//! running code that 3dsrecomp turned into a library ahead of time. the
//! library's functions work on a context holding the registers and reach
//! memory through the same page tables the interpreter uses, calling back
//! here for anything else. whatever the library has no code for stays with
//! the interpreter.

use std::cell::Cell;
use std::ffi::c_void;
use std::path::Path;

use recomp_abi::{Code, Context, Host, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND};
use zakuro_cpu::{Bus, Cpu, Exit};

use crate::memory::Memory;

pub use recomp_abi::Linked;

/// the library 3dsrecomp build installed for a title, if there is one.
pub fn installed(program_id: u64) -> Option<std::path::PathBuf> {
    recomp_abi::installed(program_id)
}

/// recompiled code, and how much of its work it handed back.
pub struct Library {
    backend: Backend,
    /// instructions the code handed to the interpreter one at a time.
    fallbacks: Cell<u64>,
}

enum Backend {
    Abi {
        code: recomp_abi::Library,
        /// the code found for addresses lately, each in a slot its address
        /// picks. a jump through a register looks its target up, which is a
        /// search through every function otherwise.
        found: Box<[Found]>,
    },
    Aot(zakuro_aot::LoadedAot),
    /// Transient placeholder while updating owned `LoadedAot` state in place.
    Empty,
}

/// a slot of Library::found, an address and the code found for it.
type Found = Cell<(u32, Option<Code>)>;

/// slots of Library::found, a power of two. Kept at 128 KiB (8192 * 16 B) on
/// Horizon so indirect-call lookups stay warm inside Tegra X1's 2 MiB L2.
#[cfg(target_os = "horizon")]
const FOUND: usize = 8192;
#[cfg(not(target_os = "horizon"))]
const FOUND: usize = 65536;

/// the slot of Library::found an address goes in, the top bits of the
/// address times the golden ratio. ARM addresses step by four and Thumb ones
/// by two with bit 0 set, and taking low bits would leave half the slots to
/// one of them.
fn slot(address: u32) -> usize {
    (address.wrapping_mul(0x9E37_79B1) >> (32 - FOUND.trailing_zeros())) as usize
}

/// Syncs AArch64 hardware FPCR control bits [25:22] (DN, FZ, RMode) with the
/// guest's FPSCR while recompiled ARMv6 VFP code runs, restoring the host
/// thread's FPCR on exit.
#[cfg(target_arch = "aarch64")]
struct FpcrGuard {
    saved: u64,
    current: u64,
}

#[cfg(target_arch = "aarch64")]
impl FpcrGuard {
    const MASK: u64 = 0x03C0_0000;

    #[inline(always)]
    fn new(guest_fpscr: u32) -> Self {
        let saved: u64;
        // SAFETY: FPCR is an unprivileged EL0 floating-point control register on AArch64.
        unsafe { core::arch::asm!("mrs {}, fpcr", out(reg) saved, options(nomem, nostack, preserves_flags)) };
        let mut guard = Self { saved, current: saved };
        guard.sync(guest_fpscr);
        guard
    }

    #[inline(always)]
    fn sync(&mut self, guest_fpscr: u32) {
        let wanted = (self.current & !Self::MASK) | ((guest_fpscr as u64) & Self::MASK);
        if wanted != self.current {
            // SAFETY: Only standard IEEE/ARMv6-compatible DN, FZ, and RMode bits [25:22] change.
            unsafe { core::arch::asm!("msr fpcr, {}", in(reg) wanted, options(nomem, nostack, preserves_flags)) };
            self.current = wanted;
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl Drop for FpcrGuard {
    #[inline(always)]
    fn drop(&mut self) {
        if self.current != self.saved {
            // SAFETY: Restores the exact FPCR value read on entry.
            unsafe { core::arch::asm!("msr fpcr, {}", in(reg) self.saved, options(nomem, nostack, preserves_flags)) };
        }
    }
}

/// Syncs x86_64 SSE MXCSR control bits (FTZ, DAZ, RC) with guest `FPSCR[25:22]`
/// while recompiled ARMv6 VFP code runs, restoring the host thread's MXCSR on drop.
#[cfg(target_arch = "x86_64")]
struct MxcsrGuard {
    saved: u32,
    current: u32,
}

#[cfg(target_arch = "x86_64")]
impl MxcsrGuard {
    // MXCSR FTZ (bit 15), RC (bits [14:13]), and DAZ (bit 6).
    const MASK: u32 = (1 << 15) | (0b11 << 13) | (1 << 6);

    #[inline(always)]
    fn read_mxcsr() -> u32 {
        let mut mxcsr: u32 = 0;
        // SAFETY: SSE is baseline on x86_64; `stmxcsr` stores the 32-bit MXCSR to a stack local.
        unsafe {
            core::arch::asm!(
                "stmxcsr [{ptr}]",
                ptr = in(reg) &mut mxcsr,
                options(nostack, preserves_flags),
            );
        }
        mxcsr
    }

    #[inline(always)]
    fn write_mxcsr(mxcsr: u32) {
        // SAFETY: SSE is baseline on x86_64; `ldmxcsr` loads a valid 32-bit MXCSR from a stack local.
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{ptr}]",
                ptr = in(reg) &mxcsr,
                options(nostack, preserves_flags),
            );
        }
    }

    /// Maps guest ARMv6 `FPSCR[25:22]` into x86_64 MXCSR FTZ/DAZ/RC bits.
    #[inline(always)]
    fn map_fpscr(guest_fpscr: u32) -> u32 {
        let fz = (guest_fpscr >> 24) & 1;
        let arm_rmode = (guest_fpscr >> 22) & 0b11;
        // ARM FPSCR[23:22]: 00=RN, 01=+Inf, 10=-Inf, 11=RZ.
        // x86 MXCSR[14:13]: 00=RN, 01=-Inf, 10=+Inf, 11=RZ (01 and 10 are swapped).
        let x86_rc = match arm_rmode {
            0b01 => 0b10,
            0b10 => 0b01,
            other => other,
        };
        (fz << 15) | (x86_rc << 13) | (fz << 6)
    }

    #[inline(always)]
    fn new(guest_fpscr: u32) -> Self {
        let saved = Self::read_mxcsr();
        let mut guard = Self { saved, current: saved };
        guard.sync(guest_fpscr);
        guard
    }

    #[inline(always)]
    fn sync(&mut self, guest_fpscr: u32) {
        let wanted = (self.current & !Self::MASK) | Self::map_fpscr(guest_fpscr);
        if wanted != self.current {
            Self::write_mxcsr(wanted);
            self.current = wanted;
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for MxcsrGuard {
    #[inline(always)]
    fn drop(&mut self) {
        if self.current != self.saved {
            Self::write_mxcsr(self.saved);
        }
    }
}

/// why a run of recompiled code ended.
pub enum Stop {
    /// an svc, the cpu is already past it.
    Svc(u32),
    /// the interpreter stopped on an instruction the code handed it.
    Exit(Exit),
    /// the budget ran out, or execution left for code the library does not
    /// have.
    Left,
}

/// what the callbacks reach through the context.
struct Machine<'a> {
    cpu: &'a mut Cpu,
    memory: &'a mut Memory,
    library: &'a Library,
    /// what the interpreter stopped with inside a callback.
    pending: Option<Exit>,
}

/// # Safety
///
/// ctx has to be a context run set up, whose user field points at a live
/// Machine.
unsafe fn machine<'a>(ctx: *mut Context) -> &'a mut Machine<'a> {
    unsafe { &mut *((*ctx).user as *mut Machine) }
}

unsafe extern "C" fn read8(ctx: *mut Context, address: u32) -> u8 {
    unsafe { machine(ctx).memory.read8(address) }
}

unsafe extern "C" fn read16(ctx: *mut Context, address: u32) -> u16 {
    unsafe { machine(ctx).memory.read16(address) }
}

unsafe extern "C" fn read32(ctx: *mut Context, address: u32) -> u32 {
    unsafe { machine(ctx).memory.read32(address) }
}

unsafe extern "C" fn write8(ctx: *mut Context, address: u32, value: u8) {
    unsafe { machine(ctx).memory.write8(address, value) }
}

unsafe extern "C" fn write16(ctx: *mut Context, address: u32, value: u16) {
    unsafe { machine(ctx).memory.write16(address, value) }
}

unsafe extern "C" fn write32(ctx: *mut Context, address: u32, value: u32) {
    unsafe { machine(ctx).memory.write32(address, value) }
}

/// runs one instruction the code left to the interpreter.
unsafe extern "C" fn interpret(ctx: *mut Context, address: u32, _opcode: u32) {
    unsafe {
        let machine = machine(ctx);
        machine.library.fallbacks.set(machine.library.fallbacks.get() + 1);
        let ctx = &mut *ctx;
        load(ctx, machine.cpu);
        machine.cpu.regs[15] = address;
        let exit = machine.cpu.step(machine.memory);
        store(machine.cpu, ctx);
        let step = if ctx.thumb != 0 { 2 } else { 4 };
        if exit.is_some() || ctx.r[15] != address.wrapping_add(step) {
            machine.pending = exit;
            ctx.exit = EXIT_UNWIND;
        }
    }
}

unsafe extern "C" fn lookup(ctx: *mut Context, address: u32) -> Option<Code> {
    unsafe { machine(ctx).library.lookup(address) }
}

static HOST: Host = Host { read8, read16, read32, write8, write16, write32, interpret, lookup };

fn load(ctx: &Context, cpu: &mut Cpu) {
    cpu.regs = ctx.r;
    cpu.cpsr.n = ctx.n != 0;
    cpu.cpsr.z = ctx.z != 0;
    cpu.cpsr.c = ctx.c != 0;
    cpu.cpsr.v = ctx.v != 0;
    cpu.cpsr.q = ctx.q != 0;
    cpu.cpsr.ge = ctx.ge;
    cpu.cpsr.thumb = ctx.thumb != 0;
    cpu.exclusive_addr = (ctx.exclusive != 0).then_some(ctx.exclusive_address);
}

fn store(cpu: &Cpu, ctx: &mut Context) {
    ctx.r = cpu.regs;
    ctx.n = cpu.cpsr.n as u8;
    ctx.z = cpu.cpsr.z as u8;
    ctx.c = cpu.cpsr.c as u8;
    ctx.v = cpu.cpsr.v as u8;
    ctx.q = cpu.cpsr.q as u8;
    ctx.ge = cpu.cpsr.ge;
    ctx.thumb = cpu.cpsr.thumb as u8;
    ctx.exclusive = cpu.exclusive_addr.is_some() as u8;
    ctx.exclusive_address = cpu.exclusive_addr.unwrap_or(0);
    ctx.tls = cpu.cp15.thread_id_ro;
}

impl Library {
    /// a library 3dsrecomp built, or a cached `UNXAOT*` AOT image.
    pub fn open(path: &Path) -> Result<Library, String> {
        #[cfg(not(target_os = "horizon"))]
        if let Ok(mut file) = std::fs::File::open(path) {
            use std::io::{Read, Seek, SeekFrom};
            let mut magic = [0u8; 8];
            if file.read_exact(&mut magic).is_ok() && &magic == zakuro_aot::CACHE_MAGIC {
                let _ = file.seek(SeekFrom::Start(0));
                let reader = std::io::BufReader::new(file);
                return zakuro_aot::LoadedAot::read_exec(reader).map(Library::from_aot);
            }
        }
        recomp_abi::Library::open(path).map(Library::new_abi)
    }

    /// the code linked into the program.
    pub fn linked(linked: &Linked) -> Result<Library, String> {
        recomp_abi::Library::linked(linked).map(Library::new_abi)
    }

    /// native AArch64 / x86_64 AOT code compiled by `zakuro-aot`.
    pub fn from_aot(aot: zakuro_aot::LoadedAot) -> Library {
        Library { backend: Backend::Aot(aot), fallbacks: Cell::new(0) }
    }

    /// Attaches native override functions and optional plugin state to an AOT backend.
    pub fn set_aot_overrides(
        &mut self,
        overrides: &[(u32, zakuro_aot::OverrideFn)],
        plugin_state: Option<zakuro_aot::PluginState>,
    ) {
        if !matches!(self.backend, Backend::Aot(_)) {
            return;
        }
        let Backend::Aot(mut aot) = std::mem::replace(&mut self.backend, Backend::Empty) else {
            unreachable!();
        };
        aot.set_overrides(overrides);
        if let Some(state) = plugin_state {
            aot = aot.with_plugin_state(state);
        }
        self.backend = Backend::Aot(aot);
    }

    fn new_abi(code: recomp_abi::Library) -> Library {
        Library {
            backend: Backend::Abi { code, found: Library::nothing_found() },
            fallbacks: Cell::new(0),
        }
    }

    /// every slot of found empty, an address no code starts at in each.
    fn nothing_found() -> Box<[Found]> {
        (0..FOUND).map(|_| Cell::new((u32::MAX, None))).collect()
    }

    /// whether the code came from an older 3dsrecomp than the one this
    /// build has, so that recompiling the game again makes it better.
    pub fn outdated(&self) -> bool {
        match &self.backend {
            Backend::Abi { code, .. } => code.generation() < recomp_abi::GENERATION,
            Backend::Aot(_) | Backend::Empty => false,
        }
    }

    /// whether the code can tell the functions whose code changed in
    /// memory, which it can from generation 2 on.
    pub fn checks_itself(&self) -> bool {
        match &self.backend {
            Backend::Abi { code, .. } => code.origins(0).is_some(),
            Backend::Aot(_) => true,
            Backend::Empty => false,
        }
    }

    /// how many instructions the code has handed to the interpreter.
    pub fn fallbacks(&self) -> u64 {
        self.fallbacks.get()
    }

    /// how many functions and modules there is code for.
    pub fn describe(&self) -> String {
        match &self.backend {
            Backend::Abi { code, .. } => code.describe(),
            Backend::Aot(aot) => aot.describe(),
            Backend::Empty => String::new(),
        }
    }

    /// the code that can run from address, bit 0 set for Thumb, in the
    /// executable or in a module that is loaded.
    fn lookup(&self, address: u32) -> Option<Code> {
        let Backend::Abi { code, found } = &self.backend else {
            return None;
        };
        let slot = &found[slot(address)];
        match slot.get() {
            (seen, code) if seen == address => code,
            _ => {
                let code = code.lookup(address);
                slot.set((address, code));
                code
            }
        }
    }

    pub fn has_code(&self, address: u32) -> bool {
        match &self.backend {
            Backend::Abi { .. } => self.lookup(address).is_some(),
            Backend::Aot(aot) => aot.has_code(address),
            Backend::Empty => false,
        }
    }

    /// marks the executable's functions whose code in memory is not what
    /// they were recompiled from, which a mod or another version of the
    /// game does, so that the interpreter runs them. how many there are.
    pub fn check(&mut self, memory: &mut Memory) -> usize {
        match &mut self.backend {
            Backend::Abi { code, found } => {
                let stale = code.check(0, 0, |address, bytes| memory.read_bytes(address, bytes));
                *found = Library::nothing_found();
                stale
            }
            Backend::Aot(aot) => aot.check(|address, bytes| memory.read_bytes(address, bytes)),
            Backend::Empty => 0,
        }
    }

    /// tells the code of the module called name where the title loaded it,
    /// zero when it unloads it, and checks the module's code once it is in
    /// memory.
    pub fn place(&mut self, name: &str, base: u32, memory: &mut Memory) {
        let Backend::Abi { code, found } = &mut self.backend else {
            return;
        };
        let Some(index) = code.module_index(name) else { return };
        code.place(index, base);
        // what was found before may have moved or gone
        *found = Library::nothing_found();
        if base == 0 {
            return;
        }
        log::info!("recompiled code for {name} runs at 0x{base:08X}");
        let stale = code.check(index + 1, base, |address, bytes| memory.read_bytes(address, bytes));
        if stale > 0 {
            log::warn!("{stale} functions of {name} changed since it was recompiled, they run in the interpreter");
        }
    }

    /// runs recompiled code from the cpu's pc for about budget instructions,
    /// returning how many ran and why it stopped.
    pub fn run(&self, cpu: &mut Cpu, memory: &mut Memory, budget: u64) -> (u64, Stop) {
        let cycles = cpu.cycles;
        let (read_pages, write_pages) = memory.page_tables();
        let mut ctx = Context {
            r: [0; 16],
            n: 0,
            z: 0,
            c: 0,
            v: 0,
            q: 0,
            thumb: 0,
            ge: 0,
            exclusive: 0,
            budget: budget.min(i32::MAX as u64) as i32,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: 0,
            tls: 0,
            read_pages,
            write_pages,
            vfp: std::ptr::null_mut(),
            fpscr: std::ptr::null_mut(),
            host: &HOST,
            user: std::ptr::null_mut(),
        };
        store(cpu, &mut ctx);
        let mut machine = Machine { cpu, memory, library: self, pending: None };
        // the code works on the VFP registers where they are, so the
        // interpreter sees its changes without copying them around
        ctx.vfp = machine.cpu.vfp.regs.as_mut_ptr();
        ctx.fpscr = &mut machine.cpu.vfp.fpscr;
        ctx.user = &mut machine as *mut Machine as *mut c_void;
        #[cfg(target_arch = "aarch64")]
        let mut fpcr = FpcrGuard::new(unsafe { *ctx.fpscr });
        #[cfg(target_arch = "x86_64")]
        let mut mxcsr = MxcsrGuard::new(unsafe { *ctx.fpscr });

        let stop = match &self.backend {
            Backend::Abi { .. } => loop {
                if ctx.budget <= 0 {
                    break Stop::Left;
                }
                let Some(code) = self.lookup(ctx.r[15] | ctx.thumb as u32) else {
                    break Stop::Left;
                };
                ctx.exit = 0;
                ctx.depth = 0;
                #[cfg(target_arch = "aarch64")]
                fpcr.sync(unsafe { *ctx.fpscr });
                #[cfg(target_arch = "x86_64")]
                mxcsr.sync(unsafe { *ctx.fpscr });
                // SAFETY: the code came from the library and ctx is set up the
                // way it expects, with user pointing at machine
                unsafe { code(&mut ctx) };
                if let Some(exit) = machine.pending.take() {
                    break Stop::Exit(exit);
                }
                match ctx.exit {
                    EXIT_SVC => break Stop::Svc(ctx.svc),
                    EXIT_BUDGET => break Stop::Left,
                    _ => {}
                }
            },
            Backend::Aot(aot) => loop {
                if ctx.budget <= 0 {
                    break Stop::Left;
                }
                let Some(target_rx) = aot.lookup(ctx.r[15] | ctx.thumb as u32) else {
                    break Stop::Left;
                };
                ctx.exit = 0;
                ctx.depth = 0;
                #[cfg(target_arch = "aarch64")]
                fpcr.sync(unsafe { *ctx.fpscr });
                #[cfg(target_arch = "x86_64")]
                mxcsr.sync(unsafe { *ctx.fpscr });
                // SAFETY: `target_rx` came from `aot.lookup`, and `ctx.user` points at `machine`.
                unsafe { aot.enter(&mut ctx, target_rx) };
                if let Some(exit) = machine.pending.take() {
                    break Stop::Exit(exit);
                }
                match ctx.exit {
                    EXIT_SVC => break Stop::Svc(ctx.svc),
                    EXIT_BUDGET => break Stop::Left,
                    _ => {}
                }
            },
            Backend::Empty => Stop::Left,
        };

        load(&ctx, machine.cpu);
        let ran = budget.saturating_sub(ctx.budget.max(0) as u64);
        // the interpreter counted the instructions it ran for the code, which
        // the budget already has
        machine.cpu.cycles = cycles + ran;
        (ran, stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_and_thumb_code_use_every_slot() {
        let arm: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0000 + 4 * i)).collect();
        let thumb: std::collections::BTreeSet<usize> = (0..FOUND as u32).map(|i| slot(0x0010_0001 + 2 * i)).collect();
        assert!(arm.len() > FOUND * 3 / 4, "{} of {FOUND}", arm.len());
        assert!(thumb.len() > FOUND * 3 / 4, "{} of {FOUND}", thumb.len());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn mxcsr_guard_maps_arm_rounding_and_fz_and_restores_host() {
        // ARM 00=RN, 01=+Inf, 10=-Inf, 11=RZ; x86 00=RN, 01=-Inf, 10=+Inf, 11=RZ.
        assert_eq!(MxcsrGuard::map_fpscr(0 << 22), 0 << 13);
        assert_eq!(MxcsrGuard::map_fpscr(0b01 << 22), 0b10 << 13);
        assert_eq!(MxcsrGuard::map_fpscr(0b10 << 22), 0b01 << 13);
        assert_eq!(MxcsrGuard::map_fpscr(0b11 << 22), 0b11 << 13);
        assert_eq!(
            MxcsrGuard::map_fpscr((1 << 24) | (0b01 << 22)),
            (1 << 15) | (0b10 << 13) | (1 << 6)
        );

        let before = MxcsrGuard::read_mxcsr();
        {
            let mut guard = MxcsrGuard::new((1 << 24) | (0b01 << 22));
            let active = MxcsrGuard::read_mxcsr();
            assert_eq!(active & MxcsrGuard::MASK, (1 << 15) | (0b10 << 13) | (1 << 6));
            guard.sync(0b10 << 22);
            let updated = MxcsrGuard::read_mxcsr();
            assert_eq!(updated & MxcsrGuard::MASK, 0b01 << 13);
        }
        let after = MxcsrGuard::read_mxcsr();
        assert_eq!(after, before);
    }

    #[cfg(not(target_os = "horizon"))]
    #[test]
    fn library_open_reads_aot_cache_and_attaches_overrides() {
        let image = zakuro_aot::AotImage {
            program_id: 0x0004_0000_0003_0100,
            text_hash: 0xD6E0_28C6_175A_2B07,
            enter_offset: 0,
            code: vec![0xC3; 16],
            entries: vec![(0x0010_0000, 0)],
            origins: zakuro_aot::discover::OriginData {
                owners: vec![recomp_abi::NO_ORIGIN],
                ..Default::default()
            },
        };
        let path = std::env::temp_dir().join(format!("zakuro-recompiled-open-{}.aot", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        image.write_to(file).unwrap();

        #[cfg(target_arch = "x86_64")]
        unsafe extern "sysv64" fn dummy_override(_: *mut zakuro_aot::AotExecContext, _: *const u8) {}
        #[cfg(not(target_arch = "x86_64"))]
        unsafe extern "C" fn dummy_override(_: *mut zakuro_aot::AotExecContext, _: *const u8) {}

        let mut lib = Library::open(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(lib.checks_itself());
        assert!(!lib.outdated());
        assert!(lib.has_code(0x0010_0000));
        lib.set_aot_overrides(
            &[(0x0010_0000, dummy_override)],
            Some(zakuro_aot::PluginState([1, 2, 3, 4])),
        );
    }
}
