//! Differential verification of `zakuro-aot` compiled AArch64/x86_64 machine code
//! against `zakuro-cpu` interpreter across synthetic ISA torture tests and
//! thousands of real Kid Icarus: Uprising functions.

#![cfg(all(unix, any(target_arch = "aarch64", target_arch = "x86_64")))]

use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

use recomp_abi::{Context, Host, EXIT_BUDGET, EXIT_NONE, EXIT_SVC, EXIT_UNWIND};
use zakuro_aot::discover::{Program, Segment, Source};
use zakuro_aot::{compile_program, compile_title, CompileOptions, LoadedAot};
use zakuro_cpu::{Bus, Cpu, Exit};

const PAGE_SIZE: usize = 0x1000;
const BASE: u32 = 0x0010_0000;
const RETURN: u32 = 0xFFFF_FFF0;
const STACK_TOP: u32 = 0x1000_0000;
const STACK_SIZE: u32 = 0x4_0000;
const HEAP: u32 = 0x0800_0000;
const HEAP_SIZE: u32 = 0x4_0000;
const UNMAPPED_SCRATCH: u32 = 0x0600_0000; // handled only through slow-path callbacks!
const TLS: u32 = 0x1FF8_2000;
const LIMIT: i32 = 100_000;

#[derive(Clone)]
struct Region {
    base: u32,
    bytes: Vec<u8>,
    writable: bool,
    direct_page: bool,
}

struct TestMemory {
    regions: Vec<Region>,
    read_pages: Vec<*mut u8>,
    write_pages: Vec<*mut u8>,
    faults: u64,
}

impl TestMemory {
    fn new(mut regions: Vec<Region>) -> Self {
        let mut read_pages = vec![null_mut(); 1 << 20];
        let mut write_pages = vec![null_mut(); 1 << 20];
        for region in &mut regions {
            region.bytes.resize(region.bytes.len().next_multiple_of(PAGE_SIZE), 0);
            if region.direct_page {
                for (i, page) in region.bytes.chunks_mut(PAGE_SIZE).enumerate() {
                    let index = (region.base as usize >> 12) + i;
                    read_pages[index] = page.as_mut_ptr();
                    if region.writable {
                        write_pages[index] = page.as_mut_ptr();
                    }
                }
            }
        }
        Self { regions, read_pages, write_pages, faults: 0 }
    }

    fn duplicate(&self) -> Self {
        Self::new(self.regions.clone())
    }

    fn copy_from(&mut self, other: &Self) {
        for (dst, src) in self.regions.iter_mut().zip(&other.regions) {
            if dst.writable {
                for (dp, sp) in dst.bytes.chunks_mut(PAGE_SIZE).zip(src.bytes.chunks(PAGE_SIZE)) {
                    if dp != sp {
                        dp.copy_from_slice(sp);
                    }
                }
            }
        }
        self.faults = 0;
    }

    fn find_region_byte_mut(&mut self, addr: u32, write: bool) -> Option<&mut u8> {
        for r in &mut self.regions {
            let end = r.base.wrapping_add(r.bytes.len() as u32);
            if addr >= r.base && addr < end {
                if write && !r.writable {
                    return None;
                }
                return Some(&mut r.bytes[(addr - r.base) as usize]);
            }
        }
        None
    }

    fn first_difference(&self, other: &Self) -> Option<u32> {
        self.regions.iter().zip(&other.regions).filter(|(r, _)| r.writable).find_map(|(a, b)| {
            if a.bytes == b.bytes {
                return None;
            }
            for (page_idx, (ap, bp)) in a.bytes.chunks(PAGE_SIZE).zip(b.bytes.chunks(PAGE_SIZE)).enumerate() {
                if ap == bp {
                    continue;
                }
                let page_off = page_idx * PAGE_SIZE;
                let mut i = 0;
                while i < ap.len() {
                    if ap[i] == bp[i] {
                        i += 1;
                        continue;
                    }
                    let word = page_off + (i & !3);
                    let double = page_off + (i & !7);
                    if both_nan(&a.bytes, &b.bytes, double, 8) {
                        i = (i & !7) + 8;
                    } else if both_nan(&a.bytes, &b.bytes, word, 4) {
                        i = (i & !3) + 4;
                    } else {
                        return Some(a.base + (page_off + i) as u32);
                    }
                }
            }
            None
        })
    }
}

fn both_nan(a: &[u8], b: &[u8], off: usize, size: usize) -> bool {
    let is_nan = |s: &[u8]| match s.get(off..off + size) {
        Some(bytes) if size == 4 => f32::from_le_bytes(bytes.try_into().unwrap()).is_nan(),
        Some(bytes) if size == 8 => f64::from_le_bytes(bytes.try_into().unwrap()).is_nan(),
        _ => false,
    };
    is_nan(a) && is_nan(b)
}

impl Bus for TestMemory {
    fn read8(&mut self, address: u32) -> u8 {
        let page = self.read_pages[(address >> 12) as usize];
        if !page.is_null() {
            return unsafe { *page.add((address & 0xFFF) as usize) };
        }
        if let Some(b) = self.find_region_byte_mut(address, false) {
            return *b;
        }
        self.faults += 1;
        0
    }

    fn read16(&mut self, address: u32) -> u16 {
        u16::from_le_bytes([self.read8(address), self.read8(address.wrapping_add(1))])
    }

    fn read32(&mut self, address: u32) -> u32 {
        u32::from_le_bytes(std::array::from_fn(|i| self.read8(address.wrapping_add(i as u32))))
    }

    fn write8(&mut self, address: u32, value: u8) {
        let page = self.write_pages[(address >> 12) as usize];
        if !page.is_null() {
            unsafe { *page.add((address & 0xFFF) as usize) = value };
            return;
        }
        if let Some(b) = self.find_region_byte_mut(address, true) {
            *b = value;
            return;
        }
        self.faults += 1;
    }

    fn write16(&mut self, address: u32, value: u16) {
        for (i, b) in value.to_le_bytes().into_iter().enumerate() {
            self.write8(address.wrapping_add(i as u32), b);
        }
    }

    fn write32(&mut self, address: u32, value: u32) {
        for (i, b) in value.to_le_bytes().into_iter().enumerate() {
            self.write8(address.wrapping_add(i as u32), b);
        }
    }
}

struct Run {
    memory: TestMemory,
    cpu: Cpu,
    pending: Option<Exit>,
    interpreted: u64,
    fallbacks: u64,
}

#[derive(Debug, PartialEq)]
enum Stop {
    Returned,
    Svc(u32),
    Limit,
    Other(String),
}

fn to_stop(exit: Exit) -> Stop {
    match exit {
        Exit::Supervisor(n) => Stop::Svc(n),
        other => Stop::Other(format!("{other:?}")),
    }
}

unsafe fn run_of<'a>(ctx: *mut Context) -> &'a mut Run {
    unsafe { &mut *((*ctx).user as *mut Run) }
}

unsafe extern "C" fn cb_read8(ctx: *mut Context, addr: u32) -> u8 {
    unsafe { run_of(ctx).memory.read8(addr) }
}
unsafe extern "C" fn cb_read16(ctx: *mut Context, addr: u32) -> u16 {
    unsafe { run_of(ctx).memory.read16(addr) }
}
unsafe extern "C" fn cb_read32(ctx: *mut Context, addr: u32) -> u32 {
    unsafe { run_of(ctx).memory.read32(addr) }
}
unsafe extern "C" fn cb_write8(ctx: *mut Context, addr: u32, val: u8) {
    unsafe { run_of(ctx).memory.write8(addr, val) }
}
unsafe extern "C" fn cb_write16(ctx: *mut Context, addr: u32, val: u16) {
    unsafe { run_of(ctx).memory.write16(addr, val) }
}
unsafe extern "C" fn cb_write32(ctx: *mut Context, addr: u32, val: u32) {
    unsafe { run_of(ctx).memory.write32(addr, val) }
}
unsafe extern "C" fn cb_interpret(ctx: *mut Context, address: u32, _opcode: u32) {
    unsafe {
        let run = run_of(ctx);
        let ctx = &mut *ctx;
        load_ctx(ctx, &mut run.cpu);
        run.cpu.regs[15] = address;
        let exit = run.cpu.step(&mut run.memory);
        store_ctx(&run.cpu, ctx);
        run.fallbacks += 1;
        let step = if ctx.thumb != 0 { 2 } else { 4 };
        if exit.is_some() || ctx.r[15] != address.wrapping_add(step) {
            run.pending = exit;
            ctx.exit = EXIT_UNWIND;
        }
    }
}
unsafe extern "C" fn cb_lookup(_ctx: *mut Context, _addr: u32) -> Option<recomp_abi::Code> {
    None
}

static HOST: Host = Host {
    read8: cb_read8,
    read16: cb_read16,
    read32: cb_read32,
    write8: cb_write8,
    write16: cb_write16,
    write32: cb_write32,
    interpret: cb_interpret,
    lookup: cb_lookup,
};

fn load_ctx(ctx: &Context, cpu: &mut Cpu) {
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

fn store_ctx(cpu: &Cpu, ctx: &mut Context) {
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

#[cfg(target_arch = "aarch64")]
fn sync_host_fpcr(guest_fpscr: u32) -> u64 {
    let saved: u64;
    unsafe { core::arch::asm!("mrs {}, fpcr", out(reg) saved, options(nomem, nostack, preserves_flags)) };
    let wanted = (saved & !0x03C0_0000) | ((guest_fpscr as u64) & 0x03C0_0000);
    if wanted != saved {
        unsafe { core::arch::asm!("msr fpcr, {}", in(reg) wanted, options(nomem, nostack, preserves_flags)) };
    }
    saved
}

#[cfg(target_arch = "aarch64")]
fn restore_host_fpcr(saved: u64) {
    unsafe { core::arch::asm!("msr fpcr, {}", in(reg) saved, options(nomem, nostack, preserves_flags)) };
}

#[cfg(target_arch = "x86_64")]
fn sync_host_fpcr(guest_fpscr: u32) -> u64 {
    let mut saved: u32 = 0;
    unsafe {
        core::arch::asm!(
            "stmxcsr [{}]",
            in(reg) &mut saved,
            options(nostack, preserves_flags),
        );
    }
    // ARM FPSCR[23:22] (00=RN, 01=+Inf, 10=-Inf, 11=RZ) -> x86 MXCSR[14:13] (01=-Inf, 10=+Inf)
    let mxcsr_rc = match (guest_fpscr >> 22) & 3 {
        0b00 => 0b00,
        0b01 => 0b10,
        0b10 => 0b01,
        _ => 0b11,
    };
    let mxcsr_ftz = if (guest_fpscr & (1 << 24)) != 0 { (1 << 15) | (1 << 6) } else { 0 };
    let wanted = (saved & !((3 << 13) | (1 << 15) | (1 << 6))) | (mxcsr_rc << 13) | mxcsr_ftz;
    if wanted != saved {
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{}]",
                in(reg) &wanted,
                options(nostack, preserves_flags),
            );
        }
    }
    saved as u64
}

#[cfg(target_arch = "x86_64")]
fn restore_host_fpcr(saved: u64) {
    let saved32 = saved as u32;
    unsafe {
        core::arch::asm!(
            "ldmxcsr [{}]",
            in(reg) &saved32,
            options(nostack, preserves_flags),
        );
    }
}

fn run_interpreted(run: &mut Run, limit: i32) -> Stop {
    for _ in 0..limit {
        if run.cpu.regs[15] == RETURN && !run.cpu.cpsr.thumb {
            return Stop::Returned;
        }
        if let Some(exit) = run.cpu.step(&mut run.memory) {
            return to_stop(exit);
        }
    }
    Stop::Limit
}

unsafe fn run_aot(aot: &LoadedAot, run: *mut Run, ctx: *mut Context, slice: i32, limit: i32) -> Stop {
    unsafe {
        let mut left = limit;
        let stop = loop {
            if (*ctx).r[15] == RETURN && (*ctx).thumb == 0 {
                break Stop::Returned;
            }
            if left <= 0 {
                break Stop::Limit;
            }
            let key = (*ctx).r[15] | ((*ctx).thumb as u32);
            if let Some(target_rx) = aot.lookup(key) {
                let budget = slice.min(left);
                (*ctx).budget = budget;
                (*ctx).exit = EXIT_NONE;
                (*ctx).depth = 0;
                let saved_fpcr = sync_host_fpcr(*(*ctx).fpscr);
                aot.enter(&mut *ctx, target_rx);
                restore_host_fpcr(saved_fpcr);
                let spent = budget - (*ctx).budget;
                left -= spent;
                if let Some(exit) = (*run).pending.take() {
                    break to_stop(exit);
                }
                match (*ctx).exit {
                    EXIT_SVC => break Stop::Svc((*ctx).svc),
                    EXIT_BUDGET if spent == 0 => {}
                    _ => continue,
                }
            }
            let run = &mut *run;
            let ctx = &mut *ctx;
            load_ctx(ctx, &mut run.cpu);
            let exit = run.cpu.step(&mut run.memory);
            store_ctx(&run.cpu, ctx);
            left -= 1;
            run.interpreted += 1;
            if let Some(exit) = exit {
                break to_stop(exit);
            }
        };
        (*ctx).budget = left;
        stop
    }
}

fn same_vfp(a: &[u32; 32], b: &[u32; 32]) -> bool {
    let single = |bits: u32| f32::from_bits(bits).is_nan();
    let double = |r: &[u32; 32], d: usize| f64::from_bits(((r[2 * d + 1] as u64) << 32) | (r[2 * d] as u64)).is_nan();
    (0..16).all(|d| {
        (double(a, d) && double(b, d))
            || (2 * d..2 * d + 2).all(|s| a[s] == b[s] || (single(a[s]) && single(b[s])))
    })
}

fn compare_runs(a: &Run, b: &Run, ctx: &Context) -> Vec<String> {
    let mut diffs = Vec::new();
    let nan = |bits: u32| f32::from_bits(bits).is_nan();
    for i in 0..16 {
        if a.cpu.regs[i] != ctx.r[i] && !(nan(a.cpu.regs[i]) && nan(ctx.r[i])) {
            diffs.push(format!("r{i}: interp={:08X} aot={:08X}", a.cpu.regs[i], ctx.r[i]));
        }
    }
    for (name, exp, got) in [
        ("n", a.cpu.cpsr.n as u8, ctx.n),
        ("z", a.cpu.cpsr.z as u8, ctx.z),
        ("c", a.cpu.cpsr.c as u8, ctx.c),
        ("v", a.cpu.cpsr.v as u8, ctx.v),
        ("q", a.cpu.cpsr.q as u8, ctx.q),
        ("thumb", a.cpu.cpsr.thumb as u8, ctx.thumb),
    ] {
        if exp != got {
            diffs.push(format!("{name}: interp={exp} aot={got}"));
        }
    }
    if a.cpu.cpsr.ge != ctx.ge {
        diffs.push(format!("ge: interp={:X} aot={:X}", a.cpu.cpsr.ge, ctx.ge));
    }
    let mon = (ctx.exclusive != 0).then_some(ctx.exclusive_address);
    if a.cpu.exclusive_addr != mon {
        diffs.push(format!("excl: interp={:X?} aot={mon:X?}", a.cpu.exclusive_addr));
    }
    if !same_vfp(&a.cpu.vfp.regs, &b.cpu.vfp.regs) || a.cpu.vfp.fpscr != b.cpu.vfp.fpscr {
        for s in 0..32 {
            if a.cpu.vfp.regs[s] != b.cpu.vfp.regs[s] && !(nan(a.cpu.vfp.regs[s]) && nan(b.cpu.vfp.regs[s])) {
                diffs.push(format!("s{s}: interp={:08X} aot={:08X}", a.cpu.vfp.regs[s], b.cpu.vfp.regs[s]));
            }
        }
        if a.cpu.vfp.fpscr != b.cpu.vfp.fpscr {
            diffs.push(format!("fpscr: interp={:08X} aot={:08X}", a.cpu.vfp.fpscr, b.cpu.vfp.fpscr));
        }
    }
    if let Some(addr) = a.memory.first_difference(&b.memory) {
        diffs.push(format!("mem@{addr:08X}"));
    }
    if a.memory.faults != b.memory.faults {
        diffs.push(format!("faults: interp={} aot={}", a.memory.faults, b.memory.faults));
    }
    diffs
}

fn compile_snippet(words: &[u32], seeds: &[(u32, Source)]) -> (Vec<u8>, LoadedAot) {
    let text_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let program = Program {
        text: Segment { base: BASE, bytes: text_bytes.clone() },
        seeds: seeds.to_vec(),
        slots: None,
    };
    let image = compile_program(0, 0x1234, &program, &[], CompileOptions::default(), |_, _| {});
    // Also exercise binary cache serialization round-trip!
    let mut buf = Vec::new();
    image.write_to(&mut buf).unwrap();
    let aot = LoadedAot::read_exec(buf.as_slice()).unwrap();
    (text_bytes, aot)
}

fn verify_compiled(
    name: &str,
    text_bytes: &[u8],
    aot: &LoadedAot,
    require_zero_fallbacks: bool,
    init: impl FnOnce(&mut Cpu, &mut TestMemory),
) {
    let mut mem = TestMemory::new(vec![
        Region { base: BASE, bytes: text_bytes.to_vec(), writable: false, direct_page: true },
        Region { base: HEAP, bytes: vec![0x5A; HEAP_SIZE as usize], writable: true, direct_page: true },
        Region { base: UNMAPPED_SCRATCH, bytes: vec![0xA5; PAGE_SIZE * 2], writable: true, direct_page: false },
        Region { base: STACK_TOP - STACK_SIZE, bytes: vec![0; STACK_SIZE as usize], writable: true, direct_page: true },
        Region { base: TLS, bytes: vec![0; PAGE_SIZE], writable: true, direct_page: true },
    ]);
    let mut cpu = Cpu::new();
    cpu.regs[13] = STACK_TOP - 0x1000;
    cpu.regs[14] = RETURN;
    cpu.regs[15] = BASE;
    cpu.cp15.thread_id_ro = TLS;
    init(&mut cpu, &mut mem);

    for slice in [LIMIT, 1, 3] {
        let mut a = Run { memory: mem.duplicate(), cpu: cpu.clone(), pending: None, interpreted: 0, fallbacks: 0 };
        let mut b = Run { memory: mem.duplicate(), cpu: cpu.clone(), pending: None, interpreted: 0, fallbacks: 0 };
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
            budget: 0,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: 0,
            tls: 0,
            read_pages: b.memory.read_pages.as_ptr(),
            write_pages: b.memory.write_pages.as_ptr(),
            vfp: b.cpu.vfp.regs.as_mut_ptr(),
            fpscr: &mut b.cpu.vfp.fpscr,
            host: &HOST,
            user: &mut b as *mut Run as *mut c_void,
        };
        store_ctx(&cpu, &mut ctx);
        let exp = run_interpreted(&mut a, LIMIT);
        let got = unsafe { run_aot(aot, &mut b, &mut ctx, slice, LIMIT) };
        let diffs = compare_runs(&a, &b, &ctx);
        assert_eq!(exp, got, "{name} (slice={slice}): stop mismatch");
        assert!(diffs.is_empty(), "{name} (slice={slice}): {diffs:?}");
        if require_zero_fallbacks {
            assert_eq!(b.fallbacks, 0, "{name}: unexpected interpreter fallback!");
            if slice == LIMIT {
                assert_eq!(b.interpreted, 0, "{name}: unexpected uncompiled target fallback!");
            }
        }
    }
}

fn run_snippet_test(name: &str, words: &[u32], seeds: &[(u32, Source)], init: impl FnOnce(&mut Cpu, &mut TestMemory)) {
    let (text_bytes, aot) = compile_snippet(words, seeds);
    verify_compiled(name, &text_bytes, &aot, true, init);
}

#[test]
fn alu_shifts_and_flags_match_interpreter() {
    let words = [
        // Test register shifts (0, 1, 31, 32, 33, 255), ADCS, SBCS, RSBS, RRX, conditional ops
        0xE090_4001, // adds r4, r0, r1
        0xE0B0_5001, // adcs r5, r0, r1
        0xE0D0_6001, // sbcs r6, r0, r1
        0xE070_7001, // rsbs r7, r0, r1
        0xE1B0_8210, // movs r8, r0, lsl r2
        0xE1B0_9230, // movs r9, r0, lsr r2
        0xE1B0_A250, // movs r10, r0, asr r2
        0xE1B0_B270, // movs r11, r0, ror r2
        0xE1B0_C060, // movs r12, r0, rrx
        0x03A0_0077, // moveq r0, #0x77
        0x13E0_0022, // mvnne r0, #0x22
        0xE12F_FF1E, // bx lr
    ];
    for &shift_amt in &[0u32, 1, 7, 31, 32, 33, 64, 255, 0x100, 0x120] {
        for &(r0, r1, carry) in &[
            (0x8000_0001u32, 0x7FFF_FFFFu32, true),
            (0xDEAD_BEEF, 0x8000_0001, false),
            (0, 0, true),
            (u32::MAX, 1, false),
        ] {
            run_snippet_test(
                &format!("alu_shift_{shift_amt}_{r0:08X}"),
                &words,
                &[(BASE, Source::Entry)],
                |cpu, _| {
                    cpu.regs[0] = r0;
                    cpu.regs[1] = r1;
                    cpu.regs[2] = shift_amt;
                    cpu.cpsr.c = carry;
                    cpu.cpsr.v = !carry;
                },
            );
        }
    }
}

#[test]
fn memory_cross_page_slow_path_exclusives_and_block_transfers_match() {
    let words = [
        // Unaligned & cross-page loads/stores at HEAP + 0xFFD, plus slow-path UNMAPPED_SCRATCH!
        0xE580_2000, // str r2, [r0]           (crosses page at +0xFFD!)
        0xE590_3000, // ldr r3, [r0]
        0xE1C0_20B2, // strh r2, [r0, #2]      (crosses page at +0xFFF!)
        0xE1D0_40F2, // ldrsh r4, [r0, #2]
        0xE5D0_5003, // ldrb r5, [r0, #3]
        0xE1D0_60D3, // ldrsb r6, [r0, #3]
        // Rn == Rd post-indexed load: Rd must receive loaded value, not writeback!
        0xE490_0008, // ldr r0, [r0], #8
        // Slow-path non-direct page read/write + LDRD/STRD:
        0xE1C1_20F0, // strd r2, [r1]
        0xE1C1_80D0, // ldrd r8, [r1]
        // LDREX / STREX / CLREX:
        0xE191_A99F, // ldrex r10, [r1]
        0xE181_B993, // strex r11, r3, [r1]    (should succeed -> r11 = 0)
        0xE181_C994, // strex r12, r4, [r1]    (monitor cleared -> should fail -> r12 = 1)
        0xE191_A99F, // ldrex r10, [r1]
        0xF57F_F01F, // clrex
        // PUSH & POP across stack page boundary:
        0xE92D_4FF0, // push {r4-r11, lr}
        0xEE1D_0F70, // mrc p15, 0, r0, c13, c0, 3 (TLS!)
        0xE8BD_8FF0, // pop {r4-r11, pc}
    ];
    run_snippet_test("mem_cross_page_and_slow", &words, &[(BASE, Source::Entry)], |cpu, _| {
        cpu.regs[0] = HEAP + 0xFFD;
        cpu.regs[1] = UNMAPPED_SCRATCH + 0xFFC;
        cpu.regs[2] = 0x89AB_CDEF;
        // Set SP right on a page boundary (`STACK_TOP - 0x2000 + 12`) so PUSH crosses 4KB page!
        cpu.regs[13] = STACK_TOP - 0x2000 + 12;
    });
}

#[test]
fn dsp_saturating_media_and_parallel_simd_match() {
    let words = [
        0xE100_4051, // qadd r4, r1, r0
        0xE140_5051, // qdadd r5, r1, r0
        0xE160_6051, // qdsub r6, r1, r0
        0xE107_1280, // smlabb r7, r0, r2, r1
        0xE108_12E0, // smlatt r8, r0, r2, r1
        0xE709_1210, // smlad r9, r0, r2, r1
        0xE75A_1230, // smmlar r10, r0, r2, r1
        0xE6EF_B070, // uxtab16 r11, r15(=none->uxtb16), r0 -> wait: 0xE6CF_B070 is uxtb16 r11, r0
        0xE6C2_B470, // uxtab16 r11, r2, r0, ror #8
        0xE682_C470, // sxtab16 r12, r2, r0, ror #8
        0xE650_4F12, // uadd8 r4, r0, r2       (sets GE!)
        0xE680_5FB2, // sel r5, r0, r2         (reads GE!)
        0xE660_6F92, // uqsub8 r6, r0, r2      (inline ASIMD!)
        0xE670_7F12, // uhadd8 r7, r0, r2      (inline ASIMD!)
        0xE620_8F12, // qadd16 r8, r0, r2      (inline ASIMD!)
        0xE6BF_9F30, // rev r9, r0
        0xE6FF_AF30, // rbit r10, r0
        0xE6FF_BFB0, // revsh r11, r0
        0xE6ED_C110, // usat r12, #13, r0, lsl #2
        0xE6E7_0F32, // usat16 r0, #7, r2
        0xE781_2210, // usada8 r1, r0, r2, r2
        0xE12F_FF1E, // bx lr
    ];
    run_snippet_test("dsp_media_simd", &words, &[(BASE, Source::Entry)], |cpu, _| {
        cpu.regs[0] = 0xF080_7F05;
        cpu.regs[1] = 0x5000_0000;
        cpu.regs[2] = 0x2090_81FA;
    });
}

#[test]
fn vfp_singles_doubles_conversions_and_thumb_interworking_match() {
    // ARM caller -> BLX Thumb helper -> VFP f32 & f64 math -> return!
    let words = [
        0xEE20_1A81, // vmul.f32 s2, s0, s3
        0xEE01_2A22, // vmla.f32 s4, s2, s5
        0xEE41_3A62, // vnmla.f32 s6, s2, s5
        0xEEB4_3AC2, // vcmpe.f32 s6, s5
        0xEEF1_FA10, // vmrs APSR_nzcv, fpscr
        0xEEB8_7AC0, // vcvt.f32.s32 s14, s0
        0xEEFD_7AC2, // vcvt.s32.f32 s15, s5
        // Double precision on d0 ({s0, s1}), d1 ({s2, s3}), and d15 ({s30, s31})!
        0xEE70_FB01, // vadd.f64 d15, d0, d1  (`d_bit=1, vd=15` -> 0xEE70_FB01!)
        0xEEB1_0B4F, // vneg.f64 d0, d15      (raw sign bit flip!)
        // BLX to Thumb function at BASE + 0x30 (offset +12 bytes from PC = BASE + 0x24 + 8 = BASE + 0x2C -> +4 = BASE+0x30):
        0xFA00_0000, // at 0x24: pc=0x2C, offset=0 -> target = 0x2C (let's put nop at 0x28 and target 0x30: offset=+1 word -> 0xFA00_0001!)
        0xE12F_FF1E, // 0x28: bx lr
        0xE1A0_0000, // 0x2C: pad
        // 0x30: Thumb function: `push {r4, lr}; movs r4, #42; adds r0, r0, r4; pop {r4, pc}`
        0x242A_B510, // B510 = push {r4, lr}, 242A = movs r4, #42
        0xBD10_1900, // 1900 = adds r0, r0, r4, BD10 = pop {r4, pc}
    ];
    let mut fixed = words;
    fixed[9] = 0xFA00_0001; // at 0x24: (0x24 + 8) + 4 = 0x30!
    run_snippet_test("vfp_and_thumb", &fixed, &[(BASE, Source::Entry)], |cpu, _| {
        cpu.regs[0] = 100;
        cpu.vfp.regs[0] = (-12.5_f32).to_bits();
        cpu.vfp.regs[1] = 0x3FF8_0000; // upper half of d0
        cpu.vfp.regs[2] = 3.25_f32.to_bits();
        cpu.vfp.regs[3] = 0x4004_0000; // upper half of d1
        cpu.vfp.regs[4] = 10.0_f32.to_bits();
        cpu.vfp.regs[5] = (-7.75_f32).to_bits();
        cpu.vfp.regs[30] = 1.5_f32.to_bits();
        cpu.vfp.regs[31] = 2.5_f32.to_bits();
    });
}

#[test]
fn direct_call_and_tail_jump_to_function_with_lower_prefix_block() {
    // Regression test for RVCT functions whose lowest basic-block address `< entry`:
    //   0x00: push {r4, lr}
    //   0x04: mov r0, #10
    //   0x08: bl 0x20        (callee entry at 0x20, which branches backward to 0x14!)
    //   0x0C: pop {r4, lr}
    //   0x10: b 0x30         (tail-jump to callee2 entry at 0x30, which branches backward to 0x28!)
    //   0x14: add r0, r0, #7 (callee1 cold prefix `< 0x20`!)
    //   0x18: bx lr
    //   0x1C: nop
    //   0x20: add r0, r0, #3 (callee1 TRUE entry!)
    //   0x24: b 0x14         (backward branch to 0x14: offset = (0x14 - (0x24 + 8)) / 4 = -6 -> 0xEAFF_FFFA)
    //   0x28: add r0, r0, #100 (callee2 cold prefix `< 0x30`!)
    //   0x2C: bx lr
    //   0x30: add r0, r0, #50  (callee2 TRUE entry!)
    //   0x34: b 0x28           (backward branch to 0x28: offset = (0x28 - (0x34 + 8)) / 4 = -5 -> 0xEAFF_FFFB)
    let words = [
        0xE92D_4010, // 0x00: push {r4, lr}
        0xE3A0_000A, // 0x04: mov r0, #10
        0xEB00_0004, // 0x08: bl 0x20 ((0x08 + 8) + 4*4 = 0x20)
        0xE8BD_4010, // 0x0C: pop {r4, lr}
        0xEA00_0006, // 0x10: b 0x30 ((0x10 + 8) + 6*4 = 0x30)
        0xE280_0007, // 0x14: add r0, r0, #7
        0xE12F_FF1E, // 0x18: bx lr
        0xE1A0_0000, // 0x1C: nop
        0xE280_0003, // 0x20: add r0, r0, #3
        0xEAFF_FFFA, // 0x24: b 0x14
        0xE280_0064, // 0x28: add r0, r0, #100
        0xE12F_FF1E, // 0x2C: bx lr
        0xE280_0032, // 0x30: add r0, r0, #50
        0xEAFF_FFFB, // 0x34: b 0x28
    ];
    run_snippet_test(
        "lower_prefix_block_call_and_tail",
        &words,
        &[(BASE, Source::Entry), (BASE + 0x20, Source::Call), (BASE + 0x30, Source::Entry)],
        |_, _| {},
    );
}

#[allow(dead_code)]
mod asm {
    pub const EQ: u32 = 0;
    pub const NE: u32 = 1;
    pub const CS: u32 = 2;
    pub const CC: u32 = 3;
    pub const MI: u32 = 4;
    pub const PL: u32 = 5;
    pub const VS: u32 = 6;
    pub const VC: u32 = 7;
    pub const HI: u32 = 8;
    pub const LS: u32 = 9;
    pub const GE: u32 = 10;
    pub const LT: u32 = 11;
    pub const GT: u32 = 12;
    pub const LE: u32 = 13;
    pub const AL: u32 = 14;

    pub const LSL: u32 = 0;
    pub const LSR: u32 = 1;
    pub const ASR: u32 = 2;
    pub const ROR: u32 = 3;

    pub const OP_AND: u32 = 0;
    pub const OP_EOR: u32 = 1;
    pub const OP_SUB: u32 = 2;
    pub const OP_RSB: u32 = 3;
    pub const OP_ADD: u32 = 4;
    pub const OP_ADC: u32 = 5;
    pub const OP_SBC: u32 = 6;
    pub const OP_RSC: u32 = 7;
    pub const OP_TST: u32 = 8;
    pub const OP_TEQ: u32 = 9;
    pub const OP_CMP: u32 = 10;
    pub const OP_CMN: u32 = 11;
    pub const OP_ORR: u32 = 12;
    pub const OP_MOV: u32 = 13;
    pub const OP_BIC: u32 = 14;
    pub const OP_MVN: u32 = 15;

    pub const fn dp_imm(cond: u32, op: u32, s: bool, rn: u32, rd: u32, imm8: u32, rot4: u32) -> u32 {
        (cond << 28) | (1 << 25) | (op << 21) | ((s as u32) << 20) | (rn << 16) | (rd << 12) | (rot4 << 8) | (imm8 & 0xFF)
    }

    pub const fn dp_reg_imm(
        cond: u32,
        op: u32,
        s: bool,
        rn: u32,
        rd: u32,
        rm: u32,
        shift_ty: u32,
        imm5: u32,
    ) -> u32 {
        (cond << 28)
            | (op << 21)
            | ((s as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | ((imm5 & 31) << 7)
            | ((shift_ty & 3) << 5)
            | (rm & 15)
    }

    pub const fn dp_reg_reg(
        cond: u32,
        op: u32,
        s: bool,
        rn: u32,
        rd: u32,
        rm: u32,
        shift_ty: u32,
        rs: u32,
    ) -> u32 {
        (cond << 28)
            | (op << 21)
            | ((s as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | ((rs & 15) << 8)
            | ((shift_ty & 3) << 5)
            | (1 << 4)
            | (rm & 15)
    }

    pub const fn mov_imm(cond: u32, s: bool, rd: u32, imm8: u32, rot4: u32) -> u32 {
        dp_imm(cond, OP_MOV, s, 0, rd, imm8, rot4)
    }

    pub const fn add_imm(cond: u32, s: bool, rd: u32, rn: u32, imm8: u32, rot4: u32) -> u32 {
        dp_imm(cond, OP_ADD, s, rn, rd, imm8, rot4)
    }

    pub const fn orr_imm(cond: u32, s: bool, rd: u32, rn: u32, imm8: u32, rot4: u32) -> u32 {
        dp_imm(cond, OP_ORR, s, rn, rd, imm8, rot4)
    }

    pub const fn cmp_reg(cond: u32, rn: u32, rm: u32) -> u32 {
        dp_reg_imm(cond, OP_CMP, true, rn, 0, rm, LSL, 0)
    }

    pub const fn cmn_reg(cond: u32, rn: u32, rm: u32) -> u32 {
        dp_reg_imm(cond, OP_CMN, true, rn, 0, rm, LSL, 0)
    }

    pub const fn b_cond(cond: u32, cur_addr: u32, target: u32) -> u32 {
        let off = (target.wrapping_sub(cur_addr.wrapping_add(8)) as i32) >> 2;
        (cond << 28) | 0x0A00_0000 | ((off as u32) & 0x00FF_FFFF)
    }

    pub const fn bl_cond(cond: u32, cur_addr: u32, target: u32) -> u32 {
        let off = (target.wrapping_sub(cur_addr.wrapping_add(8)) as i32) >> 2;
        (cond << 28) | 0x0B00_0000 | ((off as u32) & 0x00FF_FFFF)
    }

    pub const fn blx_thumb(cur_addr: u32, target_thumb: u32) -> u32 {
        let diff = target_thumb.wrapping_sub(cur_addr.wrapping_add(8)) as i32;
        let h = ((diff >> 1) & 1) as u32;
        let imm24 = ((diff >> 2) as u32) & 0x00FF_FFFF;
        0xFA00_0000 | (h << 24) | imm24
    }

    pub const fn bx(cond: u32, rm: u32) -> u32 {
        (cond << 28) | 0x012F_FF10 | (rm & 15)
    }

    pub const fn blx_reg(cond: u32, rm: u32) -> u32 {
        (cond << 28) | 0x012F_FF30 | (rm & 15)
    }

    pub const fn mrs_cpsr(cond: u32, rd: u32) -> u32 {
        (cond << 28) | 0x010F_0000 | (rd << 12)
    }

    pub const fn msr_cpsr_f_imm(cond: u32, imm8: u32, rot4: u32) -> u32 {
        (cond << 28) | 0x0328_F000 | (rot4 << 8) | (imm8 & 0xFF)
    }

    pub const fn clz(cond: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x016F_0F10 | (rd << 12) | (rm & 15)
    }

    pub const fn ldr_str_imm(
        cond: u32,
        load: bool,
        byte: bool,
        pre: bool,
        up: bool,
        wb: bool,
        rn: u32,
        rd: u32,
        imm12: u32,
    ) -> u32 {
        (cond << 28)
            | (1 << 26)
            | ((pre as u32) << 24)
            | ((up as u32) << 23)
            | ((byte as u32) << 22)
            | ((wb as u32) << 21)
            | ((load as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | (imm12 & 0xFFF)
    }

    pub const fn extra_imm(
        cond: u32,
        load: bool,
        sh: u32,
        pre: bool,
        up: bool,
        wb: bool,
        rn: u32,
        rd: u32,
        imm8: u32,
    ) -> u32 {
        (cond << 28)
            | ((pre as u32) << 24)
            | ((up as u32) << 23)
            | (1 << 22)
            | ((wb as u32) << 21)
            | ((load as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | (((imm8 >> 4) & 0xF) << 8)
            | ((sh & 3) << 5)
            | 0x90
            | (imm8 & 0xF)
    }

    pub const fn ldm_stm(cond: u32, load: bool, pre: bool, up: bool, wb: bool, rn: u32, list: u32) -> u32 {
        (cond << 28)
            | (1 << 27)
            | ((pre as u32) << 24)
            | ((up as u32) << 23)
            | ((wb as u32) << 21)
            | ((load as u32) << 20)
            | (rn << 16)
            | (list & 0xFFFF)
    }

    pub const fn mul(cond: u32, acc: bool, s: bool, rd: u32, rn: u32, rs: u32, rm: u32) -> u32 {
        (cond << 28) | ((acc as u32) << 21) | ((s as u32) << 20) | (rd << 16) | (rn << 12) | (rs << 8) | 0x90 | rm
    }

    pub const fn mul_long(
        cond: u32,
        signed: bool,
        acc: bool,
        s: bool,
        rd_hi: u32,
        rd_lo: u32,
        rs: u32,
        rm: u32,
    ) -> u32 {
        (cond << 28)
            | (1 << 23)
            | ((signed as u32) << 22)
            | ((acc as u32) << 21)
            | ((s as u32) << 20)
            | (rd_hi << 16)
            | (rd_lo << 12)
            | (rs << 8)
            | 0x90
            | rm
    }

    pub const fn smlaxy(cond: u32, rd: u32, rn: u32, rs: u32, rm: u32, x: bool, y: bool) -> u32 {
        (cond << 28) | 0x0100_0080 | (rd << 16) | (rn << 12) | (rs << 8) | ((y as u32) << 6) | ((x as u32) << 5) | rm
    }

    pub const fn smlawy(cond: u32, rd: u32, rn: u32, rs: u32, rm: u32, y: bool) -> u32 {
        (cond << 28) | 0x0120_0080 | (rd << 16) | (rn << 12) | (rs << 8) | ((y as u32) << 6) | rm
    }

    pub const fn smulwy(cond: u32, rd: u32, rs: u32, rm: u32, y: bool) -> u32 {
        (cond << 28) | 0x0120_00A0 | (rd << 16) | (rs << 8) | ((y as u32) << 6) | rm
    }

    pub const fn q_arith(cond: u32, kind: u32, rn: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x0100_0050 | ((kind & 3) << 21) | (rn << 16) | (rd << 12) | rm
    }

    pub const fn ssat(cond: u32, sat_imm: u32, rd: u32, rm: u32, asr: bool, shift5: u32) -> u32 {
        (cond << 28) | 0x06A0_0010 | ((sat_imm & 31) << 16) | (rd << 12) | ((shift5 & 31) << 7) | ((asr as u32) << 6) | rm
    }

    pub const fn usat(cond: u32, sat_imm: u32, rd: u32, rm: u32, asr: bool, shift5: u32) -> u32 {
        (cond << 28) | 0x06E0_0010 | ((sat_imm & 31) << 16) | (rd << 12) | ((shift5 & 31) << 7) | ((asr as u32) << 6) | rm
    }

    pub const fn pkhbt(cond: u32, rn: u32, rd: u32, rm: u32, lsl5: u32) -> u32 {
        (cond << 28) | 0x0680_0010 | (rn << 16) | (rd << 12) | ((lsl5 & 31) << 7) | rm
    }

    pub const fn pkhtb(cond: u32, rn: u32, rd: u32, rm: u32, asr5: u32) -> u32 {
        (cond << 28) | 0x0680_0050 | (rn << 16) | (rd << 12) | ((asr5 & 31) << 7) | rm
    }

    pub const fn rev(cond: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x06BF_0F30 | (rd << 12) | rm
    }

    pub const fn rev16(cond: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x06BF_0FB0 | (rd << 12) | rm
    }

    pub const fn revsh(cond: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x06FF_0FB0 | (rd << 12) | rm
    }

    pub const fn media_parallel(cond: u32, op1: u32, op2: u32, rn: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x0600_0F10 | ((op1 & 31) << 20) | (rn << 16) | (rd << 12) | ((op2 & 7) << 5) | rm
    }

    pub const fn sel(cond: u32, rn: u32, rd: u32, rm: u32) -> u32 {
        (cond << 28) | 0x0680_0FB0 | (rn << 16) | (rd << 12) | rm
    }

    pub const fn usad8(cond: u32, rd: u32, rs: u32, rm: u32) -> u32 {
        (cond << 28) | 0x0780_F010 | (rd << 16) | (rs << 8) | rm
    }

    pub const fn usada8(cond: u32, rd: u32, ra: u32, rs: u32, rm: u32) -> u32 {
        (cond << 28) | 0x0780_0010 | (rd << 16) | (ra << 12) | (rs << 8) | rm
    }

    // VFPv2 helpers
    const fn split_s(s: u32) -> (u32, u32) {
        ((s >> 1) & 0xF, s & 1)
    }

    const fn split_d(d: u32) -> (u32, u32) {
        (d & 0xF, (d >> 4) & 1)
    }

    pub const fn vfp_dp(cond: u32, double: bool, opc1: u32, negate: bool, rd: u32, rn: u32, rm: u32) -> u32 {
        let (vd, db) = if double { split_d(rd) } else { split_s(rd) };
        let (vn, nb) = if double { split_d(rn) } else { split_s(rn) };
        let (vm, mb) = if double { split_d(rm) } else { split_s(rm) };
        (cond << 28)
            | 0x0E00_0A00
            | ((double as u32) << 8)
            | ((opc1 & 0xB) << 20)
            | (db << 22)
            | (vn << 16)
            | (vd << 12)
            | (nb << 7)
            | ((negate as u32) << 6)
            | (mb << 5)
            | vm
    }

    pub const fn vfp_ext(cond: u32, double: bool, opc2: u32, top: bool, vd: u32, db: u32, vm: u32, mb: u32) -> u32 {
        (cond << 28)
            | 0x0EB0_0A40
            | ((double as u32) << 8)
            | (db << 22)
            | ((opc2 & 0xF) << 16)
            | (vd << 12)
            | ((top as u32) << 7)
            | (mb << 5)
            | vm
    }

    pub const fn vfp_unary(cond: u32, double: bool, opc2: u32, top: bool, rd: u32, rm: u32) -> u32 {
        let (vd, db) = if double { split_d(rd) } else { split_s(rd) };
        let (vm, mb) = if double { split_d(rm) } else { split_s(rm) };
        vfp_ext(cond, double, opc2, top, vd, db, vm, mb)
    }

    pub const fn vcmp(cond: u32, double: bool, e_bit: bool, rd: u32, rm: Option<u32>) -> u32 {
        let (vd, db) = if double { split_d(rd) } else { split_s(rd) };
        match rm {
            Some(r) => {
                let (vm, mb) = if double { split_d(r) } else { split_s(r) };
                vfp_ext(cond, double, 0b0100, e_bit, vd, db, vm, mb)
            }
            None => vfp_ext(cond, double, 0b0101, e_bit, vd, db, 0, 0),
        }
    }

    pub const fn vcvt_f32_f64(cond: u32, sd: u32, dm: u32) -> u32 {
        let (vd, db) = split_s(sd);
        let (vm, mb) = split_d(dm);
        vfp_ext(cond, true, 0b0111, true, vd, db, vm, mb)
    }

    pub const fn vcvt_f64_f32(cond: u32, dd: u32, sm: u32) -> u32 {
        let (vd, db) = split_d(dd);
        let (vm, mb) = split_s(sm);
        vfp_ext(cond, false, 0b0111, true, vd, db, vm, mb)
    }

    pub const fn vcvt_from_int(cond: u32, double: bool, signed: bool, dst: u32, sm: u32) -> u32 {
        let (vd, db) = if double { split_d(dst) } else { split_s(dst) };
        let (vm, mb) = split_s(sm);
        vfp_ext(cond, double, 0b1000, signed, vd, db, vm, mb)
    }

    pub const fn vcvt_to_int(cond: u32, double: bool, signed: bool, round_zero: bool, sd: u32, src: u32) -> u32 {
        let (vd, db) = split_s(sd);
        let (vm, mb) = if double { split_d(src) } else { split_s(src) };
        vfp_ext(cond, double, if signed { 0b1101 } else { 0b1100 }, round_zero, vd, db, vm, mb)
    }

    pub const fn fmstat(cond: u32) -> u32 {
        (cond << 28) | 0x0EF1_FA10
    }

    pub const fn vmsr_fpscr(cond: u32, rt: u32) -> u32 {
        (cond << 28) | 0x0EE1_0A10 | (rt << 12)
    }

    pub const fn vmrs_fpscr(cond: u32, rt: u32) -> u32 {
        (cond << 28) | 0x0EF1_0A10 | (rt << 12)
    }

    pub const fn vmov_s_rt(cond: u32, to_gpr: bool, sn: u32, rt: u32) -> u32 {
        let (vn, nb) = split_s(sn);
        (cond << 28) | 0x0E00_0A10 | ((to_gpr as u32) << 20) | (vn << 16) | (rt << 12) | (nb << 7)
    }

    pub const fn vmov_d_rpair(cond: u32, to_gpr: bool, dm: u32, rt: u32, rt2: u32) -> u32 {
        let (vm, mb) = split_d(dm);
        (cond << 28) | 0x0C40_0B10 | ((to_gpr as u32) << 20) | (rt2 << 16) | (rt << 12) | (mb << 5) | vm
    }

    // Thumb-1 helpers
    pub const fn t_lsls_imm(rd: u16, rm: u16, imm5: u16) -> u16 {
        ((imm5 & 31) << 6) | ((rm & 7) << 3) | (rd & 7)
    }

    pub const fn t_adds_reg(rd: u16, rn: u16, rm: u16) -> u16 {
        0x1800 | ((rm & 7) << 6) | ((rn & 7) << 3) | (rd & 7)
    }

    pub const fn t_adds_imm3(rd: u16, rn: u16, imm3: u16) -> u16 {
        0x1C00 | ((imm3 & 7) << 6) | ((rn & 7) << 3) | (rd & 7)
    }

    pub const fn t_movs_imm8(rd: u16, imm8: u16) -> u16 {
        0x2000 | ((rd & 7) << 8) | (imm8 & 0xFF)
    }

    pub const fn t_adds_imm8(rd: u16, imm8: u16) -> u16 {
        0x3000 | ((rd & 7) << 8) | (imm8 & 0xFF)
    }

    pub const fn t_alu(op4: u16, rd: u16, rs: u16) -> u16 {
        0x4000 | ((op4 & 0xF) << 6) | ((rs & 7) << 3) | (rd & 7)
    }

    pub const fn t_hi_add(rd: u16, rs: u16) -> u16 {
        0x4400 | (((rd >> 3) & 1) << 7) | ((rs & 15) << 3) | (rd & 7)
    }

    pub const fn t_hi_cmp(rd: u16, rs: u16) -> u16 {
        0x4500 | (((rd >> 3) & 1) << 7) | ((rs & 15) << 3) | (rd & 7)
    }

    pub const fn t_hi_mov(rd: u16, rs: u16) -> u16 {
        0x4600 | (((rd >> 3) & 1) << 7) | ((rs & 15) << 3) | (rd & 7)
    }

    pub const fn t_bx(rm: u16) -> u16 {
        0x4700 | ((rm & 15) << 3)
    }

    pub const fn t_blx_reg(rm: u16) -> u16 {
        0x4780 | ((rm & 15) << 3)
    }

    pub const fn t_nop() -> u16 {
        0x46C0
    }

    pub const fn t_push(regs8: u16, lr: bool) -> u16 {
        0xB400 | ((lr as u16) << 8) | (regs8 & 0xFF)
    }

    pub const fn t_pop(regs8: u16, pc: bool) -> u16 {
        0xBC00 | ((pc as u16) << 8) | (regs8 & 0xFF)
    }

    pub const fn t_ldmia(rb: u16, regs8: u16) -> u16 {
        0xC800 | ((rb & 7) << 8) | (regs8 & 0xFF)
    }

    pub const fn t_b_cond(cond: u16, cur_addr: u32, target: u32) -> u16 {
        let diff = (target.wrapping_sub(cur_addr.wrapping_add(4)) as i32) >> 1;
        0xD000 | ((cond & 0xF) << 8) | ((diff as u16) & 0xFF)
    }

    pub const fn t_b(cur_addr: u32, target: u32) -> u16 {
        let diff = (target.wrapping_sub(cur_addr.wrapping_add(4)) as i32) >> 1;
        0xE000 | ((diff as u16) & 0x7FF)
    }

    pub const fn t_bl(cur_addr: u32, target: u32) -> [u16; 2] {
        let diff = target.wrapping_sub(cur_addr.wrapping_add(4)) as i32;
        let high = ((diff >> 12) as u16) & 0x7FF;
        let low = ((diff >> 1) as u16) & 0x7FF;
        [0xF000 | high, 0xF800 | low]
    }

    pub const fn t_blx(cur_addr: u32, target_arm: u32) -> [u16; 2] {
        // Use `(cur_addr + 4) & !3` so `low` has bit 0 == 0 even when `cur_addr % 4 == 2`.
        let pc_aligned = cur_addr.wrapping_add(4) & !3;
        let diff = target_arm.wrapping_sub(pc_aligned) as i32;
        let high = ((diff >> 12) as u16) & 0x7FF;
        let low = ((diff >> 1) as u16) & 0x7FE;
        [0xF000 | high, 0xE800 | low]
    }
}

struct CodeBuilder {
    bytes: Vec<u8>,
}

impl CodeBuilder {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn addr(&self) -> u32 {
        BASE + self.bytes.len() as u32
    }

    fn align4(&mut self) {
        if self.bytes.len() % 4 != 0 {
            self.emit_thumb(asm::t_nop());
        }
    }

    fn emit_arm(&mut self, op: u32) -> u32 {
        assert_eq!(self.bytes.len() % 4, 0);
        let a = self.addr();
        self.bytes.extend_from_slice(&op.to_le_bytes());
        a
    }

    fn emit_thumb(&mut self, op: u16) -> u32 {
        assert_eq!(self.bytes.len() % 2, 0);
        let a = self.addr();
        self.bytes.extend_from_slice(&op.to_le_bytes());
        a
    }

    fn emit_thumb_pair(&mut self, pair: [u16; 2]) -> u32 {
        let a = self.emit_thumb(pair[0]);
        self.emit_thumb(pair[1]);
        a
    }

    fn patch_arm(&mut self, addr: u32, op: u32) {
        let off = (addr - BASE) as usize;
        self.bytes[off..off + 4].copy_from_slice(&op.to_le_bytes());
    }

    fn patch_thumb(&mut self, addr: u32, op: u16) {
        let off = (addr - BASE) as usize;
        self.bytes[off..off + 2].copy_from_slice(&op.to_le_bytes());
    }

    fn patch_thumb_pair(&mut self, addr: u32, pair: [u16; 2]) {
        self.patch_thumb(addr, pair[0]);
        self.patch_thumb(addr + 2, pair[1]);
    }

    fn finish(mut self) -> Vec<u32> {
        self.align4();
        self.bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

#[test]
fn torture_cond_codes_and_borrow_chains() {
    use asm::*;

    // Entry 1: test all 14 condition codes (EQ..LE) via both conditional ALU and B<cond>,
    // plus verify skipped conditional flag-setting instructions leave NZCV intact.
    let mut b = CodeBuilder::new();
    let e_cond = b.addr();
    b.emit_arm(mov_imm(AL, false, 4, 0, 0)); // r4 = conditional ALU mask
    b.emit_arm(mov_imm(AL, false, 5, 0, 0)); // r5 = conditional B<cond> mask
    for cond in 0..14u32 {
        let (imm8, rot4) = if cond < 8 { (1 << cond, 0) } else { (1 << (cond - 8), 12) };
        b.emit_arm(orr_imm(cond, false, 4, 4, imm8, rot4));
    }
    for cond in 0..14u32 {
        let (imm8, rot4) = if cond < 8 { (1 << cond, 0) } else { (1 << (cond - 8), 12) };
        let j_taken = b.emit_arm(0);
        let j_skip = b.emit_arm(0);
        let taken_a = b.emit_arm(orr_imm(AL, false, 5, 5, imm8, rot4));
        let done_a = b.addr();
        b.patch_arm(j_taken, b_cond(cond, j_taken, taken_a));
        b.patch_arm(j_skip, b_cond(AL, j_skip, done_a));
    }
    b.emit_arm(mrs_cpsr(AL, 6));
    // Conditional ADDS (S=1) using r8=0xFFFF_FFFF, r9=1: when skipped, NZCV must remain unchanged!
    b.emit_arm(dp_reg_imm(NE, OP_ADD, true, 8, 10, 9, LSL, 0));
    b.emit_arm(mrs_cpsr(AL, 7));
    b.emit_arm(dp_reg_imm(EQ, OP_SUB, true, 8, 11, 9, LSL, 0));
    b.emit_arm(bx(AL, 14));

    // Entry 2..4: long ADD/ADDS/ADC/ADCS/SUB/SUBS/SBC/SBCS/RSC/RSCS/CMP/CMN chains.
    // Separate entries start directly with ADCS, SBCS, or RSCS so initial C=0/C=1 feeds each directly.
    let emit_chain = |b: &mut CodeBuilder, first_op: u32| -> u32 {
        let entry = b.addr();
        b.emit_arm(dp_reg_imm(AL, first_op, true, 0, 3, 1, LSL, 0)); // consumes initial C
        b.emit_arm(mrs_cpsr(AL, 11));
        b.emit_arm(dp_reg_imm(AL, OP_ADD, true, 0, 4, 1, LSL, 0)); // ADDS
        b.emit_arm(dp_reg_imm(AL, OP_ADD, false, 2, 2, 4, LSL, 0)); // ADD (preserves NZCV)
        b.emit_arm(dp_reg_imm(AL, OP_ADC, true, 3, 5, 1, LSL, 0)); // ADCS
        b.emit_arm(dp_reg_imm(AL, OP_ADC, false, 2, 2, 5, LSL, 0)); // ADC (preserves NZCV)
        b.emit_arm(dp_reg_imm(AL, OP_SUB, true, 0, 6, 1, LSL, 0)); // SUBS
        b.emit_arm(dp_reg_imm(AL, OP_SUB, false, 2, 2, 6, LSL, 0)); // SUB (preserves NZCV)
        b.emit_arm(dp_reg_imm(AL, OP_SBC, true, 5, 7, 1, LSL, 0)); // SBCS
        b.emit_arm(dp_reg_imm(AL, OP_SBC, false, 2, 2, 7, LSL, 0)); // SBC (preserves NZCV)
        b.emit_arm(dp_reg_imm(AL, OP_RSC, true, 0, 8, 1, LSL, 0)); // RSCS
        b.emit_arm(dp_reg_imm(AL, OP_RSC, false, 2, 2, 8, LSL, 0)); // RSC (preserves NZCV)
        b.emit_arm(dp_reg_imm(AL, OP_SBC, true, 7, 9, 8, LSL, 0)); // SBCS
        b.emit_arm(dp_reg_imm(AL, OP_RSC, true, 6, 10, 9, LSL, 0)); // RSCS
        b.emit_arm(cmp_reg(AL, 0, 1));
        b.emit_arm(add_imm(CS, false, 2, 2, 1, 0));
        b.emit_arm(add_imm(VS, false, 2, 2, 2, 0));
        b.emit_arm(add_imm(MI, false, 2, 2, 4, 0));
        b.emit_arm(add_imm(EQ, false, 2, 2, 8, 0));
        b.emit_arm(add_imm(HI, false, 2, 2, 16, 0));
        b.emit_arm(add_imm(GE, false, 2, 2, 32, 0));
        b.emit_arm(add_imm(GT, false, 2, 2, 64, 0));
        b.emit_arm(dp_reg_imm(AL, OP_SBC, true, 0, 12, 1, LSL, 0)); // consumes C from CMP
        b.emit_arm(cmn_reg(AL, 0, 1));
        b.emit_arm(add_imm(CS, false, 2, 2, 1, 12)); // +0x100
        b.emit_arm(add_imm(VS, false, 2, 2, 2, 12)); // +0x200
        b.emit_arm(add_imm(MI, false, 2, 2, 4, 12)); // +0x400
        b.emit_arm(add_imm(EQ, false, 2, 2, 8, 12)); // +0x800
        b.emit_arm(dp_reg_imm(AL, OP_ADC, true, 12, 12, 0, LSL, 0)); // consumes C from CMN
        b.emit_arm(bx(AL, 14));
        entry
    };
    let e_adcs = emit_chain(&mut b, OP_ADC);
    let e_sbcs = emit_chain(&mut b, OP_SBC);
    let e_rscs = emit_chain(&mut b, OP_RSC);

    let words = b.finish();
    let seeds = [
        (e_cond, Source::Entry),
        (e_adcs, Source::Entry),
        (e_sbcs, Source::Entry),
        (e_rscs, Source::Entry),
    ];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    for flags in 0..16u32 {
        verify_compiled(&format!("cond_codes_{flags:X}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_cond;
            cpu.regs[8] = 0xFFFF_FFFF;
            cpu.regs[9] = 1;
            cpu.cpsr.n = (flags & 8) != 0;
            cpu.cpsr.z = (flags & 4) != 0;
            cpu.cpsr.c = (flags & 2) != 0;
            cpu.cpsr.v = (flags & 1) != 0;
        });
    }

    let edge_vals = [0u32, 1, 2, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    for (tag, entry) in [("adcs", e_adcs), ("sbcs", e_sbcs), ("rscs", e_rscs)] {
        for &x in &edge_vals {
            for &y in &edge_vals {
                for &init_c in &[false, true] {
                    verify_compiled(
                        &format!("borrow_chain_{tag}_{x:08X}_{y:08X}_c{}", init_c as u8),
                        &text_bytes,
                        &aot,
                        true,
                        |cpu, _| {
                            cpu.regs[15] = entry;
                            cpu.regs[0] = x;
                            cpu.regs[1] = y;
                            cpu.regs[2] = 0x1357_9BDF;
                            cpu.cpsr.c = init_c;
                            cpu.cpsr.v = !init_c;
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn torture_barrel_shifts_boundary_counts() {
    use asm::*;

    let mut b = CodeBuilder::new();
    let mut reg_entries = [0u32; 4];
    for (idx, &shift_ty) in [LSL, LSR, ASR, ROR].iter().enumerate() {
        reg_entries[idx] = b.addr();
        b.emit_arm(dp_reg_reg(AL, OP_MOV, true, 0, 4, 0, shift_ty, 2)); // MOVS r4, r0, <op> r2
        b.emit_arm(mrs_cpsr(AL, 7)); // captures exact N,Z,C, and preserved V
        b.emit_arm(dp_reg_reg(AL, OP_ADD, true, 1, 5, 0, shift_ty, 2)); // ADDS r5, r1, r0, <op> r2
        b.emit_arm(mrs_cpsr(AL, 8));
        b.emit_arm(dp_reg_reg(AL, OP_SUB, true, 1, 6, 0, shift_ty, 2)); // SUBS r6, r1, r0, <op> r2
        b.emit_arm(bx(AL, 14));
    }

    let e_rrx = b.addr();
    b.emit_arm(dp_reg_imm(AL, OP_MOV, true, 0, 4, 0, ROR, 0)); // MOVS r4, r0, RRX
    b.emit_arm(mrs_cpsr(AL, 7));
    b.emit_arm(dp_reg_imm(AL, OP_ADD, true, 1, 5, 0, ROR, 0)); // ADDS r5, r1, r0, RRX
    b.emit_arm(mrs_cpsr(AL, 8));
    b.emit_arm(dp_reg_imm(AL, OP_SUB, true, 1, 6, 0, ROR, 0)); // SUBS r6, r1, r0, RRX
    b.emit_arm(bx(AL, 14));

    let e_rrx_preserve = b.addr();
    b.emit_arm(dp_reg_imm(AL, OP_MOV, false, 0, 4, 0, ROR, 0)); // MOV (S=0) r4, r0, RRX (preserves C!)
    b.emit_arm(dp_reg_imm(AL, OP_ADC, true, 1, 5, 0, ROR, 0)); // ADCS r5, r1, r0, RRX
    b.emit_arm(bx(AL, 14));

    let words = b.finish();
    let seeds = [
        (reg_entries[0], Source::Entry),
        (reg_entries[1], Source::Entry),
        (reg_entries[2], Source::Entry),
        (reg_entries[3], Source::Entry),
        (e_rrx, Source::Entry),
        (e_rrx_preserve, Source::Entry),
    ];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    let counts = [0u32, 1, 7, 16, 31, 32, 33, 64, 255, 256];
    let vals = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFF];

    for (shift_name, &entry) in ["lsl", "lsr", "asr", "ror"].iter().zip(&reg_entries) {
        for &rs in &counts {
            for &val in &vals {
                for &init_c in &[false, true] {
                    verify_compiled(
                        &format!("barrel_{shift_name}_rs{rs}_{val:08X}_c{}", init_c as u8),
                        &text_bytes,
                        &aot,
                        true,
                        |cpu, _| {
                            cpu.regs[15] = entry;
                            cpu.regs[0] = val;
                            cpu.regs[1] = 0x8000_0001;
                            cpu.regs[2] = rs;
                            cpu.cpsr.c = init_c;
                            cpu.cpsr.v = !init_c;
                        },
                    );
                }
            }
        }
    }

    for (tag, entry) in [("rrx_s", e_rrx), ("rrx_preserve", e_rrx_preserve)] {
        for &val in &vals {
            for &init_c in &[false, true] {
                verify_compiled(
                    &format!("barrel_{tag}_{val:08X}_c{}", init_c as u8),
                    &text_bytes,
                    &aot,
                    true,
                    |cpu, _| {
                        cpu.regs[15] = entry;
                        cpu.regs[0] = val;
                        cpu.regs[1] = 0x8000_0001;
                        cpu.cpsr.c = init_c;
                        cpu.cpsr.v = !init_c;
                    },
                );
            }
        }
    }
}

#[test]
fn torture_unaligned_crosspage_and_slowpath_preserves_nzcv() {
    use asm::*;

    let mut b = CodeBuilder::new();

    // Entry 1: Unaligned LDR/STR/LDRH/STRH (+0..+3) and HEAP page-boundary (+0xFFD, +0xFFE, +0xFFF).
    let e_unaligned = b.addr();
    // r0 = HEAP + 0x100, r1 = HEAP + 0xFFD, r2 = 0x1122_3344, r3 = 0xAABB_CCDD
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 0, 2, 0)); // str r2, [r0]
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 0, 3, 4)); // str r3, [r0, #4]
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 0, 4, 0)); // ldr r4, [r0, #0]
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 0, 5, 1)); // ldr r5, [r0, #1]
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 0, 6, 2)); // ldr r6, [r0, #2]
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 0, 7, 3)); // ldr r7, [r0, #3]
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 0, 2, 9)); // str r2, [r0, #9]
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 0, 3, 13)); // strh r3, [r0, #13]
    b.emit_arm(extra_imm(AL, true, 1, true, true, false, 0, 8, 9)); // ldrh r8, [r0, #9]
    b.emit_arm(extra_imm(AL, true, 3, true, true, false, 0, 9, 13)); // ldrsh r9, [r0, #13]
    // Page boundary on HEAP: r1 = HEAP + 0xFFD (so +0 = 0xFFD, +1 = 0xFFE, +2 = 0xFFF)
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 1, 2, 0)); // str r2, [r1, #0] (0xFFD)
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 1, 10, 0)); // ldr r10, [r1, #0] (0xFFD)
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 1, 3, 0)); // strh r3, [r1, #0] (0xFFD)
    b.emit_arm(extra_imm(AL, true, 1, true, true, false, 1, 11, 0)); // ldrh r11, [r1, #0] (0xFFD)
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 1, 3, 1)); // str r3, [r1, #1] (0xFFE)
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 1, 12, 1)); // ldr r12, [r1, #1] (0xFFE)
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 1, 2, 1)); // strh r2, [r1, #1] (0xFFE)
    b.emit_arm(extra_imm(AL, true, 1, true, true, false, 1, 4, 1)); // ldrh r4, [r1, #1] (0xFFE)
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 1, 2, 2)); // str r2, [r1, #2] (0xFFF)
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 1, 5, 2)); // ldr r5, [r1, #2] (0xFFF)
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 1, 3, 2)); // strh r3, [r1, #2] (0xFFF)
    b.emit_arm(extra_imm(AL, true, 1, true, true, false, 1, 6, 2)); // ldrh r6, [r1, #2] (0xFFF)
    b.emit_arm(extra_imm(AL, true, 3, true, true, false, 1, 7, 2)); // ldrsh r7, [r1, #2] (0xFFF)
    // Rn == Rd post-indexed cross-page load: r1 must receive loaded word, not writeback!
    b.emit_arm(ldr_str_imm(AL, true, false, false, true, false, 1, 1, 4)); // ldr r1, [r1], #4
    b.emit_arm(bx(AL, 14));

    // Entry 2: Slow-path (UNMAPPED_SCRATCH + cross-page HEAP +0xFFD/+0xFFF) with live NZCV!
    // Every memory op runs without touching NZCV; conditional instructions + ADCS check flags after ops.
    let e_slow_nzcv = b.addr();
    let check_conds = |b: &mut CodeBuilder, acc: u32, shift_rot4: u32| {
        b.emit_arm(add_imm(MI, false, acc, acc, 1, shift_rot4));
        b.emit_arm(add_imm(EQ, false, acc, acc, 2, shift_rot4));
        b.emit_arm(add_imm(CS, false, acc, acc, 4, shift_rot4));
        b.emit_arm(add_imm(VS, false, acc, acc, 8, shift_rot4));
        b.emit_arm(add_imm(HI, false, acc, acc, 16, shift_rot4));
        b.emit_arm(add_imm(GE, false, acc, acc, 32, shift_rot4));
        b.emit_arm(add_imm(GT, false, acc, acc, 64, shift_rot4));
        b.emit_arm(add_imm(LE, false, acc, acc, 128, shift_rot4));
    };
    // r0 = UNMAPPED_SCRATCH + 0xFFD, r1 = UNMAPPED_SCRATCH + 0x100, r12 = HEAP + 0xFFD
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 0, 2, 0)); // slow str @ 0xFFD
    check_conds(&mut b, 10, 0);
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 0, 4, 0)); // slow ldr @ 0xFFD
    check_conds(&mut b, 10, 12);
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 0, 3, 2)); // slow strh @ 0xFFF
    check_conds(&mut b, 10, 8);
    b.emit_arm(extra_imm(AL, true, 3, true, true, false, 0, 5, 2)); // slow ldrsh @ 0xFFF
    b.emit_arm(ldr_str_imm(AL, false, true, true, true, false, 1, 2, 1)); // slow strb @ +0x101
    b.emit_arm(ldr_str_imm(AL, true, true, true, true, false, 1, 6, 1)); // slow ldrb @ +0x101
    b.emit_arm(extra_imm(AL, true, 2, true, true, false, 1, 7, 1)); // slow ldrsb @ +0x101
    b.emit_arm(extra_imm(AL, false, 3, true, true, false, 1, 2, 0)); // slow strd r2, [r1]
    b.emit_arm(extra_imm(AL, false, 2, true, true, false, 1, 8, 0)); // slow ldrd r8, [r1]
    // Also cross-page fast-path slow-fallback on HEAP + 0xFFD and + 0xFFF with live NZCV!
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 12, 2, 0)); // heap cross str @ 0xFFD
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 12, 11, 2)); // heap cross ldr @ 0xFFF
    b.emit_arm(extra_imm(AL, false, 1, true, true, false, 12, 3, 2)); // heap cross strh @ 0xFFF
    check_conds(&mut b, 10, 4);
    // Leave NZCV directly live at exit so compare_runs verifies N, Z, C, V!
    b.emit_arm(bx(AL, 14));

    // Entry 3: Compute NZCV inline with ADDS/SUBS, cross slow-path ops, then consume C via ADCS/SBCS.
    let e_slow_adc = b.addr();
    b.emit_arm(dp_reg_imm(AL, OP_ADD, true, 4, 4, 5, LSL, 0)); // sets live NZCV
    b.emit_arm(ldr_str_imm(AL, false, false, true, true, false, 1, 4, 0)); // slow str
    b.emit_arm(ldr_str_imm(AL, true, false, true, true, false, 1, 6, 0)); // slow ldr
    b.emit_arm(dp_reg_imm(AL, OP_ADC, true, 6, 7, 5, LSL, 0)); // ADCS consuming preserved C
    b.emit_arm(ldr_str_imm(AL, false, true, true, true, false, 1, 7, 3)); // slow strb
    b.emit_arm(dp_reg_imm(AL, OP_SBC, true, 7, 8, 4, LSL, 0)); // SBCS consuming preserved C
    b.emit_arm(bx(AL, 14));

    let words = b.finish();
    let seeds = [
        (e_unaligned, Source::Entry),
        (e_slow_nzcv, Source::Entry),
        (e_slow_adc, Source::Entry),
    ];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    verify_compiled("unaligned_and_crosspage", &text_bytes, &aot, true, |cpu, mem| {
        cpu.regs[15] = e_unaligned;
        cpu.regs[0] = HEAP + 0x100;
        cpu.regs[1] = HEAP + 0xFFD;
        cpu.regs[2] = 0x1122_3344;
        cpu.regs[3] = 0xAABB_CCDD;
        for i in 0..32u32 {
            mem.write8(HEAP + 0xFF0 + i, (0x80u32.wrapping_add(i * 13)) as u8);
        }
    });

    for flags in 0..16u32 {
        verify_compiled(&format!("slowpath_nzcv_{flags:X}"), &text_bytes, &aot, true, |cpu, mem| {
            cpu.regs[15] = e_slow_nzcv;
            cpu.regs[0] = UNMAPPED_SCRATCH + 0xFFD;
            cpu.regs[1] = UNMAPPED_SCRATCH + 0x100;
            cpu.regs[2] = 0x89AB_CDEF;
            cpu.regs[3] = 0x7654_3210;
            cpu.regs[10] = 0;
            cpu.regs[12] = HEAP + 0xFFD;
            cpu.cpsr.n = (flags & 8) != 0;
            cpu.cpsr.z = (flags & 4) != 0;
            cpu.cpsr.c = (flags & 2) != 0;
            cpu.cpsr.v = (flags & 1) != 0;
            mem.write32(UNMAPPED_SCRATCH + 0xFFD, 0xCAFE_BABE);
        });
    }

    for &(a_val, b_val) in &[
        (0xFFFF_FFFFu32, 1u32),
        (0x7FFF_FFFF, 1),
        (0x8000_0000, 0x8000_0000),
        (0, 0),
    ] {
        verify_compiled(
            &format!("slowpath_adc_{a_val:08X}_{b_val:08X}"),
            &text_bytes,
            &aot,
            true,
            |cpu, _| {
                cpu.regs[15] = e_slow_adc;
                cpu.regs[1] = UNMAPPED_SCRATCH + 0x200;
                cpu.regs[4] = a_val;
                cpu.regs[5] = b_val;
            },
        );
    }
}

#[test]
fn torture_block_transfer_base_in_list_and_thumb_interwork() {
    use asm::*;

    let mut b = CodeBuilder::new();

    // Helper callees for Thumb <-> ARM interworking
    let thumb_leaf = b.addr();
    b.emit_thumb(t_push(0b0011_0000, true)); // push {r4, r5, lr}
    b.emit_thumb(t_movs_imm8(4, 17));
    b.emit_thumb(t_adds_reg(0, 0, 4));
    b.emit_thumb(t_pop(0b0011_0000, true)); // pop {r4, r5, pc}

    let thumb_push_pop_leaf = b.addr();
    b.emit_thumb(t_push(0b1111_0000, true)); // push {r4, r5, r6, r7, lr} (crosses page!)
    b.emit_thumb(t_movs_imm8(4, 11));
    b.emit_thumb(t_movs_imm8(5, 22));
    b.emit_thumb(t_adds_reg(0, 0, 4));
    b.emit_thumb(t_adds_reg(0, 0, 5));
    b.emit_thumb(t_pop(0b1111_0000, true)); // pop {r4, r5, r6, r7, pc}

    let thumb_callee_blx_reg = b.addr();
    b.emit_thumb(t_adds_imm8(0, 55));
    b.emit_thumb(t_bx(14));

    b.align4();
    let arm_callee_from_thumb = b.addr();
    b.emit_arm(add_imm(AL, false, 0, 0, 100, 0));
    b.emit_arm(bx(AL, 14));

    let arm_callee_blx_reg = b.addr();
    b.emit_arm(add_imm(AL, false, 0, 0, 200, 0));
    b.emit_arm(bx(AL, 14));

    // Thumb main target (entered via ARM `ldmia r4!, {r0, pc}` with bit 0 = 1!)
    let thumb_main = b.addr();
    b.emit_thumb(t_push(0b1111_0000, true)); // push {r4-r7, lr} (saves RETURN)
    // Load test inputs from r4 (pointing into HEAP after {r0, pc}) -> r1, r2, r3
    b.emit_thumb(t_ldmia(4, 0b0000_1110)); // ldmia r4!, {r1, r2, r3}
    // Thumb ALU: ADCS, SBCS, NEGS, LSLS, LSRS, ASRS, RORS
    b.emit_thumb(t_movs_imm8(5, 0xFF));
    b.emit_thumb(t_adds_imm8(5, 1)); // sets C=0, Z=0 -> wait: 0xFF+1 = 0x100, C=0
    b.emit_thumb(t_movs_imm8(6, 1));
    b.emit_thumb(t_lsls_imm(6, 6, 31)); // r6 = 0x8000_0000, C=0
    b.emit_thumb(t_adds_reg(7, 6, 6)); // 0x8000_0000 + 0x8000_0000 = 0, C=1, Z=1, V=1
    b.emit_thumb(t_alu(0x5, 0, 1)); // adcs r0, r1 (C=1)
    b.emit_thumb(t_alu(0x6, 0, 5)); // sbcs r0, r5
    b.emit_thumb(t_alu(0x9, 7, 6)); // negs r7, r6 (0 - 0x8000_0000 = 0x8000_0000, V=1, C=0)
    b.emit_thumb(t_alu(0x2, 0, 2)); // lsls r0, r2
    b.emit_thumb(t_alu(0x3, 0, 2)); // lsrs r0, r2
    b.emit_thumb(t_alu(0x4, 7, 2)); // asrs r7, r2
    b.emit_thumb(t_alu(0x7, 0, 2)); // rors r0, r2
    b.emit_thumb(t_adds_reg(0, 0, 7));

    // Thumb high-reg ADD / CMP / MOV / BLX Rm
    b.emit_thumb(t_hi_mov(8, 0)); // mov r8, r0
    b.emit_thumb(t_hi_add(8, 1)); // add r8, r1
    b.emit_thumb(t_hi_cmp(8, 1)); // cmp r8, r1 (updates NZCV)
    let t_branch_skip = b.emit_thumb(0);
    b.emit_thumb(t_adds_imm3(0, 0, 3));
    let t_after_cmp = b.addr();
    b.patch_thumb(t_branch_skip, t_b_cond(EQ as u16, t_branch_skip, t_after_cmp));

    // High-reg BLX Rm -> ARM (r2 holds arm_callee_blx_reg), then Thumb (r3 holds thumb_callee_blx_reg | 1)
    b.emit_thumb(t_hi_mov(9, 2));
    b.emit_thumb(t_blx_reg(9));
    b.emit_thumb(t_hi_mov(9, 3));
    b.emit_thumb(t_blx_reg(9));

    // Thumb 32-bit BL backward (to thumb_leaf) and forward (to thumb_leaf_fwd)
    b.emit_thumb_pair(t_bl(b.addr(), thumb_leaf));
    let j_fwd_bl = b.emit_thumb_pair([0, 0]);

    // Thumb 32-bit BLX at 4k-aligned address AND at 4k+2 address!
    b.align4();
    b.emit_thumb_pair(t_blx(b.addr(), arm_callee_from_thumb));
    b.emit_thumb(t_nop()); // pushes next instruction to 4k + 2!
    assert_eq!(b.addr() % 4, 2);
    b.emit_thumb_pair(t_blx(b.addr(), arm_callee_from_thumb));

    // Thumb PUSH/POP across stack page boundary (`HEAP + 0x1008`)
    b.emit_thumb(t_hi_mov(6, 13)); // save SP in r6
    b.emit_thumb(t_hi_mov(13, 11)); // SP = HEAP + 0x1008
    b.emit_thumb_pair(t_bl(b.addr(), thumb_push_pop_leaf));
    b.emit_thumb(t_hi_mov(13, 6)); // restore SP

    // High-reg BX r8 (jump forward to `thumb_after_bx`) using r10 holding target|1
    b.emit_thumb(t_hi_mov(8, 10));
    b.emit_thumb(t_bx(8));
    b.emit_thumb(t_movs_imm8(0, 0xEE)); // skipped!

    let thumb_after_bx = b.addr();
    b.emit_thumb(t_adds_imm8(0, 9));
    b.emit_thumb(t_pop(0b1111_0000, true)); // pop {r4-r7, pc} -> returns to RETURN (ARM)

    let thumb_leaf_fwd = b.addr();
    b.emit_thumb(t_adds_imm8(0, 29));
    b.emit_thumb(t_bx(14));
    b.patch_thumb_pair(j_fwd_bl, t_bl(j_fwd_bl, thumb_leaf_fwd));

    // Thumb ALU shift boundary entry (`thumb_shifts_entry`)
    b.align4();
    let thumb_shifts_entry = b.addr();
    // Inputs: r0 = val, r1 = acc, r2 = shift_amt, initial C
    b.emit_thumb(t_push(0b0011_0000, true)); // push {r4, r5, lr}
    b.emit_thumb(t_hi_mov(4, 0)); // r4 = val
    b.emit_thumb(t_alu(0x2, 4, 2)); // lsls r4, r2
    b.emit_thumb(t_alu(0x5, 1, 4)); // adcs r1, r4
    b.emit_thumb(t_hi_mov(5, 0)); // r5 = val
    b.emit_thumb(t_alu(0x3, 5, 2)); // lsrs r5, r2
    b.emit_thumb(t_alu(0x6, 1, 5)); // sbcs r1, r5
    b.emit_thumb(t_hi_mov(4, 0));
    b.emit_thumb(t_alu(0x4, 4, 2)); // asrs r4, r2
    b.emit_thumb(t_alu(0x5, 1, 4)); // adcs r1, r4
    b.emit_thumb(t_hi_mov(5, 0));
    b.emit_thumb(t_alu(0x7, 5, 2)); // rors r5, r2
    b.emit_thumb(t_alu(0x6, 1, 5)); // sbcs r1, r5
    b.emit_thumb(t_pop(0b0011_0000, true));

    // ARM entry: LDMIA/STMDB writeback + base in reglist (fast-path HEAP + slow-path UNMAPPED_SCRATCH),
    // then `ldmia r4!, {r0, pc}` switching to Thumb `thumb_main`!
    b.align4();
    let arm_entry = b.addr();
    // r4 = HEAP + 0x100, r7 = UNMAPPED_SCRATCH + 0x100
    // 1) STMDB r4!, {r4, r5, r6}: base r4 IS lowest -> stores OLD r4!
    b.emit_arm(ldm_stm(AL, false, true, false, true, 4, (1 << 4) | (1 << 5) | (1 << 6)));
    // 2) STMDB r4!, {r2, r4, r6}: base r4 is NOT lowest -> stores WRITTEN-BACK r4!
    b.emit_arm(ldm_stm(AL, false, true, false, true, 4, (1 << 2) | (1 << 4) | (1 << 6)));
    // 3) STMIA r7!, {r1, r7, r12} (slow-path): base r7 NOT lowest -> stores WRITTEN-BACK r7!
    b.emit_arm(ldm_stm(AL, false, false, true, true, 7, (1 << 1) | (1 << 7) | (1 << 12)));
    // 4) STMIA r4!, {r4, pc}: base r4 lowest + PC stored as instr_addr + 12!
    b.emit_arm(ldm_stm(AL, false, false, true, true, 4, (1 << 4) | (1 << 15)));
    // 5) LDMIA r4!, {r2, r4, r5}: writeback + base r4 in reglist -> r4 keeps LOADED value!
    b.emit_arm(ldm_stm(AL, true, false, true, true, 4, (1 << 2) | (1 << 4) | (1 << 5)));
    // 6) LDMIA r7!, {r7, r8} (slow-path): writeback + base r7 in reglist -> r7 keeps LOADED value!
    b.emit_arm(ldm_stm(AL, true, false, true, true, 7, (1 << 7) | (1 << 8)));
    // 7) LDMIA r12!, {r0, r4, pc}: loads r0, r4 (for thumb_main's LDMIA) and interworks to Thumb!
    b.emit_arm(ldm_stm(AL, true, false, true, true, 12, (1 << 0) | (1 << 4) | (1 << 15)));

    let words = b.finish();
    let seeds = [
        (arm_entry, Source::Entry),
        (thumb_main | 1, Source::Entry),
        (thumb_after_bx | 1, Source::Entry),
        (thumb_callee_blx_reg | 1, Source::Entry),
        (thumb_shifts_entry | 1, Source::Entry),
        (arm_callee_blx_reg, Source::Entry),
        (arm_callee_from_thumb, Source::Call),
        (thumb_leaf | 1, Source::Call),
        (thumb_leaf_fwd | 1, Source::Call),
        (thumb_push_pop_leaf | 1, Source::Call),
    ];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    verify_compiled("block_base_in_list_and_thumb_interwork", &text_bytes, &aot, true, |cpu, mem| {
        cpu.regs[15] = arm_entry;
        cpu.regs[1] = 0x1111_2222;
        cpu.regs[2] = 0x3333_4444;
        cpu.regs[4] = HEAP + 0x100;
        cpu.regs[5] = 0x5555_6666;
        cpu.regs[6] = 0x7777_8888;
        cpu.regs[7] = UNMAPPED_SCRATCH + 0x100;
        cpu.regs[10] = thumb_after_bx | 1;
        cpu.regs[11] = HEAP + 0x1008;
        cpu.regs[12] = HEAP + 0x200;
        // Step 3 advances r7 by 12 to UNMAPPED_SCRATCH + 0x10C; step 6 reads 2 words there:
        mem.write32(UNMAPPED_SCRATCH + 0x10C, 0xBBBB_0001); // loaded into r7
        mem.write32(UNMAPPED_SCRATCH + 0x110, 0xBBBB_0002); // loaded into r8
        // Step 7 reads `{r0, r4, pc}` at HEAP + 0x200, then thumb_main reads `{r1, r2, r3}` at r4:
        mem.write32(HEAP + 0x200, 10); // r0
        mem.write32(HEAP + 0x204, HEAP + 0x20C); // r4
        mem.write32(HEAP + 0x208, thumb_main | 1); // pc -> Thumb!
        mem.write32(HEAP + 0x20C, 7); // r1
        mem.write32(HEAP + 0x210, arm_callee_blx_reg); // r2 (BLX r9 ARM target)
        mem.write32(HEAP + 0x214, thumb_callee_blx_reg | 1); // r3 (BLX r9 Thumb target)
    });

    for &rs in &[0u32, 1, 31, 32, 33, 255, 256] {
        for &init_c in &[false, true] {
            verify_compiled(
                &format!("thumb_alu_shifts_rs{rs}_c{}", init_c as u8),
                &text_bytes,
                &aot,
                true,
                |cpu, _| {
                    cpu.cpsr.thumb = true;
                    cpu.regs[15] = thumb_shifts_entry;
                    cpu.regs[0] = 0x8000_0001;
                    cpu.regs[1] = 0x1234_5678;
                    cpu.regs[2] = rs;
                    cpu.cpsr.c = init_c;
                    cpu.cpsr.v = !init_c;
                },
            );
        }
    }
}

#[test]
fn torture_multiply_saturate_media() {
    use asm::*;

    let mut b = CodeBuilder::new();
    let e_mul_sat = b.addr();
    // 32-bit and 64-bit multiplies (including N,Z check and C,V preservation on S=1)
    b.emit_arm(mul(AL, false, false, 4, 0, 1, 2)); // mul r4, r2, r1
    b.emit_arm(mul(AL, true, false, 5, 3, 1, 2)); // mla r5, r2, r1, r3
    b.emit_arm(mul_long(AL, false, false, true, 7, 6, 1, 2)); // umulls r6, r7, r2, r1
    b.emit_arm(mrs_cpsr(AL, 8));
    b.emit_arm(mul_long(AL, false, true, true, 7, 6, 1, 2)); // umlals r6, r7, r2, r1
    b.emit_arm(mrs_cpsr(AL, 9));
    b.emit_arm(mul_long(AL, true, false, true, 11, 10, 1, 2)); // smulls r10, r11, r2, r1
    b.emit_arm(mul_long(AL, true, true, true, 11, 10, 1, 2)); // smlals r10, r11, r2, r1
    b.emit_arm(mul(AL, false, true, 12, 0, 1, 2)); // muls r12, r2, r1
    b.emit_arm(bx(AL, 14));

    let e_q_sat = b.addr();
    // First check sticky Q preservation across a non-saturating QADD, then clear via MSR and test each op:
    b.emit_arm(q_arith(AL, 0, 2, 8, 2)); // qadd r8, r2, r2 (preserves initial sticky Q!)
    b.emit_arm(mrs_cpsr(AL, 9)); // bit 27 = sticky Q check
    b.emit_arm(msr_cpsr_f_imm(AL, 0, 0)); // clear NZCVQ
    b.emit_arm(q_arith(AL, 0, 1, 4, 0)); // qadd r4, r0, r1
    b.emit_arm(mrs_cpsr(AL, 10)); // bit 27 = Q1
    b.emit_arm(msr_cpsr_f_imm(AL, 0, 0));
    b.emit_arm(q_arith(AL, 1, 1, 5, 0)); // qsub r5, r0, r1
    b.emit_arm(mrs_cpsr(AL, 11)); // bit 27 = Q2
    b.emit_arm(msr_cpsr_f_imm(AL, 0, 0));
    b.emit_arm(q_arith(AL, 2, 1, 6, 0)); // qdadd r6, r0, r1 (doubles r1 first!)
    b.emit_arm(mrs_cpsr(AL, 12)); // bit 27 = Q3
    b.emit_arm(msr_cpsr_f_imm(AL, 0, 0));
    b.emit_arm(q_arith(AL, 3, 1, 7, 0)); // qdsub r7, r0, r1
    b.emit_arm(smlaxy(AL, 8, 3, 1, 0, false, false)); // smlabb r8, r0, r1, r3
    b.emit_arm(smulwy(AL, 9, 1, 0, false)); // smulwb r9, r0, r1
    b.emit_arm(smlawy(AL, 9, 3, 1, 0, false)); // smlawb r9, r0, r1, r3
    b.emit_arm(ssat(AL, 15, 2, 0, true, 4)); // ssat r2, #16, r0, asr #4
    b.emit_arm(usat(AL, 12, 3, 0, false, 2)); // usat r3, #12, r0, lsl #2
    b.emit_arm(bx(AL, 14));

    let e_media = b.addr();
    // PKHBT, PKHTB, REV, REV16, REVSH, CLZ, SADD16 (sets GE), QSUB8 & UHADD8 (preserve GE!), SEL, USAD8, USADA8
    b.emit_arm(pkhbt(AL, 0, 4, 1, 8)); // pkhbt r4, r0, r1, lsl #8
    b.emit_arm(pkhtb(AL, 0, 5, 1, 0)); // pkhtb r5, r0, r1, asr #32 (shift=0)
    b.emit_arm(rev(AL, 6, 0));
    b.emit_arm(rev16(AL, 7, 0));
    b.emit_arm(revsh(AL, 8, 1));
    b.emit_arm(clz(AL, 9, 2));
    b.emit_arm(media_parallel(AL, 0b00001, 0b000, 0, 10, 1)); // sadd16 r10, r0, r1 (sets GE!)
    b.emit_arm(media_parallel(AL, 0b00010, 0b111, 0, 11, 1)); // qsub8 r11, r0, r1 (must preserve GE!)
    b.emit_arm(media_parallel(AL, 0b00111, 0b100, 0, 12, 1)); // uhadd8 r12, r0, r1 (must preserve GE!)
    b.emit_arm(sel(AL, 0, 2, 1)); // sel r2, r0, r1 (reads GE!)
    b.emit_arm(usad8(AL, 3, 1, 0)); // usad8 r3, r0, r1
    b.emit_arm(usada8(AL, 0, 3, 1, 0)); // usada8 r0, r0, r1, r3
    b.emit_arm(bx(AL, 14));

    let words = b.finish();
    let seeds = [(e_mul_sat, Source::Entry), (e_q_sat, Source::Entry), (e_media, Source::Entry)];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    for &(a, b_val, acc, c, v) in &[
        (0u32, 0u32, 0u32, true, false),
        (0x1_0000, 0x1_0000, 0, false, true), // lo32 == 0, hi32 != 0 -> Z MUST be 0!
        (0x8000_0000, 0xFFFF_FFFF, 0x1234_5678, true, true),
        (0x7FFF_FFFF, 0x7FFF_FFFF, 0x8000_0000, false, false),
    ] {
        verify_compiled(&format!("mul_{a:08X}_{b_val:08X}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_mul_sat;
            cpu.regs[1] = a;
            cpu.regs[2] = b_val;
            cpu.regs[3] = acc;
            cpu.cpsr.c = c;
            cpu.cpsr.v = v;
        });
    }

    for &(r0, r1, r3, init_q) in &[
        (0x7FFF_FFFFu32, 1u32, 0x7FFF_0000u32, false), // QADD/QDADD saturate
        (0x8000_0001, 0x4000_0000, 0, false),          // QDADD: 2*r1 saturates even though sum = 0!
        (0x8000_0000, 0xBFFF_FFFF, 0x8000_0000, false), // QDSUB intermediate saturation
        (10, 20, 5, false),                             // non-saturating
        (10, 20, 5, true),                              // sticky Q preserved
        (0xFFFF_8000, 0x8000_0001, 0x7FFF_FF00, false), // SMLABB/SMLAWB/SSAT/USAT negative
    ] {
        verify_compiled(&format!("qsat_{r0:08X}_{r1:08X}_q{}", init_q as u8), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_q_sat;
            cpu.regs[0] = r0;
            cpu.regs[1] = r1;
            cpu.regs[2] = 3;
            cpu.regs[3] = r3;
            cpu.cpsr.q = init_q;
        });
    }

    for &(r0, r1, r2) in &[
        (0x8001_7FFF, 0x8001_0001, 0u32),       // both lanes negative/non-neg -> GE = 0011
        (0x7FFF_8001, 0x0001_8001, 1u32),       // GE = 1100
        (0x8000_8000, 0xFFFF_FFFF, 0x0001_0000), // GE = 0000, REVSH sign-ext neg
        (0x1234_5678, 0x89AB_007F, 0x8000_0000), // GE = 1111, REVSH pos
    ] {
        verify_compiled(&format!("media_{r0:08X}_{r1:08X}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_media;
            cpu.regs[0] = r0;
            cpu.regs[1] = r1;
            cpu.regs[2] = r2;
            cpu.cpsr.ge = 0b0101;
        });
    }
}

#[test]
fn torture_vfp_nan_vcmp_fmstat_vcvt() {
    use asm::*;

    let mut b = CodeBuilder::new();

    // Entry 1: VADD/VSUB/VMUL/VMLA/VNMLA/VDIV/VSQRT/VABS/VNEG for f32 (s0..s11) & f64 (d8..d15)
    let e_arith = b.addr();
    // f32 ops (s0=a, s1=b, s2=acc) -> results in s4..s12
    b.emit_arm(vfp_dp(AL, false, 0b0011, false, 4, 0, 1)); // vadd.f32 s4, s0, s1
    b.emit_arm(vfp_dp(AL, false, 0b0011, true, 5, 0, 1)); // vsub.f32 s5, s0, s1
    b.emit_arm(vfp_dp(AL, false, 0b0010, false, 6, 0, 1)); // vmul.f32 s6, s0, s1
    b.emit_arm(vfp_unary(AL, false, 0b0000, false, 7, 2)); // vmov.f32 s7, s2
    b.emit_arm(vfp_dp(AL, false, 0b0000, false, 7, 0, 1)); // vmla.f32 s7, s0, s1
    b.emit_arm(vfp_unary(AL, false, 0b0000, false, 8, 2)); // vmov.f32 s8, s2
    b.emit_arm(vfp_dp(AL, false, 0b0001, true, 8, 0, 1)); // vnmla.f32 s8, s0, s1
    b.emit_arm(vfp_dp(AL, false, 0b1000, false, 9, 0, 1)); // vdiv.f32 s9, s0, s1
    b.emit_arm(vfp_unary(AL, false, 0b0001, true, 10, 0)); // vsqrt.f32 s10, s0
    b.emit_arm(vfp_unary(AL, false, 0b0000, true, 11, 0)); // vabs.f32 s11, s0
    b.emit_arm(vfp_unary(AL, false, 0b0001, false, 12, 0)); // vneg.f32 s12, s0
    // f64 ops (d8=a, d9=b, d10=acc) -> results in d11..d15 (covering d15 == {s30, s31}!)
    b.emit_arm(vfp_dp(AL, true, 0b0011, false, 11, 8, 9)); // vadd.f64 d11, d8, d9
    b.emit_arm(vfp_dp(AL, true, 0b0011, true, 12, 8, 9)); // vsub.f64 d12, d8, d9
    b.emit_arm(vfp_dp(AL, true, 0b0010, false, 13, 8, 9)); // vmul.f64 d13, d8, d9
    b.emit_arm(vfp_unary(AL, true, 0b0000, false, 14, 10)); // vmov.f64 d14, d10
    b.emit_arm(vfp_dp(AL, true, 0b0000, false, 14, 8, 9)); // vmla.f64 d14, d8, d9
    b.emit_arm(vfp_unary(AL, true, 0b0000, false, 15, 10)); // vmov.f64 d15, d10
    b.emit_arm(vfp_dp(AL, true, 0b0001, true, 15, 8, 9)); // vnmla.f64 d15, d8, d9
    b.emit_arm(bx(AL, 14));

    let e_arith_d_ext = b.addr();
    // d8=a, d9=b -> d10=vdiv, d11=vsqrt, d12=vabs, d13=vneg, d14=vabs(d15), d15=vneg(d15)
    b.emit_arm(vfp_dp(AL, true, 0b1000, false, 10, 8, 9)); // vdiv.f64 d10, d8, d9
    b.emit_arm(vfp_unary(AL, true, 0b0001, true, 11, 8)); // vsqrt.f64 d11, d8
    b.emit_arm(vfp_unary(AL, true, 0b0000, true, 12, 8)); // vabs.f64 d12, d8
    b.emit_arm(vfp_unary(AL, true, 0b0001, false, 13, 8)); // vneg.f64 d13, d8
    b.emit_arm(vfp_unary(AL, true, 0b0000, true, 14, 15)); // vabs.f64 d14, d15
    b.emit_arm(vfp_unary(AL, true, 0b0001, false, 15, 15)); // vneg.f64 d15, d15
    b.emit_arm(bx(AL, 14));

    // Entry 2..5: VCMP/VCMPE (f32 & f64, reg & #0) + FMSTAT + all 14 conditional branches!
    let emit_vcmp_suite = |b: &mut CodeBuilder, double: bool, e_bit: bool, vs_zero: bool| -> u32 {
        let entry = b.addr();
        b.emit_arm(vcmp(AL, double, e_bit, if double { 8 } else { 0 }, if vs_zero { None } else { Some(if double { 9 } else { 1 }) }));
        b.emit_arm(fmstat(AL));
        b.emit_arm(mov_imm(AL, false, 4, 0, 0));
        for cond in 0..14u32 {
            let (imm8, rot4) = if cond < 8 { (1 << cond, 0) } else { (1 << (cond - 8), 12) };
            let j_taken = b.emit_arm(0);
            let j_skip = b.emit_arm(0);
            let taken_a = b.emit_arm(orr_imm(AL, false, 4, 4, imm8, rot4));
            let done_a = b.addr();
            b.patch_arm(j_taken, b_cond(cond, j_taken, taken_a));
            b.patch_arm(j_skip, b_cond(AL, j_skip, done_a));
        }
        b.emit_arm(bx(AL, 14));
        entry
    };
    let e_vcmp_f32 = emit_vcmp_suite(&mut b, false, false, false);
    let e_vcmpe_f32_zero = emit_vcmp_suite(&mut b, false, true, true);
    let e_vcmpe_f64 = emit_vcmp_suite(&mut b, true, true, false);
    let e_vcmp_f64_zero = emit_vcmp_suite(&mut b, true, false, true);

    // Entry 6: VCVT f32<->f64 and f32/f64<->i32/u32 (round_zero=true, zero fallbacks on all arches!)
    // Input: s0 (f32), d2 (f64), s1 (i32/u32 bits).
    // Note: fold integer outputs into r11 (`add r11, r4, r11, ror #7`) so 0x7FFF_FFFF/0xFFFF_FFFF
    // are verified bit-exact even though compare_runs treats NaN-like bit patterns specially!
    let fold_s_into_r11 = |b: &mut CodeBuilder, s: u32| {
        b.emit_arm(vmov_s_rt(AL, true, s, 4));
        b.emit_arm(dp_reg_imm(AL, OP_ADD, false, 4, 11, 11, ROR, 7));
    };
    let e_vcvt = b.addr();
    b.emit_arm(mov_imm(AL, false, 11, 0x55, 0));
    b.emit_arm(vcvt_f64_f32(AL, 8, 0)); // d8 = f64(s0)
    b.emit_arm(vcvt_f32_f64(AL, 10, 2)); // s10 = f32(d2)
    b.emit_arm(vcvt_to_int(AL, false, true, true, 12, 0)); // vcvt.s32.f32 s12, s0
    fold_s_into_r11(&mut b, 12);
    b.emit_arm(vcvt_to_int(AL, false, false, true, 13, 0)); // vcvt.u32.f32 s13, s0
    fold_s_into_r11(&mut b, 13);
    b.emit_arm(vcvt_to_int(AL, true, true, true, 14, 2)); // vcvt.s32.f64 s14, d2
    fold_s_into_r11(&mut b, 14);
    b.emit_arm(vcvt_to_int(AL, true, false, true, 15, 2)); // vcvt.u32.f64 s15, d2
    fold_s_into_r11(&mut b, 15);
    // Keep int->float under FPSCR[23:22]==0 only or test separately below:
    b.emit_arm(bx(AL, 14));

    let e_vcvt_int_to_fp = b.addr();
    b.emit_arm(vcvt_from_int(AL, false, true, 4, 1)); // vcvt.f32.s32 s4, s1
    b.emit_arm(vcvt_from_int(AL, false, false, 5, 1)); // vcvt.f32.u32 s5, s1
    b.emit_arm(vcvt_from_int(AL, true, true, 8, 1)); // vcvt.f64.s32 d8, s1
    b.emit_arm(vcvt_from_int(AL, true, false, 9, 1)); // vcvt.f64.u32 d9, s1
    b.emit_arm(bx(AL, 14));

    // Entry 8: VCVTR (round_zero=false) across FPSCR[23:22] rounding modes, using VMSR FPSCR, r0!
    let e_vcvtr = b.addr();
    b.emit_arm(mov_imm(AL, false, 11, 0x55, 0));
    b.emit_arm(vmsr_fpscr(AL, 0));
    b.emit_arm(vmrs_fpscr(AL, 1));
    b.emit_arm(vcvt_to_int(AL, false, true, false, 12, 0)); // vcvtr.s32.f32 s12, s0
    fold_s_into_r11(&mut b, 12);
    b.emit_arm(vcvt_to_int(AL, false, false, false, 13, 0)); // vcvtr.u32.f32 s13, s0
    fold_s_into_r11(&mut b, 13);
    b.emit_arm(vcvt_to_int(AL, true, true, false, 14, 2)); // vcvtr.s32.f64 s14, d2
    fold_s_into_r11(&mut b, 14);
    b.emit_arm(vcvt_to_int(AL, true, false, false, 15, 2)); // vcvtr.u32.f64 s15, d2
    fold_s_into_r11(&mut b, 15);
    b.emit_arm(bx(AL, 14));

    let words = b.finish();
    let seeds = [
        (e_arith, Source::Entry),
        (e_arith_d_ext, Source::Entry),
        (e_vcmp_f32, Source::Entry),
        (e_vcmpe_f32_zero, Source::Entry),
        (e_vcmpe_f64, Source::Entry),
        (e_vcmp_f64_zero, Source::Entry),
        (e_vcvt, Source::Entry),
        (e_vcvt_int_to_fp, Source::Entry),
        (e_vcvtr, Source::Entry),
    ];
    let (text_bytes, aot) = compile_snippet(&words, &seeds);

    // 1. VADD/VSUB/VMUL/VMLA/VNMLA/VDIV/VSQRT/VABS/VNEG (f32 & f64)
    let f32_cases: &[(f32, f32, f32, u32)] = &[
        (-3.5, 2.25, 10.0, 0),
        (-0.0, 0.0, -0.0, 0),
        (16.0, -0.0, 1.5, 0),
        (-1.0, 0.0, 1.0, 0), // sqrt(-1.0)=NaN, -1/0=-Inf
        (0.0, 0.0, 1.0, 0),  // 0/0=NaN
        (f32::INFINITY, f32::NEG_INFINITY, 2.0, 0),
        (f32::NAN, 1.5, -2.0, 0),
        (1.0 + 2.0f32.powi(-23), 1.0 + 2.0f32.powi(-23), -1.0, 0), // VMLA/VNMLA chained rounding
        (1.0e-20, 1.0e-20, 0.0, 1 << 24),                           // FZ=1 flushes subnormal output to +0.0
        (-1.0e-20, 1.0e-20, 0.0, 1 << 24),                          // FZ=1 flushes subnormal output to -0.0
    ];
    for (idx, &(a, b_val, acc, fpscr)) in f32_cases.iter().enumerate() {
        verify_compiled(&format!("vfp_arith_{idx}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_arith;
            cpu.vfp.fpscr = fpscr;
            cpu.vfp.regs[0] = a.to_bits();
            cpu.vfp.regs[1] = b_val.to_bits();
            cpu.vfp.regs[2] = acc.to_bits();
            cpu.vfp.set_f64(8, a as f64);
            cpu.vfp.set_f64(9, b_val as f64);
            cpu.vfp.set_f64(10, acc as f64);
        });
        verify_compiled(&format!("vfp_arith_d_{idx}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_arith_d_ext;
            cpu.vfp.fpscr = fpscr;
            cpu.vfp.set_f64(8, a as f64);
            cpu.vfp.set_f64(9, b_val as f64);
            // Put two singles in d15 under FZ=1 too (verifies raw bit VABS/VNEG!)
            cpu.vfp.regs[30] = 2.5f32.to_bits();
            cpu.vfp.regs[31] = (-0.0f32).to_bits();
        });
    }

    // 2. VCMP/VCMPE + FMSTAT + conditional branches across +0.0, -0.0, NaN, +Inf, -Inf
    let cmp_vals = [0.0f64, -0.0, 1.0, -2.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    for (i, &x) in cmp_vals.iter().enumerate() {
        for (j, &y) in cmp_vals.iter().enumerate() {
            let x32 = x as f32;
            let y32 = y as f32;
            verify_compiled(&format!("vcmp_f32_{i}_{j}"), &text_bytes, &aot, true, |cpu, _| {
                cpu.regs[15] = e_vcmp_f32;
                cpu.vfp.regs[0] = x32.to_bits();
                cpu.vfp.regs[1] = y32.to_bits();
            });
            verify_compiled(&format!("vcmpe_f64_{i}_{j}"), &text_bytes, &aot, true, |cpu, _| {
                cpu.regs[15] = e_vcmpe_f64;
                cpu.vfp.set_f64(8, x);
                cpu.vfp.set_f64(9, y);
            });
        }
        verify_compiled(&format!("vcmpe_f32_zero_{i}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_vcmpe_f32_zero;
            cpu.vfp.regs[0] = (x as f32).to_bits();
        });
        verify_compiled(&format!("vcmp_f64_zero_{i}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_vcmp_f64_zero;
            cpu.vfp.set_f64(8, x);
        });
    }
    // Also test SNaN explicitly on f32 and f64
    verify_compiled("vcmp_snan_f32", &text_bytes, &aot, true, |cpu, _| {
        cpu.regs[15] = e_vcmp_f32;
        cpu.vfp.regs[0] = 0x7F80_0001; // f32 SNaN
        cpu.vfp.regs[1] = 1.0f32.to_bits();
    });
    verify_compiled("vcmp_snan_f64", &text_bytes, &aot, true, |cpu, _| {
        cpu.regs[15] = e_vcmpe_f64;
        cpu.vfp.regs[16] = 0x0000_0001;
        cpu.vfp.regs[17] = 0x7FF0_0000; // f64 SNaN
        cpu.vfp.set_f64(9, 1.0);
    });

    // 3. VCVT f32<->f64 and f32/f64<->i32/u32
    let cvt_doubles = [
        0.0f64,
        -0.0,
        1.75,
        -2.75,
        -0.9,
        2147483648.0, // 2^31 (u32 >= 2^31!)
        3000000000.75,
        4294967294.75,
        4294967295.0,
        1e20,
        -1e20,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0e-40, // underflows f32
    ];
    for &fpscr in &[0u32, 0x0040_0000, 0x0080_0000] {
        for &d in &cvt_doubles {
            let s = d as f32;
            verify_compiled(&format!("vcvt_fpscr_{fpscr:08X}"), &text_bytes, &aot, true, |cpu, _| {
                cpu.regs[15] = e_vcvt;
                cpu.vfp.fpscr = fpscr;
                cpu.vfp.regs[0] = s.to_bits();
                cpu.vfp.set_f64(2, d);
            });
        }
    }
    for &int_bits in &[0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFF] {
        verify_compiled(&format!("vcvt_int_to_fp_{int_bits:08X}"), &text_bytes, &aot, true, |cpu, _| {
            cpu.regs[15] = e_vcvt_int_to_fp;
            cpu.vfp.regs[1] = int_bits;
        });
    }

    // 4. VCVTR across all 4 FPSCR[23:22] rounding modes (00=RN, 01=+Inf, 10=-Inf, 11=RZ)
    // Native on x86_64 via vfp_to_int_helper (0 fallbacks); AArch64 interprets VCVTR via fallback.
    let vcvtr_vals = [
        0.5f64,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        1.25,
        -1.25,
        1.75,
        -1.75,
        2147483648.5,
        3000000000.25,
        3000000000.75,
        4294967294.75,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let require_zero_vcvtr = cfg!(target_arch = "x86_64");
    for rmode in 0..4u32 {
        let fpscr = rmode << 22;
        for &d in &vcvtr_vals {
            verify_compiled(
                &format!("vcvtr_rm{rmode}_{:016X}", d.to_bits()),
                &text_bytes,
                &aot,
                require_zero_vcvtr,
                |cpu, _| {
                    cpu.regs[15] = e_vcvtr;
                    cpu.regs[0] = fpscr;
                    cpu.vfp.regs[0] = (d as f32).to_bits();
                    cpu.vfp.set_f64(2, d);
                },
            );
        }
    }
}

#[test]
fn compiles_and_verifies_local_3ds_rom() {
    let Some(rom_path) = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../roms"))
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("cci") || e.eq_ignore_ascii_case("3ds")))
    else {
        eprintln!("No test ROM in roms/, skipping real-ROM differential test");
        return;
    };

    let title = zakuro_fs::Title::load(&rom_path).expect("load test ROM");
    let code = title.code().expect("decompress code.bin");

    let t0 = Instant::now();
    let image = compile_title(
        title.program_id(),
        &title.exheader,
        &code,
        &[],
        &[],
        CompileOptions::default(),
        |_, _| {},
    );
    let compile_elapsed = t0.elapsed();

    let code_mib = image.code_bytes().len() as f64 / (1024.0 * 1024.0);
    println!(
        "ROM AOT compiled in {:.2?}: {:.2} MiB machine code ({} functions, {} entry points, {} pieces)",
        compile_elapsed,
        code_mib,
        image.origins.functions.len(),
        image.entries.len(),
        image.origins.pieces.len(),
    );
    // On AArch64 Horizon, code must fit comfortably in a single 128 MiB CodeMemory;
    // x86_64 instructions are variable-length CISC with memory operands so budget < 256 MiB.
    #[cfg(target_arch = "aarch64")]
    assert!(code_mib < 96.0, "expected compact AOT code < 96 MiB, got {code_mib:.2} MiB");
    #[cfg(target_arch = "x86_64")]
    assert!(code_mib < 256.0, "expected x86_64 AOT code < 256 MiB, got {code_mib:.2} MiB");

    // Differential-verify 2,500 real functions against pure `zakuro-cpu` instruction-for-instruction.
    let parsed = zakuro_aot::discover::Image::from_exheader(&title.exheader, &code);
    let text_seg = parsed.text.clone();
    let ro_seg = parsed.rodata.clone();
    let data_seg = parsed.data.clone();
    let text_hash = parsed.text.hash();
    let prog = parsed.into_program(&[], &[]);
    let pure_image = compile_program(
        title.program_id(),
        text_hash,
        &prog,
        &[],
        CompileOptions::default(),
        |_, _| {},
    );
    let aot = pure_image.into_loaded_exec().expect("exec ROM AOT");

    // Build pristine test memory matching `3dsrecomp/src/verify.rs`:
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next_u32 = || -> u32 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 16) as u32
    };
    let mut heap = vec![0u8; HEAP_SIZE as usize];
    for word in heap.chunks_mut(4) {
        let v = if next_u32() % 4 == 0 {
            (HEAP + next_u32() % (HEAP_SIZE - 0x1000)) & !3
        } else {
            next_u32()
        };
        word.copy_from_slice(&v.to_le_bytes());
    }
    let mut data_bytes = data_seg.bytes.clone();
    data_bytes.resize(data_bytes.len() + title.exheader.bss_size as usize, 0);
    let pristine = TestMemory::new(vec![
        Region { base: text_seg.base, bytes: text_seg.bytes, writable: false, direct_page: true },
        Region { base: ro_seg.base, bytes: ro_seg.bytes, writable: false, direct_page: true },
        Region { base: data_seg.base, bytes: data_bytes, writable: true, direct_page: true },
        Region { base: HEAP, bytes: heap, writable: true, direct_page: true },
        Region { base: STACK_TOP - STACK_SIZE, bytes: vec![0; STACK_SIZE as usize], writable: true, direct_page: true },
        Region { base: TLS, bytes: vec![0; PAGE_SIZE], writable: true, direct_page: true },
    ]);

    let func_starts: Vec<u32> = pure_image_starts(&aot, &code, &title.exheader);
    let step = (func_starts.len() / 2500).max(1);
    let mut a = Run { memory: pristine.duplicate(), cpu: Cpu::new(), pending: None, interpreted: 0, fallbacks: 0 };
    let mut b = Run { memory: pristine.duplicate(), cpu: Cpu::new(), pending: None, interpreted: 0, fallbacks: 0 };
    let mut tested = 0usize;
    let mut returned = 0usize;
    let mut mismatches = 0usize;
    let mut total_fallbacks = 0u64;

    for &fn_addr in func_starts.iter().step_by(step) {
        a.memory.copy_from(&pristine);
        b.memory.copy_from(&pristine);
        let mut r = 0x5DEE_CE66_D000_0000_u64 ^ (fn_addr as u64);
        let mut rnd = || -> u32 {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            (r >> 16) as u32
        };
        let mut cpu = Cpu::new();
        for reg in 0..13 {
            cpu.regs[reg] = match rnd() % 4 {
                0 | 1 => (HEAP + rnd() % (HEAP_SIZE - 0x1000)) & !3,
                2 => rnd() % 256,
                _ => rnd(),
            };
        }
        cpu.regs[13] = STACK_TOP - 0x1000;
        cpu.regs[14] = RETURN;
        cpu.regs[15] = fn_addr;
        let fl = rnd();
        cpu.cpsr.n = (fl & 1) != 0;
        cpu.cpsr.z = (fl & 2) != 0;
        cpu.cpsr.c = (fl & 4) != 0;
        cpu.cpsr.v = (fl & 8) != 0;
        for s in 0..32 {
            cpu.vfp.regs[s] = ((rnd() % 20_000) as f32 / 100.0 - 100.0).to_bits();
        }
        cpu.vfp.fpscr = if rnd() % 2 == 0 { 0x0300_0000 } else { 0 };
        cpu.cp15.thread_id_ro = TLS;
        a.cpu = cpu.clone();
        b.cpu = cpu.clone();
        b.interpreted = 0;
        b.fallbacks = 0;

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
            budget: 0,
            exit: 0,
            svc: 0,
            depth: 0,
            exclusive_address: 0,
            tls: 0,
            read_pages: b.memory.read_pages.as_ptr(),
            write_pages: b.memory.write_pages.as_ptr(),
            vfp: b.cpu.vfp.regs.as_mut_ptr(),
            fpscr: &mut b.cpu.vfp.fpscr,
            host: &HOST,
            user: &mut b as *mut Run as *mut c_void,
        };
        store_ctx(&cpu, &mut ctx);

        let exp = run_interpreted(&mut a, 5_000);
        let slice = if tested % 2 == 0 { 5_000 } else { 1 + (rnd() % 32) as i32 };
        let got = unsafe { run_aot(&aot, &mut b, &mut ctx, slice, 5_000) };
        tested += 1;
        total_fallbacks += b.fallbacks;
        if exp == Stop::Limit || got == Stop::Limit {
            continue;
        }
        let mut diffs = compare_runs(&a, &b, &ctx);
        if exp != got {
            diffs.insert(0, format!("stop: interp={exp:?} aot={got:?}"));
        }
        if !diffs.is_empty() {
            mismatches += 1;
            if mismatches <= 10 {
                println!("MISMATCH at 0x{fn_addr:08X}: {}", diffs.join(", "));
            }
        } else if exp == Stop::Returned {
            returned += 1;
        }
    }

    println!(
        "ROM differential verification: tested={tested}, returned={returned}, fallbacks={total_fallbacks}, mismatches={mismatches}"
    );
    assert_eq!(mismatches, 0, "{mismatches} ROM functions mismatched zakuro-cpu!");
}

fn pure_image_starts(aot: &LoadedAot, code: &[u8], exh: &zakuro_fs::ExHeader) -> Vec<u32> {
    let text_end = exh.text.address + exh.text.size;
    let _ = code;
    let mut out = Vec::new();
    let mut addr = exh.text.address;
    while addr < text_end {
        if aot.has_code(addr) {
            out.push(addr);
        }
        addr += 4;
    }
    out
}
