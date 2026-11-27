//! Live single-block "shadow CPU" differential verification for AOT execution.
//!
//! When `ZAKURO_VERIFY_LOCKSTEP` is set to a nonzero stride `N`:
//! - `N = 1`: verifies every AOT basic block executed in `Library::run`.
//! - `N > 1`: verifies the first AOT basic block on 1 of every `N` `Library::run` invocations.
//!
//! Each verified block is first stepped on a cloned shadow `Cpu` backed by [`ShadowBus`]
//! (which intercepts all RAM and host-mailbox reads/writes without mutating real [`Memory`]),
//! then executed as a single block in AOT by probing `budget = 1..=max_budget` until AOT
//! commits one block (`spent > 0`), and finally compared state-for-state.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

use recomp_abi::{Context, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND};
use zakuro_aot::LoadedAot;
use zakuro_common::memory_map::PAGE_MASK;
use zakuro_cpu::{Bus, Cpu, Exit};

use crate::memory::Memory;
use crate::recompiled::{load, Machine, Stop};

static STRIDE: OnceLock<u64> = OnceLock::new();
static VERIFIED_BLOCKS: AtomicU64 = AtomicU64::new(0);
static VERIFIED_INSNS: AtomicU64 = AtomicU64::new(0);
static INVOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Returns the lockstep verification stride parsed once from `ZAKURO_VERIFY_LOCKSTEP`.
///
/// Unset, invalid, or `"0"` returns `0` (disabled). `"1"` verifies every AOT basic block;
/// `N > 1` verifies the first AOT basic block on 1 of every `N` `Library::run` invocations.
pub fn stride() -> u64 {
    *STRIDE.get_or_init(|| {
        std::env::var("ZAKURO_VERIFY_LOCKSTEP")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}

/// Returns whether lockstep verification is enabled (`stride() != 0`).
pub fn is_enabled() -> bool {
    stride() != 0
}

/// Returns `(verified_blocks, verified_guest_insns)` accumulated so far.
pub fn stats() -> (u64, u64) {
    (
        VERIFIED_BLOCKS.load(Ordering::Relaxed),
        VERIFIED_INSNS.load(Ordering::Relaxed),
    )
}

/// Advances the `Library::run` invocation counter and returns `true` for 1 in every `stride` calls.
pub(crate) fn take_invocation(stride: u64) -> bool {
    if stride == 0 {
        return false;
    }
    if stride == 1 {
        return true;
    }
    INVOCATIONS.fetch_add(1, Ordering::Relaxed).is_multiple_of(stride)
}

/// Outcome of verifying a single AOT basic block in [`verify_block`].
pub(crate) struct BlockOutcome {
    #[allow(dead_code)]
    pub ran: u64,
    pub stop: Option<Stop>,
    pub switch_to_fast: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShadowLoc {
    Ram(usize),
    MailboxByte(u8, u8),
}

/// Persistent shadow memory overlay state across lazy stepping calls within one block.
#[derive(Default)]
struct ShadowState {
    ram_overlay: HashMap<usize, u8>,
    initial_ram: HashMap<usize, u8>,
    mailbox_overlay: HashMap<u8, u32>,
    initial_mailbox: HashMap<u8, u32>,
    write_events: Vec<(usize, ShadowLoc, u8)>,
    current_step: usize,
    unmapped_accesses: u64,
}

/// Non-mutating [`Bus`] wrapper over `&mut Memory` for shadow CPU stepping.
struct ShadowBus<'a> {
    memory: &'a mut Memory,
    state: &'a mut ShadowState,
}

impl<'a> ShadowBus<'a> {
    fn new(memory: &'a mut Memory, state: &'a mut ShadowState) -> Self {
        Self { memory, state }
    }

    fn set_step(&mut self, step: usize) {
        self.state.current_step = step;
    }

    fn read_ram_byte(&mut self, addr: u32) -> Option<u8> {
        let ptr = self.memory.shadow_read_ptr(addr)?;
        let key = ptr as usize;
        if let Some(&val) = self.state.ram_overlay.get(&key) {
            return Some(val);
        }
        // SAFETY: `shadow_read_ptr` returns a valid byte pointer inside live guest RAM.
        Some(unsafe { *ptr })
    }

    fn write_ram_byte(&mut self, addr: u32, value: u8) {
        if let Some(ptr) = self.memory.shadow_write_ptr(addr) {
            let key = ptr as usize;
            // SAFETY: `shadow_write_ptr` returns a valid byte pointer inside live guest RAM.
            self.state.initial_ram.entry(key).or_insert_with(|| unsafe { *ptr });
            self.state.ram_overlay.insert(key, value);
            self.state
                .write_events
                .push((self.state.current_step, ShadowLoc::Ram(key), value));
        } else {
            self.state.unmapped_accesses += 1;
        }
    }
}

impl Bus for ShadowBus<'_> {
    fn read8(&mut self, addr: u32) -> u8 {
        match self.read_ram_byte(addr) {
            Some(byte) => byte,
            None => {
                self.state.unmapped_accesses += 1;
                0
            }
        }
    }

    fn read16(&mut self, addr: u32) -> u16 {
        u16::from_le_bytes([self.read8(addr), self.read8(addr.wrapping_add(1))])
    }

    fn read32(&mut self, addr: u32) -> u32 {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= zakuro_common::memory_map::PAGE_SIZE as usize - 4 {
            if self.memory.shadow_read_ptr(addr).is_some() {
                let b = std::array::from_fn(|i| self.read8(addr.wrapping_add(i as u32)));
                return u32::from_le_bytes(b);
            }
            if let Some(idx) = Memory::shadow_mailbox_word(addr) {
                let idx_u8 = idx as u8;
                return self
                    .state
                    .mailbox_overlay
                    .get(&idx_u8)
                    .copied()
                    .unwrap_or(self.memory.host_mailbox[idx]);
            }
            self.state.unmapped_accesses += 1;
            0
        } else {
            let b = std::array::from_fn(|i| self.read8(addr.wrapping_add(i as u32)));
            u32::from_le_bytes(b)
        }
    }

    fn write8(&mut self, addr: u32, value: u8) {
        self.write_ram_byte(addr, value);
    }

    fn write16(&mut self, addr: u32, value: u16) {
        for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
            self.write_ram_byte(addr.wrapping_add(i as u32), byte);
        }
    }

    fn write32(&mut self, addr: u32, value: u32) {
        let offset = (addr & PAGE_MASK) as usize;
        if offset <= zakuro_common::memory_map::PAGE_SIZE as usize - 4 {
            if self.memory.shadow_write_ptr(addr).is_some() {
                for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                    self.write_ram_byte(addr.wrapping_add(i as u32), byte);
                }
            } else if let Some(idx) = Memory::shadow_mailbox_word(addr) {
                let idx_u8 = idx as u8;
                self.state
                    .initial_mailbox
                    .entry(idx_u8)
                    .or_insert_with(|| self.memory.host_mailbox[idx]);
                self.state.mailbox_overlay.insert(idx_u8, value);
                for (lane, byte) in value.to_le_bytes().into_iter().enumerate() {
                    self.state.write_events.push((
                        self.state.current_step,
                        ShadowLoc::MailboxByte(idx_u8, lane as u8),
                        byte,
                    ));
                }
            } else {
                self.state.unmapped_accesses += 1;
            }
        } else {
            for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                self.write_ram_byte(addr.wrapping_add(i as u32), byte);
            }
        }
    }
}

struct ShadowStep {
    cpu: Cpu,
    exit: Option<Exit>,
    took_branch: bool,
    write_end: usize,
}

/// Executes one AOT-equivalent instruction on `shadow_cpu`, fusing Thumb-1 32-bit `BL`/`BLX`.
fn step_shadow_aot_insn(shadow_cpu: &mut Cpu, bus: &mut ShadowBus<'_>) -> (Option<Exit>, bool) {
    let was_thumb = shadow_cpu.cpsr.thumb;
    let (hw1, hw2) = if was_thumb {
        let addr = shadow_cpu.regs[15];
        (bus.fetch16(addr), bus.fetch16(addr.wrapping_add(2)))
    } else {
        (0, 0)
    };
    let (exit1, b1) = shadow_cpu.step_with_branch(bus);
    if was_thumb
        && exit1.is_none()
        && !b1
        && (hw1 >> 11) == 0b11110
        && matches!(hw2 >> 11, 0b11111 | 0b11101)
    {
        let (exit2, b2) = shadow_cpu.step_with_branch(bus);
        (exit2, b1 || b2)
    } else {
        (exit1, b1)
    }
}

fn step_shadow_to(
    target_len: usize,
    shadow_cpu: &mut Cpu,
    memory: &mut Memory,
    state: &mut ShadowState,
    history: &mut Vec<ShadowStep>,
    stopped: &mut bool,
) {
    let mut bus = ShadowBus::new(memory, state);
    while !*stopped && history.len() < target_len {
        let step_num = history.len() + 1;
        bus.set_step(step_num);
        let (exit, took_branch) = step_shadow_aot_insn(shadow_cpu, &mut bus);
        let write_end = bus.state.write_events.len();
        if exit.is_some() || took_branch {
            *stopped = true;
        }
        history.push(ShadowStep {
            cpu: shadow_cpu.clone(),
            exit,
            took_branch,
            write_end,
        });
    }
    let _ = bus.state.unmapped_accesses;
}

/// Verifies a single AOT basic block starting at `ctx.r[15]` (`target_rx`) against `zakuro-cpu`.
pub(crate) fn verify_block(
    aot: &LoadedAot,
    ctx: &mut Context,
    machine: &mut Machine<'_>,
    target_rx: *const u8,
) -> BlockOutcome {
    verify_block_with_hook(aot, ctx, machine, target_rx, |_, _| {})
}

fn verify_block_with_hook(
    aot: &LoadedAot,
    ctx: &mut Context,
    machine: &mut Machine<'_>,
    target_rx: *const u8,
    mut after_aot: impl FnMut(&mut Context, &mut Memory),
) -> BlockOutcome {
    if aot.has_active_overrides() {
        static WARNED_OVERRIDES: Once = Once::new();
        WARNED_OVERRIDES.call_once(|| {
            log::warn!("lockstep verification bypassed: active native AOT overrides detected");
        });
        return BlockOutcome {
            ran: 0,
            stop: None,
            switch_to_fast: true,
        };
    }

    let max_budget = ctx.budget;
    if max_budget <= 0 {
        return BlockOutcome {
            ran: 0,
            stop: Some(Stop::Left),
            switch_to_fast: false,
        };
    }

    let start_pc = ctx.r[15];
    let start_thumb = ctx.thumb;

    let mut cpu_before = machine.cpu.clone();
    load(ctx, &mut cpu_before);
    cpu_before.cp15.thread_id_ro = ctx.tls;

    let mut shadow_cpu = cpu_before.clone();
    let mut shadow_state = ShadowState::default();
    let mut history: Vec<ShadowStep> = Vec::new();
    let mut shadow_stopped = false;

    let initial_cap = (max_budget as usize).min(256);
    step_shadow_to(
        initial_cap,
        &mut shadow_cpu,
        machine.memory,
        &mut shadow_state,
        &mut history,
        &mut shadow_stopped,
    );

    #[cfg(target_arch = "aarch64")]
    let mut fp_guard = crate::recompiled::FpcrGuard::new(unsafe { *ctx.fpscr });
    #[cfg(target_arch = "x86_64")]
    let mut fp_guard = crate::recompiled::MxcsrGuard::new(unsafe { *ctx.fpscr });

    let mut spent_out: Option<i32> = None;
    for probe_budget in 1..=max_budget {
        if (probe_budget as usize) > history.len() && !shadow_stopped {
            step_shadow_to(
                probe_budget as usize,
                &mut shadow_cpu,
                machine.memory,
                &mut shadow_state,
                &mut history,
                &mut shadow_stopped,
            );
        }

        ctx.budget = probe_budget;
        ctx.exit = 0;
        ctx.depth = 0;
        ctx.vfp = machine.cpu.vfp.regs.as_mut_ptr();
        ctx.fpscr = &mut machine.cpu.vfp.fpscr;
        ctx.user = machine as *mut Machine<'_> as *mut c_void;

        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        fp_guard.sync(unsafe { *ctx.fpscr });

        // SAFETY: `target_rx` came from `aot.lookup`, and `ctx` pointers were just refreshed.
        unsafe { aot.enter(ctx, target_rx) };

        let spent = probe_budget - ctx.budget;
        if spent == 0 && ctx.exit == EXIT_BUDGET && machine.pending.is_none() {
            ctx.r[15] = start_pc;
            ctx.thumb = start_thumb;
            ctx.exit = 0;
            ctx.depth = 0;
            continue;
        }
        spent_out = Some(spent);
        break;
    }

    let Some(spent) = spent_out else {
        ctx.r[15] = start_pc;
        ctx.thumb = start_thumb;
        ctx.exit = 0;
        ctx.depth = 0;
        ctx.budget = max_budget;
        return BlockOutcome {
            ran: 0,
            stop: Some(Stop::Left),
            switch_to_fast: false,
        };
    };

    after_aot(ctx, machine.memory);

    let pending = machine.pending.take();

    if spent == 0 && ctx.exit == EXIT_UNWIND && pending.is_none() {
        ctx.budget = max_budget;
        ctx.exit = 0;
        return BlockOutcome {
            ran: 0,
            stop: Some(Stop::Left),
            switch_to_fast: false,
        };
    }

    let (expected_cpu, expected_exit, write_end, step_k): (&Cpu, Option<Exit>, usize, usize) =
        if spent <= 0 {
            (&cpu_before, None, 0, 0)
        } else {
            let idx = (spent as usize).saturating_sub(1).min(history.len() - 1);
            let step = &history[idx];
            let _ = step.took_branch;
            (&step.cpu, step.exit, step.write_end, idx + 1)
        };

    let actual_exit = if let Some(exit) = pending {
        Some(exit)
    } else if ctx.exit == EXIT_SVC {
        Some(Exit::Supervisor(ctx.svc))
    } else {
        None
    };

    let diffs = compare_state(
        expected_cpu,
        expected_exit,
        ctx,
        &machine.cpu.vfp,
        actual_exit,
        machine.memory,
        &shadow_state.write_events[..write_end],
        &shadow_state.initial_ram,
        &shadow_state.initial_mailbox,
    );
    if !diffs.is_empty() {
        panic!(
            "LOCKSTEP MISMATCH at start_pc=0x{start_pc:08X} (thumb={}, block_c={spent}, step_k={step_k}): {}",
            start_thumb != 0,
            diffs.join("; ")
        );
    }

    let remaining = (max_budget - spent).max(0);
    ctx.budget = remaining;

    let verified_insns = step_k as u64;
    let blocks = VERIFIED_BLOCKS.fetch_add(1, Ordering::Relaxed) + 1;
    let insns = VERIFIED_INSNS.fetch_add(verified_insns, Ordering::Relaxed) + verified_insns;
    if blocks.is_multiple_of(250_000) {
        log::info!("lockstep: verified {blocks} blocks ({insns} guest insns), 0 mismatches");
    }

    let stop = if let Some(exit) = pending {
        Some(Stop::Exit(exit))
    } else if ctx.exit == EXIT_SVC {
        Some(Stop::Svc(ctx.svc))
    } else if remaining <= 0 {
        Some(Stop::Left)
    } else {
        None
    };

    BlockOutcome {
        ran: spent.max(0) as u64,
        stop,
        switch_to_fast: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_state(
    expected_cpu: &Cpu,
    expected_exit: Option<Exit>,
    ctx: &Context,
    actual_vfp: &zakuro_cpu::Vfp,
    actual_exit: Option<Exit>,
    memory: &Memory,
    write_events: &[(usize, ShadowLoc, u8)],
    initial_ram: &HashMap<usize, u8>,
    initial_mailbox: &HashMap<u8, u32>,
) -> Vec<String> {
    let mut diffs = Vec::new();
    let f32_nan = |bits: u32| f32::from_bits(bits).is_nan();
    let f64_nan = |r: &[u32; 32], d: usize| {
        f64::from_bits(((r[2 * d + 1] as u64) << 32) | (r[2 * d] as u64)).is_nan()
    };

    for i in 0..16 {
        let exp = expected_cpu.regs[i];
        let got = ctx.r[i];
        if exp != got && !(f32_nan(exp) && f32_nan(got)) {
            diffs.push(format!("r{i}: shadow=0x{exp:08X} aot=0x{got:08X}"));
        }
    }

    for (name, exp, got) in [
        ("n", expected_cpu.cpsr.n as u8, ctx.n),
        ("z", expected_cpu.cpsr.z as u8, ctx.z),
        ("c", expected_cpu.cpsr.c as u8, ctx.c),
        ("v", expected_cpu.cpsr.v as u8, ctx.v),
        ("q", expected_cpu.cpsr.q as u8, ctx.q),
        ("thumb", expected_cpu.cpsr.thumb as u8, ctx.thumb),
    ] {
        if exp != got {
            diffs.push(format!("{name}: shadow={exp} aot={got}"));
        }
    }

    if expected_cpu.cpsr.ge != ctx.ge {
        diffs.push(format!("ge: shadow=0x{:X} aot=0x{:X}", expected_cpu.cpsr.ge, ctx.ge));
    }

    let got_excl = (ctx.exclusive != 0).then_some(ctx.exclusive_address);
    if expected_cpu.exclusive_addr != got_excl {
        diffs.push(format!("excl: shadow={:X?} aot={got_excl:X?}", expected_cpu.exclusive_addr));
    }

    for d in 0..16 {
        if f64_nan(&expected_cpu.vfp.regs, d) && f64_nan(&actual_vfp.regs, d) {
            continue;
        }
        for s in 2 * d..2 * d + 2 {
            let exp = expected_cpu.vfp.regs[s];
            let got = actual_vfp.regs[s];
            if exp != got && !(f32_nan(exp) && f32_nan(got)) {
                diffs.push(format!("s{s}: shadow=0x{exp:08X} aot=0x{got:08X}"));
            }
        }
    }

    if expected_cpu.vfp.fpscr != actual_vfp.fpscr {
        diffs.push(format!(
            "fpscr: shadow=0x{:08X} aot=0x{:08X}",
            expected_cpu.vfp.fpscr, actual_vfp.fpscr
        ));
    }

    if expected_exit != actual_exit {
        diffs.push(format!("exit: shadow={expected_exit:?} aot={actual_exit:?}"));
    }

    let mut expected_ram: HashMap<usize, u8> = HashMap::new();
    let mut expected_mailbox: HashMap<u8, u32> = HashMap::new();
    for &(_, loc, value) in write_events {
        match loc {
            ShadowLoc::Ram(ptr) => {
                expected_ram.insert(ptr, value);
            }
            ShadowLoc::MailboxByte(word_idx, lane) => {
                let entry = expected_mailbox
                    .entry(word_idx)
                    .or_insert_with(|| initial_mailbox.get(&word_idx).copied().unwrap_or(0));
                let shift = lane * 8;
                *entry = (*entry & !(0xFFu32 << shift)) | ((value as u32) << shift);
            }
        }
    }

    let mut ram_ptrs: Vec<usize> = expected_ram.keys().copied().collect();
    ram_ptrs.sort_unstable();
    let mut idx = 0usize;
    while idx < ram_ptrs.len() {
        let ptr = ram_ptrs[idx];
        let exp = expected_ram[&ptr];
        // SAFETY: `ptr` came from `Memory::shadow_write_ptr`, pointing into live guest RAM.
        let got = unsafe { *(ptr as *const u8) };
        if exp == got {
            idx += 1;
            continue;
        }
        let base8 = ptr & !7;
        let base4 = ptr & !3;
        if ram_nan_f64(&expected_ram, base8) {
            while idx < ram_ptrs.len() && ram_ptrs[idx] < base8 + 8 {
                idx += 1;
            }
            continue;
        }
        if ram_nan_f32(&expected_ram, base4) {
            while idx < ram_ptrs.len() && ram_ptrs[idx] < base4 + 4 {
                idx += 1;
            }
            continue;
        }
        let before = initial_ram.get(&ptr).copied().unwrap_or(0);
        diffs.push(format!(
            "ram@0x{ptr:X}: before=0x{before:02X} shadow=0x{exp:02X} aot=0x{got:02X}"
        ));
        idx += 1;
    }

    for (&word_idx, &exp) in &expected_mailbox {
        let got = memory.host_mailbox[word_idx as usize];
        if exp != got && !(f32_nan(exp) && f32_nan(got)) {
            let before = initial_mailbox.get(&word_idx).copied().unwrap_or(0);
            diffs.push(format!(
                "mailbox[{word_idx}]: before=0x{before:08X} shadow=0x{exp:08X} aot=0x{got:08X}"
            ));
        }
    }

    diffs
}

fn ram_nan_f64(expected_ram: &HashMap<usize, u8>, base: usize) -> bool {
    let mut exp = [0u8; 8];
    let mut got = [0u8; 8];
    for i in 0..8 {
        let Some(&b) = expected_ram.get(&(base + i)) else {
            return false;
        };
        exp[i] = b;
        // SAFETY: all 8 bytes exist in `expected_ram` from `shadow_write_ptr`.
        got[i] = unsafe { *((base + i) as *const u8) };
    }
    f64::from_le_bytes(exp).is_nan() && f64::from_le_bytes(got).is_nan()
}

fn ram_nan_f32(expected_ram: &HashMap<usize, u8>, base: usize) -> bool {
    let mut exp = [0u8; 4];
    let mut got = [0u8; 4];
    for i in 0..4 {
        let Some(&b) = expected_ram.get(&(base + i)) else {
            return false;
        };
        exp[i] = b;
        // SAFETY: all 4 bytes exist in `expected_ram` from `shadow_write_ptr`.
        got[i] = unsafe { *((base + i) as *const u8) };
    }
    f32::from_le_bytes(exp).is_nan() && f32::from_le_bytes(got).is_nan()
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod tests {
    use super::*;
    use crate::memory::{MemoryRegion, MemoryState, Permission};
    use crate::recompiled::{store, Library, HOST};
    use zakuro_aot::discover::{Program, Segment, Source};
    use zakuro_aot::{compile_program, CompileOptions};
    use zakuro_common::memory_map::{HOST_MAILBOX_VADDR, PAGE_SIZE};

    const CODE_BASE: u32 = 0x0010_0000;
    const RAM_BASE: u32 = 0x0800_0000;

    fn setup_memory(code_bytes: &[u8]) -> Memory {
        let mut memory = Memory::new(false, 0x0400_0000);
        let code_block = memory
            .phys
            .allocate(MemoryRegion::Application, PAGE_SIZE)
            .unwrap();
        memory.map(
            CODE_BASE,
            code_block.addr,
            PAGE_SIZE,
            Permission::RX,
            MemoryState::Code,
        );
        memory.write_physical(code_block.addr, code_bytes);

        let ram_block = memory
            .phys
            .allocate(MemoryRegion::Application, PAGE_SIZE * 2)
            .unwrap();
        memory.map(
            RAM_BASE,
            ram_block.addr,
            PAGE_SIZE * 2,
            Permission::RW,
            MemoryState::Private,
        );
        memory.host_mailbox = [0x1122_3344, 0x5566_7788, 0, 0];
        memory
    }

    fn compile_test_aot(code_bytes: &[u8], seeds: &[(u32, Source)]) -> LoadedAot {
        let program = Program {
            text: Segment {
                base: CODE_BASE,
                bytes: code_bytes.to_vec(),
            },
            seeds: seeds.to_vec(),
            slots: None,
        };
        let image = compile_program(0, 0x1234, &program, &[], CompileOptions::default(), |_, _| {});
        image.into_loaded_exec().unwrap()
    }

    fn put_u32(code: &mut [u8], off: usize, word: u32) {
        code[off..off + 4].copy_from_slice(&word.to_le_bytes());
    }

    fn put_u16(code: &mut [u8], off: usize, half: u16) {
        code[off..off + 2].copy_from_slice(&half.to_le_bytes());
    }

    fn make_mixed_arm_thumb_vfp_program() -> Vec<u8> {
        let mut code = vec![0u8; 0xA0];

        // Entry (ARM @ CODE_BASE + 0x00):
        // Reads mailbox[0], mutates + writes mailbox[2] and RAM, exercises BLX->Thumb,
        // conditional ARM branch, BL->VFP (including NaN sqrt), and SVC #0x2A.
        put_u32(&mut code, 0x00, 0xE591_2000); // LDR r2, [r1]
        put_u32(&mut code, 0x04, 0xE282_2001); // ADD r2, r2, #1
        put_u32(&mut code, 0x08, 0xE581_2008); // STR r2, [r1, #8]
        put_u32(&mut code, 0x0C, 0xE580_2000); // STR r2, [r0, #0]
        put_u32(&mut code, 0x10, 0xE5D0_3001); // LDRB r3, [r0, #1]
        put_u32(&mut code, 0x14, 0xE5C0_3004); // STRB r3, [r0, #4]
        put_u32(&mut code, 0x18, 0xFA00_0008); // BLX CODE_BASE + 0x40 (Thumb)
        put_u32(&mut code, 0x1C, 0xEB00_000F); // BL CODE_BASE + 0x60 (ARM VFP)
        put_u32(&mut code, 0x20, 0xE353_0033); // CMP r3, #0x33
        put_u32(&mut code, 0x24, 0x0A00_0001); // BEQ CODE_BASE + 0x30 (taken!)
        put_u32(&mut code, 0x28, 0xE3A0_2000); // MOV r2, #0 (skipped)
        put_u32(&mut code, 0x2C, 0xE1A0_0000); // NOP
        put_u32(&mut code, 0x30, 0xEF00_002A); // SVC #0x2A
        put_u32(&mut code, 0x34, 0xE1A0_0000); // NOP
        put_u32(&mut code, 0x38, 0xE1A0_0000); // NOP
        put_u32(&mut code, 0x3C, 0xE1A0_0000); // NOP

        // Thumb function @ CODE_BASE + 0x40:
        put_u16(&mut code, 0x40, 0xB510); // PUSH {r4, lr}
        put_u16(&mut code, 0x42, 0x2407); // MOVS r4, #7
        put_u16(&mut code, 0x44, 0x3405); // ADDS r4, #5 (r4 = 12)
        put_u16(&mut code, 0x46, 0x6084); // STR r4, [r0, #8]
        put_u16(&mut code, 0x48, 0xF000); // BL prefix -> target CODE_BASE + 0x54
        put_u16(&mut code, 0x4A, 0xF804); // BL suffix
        put_u16(&mut code, 0x4C, 0x60C4); // STR r4, [r0, #12] (r4 = 15)
        put_u16(&mut code, 0x4E, 0xBD10); // POP {r4, pc}
        put_u16(&mut code, 0x50, 0x46C0); // NOP
        put_u16(&mut code, 0x52, 0x46C0); // NOP

        // Thumb helper @ CODE_BASE + 0x54:
        put_u16(&mut code, 0x54, 0x3403); // ADDS r4, #3 (r4 = 15)
        put_u16(&mut code, 0x56, 0x4770); // BX lr
        put_u16(&mut code, 0x58, 0x46C0); // NOP
        put_u16(&mut code, 0x5A, 0x46C0); // NOP
        put_u16(&mut code, 0x5C, 0x46C0); // NOP
        put_u16(&mut code, 0x5E, 0x46C0); // NOP

        // ARM VFP helper @ CODE_BASE + 0x60:
        put_u32(&mut code, 0x60, 0xEE30_3A20); // VADD.F32 s6, s0, s1
        put_u32(&mut code, 0x64, 0xEE21_4B02); // VMUL.F64 d4, d1, d2
        put_u32(&mut code, 0x68, 0xEEF1_3AC5); // VSQRT.F32 s7, s10 (-1.0 -> NaN)
        put_u32(&mut code, 0x6C, 0xED80_3A04); // VSTR.F32 s6, [r0, #16]
        put_u32(&mut code, 0x70, 0xED80_4B06); // VSTR.F64 d4, [r0, #24]
        put_u32(&mut code, 0x74, 0xEDC0_3A08); // VSTR.F32 s7, [r0, #32]
        put_u32(&mut code, 0x78, 0xEEB4_3A40); // VCMP.F32 s6, s0
        put_u32(&mut code, 0x7C, 0xEEF1_FA10); // VMRS APSR_nzcv, FPSCR
        put_u32(&mut code, 0x80, 0xE12F_FF1E); // BX lr
        code
    }

    #[test]
    fn lockstep_verifies_arm_thumb_vfp_ram_and_mailbox_program() {
        let code = make_mixed_arm_thumb_vfp_program();
        let mut memory = setup_memory(&code);
        let aot = compile_test_aot(&code, &[(CODE_BASE, Source::Entry)]);
        let library = Library::from_aot(aot);

        let mut cpu = Cpu::new();
        cpu.regs[0] = RAM_BASE;
        cpu.regs[1] = HOST_MAILBOX_VADDR;
        cpu.regs[13] = RAM_BASE + 0x1800;
        cpu.regs[14] = 0xFFFF_FFF0;
        cpu.regs[15] = CODE_BASE;
        cpu.vfp.regs[0] = 1.5f32.to_bits();
        cpu.vfp.regs[1] = 2.25f32.to_bits();
        cpu.vfp.set_f64(1, 3.5);
        cpu.vfp.set_f64(2, 1.25);
        cpu.vfp.regs[10] = (-1.0f32).to_bits();

        let (ran, stop) = library.run_with_stride(&mut cpu, &mut memory, 1000, 1);
        assert!(matches!(stop, Stop::Svc(0x2A)), "expected SVC 0x2A, got {stop:?}");
        assert!(ran > 0);
        assert_eq!(memory.host_mailbox[2], 0x1122_3345);
        assert_eq!(memory.peek32(RAM_BASE), 0x1122_3345);
        assert_eq!(memory.read8(RAM_BASE + 4), 0x33);
        assert_eq!(memory.peek32(RAM_BASE + 8), 12);
        assert_eq!(memory.peek32(RAM_BASE + 12), 15);
        assert_eq!(f32::from_bits(memory.peek32(RAM_BASE + 16)), 3.75);
        let d3 = ((memory.peek32(RAM_BASE + 28) as u64) << 32) | (memory.peek32(RAM_BASE + 24) as u64);
        assert_eq!(f64::from_bits(d3), 4.375);
        assert!(f32::from_bits(memory.peek32(RAM_BASE + 32)).is_nan());
    }

    fn run_fault_injection(
        words: &[u32],
        init_cpu: impl FnOnce(&mut Cpu),
        inject: impl FnMut(&mut Context, &mut Memory),
    ) {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut memory = setup_memory(&code);
        let aot = compile_test_aot(&code, &[(CODE_BASE, Source::Entry)]);
        let library = Library::from_aot(aot);

        let mut cpu = Cpu::new();
        cpu.regs[0] = RAM_BASE;
        cpu.regs[13] = RAM_BASE + 0x1800;
        cpu.regs[14] = 0xFFFF_FFF0;
        cpu.regs[15] = CODE_BASE;
        init_cpu(&mut cpu);

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
            budget: 64,
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
        store(&cpu, &mut ctx);
        let mut machine = Machine {
            cpu: &mut cpu,
            memory: &mut memory,
            library: &library,
            pending: None,
        };
        ctx.vfp = machine.cpu.vfp.regs.as_mut_ptr();
        ctx.fpscr = &mut machine.cpu.vfp.fpscr;
        ctx.user = &mut machine as *mut Machine<'_> as *mut c_void;

        let target_rx = match &library.backend {
            crate::recompiled::Backend::Aot(loaded) => loaded.lookup(CODE_BASE).unwrap(),
            _ => unreachable!(),
        };
        let loaded = match &library.backend {
            crate::recompiled::Backend::Aot(loaded) => loaded,
            _ => unreachable!(),
        };
        verify_block_with_hook(loaded, &mut ctx, &mut machine, target_rx, inject);
    }

    #[test]
    #[should_panic(expected = "LOCKSTEP MISMATCH")]
    fn fault_injection_corrupted_r0_panics() {
        run_fault_injection(
            &[0xE280_0001, 0xE12F_FF1E], // ADD r0, r0, #1; BX lr
            |_| {},
            |ctx, _| {
                ctx.r[0] ^= 0x10;
            },
        );
    }

    #[test]
    #[should_panic(expected = "LOCKSTEP MISMATCH")]
    fn fault_injection_corrupted_c_flag_panics() {
        run_fault_injection(
            &[0xE290_0001, 0xE12F_FF1E], // ADDS r0, r0, #1; BX lr
            |_| {},
            |ctx, _| {
                ctx.c ^= 1;
            },
        );
    }

    #[test]
    fn budget_too_small_returns_zero_ran_and_preserves_state() {
        // 4-instruction block: ADD r0, r0, #1; ADD r0, r0, #2; ADD r0, r0, #3; SVC #1
        let words = [0xE280_0001u32, 0xE280_0002, 0xE280_0003, 0xEF00_0001];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut memory = setup_memory(&code);
        let aot = compile_test_aot(&code, &[(CODE_BASE, Source::Entry)]);
        let library = Library::from_aot(aot);

        let mut cpu = Cpu::new();
        cpu.regs[0] = 10;
        cpu.regs[15] = CODE_BASE;

        let (ran, stop) = library.run_with_stride(&mut cpu, &mut memory, 2, 1);
        assert_eq!(ran, 0);
        assert!(matches!(stop, Stop::Left));
        assert_eq!(cpu.regs[0], 10);
        assert_eq!(cpu.regs[15], CODE_BASE);
    }

    #[test]
    fn stride_gt_1_verifies_first_block_and_completes_program() {
        let code = make_mixed_arm_thumb_vfp_program();
        let mut memory = setup_memory(&code);
        let aot = compile_test_aot(&code, &[(CODE_BASE, Source::Entry)]);
        let library = Library::from_aot(aot);

        let mut cpu = Cpu::new();
        cpu.regs[0] = RAM_BASE;
        cpu.regs[1] = HOST_MAILBOX_VADDR;
        cpu.regs[13] = RAM_BASE + 0x1800;
        cpu.regs[14] = 0xFFFF_FFF0;
        cpu.regs[15] = CODE_BASE;
        cpu.vfp.regs[0] = 1.5f32.to_bits();
        cpu.vfp.regs[1] = 2.25f32.to_bits();
        cpu.vfp.set_f64(1, 3.5);
        cpu.vfp.set_f64(2, 1.25);
        cpu.vfp.regs[10] = (-1.0f32).to_bits();

        let (ran, stop) = library.run_with_stride(&mut cpu, &mut memory, 1000, 3);
        assert!(matches!(stop, Stop::Svc(0x2A)));
        assert!(ran > 0);
        assert_eq!(memory.peek32(RAM_BASE + 12), 15);
    }

    #[test]
    fn shadow_bus_does_not_mutate_real_memory_or_faults() {
        let code = [0x01u8, 0x02, 0x03, 0x04];
        let mut memory = setup_memory(&code);
        memory.write32(RAM_BASE, 0xDEAD_BEEF);

        let mut state = ShadowState::default();
        let mut bus = ShadowBus::new(&mut memory, &mut state);
        bus.set_step(1);
        bus.write32(RAM_BASE, 0x1234_5678);
        assert_eq!(bus.read32(RAM_BASE), 0x1234_5678);
        // Write to RX-only code page & unmapped address should not mutate or record faults.
        bus.write8(CODE_BASE, 0xFF);
        assert_eq!(bus.read8(CODE_BASE), 0x01);
        assert_eq!(bus.read32(0x2000_0000), 0);
        bus.write32(0x2000_0000, 0xABCD);
        // Mailbox write stays in shadow overlay.
        bus.write32(HOST_MAILBOX_VADDR + 4, 0x9999_8888);
        assert_eq!(bus.read32(HOST_MAILBOX_VADDR + 4), 0x9999_8888);
        assert!(state.unmapped_accesses >= 3);

        assert_eq!(memory.peek32(RAM_BASE), 0xDEAD_BEEF);
        assert_eq!(memory.host_mailbox[1], 0x5566_7788);
        assert!(memory.fault_summary().is_empty());
    }

    #[test]
    #[should_panic(expected = "LOCKSTEP MISMATCH")]
    fn fault_injection_corrupted_guest_memory_panics() {
        run_fault_injection(
            &[0xE580_1000, 0xE12F_FF1E], // STR r1, [r0]; BX lr
            |cpu| {
                cpu.regs[1] = 0xCAFE_BABE;
            },
            |_, memory| {
                let prev = memory.read8(RAM_BASE + 1);
                memory.write8(RAM_BASE + 1, prev ^ 0xFF);
            },
        );
    }
}
