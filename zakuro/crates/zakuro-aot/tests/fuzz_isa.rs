//! Randomized single-instruction differential fuzzer (`dynarmic`-style)
//! comparing `zakuro-aot` compiled native code against `zakuro-cpu` across
//! ARMv6K ALU/shifts/flags, MUL/DSP/saturating/parallel media, ARM & Thumb
//! memory (fast-page, cross-page, unaligned, slow-path), Thumb-1 ALU/high-reg/
//! extends, and VFPv2 scalar/vector/conversions/rounding modes.

#![cfg(all(unix, any(target_arch = "aarch64", target_arch = "x86_64")))]

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use recomp_abi::{Context, Host, EXIT_BUDGET, EXIT_NONE, EXIT_SVC, EXIT_UNWIND};
use zakuro_aot::discover::{Program, Segment, Source};
use zakuro_aot::{compile_program, CompileOptions, LoadedAot};
use zakuro_cpu::{Bus, Cpu, Exit};

const PAGE_SIZE: usize = 0x1000;
const BASE: u32 = 0x0010_0000;
const RETURN: u32 = 0xFFFF_FFF0;
const STACK_TOP: u32 = 0x1000_0000;
const STACK_SIZE: u32 = 0x4000;
const HEAP: u32 = 0x0800_0000;
const HEAP_SIZE: u32 = 0x4000;
const UNMAPPED_SCRATCH: u32 = 0x0600_0000; // handled only through slow-path callbacks!
const TLS: u32 = 0x1FF8_2000;
const LIMIT: i32 = 1000;

const BATCH_FUNCS: usize = 1024;
const SLOT_BYTES: usize = 16;
const TEXT_BYTES_LEN: usize = BATCH_FUNCS * SLOT_BYTES;

const DEFAULT_BATCHES_PER_SUITE: usize = 50; // 50 * 1024 = 51,200 per suite -> 256,000 total
const STRESS_BATCHES_PER_SUITE: usize = 200; // 200 * 1024 = 204,800 per suite -> 1,024,000 total

const MAX_PRINT_FAILS: usize = 16;

// ---------------------------------------------------------------------------
// Deterministic SplitMix64 PRNG & Edge Operand Pools
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn gen_range(&mut self, upper: u32) -> u32 {
        (self.next_u64() % (upper as u64)) as u32
    }

    fn gen_bool(&mut self) -> bool {
        (self.next_u64() >> 63) != 0
    }

    fn pick<T: Copy>(&mut self, slice: &[T]) -> T {
        slice[self.gen_range(slice.len() as u32) as usize]
    }
}

const EDGE_U32: &[u32] = &[
    0,
    1,
    2,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFE,
    0xFFFF_FFFF,
    0x5555_5555,
    0xAAAA_AAAA,
    // Extra halfword/byte boundary values for DSP/SIMD/saturating coverage:
    0x0000_7FFF,
    0x0000_8000,
    0x7FFF_8000,
    0x8000_7FFF,
    0x00FF_00FF,
    0xFF00_FF00,
    0x7F7F_7F7F,
    0x8080_8080,
];

const EDGE_SHIFTS: &[u32] = &[
    0,
    1,
    7,
    8,
    15,
    16,
    31,
    32,
    33,
    63,
    64,
    128,
    255,
    0x100,
    0x120,
    0xFFFF_FF00,
    0xDEAD_0020,
];

const EDGE_BYTES: &[u32] = &[0x00, 0x01, 0x02, 0x7E, 0x7F, 0x80, 0x81, 0xFE, 0xFF];

const EDGE_F32_BITS: &[u32] = &[
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x0040_0000, // +subnormal 2^-127
    0x8040_0000, // -subnormal -2^-127
    0x0000_0001, // +smallest subnormal
    0x8000_0001, // -smallest subnormal
    0x007F_FFFF, // +largest subnormal
    0x807F_FFFF, // -largest subnormal
    0x0080_0000, // +smallest normal
    0x8080_0000, // -smallest normal
    0x3F80_0000, // +1.0
    0xBF80_0000, // -1.0
    0x3F00_0000, // +0.5
    0xBF00_0000, // -0.5
    0x3FC0_0000, // +1.5
    0xBFC0_0000, // -1.5
    0x4020_0000, // +2.5
    0xC020_0000, // -2.5
    0x3F80_0001, // 1.0 + 2^-23
    0x3380_0000, // 2^-24
    0xB380_0000, // -2^-24
    0x3FE0_0000, // +1.75
    0xBF66_6666, // ~ -0.9
    0x7F7F_FFFF, // +f32::MAX
    0xFF7F_FFFF, // -f32::MAX
    0x7F80_0000, // +Inf
    0xFF80_0000, // -Inf
    0x7FC0_0000, // +qNaN
    0xFFC0_1234, // -qNaN with payload
    0x7F80_0001, // +sNaN
    0xFF80_0001, // -sNaN
    // Boundary floats near i32 / u32 limits:
    0x4EFF_FFFF, // 2147483520.0 (largest f32 < 2^31)
    0x4F00_0000, // 2147483648.0 (2^31)
    0xCF00_0000, // -2147483648.0 (-2^31)
    0xCF00_0001, // -2147483904.0 (largest magnitude neg f32 < -2^31)
    0x4F7F_FFFF, // 4294967040.0 (largest f32 < 2^32)
    0x4F80_0000, // 4294967296.0 (2^32)
    0x60AD_78EC, // ~1e20
    0xE0AD_78EC, // ~-1e20
];

const EDGE_F64_BITS: &[u64] = &[
    0x0000_0000_0000_0000, // +0.0
    0x8000_0000_0000_0000, // -0.0
    0x0008_0000_0000_0000, // +subnormal
    0x8008_0000_0000_0000, // -subnormal
    0x0000_0000_0000_0001, // +smallest subnormal
    0x8000_0000_0000_0001, // -smallest subnormal
    0x3800_0000_0000_0000, // 2^-127 (subnormal in f32!)
    0xB800_0000_0000_0000, // -2^-127
    0x36A0_0000_0000_0000, // 2^-149
    0x3FF0_0000_0000_0000, // +1.0
    0xBFF0_0000_0000_0000, // -1.0
    0x3FE0_0000_0000_0000, // +0.5
    0xBFE0_0000_0000_0000, // -0.5
    0x3FF8_0000_0000_0000, // +1.5
    0xBFF8_0000_0000_0000, // -1.5
    0x4004_0000_0000_0000, // +2.5
    0xC004_0000_0000_0000, // -2.5
    0x3FF0_0000_0000_0001, // 1.0 + 2^-52
    0x3CA0_0000_0000_0000, // 2^-53
    0xBCA0_0000_0000_0000, // -2^-53
    0x7FEF_FFFF_FFFF_FFFF, // +f64::MAX
    0xFFEF_FFFF_FFFF_FFFF, // -f64::MAX
    0x7FF0_0000_0000_0000, // +Inf
    0xFFF0_0000_0000_0000, // -Inf
    0x7FF8_0000_0000_0000, // +qNaN
    0xFFF8_0000_0000_1234, // -qNaN with payload
    0x7FF0_0000_0000_0001, // +sNaN
    0xFFF0_0000_0000_0001, // -sNaN
    // Boundary doubles near i32 / u32 limits:
    0x41DF_FFFF_FE00_0000, // 2147483520.0
    0x41DF_FFFF_FFC0_0000, // 2147483647.0
    0x41DF_FFFF_FFD0_0000, // 2147483647.25
    0x41DF_FFFF_FFE0_0000, // 2147483647.5
    0x41DF_FFFF_FFF0_0000, // 2147483647.75
    0x41E0_0000_0000_0000, // 2147483648.0
    0xC1E0_0000_0000_0000, // -2147483648.0
    0xC1E0_0000_0010_0000, // -2147483648.5
    0xC1E0_0000_0020_0000, // -2147483649.0
    0xC1E0_0000_0200_0000, // -2147483904.0
    0x41EF_FFFF_FE00_0000, // 4294967040.0
    0x41EF_FFFF_FFE0_0000, // 4294967295.0
    0x41EF_FFFF_FFE8_0000, // 4294967295.25
    0x41EF_FFFF_FFF0_0000, // 4294967295.5
    0x41EF_FFFF_FFF8_0000, // 4294967295.75
    0x41F0_0000_0000_0000, // 4294967296.0
    0x4415_AF1D_78B5_8C40, // 1e20
    0xC415_AF1D_78B5_8C40, // -1e20
];

fn rand_u32_biased(rng: &mut SplitMix64) -> u32 {
    if rng.gen_bool() {
        rng.pick(EDGE_U32)
    } else {
        rng.next_u32()
    }
}

fn rand_shift_count(rng: &mut SplitMix64) -> u32 {
    if rng.gen_bool() {
        rng.pick(EDGE_SHIFTS)
    } else {
        rng.next_u32()
    }
}

fn rand_f32_bits(rng: &mut SplitMix64) -> u32 {
    if rng.gen_bool() {
        rng.pick(EDGE_F32_BITS)
    } else {
        rng.next_u32()
    }
}

fn rand_f64_bits(rng: &mut SplitMix64) -> u64 {
    if rng.gen_bool() {
        rng.pick(EDGE_F64_BITS)
    } else {
        rng.next_u64()
    }
}

// ---------------------------------------------------------------------------
// Differential Test Harness (self-contained, matching `verify.rs`)
// ---------------------------------------------------------------------------

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
        let mut read_pages = vec![std::ptr::null_mut(); 1 << 20];
        let mut write_pages = vec![std::ptr::null_mut(); 1 << 20];
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

    fn set_text(&mut self, text_bytes: &[u8]) {
        debug_assert_eq!(self.regions[0].base, BASE);
        debug_assert_eq!(self.regions[0].bytes.len(), text_bytes.len());
        self.regions[0].bytes.copy_from_slice(text_bytes);
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
        let was_thumb = ctx.thumb != 0;
        load_ctx(ctx, &mut run.cpu);
        run.cpu.regs[15] = address;
        let exit = run.cpu.step(&mut run.memory);
        store_ctx(&run.cpu, ctx);
        run.fallbacks += 1;
        let step = if was_thumb { 2 } else { 4 };
        if exit.is_some() || (ctx.thumb != 0) != was_thumb || ctx.r[15] != address.wrapping_add(step) {
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum VfpCheckMode {
    /// Default check matching `verify.rs::same_vfp` (f64 NaN or f32 NaN).
    Standard,
    /// Double-precision arithmetic: allows f64 NaN equivalence per double register.
    DoubleFloat,
    /// Single-precision arithmetic: only allows f32 NaN equivalence per single register.
    SingleFloat,
    /// Exact integer bit comparison in `s[sd]` (plus standard NaN checks elsewhere).
    IntDst(usize),
}

fn compare_runs(a: &Run, b: &Run, ctx: &Context, vfp_mode: VfpCheckMode) -> Vec<String> {
    let mut diffs = Vec::new();
    let nan = |bits: u32| f32::from_bits(bits).is_nan();
    for i in 0..16 {
        if a.cpu.regs[i] != ctx.r[i] {
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
    let double = |r: &[u32; 32], d: usize| {
        f64::from_bits(((r[2 * d + 1] as u64) << 32) | (r[2 * d] as u64)).is_nan()
    };
    for s in 0..32 {
        let va = a.cpu.vfp.regs[s];
        let vb = b.cpu.vfp.regs[s];
        if va == vb {
            continue;
        }
        let allowed = match vfp_mode {
            VfpCheckMode::DoubleFloat | VfpCheckMode::Standard => {
                (double(&a.cpu.vfp.regs, s / 2) && double(&b.cpu.vfp.regs, s / 2))
                    || (nan(va) && nan(vb))
            }
            VfpCheckMode::SingleFloat => nan(va) && nan(vb),
            VfpCheckMode::IntDst(sd) => s != sd && nan(va) && nan(vb),
        };
        if !allowed {
            diffs.push(format!("s{s}: interp={va:08X} aot={vb:08X}"));
        }
    }
    if a.cpu.vfp.fpscr != b.cpu.vfp.fpscr {
        diffs.push(format!(
            "fpscr: interp={:08X} aot={:08X}",
            a.cpu.vfp.fpscr, b.cpu.vfp.fpscr
        ));
    }
    if let Some(addr) = a.memory.first_difference(&b.memory) {
        diffs.push(format!("mem@{addr:08X}"));
    }
    if a.memory.faults != b.memory.faults {
        diffs.push(format!("faults: interp={} aot={}", a.memory.faults, b.memory.faults));
    }
    diffs
}

// ---------------------------------------------------------------------------
// Batched Slot & Pre-State Model
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Slot {
    opcodes: Vec<u32>,
    thumb: bool,
    desc: &'static str,
    init_regs: [u32; 13],
    init_nzcvq: u8, // bits [4:0] = N,Z,C,V,Q
    init_ge: u8,
    init_excl: Option<u32>,
    init_fpscr: u32,
    init_vfp: [u32; 32],
    vfp_mode: VfpCheckMode,
}

impl Slot {
    fn encode_into(&self, slot_bytes: &mut [u8]) {
        slot_bytes.fill(0);
        if self.thumb {
            let mut off = 0usize;
            for &hw in &self.opcodes {
                slot_bytes[off..off + 2].copy_from_slice(&(hw as u16).to_le_bytes());
                off += 2;
            }
            slot_bytes[off..off + 2].copy_from_slice(&0x4770u16.to_le_bytes()); // BX LR
        } else {
            let mut off = 0usize;
            for &w in &self.opcodes {
                slot_bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
                off += 4;
            }
            slot_bytes[off..off + 4].copy_from_slice(&0xE12F_FF1Eu32.to_le_bytes()); // BX LR
        }
    }

    fn apply_to_cpu(&self, cpu: &mut Cpu, addr: u32) {
        cpu.regs[..13].copy_from_slice(&self.init_regs);
        cpu.regs[13] = STACK_TOP - 0x1000;
        cpu.regs[14] = RETURN;
        cpu.regs[15] = addr;
        cpu.cpsr = zakuro_cpu::Psr::default();
        cpu.cpsr.n = (self.init_nzcvq & 16) != 0;
        cpu.cpsr.z = (self.init_nzcvq & 8) != 0;
        cpu.cpsr.c = (self.init_nzcvq & 4) != 0;
        cpu.cpsr.v = (self.init_nzcvq & 2) != 0;
        cpu.cpsr.q = (self.init_nzcvq & 1) != 0;
        cpu.cpsr.ge = self.init_ge & 0xF;
        cpu.cpsr.thumb = self.thumb;
        cpu.exclusive_addr = self.init_excl;
        cpu.cp15.thread_id_ro = TLS;
        cpu.vfp.regs = self.init_vfp;
        cpu.vfp.fpscr = self.init_fpscr;
        cpu.vfp.fpexc = 1 << 30;
    }

    fn format_opcodes(&self) -> String {
        if self.thumb {
            let parts: Vec<String> = self.opcodes.iter().map(|w| format!("{:04X}", *w as u16)).collect();
            format!("[{}] ({})", parts.join(", "), self.desc)
        } else {
            let parts: Vec<String> = self.opcodes.iter().map(|w| format!("{w:08X}")).collect();
            format!("[{}] ({})", parts.join(", "), self.desc)
        }
    }

    fn format_pre_state(&self) -> String {
        let regs: Vec<String> = self.init_regs.iter().enumerate().map(|(i, v)| format!("r{i}={v:08X}")).collect();
        format!(
            "thumb={} nzcvq={:05b} ge={:X} excl={:X?} fpscr={:08X} regs=[{}]",
            self.thumb as u8,
            self.init_nzcvq,
            self.init_ge,
            self.init_excl,
            self.init_fpscr,
            regs.join(" "),
        )
    }
}

fn make_pristine_memory() -> TestMemory {
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
    let mut heap = vec![0u8; HEAP_SIZE as usize];
    for w in heap.chunks_exact_mut(4) {
        let v = rand_u32_biased(&mut rng);
        w.copy_from_slice(&v.to_le_bytes());
    }
    let mut scratch = vec![0u8; PAGE_SIZE * 2];
    for w in scratch.chunks_exact_mut(4) {
        let v = rand_u32_biased(&mut rng);
        w.copy_from_slice(&v.to_le_bytes());
    }
    TestMemory::new(vec![
        Region { base: BASE, bytes: vec![0u8; TEXT_BYTES_LEN], writable: false, direct_page: true },
        Region { base: HEAP, bytes: heap, writable: true, direct_page: true },
        Region { base: UNMAPPED_SCRATCH, bytes: scratch, writable: true, direct_page: false },
        Region { base: STACK_TOP - STACK_SIZE, bytes: vec![0u8; STACK_SIZE as usize], writable: true, direct_page: true },
        Region { base: TLS, bytes: vec![0u8; PAGE_SIZE], writable: true, direct_page: true },
    ])
}

struct ScratchRunner {
    pristine: TestMemory,
    a: Run,
    b: Run,
}

impl ScratchRunner {
    fn new() -> Self {
        let pristine = make_pristine_memory();
        let a = Run {
            memory: TestMemory::new(pristine.regions.clone()),
            cpu: Cpu::new(),
            pending: None,
            interpreted: 0,
            fallbacks: 0,
        };
        let b = Run {
            memory: TestMemory::new(pristine.regions.clone()),
            cpu: Cpu::new(),
            pending: None,
            interpreted: 0,
            fallbacks: 0,
        };
        Self { pristine, a, b }
    }
}

struct CompiledBatch {
    batch_idx: usize,
    batch_seed: u64,
    text_bytes: Vec<u8>,
    slots: Vec<Slot>,
    aot: LoadedAot,
}

fn build_and_compile_batch(
    batch_idx: usize,
    batch_seed: u64,
    gen: impl Fn(&mut SplitMix64) -> Slot,
) -> CompiledBatch {
    let mut rng = SplitMix64::new(batch_seed);
    let mut text_bytes = vec![0u8; TEXT_BYTES_LEN];
    let mut seeds = Vec::with_capacity(BATCH_FUNCS);
    let mut slots = Vec::with_capacity(BATCH_FUNCS);

    for idx in 0..BATCH_FUNCS {
        let slot = gen(&mut rng);
        let addr = BASE + (idx * SLOT_BYTES) as u32;
        slot.encode_into(&mut text_bytes[idx * SLOT_BYTES..(idx + 1) * SLOT_BYTES]);
        seeds.push((addr | (slot.thumb as u32), Source::Entry));
        slots.push(slot);
    }

    let program = Program {
        text: Segment { base: BASE, bytes: text_bytes.clone() },
        seeds,
        slots: None,
    };
    let options = CompileOptions { fold_text_literals: true, emit_vfp_vectors: true };
    let image = compile_program(0, batch_seed, &program, &[], options, |_, _| {});
    let aot = image.into_loaded_exec().expect("load fuzzed AOT batch");
    CompiledBatch { batch_idx, batch_seed, text_bytes, slots, aot }
}

fn run_fuzz_suite_raw(
    test_name: &str,
    base_seed: u64,
    num_batches: usize,
    gen: &(impl Fn(&mut SplitMix64) -> Slot + Sync),
) -> (usize, u64, u64, usize) {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    let t0 = Instant::now();
    let mismatches = AtomicUsize::new(0);
    let total_execs = AtomicUsize::new(0);
    let total_fallbacks = AtomicUsize::new(0);
    let total_interpreted = AtomicUsize::new(0);
    let seen_kinds: Mutex<BTreeSet<(&'static str, bool, u32)>> = Mutex::new(BTreeSet::new());

    // Compile in parallel-friendly chunks so `compile_program` uses its own thread pool
    // cleanly without oversubscribing N^2 nested threads, then execute batches in parallel
    // across Rayon worker threads with per-thread reusable `(Run, Run)` scratch state.
    const CHUNK_BATCHES: usize = 16;
    for chunk_start in (0..num_batches).step_by(CHUNK_BATCHES) {
        let chunk_end = (chunk_start + CHUNK_BATCHES).min(num_batches);
        let compiled: Vec<CompiledBatch> = (chunk_start..chunk_end)
            .map(|batch_idx| {
                let mut mix = SplitMix64::new(base_seed ^ (batch_idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
                let batch_seed = mix.next_u64();
                build_and_compile_batch(batch_idx, batch_seed, gen)
            })
            .collect();

        compiled.into_par_iter().for_each_init(ScratchRunner::new, |scratch, batch| {
            scratch.a.memory.set_text(&batch.text_bytes);
            scratch.b.memory.set_text(&batch.text_bytes);

            for (idx, slot) in batch.slots.iter().enumerate() {
                let addr = BASE + (idx * SLOT_BYTES) as u32;
                scratch.a.memory.copy_from(&scratch.pristine);
                scratch.b.memory.copy_from(&scratch.pristine);

                slot.apply_to_cpu(&mut scratch.a.cpu, addr);
                scratch.b.cpu = scratch.a.cpu.clone();
                scratch.a.pending = None;
                scratch.b.pending = None;
                scratch.b.interpreted = 0;
                scratch.b.fallbacks = 0;

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
                    tls: TLS,
                    read_pages: scratch.b.memory.read_pages.as_ptr(),
                    write_pages: scratch.b.memory.write_pages.as_ptr(),
                    vfp: scratch.b.cpu.vfp.regs.as_mut_ptr(),
                    fpscr: &mut scratch.b.cpu.vfp.fpscr,
                    host: &HOST,
                    user: &mut scratch.b as *mut Run as *mut c_void,
                };
                store_ctx(&scratch.b.cpu, &mut ctx);

                let exp_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_interpreted(&mut scratch.a, LIMIT)
                }));
                let got_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    run_aot(&batch.aot, &mut scratch.b, &mut ctx, LIMIT, LIMIT)
                }));

                total_execs.fetch_add(1, Ordering::Relaxed);
                total_fallbacks.fetch_add(scratch.b.fallbacks as usize, Ordering::Relaxed);
                total_interpreted.fetch_add(scratch.b.interpreted as usize, Ordering::Relaxed);

                let mut diffs = match (&exp_res, &got_res) {
                    (Ok(exp), Ok(got)) => {
                        let mut d = compare_runs(&scratch.a, &scratch.b, &ctx, slot.vfp_mode);
                        if exp != got {
                            d.insert(0, format!("stop: interp={exp:?} aot={got:?}"));
                        }
                        d
                    }
                    (Err(_), Ok(got)) => vec![format!("PANIC in zakuro-cpu interpreter (aot={got:?})")],
                    (Ok(exp), Err(_)) => vec![format!("PANIC in zakuro-aot (interp={exp:?})")],
                    (Err(_), Err(_)) => vec!["PANIC in both zakuro-cpu and zakuro-aot".to_owned()],
                };

                if !diffs.is_empty() {
                    let count = mismatches.fetch_add(1, Ordering::Relaxed);
                    let is_double = slot.vfp_mode == VfpCheckMode::DoubleFloat;
                    let rmode = (slot.init_fpscr >> 22) & 3;
                    let is_new_kind = seen_kinds
                        .lock()
                        .unwrap()
                        .insert((slot.desc, is_double, rmode));
                    if count < MAX_PRINT_FAILS || is_new_kind {
                        eprintln!(
                            "MISMATCH #{}: test={test_name} batch={} idx={} seed=0x{:016X} addr=0x{addr:08X} opcodes={} pre=({}) diffs={:?}",
                            count + 1,
                            batch.batch_idx,
                            idx,
                            batch.batch_seed,
                            slot.format_opcodes(),
                            slot.format_pre_state(),
                            diffs,
                        );
                    }
                    let _ = &mut diffs;
                }
            }
        });
    }

    let execs = total_execs.load(Ordering::Relaxed);
    let fallbacks = total_fallbacks.load(Ordering::Relaxed) as u64;
    let interpreted = total_interpreted.load(Ordering::Relaxed) as u64;
    let fails = mismatches.load(Ordering::Relaxed);

    println!(
        "[{test_name}] execs={execs} batches={num_batches} fallbacks={fallbacks} uncompiled={interpreted} mismatches={fails} elapsed={:.2?}",
        t0.elapsed()
    );
    (execs, fallbacks, interpreted, fails)
}

fn run_fuzz_suite(
    test_name: &str,
    base_seed: u64,
    num_batches: usize,
    gen: &(impl Fn(&mut SplitMix64) -> Slot + Sync),
) {
    let (execs, _fallbacks, interpreted, fails) =
        run_fuzz_suite_raw(test_name, base_seed, num_batches, gen);
    assert_eq!(
        interpreted, 0,
        "{test_name}: {interpreted} slots fell back to interpreter due to missing compiled entry!"
    );
    assert_eq!(
        fails, 0,
        "{test_name}: {fails} differential mismatches out of {execs} executions!"
    );
}

fn rand_regs_13(rng: &mut SplitMix64) -> [u32; 13] {
    std::array::from_fn(|_| rand_u32_biased(rng))
}

fn rand_vfp_32(rng: &mut SplitMix64) -> [u32; 32] {
    std::array::from_fn(|_| rand_f32_bits(rng))
}

fn rand_arm_cond(rng: &mut SplitMix64) -> u32 {
    if rng.gen_bool() {
        14 // AL (always)
    } else {
        rng.gen_range(15) // 0..=14
    }
}

// ---------------------------------------------------------------------------
// Suite 1: ARM ALU, Shifts, Conditions, R15 Reads, and Flags
// ---------------------------------------------------------------------------

fn gen_arm_alu(rng: &mut SplitMix64) -> Slot {
    let mut init_regs = rand_regs_13(rng);
    let opcode = rng.gen_range(16);
    let is_compare = (8..=11).contains(&opcode);
    let s_bit = if is_compare { true } else { rng.gen_bool() };
    let cond = rand_arm_cond(rng);
    let rd = if is_compare { 0 } else { rng.gen_range(13) };

    let desc;
    let mode = rng.gen_range(10);
    let word = if mode < 3 {
        // Immediate
        desc = "arm_alu_imm";
        let rn = if matches!(opcode, 13 | 15) {
            0
        } else if rng.gen_range(8) == 0 {
            15
        } else {
            rng.gen_range(15)
        };
        let imm8 = if rng.gen_bool() { rng.pick(EDGE_BYTES) } else { rng.next_u32() & 0xFF };
        let rot4 = if rng.gen_bool() {
            rng.pick(&[0u32, 1, 2, 4, 8, 12, 15])
        } else {
            rng.gen_range(16)
        };
        (cond << 28)
            | (1 << 25)
            | (opcode << 21)
            | ((s_bit as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | (rot4 << 8)
            | imm8
    } else if mode < 7 {
        // Immediate shift (including RRX via ROR #0 and LSR/ASR #0 -> #32)
        desc = "arm_alu_imm_shift";
        let rn = if matches!(opcode, 13 | 15) {
            0
        } else if rng.gen_range(8) == 0 {
            15
        } else {
            rng.gen_range(15)
        };
        let rm = if rng.gen_range(8) == 0 { 15 } else { rng.gen_range(15) };
        let shift_ty = rng.gen_range(4);
        let imm5 = if rng.gen_bool() {
            rng.pick(&[0u32, 1, 2, 7, 8, 15, 16, 24, 30, 31])
        } else {
            rng.gen_range(32)
        };
        (cond << 28)
            | (opcode << 21)
            | ((s_bit as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | (imm5 << 7)
            | (shift_ty << 5)
            | rm
    } else {
        // Register shift (never use R15 in reg-shift!)
        desc = "arm_alu_reg_shift";
        let rn = if matches!(opcode, 13 | 15) { 0 } else { rng.gen_range(15) };
        let rm = rng.gen_range(15);
        let rs = rng.gen_range(15);
        let shift_ty = rng.gen_range(4);
        // If rs < 13 and rs != rm (and rs != rn when used), bias Rs specifically to shift-count edge cases:
        if rs < 13 && rs != rm && (matches!(opcode, 13 | 15) || rs != rn) {
            init_regs[rs as usize] = rand_shift_count(rng);
        }
        (cond << 28)
            | (opcode << 21)
            | ((s_bit as u32) << 20)
            | (rn << 16)
            | (rd << 12)
            | (rs << 8)
            | (shift_ty << 5)
            | (1 << 4)
            | rm
    };

    Slot {
        opcodes: vec![word],
        thumb: false,
        desc,
        init_regs,
        init_nzcvq: (rng.gen_range(32) as u8),
        init_ge: rng.gen_range(16) as u8,
        init_excl: None,
        init_fpscr: 0,
        init_vfp: [0; 32],
        vfp_mode: VfpCheckMode::Standard,
    }
}

// ---------------------------------------------------------------------------
// Suite 2: ARM Multiply, Halfword/Dual Multiply, Saturating, Media & Parallel SIMD
// ---------------------------------------------------------------------------

fn pick_distinct_pair(rng: &mut SplitMix64) -> (u32, u32) {
    let a = rng.gen_range(13);
    let mut b = rng.gen_range(12);
    if b >= a {
        b += 1;
    }
    (a, b)
}

fn gen_arm_mul_dsp_media(rng: &mut SplitMix64) -> Slot {
    let init_regs = rand_regs_13(rng);
    let cond = rand_arm_cond(rng);
    let r = |rng: &mut SplitMix64| rng.gen_range(13);

    let (word, desc) = match rng.gen_range(26) {
        0 => {
            // MUL / MLA
            let acc = rng.gen_bool();
            let s = rng.gen_bool();
            let rd = r(rng);
            let rn = if acc { r(rng) } else { 0 };
            let rs = r(rng);
            let rm = r(rng);
            (
                (cond << 28)
                    | ((acc as u32) << 21)
                    | ((s as u32) << 20)
                    | (rd << 16)
                    | (rn << 12)
                    | (rs << 8)
                    | 0x90
                    | rm,
                if acc { "mla" } else { "mul" },
            )
        }
        1 => {
            // UMULL / UMLAL / SMULL / SMLAL / UMAAL (RdHi != RdLo)
            let (rd_hi, rd_lo) = pick_distinct_pair(rng);
            let rs = r(rng);
            let rm = r(rng);
            if rng.gen_range(5) == 0 {
                (
                    (cond << 28) | 0x0040_0090 | (rd_hi << 16) | (rd_lo << 12) | (rs << 8) | rm,
                    "umaal",
                )
            } else {
                let signed = rng.gen_bool();
                let acc = rng.gen_bool();
                let s = rng.gen_bool();
                (
                    (cond << 28)
                        | (1 << 23)
                        | ((signed as u32) << 22)
                        | ((acc as u32) << 21)
                        | ((s as u32) << 20)
                        | (rd_hi << 16)
                        | (rd_lo << 12)
                        | (rs << 8)
                        | 0x90
                        | rm,
                    "mul_long",
                )
            }
        }
        2 => {
            // SMLAxy
            let (rd, rn, rs, rm) = (r(rng), r(rng), r(rng), r(rng));
            let x = rng.gen_bool() as u32;
            let y = rng.gen_bool() as u32;
            (
                (cond << 28)
                    | 0x0100_0080
                    | (rd << 16)
                    | (rn << 12)
                    | (rs << 8)
                    | (y << 6)
                    | (x << 5)
                    | rm,
                "smlaxy",
            )
        }
        3 => {
            // SMLAWy / SMULWy
            let is_smulw = rng.gen_bool();
            let rd = r(rng);
            let rn = if is_smulw { 0 } else { r(rng) };
            let rs = r(rng);
            let rm = r(rng);
            let y = rng.gen_bool() as u32;
            (
                (cond << 28)
                    | 0x0120_0080
                    | (rd << 16)
                    | (rn << 12)
                    | (rs << 8)
                    | (y << 6)
                    | ((is_smulw as u32) << 5)
                    | rm,
                if is_smulw { "smulwy" } else { "smlawy" },
            )
        }
        4 => {
            // SMLALxy (RdHi != RdLo)
            let (rd_hi, rd_lo) = pick_distinct_pair(rng);
            let rs = r(rng);
            let rm = r(rng);
            let x = rng.gen_bool() as u32;
            let y = rng.gen_bool() as u32;
            (
                (cond << 28)
                    | 0x0140_0080
                    | (rd_hi << 16)
                    | (rd_lo << 12)
                    | (rs << 8)
                    | (y << 6)
                    | (x << 5)
                    | rm,
                "smlalxy",
            )
        }
        5 => {
            // SMULxy
            let (rd, rs, rm) = (r(rng), r(rng), r(rng));
            let x = rng.gen_bool() as u32;
            let y = rng.gen_bool() as u32;
            (
                (cond << 28) | 0x0160_0080 | (rd << 16) | (rs << 8) | (y << 6) | (x << 5) | rm,
                "smulxy",
            )
        }
        6 | 7 => {
            // SMUAD[X] / SMLAD[X] / SMUSD[X] / SMLSD[X]
            let sub = rng.gen_bool() as u32;
            let m_swap = rng.gen_bool() as u32;
            let acc = rng.gen_bool();
            let rd = r(rng);
            let ra = if acc { r(rng) } else { 15 };
            let rs = r(rng);
            let rm = r(rng);
            (
                (cond << 28)
                    | 0x0700_0010
                    | (rd << 16)
                    | (ra << 12)
                    | (rs << 8)
                    | (sub << 6)
                    | (m_swap << 5)
                    | rm,
                "smuad_smlad_smusd_smlsd",
            )
        }
        8 => {
            // SMLALD[X] / SMLSLD[X] (RdHi != RdLo)
            let sub = rng.gen_bool() as u32;
            let m_swap = rng.gen_bool() as u32;
            let (rd_hi, rd_lo) = pick_distinct_pair(rng);
            let rs = r(rng);
            let rm = r(rng);
            (
                (cond << 28)
                    | 0x0740_0010
                    | (rd_hi << 16)
                    | (rd_lo << 12)
                    | (rs << 8)
                    | (sub << 6)
                    | (m_swap << 5)
                    | rm,
                "smlald_smlsld",
            )
        }
        9 => {
            // SMMUL[R] / SMMLA[R] / SMMLS[R]
            let sub = rng.gen_bool() as u32;
            let round = rng.gen_bool() as u32;
            let rd = r(rng);
            let ra = if sub == 0 && rng.gen_bool() { 15 } else { r(rng) };
            let rs = r(rng);
            let rm = r(rng);
            (
                (cond << 28)
                    | 0x0750_0010
                    | (rd << 16)
                    | (ra << 12)
                    | (rs << 8)
                    | (sub << 6)
                    | (round << 5)
                    | rm,
                "smmul_smmla_smmls",
            )
        }
        10 => {
            // USAD8 / USADA8
            let acc = rng.gen_bool();
            let rd = r(rng);
            let ra = if acc { r(rng) } else { 15 };
            let rs = r(rng);
            let rm = r(rng);
            (
                (cond << 28) | 0x0780_0010 | (rd << 16) | (ra << 12) | (rs << 8) | rm,
                if acc { "usada8" } else { "usad8" },
            )
        }
        11 => {
            // QADD / QSUB / QDADD / QDSUB
            let kind = rng.gen_range(4);
            let (rn, rd, rm) = (r(rng), r(rng), r(rng));
            (
                (cond << 28) | 0x0100_0050 | (kind << 21) | (rn << 16) | (rd << 12) | rm,
                "q_arith",
            )
        }
        12 => {
            // SSAT / USAT
            let unsigned = rng.gen_bool();
            let sat_imm = if rng.gen_bool() {
                rng.pick(&[0u32, 1, 7, 8, 15, 16, 30, 31])
            } else {
                rng.gen_range(32)
            };
            let rd = r(rng);
            let rm = r(rng);
            let asr = rng.gen_bool() as u32;
            let shift5 = if rng.gen_bool() {
                rng.pick(&[0u32, 1, 2, 7, 15, 16, 31])
            } else {
                rng.gen_range(32)
            };
            let base = if unsigned { 0x06E0_0010 } else { 0x06A0_0010 };
            (
                (cond << 28) | base | (sat_imm << 16) | (rd << 12) | (shift5 << 7) | (asr << 6) | rm,
                if unsigned { "usat" } else { "ssat" },
            )
        }
        13 => {
            // SSAT16 / USAT16
            let unsigned = rng.gen_bool();
            let sat_imm = rng.gen_range(16);
            let rd = r(rng);
            let rm = r(rng);
            let base = if unsigned { 0x06E0_0F30 } else { 0x06A0_0F30 };
            (
                (cond << 28) | base | (sat_imm << 16) | (rd << 12) | rm,
                if unsigned { "usat16" } else { "ssat16" },
            )
        }
        14 => {
            // PKHBT / PKHTB
            let tb = rng.gen_bool() as u32;
            let (rn, rd, rm) = (r(rng), r(rng), r(rng));
            let shift5 = if rng.gen_bool() {
                rng.pick(&[0u32, 1, 8, 16, 31])
            } else {
                rng.gen_range(32)
            };
            (
                (cond << 28)
                    | 0x0680_0010
                    | (rn << 16)
                    | (rd << 12)
                    | (shift5 << 7)
                    | (tb << 6)
                    | rm,
                if tb != 0 { "pkhtb" } else { "pkhbt" },
            )
        }
        15..=17 => {
            // SXTH/SXTB/UXTH/UXTB/SXTAH/SXTAB/UXTAH/UXTAB/SXTB16/SXTAB16/UXTB16/UXTAB16
            let op1_low3 = rng.pick(&[0b000u32, 0b010, 0b011, 0b100, 0b110, 0b111]);
            let acc = rng.gen_bool();
            let rn = if acc { r(rng) } else { 15 };
            let rd = r(rng);
            let rm = r(rng);
            let rot2 = rng.gen_range(4);
            (
                (cond << 28)
                    | 0x0680_0070
                    | (op1_low3 << 20)
                    | (rn << 16)
                    | (rd << 12)
                    | (rot2 << 10)
                    | rm,
                "extend",
            )
        }
        18 => {
            // REV / REV16 / REVSH / CLZ / SEL
            let (rn, rd, rm) = (r(rng), r(rng), r(rng));
            match rng.gen_range(5) {
                0 => ((cond << 28) | 0x06BF_0F30 | (rd << 12) | rm, "rev"),
                1 => ((cond << 28) | 0x06BF_0FB0 | (rd << 12) | rm, "rev16"),
                2 => ((cond << 28) | 0x06FF_0FB0 | (rd << 12) | rm, "revsh"),
                3 => ((cond << 28) | 0x016F_0F10 | (rd << 12) | rm, "clz"),
                _ => ((cond << 28) | 0x0680_0FB0 | (rn << 16) | (rd << 12) | rm, "sel"),
            }
        }
        _ => {
            // ALL ARMv6 parallel add/sub media ops:
            // {S, Q, SH, U, UQ, UH} x {ADD16, ASX, SAX, SUB16, ADD8, SUB8}
            let op1 = rng.pick(&[0b00001u32, 0b00010, 0b00011, 0b00101, 0b00110, 0b00111]);
            let op2 = rng.pick(&[0b000u32, 0b001, 0b010, 0b011, 0b100, 0b111]);
            let (rn, rd, rm) = (r(rng), r(rng), r(rng));
            (
                (cond << 28)
                    | 0x0600_0F10
                    | (op1 << 20)
                    | (rn << 16)
                    | (rd << 12)
                    | (op2 << 5)
                    | rm,
                "parallel_media",
            )
        }
    };

    Slot {
        opcodes: vec![word],
        thumb: false,
        desc,
        init_regs,
        init_nzcvq: rng.gen_range(32) as u8,
        init_ge: rng.gen_range(16) as u8,
        init_excl: None,
        init_fpscr: 0,
        init_vfp: [0; 32],
        vfp_mode: VfpCheckMode::Standard,
    }
}

// ---------------------------------------------------------------------------
// Suite 3: ARM & Thumb Memory (Fast-Page, Cross-Page, Unaligned, Slow-Path)
// ---------------------------------------------------------------------------

fn pick_target_addr(rng: &mut SplitMix64, align: u32) -> u32 {
    let raw = match rng.gen_range(10) {
        0..=3 => {
            // Fast-page HEAP (interior, avoiding out-of-bounds)
            HEAP + 0x40 + rng.gen_range(HEAP_SIZE - 0x100)
        }
        4..=6 => {
            // Cross-page HEAP boundary (`0xFFD..=0x1002` across interior boundaries at pages 1..=3)
            let page = rng.gen_range(3); // 0, 1, or 2 -> +0x1000, +0x2000, +0x3000
            let off = match align {
                8 => if rng.gen_bool() { 0xFF8 } else { 0x1000 },
                4 if rng.gen_bool() => 0xFFC, // cross-page 8-byte LDRD/STRD (`0xFFC` & `0x1000`)
                2 if rng.gen_bool() => 0xFFE,
                _ => rng.pick(&[0xFFDu32, 0xFFE, 0xFFF, 0x1000, 0x1001, 0x1002]),
            };
            HEAP + page * (PAGE_SIZE as u32) + off
        }
        _ => {
            // Slow-path UNMAPPED_SCRATCH (including its `0xFFD..=0x1002` boundary!)
            if rng.gen_bool() {
                let off = match align {
                    8 => if rng.gen_bool() { 0xFF8 } else { 0x1000 },
                    4 if rng.gen_bool() => 0xFFC,
                    2 if rng.gen_bool() => 0xFFE,
                    _ => rng.pick(&[0xFFDu32, 0xFFE, 0xFFF, 0x1000, 0x1001, 0x1002]),
                };
                UNMAPPED_SCRATCH + off
            } else {
                UNMAPPED_SCRATCH + 0x20 + rng.gen_range(PAGE_SIZE as u32 * 2 - 0x80)
            }
        }
    };
    raw & !(align - 1)
}

fn shift_imm_eval(kind: u32, val: u32, imm5: u32, carry_in: bool) -> u32 {
    match kind {
        0 => val.wrapping_shl(imm5),
        1 => if imm5 == 0 { 0 } else { val.wrapping_shr(imm5) },
        2 => {
            let s = if imm5 == 0 { 31 } else { imm5 };
            ((val as i32) >> s) as u32
        }
        _ => {
            if imm5 == 0 {
                ((carry_in as u32) << 31) | (val >> 1)
            } else {
                val.rotate_right(imm5)
            }
        }
    }
}

fn gen_arm_thumb_memory(rng: &mut SplitMix64) -> Slot {
    let mut init_regs = rand_regs_13(rng);
    let init_nzcvq = rng.gen_range(32) as u8;
    let carry_in = (init_nzcvq & 4) != 0;

    if rng.gen_range(4) == 0 {
        // Thumb Formats 7-10
        let rd = rng.gen_range(8);
        let rb = rng.gen_range(8);
        let mut ro = rng.gen_range(7);
        if ro >= rb {
            ro += 1;
        }
        let eff_addr = pick_target_addr(rng, 1);
        let (hw, desc) = match rng.gen_range(4) {
            0 => {
                // Format 7: reg-offset STR/STRB/LDR/LDRB
                let op2 = rng.gen_range(4);
                let ro_val = rng.pick(&[0u32, 1, 2, 3, 4, 8, 12, 0xFFFF_FFFC]);
                init_regs[ro as usize] = ro_val;
                init_regs[rb as usize] = eff_addr.wrapping_sub(ro_val);
                (0x5000u32 | (op2 << 10) | (ro << 6) | (rb << 3) | rd, "t_ls_reg")
            }
            1 => {
                // Format 8: reg-offset STRH/LDRSB/LDRH/LDRSH
                let op2 = rng.gen_range(4);
                let ro_val = rng.pick(&[0u32, 1, 2, 3, 4, 8, 0xFFFF_FFFE]);
                init_regs[ro as usize] = ro_val;
                init_regs[rb as usize] = eff_addr.wrapping_sub(ro_val);
                (0x5200u32 | (op2 << 10) | (ro << 6) | (rb << 3) | rd, "t_lsh_reg")
            }
            2 => {
                // Format 9: imm5 STR/LDR/STRB/LDRB
                let byte = rng.gen_bool();
                let load = rng.gen_bool();
                let imm5 = rng.gen_range(32);
                let delta = imm5 * if byte { 1 } else { 4 };
                init_regs[rb as usize] = eff_addr.wrapping_sub(delta);
                (
                    0x6000u32 | ((byte as u32) << 12) | ((load as u32) << 11) | (imm5 << 6) | (rb << 3) | rd,
                    "t_ls_imm",
                )
            }
            _ => {
                // Format 10: imm5 STRH/LDRH
                let load = rng.gen_bool();
                let imm5 = rng.gen_range(32);
                let delta = imm5 * 2;
                init_regs[rb as usize] = eff_addr.wrapping_sub(delta);
                (0x8000u32 | ((load as u32) << 11) | (imm5 << 6) | (rb << 3) | rd, "t_lsh_imm")
            }
        };
        return Slot {
            opcodes: vec![hw],
            thumb: true,
            desc,
            init_regs,
            init_nzcvq,
            init_ge: rng.gen_range(16) as u8,
            init_excl: None,
            init_fpscr: 0,
            init_vfp: [0; 32],
            vfp_mode: VfpCheckMode::Standard,
        };
    }

    // ARM memory instructions
    let cond = rand_arm_cond(rng);
    let mut init_excl = None;

    let (word, desc) = match rng.gen_range(10) {
        0..=4 => {
            // LDR / STR / LDRB / STRB (imm12 or shifted reg, offset/pre-wb/post-wb)
            let load = rng.gen_bool();
            let byte = rng.gen_bool();
            let pre = rng.gen_bool();
            let up = rng.gen_bool();
            let wb = if pre { rng.gen_bool() } else { false };
            let writeback = !pre || wb;
            let reg_off = rng.gen_bool();

            let rd = rng.gen_range(13);
            let rn = if writeback {
                let mut r = rng.gen_range(12);
                if r >= rd {
                    r += 1;
                }
                r
            } else {
                rng.gen_range(13)
            };

            let eff_addr = pick_target_addr(rng, 1);

            if !reg_off {
                let imm12 = if rng.gen_bool() {
                    rng.pick(&[0u32, 1, 2, 3, 4, 8, 12, 16, 64, 255, 0xFFF])
                } else {
                    rng.gen_range(0x1000)
                };
                if pre {
                    init_regs[rn as usize] = if up {
                        eff_addr.wrapping_sub(imm12)
                    } else {
                        eff_addr.wrapping_add(imm12)
                    };
                } else {
                    init_regs[rn as usize] = eff_addr;
                }
                (
                    (cond << 28)
                        | (0b010 << 25)
                        | ((pre as u32) << 24)
                        | ((up as u32) << 23)
                        | ((byte as u32) << 22)
                        | ((wb as u32) << 21)
                        | ((load as u32) << 20)
                        | (rn << 16)
                        | (rd << 12)
                        | imm12,
                    "arm_ls_imm",
                )
            } else {
                // Keep rm != rn (and rm != rd when load+wb? not required, rm != rn is enough)
                let mut rm = rng.gen_range(12);
                if rm >= rn {
                    rm += 1;
                }
                let kind = rng.gen_range(4);
                let imm5 = if rng.gen_bool() {
                    rng.pick(&[0u32, 1, 2, 3, 16, 31])
                } else {
                    rng.gen_range(32)
                };
                let rm_val = rng.pick(&[0u32, 1, 2, 3, 4, 8, 16, 0x8000_0001, 0xFFFF_FFFC]);
                init_regs[rm as usize] = rm_val;
                let offset = shift_imm_eval(kind, rm_val, imm5, carry_in);
                if pre {
                    init_regs[rn as usize] = if up {
                        eff_addr.wrapping_sub(offset)
                    } else {
                        eff_addr.wrapping_add(offset)
                    };
                } else {
                    init_regs[rn as usize] = eff_addr;
                }
                (
                    (cond << 28)
                        | (0b011 << 25)
                        | ((pre as u32) << 24)
                        | ((up as u32) << 23)
                        | ((byte as u32) << 22)
                        | ((wb as u32) << 21)
                        | ((load as u32) << 20)
                        | (rn << 16)
                        | (rd << 12)
                        | (imm5 << 7)
                        | (kind << 5)
                        | rm,
                    "arm_ls_reg",
                )
            }
        }
        5..=7 => {
            // Extra load/store: LDRH / STRH / LDRSB / LDRSH / LDRD / STRD
            let kind_pick = rng.gen_range(6);
            let (sh, load, is_pair) = match kind_pick {
                0 => (0b01u32, true, false),  // LDRH
                1 => (0b01, false, false),    // STRH
                2 => (0b10, true, false),     // LDRSB
                3 => (0b11, true, false),     // LDRSH
                4 => (0b10, false, true),     // LDRD
                _ => (0b11, false, true),     // STRD
            };
            let pre = rng.gen_bool();
            let up = rng.gen_bool();
            let wb = if pre { rng.gen_bool() } else { false };
            let writeback = !pre || wb;
            let imm_mode = rng.gen_bool();

            let (rd, rn, rm_opt) = if is_pair {
                // Rd even < 12; Rn and Rm disjoint from {Rd, Rd+1} and each other
                let rd = rng.gen_range(6) * 2;
                let mut pool: Vec<u32> = (0..13).filter(|&r| r != rd && r != rd + 1).collect();
                let rn_idx = rng.gen_range(pool.len() as u32) as usize;
                let rn = pool.swap_remove(rn_idx);
                let rm_opt = if !imm_mode {
                    let rm_idx = rng.gen_range(pool.len() as u32) as usize;
                    Some(pool[rm_idx])
                } else {
                    None
                };
                (rd, rn, rm_opt)
            } else {
                let rd = rng.gen_range(13);
                let rn = if writeback {
                    let mut r = rng.gen_range(12);
                    if r >= rd {
                        r += 1;
                    }
                    r
                } else {
                    rng.gen_range(13)
                };
                let rm_opt = if !imm_mode {
                    let mut rm = rng.gen_range(12);
                    if rm >= rn {
                        rm += 1;
                    }
                    Some(rm)
                } else {
                    None
                };
                (rd, rn, rm_opt)
            };

            let eff_addr = if is_pair {
                let pair_align = if rng.gen_bool() { 8 } else { 4 };
                pick_target_addr(rng, pair_align)
            } else {
                pick_target_addr(rng, 1)
            };

            if imm_mode {
                let imm8 = if rng.gen_bool() {
                    rng.pick(&[0u32, 1, 2, 3, 4, 8, 12, 16, 0xFF])
                } else {
                    rng.gen_range(256)
                };
                if pre {
                    init_regs[rn as usize] = if up {
                        eff_addr.wrapping_sub(imm8)
                    } else {
                        eff_addr.wrapping_add(imm8)
                    };
                } else {
                    init_regs[rn as usize] = eff_addr;
                }
                (
                    (cond << 28)
                        | ((pre as u32) << 24)
                        | ((up as u32) << 23)
                        | (1 << 22)
                        | ((wb as u32) << 21)
                        | ((load as u32) << 20)
                        | (rn << 16)
                        | (rd << 12)
                        | (((imm8 >> 4) & 0xF) << 8)
                        | (sh << 5)
                        | 0x90
                        | (imm8 & 0xF),
                    if is_pair { "arm_ldrd_strd_imm" } else { "arm_extra_imm" },
                )
            } else {
                let rm = rm_opt.unwrap();
                let rm_val = rng.pick(&[0u32, 1, 2, 3, 4, 8, 16, 0xFFFF_FFF8]);
                init_regs[rm as usize] = rm_val;
                if pre {
                    init_regs[rn as usize] = if up {
                        eff_addr.wrapping_sub(rm_val)
                    } else {
                        eff_addr.wrapping_add(rm_val)
                    };
                } else {
                    init_regs[rn as usize] = eff_addr;
                }
                (
                    (cond << 28)
                        | ((pre as u32) << 24)
                        | ((up as u32) << 23)
                        | ((wb as u32) << 21)
                        | ((load as u32) << 20)
                        | (rn << 16)
                        | (rd << 12)
                        | (sh << 5)
                        | 0x90
                        | rm,
                    if is_pair { "arm_ldrd_strd_reg" } else { "arm_extra_reg" },
                )
            }
        }
        _ => {
            // LDREX / STREX (mostly word 0b00, plus B/H/D)
            let load = rng.gen_bool();
            let size = match rng.gen_range(6) {
                0..=2 => 0b00u32, // word
                3 => 0b10,        // byte
                4 => 0b11,        // halfword
                _ => 0b01,        // doubleword
            };
            let align = match size {
                0b00 => 4,
                0b01 => 8,
                0b10 => 1,
                _ => 2,
            };
            let eff_addr = pick_target_addr(rng, align);
            init_excl = match rng.gen_range(3) {
                0 => None,
                1 => Some(eff_addr),
                _ => Some(eff_addr ^ 4),
            };

            if load {
                let rd = if size == 0b01 { rng.gen_range(6) * 2 } else { rng.gen_range(13) };
                let rn = rng.gen_range(13);
                init_regs[rn as usize] = eff_addr;
                (
                    (cond << 28) | 0x0190_0F9F | (size << 21) | (rn << 16) | (rd << 12),
                    "arm_ldrex",
                )
            } else {
                // STREX: Rd != Rn and Rd != Rm (and for D: Rm even < 12, Rd != Rm+1)
                let rm = if size == 0b01 { rng.gen_range(6) * 2 } else { rng.gen_range(13) };
                let rn = rng.gen_range(13);
                let rd_pool: Vec<u32> = (0..13)
                    .filter(|&r| r != rn && r != rm && (size != 0b01 || r != rm + 1))
                    .collect();
                let rd = rd_pool[rng.gen_range(rd_pool.len() as u32) as usize];
                init_regs[rn as usize] = eff_addr;
                (
                    (cond << 28) | 0x0180_0F90 | (size << 21) | (rn << 16) | (rd << 12) | rm,
                    "arm_strex",
                )
            }
        }
    };

    Slot {
        opcodes: vec![word],
        thumb: false,
        desc,
        init_regs,
        init_nzcvq,
        init_ge: rng.gen_range(16) as u8,
        init_excl,
        init_fpscr: 0,
        init_vfp: [0; 32],
        vfp_mode: VfpCheckMode::Standard,
    }
}

// ---------------------------------------------------------------------------
// Suite 4: Thumb Formats 1-5 + Extends & Byte Reversals
// ---------------------------------------------------------------------------

fn gen_thumb_alu(rng: &mut SplitMix64) -> Slot {
    let mut init_regs = rand_regs_13(rng);

    let (hw, desc) = match rng.gen_range(6) {
        0 => {
            // Format 1: LSL / LSR / ASR imm5
            let ty = rng.gen_range(3); // 0=LSL, 1=LSR, 2=ASR
            let imm5 = if rng.gen_bool() {
                rng.pick(&[0u32, 1, 7, 8, 15, 16, 30, 31])
            } else {
                rng.gen_range(32)
            };
            let rm = rng.gen_range(8);
            let rd = rng.gen_range(8);
            ((ty << 11) | (imm5 << 6) | (rm << 3) | rd, "t_shift_imm")
        }
        1 => {
            // Format 2: ADD / SUB reg or imm3
            let imm_flag = rng.gen_bool() as u32;
            let sub = rng.gen_bool() as u32;
            let field = rng.gen_range(8);
            let rn = rng.gen_range(8);
            let rd = rng.gen_range(8);
            (0x1800 | (imm_flag << 10) | (sub << 9) | (field << 6) | (rn << 3) | rd, "t_add_sub3")
        }
        2 => {
            // Format 3: MOV / CMP / ADD / SUB imm8
            let op2 = rng.gen_range(4);
            let rd = rng.gen_range(8);
            let imm8 = if rng.gen_bool() { rng.pick(EDGE_BYTES) } else { rng.next_u32() & 0xFF };
            (0x2000 | (op2 << 11) | (rd << 8) | imm8, "t_imm8")
        }
        3 => {
            // Format 4: all 16 ALU ops
            let op4 = rng.gen_range(16);
            let rd = rng.gen_range(8);
            let rs = rng.gen_range(8);
            if matches!(op4, 2 | 3 | 4 | 7) && rs != rd {
                init_regs[rs as usize] = rand_shift_count(rng);
            }
            (0x4000 | (op4 << 6) | (rs << 3) | rd, "t_alu4")
        }
        4 => {
            // Format 5: ADD / CMP / MOV high registers (r0..r12, plus r15 read where valid)
            let op2 = rng.gen_range(3); // 0=ADD, 1=CMP, 2=MOV
            match op2 {
                0 => {
                    // ADD Rd, Rs: rd in 0..=12, rs in 0..=15 (including r15!), at least one >= 8
                    let rd = rng.gen_range(13);
                    let rs = if rd < 8 {
                        if rng.gen_range(5) == 0 {
                            15
                        } else {
                            8 + rng.gen_range(7) // 8..=14 (or 15 above)
                        }
                    } else if rng.gen_range(6) == 0 {
                        15
                    } else {
                        rng.gen_range(15)
                    };
                    (
                        0x4400 | (((rd >> 3) & 1) << 7) | (rs << 3) | (rd & 7),
                        "t_hi_add",
                    )
                }
                1 => {
                    // CMP Rn, Rm: rn, rm in 0..=14 (no R15), at least one >= 8
                    let rn = rng.gen_range(15);
                    let rm = if rn < 8 { 8 + rng.gen_range(7) } else { rng.gen_range(15) };
                    (
                        0x4500 | (((rn >> 3) & 1) << 7) | (rm << 3) | (rn & 7),
                        "t_hi_cmp",
                    )
                }
                _ => {
                    // MOV Rd, Rs: rd in 0..=12, rs in 0..=15 (including r15!)
                    let rd = rng.gen_range(13);
                    let rs = if rng.gen_range(6) == 0 { 15 } else { rng.gen_range(15) };
                    (
                        0x4600 | (((rd >> 3) & 1) << 7) | (rs << 3) | (rd & 7),
                        "t_hi_mov",
                    )
                }
            }
        }
        _ => {
            // SXTH / SXTB / UXTH / UXTB / REV / REV16 / REVSH
            let rd = rng.gen_range(8);
            let rm = rng.gen_range(8);
            match rng.gen_range(7) {
                0 => (0xB200 | (rm << 3) | rd, "t_sxth"),
                1 => (0xB240 | (rm << 3) | rd, "t_sxtb"),
                2 => (0xB280 | (rm << 3) | rd, "t_uxth"),
                3 => (0xB2C0 | (rm << 3) | rd, "t_uxtb"),
                4 => (0xBA00 | (rm << 3) | rd, "t_rev"),
                5 => (0xBA40 | (rm << 3) | rd, "t_rev16"),
                _ => (0xBAC0 | (rm << 3) | rd, "t_revsh"),
            }
        }
    };

    let opcodes = if rng.gen_bool() {
        vec![0x46C0, hw] // NOP prefix so `hw` also runs at `addr % 4 == 2`
    } else {
        vec![hw]
    };

    Slot {
        opcodes,
        thumb: true,
        desc,
        init_regs,
        init_nzcvq: rng.gen_range(32) as u8,
        init_ge: rng.gen_range(16) as u8,
        init_excl: None,
        init_fpscr: 0,
        init_vfp: [0; 32],
        vfp_mode: VfpCheckMode::Standard,
    }
}

// ---------------------------------------------------------------------------
// Suite 5: VFPv2 Scalar, Short-Vector, Conversions, Rounding Modes, & FZ
// ---------------------------------------------------------------------------

const fn vfp_split_s(s: u32) -> (u32, u32) {
    ((s >> 1) & 0xF, s & 1)
}

const fn vfp_split_d(d: u32) -> (u32, u32) {
    (d & 0xF, (d >> 4) & 1)
}

const fn vfp_dp(cond: u32, double: bool, opc1: u32, negate: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    let (vd, db) = if double { vfp_split_d(rd) } else { vfp_split_s(rd) };
    let (vn, nb) = if double { vfp_split_d(rn) } else { vfp_split_s(rn) };
    let (vm, mb) = if double { vfp_split_d(rm) } else { vfp_split_s(rm) };
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

const fn vfp_ext(cond: u32, double: bool, opc2: u32, top: bool, vd: u32, db: u32, vm: u32, mb: u32) -> u32 {
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

const fn vfp_unary(cond: u32, double: bool, opc2: u32, top: bool, rd: u32, rm: u32) -> u32 {
    let (vd, db) = if double { vfp_split_d(rd) } else { vfp_split_s(rd) };
    let (vm, mb) = if double { vfp_split_d(rm) } else { vfp_split_s(rm) };
    vfp_ext(cond, double, opc2, top, vd, db, vm, mb)
}

fn fill_vfp_pool(rng: &mut SplitMix64, double: bool) -> [u32; 32] {
    if double {
        let mut out = [0u32; 32];
        for d in 0..16 {
            let bits = rand_f64_bits(rng);
            out[2 * d] = bits as u32;
            out[2 * d + 1] = (bits >> 32) as u32;
        }
        out
    } else {
        rand_vfp_32(rng)
    }
}

fn gen_vfp_fpscr(rng: &mut SplitMix64, double: bool, allow_vector: bool) -> u32 {
    let nzcv = rng.gen_range(16) << 28;
    let fz = (rng.gen_bool() as u32) << 24;
    let rmode = rng.gen_range(4) << 22;
    let (stride, len) = if allow_vector && rng.gen_bool() {
        let stride2 = rng.gen_bool();
        let max_len = match (double, stride2) {
            (false, false) => 8, // len 0..=7
            (false, true) => 4,  // len 0..=3
            (true, false) => 4,  // len 0..=3
            (true, true) => 2,   // len 0..=1
        };
        let stride_bits = if stride2 { 0b11u32 } else { 0b00 };
        let len_bits = rng.gen_range(max_len);
        (stride_bits << 20, len_bits << 16)
    } else {
        (0, 0)
    };
    nzcv | fz | rmode | stride | len
}

fn gen_vfp(rng: &mut SplitMix64) -> Slot {
    let init_regs = rand_regs_13(rng);
    let cond = rand_arm_cond(rng);
    let double = rng.gen_bool();
    let max_reg = if double { 16 } else { 32 };
    let bank = if double { 4 } else { 8 };

    let mut init_vfp = fill_vfp_pool(rng, double);
    let mut init_fpscr = gen_vfp_fpscr(rng, double, false);
    let mut vfp_mode = if double { VfpCheckMode::DoubleFloat } else { VfpCheckMode::SingleFloat };
    let mut opcodes = Vec::with_capacity(2);
    let desc: &'static str;

    let pick_vfp_dst = |rng: &mut SplitMix64, vec_active: bool| -> u32 {
        if vec_active && rng.gen_range(5) != 0 {
            bank + rng.gen_range(max_reg - bank) // vector bank
        } else {
            rng.gen_range(max_reg)
        }
    };

    match rng.gen_range(10) {
        0..=3 => {
            // VADD, VSUB, VMUL, VNMUL, VDIV, VMLA, VMLS, VNMLA, VNMLS (scalar + short vectors)
            init_fpscr = gen_vfp_fpscr(rng, double, true);
            let vec_active = ((init_fpscr >> 16) & 7) != 0;
            let rd = pick_vfp_dst(rng, vec_active);
            let rn = rng.gen_range(max_reg);
            let rm = rng.gen_range(max_reg);
            let (opc1, negate, name) = match rng.gen_range(9) {
                0 => (0b0011, false, "vadd"),
                1 => (0b0011, true, "vsub"),
                2 => (0b0010, false, "vmul"),
                3 => (0b0010, true, "vnmul"),
                4 => (0b1000, false, "vdiv"),
                5 => (0b0000, false, "vmla"),
                6 => (0b0000, true, "vmls"),
                7 => (0b0001, false, "vnmls"),
                _ => (0b0001, true, "vnmla"),
            };
            opcodes.push(vfp_dp(cond, double, opc1, negate, rd, rn, rm));
            desc = name;
        }
        4 => {
            // VSQRT, VABS, VNEG, VMOV (reg) (scalar + short vectors)
            init_fpscr = gen_vfp_fpscr(rng, double, true);
            let vec_active = ((init_fpscr >> 16) & 7) != 0;
            let rd = pick_vfp_dst(rng, vec_active);
            let rm = rng.gen_range(max_reg);
            let (opc2, top, name) = match rng.gen_range(4) {
                0 => (0b0000, false, "vmov"),
                1 => (0b0000, true, "vabs"),
                2 => (0b0001, false, "vneg"),
                _ => (0b0001, true, "vsqrt"),
            };
            opcodes.push(vfp_unary(cond, double, opc2, top, rd, rm));
            desc = name;
        }
        5 => {
            // [VCMP/VCMPE, FMSTAT] (vs register or vs #0.0)
            let e_bit = rng.gen_bool();
            let vs_zero = rng.gen_bool();
            let rd = rng.gen_range(max_reg);
            let (vd, db) = if double { vfp_split_d(rd) } else { vfp_split_s(rd) };
            let vcmp_op = if vs_zero {
                vfp_ext(cond, double, 0b0101, e_bit, vd, db, 0, 0)
            } else {
                let rm = rng.gen_range(max_reg);
                let (vm, mb) = if double { vfp_split_d(rm) } else { vfp_split_s(rm) };
                vfp_ext(cond, double, 0b0100, e_bit, vd, db, vm, mb)
            };
            opcodes.push(vcmp_op);
            if rng.gen_range(4) != 0 {
                // Append FMSTAT (`VMRS APSR_nzcv, FPSCR`)
                opcodes.push((cond << 28) | 0x0EF1_FA10);
                desc = "vcmp_fmstat";
            } else {
                desc = "vcmp";
            }
        }
        6 => {
            // VCVT.F64.F32 / VCVT.F32.F64
            if double {
                // VCVT.F32.F64 sd, dm
                let sd = rng.gen_range(32);
                let dm = rng.gen_range(16);
                let (vd, db) = vfp_split_s(sd);
                let (vm, mb) = vfp_split_d(dm);
                opcodes.push(vfp_ext(cond, true, 0b0111, true, vd, db, vm, mb));
                vfp_mode = VfpCheckMode::SingleFloat;
                desc = "vcvt_f32_f64";
            } else {
                // VCVT.F64.F32 dd, sm
                let dd = rng.gen_range(16);
                let sm = rng.gen_range(32);
                let (vd, db) = vfp_split_d(dd);
                let (vm, mb) = vfp_split_s(sm);
                opcodes.push(vfp_ext(cond, false, 0b0111, true, vd, db, vm, mb));
                vfp_mode = VfpCheckMode::DoubleFloat;
                desc = "vcvt_f64_f32";
            }
        }
        7 | 8 => {
            // VCVT / VCVTR S32/U32 <-> F32/F64
            let to_int = rng.gen_bool();
            let signed = rng.gen_bool();
            if to_int {
                let round_zero = rng.gen_bool();
                let sd = rng.gen_range(32);
                let src = rng.gen_range(max_reg);
                let (vd, db) = vfp_split_s(sd);
                let (vm, mb) = if double { vfp_split_d(src) } else { vfp_split_s(src) };
                let opc2 = if signed { 0b1101 } else { 0b1100 };
                opcodes.push(vfp_ext(cond, double, opc2, round_zero, vd, db, vm, mb));
                vfp_mode = VfpCheckMode::IntDst(sd as usize);
                desc = if round_zero { "vcvt_to_int" } else { "vcvtr_to_int" };
            } else {
                let dst = rng.gen_range(max_reg);
                let sm = rng.gen_range(32);
                init_vfp[sm as usize] = rand_u32_biased(rng);
                let (vd, db) = if double { vfp_split_d(dst) } else { vfp_split_s(dst) };
                let (vm, mb) = vfp_split_s(sm);
                opcodes.push(vfp_ext(cond, double, 0b1000, signed, vd, db, vm, mb));
                desc = "vcvt_from_int";
            }
        }
        _ => {
            // VMOV GPR <-> VFP: FMRS, FMSR, FMRDL, FMRDH, FMDRR, FMRRD
            vfp_mode = VfpCheckMode::Standard;
            match rng.gen_range(6) {
                0 => {
                    // FMRS Rt, Sn
                    let rt = rng.gen_range(13);
                    let sn = rng.gen_range(32);
                    let (vn, nb) = vfp_split_s(sn);
                    opcodes.push((cond << 28) | 0x0E10_0A10 | (vn << 16) | (rt << 12) | (nb << 7));
                    desc = "fmrs";
                }
                1 => {
                    // FMSR Sn, Rt
                    let rt = rng.gen_range(13);
                    let sn = rng.gen_range(32);
                    let (vn, nb) = vfp_split_s(sn);
                    opcodes.push((cond << 28) | 0x0E00_0A10 | (vn << 16) | (rt << 12) | (nb << 7));
                    desc = "fmsr";
                }
                2 => {
                    // FMRDL Rt, Dn
                    let rt = rng.gen_range(13);
                    let dn = rng.gen_range(16);
                    opcodes.push((cond << 28) | 0x0E10_0B10 | (dn << 16) | (rt << 12));
                    desc = "fmrdl";
                }
                3 => {
                    // FMRDH Rt, Dn
                    let rt = rng.gen_range(13);
                    let dn = rng.gen_range(16);
                    opcodes.push((cond << 28) | 0x0E30_0B10 | (dn << 16) | (rt << 12));
                    desc = "fmrdh";
                }
                4 => {
                    // FMDRR Dm, Rt, Rt2
                    let rt = rng.gen_range(13);
                    let rt2 = rng.gen_range(13);
                    let dm = rng.gen_range(16);
                    let (vm, mb) = vfp_split_d(dm);
                    opcodes.push(
                        (cond << 28) | 0x0C40_0B10 | (rt2 << 16) | (rt << 12) | (mb << 5) | vm,
                    );
                    desc = "fmdrr";
                }
                _ => {
                    // FMRRD Rt, Rt2, Dm (`Rt != Rt2` architecturally required!)
                    let (rt, rt2) = pick_distinct_pair(rng);
                    let dm = rng.gen_range(16);
                    let (vm, mb) = vfp_split_d(dm);
                    opcodes.push(
                        (cond << 28) | 0x0C50_0B10 | (rt2 << 16) | (rt << 12) | (mb << 5) | vm,
                    );
                    desc = "fmrrd";
                }
            }
        }
    }

    Slot {
        opcodes,
        thumb: false,
        desc,
        init_regs,
        init_nzcvq: rng.gen_range(32) as u8,
        init_ge: rng.gen_range(16) as u8,
        init_excl: None,
        init_fpscr,
        init_vfp,
        vfp_mode,
    }
}

// ---------------------------------------------------------------------------
// Test Entry Points (5 Suites + 1M-Instruction Ignored Stress Test)
// ---------------------------------------------------------------------------

#[test]
fn fuzz_arm_alu_shifts_and_flags() {
    run_fuzz_suite(
        "fuzz_arm_alu_shifts_and_flags",
        0xA11C_0001_0000_0001,
        DEFAULT_BATCHES_PER_SUITE,
        &gen_arm_alu,
    );
}

#[test]
fn fuzz_arm_mul_dsp_saturating_and_parallel_simd() {
    run_fuzz_suite(
        "fuzz_arm_mul_dsp_saturating_and_parallel_simd",
        0xD59A_0002_0000_0002,
        DEFAULT_BATCHES_PER_SUITE,
        &gen_arm_mul_dsp_media,
    );
}

#[test]
fn fuzz_arm_and_thumb_memory_fast_crosspage_and_slow() {
    run_fuzz_suite(
        "fuzz_arm_and_thumb_memory_fast_crosspage_and_slow",
        0x7E30_0003_0000_0003,
        DEFAULT_BATCHES_PER_SUITE,
        &gen_arm_thumb_memory,
    );
}

#[test]
fn fuzz_thumb_alu_shifts_highreg_and_extends() {
    run_fuzz_suite(
        "fuzz_thumb_alu_shifts_highreg_and_extends",
        0x780B_0004_0000_0004,
        DEFAULT_BATCHES_PER_SUITE,
        &gen_thumb_alu,
    );
}

#[test]
fn fuzz_vfp_scalar_vector_conversions_and_rmodes() {
    run_fuzz_suite(
        "fuzz_vfp_scalar_vector_conversions_and_rmodes",
        0xF902_0005_0000_0005,
        DEFAULT_BATCHES_PER_SUITE,
        &gen_vfp,
    );
}

#[ignore]
#[test]
fn fuzz_million_instructions_stress() {
    let t0 = Instant::now();
    let suites: [(&str, u64, &(dyn Fn(&mut SplitMix64) -> Slot + Sync)); 5] = [
        ("stress_arm_alu", 0x1111_0000_0000_0001, &gen_arm_alu),
        ("stress_arm_mul_dsp_media", 0x2222_0000_0000_0002, &gen_arm_mul_dsp_media),
        ("stress_memory", 0x3333_0000_0000_0003, &gen_arm_thumb_memory),
        ("stress_thumb_alu", 0x4444_0000_0000_0004, &gen_thumb_alu),
        ("stress_vfp", 0x5555_0000_0000_0005, &gen_vfp),
    ];

    let mut grand_execs = 0usize;
    let mut grand_fallbacks = 0u64;
    let mut grand_interpreted = 0u64;
    let mut grand_fails = 0usize;
    for (name, seed, gen) in suites {
        let (execs, fb, interp, fails) =
            run_fuzz_suite_raw(name, seed, STRESS_BATCHES_PER_SUITE, &|rng| gen(rng));
        assert!(execs > 200_000, "{name}: expected >200,000 executions, got {execs}");
        grand_execs += execs;
        grand_fallbacks += fb;
        grand_interpreted += interp;
        grand_fails += fails;
    }

    println!(
        "fuzz_million_instructions_stress TOTAL: execs={grand_execs} fallbacks={grand_fallbacks} uncompiled={grand_interpreted} mismatches={grand_fails} elapsed={:.2?}",
        t0.elapsed()
    );
    assert!(grand_execs >= 1_000_000);
    assert_eq!(grand_interpreted, 0, "uncompiled entries fell back to interpreter");
    assert_eq!(grand_fails, 0, "{grand_fails} total differential mismatches across 1M+ stress run");
}
