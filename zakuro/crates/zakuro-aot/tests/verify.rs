//! Differential verification of `zakuro-aot` compiled AArch64 machine code
//! against `zakuro-cpu` interpreter across synthetic ISA torture tests and
//! thousands of real Kid Icarus: Uprising functions.

#![cfg(all(unix, target_arch = "aarch64"))]

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

fn sync_host_fpcr(guest_fpscr: u32) -> u64 {
    let saved: u64;
    unsafe { core::arch::asm!("mrs {}, fpcr", out(reg) saved, options(nomem, nostack, preserves_flags)) };
    let wanted = (saved & !0x03C0_0000) | ((guest_fpscr as u64) & 0x03C0_0000);
    if wanted != saved {
        unsafe { core::arch::asm!("msr fpcr, {}", in(reg) wanted, options(nomem, nostack, preserves_flags)) };
    }
    saved
}

fn restore_host_fpcr(saved: u64) {
    unsafe { core::arch::asm!("msr fpcr, {}", in(reg) saved, options(nomem, nostack, preserves_flags)) };
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

fn run_snippet_test(name: &str, words: &[u32], seeds: &[(u32, Source)], init: impl FnOnce(&mut Cpu, &mut TestMemory)) {
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
    let aot = LoadedAot::read_mmap(buf.as_slice()).unwrap();

    let mut mem = TestMemory::new(vec![
        Region { base: BASE, bytes: text_bytes, writable: false, direct_page: true },
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
        let got = unsafe { run_aot(&aot, &mut b, &mut ctx, slice, LIMIT) };
        let diffs = compare_runs(&a, &b, &ctx);
        assert_eq!(exp, got, "{name} (slice={slice}): stop mismatch");
        assert!(diffs.is_empty(), "{name} (slice={slice}): {diffs:?}");
        assert_eq!(b.fallbacks, 0, "{name}: unexpected interpreter fallback!");
    }
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
    // Must fit comfortably in one single Horizon CodeMemory (`<= 128 MiB`)
    assert!(code_mib < 96.0, "expected compact AOT code < 96 MiB, got {code_mib:.2} MiB");

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
    let aot = pure_image.into_loaded_mmap().expect("mmap ROM AOT");

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
