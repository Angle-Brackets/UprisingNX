//! Runtime execution state, executable buffer management (`mmap` on Unix/QEMU,
//! `jitCreate` streaming callback on Horizon Switch), cache serialization, and
//! slow-path C-ABI helpers for `zakuro-aot`.

use std::cell::UnsafeCell;
use std::io::{Read, Write};
use std::mem::offset_of;

use recomp_abi::{CodeHash, Context, Piece, Run, NO_ORIGIN};

use crate::discover::{FuncOrigin, OriginData};

#[cfg(target_arch = "x86_64")]
pub type OverrideFn = unsafe extern "sysv64" fn(*mut AotExecContext, *const u8);
#[cfg(not(target_arch = "x86_64"))]
pub type OverrideFn = unsafe extern "C" fn(*mut AotExecContext, *const u8);

/// Opaque game-specific state block carried inside [`AotExecContext`] and [`LoadedAot`].
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub struct PluginState(pub [u64; 4]);

#[cfg(target_arch = "aarch64")]
pub const CACHE_MAGIC: &[u8; 8] = b"UNXAOTA2";
#[cfg(target_arch = "x86_64")]
pub const CACHE_MAGIC: &[u8; 8] = b"UNXAOTX2";
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const CACHE_MAGIC: &[u8; 8] = b"UNXAOT00";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    /// Fold PC-relative loads from immutable `.text` literal pools into immediate moves.
    pub fold_text_literals: bool,
    /// Emit `FPSCR.LEN != 0` short-vector checks on VFP arithmetic.
    pub emit_vfp_vectors: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self { fold_text_literals: true, emit_vfp_vectors: true }
    }
}

pub const ICACHE_SLOTS: usize = 8192;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ICacheSlot {
    pub key: u64,
    pub rx_body: u64,
}

impl Default for ICacheSlot {
    fn default() -> Self {
        Self { key: u64::MAX, rx_body: 0 }
    }
}

/// Runtime context pointed to by `x19` throughout AOT execution.
/// Field 0 (`offset 0`) is `recomp_abi::Context` so `x19` is also a valid `*mut Context`.
#[repr(C)]
pub struct AotExecContext {
    pub ctx: Context,                // 0..144
    pub scratch_fp: [u32; 4],        // 144..160
    pub icache_ptr: *mut ICacheSlot, // 160
    pub aot_enter_ptr: usize,        // 168
    pub rx_base: *const u8,          // 176
    pub loaded_ptr: *const LoadedAot,// 184
    pub fn_slow_read8: usize,        // 192
    pub fn_slow_read16: usize,       // 200
    pub fn_slow_read32: usize,       // 208
    pub fn_slow_write8: usize,       // 216
    pub fn_slow_write16: usize,      // 224
    pub fn_slow_write32: usize,      // 232
    pub fn_lookup_body: usize,       // 240
    pub fn_interpret: usize,         // 248
    pub fn_media_parallel: usize,    // 256
    pub fn_vfp_vector: usize,        // 264
    pub fn_vfp_to_int: usize,        // 272
    pub overrides: [usize; 8],       // 280..344
    pub plugin_state: PluginState,   // 344..376
    pub stale_ptr: *const u8,        // 376..384
}

pub mod offsets {
    pub const CTX_R0: u16 = 0;
    pub const CTX_N: u16 = 64;
    pub const CTX_Z: u16 = 65;
    pub const CTX_C: u16 = 66;
    pub const CTX_V: u16 = 67;
    pub const CTX_Q: u16 = 68;
    pub const CTX_THUMB: u16 = 69;
    pub const CTX_GE: u16 = 70;
    pub const CTX_EXCLUSIVE: u16 = 71;
    pub const CTX_BUDGET: u16 = 72;
    pub const CTX_EXIT: u16 = 76;
    pub const CTX_SVC: u16 = 80;
    pub const CTX_DEPTH: u16 = 84;
    pub const CTX_EXCL_ADDR: u16 = 88;
    pub const CTX_TLS: u16 = 92;
    pub const CTX_READ_PAGES: u16 = 96;
    pub const CTX_WRITE_PAGES: u16 = 104;
    pub const CTX_VFP: u16 = 112;
    pub const CTX_FPSCR: u16 = 120;
    pub const SCRATCH_FP: i16 = 144;
    pub const HOST_MXCSR: i16 = 148;
    pub const ICACHE_PTR: u16 = 160;
    pub const AOT_ENTER_PTR: u16 = 168;
    pub const RX_BASE: u16 = 176;
    pub const LOADED_PTR: u16 = 184;
    pub const FN_SLOW_READ8: u16 = 192;
    pub const FN_SLOW_READ16: u16 = 200;
    pub const FN_SLOW_READ32: u16 = 208;
    pub const FN_SLOW_WRITE8: u16 = 216;
    pub const FN_SLOW_WRITE16: u16 = 224;
    pub const FN_SLOW_WRITE32: u16 = 232;
    pub const FN_LOOKUP_BODY: u16 = 240;
    pub const FN_INTERPRET: u16 = 248;
    pub const FN_MEDIA_PARALLEL: u16 = 256;
    pub const FN_VFP_VECTOR: u16 = 264;
    pub const FN_VFP_TO_INT: u16 = 272;
    pub const OVERRIDES_BASE: u16 = 280;
    pub const STALE_PTR: u16 = 376;
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(offset_of!(AotExecContext, ctx) == 0);
    assert!(offset_of!(Context, r) == offsets::CTX_R0 as usize);
    assert!(offset_of!(Context, n) == offsets::CTX_N as usize);
    assert!(offset_of!(Context, z) == offsets::CTX_Z as usize);
    assert!(offset_of!(Context, c) == offsets::CTX_C as usize);
    assert!(offset_of!(Context, v) == offsets::CTX_V as usize);
    assert!(offset_of!(Context, q) == offsets::CTX_Q as usize);
    assert!(offset_of!(Context, thumb) == offsets::CTX_THUMB as usize);
    assert!(offset_of!(Context, ge) == offsets::CTX_GE as usize);
    assert!(offset_of!(Context, exclusive) == offsets::CTX_EXCLUSIVE as usize);
    assert!(offset_of!(Context, budget) == offsets::CTX_BUDGET as usize);
    assert!(offset_of!(Context, exit) == offsets::CTX_EXIT as usize);
    assert!(offset_of!(Context, svc) == offsets::CTX_SVC as usize);
    assert!(offset_of!(Context, depth) == offsets::CTX_DEPTH as usize);
    assert!(offset_of!(Context, exclusive_address) == offsets::CTX_EXCL_ADDR as usize);
    assert!(offset_of!(Context, tls) == offsets::CTX_TLS as usize);
    assert!(offset_of!(Context, read_pages) == offsets::CTX_READ_PAGES as usize);
    assert!(offset_of!(Context, write_pages) == offsets::CTX_WRITE_PAGES as usize);
    assert!(offset_of!(Context, vfp) == offsets::CTX_VFP as usize);
    assert!(offset_of!(Context, fpscr) == offsets::CTX_FPSCR as usize);
    assert!(offset_of!(AotExecContext, scratch_fp) == offsets::SCRATCH_FP as usize);
    assert!(offset_of!(AotExecContext, scratch_fp) + 4 == offsets::HOST_MXCSR as usize);
    assert!(offset_of!(AotExecContext, icache_ptr) == offsets::ICACHE_PTR as usize);
    assert!(offset_of!(AotExecContext, aot_enter_ptr) == offsets::AOT_ENTER_PTR as usize);
    assert!(offset_of!(AotExecContext, rx_base) == offsets::RX_BASE as usize);
    assert!(offset_of!(AotExecContext, loaded_ptr) == offsets::LOADED_PTR as usize);
    assert!(offset_of!(AotExecContext, fn_slow_read8) == offsets::FN_SLOW_READ8 as usize);
    assert!(offset_of!(AotExecContext, fn_slow_read16) == offsets::FN_SLOW_READ16 as usize);
    assert!(offset_of!(AotExecContext, fn_slow_read32) == offsets::FN_SLOW_READ32 as usize);
    assert!(offset_of!(AotExecContext, fn_slow_write8) == offsets::FN_SLOW_WRITE8 as usize);
    assert!(offset_of!(AotExecContext, fn_slow_write16) == offsets::FN_SLOW_WRITE16 as usize);
    assert!(offset_of!(AotExecContext, fn_slow_write32) == offsets::FN_SLOW_WRITE32 as usize);
    assert!(offset_of!(AotExecContext, fn_lookup_body) == offsets::FN_LOOKUP_BODY as usize);
    assert!(offset_of!(AotExecContext, fn_interpret) == offsets::FN_INTERPRET as usize);
    assert!(offset_of!(AotExecContext, fn_media_parallel) == offsets::FN_MEDIA_PARALLEL as usize);
    assert!(offset_of!(AotExecContext, fn_vfp_vector) == offsets::FN_VFP_VECTOR as usize);
    assert!(offset_of!(AotExecContext, fn_vfp_to_int) == offsets::FN_VFP_TO_INT as usize);
    assert!(offset_of!(AotExecContext, overrides) == offsets::OVERRIDES_BASE as usize);
    assert!(offset_of!(AotExecContext, stale_ptr) == offsets::STALE_PTR as usize);
};

// ---------------------------------------------------------------------------
// Fast inline memory helpers (for overrides & helpers)
// ---------------------------------------------------------------------------

#[inline(always)]
pub unsafe fn mem_read8(ctx: &mut Context, addr: u32) -> u8 {
    let page = unsafe { *ctx.read_pages.add((addr >> 12) as usize) };
    if !page.is_null() {
        return unsafe { *page.add((addr & 0xFFF) as usize) };
    }
    unsafe { ((*ctx.host).read8)(ctx, addr) }
}

#[inline(always)]
pub unsafe fn mem_read16(ctx: &mut Context, addr: u32) -> u16 {
    let page = unsafe { *ctx.read_pages.add((addr >> 12) as usize) };
    let off = (addr & 0xFFF) as usize;
    if !page.is_null() && off <= 0xFFE {
        return unsafe { (page.add(off) as *const u16).read_unaligned() }.to_le();
    }
    unsafe { ((*ctx.host).read16)(ctx, addr) }
}

#[inline(always)]
pub unsafe fn mem_read32(ctx: &mut Context, addr: u32) -> u32 {
    let page = unsafe { *ctx.read_pages.add((addr >> 12) as usize) };
    let off = (addr & 0xFFF) as usize;
    if !page.is_null() && off <= 0xFFC {
        return unsafe { (page.add(off) as *const u32).read_unaligned() }.to_le();
    }
    unsafe { ((*ctx.host).read32)(ctx, addr) }
}

#[inline(always)]
pub unsafe fn mem_write8(ctx: &mut Context, addr: u32, val: u8) {
    let page = unsafe { *ctx.write_pages.add((addr >> 12) as usize) };
    if !page.is_null() {
        unsafe { *page.add((addr & 0xFFF) as usize) = val };
    } else {
        unsafe { ((*ctx.host).write8)(ctx, addr, val) };
    }
}

#[inline(always)]
pub unsafe fn mem_write16(ctx: &mut Context, addr: u32, val: u16) {
    let page = unsafe { *ctx.write_pages.add((addr >> 12) as usize) };
    let off = (addr & 0xFFF) as usize;
    if !page.is_null() && off <= 0xFFE {
        unsafe { (page.add(off) as *mut u16).write_unaligned(val.to_le()) };
    } else {
        unsafe { ((*ctx.host).write16)(ctx, addr, val) };
    }
}

#[inline(always)]
pub unsafe fn mem_write32(ctx: &mut Context, addr: u32, val: u32) {
    let page = unsafe { *ctx.write_pages.add((addr >> 12) as usize) };
    let off = (addr & 0xFFF) as usize;
    if !page.is_null() && off <= 0xFFC {
        unsafe { (page.add(off) as *mut u32).write_unaligned(val.to_le()) };
    } else {
        unsafe { ((*ctx.host).write32)(ctx, addr, val) };
    }
}

// ---------------------------------------------------------------------------
// ABI-uniform slow-path & ISA helpers called from AOT machine code
// ---------------------------------------------------------------------------

macro_rules! aot_helper_fn {
    (unsafe fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        #[cfg(target_arch = "x86_64")]
        unsafe extern "sysv64" fn $name($($arg: $ty),*) $(-> $ret)? $body
        #[cfg(not(target_arch = "x86_64"))]
        unsafe extern "C" fn $name($($arg: $ty),*) $(-> $ret)? $body
    };
}

aot_helper_fn!(unsafe fn slow_read8(ec: *mut AotExecContext, addr: u32) -> u32 {
    unsafe { ((*(*ec).ctx.host).read8)(&mut (*ec).ctx, addr) as u32 }
});

aot_helper_fn!(unsafe fn slow_read16(ec: *mut AotExecContext, addr: u32) -> u32 {
    unsafe { ((*(*ec).ctx.host).read16)(&mut (*ec).ctx, addr) as u32 }
});

aot_helper_fn!(unsafe fn slow_read32(ec: *mut AotExecContext, addr: u32) -> u32 {
    unsafe { ((*(*ec).ctx.host).read32)(&mut (*ec).ctx, addr) }
});

aot_helper_fn!(unsafe fn slow_write8(ec: *mut AotExecContext, addr: u32, val: u32) {
    unsafe { ((*(*ec).ctx.host).write8)(&mut (*ec).ctx, addr, val as u8) };
});

aot_helper_fn!(unsafe fn slow_write16(ec: *mut AotExecContext, addr: u32, val: u32) {
    unsafe { ((*(*ec).ctx.host).write16)(&mut (*ec).ctx, addr, val as u16) };
});

aot_helper_fn!(unsafe fn slow_write32(ec: *mut AotExecContext, addr: u32, val: u32) {
    unsafe { ((*(*ec).ctx.host).write32)(&mut (*ec).ctx, addr, val) };
});

#[inline(always)]
pub fn icache_slot(address: u32) -> usize {
    (address.wrapping_mul(0x9E37_79B1) >> (32 - ICACHE_SLOTS.trailing_zeros())) as usize
}

aot_helper_fn!(unsafe fn lookup_body(ec: *mut AotExecContext, key: u32) -> *const u8 {
    let ec = unsafe { &mut *ec };
    let loaded = unsafe { &*ec.loaded_ptr };
    let ptr = loaded.lookup_uncached(key).unwrap_or(std::ptr::null());
    if !ec.icache_ptr.is_null() {
        let slot = icache_slot(key);
        unsafe {
            *ec.icache_ptr.add(slot) = ICacheSlot { key: key as u64, rx_body: ptr as u64 };
        }
    }
    ptr
});

aot_helper_fn!(unsafe fn do_interpret(ec: *mut AotExecContext, addr: u32, opcode: u32) {
    unsafe { ((*(*ec).ctx.host).interpret)(&mut (*ec).ctx, addr, opcode) };
});

fn saturate_signed(value: i64, bits: u32) -> i64 {
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    value.clamp(min, max)
}

fn saturate_unsigned(value: i64, bits: u32) -> i64 {
    let max = if bits >= 32 { 0xFFFF_FFFF } else { (1i64 << bits) - 1 };
    value.clamp(0, max)
}

fn media_lane(
    a: u32,
    b: u32,
    width: u32,
    sub: bool,
    is_signed: bool,
    saturating: bool,
    halving: bool,
) -> (u32, bool) {
    let mask = if width == 16 { 0xFFFFu32 } else { 0xFFu32 };
    let (wide, ge) = if is_signed {
        let (x, y) = if width == 16 {
            (a as u16 as i16 as i32, b as u16 as i16 as i32)
        } else {
            (a as u8 as i8 as i32, b as u8 as i8 as i32)
        };
        let w = if sub { x - y } else { x + y };
        (w, w >= 0)
    } else {
        let (x, y) = ((a & mask) as i32, (b & mask) as i32);
        let w = if sub { x - y } else { x + y };
        let g = if sub { w >= 0 } else { w > mask as i32 };
        (w, g)
    };
    let val = if halving {
        (wide >> 1) as i64
    } else if saturating {
        if is_signed {
            saturate_signed(wide as i64, width)
        } else {
            saturate_unsigned(wide as i64, width)
        }
    } else {
        wide as i64
    };
    ((val as u32) & mask, ge)
}

aot_helper_fn!(unsafe fn media_parallel_helper(
    ec: *mut AotExecContext,
    packed: u32,
    a: u32,
    b: u32,
) -> u32 {
    let op1 = (packed >> 3) & 0x1F;
    let op2 = packed & 7;
    let is_signed = (op1 & 4) == 0;
    let saturating = (op1 & 3) == 2;
    let halving = (op1 & 3) == 3;
    let mut result = 0u32;
    let mut ge = 0u8;
    if op2 <= 3 {
        let cross = op2 == 1 || op2 == 2;
        let b_lo = if cross { b >> 16 } else { b & 0xFFFF };
        let b_hi = if cross { b & 0xFFFF } else { b >> 16 };
        let (lo, g0) = media_lane(a & 0xFFFF, b_lo, 16, op2 == 1 || op2 == 3, is_signed, saturating, halving);
        if g0 {
            ge |= 0b0011;
        }
        let (hi, g1) = media_lane(a >> 16, b_hi, 16, op2 == 2 || op2 == 3, is_signed, saturating, halving);
        if g1 {
            ge |= 0b1100;
        }
        result = lo | (hi << 16);
    } else {
        for i in 0..4 {
            let (lane, g) = media_lane(
                a >> (8 * i),
                b >> (8 * i),
                8,
                op2 == 7,
                is_signed,
                saturating,
                halving,
            );
            result |= lane << (8 * i);
            if g {
                ge |= 1 << i;
            }
        }
    }
    if !saturating && !halving {
        unsafe { (*ec).ctx.ge = ge };
    }
    result
});

// Short-vector VFPv2 helper (`packed`: `wide << 20 | op << 16 | d << 10 | n << 5 | m`).
aot_helper_fn!(unsafe fn vfp_vector_helper(ec: *mut AotExecContext, packed: u32, _unused: u32) {
    let ctx = unsafe { &mut (*ec).ctx };
    let wide = ((packed >> 20) & 1) != 0;
    let op = (packed >> 16) & 0xF;
    let d = ((packed >> 10) & 31) as usize;
    let n = ((packed >> 5) & 31) as usize;
    let m = (packed & 31) as usize;

    let fpscr = unsafe { *ctx.fpscr };
    let length = (((fpscr >> 16) & 7) + 1) as usize;
    let stride = if ((fpscr >> 20) & 3) == 3 { 2 } else { 1 };
    let bank = if wide { 4 } else { 8 };
    let step = |reg: usize, i: usize| (reg & !(bank - 1)) | ((reg + i * stride) & (bank - 1));
    let fz = (fpscr & (1 << 24)) != 0;

    let flush_s = |v: f32| -> u32 {
        let b = v.to_bits();
        if fz && (b & 0x7F80_0000) == 0 { b & 0x8000_0000 } else { b }
    };
    let flush_d = |v: f64| -> u64 {
        let b = v.to_bits();
        if fz && (b & 0x7FF0_0000_0000_0000) == 0 { b & 0x8000_0000_0000_0000 } else { b }
    };
    let get_d = |vfp: *mut u32, idx: usize| -> f64 {
        let lo = unsafe { *vfp.add((idx * 2) & 31) } as u64;
        let hi = unsafe { *vfp.add((idx * 2 + 1) & 31) } as u64;
        f64::from_bits(lo | (hi << 32))
    };

    for i in 0..length {
        let dd = step(d, i);
        let nn = step(n, i);
        let mm = if m < bank { m } else { step(m, i) };
        if wide && (9..=11).contains(&op) {
            let lo = unsafe { *ctx.vfp.add(2 * (mm & 15)) };
            let hi = unsafe { *ctx.vfp.add(2 * (mm & 15) + 1) };
            let out_hi = match op {
                9 => hi,
                10 => hi & 0x7FFF_FFFF,
                _ => hi ^ 0x8000_0000,
            };
            unsafe {
                *ctx.vfp.add(2 * (dd & 15)) = lo;
                *ctx.vfp.add(2 * (dd & 15) + 1) = out_hi;
            }
        } else if wide {
            let raw_bits = |vfp: *mut u32, idx: usize| -> f64 { get_d(vfp, idx) };
            let read_d = |vfp: *mut u32, idx: usize| -> f64 { f64::from_bits(flush_d(raw_bits(vfp, idx))) };
            let a = read_d(ctx.vfp, nn & 15);
            let b = read_d(ctx.vfp, mm & 15);
            let acc = read_d(ctx.vfp, dd & 15);
            let prod = || f64::from_bits(flush_d(a * b));
            let res = match op {
                0 => acc + prod(),
                1 => acc - prod(),
                2 => prod() - acc,
                3 => (-prod()) - acc,
                4 => prod(),
                5 => -prod(),
                6 => a + b,
                7 => a - b,
                8 => a / b,
                _ => b.sqrt(),
            };
            let bits = flush_d(res);
            unsafe {
                *ctx.vfp.add(2 * (dd & 15)) = bits as u32;
                *ctx.vfp.add(2 * (dd & 15) + 1) = (bits >> 32) as u32;
            }
        } else if (9..=11).contains(&op) {
            let bits = unsafe { *ctx.vfp.add(mm) };
            let out = match op {
                9 => bits,
                10 => bits & 0x7FFF_FFFF,
                _ => bits ^ 0x8000_0000,
            };
            unsafe { *ctx.vfp.add(dd) = out };
        } else {
            let read_s = |idx: usize| -> f32 { f32::from_bits(flush_s(f32::from_bits(unsafe { *ctx.vfp.add(idx) }))) };
            let a = read_s(nn);
            let b = read_s(mm);
            let acc = read_s(dd);
            let prod = || f32::from_bits(flush_s(a * b));
            let res = match op {
                0 => acc + prod(),
                1 => acc - prod(),
                2 => prod() - acc,
                3 => (-prod()) - acc,
                4 => prod(),
                5 => -prod(),
                6 => a + b,
                7 => a - b,
                8 => a / b,
                _ => b.sqrt(),
            };
            unsafe { *ctx.vfp.add(dd) = flush_s(res) };
        }
    }
});

// Exact ARM VFPv2 float-to-integer conversion matching `zakuro-cpu/src/vfp.rs`
// (`NaN -> 0`, rounding mode from `FPSCR[23:22]` or `trunc()` when `round_zero`,
// saturating to `i32` / `u32`).
//
// `packed`: `(double << 12) | (signed << 11) | (round_zero << 10) | (sd << 5) | rm`
aot_helper_fn!(unsafe fn vfp_to_int_helper(ec: *mut AotExecContext, packed: u32, _unused: u32) {
    let ctx = unsafe { &mut (*ec).ctx };
    let double = ((packed >> 12) & 1) != 0;
    let signed = ((packed >> 11) & 1) != 0;
    let round_zero = ((packed >> 10) & 1) != 0;
    let sd = ((packed >> 5) & 31) as usize;
    let rm = (packed & 31) as usize;
    let fpscr = unsafe { *ctx.fpscr };
    let fz = (fpscr & (1 << 24)) != 0;

    let source = if double {
        let lo = unsafe { *ctx.vfp.add((rm & 15) * 2) } as u64;
        let hi = unsafe { *ctx.vfp.add((rm & 15) * 2 + 1) } as u64;
        let mut bits = lo | (hi << 32);
        if fz && (bits & 0x7FF0_0000_0000_0000) == 0 {
            bits &= 0x8000_0000_0000_0000;
        }
        f64::from_bits(bits)
    } else {
        let mut bits = unsafe { *ctx.vfp.add(rm) };
        if fz && (bits & 0x7F80_0000) == 0 {
            bits &= 0x8000_0000;
        }
        f32::from_bits(bits) as f64
    };

    let rounded = if round_zero {
        source.trunc()
    } else {
        match (fpscr >> 22) & 3 {
            0b00 => {
                let r = source.round();
                if (source - source.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
                    r - source.signum()
                } else {
                    r
                }
            }
            0b01 => source.ceil(),
            0b10 => source.floor(),
            _ => source.trunc(),
        }
    };

    let bits_out = if signed {
        let val = if rounded.is_nan() {
            0
        } else if rounded >= i32::MAX as f64 {
            i32::MAX
        } else if rounded <= i32::MIN as f64 {
            i32::MIN
        } else {
            rounded as i32
        };
        val as u32
    } else if rounded.is_nan() {
        0
    } else if rounded >= u32::MAX as f64 {
        u32::MAX
    } else if rounded <= 0.0 {
        0
    } else {
        rounded as u32
    };
    unsafe { *ctx.vfp.add(sd) = bits_out };
});

// ---------------------------------------------------------------------------
// Serializable AOT Image & Executable Loader
// ---------------------------------------------------------------------------

pub struct AotImage {
    pub program_id: u64,
    pub text_hash: u64,
    pub enter_offset: u32,
    pub code: Vec<u8>,
    /// Sorted by `address` (`guest_pc | thumb`). `offset` is byte offset in `code`.
    pub entries: Vec<(u32, u32)>,
    pub origins: OriginData,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheHeader {
    pub format_version: u32,
    pub program_id: u64,
    pub text_hash: u64,
    pub enter_offset: u32,
    pub code_bytes_len: u32,
    pub entry_count: u32,
    pub piece_count: u32,
    pub func_count: u32,
    pub run_count: u32,
}

impl AotImage {
    pub fn code_bytes(&self) -> &[u8] {
        &self.code
    }

    pub fn write_to(&self, mut w: impl Write) -> std::io::Result<()> {
        debug_assert_eq!(self.entries.len(), self.origins.owners.len());
        w.write_all(CACHE_MAGIC)?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&self.program_id.to_le_bytes())?;
        w.write_all(&self.text_hash.to_le_bytes())?;
        w.write_all(&self.enter_offset.to_le_bytes())?;
        let code_bytes_len = self.code.len() as u32;
        w.write_all(&code_bytes_len.to_le_bytes())?;
        w.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        w.write_all(&(self.origins.pieces.len() as u32).to_le_bytes())?;
        w.write_all(&(self.origins.functions.len() as u32).to_le_bytes())?;
        w.write_all(&(self.origins.runs.len() as u32).to_le_bytes())?;

        for &(addr, off) in &self.entries {
            w.write_all(&addr.to_le_bytes())?;
            w.write_all(&off.to_le_bytes())?;
        }
        for &owner in &self.origins.owners {
            w.write_all(&owner.to_le_bytes())?;
        }
        for p in &self.origins.pieces {
            w.write_all(&p.start.to_le_bytes())?;
            w.write_all(&p.end.to_le_bytes())?;
            w.write_all(&p.hash.to_le_bytes())?;
        }
        for f in &self.origins.functions {
            w.write_all(&f.first.to_le_bytes())?;
            w.write_all(&f.count.to_le_bytes())?;
            w.write_all(&f.start.to_le_bytes())?;
        }
        for r in &self.origins.runs {
            w.write_all(&r.first.to_le_bytes())?;
            w.write_all(&r.end.to_le_bytes())?;
        }
        w.write_all(self.code_bytes())?;
        w.flush()
    }

    pub fn read_header(mut r: impl Read) -> Result<CacheHeader, String> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != CACHE_MAGIC {
            return Err("invalid AOT cache magic".to_owned());
        }
        let u32_le = |r: &mut dyn Read| -> Result<u32, String> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(u32::from_le_bytes(b))
        };
        let u64_le = |r: &mut dyn Read| -> Result<u64, String> {
            let mut b = [0u8; 8];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(u64::from_le_bytes(b))
        };
        let format_version = u32_le(&mut r)?;
        if format_version != FORMAT_VERSION {
            return Err(format!("AOT cache version {format_version} != current {FORMAT_VERSION}"));
        }
        Ok(CacheHeader {
            format_version,
            program_id: u64_le(&mut r)?,
            text_hash: u64_le(&mut r)?,
            enter_offset: u32_le(&mut r)?,
            code_bytes_len: u32_le(&mut r)?,
            entry_count: u32_le(&mut r)?,
            piece_count: u32_le(&mut r)?,
            func_count: u32_le(&mut r)?,
            run_count: u32_le(&mut r)?,
        })
    }

    #[cfg(not(target_os = "horizon"))]
    pub fn into_loaded_exec(self) -> Result<LoadedAot, String> {
        let exec = ExecBuffer::exec_copy(self.code_bytes())?;
        Ok(LoadedAot::new(
            self.program_id,
            self.text_hash,
            self.enter_offset,
            exec,
            self.entries,
            self.origins,
        ))
    }

    #[cfg(all(unix, not(target_os = "horizon")))]
    pub fn into_loaded_mmap(self) -> Result<LoadedAot, String> {
        self.into_loaded_exec()
    }
}

pub struct ExecBuffer {
    rx_ptr: *const u8,
    len: usize,
    drop_fn: Option<Box<dyn FnOnce() + Send>>,
}

unsafe impl Send for ExecBuffer {}
unsafe impl Sync for ExecBuffer {}

impl ExecBuffer {
    /// Wraps externally-allocated executable code memory (such as libnx `jitCreate`).
    ///
    /// # Safety
    /// `rx_ptr` must point to `len` readable and executable bytes valid until `drop_fn` runs.
    pub unsafe fn from_external(
        rx_ptr: *const u8,
        len: usize,
        drop_fn: Option<Box<dyn FnOnce() + Send>>,
    ) -> Self {
        Self { rx_ptr, len, drop_fn }
    }

    #[cfg(not(target_os = "horizon"))]
    pub fn exec_copy(bytes: &[u8]) -> Result<Self, String> {
        Self::exec_stream(bytes.len(), |dst| {
            dst.copy_from_slice(bytes);
            Ok(())
        })
    }

    #[cfg(not(target_os = "horizon"))]
    pub fn exec_stream(
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        #[cfg(unix)]
        {
            Self::mmap_stream(len, fill)
        }
        #[cfg(windows)]
        {
            Self::win_alloc_stream(len, fill)
        }
    }

    #[cfg(all(unix, not(target_os = "horizon")))]
    pub fn mmap_copy(bytes: &[u8]) -> Result<Self, String> {
        Self::mmap_stream(bytes.len(), |dst| {
            dst.copy_from_slice(bytes);
            Ok(())
        })
    }

    #[cfg(all(unix, not(target_os = "horizon")))]
    pub fn mmap_stream(
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        let alloc_len = len.max(4096).next_multiple_of(4096);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                alloc_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap failed for AOT executable buffer".to_owned());
        }
        let rw = ptr as *mut u8;
        if let Err(e) = fill(unsafe { std::slice::from_raw_parts_mut(rw, len) }) {
            unsafe { libc::munmap(ptr, alloc_len) };
            return Err(e);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            clear_icache(rw, len);
        }
        let rc = unsafe { libc::mprotect(ptr, alloc_len, libc::PROT_READ | libc::PROT_EXEC) };
        if rc != 0 {
            unsafe { libc::munmap(ptr, alloc_len) };
            return Err("mprotect(PROT_READ | PROT_EXEC) failed".to_owned());
        }
        let addr = ptr as usize;
        Ok(Self {
            rx_ptr: rw as *const u8,
            len,
            drop_fn: Some(Box::new(move || unsafe {
                libc::munmap(addr as *mut libc::c_void, alloc_len);
            })),
        })
    }

    #[cfg(windows)]
    pub fn win_alloc_stream(
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        extern "system" {
            fn VirtualAlloc(
                addr: *mut core::ffi::c_void,
                size: usize,
                fl_allocation_type: u32,
                fl_protect: u32,
            ) -> *mut core::ffi::c_void;
            fn VirtualProtect(
                addr: *mut core::ffi::c_void,
                size: usize,
                fl_new_protect: u32,
                lpfl_old_protect: *mut u32,
            ) -> i32;
            fn VirtualFree(
                addr: *mut core::ffi::c_void,
                size: usize,
                dw_free_type: u32,
            ) -> i32;
        }
        const MEM_COMMIT_RESERVE: u32 = 0x3000;
        const MEM_RELEASE: u32 = 0x8000;
        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_EXECUTE_READ: u32 = 0x20;

        let alloc_len = len.max(4096).next_multiple_of(4096);
        let ptr = unsafe { VirtualAlloc(core::ptr::null_mut(), alloc_len, MEM_COMMIT_RESERVE, PAGE_READWRITE) };
        if ptr.is_null() {
            return Err("VirtualAlloc failed for AOT executable buffer".to_owned());
        }
        let rw = ptr as *mut u8;
        if let Err(e) = fill(unsafe { core::slice::from_raw_parts_mut(rw, len) }) {
            unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
            return Err(e);
        }
        let mut old_protect = 0u32;
        if unsafe { VirtualProtect(ptr, alloc_len, PAGE_EXECUTE_READ, &mut old_protect) } == 0 {
            unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
            return Err("VirtualProtect(PAGE_EXECUTE_READ) failed".to_owned());
        }
        let addr = ptr as usize;
        Ok(Self {
            rx_ptr: rw as *const u8,
            len,
            drop_fn: Some(Box::new(move || unsafe {
                VirtualFree(addr as *mut core::ffi::c_void, 0, MEM_RELEASE);
            })),
        })
    }

    #[inline(always)]
    pub fn rx_ptr(&self) -> *const u8 {
        self.rx_ptr
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ExecBuffer {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f();
        }
    }
}

#[cfg(all(unix, not(target_os = "horizon"), target_arch = "aarch64"))]
unsafe fn clear_icache(ptr: *const u8, len: usize) {
    let mut addr = (ptr as usize) & !63;
    let end = (ptr as usize) + len;
    while addr < end {
        unsafe { core::arch::asm!("dc cvau, {}", in(reg) addr, options(nostack, preserves_flags)) };
        addr += 64;
    }
    unsafe { core::arch::asm!("dsb ish", options(nostack, preserves_flags)) };
    let mut addr = (ptr as usize) & !63;
    while addr < end {
        unsafe { core::arch::asm!("ic ivau, {}", in(reg) addr, options(nostack, preserves_flags)) };
        addr += 64;
    }
    unsafe { core::arch::asm!("dsb ish\n\tisb", options(nostack, preserves_flags)) };
}

pub struct LoadedAot {
    pub program_id: u64,
    pub text_hash: u64,
    enter_offset: u32,
    exec: ExecBuffer,
    entries: Vec<(u32, u32)>,
    origins: OriginData,
    stale: Vec<u8>,
    any_stale: bool,
    icache: UnsafeCell<Box<[ICacheSlot]>>,
    overrides: [usize; 8],
    plugin_state: UnsafeCell<PluginState>,
}

// SAFETY: `LoadedAot` is accessed exclusively on the emulated CPU thread during `enter()`.
unsafe impl Send for LoadedAot {}

impl LoadedAot {
    pub fn new(
        program_id: u64,
        text_hash: u64,
        enter_offset: u32,
        exec: ExecBuffer,
        entries: Vec<(u32, u32)>,
        origins: OriginData,
    ) -> Self {
        let stale = vec![0u8; origins.functions.len().max(1)];
        let icache = (0..ICACHE_SLOTS).map(|_| ICacheSlot::default()).collect();
        Self {
            program_id,
            text_hash,
            enter_offset,
            exec,
            entries,
            origins,
            stale,
            any_stale: false,
            icache: UnsafeCell::new(icache),
            overrides: [0usize; 8],
            plugin_state: UnsafeCell::new(PluginState::default()),
        }
    }

    pub fn set_overrides(&mut self, overrides: &[(u32, OverrideFn)]) {
        self.overrides = [0usize; 8];
        for (i, &(_, func)) in overrides.iter().take(8).enumerate() {
            self.overrides[i] = func as usize;
        }
    }

    pub fn has_active_overrides(&self) -> bool {
        self.overrides.iter().any(|&fn_ptr| fn_ptr != 0)
    }

    pub fn with_overrides(mut self, overrides: &[(u32, OverrideFn)]) -> Self {
        self.set_overrides(overrides);
        self
    }

    pub fn with_plugin_state(mut self, state: PluginState) -> Self {
        *self.plugin_state.get_mut() = state;
        self
    }

    /// Streams a cached `UNXAOT01` file directly into a caller-provided executable allocator
    /// without allocating a duplicate `Vec<u8>` for the machine code. Reads metadata in a
    /// single bulk I/O call and streams the machine code directly into the executable buffer.
    pub fn read_streaming<R: Read>(
        mut r: R,
        alloc_exec: impl FnOnce(usize, &mut dyn FnMut(&mut [u8]) -> Result<(), String>) -> Result<ExecBuffer, String>,
    ) -> Result<Self, String> {
        let hdr = AotImage::read_header(&mut r)?;
        let n_entries = hdr.entry_count as usize;
        let n_pieces = hdr.piece_count as usize;
        let n_funcs = hdr.func_count as usize;
        let n_runs = hdr.run_count as usize;

        let meta_bytes = n_entries
            .checked_mul(12)
            .and_then(|v| v.checked_add(n_pieces.checked_mul(16)?))
            .and_then(|v| v.checked_add(n_funcs.checked_mul(12)?))
            .and_then(|v| v.checked_add(n_runs.checked_mul(8)?))
            .ok_or_else(|| "AOT header metadata size overflow".to_owned())?;

        let mut meta = vec![0u8; meta_bytes];
        r.read_exact(&mut meta).map_err(|e| e.to_string())?;
        let mut cursor = 0usize;
        let next_u32 = |buf: &[u8], cur: &mut usize| -> u32 {
            let v = u32::from_le_bytes([buf[*cur], buf[*cur + 1], buf[*cur + 2], buf[*cur + 3]]);
            *cur += 4;
            v
        };
        let next_u64 = |buf: &[u8], cur: &mut usize| -> u64 {
            let v = u64::from_le_bytes([
                buf[*cur],
                buf[*cur + 1],
                buf[*cur + 2],
                buf[*cur + 3],
                buf[*cur + 4],
                buf[*cur + 5],
                buf[*cur + 6],
                buf[*cur + 7],
            ]);
            *cur += 8;
            v
        };

        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let a = next_u32(&meta, &mut cursor);
            let b = next_u32(&meta, &mut cursor);
            entries.push((a, b));
        }
        let mut owners = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            owners.push(next_u32(&meta, &mut cursor));
        }
        let mut pieces = Vec::with_capacity(n_pieces);
        for _ in 0..n_pieces {
            let start = next_u32(&meta, &mut cursor);
            let end = next_u32(&meta, &mut cursor);
            let hash = next_u64(&meta, &mut cursor);
            pieces.push(Piece { start, end, hash });
        }
        let mut functions = Vec::with_capacity(n_funcs);
        for _ in 0..n_funcs {
            let first = next_u32(&meta, &mut cursor);
            let count = next_u32(&meta, &mut cursor);
            let start = next_u32(&meta, &mut cursor);
            functions.push(FuncOrigin { first, count, start });
        }
        let mut runs = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let first = next_u32(&meta, &mut cursor);
            let end = next_u32(&meta, &mut cursor);
            runs.push(Run { first, end });
        }
        drop(meta);

        let code_len = hdr.code_bytes_len as usize;
        if code_len == 0 || (hdr.enter_offset as usize) >= code_len {
            return Err(format!(
                "corrupt AOT image: invalid enter_offset {} for code length {}",
                hdr.enter_offset, code_len
            ));
        }
        if entries.windows(2).any(|w| w[0].0 >= w[1].0) {
            return Err("corrupt AOT image: entry table is not strictly sorted".to_owned());
        }
        if let Some(&(addr, off)) = entries.iter().find(|&&(_, off)| (off as usize) >= code_len) {
            return Err(format!(
                "corrupt AOT image: entry 0x{addr:08X} offset {off} out of bounds ({code_len})"
            ));
        }
        if owners.iter().any(|&owner| owner != NO_ORIGIN && (owner as usize) >= n_funcs) {
            return Err("corrupt AOT image: owner index out of bounds".to_owned());
        }
        if let (Some(first), Some(last)) = (pieces.first(), pieces.last()) {
            if pieces.iter().any(|p| p.start > p.end)
                || pieces.windows(2).any(|w| w[0].end > w[1].start)
            {
                return Err("corrupt AOT image: origin piece ranges invalid or unsorted".to_owned());
            }
            let span = last.end.checked_sub(first.start).ok_or_else(|| {
                "corrupt AOT image: origin span underflow".to_owned()
            })?;
            if span > 128 * 1024 * 1024 {
                return Err(format!("corrupt AOT image: origin span {span} exceeds 128 MiB"));
            }
        }
        if functions
            .iter()
            .any(|f| f.first.checked_add(f.count).is_none_or(|end| end as usize > n_runs))
        {
            return Err("corrupt AOT image: function run range out of bounds".to_owned());
        }
        if runs.iter().any(|r| r.first > r.end || (r.end as usize) > n_pieces) {
            return Err("corrupt AOT image: origin run piece index out of bounds".to_owned());
        }

        let exec = alloc_exec(code_len, &mut |dst| {
            r.read_exact(dst).map_err(|e| e.to_string())
        })?;

        Ok(Self::new(
            hdr.program_id,
            hdr.text_hash,
            hdr.enter_offset,
            exec,
            entries,
            OriginData { pieces, functions, runs, owners },
        ))
    }

    #[cfg(not(target_os = "horizon"))]
    pub fn read_exec<R: Read>(r: R) -> Result<Self, String> {
        Self::read_streaming(r, |len, fill| ExecBuffer::exec_stream(len, fill))
    }

    #[cfg(all(unix, not(target_os = "horizon")))]
    pub fn read_mmap<R: Read>(r: R) -> Result<Self, String> {
        Self::read_exec(r)
    }

    pub fn describe(&self) -> String {
        let arch = if cfg!(target_arch = "x86_64") { "x86_64" } else { "AArch64" };
        format!(
            "{} entry points, {:.1} MiB native {arch} AOT",
            self.entries.len(),
            self.exec.len() as f64 / (1024.0 * 1024.0),
        )
    }

    pub fn code_size_bytes(&self) -> usize {
        self.exec.len()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn clear_icache(&self) {
        let cache = unsafe { &mut *self.icache.get() };
        cache.fill(ICacheSlot::default());
    }

    #[inline]
    pub fn lookup_uncached(&self, key: u32) -> Option<*const u8> {
        let idx = self.entries.binary_search_by_key(&key, |&(a, _)| a).ok()?;
        let owner = *self.origins.owners.get(idx)?;
        if owner != NO_ORIGIN && self.stale.get(owner as usize).copied().unwrap_or(0) != 0 {
            return None;
        }
        let byte_off = self.entries[idx].1 as usize;
        Some(unsafe { self.exec.rx_ptr().add(byte_off) })
    }

    #[inline]
    pub fn lookup(&self, key: u32) -> Option<*const u8> {
        let slot = icache_slot(key);
        let cache = unsafe { &mut *self.icache.get() };
        let entry = cache[slot];
        if entry.key == key as u64 {
            return (entry.rx_body != 0).then_some(entry.rx_body as *const u8);
        }
        let found = self.lookup_uncached(key);
        cache[slot] = ICacheSlot {
            key: key as u64,
            rx_body: found.map_or(0, |p| p as u64),
        };
        found
    }

    pub fn has_code(&self, key: u32) -> bool {
        self.lookup(key).is_some()
    }

    /// Checks guest memory against `Origins` pieces and marks modified functions stale.
    pub fn check(&mut self, mut read: impl FnMut(u32, &mut [u8])) -> usize {
        let pieces = &self.origins.pieces;
        let (Some(first), Some(last)) = (pieces.first(), pieces.last()) else {
            return 0;
        };
        let mut code = vec![0u8; (last.end - first.start) as usize];
        read(first.start, &mut code);
        let mut changed_before = Vec::with_capacity(pieces.len() + 1);
        changed_before.push(0u32);
        for piece in pieces {
            let mut hash = CodeHash::default();
            hash.add(&code[(piece.start - first.start) as usize..(piece.end - first.start) as usize]);
            let changed = (hash.value() != piece.hash) as u32;
            changed_before.push(changed_before.last().unwrap() + changed);
        }
        let runs = &self.origins.runs;
        let mut stale_count = 0;
        for (i, func) in self.origins.functions.iter().enumerate() {
            let f_runs = &runs[func.first as usize..(func.first + func.count) as usize];
            let changed = f_runs.iter().any(|r| changed_before[r.end as usize] > changed_before[r.first as usize]);
            self.stale[i] = changed as u8;
            stale_count += changed as usize;
        }
        self.any_stale = stale_count > 0;
        self.clear_icache();
        stale_count
    }

    /// Enters compiled AOT code starting at `target_body_rx`, mutating `ctx` in place.
    ///
    /// # Safety
    /// Must only be called on `aarch64` or `x86_64` with a valid `Context` and `target_body_rx`
    /// returned by `self.lookup(...)`.
    pub unsafe fn enter(&self, ctx: &mut Context, target_body_rx: *const u8) {
        let enter_ptr = unsafe { self.exec.rx_ptr().add(self.enter_offset as usize) };
        let icache_ptr = unsafe { (*self.icache.get()).as_mut_ptr() };
        let plugin_state = unsafe { *self.plugin_state.get() };

        let mut ec = AotExecContext {
            ctx: Context {
                r: ctx.r,
                n: ctx.n,
                z: ctx.z,
                c: ctx.c,
                v: ctx.v,
                q: ctx.q,
                thumb: ctx.thumb,
                ge: ctx.ge,
                exclusive: ctx.exclusive,
                budget: ctx.budget,
                exit: ctx.exit,
                svc: ctx.svc,
                depth: ctx.depth,
                exclusive_address: ctx.exclusive_address,
                tls: ctx.tls,
                read_pages: ctx.read_pages,
                write_pages: ctx.write_pages,
                vfp: ctx.vfp,
                fpscr: ctx.fpscr,
                host: ctx.host,
                user: ctx.user,
            },
            scratch_fp: [0; 4],
            icache_ptr,
            aot_enter_ptr: enter_ptr as usize,
            rx_base: self.exec.rx_ptr(),
            loaded_ptr: self as *const LoadedAot,
            fn_slow_read8: slow_read8 as *const () as usize,
            fn_slow_read16: slow_read16 as *const () as usize,
            fn_slow_read32: slow_read32 as *const () as usize,
            fn_slow_write8: slow_write8 as *const () as usize,
            fn_slow_write16: slow_write16 as *const () as usize,
            fn_slow_write32: slow_write32 as *const () as usize,
            fn_lookup_body: lookup_body as *const () as usize,
            fn_interpret: do_interpret as *const () as usize,
            fn_media_parallel: media_parallel_helper as *const () as usize,
            fn_vfp_vector: vfp_vector_helper as *const () as usize,
            fn_vfp_to_int: vfp_to_int_helper as *const () as usize,
            overrides: self.overrides,
            plugin_state,
            stale_ptr: if self.any_stale { self.stale.as_ptr() } else { std::ptr::null() },
        };

        let enter_fn: OverrideFn = unsafe { std::mem::transmute(enter_ptr) };
        unsafe { enter_fn(&mut ec, target_body_rx) };

        unsafe { *self.plugin_state.get() = ec.plugin_state };
        *ctx = ec.ctx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_streaming_roundtrips_bulk_metadata() {
        let image = AotImage {
            program_id: 0x0004_0000_0003_0100,
            text_hash: 0xD6E0_28C6_175A_2B07,
            enter_offset: 0x40,
            entries: vec![(0x0010_0000, 0x100), (0x0010_0010, 0x180)],
            origins: OriginData {
                pieces: vec![Piece { start: 0x0010_0000, end: 0x0010_0020, hash: 0x1234_5678_9ABC_DEF0 }],
                functions: vec![FuncOrigin { first: 0, count: 1, start: 0x0010_0000 }],
                runs: vec![Run { first: 0, end: 1 }],
                owners: vec![0, 0],
            },
            code: {
                let mut code = vec![0x90u8; 0x190];
                let head: Vec<u8> = [0xD503_201Fu32, 0xD65F_03C0u32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect();
                code[..head.len()].copy_from_slice(&head);
                code
            },
        };
        let mut buf = Vec::new();
        image.write_to(&mut buf).unwrap();

        let mut captured = Vec::new();
        let loaded = LoadedAot::read_streaming(Cursor::new(&buf), |len, fill| {
            let mut mem = vec![0u8; len].into_boxed_slice();
            fill(&mut mem)?;
            captured.extend_from_slice(&mem);
            let ptr = mem.as_ptr();
            Ok(unsafe {
                ExecBuffer::from_external(
                    ptr,
                    len,
                    Some(Box::new(move || {
                        drop(mem);
                    })),
                )
            })
        })
        .unwrap();

        assert_eq!(loaded.program_id, image.program_id);
        assert_eq!(loaded.text_hash, image.text_hash);
        assert_eq!(loaded.enter_offset, image.enter_offset);
        assert_eq!(loaded.entries, image.entries);
        assert_eq!(loaded.origins.owners, image.origins.owners);
        assert_eq!(loaded.origins.pieces.len(), 1);
        assert_eq!(loaded.origins.pieces[0].hash, 0x1234_5678_9ABC_DEF0);
        assert_eq!(captured, image.code_bytes());
    }
}
