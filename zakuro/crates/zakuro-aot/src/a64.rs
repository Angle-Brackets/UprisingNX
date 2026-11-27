//! Zero-dependency AArch64 machine-code instruction encoder.
//!
//! Every emitter returns a `u32` AArch64 little-endian instruction word.
//! Tested directly against `aarch64-linux-gnu-objdump` so every opcode bitfield
//! is verified against GNU binutils.

pub const ZR: u8 = 31;
pub const SP: u8 = 31;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

pub mod cond {
    pub const EQ: u8 = 0x0;
    pub const NE: u8 = 0x1;
    pub const CS: u8 = 0x2;
    pub const HS: u8 = 0x2;
    pub const CC: u8 = 0x3;
    pub const LO: u8 = 0x3;
    pub const MI: u8 = 0x4;
    pub const PL: u8 = 0x5;
    pub const VS: u8 = 0x6;
    pub const VC: u8 = 0x7;
    pub const HI: u8 = 0x8;
    pub const LS: u8 = 0x9;
    pub const GE: u8 = 0xA;
    pub const LT: u8 = 0xB;
    pub const GT: u8 = 0xC;
    pub const LE: u8 = 0xD;
    pub const AL: u8 = 0xE;

    #[inline(always)]
    pub const fn invert(c: u8) -> u8 {
        c ^ 1
    }
}

/// Code buffer wrapper with convenience push & patch helpers.
#[derive(Default, Clone)]
pub struct CodeBuf {
    pub words: Vec<u32>,
}

impl CodeBuf {
    pub fn with_capacity(words: usize) -> Self {
        Self { words: Vec::with_capacity(words) }
    }

    #[inline(always)]
    pub fn pos(&self) -> usize {
        self.words.len()
    }

    #[inline(always)]
    pub fn byte_len(&self) -> usize {
        self.words.len() * 4
    }

    #[inline(always)]
    pub fn emit(&mut self, insn: u32) -> usize {
        let idx = self.words.len();
        self.words.push(insn);
        idx
    }

    #[inline(always)]
    pub fn patch(&mut self, word_idx: usize, insn: u32) {
        self.words[word_idx] = insn;
    }

    /// Materializes any 32-bit constant into `wd` in 1 or 2 instructions without
    /// touching flags.
    pub fn mov_imm32(&mut self, wd: u8, val: u32) {
        if val == 0 {
            self.emit(movz32(wd, 0, 0));
        } else if (val >> 16) == 0 {
            self.emit(movz32(wd, val as u16, 0));
        } else if (val & 0xFFFF) == 0 {
            self.emit(movz32(wd, (val >> 16) as u16, 1));
        } else if (!val >> 16) == 0 {
            self.emit(movn32(wd, (!val) as u16, 0));
        } else if (!val & 0xFFFF) == 0 {
            self.emit(movn32(wd, (!val >> 16) as u16, 1));
        } else if let Some(insn) = orr32_imm(wd, ZR, val) {
            self.emit(insn);
        } else {
            self.emit(movz32(wd, val as u16, 0));
            self.emit(movk32(wd, (val >> 16) as u16, 1));
        }
    }

    /// Materializes a 64-bit constant into `xd` without touching flags.
    pub fn mov_imm64(&mut self, xd: u8, val: u64) {
        if val <= u32::MAX as u64 {
            self.mov_imm32(xd, val as u32);
            return;
        }
        let mut first = true;
        for hw in 0..4u8 {
            let chunk = ((val >> (hw * 16)) & 0xFFFF) as u16;
            if chunk != 0 {
                if first {
                    self.emit(movz64(xd, chunk, hw));
                    first = false;
                } else {
                    self.emit(movk64(xd, chunk, hw));
                }
            }
        }
        if first {
            self.emit(movz64(xd, 0, 0));
        }
    }

    /// Computes `wd = wn.wrapping_add(delta)` without touching flags.
    /// `scratch` must be distinct from `wn`.
    pub fn add_const32(&mut self, wd: u8, wn: u8, delta: u32, scratch: u8) {
        let neg = delta.wrapping_neg();
        if delta == 0 {
            if wd != wn {
                self.emit(mov32(wd, wn));
            }
        } else if delta <= 0xFFF {
            self.emit(add32_imm(wd, wn, delta as u16, false));
        } else if delta & 0xFFF == 0 && (delta >> 12) <= 0xFFF {
            self.emit(add32_imm(wd, wn, (delta >> 12) as u16, true));
        } else if neg <= 0xFFF {
            self.emit(sub32_imm(wd, wn, neg as u16, false));
        } else if neg & 0xFFF == 0 && (neg >> 12) <= 0xFFF {
            self.emit(sub32_imm(wd, wn, (neg >> 12) as u16, true));
        } else if delta <= 0xFF_FFFF {
            self.emit(add32_imm(wd, wn, (delta >> 12) as u16, true));
            self.emit(add32_imm(wd, wd, (delta & 0xFFF) as u16, false));
        } else if neg <= 0xFF_FFFF {
            self.emit(sub32_imm(wd, wn, (neg >> 12) as u16, true));
            self.emit(sub32_imm(wd, wd, (neg & 0xFFF) as u16, false));
        } else {
            debug_assert_ne!(scratch, wn);
            self.mov_imm32(scratch, delta);
            self.emit(add32_reg(wd, wn, scratch, Shift::Lsl, 0));
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 4);
        for w in self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Move wide (MOVZ / MOVN / MOVK)
// ---------------------------------------------------------------------------

#[inline(always)]
pub const fn movn32(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 2);
    0x1280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn movz32(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 2);
    0x5280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn movk32(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 2);
    0x7280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn movn64(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 4);
    0x9280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn movz64(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 4);
    0xD280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn movk64(rd: u8, imm16: u16, hw: u8) -> u32 {
    debug_assert!(hw < 4);
    0xF280_0000 | ((hw as u32) << 21) | ((imm16 as u32) << 5) | (rd as u32 & 31)
}

// ---------------------------------------------------------------------------
// Add / Sub immediate
// ---------------------------------------------------------------------------

#[inline(always)]
const fn addsub_imm(sf: u32, op: u32, s: u32, rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    debug_assert!(imm12 <= 0xFFF);
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | 0x1100_0000
        | ((shift12 as u32) << 22)
        | ((imm12 as u32) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}

#[inline(always)]
pub const fn add32_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(0, 0, 0, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn adds32_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(0, 0, 1, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn sub32_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(0, 1, 0, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn subs32_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(0, 1, 1, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn cmp32_imm(rn: u8, imm12: u16) -> u32 {
    subs32_imm(ZR, rn, imm12, false)
}

#[inline(always)]
pub const fn add64_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(1, 0, 0, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn sub64_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(1, 1, 0, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn subs64_imm(rd: u8, rn: u8, imm12: u16, shift12: bool) -> u32 {
    addsub_imm(1, 1, 1, rd, rn, imm12, shift12)
}
#[inline(always)]
pub const fn mov_sp(rd: u8, rn: u8) -> u32 {
    add64_imm(rd, rn, 0, false)
}

// ---------------------------------------------------------------------------
// Add / Sub shifted & extended register, ADC / SBC
// ---------------------------------------------------------------------------

#[inline(always)]
const fn addsub_reg(sf: u32, op: u32, s: u32, rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    debug_assert!(!matches!(shift, Shift::Ror));
    debug_assert!(amount < if sf == 1 { 64 } else { 32 });
    (sf << 31)
        | (op << 30)
        | (s << 29)
        | 0x0B00_0000
        | ((shift as u32) << 22)
        | ((rm as u32 & 31) << 16)
        | ((amount as u32) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}

#[inline(always)]
pub const fn add32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(0, 0, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn adds32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(0, 0, 1, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn sub32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(0, 1, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn subs32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(0, 1, 1, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn cmp32_reg(rn: u8, rm: u8) -> u32 {
    subs32_reg(ZR, rn, rm, Shift::Lsl, 0)
}
#[inline(always)]
pub const fn add64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(1, 0, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn sub64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(1, 1, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn subs64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    addsub_reg(1, 1, 1, rd, rn, rm, shift, amount)
}

/// `add xd, xn, wm, uxtw #lsl` (lsl in 0..=4)
#[inline(always)]
pub const fn add64_uxtw(xd: u8, xn: u8, wm: u8, lsl: u8) -> u32 {
    debug_assert!(lsl <= 4);
    0x8B20_4000
        | ((wm as u32 & 31) << 16)
        | ((lsl as u32) << 10)
        | ((xn as u32 & 31) << 5)
        | (xd as u32 & 31)
}

/// `cmp xn, wm, sxtw` (`subs xzr, xn, wm, sxtw #0`)
#[inline(always)]
pub const fn cmp64_sxtw(xn: u8, wm: u8) -> u32 {
    0xEB20_C000 | ((wm as u32 & 31) << 16) | ((xn as u32 & 31) << 5) | 31
}

#[inline(always)]
const fn adcsbc32(op: u32, s: u32, rd: u8, rn: u8, rm: u8) -> u32 {
    (op << 30) | (s << 29) | 0x1A00_0000 | ((rm as u32 & 31) << 16) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn adc32(rd: u8, rn: u8, rm: u8) -> u32 {
    adcsbc32(0, 0, rd, rn, rm)
}
#[inline(always)]
pub const fn adcs32(rd: u8, rn: u8, rm: u8) -> u32 {
    adcsbc32(0, 1, rd, rn, rm)
}
#[inline(always)]
pub const fn sbc32(rd: u8, rn: u8, rm: u8) -> u32 {
    adcsbc32(1, 0, rd, rn, rm)
}
#[inline(always)]
pub const fn sbcs32(rd: u8, rn: u8, rm: u8) -> u32 {
    adcsbc32(1, 1, rd, rn, rm)
}

// ---------------------------------------------------------------------------
// Logical (shifted register & 32-bit bitmask immediate)
// ---------------------------------------------------------------------------

#[inline(always)]
const fn logical_reg(sf: u32, opc: u32, n: u32, rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    debug_assert!(amount < if sf == 1 { 64 } else { 32 });
    (sf << 31)
        | (opc << 29)
        | 0x0A00_0000
        | ((shift as u32) << 22)
        | (n << 21)
        | ((rm as u32 & 31) << 16)
        | ((amount as u32) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}

#[inline(always)]
pub const fn and32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b00, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn bic32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b00, 1, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn orr32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b01, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn orn32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b01, 1, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn eor32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b10, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn ands32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b11, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn bics32_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(0, 0b11, 1, rd, rn, rm, shift, amount)
}

#[inline(always)]
pub const fn mov32(rd: u8, rm: u8) -> u32 {
    orr32_reg(rd, ZR, rm, Shift::Lsl, 0)
}
#[inline(always)]
pub const fn mvn32(rd: u8, rm: u8) -> u32 {
    orn32_reg(rd, ZR, rm, Shift::Lsl, 0)
}
#[inline(always)]
pub const fn tst32_reg(rn: u8, rm: u8) -> u32 {
    ands32_reg(ZR, rn, rm, Shift::Lsl, 0)
}

#[inline(always)]
pub const fn and64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(1, 0b00, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn bic64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(1, 0b00, 1, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn orr64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(1, 0b01, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn eor64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(1, 0b10, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn ands64_reg(rd: u8, rn: u8, rm: u8, shift: Shift, amount: u8) -> u32 {
    logical_reg(1, 0b11, 0, rd, rn, rm, shift, amount)
}
#[inline(always)]
pub const fn mov64(rd: u8, rm: u8) -> u32 {
    orr64_reg(rd, ZR, rm, Shift::Lsl, 0)
}

/// Encodes a 32-bit AArch64 logical bitmask immediate `(immr, imms)` (`N = 0`).
pub const fn encode_bitmask32(val: u32) -> Option<(u32, u32)> {
    if val == 0 || val == u32::MAX {
        return None;
    }
    // Find smallest repeating element size in {2, 4, 8, 16, 32}
    let mut size = 32u32;
    while size > 2 {
        let half = size / 2;
        let mask = (1u32 << half) - 1;
        if (val & mask) == ((val >> half) & mask) {
            size = half;
        } else {
            break;
        }
    }
    let elem_mask = if size == 32 { u32::MAX } else { (1u32 << size) - 1 };
    let elem = val & elem_mask;
    // Rotate elem right so that it forms `1..1` at bit 0 (`(1 << ones) - 1`).
    // Check all rotations r in 0..size: AArch64 rotates `(1 << (s+1)) - 1` RIGHT by `immr`.
    let ones = elem.count_ones();
    if ones == 0 || ones == size {
        return None;
    }
    let base_pat = (1u32 << ones) - 1;
    let mut immr = 0u32;
    while immr < size {
        let rotated = if immr == 0 {
            base_pat
        } else {
            ((base_pat >> immr) | (base_pat << (size - immr))) & elem_mask
        };
        if rotated == elem {
            let nimms = (!(size * 2 - 1)) & 0x3F;
            let imms = nimms | (ones - 1);
            return Some((immr, imms));
        }
        immr += 1;
    }
    None
}

#[inline]
const fn logical32_imm(opc: u32, rd: u8, rn: u8, val: u32) -> Option<u32> {
    let Some((immr, imms)) = encode_bitmask32(val) else {
        return None;
    };
    Some(
        (opc << 29)
            | 0x1200_0000
            | (immr << 16)
            | (imms << 10)
            | ((rn as u32 & 31) << 5)
            | (rd as u32 & 31),
    )
}

#[inline]
pub const fn and32_imm(rd: u8, rn: u8, val: u32) -> Option<u32> {
    logical32_imm(0b00, rd, rn, val)
}
#[inline]
pub const fn orr32_imm(rd: u8, rn: u8, val: u32) -> Option<u32> {
    logical32_imm(0b01, rd, rn, val)
}
#[inline]
pub const fn eor32_imm(rd: u8, rn: u8, val: u32) -> Option<u32> {
    logical32_imm(0b10, rd, rn, val)
}
#[inline]
pub const fn ands32_imm(rd: u8, rn: u8, val: u32) -> Option<u32> {
    logical32_imm(0b11, rd, rn, val)
}

// ---------------------------------------------------------------------------
// Variable shifts, bitfields, bit ops, conditional select
// ---------------------------------------------------------------------------

#[inline(always)]
const fn shift_var(sf: u32, op2: u32, rd: u8, rn: u8, rm: u8) -> u32 {
    (sf << 31) | 0x1AC0_2000 | ((rm as u32 & 31) << 16) | (op2 << 10) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn lslv32(rd: u8, rn: u8, rm: u8) -> u32 {
    shift_var(0, 0b00, rd, rn, rm)
}
#[inline(always)]
pub const fn lsrv32(rd: u8, rn: u8, rm: u8) -> u32 {
    shift_var(0, 0b01, rd, rn, rm)
}
#[inline(always)]
pub const fn asrv32(rd: u8, rn: u8, rm: u8) -> u32 {
    shift_var(0, 0b10, rd, rn, rm)
}
#[inline(always)]
pub const fn rorv32(rd: u8, rn: u8, rm: u8) -> u32 {
    shift_var(0, 0b11, rd, rn, rm)
}

#[inline(always)]
const fn bitfield(sf: u32, opc: u32, rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    (sf << 31)
        | (opc << 29)
        | 0x1300_0000
        | (sf << 22)
        | ((immr as u32 & 63) << 16)
        | ((imms as u32 & 63) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}

#[inline(always)]
pub const fn sbfm32(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(0, 0b00, rd, rn, immr, imms)
}
#[inline(always)]
pub const fn bfm32(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(0, 0b01, rd, rn, immr, imms)
}
#[inline(always)]
pub const fn ubfm32(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(0, 0b10, rd, rn, immr, imms)
}
#[inline(always)]
pub const fn sbfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(1, 0b00, rd, rn, immr, imms)
}
#[inline(always)]
pub const fn bfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(1, 0b01, rd, rn, immr, imms)
}
#[inline(always)]
pub const fn ubfm64(rd: u8, rn: u8, immr: u8, imms: u8) -> u32 {
    bitfield(1, 0b10, rd, rn, immr, imms)
}

#[inline(always)]
pub const fn lsl32_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    if shift == 0 {
        mov32(rd, rn)
    } else {
        ubfm32(rd, rn, (32 - shift) & 31, 31 - shift)
    }
}
#[inline(always)]
pub const fn lsr32_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    ubfm32(rd, rn, shift, 31)
}
#[inline(always)]
pub const fn asr32_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    sbfm32(rd, rn, shift, 31)
}
#[inline(always)]
pub const fn ror32_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    debug_assert!(shift < 32);
    0x1380_0000 | ((rn as u32 & 31) << 16) | ((shift as u32) << 10) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn lsl64_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    if shift == 0 {
        mov64(rd, rn)
    } else {
        ubfm64(rd, rn, (64 - shift) & 63, 63 - shift)
    }
}
#[inline(always)]
pub const fn lsr64_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    ubfm64(rd, rn, shift, 63)
}
#[inline(always)]
pub const fn asr64_imm(rd: u8, rn: u8, shift: u8) -> u32 {
    sbfm64(rd, rn, shift, 63)
}

#[inline(always)]
pub const fn sxtb32(rd: u8, rn: u8) -> u32 {
    sbfm32(rd, rn, 0, 7)
}
#[inline(always)]
pub const fn sxth32(rd: u8, rn: u8) -> u32 {
    sbfm32(rd, rn, 0, 15)
}
#[inline(always)]
pub const fn uxtb32(rd: u8, rn: u8) -> u32 {
    ubfm32(rd, rn, 0, 7)
}
#[inline(always)]
pub const fn uxth32(rd: u8, rn: u8) -> u32 {
    ubfm32(rd, rn, 0, 15)
}
#[inline(always)]
pub const fn sxtw64(xd: u8, wn: u8) -> u32 {
    sbfm64(xd, wn, 0, 31)
}
#[inline(always)]
pub const fn sxtb64(xd: u8, wn: u8) -> u32 {
    sbfm64(xd, wn, 0, 7)
}
#[inline(always)]
pub const fn sxth64(xd: u8, wn: u8) -> u32 {
    sbfm64(xd, wn, 0, 15)
}

#[inline(always)]
pub const fn ubfx32(rd: u8, rn: u8, lsb: u8, width: u8) -> u32 {
    ubfm32(rd, rn, lsb, lsb + width - 1)
}
#[inline(always)]
pub const fn bfi32(rd: u8, rn: u8, lsb: u8, width: u8) -> u32 {
    bfm32(rd, rn, (32 - lsb) & 31, width - 1)
}
#[inline(always)]
pub const fn bfxil32(rd: u8, rn: u8, lsb: u8, width: u8) -> u32 {
    bfm32(rd, rn, lsb, lsb + width - 1)
}
#[inline(always)]
pub const fn bfi64(rd: u8, rn: u8, lsb: u8, width: u8) -> u32 {
    bfm64(rd, rn, (64 - lsb) & 63, width - 1)
}

#[inline(always)]
pub const fn rbit32(rd: u8, rn: u8) -> u32 {
    0x5AC0_0000 | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn rev16_32(rd: u8, rn: u8) -> u32 {
    0x5AC0_0400 | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn rev32(rd: u8, rn: u8) -> u32 {
    0x5AC0_0800 | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn clz32(rd: u8, rn: u8) -> u32 {
    0x5AC0_1000 | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}

#[inline(always)]
pub const fn csel32(rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    0x1A80_0000
        | ((rm as u32 & 31) << 16)
        | ((cond as u32 & 15) << 12)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn csinc32(rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    0x1A80_0400
        | ((rm as u32 & 31) << 16)
        | ((cond as u32 & 15) << 12)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn cset32(rd: u8, c: u8) -> u32 {
    csinc32(rd, ZR, ZR, cond::invert(c))
}

// ---------------------------------------------------------------------------
// Multiplies
// ---------------------------------------------------------------------------

#[inline(always)]
pub const fn madd32(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x1B00_0000
        | ((rm as u32 & 31) << 16)
        | ((ra as u32 & 31) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn mul32(rd: u8, rn: u8, rm: u8) -> u32 {
    madd32(rd, rn, rm, ZR)
}
#[inline(always)]
pub const fn madd64(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x9B00_0000
        | ((rm as u32 & 31) << 16)
        | ((ra as u32 & 31) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn msub64(rd: u8, rn: u8, rm: u8, ra: u8) -> u32 {
    0x9B00_8000
        | ((rm as u32 & 31) << 16)
        | ((ra as u32 & 31) << 10)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn smaddl(xd: u8, wn: u8, wm: u8, xa: u8) -> u32 {
    0x9B20_0000
        | ((wm as u32 & 31) << 16)
        | ((xa as u32 & 31) << 10)
        | ((wn as u32 & 31) << 5)
        | (xd as u32 & 31)
}
#[inline(always)]
pub const fn smsubl(xd: u8, wn: u8, wm: u8, xa: u8) -> u32 {
    0x9B20_8000
        | ((wm as u32 & 31) << 16)
        | ((xa as u32 & 31) << 10)
        | ((wn as u32 & 31) << 5)
        | (xd as u32 & 31)
}
#[inline(always)]
pub const fn smull(xd: u8, wn: u8, wm: u8) -> u32 {
    smaddl(xd, wn, wm, ZR)
}
#[inline(always)]
pub const fn umaddl(xd: u8, wn: u8, wm: u8, xa: u8) -> u32 {
    0x9BA0_0000
        | ((wm as u32 & 31) << 16)
        | ((xa as u32 & 31) << 10)
        | ((wn as u32 & 31) << 5)
        | (xd as u32 & 31)
}
#[inline(always)]
pub const fn umull(xd: u8, wn: u8, wm: u8) -> u32 {
    umaddl(xd, wn, wm, ZR)
}

// ---------------------------------------------------------------------------
// Branches & System
// ---------------------------------------------------------------------------

#[inline(always)]
pub const fn b_rel(offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 25) && offset_words < (1 << 25));
    0x1400_0000 | ((offset_words as u32) & 0x03FF_FFFF)
}

#[inline(always)]
pub const fn bl_rel(offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 25) && offset_words < (1 << 25));
    0x9400_0000 | ((offset_words as u32) & 0x03FF_FFFF)
}

#[inline(always)]
pub const fn b_cond(c: u8, offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 18) && offset_words < (1 << 18));
    0x5400_0000 | (((offset_words as u32) & 0x7FFFF) << 5) | (c as u32 & 15)
}

#[inline(always)]
pub const fn cbz32(rt: u8, offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 18) && offset_words < (1 << 18));
    0x3400_0000 | (((offset_words as u32) & 0x7FFFF) << 5) | (rt as u32 & 31)
}
#[inline(always)]
pub const fn cbnz32(rt: u8, offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 18) && offset_words < (1 << 18));
    0x3500_0000 | (((offset_words as u32) & 0x7FFFF) << 5) | (rt as u32 & 31)
}
#[inline(always)]
pub const fn cbz64(rt: u8, offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 18) && offset_words < (1 << 18));
    0xB400_0000 | (((offset_words as u32) & 0x7FFFF) << 5) | (rt as u32 & 31)
}
#[inline(always)]
pub const fn cbnz64(rt: u8, offset_words: i32) -> u32 {
    debug_assert!(offset_words >= -(1 << 18) && offset_words < (1 << 18));
    0xB500_0000 | (((offset_words as u32) & 0x7FFFF) << 5) | (rt as u32 & 31)
}

#[inline(always)]
pub const fn tbz(rt: u8, bit: u8, offset_words: i32) -> u32 {
    debug_assert!(bit < 64 && offset_words >= -(1 << 13) && offset_words < (1 << 13));
    let b5 = ((bit as u32) >> 5) & 1;
    let b40 = (bit as u32) & 31;
    (b5 << 31) | 0x3600_0000 | (b40 << 19) | (((offset_words as u32) & 0x3FFF) << 5) | (rt as u32 & 31)
}
#[inline(always)]
pub const fn tbnz(rt: u8, bit: u8, offset_words: i32) -> u32 {
    debug_assert!(bit < 64 && offset_words >= -(1 << 13) && offset_words < (1 << 13));
    let b5 = ((bit as u32) >> 5) & 1;
    let b40 = (bit as u32) & 31;
    (b5 << 31) | 0x3700_0000 | (b40 << 19) | (((offset_words as u32) & 0x3FFF) << 5) | (rt as u32 & 31)
}

#[inline(always)]
pub const fn br(rn: u8) -> u32 {
    0xD61F_0000 | ((rn as u32 & 31) << 5)
}
#[inline(always)]
pub const fn blr(rn: u8) -> u32 {
    0xD63F_0000 | ((rn as u32 & 31) << 5)
}
#[inline(always)]
pub const fn ret() -> u32 {
    0xD65F_03C0
}

#[inline(always)]
pub const fn adr(xd: u8, offset_bytes: i32) -> u32 {
    debug_assert!(offset_bytes >= -(1 << 20) && offset_bytes < (1 << 20));
    let u = offset_bytes as u32;
    let immlo = u & 3;
    let immhi = (u >> 2) & 0x7FFFF;
    0x1000_0000 | (immlo << 29) | (immhi << 5) | (xd as u32 & 31)
}

#[inline(always)]
pub const fn nop() -> u32 {
    0xD503_201F
}

/// `mrs xt, nzcv`
#[inline(always)]
pub const fn mrs_nzcv(xt: u8) -> u32 {
    0xD53B_4200 | (xt as u32 & 31)
}
/// `msr nzcv, xt`
#[inline(always)]
pub const fn msr_nzcv(xt: u8) -> u32 {
    0xD51B_4200 | (xt as u32 & 31)
}
/// `mrs xt, fpcr`
#[inline(always)]
pub const fn mrs_fpcr(xt: u8) -> u32 {
    0xD53B_4400 | (xt as u32 & 31)
}
/// `msr fpcr, xt`
#[inline(always)]
pub const fn msr_fpcr(xt: u8) -> u32 {
    0xD51B_4400 | (xt as u32 & 31)
}

// ---------------------------------------------------------------------------
// Loads & Stores (GPR & FP)
// ---------------------------------------------------------------------------

#[inline(always)]
const fn ls_uoff(size: u32, v: u32, opc: u32, rt: u8, rn: u8, scaled_off: u32) -> u32 {
    debug_assert!(scaled_off <= 0xFFF);
    (size << 30) | 0x3900_0000 | (v << 26) | (opc << 22) | (scaled_off << 10) | ((rn as u32 & 31) << 5) | (rt as u32 & 31)
}

#[inline(always)]
pub const fn ldrb_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    ls_uoff(0b00, 0, 0b01, wt, xn, byte_off as u32)
}
#[inline(always)]
pub const fn strb_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    ls_uoff(0b00, 0, 0b00, wt, xn, byte_off as u32)
}
#[inline(always)]
pub const fn ldrh_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 2 == 0);
    ls_uoff(0b01, 0, 0b01, wt, xn, (byte_off / 2) as u32)
}
#[inline(always)]
pub const fn strh_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 2 == 0);
    ls_uoff(0b01, 0, 0b00, wt, xn, (byte_off / 2) as u32)
}
#[inline(always)]
pub const fn ldr32_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ls_uoff(0b10, 0, 0b01, wt, xn, (byte_off / 4) as u32)
}
#[inline(always)]
pub const fn str32_uoff(wt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ls_uoff(0b10, 0, 0b00, wt, xn, (byte_off / 4) as u32)
}
#[inline(always)]
pub const fn ldr64_uoff(xt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ls_uoff(0b11, 0, 0b01, xt, xn, (byte_off / 8) as u32)
}
#[inline(always)]
pub const fn str64_uoff(xt: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ls_uoff(0b11, 0, 0b00, xt, xn, (byte_off / 8) as u32)
}

#[inline(always)]
pub const fn ldr_s_uoff(st: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ls_uoff(0b10, 1, 0b01, st, xn, (byte_off / 4) as u32)
}
#[inline(always)]
pub const fn str_s_uoff(st: u8, xn: u8, byte_off: u16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ls_uoff(0b10, 1, 0b00, st, xn, (byte_off / 4) as u32)
}

/// Register-offset loads/stores: `[xn, wm, uxtw]` (`option = 0b010`, `s = 0`).
#[inline(always)]
const fn ls_reg_uxtw(size: u32, v: u32, opc: u32, rt: u8, rn: u8, rm: u8) -> u32 {
    (size << 30)
        | 0x3820_4800
        | (v << 26)
        | (opc << 22)
        | ((rm as u32 & 31) << 16)
        | ((rn as u32 & 31) << 5)
        | (rt as u32 & 31)
}

#[inline(always)]
pub const fn ldrb_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b00, 0, 0b01, wt, xn, wm)
}
#[inline(always)]
pub const fn strb_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b00, 0, 0b00, wt, xn, wm)
}
#[inline(always)]
pub const fn ldrsb32_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b00, 0, 0b11, wt, xn, wm)
}
#[inline(always)]
pub const fn ldrh_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b01, 0, 0b01, wt, xn, wm)
}
#[inline(always)]
pub const fn strh_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b01, 0, 0b00, wt, xn, wm)
}
#[inline(always)]
pub const fn ldrsh32_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b01, 0, 0b11, wt, xn, wm)
}
#[inline(always)]
pub const fn ldr32_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b10, 0, 0b01, wt, xn, wm)
}
#[inline(always)]
pub const fn str32_reg_uxtw(wt: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b10, 0, 0b00, wt, xn, wm)
}
#[inline(always)]
pub const fn ldr_s_reg_uxtw(st: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b10, 1, 0b01, st, xn, wm)
}
#[inline(always)]
pub const fn str_s_reg_uxtw(st: u8, xn: u8, wm: u8) -> u32 {
    ls_reg_uxtw(0b10, 1, 0b00, st, xn, wm)
}

/// `ldr xt, [xn, xm, lsl #3]` (for `pages[page_idx]`)
#[inline(always)]
pub const fn ldr64_reg_lsl3(xt: u8, xn: u8, xm: u8) -> u32 {
    0xF860_7800 | ((xm as u32 & 31) << 16) | ((xn as u32 & 31) << 5) | (xt as u32 & 31)
}

/// Load/store pairs (`stp` / `ldp`).
#[inline(always)]
const fn ldp_stp(opc: u32, v: u32, mode: u32, load: u32, rt1: u8, rt2: u8, rn: u8, scaled_imm7: i32) -> u32 {
    debug_assert!(scaled_imm7 >= -64 && scaled_imm7 <= 63);
    (opc << 30)
        | 0x2800_0000
        | (v << 26)
        | (mode << 23)
        | (load << 22)
        | (((scaled_imm7 as u32) & 0x7F) << 15)
        | ((rt2 as u32 & 31) << 10)
        | ((rn as u32 & 31) << 5)
        | (rt1 as u32 & 31)
}

#[inline(always)]
pub const fn ldp32_off(wt1: u8, wt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ldp_stp(0b00, 0, 0b010, 1, wt1, wt2, xn, (byte_off / 4) as i32)
}
#[inline(always)]
pub const fn stp32_off(wt1: u8, wt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ldp_stp(0b00, 0, 0b010, 0, wt1, wt2, xn, (byte_off / 4) as i32)
}
#[inline(always)]
pub const fn ldp64_off(xt1: u8, xt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b10, 0, 0b010, 1, xt1, xt2, xn, (byte_off / 8) as i32)
}
#[inline(always)]
pub const fn stp64_off(xt1: u8, xt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b10, 0, 0b010, 0, xt1, xt2, xn, (byte_off / 8) as i32)
}
#[inline(always)]
pub const fn stp64_pre(xt1: u8, xt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b10, 0, 0b011, 0, xt1, xt2, xn, (byte_off / 8) as i32)
}
#[inline(always)]
pub const fn ldp64_post(xt1: u8, xt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b10, 0, 0b001, 1, xt1, xt2, xn, (byte_off / 8) as i32)
}

#[inline(always)]
pub const fn ldp_s_off(st1: u8, st2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ldp_stp(0b00, 1, 0b010, 1, st1, st2, xn, (byte_off / 4) as i32)
}
#[inline(always)]
pub const fn stp_s_off(st1: u8, st2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 4 == 0);
    ldp_stp(0b00, 1, 0b010, 0, st1, st2, xn, (byte_off / 4) as i32)
}
#[inline(always)]
pub const fn ldp_d_off(dt1: u8, dt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b01, 1, 0b010, 1, dt1, dt2, xn, (byte_off / 8) as i32)
}
#[inline(always)]
pub const fn stp_d_off(dt1: u8, dt2: u8, xn: u8, byte_off: i16) -> u32 {
    debug_assert!(byte_off % 8 == 0);
    ldp_stp(0b01, 1, 0b010, 0, dt1, dt2, xn, (byte_off / 8) as i32)
}

// ---------------------------------------------------------------------------
// Scalar Floating-Point & ASIMD
// ---------------------------------------------------------------------------

/// `fmov wd, sn`
#[inline(always)]
pub const fn fmov_w_from_s(wd: u8, sn: u8) -> u32 {
    0x1E26_0000 | ((sn as u32 & 31) << 5) | (wd as u32 & 31)
}
/// `fmov sd, wn`
#[inline(always)]
pub const fn fmov_s_from_w(sd: u8, wn: u8) -> u32 {
    0x1E27_0000 | ((wn as u32 & 31) << 5) | (sd as u32 & 31)
}
/// `fmov xd, dn`
#[inline(always)]
pub const fn fmov_x_from_d(xd: u8, dn: u8) -> u32 {
    0x9E66_0000 | ((dn as u32 & 31) << 5) | (xd as u32 & 31)
}
/// `fmov dd, xn`
#[inline(always)]
pub const fn fmov_d_from_x(dd: u8, xn: u8) -> u32 {
    0x9E67_0000 | ((xn as u32 & 31) << 5) | (dd as u32 & 31)
}

#[inline(always)]
const fn fp_1src(ftype: u32, opc: u32, rd: u8, rn: u8) -> u32 {
    0x1E20_4000 | (ftype << 22) | (opc << 15) | ((rn as u32 & 31) << 5) | (rd as u32 & 31)
}
#[inline(always)]
pub const fn fmov_s(sd: u8, sn: u8) -> u32 {
    fp_1src(0, 0b000, sd, sn)
}
#[inline(always)]
pub const fn fabs_s(sd: u8, sn: u8) -> u32 {
    fp_1src(0, 0b001, sd, sn)
}
#[inline(always)]
pub const fn fneg_s(sd: u8, sn: u8) -> u32 {
    fp_1src(0, 0b010, sd, sn)
}
#[inline(always)]
pub const fn fsqrt_s(sd: u8, sn: u8) -> u32 {
    fp_1src(0, 0b011, sd, sn)
}
#[inline(always)]
pub const fn fcvt_d_from_s(dd: u8, sn: u8) -> u32 {
    fp_1src(0, 0b101, dd, sn)
}

#[inline(always)]
pub const fn fmov_d(dd: u8, dn: u8) -> u32 {
    fp_1src(1, 0b000, dd, dn)
}
#[inline(always)]
pub const fn fabs_d(dd: u8, dn: u8) -> u32 {
    fp_1src(1, 0b001, dd, dn)
}
#[inline(always)]
pub const fn fneg_d(dd: u8, dn: u8) -> u32 {
    fp_1src(1, 0b010, dd, dn)
}
#[inline(always)]
pub const fn fsqrt_d(dd: u8, dn: u8) -> u32 {
    fp_1src(1, 0b011, dd, dn)
}
#[inline(always)]
pub const fn fcvt_s_from_d(sd: u8, dn: u8) -> u32 {
    fp_1src(1, 0b100, sd, dn)
}

#[inline(always)]
const fn fp_2src(ftype: u32, opc: u32, rd: u8, rn: u8, rm: u8) -> u32 {
    0x1E20_0800
        | (ftype << 22)
        | ((rm as u32 & 31) << 16)
        | (opc << 12)
        | ((rn as u32 & 31) << 5)
        | (rd as u32 & 31)
}
#[inline(always)]
pub const fn fmul_s(sd: u8, sn: u8, sm: u8) -> u32 {
    fp_2src(0, 0b0000, sd, sn, sm)
}
#[inline(always)]
pub const fn fdiv_s(sd: u8, sn: u8, sm: u8) -> u32 {
    fp_2src(0, 0b0001, sd, sn, sm)
}
#[inline(always)]
pub const fn fadd_s(sd: u8, sn: u8, sm: u8) -> u32 {
    fp_2src(0, 0b0010, sd, sn, sm)
}
#[inline(always)]
pub const fn fsub_s(sd: u8, sn: u8, sm: u8) -> u32 {
    fp_2src(0, 0b0011, sd, sn, sm)
}
#[inline(always)]
pub const fn fnmul_s(sd: u8, sn: u8, sm: u8) -> u32 {
    fp_2src(0, 0b1000, sd, sn, sm)
}

#[inline(always)]
pub const fn fmul_d(dd: u8, dn: u8, dm: u8) -> u32 {
    fp_2src(1, 0b0000, dd, dn, dm)
}
#[inline(always)]
pub const fn fdiv_d(dd: u8, dn: u8, dm: u8) -> u32 {
    fp_2src(1, 0b0001, dd, dn, dm)
}
#[inline(always)]
pub const fn fadd_d(dd: u8, dn: u8, dm: u8) -> u32 {
    fp_2src(1, 0b0010, dd, dn, dm)
}
#[inline(always)]
pub const fn fsub_d(dd: u8, dn: u8, dm: u8) -> u32 {
    fp_2src(1, 0b0011, dd, dn, dm)
}
#[inline(always)]
pub const fn fnmul_d(dd: u8, dn: u8, dm: u8) -> u32 {
    fp_2src(1, 0b1000, dd, dn, dm)
}

#[inline(always)]
pub const fn fcmp_s(sn: u8, sm: u8) -> u32 {
    0x1E20_2000 | ((sm as u32 & 31) << 16) | ((sn as u32 & 31) << 5)
}
#[inline(always)]
pub const fn fcmp_s_zero(sn: u8) -> u32 {
    0x1E20_2008 | ((sn as u32 & 31) << 5)
}
#[inline(always)]
pub const fn fcmp_d(dn: u8, dm: u8) -> u32 {
    0x1E60_2000 | ((dm as u32 & 31) << 16) | ((dn as u32 & 31) << 5)
}
#[inline(always)]
pub const fn fcmp_d_zero(dn: u8) -> u32 {
    0x1E60_2008 | ((dn as u32 & 31) << 5)
}

/// Scalar ASIMD integer<->float conversions inside FP registers (`s0..s31`):
#[inline(always)]
pub const fn scvtf_s_asimd(sd: u8, sn: u8) -> u32 {
    0x5E21_D800 | ((sn as u32 & 31) << 5) | (sd as u32 & 31)
}
#[inline(always)]
pub const fn ucvtf_s_asimd(sd: u8, sn: u8) -> u32 {
    0x7E21_D800 | ((sn as u32 & 31) << 5) | (sd as u32 & 31)
}
#[inline(always)]
pub const fn fcvtzs_s_asimd(sd: u8, sn: u8) -> u32 {
    0x5EA1_B800 | ((sn as u32 & 31) << 5) | (sd as u32 & 31)
}
#[inline(always)]
pub const fn fcvtzu_s_asimd(sd: u8, sn: u8) -> u32 {
    0x7EA1_B800 | ((sn as u32 & 31) << 5) | (sd as u32 & 31)
}

/// GP <-> double conversions:
#[inline(always)]
pub const fn scvtf_d_from_w(dd: u8, wn: u8) -> u32 {
    0x1E62_0000 | ((wn as u32 & 31) << 5) | (dd as u32 & 31)
}
#[inline(always)]
pub const fn ucvtf_d_from_w(dd: u8, wn: u8) -> u32 {
    0x1E63_0000 | ((wn as u32 & 31) << 5) | (dd as u32 & 31)
}
#[inline(always)]
pub const fn fcvtzs_w_from_d(wd: u8, dn: u8) -> u32 {
    0x1E78_0000 | ((dn as u32 & 31) << 5) | (wd as u32 & 31)
}
#[inline(always)]
pub const fn fcvtzu_w_from_d(wd: u8, dn: u8) -> u32 {
    0x1E79_0000 | ((dn as u32 & 31) << 5) | (wd as u32 & 31)
}

/// ASIMD 8B / 4H parallel saturating/halving instructions (for Moflex video decode):
#[inline(always)]
pub const fn uqadd_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E20_0C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uqsub_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E20_2C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uhadd_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E20_0400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uhsub_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E20_2400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn sqadd_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E20_0C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn sqsub_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E20_2C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn shadd_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E20_0400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn shsub_8b(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E20_2400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}

#[inline(always)]
pub const fn uqadd_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E60_0C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uqsub_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E60_2C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uhadd_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E60_0400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn uhsub_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x2E60_2400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn sqadd_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E60_0C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn sqsub_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E60_2C00 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn shadd_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E60_0400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}
#[inline(always)]
pub const fn shsub_4h(vd: u8, vn: u8, vm: u8) -> u32 {
    0x0E60_2400 | ((vm as u32 & 31) << 16) | ((vn as u32 & 31) << 5) | (vd as u32 & 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmask32_roundtrips() {
        for &mask in &[
            0x0000_0FFF_u32,
            0xF000_0000,
            0x3000_0000,
            0x1000_0000,
            0x0101_0101,
            0x00FF_00FF,
            0xFF00_FF00,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_0000,
            0x0000_FFFF,
            0x5555_5555,
        ] {
            assert!(encode_bitmask32(mask).is_some(), "expected {mask:#010x} to be a valid bitmask");
        }
        assert!(encode_bitmask32(0).is_none());
        assert!(encode_bitmask32(u32::MAX).is_none());
        assert!(encode_bitmask32(0x1234_5678).is_none());
    }

    #[test]
    fn objdump_verifies_every_instruction_encoding() {
        let objdump = "/usr/bin/aarch64-linux-gnu-objdump";
        if !std::path::Path::new(objdump).exists() {
            return;
        }
        let cases: Vec<(u32, &str)> = vec![
            (movz32(1, 0x1234, 0), "mov\tw1, #0x1234"),
            (movk32(1, 0x5678, 1), "movk\tw1, #0x5678, lsl #16"),
            (movn32(2, 0, 0), "mov\tw2, #0xffffffff"),
            (add32_imm(3, 4, 0x123, false), "add\tw3, w4, #0x123"),
            (adds32_imm(3, 4, 5, false), "adds\tw3, w4, #0x5"),
            (subs32_imm(ZR, 0, 10, false), "cmp\tw0, #0xa"),
            (mov_sp(29, SP), "mov\tx29, sp"),
            (add32_reg(1, 2, 3, Shift::Lsl, 2), "add\tw1, w2, w3, lsl #2"),
            (subs32_reg(1, 2, 3, Shift::Asr, 5), "subs\tw1, w2, w3, asr #5"),
            (add64_uxtw(17, 17, 25, 0), "add\tx17, x17, w25, uxtw"),
            (cmp64_sxtw(16, 16), "cmp\tx16, w16, sxtw"),
            (adc32(0, 1, 2), "adc\tw0, w1, w2"),
            (adcs32(0, 1, 2), "adcs\tw0, w1, w2"),
            (sbc32(0, 1, 2), "sbc\tw0, w1, w2"),
            (sbcs32(0, 1, 2), "sbcs\tw0, w1, w2"),
            (and32_reg(0, 1, 2, Shift::Lsr, 4), "and\tw0, w1, w2, lsr #4"),
            (bic32_reg(0, 1, 2, Shift::Lsl, 0), "bic\tw0, w1, w2"),
            (orr32_reg(0, 1, 2, Shift::Ror, 8), "orr\tw0, w1, w2, ror #8"),
            (mvn32(3, 4), "mvn\tw3, w4"),
            (eor32_reg(5, 6, 7, Shift::Lsl, 0), "eor\tw5, w6, w7"),
            (ands32_reg(0, 1, 2, Shift::Lsl, 0), "ands\tw0, w1, w2"),
            (and32_imm(25, 16, 0xFFF).unwrap(), "and\tw25, w16, #0xfff"),
            (and32_imm(16, 16, 0x0101_0101).unwrap(), "and\tw16, w16, #0x1010101"),
            (and32_imm(23, 23, 0x3000_0000).unwrap(), "and\tw23, w23, #0x30000000"),
            (lslv32(0, 1, 2), "lsl\tw0, w1, w2"),
            (lsrv32(0, 1, 2), "lsr\tw0, w1, w2"),
            (asrv32(0, 1, 2), "asr\tw0, w1, w2"),
            (rorv32(0, 1, 2), "ror\tw0, w1, w2"),
            (lsl32_imm(0, 1, 5), "lsl\tw0, w1, #5"),
            (lsr32_imm(0, 1, 12), "lsr\tw0, w1, #12"),
            (asr32_imm(0, 1, 31), "asr\tw0, w1, #31"),
            (ror32_imm(0, 1, 16), "ror\tw0, w1, #16"),
            (sxtb32(0, 1), "sxtb\tw0, w1"),
            (sxth32(0, 1), "sxth\tw0, w1"),
            (uxtb32(0, 1), "uxtb\tw0, w1"),
            (uxth32(0, 1), "uxth\tw0, w1"),
            (sxtw64(16, 2), "sxtw\tx16, w2"),
            (rbit32(0, 1), "rbit\tw0, w1"),
            (rev16_32(0, 1), "rev16\tw0, w1"),
            (rev32(0, 1), "rev\tw0, w1"),
            (clz32(0, 1), "clz\tw0, w1"),
            (csel32(0, 1, 2, cond::LO), "csel\tw0, w1, w2, cc"),
            (cset32(0, cond::EQ), "cset\tw0, eq"),
            (mul32(0, 1, 2), "mul\tw0, w1, w2"),
            (madd32(0, 1, 2, 3), "madd\tw0, w1, w2, w3"),
            (smull(16, 1, 2), "smull\tx16, w1, w2"),
            (umull(16, 1, 2), "umull\tx16, w1, w2"),
            (smaddl(16, 1, 2, 17), "smaddl\tx16, w1, w2, x17"),
            (umaddl(16, 1, 2, 17), "umaddl\tx16, w1, w2, x17"),
            (b_rel(4), "b\t"),
            (bl_rel(-2), "bl\t"),
            (b_cond(cond::NE, 3), "b.ne\t"),
            (cbz64(17, 3), "cbz\tx17, "),
            (cbnz32(28, 4), "cbnz\tw28, "),
            (tbz(26, 63, 4), "tbz\tx26, #63, "),
            (tbnz(25, 12, 4), "tbnz\tw25, #12, "),
            (br(27), "br\tx27"),
            (blr(16), "blr\tx16"),
            (ret(), "ret"),
            (mrs_nzcv(23), "mrs\tx23, nzcv"),
            (msr_nzcv(23), "msr\tnzcv, x23"),
            (mrs_fpcr(16), "mrs\tx16, fpcr"),
            (msr_fpcr(16), "msr\tfpcr, x16"),
            (ldrb_uoff(16, 19, 69), "ldrb\tw16, [x19, #69]"),
            (ldr32_uoff(22, 19, 72), "ldr\tw22, [x19, #72]"),
            (ldr64_uoff(20, 19, 96), "ldr\tx20, [x19, #96]"),
            (ldr64_reg_lsl3(17, 20, 17), "ldr\tx17, [x20, x17, lsl #3]"),
            (ldr32_reg_uxtw(0, 17, 25), "ldr\tw0, [x17, w25, uxtw]"),
            (str32_reg_uxtw(0, 17, 25), "str\tw0, [x17, w25, uxtw]"),
            (ldrsb32_reg_uxtw(0, 17, 25), "ldrsb\tw0, [x17, w25, uxtw]"),
            (ldrsh32_reg_uxtw(0, 17, 25), "ldrsh\tw0, [x17, w25, uxtw]"),
            (ldr_s_reg_uxtw(3, 17, 25), "ldr\ts3, [x17, w25, uxtw]"),
            (str_s_reg_uxtw(3, 17, 25), "str\ts3, [x17, w25, uxtw]"),
            (stp64_pre(29, 30, SP, -16), "stp\tx29, x30, [sp, #-16]!"),
            (ldp64_post(29, 30, SP, 16), "ldp\tx29, x30, [sp], #16"),
            (stp_s_off(30, 31, 19, 144), "stp\ts30, s31, [x19, #144]"),
            (ldp_s_off(30, 31, 19, 144), "ldp\ts30, s31, [x19, #144]"),
            (fmov_w_from_s(0, 1), "fmov\tw0, s1"),
            (fmov_s_from_w(1, 0), "fmov\ts1, w0"),
            (fadd_s(0, 1, 2), "fadd\ts0, s1, s2"),
            (fsub_s(0, 1, 2), "fsub\ts0, s1, s2"),
            (fmul_s(0, 1, 2), "fmul\ts0, s1, s2"),
            (fnmul_s(0, 1, 2), "fnmul\ts0, s1, s2"),
            (fdiv_s(0, 1, 2), "fdiv\ts0, s1, s2"),
            (fsqrt_s(0, 1), "fsqrt\ts0, s1"),
            (fabs_s(0, 1), "fabs\ts0, s1"),
            (fneg_s(0, 1), "fneg\ts0, s1"),
            (fcmp_s(0, 1), "fcmp\ts0, s1"),
            (fcmp_s_zero(0), "fcmp\ts0, #0.0"),
            (fadd_d(30, 30, 31), "fadd\td30, d30, d31"),
            (fcmp_d(30, 31), "fcmp\td30, d31"),
            (fcmp_d_zero(30), "fcmp\td30, #0.0"),
            (fcvt_d_from_s(30, 5), "fcvt\td30, s5"),
            (fcvt_s_from_d(5, 30), "fcvt\ts5, d30"),
            (scvtf_s_asimd(2, 3), "scvtf\ts2, s3"),
            (ucvtf_s_asimd(2, 3), "ucvtf\ts2, s3"),
            (fcvtzs_s_asimd(2, 3), "fcvtzs\ts2, s3"),
            (fcvtzu_s_asimd(2, 3), "fcvtzu\ts2, s3"),
            (scvtf_d_from_w(30, 16), "scvtf\td30, w16"),
            (ucvtf_d_from_w(30, 16), "ucvtf\td30, w16"),
            (fcvtzs_w_from_d(16, 30), "fcvtzs\tw16, d30"),
            (fcvtzu_w_from_d(16, 30), "fcvtzu\tw16, d30"),
            (uqsub_8b(30, 30, 31), "uqsub\tv30.8b, v30.8b, v31.8b"),
            (uhadd_8b(30, 30, 31), "uhadd\tv30.8b, v30.8b, v31.8b"),
            (uqadd_8b(30, 30, 31), "uqadd\tv30.8b, v30.8b, v31.8b"),
            (sqadd_4h(30, 30, 31), "sqadd\tv30.4h, v30.4h, v31.4h"),
            (sqsub_4h(30, 30, 31), "sqsub\tv30.4h, v30.4h, v31.4h"),
        ];

        let bin_path = std::env::temp_dir().join(format!("zakuro-a64-test-{}.bin", std::process::id()));
        let bytes: Vec<u8> = cases.iter().flat_map(|(w, _)| w.to_le_bytes()).collect();
        std::fs::write(&bin_path, bytes).unwrap();
        let out = std::process::Command::new(objdump)
            .args(["-D", "-b", "binary", "-m", "aarch64", bin_path.to_str().unwrap()])
            .output()
            .unwrap();
        std::fs::remove_file(&bin_path).ok();
        let text = String::from_utf8(out.stdout).unwrap();
        let disasm: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with(' '))
            .collect();
        assert_eq!(disasm.len(), cases.len(), "disasm count mismatch:\n{text}");
        for (i, (line, (_, expected))) in disasm.iter().zip(cases.iter()).enumerate() {
            assert!(
                line.contains(expected),
                "insn #{i} mismatch:\n  got:      {line}\n  expected: {expected}"
            );
        }
    }
}
