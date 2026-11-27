//! Zero-dependency x86_64 machine-code instruction encoder for `zakuro-aot`.

pub const RAX: u8 = 0;
pub const RCX: u8 = 1;
pub const RDX: u8 = 2;
pub const RBX: u8 = 3;
pub const RSP: u8 = 4;
pub const RBP: u8 = 5;
pub const RSI: u8 = 6;
pub const RDI: u8 = 7;
pub const R8: u8 = 8;
pub const R9: u8 = 9;
pub const R10: u8 = 10;
pub const R11: u8 = 11;
pub const R12: u8 = 12;
pub const R13: u8 = 13;
pub const R14: u8 = 14;
pub const R15: u8 = 15;

pub const XMM0: u8 = 0;
pub const XMM1: u8 = 1;
pub const XMM2: u8 = 2;
pub const XMM3: u8 = 3;
pub const XMM4: u8 = 4;
pub const XMM5: u8 = 5;
pub const XMM6: u8 = 6;
pub const XMM7: u8 = 7;
pub const XMM8: u8 = 8;
pub const XMM9: u8 = 9;
pub const XMM10: u8 = 10;
pub const XMM11: u8 = 11;
pub const XMM12: u8 = 12;
pub const XMM13: u8 = 13;
pub const XMM14: u8 = 14;
pub const XMM15: u8 = 15;

pub mod cc {
    pub const O: u8 = 0x0;
    pub const NO: u8 = 0x1;
    pub const B: u8 = 0x2;
    pub const C: u8 = 0x2;
    pub const NAE: u8 = 0x2;
    pub const AE: u8 = 0x3;
    pub const NC: u8 = 0x3;
    pub const NB: u8 = 0x3;
    pub const E: u8 = 0x4;
    pub const Z: u8 = 0x4;
    pub const NE: u8 = 0x5;
    pub const NZ: u8 = 0x5;
    pub const BE: u8 = 0x6;
    pub const NA: u8 = 0x6;
    pub const A: u8 = 0x7;
    pub const NBE: u8 = 0x7;
    pub const S: u8 = 0x8;
    pub const NS: u8 = 0x9;
    pub const P: u8 = 0xA;
    pub const PE: u8 = 0xA;
    pub const NP: u8 = 0xB;
    pub const PO: u8 = 0xB;
    pub const L: u8 = 0xC;
    pub const NGE: u8 = 0xC;
    pub const GE: u8 = 0xD;
    pub const NL: u8 = 0xD;
    pub const LE: u8 = 0xE;
    pub const NG: u8 = 0xE;
    pub const G: u8 = 0xF;
    pub const NLE: u8 = 0xF;

    #[inline(always)]
    pub const fn invert(c: u8) -> u8 {
        c ^ 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AluOp {
    Add = 0,
    Or = 1,
    Adc = 2,
    Sbb = 3,
    And = 4,
    Sub = 5,
    Xor = 6,
    Cmp = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ShiftOp {
    Rol = 0,
    Ror = 1,
    Shl = 4,
    Shr = 5,
    Sar = 7,
}

#[derive(Default, Clone)]
pub struct X64Buf {
    pub bytes: Vec<u8>,
}

impl X64Buf {
    pub fn with_capacity(cap: usize) -> Self {
        Self { bytes: Vec::with_capacity(cap) }
    }

    #[inline(always)]
    pub fn pos(&self) -> usize {
        self.bytes.len()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline(always)]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    #[inline(always)]
    pub fn emit_u8(&mut self, b: u8) {
        self.bytes.push(b);
    }

    #[inline(always)]
    pub fn emit_u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    #[inline(always)]
    pub fn emit_u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    #[inline(always)]
    pub fn patch_i32(&mut self, disp_pos: usize, rel: i32) {
        self.bytes[disp_pos..disp_pos + 4].copy_from_slice(&rel.to_le_bytes());
    }

    #[inline(always)]
    pub fn patch_rel32(&mut self, disp_pos: usize, target_pos: usize) {
        let rel = (target_pos as i64) - ((disp_pos as i64) + 4);
        let rel32 = i32::try_from(rel).expect("x86_64 rel32 branch displacement overflow");
        self.patch_i32(disp_pos, rel32);
    }

    #[inline(always)]
    fn modrm(mod_bits: u8, reg: u8, rm: u8) -> u8 {
        ((mod_bits & 3) << 6) | ((reg & 7) << 3) | (rm & 7)
    }

    #[inline(always)]
    fn sib(scale_bits: u8, idx: u8, base: u8) -> u8 {
        ((scale_bits & 3) << 6) | ((idx & 7) << 3) | (base & 7)
    }

    #[inline(always)]
    fn disp_encoding(base: u8, disp: i32) -> (u8, u8) {
        // RBP (5) and R13 (13) with mod=00 encode [rip+disp32] / SIB disp32-without-base,
        // so disp == 0 must use mod=01 with disp8=0.
        if disp == 0 && (base & 7) != 5 {
            (0b00, 0)
        } else if i8::try_from(disp).is_ok() {
            (0b01, 1)
        } else {
            (0b10, 4)
        }
    }

    #[inline(always)]
    fn emit_disp(&mut self, disp_len: u8, disp: i32) {
        match disp_len {
            0 => {}
            1 => self.bytes.push(disp as i8 as u8),
            4 => self.emit_u32(disp as u32),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    fn emit_rex(&mut self, w: bool, reg: u8, idx: u8, rm: u8, force_rex: bool) {
        debug_assert!(reg < 16);
        debug_assert!(idx < 16);
        debug_assert!(rm < 16);
        let rex = 0x40
            | ((w as u8) << 3)
            | (((reg >> 3) & 1) << 2)
            | (((idx >> 3) & 1) << 1)
            | ((rm >> 3) & 1);
        if rex != 0x40 || force_rex {
            self.bytes.push(rex);
        }
    }

    #[inline(always)]
    fn emit_prefix_and_op(
        &mut self,
        prefix: Option<u8>,
        w: bool,
        reg: u8,
        idx: u8,
        rm: u8,
        force_rex: bool,
        op1: u8,
        op2: Option<u8>,
    ) {
        if let Some(p) = prefix {
            self.bytes.push(p);
        }
        self.emit_rex(w, reg, idx, rm, force_rex);
        self.bytes.push(op1);
        if let Some(o2) = op2 {
            self.bytes.push(o2);
        }
    }

    #[inline]
    fn emit_rr(
        &mut self,
        prefix: Option<u8>,
        w: bool,
        op1: u8,
        op2: Option<u8>,
        reg: u8,
        rm: u8,
        force_rex: bool,
    ) {
        self.emit_prefix_and_op(prefix, w, reg, 0, rm, force_rex, op1, op2);
        self.bytes.push(Self::modrm(3, reg, rm));
    }

    #[inline]
    fn emit_mem(
        &mut self,
        prefix: Option<u8>,
        w: bool,
        op1: u8,
        op2: Option<u8>,
        reg: u8,
        base: u8,
        disp: i32,
        force_rex: bool,
    ) {
        self.emit_prefix_and_op(prefix, w, reg, 0, base, force_rex, op1, op2);
        let (mod_bits, disp_len) = Self::disp_encoding(base, disp);
        self.bytes.push(Self::modrm(mod_bits, reg, base));
        if (base & 7) == 4 {
            // RSP (4) or R12 (12): rm=4 requires SIB 0x24 (index=4 [none], base=4).
            self.bytes.push(0x24);
        }
        self.emit_disp(disp_len, disp);
    }

    #[inline]
    fn emit_sib(
        &mut self,
        prefix: Option<u8>,
        w: bool,
        op1: u8,
        op2: Option<u8>,
        reg: u8,
        base: u8,
        idx: u8,
        scale: u8,
        disp: i32,
        force_rex: bool,
    ) {
        debug_assert_ne!(idx, RSP, "SIB index cannot be RSP (encoding 4 means no index)");
        let ss = match scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => panic!("invalid x86_64 SIB scale {scale} (expected 1, 2, 4, or 8)"),
        };
        self.emit_prefix_and_op(prefix, w, reg, idx, base, force_rex, op1, op2);
        let (mod_bits, disp_len) = Self::disp_encoding(base, disp);
        self.bytes.push(Self::modrm(mod_bits, reg, 4));
        self.bytes.push(Self::sib(ss, idx, base));
        self.emit_disp(disp_len, disp);
    }

    #[inline]
    fn alu_imm32(&mut self, w: bool, op: AluOp, dst: u8, imm: i32) {
        if let Ok(imm8) = i8::try_from(imm) {
            self.emit_rr(None, w, 0x83, None, op as u8, dst, false);
            self.emit_u8(imm8 as u8);
        } else {
            self.emit_rr(None, w, 0x81, None, op as u8, dst, false);
            self.emit_u32(imm as u32);
        }
    }

    #[inline]
    fn shift_imm(&mut self, w: bool, op: ShiftOp, dst: u8, imm: u8) {
        if imm == 0 {
            return;
        }
        if imm == 1 {
            self.emit_rr(None, w, 0xD1, None, op as u8, dst, false);
        } else {
            self.emit_rr(None, w, 0xC1, None, op as u8, dst, false);
            self.emit_u8(imm);
        }
    }

    // Stack & control flow
    pub fn push_r64(&mut self, r: u8) {
        self.emit_rex(false, 0, 0, r, false);
        self.emit_u8(0x50 | (r & 7));
    }

    pub fn pop_r64(&mut self, r: u8) {
        self.emit_rex(false, 0, 0, r, false);
        self.emit_u8(0x58 | (r & 7));
    }

    pub fn ret(&mut self) {
        self.bytes.push(0xC3);
    }

    pub fn nop(&mut self) {
        self.bytes.push(0x90);
    }

    pub fn nop_n(&mut self, count: usize) { for _ in 0..count { self.bytes.push(0x90); } }
    pub fn int3(&mut self) { self.bytes.push(0xCC); }
    pub fn clc(&mut self) { self.bytes.push(0xF8); }
    pub fn stc(&mut self) { self.bytes.push(0xF9); }
    pub fn cmc(&mut self) { self.bytes.push(0xF5); }

    pub fn jmp_rel32(&mut self) -> usize {
        self.bytes.push(0xE9);
        let disp_pos = self.pos();
        self.emit_u32(0);
        disp_pos
    }

    pub fn jmp_to(&mut self, target: usize) {
        let disp_pos = self.jmp_rel32();
        self.patch_rel32(disp_pos, target);
    }

    pub fn jcc_rel32(&mut self, cond: u8) -> usize {
        debug_assert!(cond < 16);
        self.bytes.push(0x0F);
        self.bytes.push(0x80 | (cond & 0xF));
        let disp_pos = self.pos();
        self.emit_u32(0);
        disp_pos
    }

    pub fn jcc_to(&mut self, cond: u8, target: usize) {
        let disp_pos = self.jcc_rel32(cond);
        self.patch_rel32(disp_pos, target);
    }

    pub fn call_rel32(&mut self) -> usize {
        self.bytes.push(0xE8);
        let disp_pos = self.pos();
        self.emit_u32(0);
        disp_pos
    }

    pub fn call_to(&mut self, target: usize) {
        let disp_pos = self.call_rel32();
        self.patch_rel32(disp_pos, target);
    }

    pub fn jmp_r64(&mut self, r: u8) {
        self.emit_rr(None, false, 0xFF, None, 4, r, false);
    }

    pub fn call_r64(&mut self, r: u8) {
        self.emit_rr(None, false, 0xFF, None, 2, r, false);
    }

    pub fn jmp_mem64(&mut self, base: u8, disp: i32) {
        self.emit_mem(None, false, 0xFF, None, 4, base, disp, false);
    }

    pub fn call_mem64(&mut self, base: u8, disp: i32) {
        self.emit_mem(None, false, 0xFF, None, 2, base, disp, false);
    }

    // Moves & extensions (NEVER clobber flags!)
    pub fn mov_r64_r64(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, true, 0x89, None, src, dst, false);
    }

    pub fn mov_r32_r32(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x89, None, src, dst, false);
    }

    pub fn mov_r32_imm32(&mut self, dst: u8, imm: u32) {
        // Never optimize imm == 0 to `xor`: `mov r32, imm32` preserves EFLAGS.
        self.emit_rex(false, 0, 0, dst, false);
        self.emit_u8(0xB8 | (dst & 7));
        self.emit_u32(imm);
    }

    pub fn mov_r64_imm64(&mut self, dst: u8, imm: u64) {
        // Never emit `xor`: all three encodings preserve EFLAGS.
        if imm <= u32::MAX as u64 {
            self.mov_r32_imm32(dst, imm as u32);
        } else if let Ok(imm32) = i32::try_from(imm as i64) {
            self.emit_rr(None, true, 0xC7, None, 0, dst, false);
            self.emit_u32(imm32 as u32);
        } else {
            self.emit_rex(true, 0, 0, dst, false);
            self.emit_u8(0xB8 | (dst & 7));
            self.emit_u64(imm);
        }
    }

    pub fn movsx_r32_r8(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBE), dst, src, src >= 4);
    }

    pub fn movsx_r32_r16(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBF), dst, src, false);
    }

    pub fn movzx_r32_r8(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xB6), dst, src, src >= 4);
    }

    pub fn movzx_r32_r16(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xB7), dst, src, false);
    }

    pub fn movsxd_r64_r32(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, true, 0x63, None, dst, src, false);
    }

    pub fn cmovcc_r32_r32(&mut self, cond: u8, dst: u8, src: u8) {
        debug_assert!(cond < 16);
        self.emit_rr(None, false, 0x0F, Some(0x40 | (cond & 0xF)), dst, src, false);
    }

    pub fn cmovcc_r64_r64(&mut self, cond: u8, dst: u8, src: u8) {
        debug_assert!(cond < 16);
        self.emit_rr(None, true, 0x0F, Some(0x40 | (cond & 0xF)), dst, src, false);
    }

    pub fn setcc_r8(&mut self, cond: u8, dst: u8) {
        debug_assert!(cond < 16);
        self.emit_rr(None, false, 0x0F, Some(0x90 | (cond & 0xF)), 0, dst, dst >= 4);
    }

    // Memory [base + disp]
    pub fn ldr_r64_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, true, 0x8B, None, dst, base, disp, false);
    }

    pub fn str_r64_mem(&mut self, src: u8, base: u8, disp: i32) {
        self.emit_mem(None, true, 0x89, None, src, base, disp, false);
    }

    pub fn ldr_r32_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x8B, None, dst, base, disp, false);
    }

    pub fn str_r32_mem(&mut self, src: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x89, None, src, base, disp, false);
    }

    pub fn ldr_u8_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x0F, Some(0xB6), dst, base, disp, false);
    }

    pub fn ldr_s8_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x0F, Some(0xBE), dst, base, disp, false);
    }

    pub fn ldr_u16_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x0F, Some(0xB7), dst, base, disp, false);
    }

    pub fn ldr_s16_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x0F, Some(0xBF), dst, base, disp, false);
    }

    pub fn str_r8_mem(&mut self, src: u8, base: u8, disp: i32) {
        self.emit_mem(None, false, 0x88, None, src, base, disp, src >= 4);
    }

    pub fn str_r16_mem(&mut self, src: u8, base: u8, disp: i32) {
        self.emit_mem(Some(0x66), false, 0x89, None, src, base, disp, false);
    }

    pub fn str_imm32_mem(&mut self, base: u8, disp: i32, imm: u32) {
        self.emit_mem(None, false, 0xC7, None, 0, base, disp, false);
        self.emit_u32(imm);
    }

    pub fn str_imm8_mem(&mut self, base: u8, disp: i32, imm: u8) {
        self.emit_mem(None, false, 0xC6, None, 0, base, disp, false);
        self.emit_u8(imm);
    }

    pub fn lea_r64_mem(&mut self, dst: u8, base: u8, disp: i32) {
        self.emit_mem(None, true, 0x8D, None, dst, base, disp, false);
    }

    // Memory [base + idx*scale + disp] (`scale` in {1, 2, 4, 8})
    pub fn ldr_r64_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, true, 0x8B, None, dst, base, idx, scale, disp, false);
    }

    pub fn str_r64_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, true, 0x89, None, src, base, idx, scale, disp, false);
    }

    pub fn ldr_r32_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x8B, None, dst, base, idx, scale, disp, false);
    }

    pub fn str_r32_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x89, None, src, base, idx, scale, disp, false);
    }

    pub fn ldr_u8_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x0F, Some(0xB6), dst, base, idx, scale, disp, false);
    }

    pub fn ldr_s8_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x0F, Some(0xBE), dst, base, idx, scale, disp, false);
    }

    pub fn ldr_u16_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x0F, Some(0xB7), dst, base, idx, scale, disp, false);
    }

    pub fn ldr_s16_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x0F, Some(0xBF), dst, base, idx, scale, disp, false);
    }

    pub fn str_r8_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, false, 0x88, None, src, base, idx, scale, disp, src >= 4);
    }

    pub fn str_r16_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(Some(0x66), false, 0x89, None, src, base, idx, scale, disp, false);
    }

    pub fn lea_r64_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) {
        self.emit_sib(None, true, 0x8D, None, dst, base, idx, scale, disp, false);
    }

    // Integer ALU
    pub fn alu_r32_r32(&mut self, op: AluOp, dst: u8, src: u8) {
        self.emit_rr(None, false, ((op as u8) << 3) | 0x01, None, src, dst, false);
    }

    pub fn alu_r32_imm32(&mut self, op: AluOp, dst: u8, imm: i32) {
        self.alu_imm32(false, op, dst, imm);
    }

    pub fn alu_r64_r64(&mut self, op: AluOp, dst: u8, src: u8) {
        self.emit_rr(None, true, ((op as u8) << 3) | 0x01, None, src, dst, false);
    }

    pub fn alu_r64_imm32(&mut self, op: AluOp, dst: u8, imm: i32) {
        self.alu_imm32(true, op, dst, imm);
    }

    pub fn test_r32_r32(&mut self, a: u8, b: u8) {
        self.emit_rr(None, false, 0x85, None, b, a, false);
    }

    pub fn test_r64_r64(&mut self, a: u8, b: u8) {
        self.emit_rr(None, true, 0x85, None, b, a, false);
    }

    pub fn test_r32_imm32(&mut self, a: u8, imm: u32) {
        self.emit_rr(None, false, 0xF7, None, 0, a, false);
        self.emit_u32(imm);
    }

    pub fn test_r8_imm8(&mut self, a: u8, imm: u8) {
        self.emit_rr(None, false, 0xF6, None, 0, a, a >= 4);
        self.emit_u8(imm);
    }

    pub fn neg_r32(&mut self, r: u8) {
        self.emit_rr(None, false, 0xF7, None, 3, r, false);
    }

    pub fn not_r32(&mut self, r: u8) {
        self.emit_rr(None, false, 0xF7, None, 2, r, false);
    }

    pub fn imul_r32_r32(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xAF), dst, src, false);
    }

    pub fn imul_r64_r64(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, true, 0x0F, Some(0xAF), dst, src, false);
    }

    pub fn mul_rax_r32(&mut self, src: u8) {
        self.emit_rr(None, false, 0xF7, None, 4, src, false);
    }

    pub fn imul_rax_r32(&mut self, src: u8) {
        self.emit_rr(None, false, 0xF7, None, 5, src, false);
    }

    pub fn mul_rax_r64(&mut self, src: u8) {
        self.emit_rr(None, true, 0xF7, None, 4, src, false);
    }

    pub fn imul_rax_r64(&mut self, src: u8) {
        self.emit_rr(None, true, 0xF7, None, 5, src, false);
    }

    pub fn shift_r32_imm(&mut self, op: ShiftOp, dst: u8, imm: u8) {
        self.shift_imm(false, op, dst, imm);
    }

    pub fn shift_r32_cl(&mut self, op: ShiftOp, dst: u8) {
        self.emit_rr(None, false, 0xD3, None, op as u8, dst, false);
    }

    pub fn shift_r64_imm(&mut self, op: ShiftOp, dst: u8, imm: u8) {
        self.shift_imm(true, op, dst, imm);
    }

    pub fn shift_r64_cl(&mut self, op: ShiftOp, dst: u8) {
        self.emit_rr(None, true, 0xD3, None, op as u8, dst, false);
    }

    pub fn bswap_r32(&mut self, r: u8) {
        self.emit_rex(false, 0, 0, r, false);
        self.emit_u8(0x0F);
        self.emit_u8(0xC8 | (r & 7));
    }

    pub fn bsr_r32(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBD), dst, src, false);
    }

    pub fn bt_r32_imm8(&mut self, r: u8, bit: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBA), 4, r, false);
        self.emit_u8(bit);
    }

    pub fn bts_r32_imm8(&mut self, r: u8, bit: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBA), 5, r, false);
        self.emit_u8(bit);
    }

    pub fn btr_r32_imm8(&mut self, r: u8, bit: u8) {
        self.emit_rr(None, false, 0x0F, Some(0xBA), 6, r, false);
        self.emit_u8(bit);
    }

    // SSE2 scalar float / double
    pub fn movd_xmm_r32(&mut self, xmm: u8, r32: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x6E), xmm, r32, false);
    }

    pub fn movd_r32_xmm(&mut self, r32: u8, xmm: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x7E), xmm, r32, false);
    }

    pub fn movq_xmm_r64(&mut self, xmm: u8, r64: u8) {
        self.emit_rr(Some(0x66), true, 0x0F, Some(0x6E), xmm, r64, false);
    }

    pub fn movq_r64_xmm(&mut self, r64: u8, xmm: u8) {
        self.emit_rr(Some(0x66), true, 0x0F, Some(0x7E), xmm, r64, false);
    }

    pub fn movss_xmm_xmm(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x10), dst, src, false);
    }

    pub fn movsd_xmm_xmm(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x10), dst, src, false);
    }

    pub fn movss_load(&mut self, xmm: u8, base: u8, disp: i32) {
        self.emit_mem(Some(0xF3), false, 0x0F, Some(0x10), xmm, base, disp, false);
    }

    pub fn movss_store(&mut self, base: u8, disp: i32, xmm: u8) {
        self.emit_mem(Some(0xF3), false, 0x0F, Some(0x11), xmm, base, disp, false);
    }

    pub fn movsd_load(&mut self, xmm: u8, base: u8, disp: i32) {
        self.emit_mem(Some(0xF2), false, 0x0F, Some(0x10), xmm, base, disp, false);
    }

    pub fn movsd_store(&mut self, base: u8, disp: i32, xmm: u8) {
        self.emit_mem(Some(0xF2), false, 0x0F, Some(0x11), xmm, base, disp, false);
    }

    pub fn addss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x58), dst, src, false);
    }

    pub fn subss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x5C), dst, src, false);
    }

    pub fn mulss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x59), dst, src, false);
    }

    pub fn divss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x5E), dst, src, false);
    }

    pub fn sqrtss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x51), dst, src, false);
    }

    pub fn addsd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x58), dst, src, false);
    }

    pub fn subsd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x5C), dst, src, false);
    }

    pub fn mulsd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x59), dst, src, false);
    }

    pub fn divsd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x5E), dst, src, false);
    }

    pub fn sqrtsd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x51), dst, src, false);
    }

    pub fn xorps(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0x57), dst, src, false);
    }

    pub fn andps(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0x54), dst, src, false);
    }

    pub fn orps(&mut self, dst: u8, src: u8) {
        self.emit_rr(None, false, 0x0F, Some(0x56), dst, src, false);
    }

    pub fn xorpd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x57), dst, src, false);
    }

    pub fn andpd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x54), dst, src, false);
    }

    pub fn orpd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x56), dst, src, false);
    }

    pub fn ucomiss(&mut self, a: u8, b: u8) {
        self.emit_rr(None, false, 0x0F, Some(0x2E), a, b, false);
    }

    pub fn ucomisd(&mut self, a: u8, b: u8) {
        self.emit_rr(Some(0x66), false, 0x0F, Some(0x2E), a, b, false);
    }

    pub fn cvtss2sd(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x5A), dst, src, false);
    }

    pub fn cvtsd2ss(&mut self, dst: u8, src: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x5A), dst, src, false);
    }

    pub fn cvtsi2ss_r32(&mut self, dst: u8, src32: u8) {
        self.emit_rr(Some(0xF3), false, 0x0F, Some(0x2A), dst, src32, false);
    }

    pub fn cvtsi2sd_r32(&mut self, dst: u8, src32: u8) {
        self.emit_rr(Some(0xF2), false, 0x0F, Some(0x2A), dst, src32, false);
    }

    pub fn cvtsi2ss_r64(&mut self, dst: u8, src64: u8) {
        self.emit_rr(Some(0xF3), true, 0x0F, Some(0x2A), dst, src64, false);
    }

    pub fn cvtsi2sd_r64(&mut self, dst: u8, src64: u8) {
        self.emit_rr(Some(0xF2), true, 0x0F, Some(0x2A), dst, src64, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_objdump() -> Option<&'static str> {
        for candidate in ["/usr/bin/objdump", "objdump", "gobjdump"] {
            if let Ok(out) = std::process::Command::new(candidate).arg("--version").output() {
                if out.status.success() && String::from_utf8_lossy(&out.stdout).contains("GNU objdump") {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Disassembles `bytes` with `objdump -D -b binary -m i386:x86-64 -M intel`,
    /// returning `(byte_offset, asm_text)` per decoded instruction.
    fn disasm_x64(objdump: &str, bytes: &[u8], tag: &str) -> Vec<(usize, String)> {
        let bin_path = std::env::temp_dir().join(format!(
            "zakuro-x64-test-{}-{tag}.bin",
            std::process::id()
        ));
        std::fs::write(&bin_path, bytes).unwrap();
        let out = std::process::Command::new(objdump)
            .args([
                "-D",
                "-b",
                "binary",
                "-m",
                "i386:x86-64",
                "-M",
                "intel",
                bin_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        std::fs::remove_file(&bin_path).ok();
        assert!(
            out.status.success(),
            "objdump failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        let mut insns = Vec::new();
        for line in text.lines() {
            if !line.starts_with(' ') {
                continue;
            }
            // GNU objdump wraps instructions > 7 bytes onto a second line with only 2 tab columns.
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 3 {
                continue;
            }
            let addr_hex = cols[0].trim().trim_end_matches(':');
            let offset = usize::from_str_radix(addr_hex, 16)
                .unwrap_or_else(|e| panic!("failed to parse objdump offset {addr_hex:?}: {e}\n{text}"));
            let asm = cols[2..].join("\t").trim().to_string();
            assert!(!asm.contains("(bad)"), "objdump reported (bad) at {offset:#x}: {line}\n{text}");
            insns.push((offset, asm));
        }
        insns
    }

    struct CaseCollector {
        buf: X64Buf,
        cases: Vec<(usize, String)>,
    }

    impl CaseCollector {
        fn new() -> Self {
            Self { buf: X64Buf::with_capacity(4096), cases: Vec::new() }
        }

        fn emit(&mut self, expected: &str, f: impl FnOnce(&mut X64Buf)) {
            let start = self.buf.pos();
            f(&mut self.buf);
            assert!(
                self.buf.pos() > start,
                "expected at least 1 byte for `{expected}`, emitted 0 bytes"
            );
            self.cases.push((start, expected.to_string()));
        }

        fn verify(self, objdump: &str, tag: &str) {
            let disasm = disasm_x64(objdump, self.buf.bytes(), tag);
            assert_eq!(
                disasm.len(),
                self.cases.len(),
                "disasm count mismatch (got {}, expected {})",
                disasm.len(),
                self.cases.len()
            );
            for (i, ((got_off, got_asm), (exp_off, exp_sub))) in
                disasm.iter().zip(self.cases.iter()).enumerate()
            {
                assert_eq!(
                    got_off, exp_off,
                    "insn #{i} offset mismatch (`{got_asm}` vs expected `{exp_sub}`)"
                );
                assert!(
                    got_asm.contains(exp_sub.as_str()),
                    "insn #{i} @ {got_off:#x} mismatch:\n  got:      {got_asm}\n  expected: {exp_sub}"
                );
            }
        }
    }

    #[test]
    fn objdump_verifies_every_instruction_family() {
        let Some(objdump) = find_objdump() else {
            eprintln!("skipping x64 GNU objdump test: GNU objdump not found");
            return;
        };
        let mut c = CaseCollector::new();

        // Stack & control flow
        c.emit("push   rax", |b| b.push_r64(RAX));
        c.emit("push   rsp", |b| b.push_r64(RSP));
        c.emit("push   r12", |b| b.push_r64(R12));
        c.emit("push   r15", |b| b.push_r64(R15));
        c.emit("pop    rbx", |b| b.pop_r64(RBX));
        c.emit("pop    r13", |b| b.pop_r64(R13));
        c.emit("ret", |b| b.ret());
        c.emit("nop", |b| b.nop());
        c.emit("nop", |b| b.nop_n(1));
        c.emit("int3", |b| b.int3());
        c.emit("clc", |b| b.clc());
        c.emit("stc", |b| b.stc());
        c.emit("cmc", |b| b.cmc());
        c.emit("jmp    rax", |b| b.jmp_r64(RAX));
        c.emit("jmp    r11", |b| b.jmp_r64(R11));
        c.emit("call   rcx", |b| b.call_r64(RCX));
        c.emit("call   r14", |b| b.call_r64(R14));
        c.emit("jmp    QWORD PTR [rax]", |b| b.jmp_mem64(RAX, 0));
        c.emit("jmp    QWORD PTR [rsp]", |b| b.jmp_mem64(RSP, 0));
        c.emit("jmp    QWORD PTR [rbp+0x0]", |b| b.jmp_mem64(RBP, 0));
        c.emit("jmp    QWORD PTR [r12+0x8]", |b| b.jmp_mem64(R12, 8));
        c.emit("jmp    QWORD PTR [r13+0x0]", |b| b.jmp_mem64(R13, 0));
        c.emit("call   QWORD PTR [rbx+0x100]", |b| b.call_mem64(RBX, 0x100));
        c.emit("call   QWORD PTR [r12]", |b| b.call_mem64(R12, 0));
        c.emit("call   QWORD PTR [r13-0x8]", |b| b.call_mem64(R13, -8));

        // Moves & extensions (including flag-preserving zero loads and byte reg traps)
        c.emit("mov    rbp,rsp", |b| b.mov_r64_r64(RBP, RSP));
        c.emit("mov    r12,r15", |b| b.mov_r64_r64(R12, R15));
        c.emit("mov    eax,ecx", |b| b.mov_r32_r32(RAX, RCX));
        c.emit("mov    r8d,r15d", |b| b.mov_r32_r32(R8, R15));
        c.emit("mov    eax,0x0", |b| b.mov_r32_imm32(RAX, 0));
        c.emit("mov    r15d,0xdeadbeef", |b| b.mov_r32_imm32(R15, 0xDEAD_BEEF));
        c.emit("mov    eax,0x0", |b| b.mov_r64_imm64(RAX, 0));
        c.emit("mov    r12d,0xffffffff", |b| b.mov_r64_imm64(R12, u32::MAX as u64));
        c.emit("movabs rax,0x100000000", |b| b.mov_r64_imm64(RAX, (u32::MAX as u64) + 1));
        c.emit("mov    r12,0xffffffff80000000", |b| b.mov_r64_imm64(R12, i32::MIN as i64 as u64));
        c.emit("movabs r12,0xffffffff7fffffff", |b| b.mov_r64_imm64(R12, (i32::MIN as i64 - 1) as u64));
        c.emit("mov    r12,0xffffffffffffffff", |b| b.mov_r64_imm64(R12, -1i64 as u64));
        c.emit("movabs r15,0x1122334455667788", |b| b.mov_r64_imm64(R15, 0x1122_3344_5566_7788));

        c.emit("movsx  eax,al", |b| b.movsx_r32_r8(RAX, RAX));
        c.emit("movsx  eax,spl", |b| b.movsx_r32_r8(RAX, RSP));
        c.emit("movsx  eax,bpl", |b| b.movsx_r32_r8(RAX, RBP));
        c.emit("movsx  eax,sil", |b| b.movsx_r32_r8(RAX, RSI));
        c.emit("movsx  r8d,dil", |b| b.movsx_r32_r8(R8, RDI));
        c.emit("movsx  esi,r15b", |b| b.movsx_r32_r8(RSI, R15));
        c.emit("movsx  eax,si", |b| b.movsx_r32_r16(RAX, RSI));
        c.emit("movsx  r12d,r13w", |b| b.movsx_r32_r16(R12, R13));
        c.emit("movzx  eax,spl", |b| b.movzx_r32_r8(RAX, RSP));
        c.emit("movzx  eax,bpl", |b| b.movzx_r32_r8(RAX, RBP));
        c.emit("movzx  eax,sil", |b| b.movzx_r32_r8(RAX, RSI));
        c.emit("movzx  edi,dil", |b| b.movzx_r32_r8(RDI, RDI));
        c.emit("movzx  r9d,r14b", |b| b.movzx_r32_r8(R9, R14));
        c.emit("movzx  eax,di", |b| b.movzx_r32_r16(RAX, RDI));
        c.emit("movzx  r10d,r11w", |b| b.movzx_r32_r16(R10, R11));
        c.emit("movsxd rax,ecx", |b| b.movsxd_r64_r32(RAX, RCX));
        c.emit("movsxd r12,r8d", |b| b.movsxd_r64_r32(R12, R8));
        c.emit("cmove  eax,ecx", |b| b.cmovcc_r32_r32(cc::E, RAX, RCX));
        c.emit("cmovge r8d,r15d", |b| b.cmovcc_r32_r32(cc::GE, R8, R15));
        c.emit("cmovne rdx,rsi", |b| b.cmovcc_r64_r64(cc::NE, RDX, RSI));
        c.emit("cmovl  r12,r13", |b| b.cmovcc_r64_r64(cc::L, R12, R13));
        c.emit("sete   al", |b| b.setcc_r8(cc::E, RAX));
        c.emit("setne  spl", |b| b.setcc_r8(cc::NE, RSP));
        c.emit("setl   bpl", |b| b.setcc_r8(cc::L, RBP));
        c.emit("setge  sil", |b| b.setcc_r8(cc::GE, RSI));
        c.emit("setg   dil", |b| b.setcc_r8(cc::G, RDI));
        c.emit("setb   r8b", |b| b.setcc_r8(cc::B, R8));
        c.emit("setae  r15b", |b| b.setcc_r8(cc::AE, R15));

        // Memory [base + disp] across RSP/RBP/R12/R13 and disp 0/disp8/disp32
        c.emit("mov    rax,QWORD PTR [rsp]", |b| b.ldr_r64_mem(RAX, RSP, 0));
        c.emit("mov    rax,QWORD PTR [rbp+0x0]", |b| b.ldr_r64_mem(RAX, RBP, 0));
        c.emit("mov    rax,QWORD PTR [r12]", |b| b.ldr_r64_mem(RAX, R12, 0));
        c.emit("mov    rax,QWORD PTR [r13+0x0]", |b| b.ldr_r64_mem(RAX, R13, 0));
        c.emit("mov    r14,QWORD PTR [r12-0x80]", |b| b.ldr_r64_mem(R14, R12, -128));
        c.emit("mov    r15,QWORD PTR [r13+0x7f]", |b| b.ldr_r64_mem(R15, R13, 127));
        c.emit("mov    rcx,QWORD PTR [rsp+0x80]", |b| b.ldr_r64_mem(RCX, RSP, 128));
        c.emit("mov    rdx,QWORD PTR [rbp-0x81]", |b| b.ldr_r64_mem(RDX, RBP, -129));
        c.emit("mov    QWORD PTR [r12+0x12345678],r9", |b| b.str_r64_mem(R9, R12, 0x1234_5678));
        c.emit("mov    QWORD PTR [r13+0x0],r10", |b| b.str_r64_mem(R10, R13, 0));
        c.emit("mov    eax,DWORD PTR [rsp+0x10]", |b| b.ldr_r32_mem(RAX, RSP, 16));
        c.emit("mov    r8d,DWORD PTR [r12]", |b| b.ldr_r32_mem(R8, R12, 0));
        c.emit("mov    DWORD PTR [r13+0x0],esi", |b| b.str_r32_mem(RSI, R13, 0));
        c.emit("movzx  eax,BYTE PTR [r12]", |b| b.ldr_u8_mem(RAX, R12, 0));
        c.emit("movsx  r8d,BYTE PTR [r13+0x0]", |b| b.ldr_s8_mem(R8, R13, 0));
        c.emit("movzx  ecx,WORD PTR [rsp+0x4]", |b| b.ldr_u16_mem(RCX, RSP, 4));
        c.emit("movsx  r11d,WORD PTR [rbp+0x0]", |b| b.ldr_s16_mem(R11, RBP, 0));
        c.emit("mov    BYTE PTR [rbp+0x0],sil", |b| b.str_r8_mem(RSI, RBP, 0));
        c.emit("mov    BYTE PTR [r12],dil", |b| b.str_r8_mem(RDI, R12, 0));
        c.emit("mov    BYTE PTR [r13+0x0],spl", |b| b.str_r8_mem(RSP, R13, 0));
        c.emit("mov    BYTE PTR [rsp],r15b", |b| b.str_r8_mem(R15, RSP, 0));
        c.emit("mov    WORD PTR [r12+0x2],si", |b| b.str_r16_mem(RSI, R12, 2));
        c.emit("mov    WORD PTR [r13+0x0],r14w", |b| b.str_r16_mem(R14, R13, 0));
        c.emit("mov    DWORD PTR [rsp],0xdeadbeef", |b| b.str_imm32_mem(RSP, 0, 0xDEAD_BEEF));
        c.emit("mov    DWORD PTR [r13+0x100],0x42", |b| b.str_imm32_mem(R13, 0x100, 0x42));
        c.emit("mov    BYTE PTR [r12],0x7f", |b| b.str_imm8_mem(R12, 0, 0x7F));
        c.emit("mov    BYTE PTR [rbp+0x0],0x1", |b| b.str_imm8_mem(RBP, 0, 1));
        c.emit("lea    rax,[rsp]", |b| b.lea_r64_mem(RAX, RSP, 0));
        c.emit("lea    rbx,[rbp+0x0]", |b| b.lea_r64_mem(RBX, RBP, 0));
        c.emit("lea    r8,[r12]", |b| b.lea_r64_mem(R8, R12, 0));
        c.emit("lea    r9,[r13+0x0]", |b| b.lea_r64_mem(R9, R13, 0));
        c.emit("lea    r10,[rbx+0x200]", |b| b.lea_r64_mem(R10, RBX, 0x200));

        // Memory [base + idx*scale + disp]
        c.emit("mov    rax,QWORD PTR [r12+rax*8]", |b| b.ldr_r64_sib(RAX, R12, RAX, 8, 0));
        c.emit("mov    rax,QWORD PTR [r12+r12*8]", |b| b.ldr_r64_sib(RAX, R12, R12, 8, 0));
        c.emit("mov    rax,QWORD PTR [r13+r13*4+0x0]", |b| b.ldr_r64_sib(RAX, R13, R13, 4, 0));
        c.emit("mov    rax,QWORD PTR [rsp+r12*2]", |b| b.ldr_r64_sib(RAX, RSP, R12, 2, 0));
        c.emit("mov    rcx,QWORD PTR [r13+rax*8+0x0]", |b| b.ldr_r64_sib(RCX, R13, RAX, 8, 0));
        c.emit("mov    QWORD PTR [rbp+r12*4+0x0],r8", |b| b.str_r64_sib(R8, RBP, R12, 4, 0));
        c.emit("mov    edx,DWORD PTR [rsp+r15*2+0x10]", |b| b.ldr_r32_sib(RDX, RSP, R15, 2, 16));
        c.emit("mov    DWORD PTR [r12+rcx*1+0x100],r9d", |b| b.str_r32_sib(R9, R12, RCX, 1, 0x100));
        c.emit("movzx  eax,BYTE PTR [r13+r8*1+0x0]", |b| b.ldr_u8_sib(RAX, R13, R8, 1, 0));
        c.emit("movsx  esi,BYTE PTR [r12+rdx*2]", |b| b.ldr_s8_sib(RSI, R12, RDX, 2, 0));
        c.emit("movzx  edi,WORD PTR [rbp+rax*2+0x4]", |b| b.ldr_u16_sib(RDI, RBP, RAX, 2, 4));
        c.emit("movsx  r10d,WORD PTR [r13+r12*8+0x0]", |b| b.ldr_s16_sib(R10, R13, R12, 8, 0));
        c.emit("mov    BYTE PTR [r12+rax*1],sil", |b| b.str_r8_sib(RSI, R12, RAX, 1, 0));
        c.emit("mov    BYTE PTR [r13+rcx*4+0x0],dil", |b| b.str_r8_sib(RDI, R13, RCX, 4, 0));
        c.emit("mov    WORD PTR [rsp+rax*2],r15w", |b| b.str_r16_sib(R15, RSP, RAX, 2, 0));
        c.emit("lea    rax,[r12+rax*8]", |b| b.lea_r64_sib(RAX, R12, RAX, 8, 0));
        c.emit("lea    rax,[r12+r12*8]", |b| b.lea_r64_sib(RAX, R12, R12, 8, 0));
        c.emit("lea    r11,[r13+r14*4+0x0]", |b| b.lea_r64_sib(R11, R13, R14, 4, 0));

        // Integer ALU
        c.emit("add    eax,ecx", |b| b.alu_r32_r32(AluOp::Add, RAX, RCX));
        c.emit("or     edx,esi", |b| b.alu_r32_r32(AluOp::Or, RDX, RSI));
        c.emit("adc    ebx,edi", |b| b.alu_r32_r32(AluOp::Adc, RBX, RDI));
        c.emit("sbb    r8d,r9d", |b| b.alu_r32_r32(AluOp::Sbb, R8, R9));
        c.emit("and    r10d,r11d", |b| b.alu_r32_r32(AluOp::And, R10, R11));
        c.emit("sub    r12d,r13d", |b| b.alu_r32_r32(AluOp::Sub, R12, R13));
        c.emit("xor    r14d,r15d", |b| b.alu_r32_r32(AluOp::Xor, R14, R15));
        c.emit("cmp    eax,r15d", |b| b.alu_r32_r32(AluOp::Cmp, RAX, R15));

        c.emit("add    eax,0x7f", |b| b.alu_r32_imm32(AluOp::Add, RAX, 127));
        c.emit("sub    r8d,0xffffff80", |b| b.alu_r32_imm32(AluOp::Sub, R8, -128));
        c.emit("cmp    r12d,0x80", |b| b.alu_r32_imm32(AluOp::Cmp, R12, 128));
        c.emit("and    esi,0x12345678", |b| b.alu_r32_imm32(AluOp::And, RSI, 0x1234_5678));

        c.emit("add    rax,r12", |b| b.alu_r64_r64(AluOp::Add, RAX, R12));
        c.emit("or     rbx,r8", |b| b.alu_r64_r64(AluOp::Or, RBX, R8));
        c.emit("adc    rcx,r9", |b| b.alu_r64_r64(AluOp::Adc, RCX, R9));
        c.emit("sbb    rdx,r10", |b| b.alu_r64_r64(AluOp::Sbb, RDX, R10));
        c.emit("and    rsi,r11", |b| b.alu_r64_r64(AluOp::And, RSI, R11));
        c.emit("sub    r15,r13", |b| b.alu_r64_r64(AluOp::Sub, R15, R13));
        c.emit("xor    r14,rdi", |b| b.alu_r64_r64(AluOp::Xor, R14, RDI));
        c.emit("cmp    rsp,rbp", |b| b.alu_r64_r64(AluOp::Cmp, RSP, RBP));
        c.emit("add    rsp,0xfffffffffffffff0", |b| b.alu_r64_imm32(AluOp::Add, RSP, -16));
        c.emit("adc    r8,0x7f", |b| b.alu_r64_imm32(AluOp::Adc, R8, 127));
        c.emit("sbb    r9,0xffffffffffffff80", |b| b.alu_r64_imm32(AluOp::Sbb, R9, -128));
        c.emit("xor    r12,0x1000", |b| b.alu_r64_imm32(AluOp::Xor, R12, 0x1000));
        c.emit("cmp    r13,0xffffffffffffff7f", |b| b.alu_r64_imm32(AluOp::Cmp, R13, -129));

        c.emit("test   eax,r8d", |b| b.test_r32_r32(RAX, R8));
        c.emit("test   r12,r13", |b| b.test_r64_r64(R12, R13));
        c.emit("test   r9d,0xf0000000", |b| b.test_r32_imm32(R9, 0xF000_0000));
        c.emit("test   al,0x1", |b| b.test_r8_imm8(RAX, 1));
        c.emit("test   spl,0x2", |b| b.test_r8_imm8(RSP, 2));
        c.emit("test   bpl,0x4", |b| b.test_r8_imm8(RBP, 4));
        c.emit("test   sil,0x8", |b| b.test_r8_imm8(RSI, 8));
        c.emit("test   dil,0x10", |b| b.test_r8_imm8(RDI, 0x10));
        c.emit("test   r8b,0x20", |b| b.test_r8_imm8(R8, 0x20));
        c.emit("test   r15b,0x80", |b| b.test_r8_imm8(R15, 0x80));

        c.emit("neg    eax", |b| b.neg_r32(RAX));
        c.emit("neg    r14d", |b| b.neg_r32(R14));
        c.emit("not    ecx", |b| b.not_r32(RCX));
        c.emit("not    r15d", |b| b.not_r32(R15));
        c.emit("imul   eax,ecx", |b| b.imul_r32_r32(RAX, RCX));
        c.emit("imul   r8d,r15d", |b| b.imul_r32_r32(R8, R15));
        c.emit("imul   r12,r13", |b| b.imul_r64_r64(R12, R13));
        c.emit("mul    ecx", |b| b.mul_rax_r32(RCX));
        c.emit("imul   r8d", |b| b.imul_rax_r32(R8));
        c.emit("mul    rcx", |b| b.mul_rax_r64(RCX));
        c.emit("imul   r12", |b| b.imul_rax_r64(R12));

        c.emit("rol    eax,1", |b| b.shift_r32_imm(ShiftOp::Rol, RAX, 1));
        c.emit("ror    r8d,0x5", |b| b.shift_r32_imm(ShiftOp::Ror, R8, 5));
        c.emit("shl    ecx,1", |b| b.shift_r32_imm(ShiftOp::Shl, RCX, 1));
        c.emit("shr    edx,0xc", |b| b.shift_r32_imm(ShiftOp::Shr, RDX, 12));
        c.emit("sar    r15d,0x1f", |b| b.shift_r32_imm(ShiftOp::Sar, R15, 31));
        c.emit("rol    ebx,cl", |b| b.shift_r32_cl(ShiftOp::Rol, RBX));
        c.emit("ror    esi,cl", |b| b.shift_r32_cl(ShiftOp::Ror, RSI));
        c.emit("shl    eax,cl", |b| b.shift_r32_cl(ShiftOp::Shl, RAX));
        c.emit("shr    r8d,cl", |b| b.shift_r32_cl(ShiftOp::Shr, R8));
        c.emit("sar    r12d,cl", |b| b.shift_r32_cl(ShiftOp::Sar, R12));
        c.emit("rol    r9,0x7", |b| b.shift_r64_imm(ShiftOp::Rol, R9, 7));
        c.emit("ror    r10,1", |b| b.shift_r64_imm(ShiftOp::Ror, R10, 1));
        c.emit("shl    rax,1", |b| b.shift_r64_imm(ShiftOp::Shl, RAX, 1));
        c.emit("shr    r13,0x1c", |b| b.shift_r64_imm(ShiftOp::Shr, R13, 28));
        c.emit("sar    r11,0x3f", |b| b.shift_r64_imm(ShiftOp::Sar, R11, 63));
        c.emit("ror    r14,cl", |b| b.shift_r64_cl(ShiftOp::Ror, R14));
        c.emit("sar    r15,cl", |b| b.shift_r64_cl(ShiftOp::Sar, R15));

        c.emit("bswap  eax", |b| b.bswap_r32(RAX));
        c.emit("bswap  r15d", |b| b.bswap_r32(R15));
        c.emit("bsr    eax,ecx", |b| b.bsr_r32(RAX, RCX));
        c.emit("bsr    r8d,r15d", |b| b.bsr_r32(R8, R15));
        c.emit("bt     eax,0x1f", |b| b.bt_r32_imm8(RAX, 31));
        c.emit("bts    r12d,0x5", |b| b.bts_r32_imm8(R12, 5));
        c.emit("btr    r15d,0x0", |b| b.btr_r32_imm8(R15, 0));

        // SSE2 scalar float / double
        c.emit("movd   xmm0,eax", |b| b.movd_xmm_r32(XMM0, RAX));
        c.emit("movd   xmm15,r12d", |b| b.movd_xmm_r32(XMM15, R12));
        c.emit("movd   eax,xmm0", |b| b.movd_r32_xmm(RAX, XMM0));
        c.emit("movd   r13d,xmm9", |b| b.movd_r32_xmm(R13, XMM9));
        c.emit("movq   xmm0,rax", |b| b.movq_xmm_r64(XMM0, RAX));
        c.emit("movq   xmm14,r15", |b| b.movq_xmm_r64(XMM14, R15));
        c.emit("movq   rax,xmm0", |b| b.movq_r64_xmm(RAX, XMM0));
        c.emit("movq   r8,xmm15", |b| b.movq_r64_xmm(R8, XMM15));
        c.emit("movss  xmm1,xmm10", |b| b.movss_xmm_xmm(XMM1, XMM10));
        c.emit("movsd  xmm11,xmm2", |b| b.movsd_xmm_xmm(XMM11, XMM2));
        c.emit("movss  xmm0,DWORD PTR [rsp]", |b| b.movss_load(XMM0, RSP, 0));
        c.emit("movss  xmm8,DWORD PTR [rbp+0x0]", |b| b.movss_load(XMM8, RBP, 0));
        c.emit("movss  DWORD PTR [r12],xmm0", |b| b.movss_store(R12, 0, XMM0));
        c.emit("movss  DWORD PTR [r13+0x0],xmm12", |b| b.movss_store(R13, 0, XMM12));
        c.emit("movsd  xmm3,QWORD PTR [rbx+0x200]", |b| b.movsd_load(XMM3, RBX, 0x200));
        c.emit("movsd  QWORD PTR [rsp+0x8],xmm15", |b| b.movsd_store(RSP, 8, XMM15));

        c.emit("addss  xmm0,xmm1", |b| b.addss(XMM0, XMM1));
        c.emit("subss  xmm8,xmm9", |b| b.subss(XMM8, XMM9));
        c.emit("mulss  xmm2,xmm10", |b| b.mulss(XMM2, XMM10));
        c.emit("divss  xmm11,xmm3", |b| b.divss(XMM11, XMM3));
        c.emit("sqrtss xmm4,xmm12", |b| b.sqrtss(XMM4, XMM12));
        c.emit("addsd  xmm0,xmm1", |b| b.addsd(XMM0, XMM1));
        c.emit("subsd  xmm8,xmm9", |b| b.subsd(XMM8, XMM9));
        c.emit("mulsd  xmm2,xmm10", |b| b.mulsd(XMM2, XMM10));
        c.emit("divsd  xmm11,xmm3", |b| b.divsd(XMM11, XMM3));
        c.emit("sqrtsd xmm4,xmm12", |b| b.sqrtsd(XMM4, XMM12));

        c.emit("xorps  xmm0,xmm8", |b| b.xorps(XMM0, XMM8));
        c.emit("andps  xmm9,xmm1", |b| b.andps(XMM9, XMM1));
        c.emit("orps   xmm10,xmm11", |b| b.orps(XMM10, XMM11));
        c.emit("xorpd  xmm0,xmm8", |b| b.xorpd(XMM0, XMM8));
        c.emit("andpd  xmm9,xmm1", |b| b.andpd(XMM9, XMM1));
        c.emit("orpd   xmm10,xmm11", |b| b.orpd(XMM10, XMM11));
        c.emit("ucomiss xmm0,xmm15", |b| b.ucomiss(XMM0, XMM15));
        c.emit("ucomisd xmm14,xmm2", |b| b.ucomisd(XMM14, XMM2));
        c.emit("cvtss2sd xmm0,xmm9", |b| b.cvtss2sd(XMM0, XMM9));
        c.emit("cvtsd2ss xmm10,xmm1", |b| b.cvtsd2ss(XMM10, XMM1));
        c.emit("cvtsi2ss xmm0,eax", |b| b.cvtsi2ss_r32(XMM0, RAX));
        c.emit("cvtsi2sd xmm8,r12d", |b| b.cvtsi2sd_r32(XMM8, R12));
        c.emit("cvtsi2ss xmm1,rcx", |b| b.cvtsi2ss_r64(XMM1, RCX));
        c.emit("cvtsi2sd xmm15,r13", |b| b.cvtsi2sd_r64(XMM15, R13));

        c.verify(objdump, "all-families");
    }

    #[test]
    fn branch_patching_forward_and_backward() {
        let Some(objdump) = find_objdump() else {
            eprintln!("skipping x64 GNU objdump test: GNU objdump not found");
            return;
        };
        let mut b = X64Buf::default();
        let entry = b.pos();
        let j_fwd = b.jmp_rel32();
        let jcc_fwd = b.jcc_rel32(cc::NE);
        let call_fwd = b.call_rel32();
        let target_fwd = b.pos();
        b.nop();
        b.patch_rel32(j_fwd, target_fwd);
        b.patch_rel32(jcc_fwd, target_fwd);
        b.patch_rel32(call_fwd, target_fwd);

        b.jmp_to(entry);
        b.jcc_to(cc::E, entry);
        b.call_to(target_fwd);

        let disasm = disasm_x64(objdump, b.bytes(), "branch-patch");
        assert_eq!(disasm.len(), 7);
        assert_eq!(disasm[0], (0x00, format!("jmp    {target_fwd:#x}")));
        assert_eq!(disasm[1], (0x05, format!("jne    {target_fwd:#x}")));
        assert_eq!(disasm[2], (0x0b, format!("call   {target_fwd:#x}")));
        assert_eq!(disasm[3], (target_fwd, "nop".to_string()));
        assert_eq!(disasm[4], (0x11, format!("jmp    {entry:#x}")));
        assert_eq!(disasm[5], (0x16, format!("je     {entry:#x}")));
        assert_eq!(disasm[6], (0x1c, format!("call   {target_fwd:#x}")));
    }

    #[test]
    fn r8_operands_never_encode_high_byte_regs() {
        let Some(objdump) = find_objdump() else {
            eprintln!("skipping x64 GNU objdump test: GNU objdump not found");
            return;
        };
        let mut b = X64Buf::default();
        for r in 0..16 {
            b.setcc_r8(cc::E, r);
            b.movsx_r32_r8(RAX, r);
            b.movzx_r32_r8(R15, r);
            b.str_r8_mem(r, RBP, 0);
            b.str_r8_sib(r, R12, RAX, 4, 0);
            b.test_r8_imm8(r, 0x5A);
        }
        let disasm = disasm_x64(objdump, b.bytes(), "r8-no-high-byte");
        assert_eq!(disasm.len(), 16 * 6);
        for (off, asm) in &disasm {
            for tok in asm.split(|c: char| !c.is_ascii_alphanumeric()) {
                assert!(
                    !matches!(tok, "ah" | "ch" | "dh" | "bh"),
                    "encoded high-byte register `{tok}` at {off:#x}: {asm}"
                );
            }
        }
    }

    #[test]
    fn flag_invariants_and_zero_shift_and_exact_traps() {
        // mov_r32_imm32(..., 0) and mov_r64_imm64(..., 0) must use 0xB8 + r, never xor.
        for r in 0..16 {
            let mut b32 = X64Buf::default();
            b32.mov_r32_imm32(r, 0);
            let expected32: Vec<u8> = if r >= 8 {
                vec![0x41, 0xB8 | (r & 7), 0, 0, 0, 0]
            } else {
                vec![0xB8 | r, 0, 0, 0, 0]
            };
            assert_eq!(b32.bytes(), expected32.as_slice());

            let mut b64 = X64Buf::default();
            b64.mov_r64_imm64(r, 0);
            assert_eq!(b64.bytes(), expected32.as_slice());
        }

        // Shift by 0 must emit nothing.
        let mut b = X64Buf::default();
        for op in [ShiftOp::Rol, ShiftOp::Ror, ShiftOp::Shl, ShiftOp::Shr, ShiftOp::Sar] {
            b.shift_r32_imm(op, RAX, 0);
            b.shift_r64_imm(op, R15, 0);
        }
        assert!(b.is_empty());

        // Exact encoding of RSP/R12 SIB 0x24 and RBP/R13 disp==0 -> mod=01 disp8=0.
        let mut t = X64Buf::default();
        t.ldr_r64_mem(RAX, RSP, 0);
        assert_eq!(t.len(), 4);
        assert_eq!(t.bytes(), &[0x48, 0x8B, 0x04, 0x24]);

        t.bytes.clear();
        t.ldr_r64_mem(RAX, R12, 0);
        assert_eq!(t.bytes(), &[0x49, 0x8B, 0x04, 0x24]);

        t.bytes.clear();
        t.ldr_r64_mem(RAX, RBP, 0);
        assert_eq!(t.bytes(), &[0x48, 0x8B, 0x45, 0x00]);

        t.bytes.clear();
        t.ldr_r64_mem(RAX, R13, 0);
        assert_eq!(t.bytes(), &[0x49, 0x8B, 0x45, 0x00]);

        t.bytes.clear();
        t.ldr_r64_sib(RAX, R12, RAX, 8, 0);
        assert_eq!(t.bytes(), &[0x49, 0x8B, 0x04, 0xC4]);

        t.bytes.clear();
        t.ldr_r64_sib(RAX, R13, RAX, 8, 0);
        assert_eq!(t.bytes(), &[0x49, 0x8B, 0x44, 0xC5, 0x00]);

        // Verify all 16 condition codes invert cleanly and encode without bad opcodes.
        for cond in 0..16u8 {
            assert_eq!(cc::invert(cc::invert(cond)), cond);
            t.setcc_r8(cond, RSI);
            t.cmovcc_r32_r32(cond, R8, R9);
            t.cmovcc_r64_r64(cond, R12, R13);
        }
        assert!(!t.into_bytes().is_empty());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "SIB index cannot be RSP")]
    fn sib_disallows_rsp_index() {
        let mut b = X64Buf::default();
        b.ldr_r64_sib(RAX, RBX, RSP, 4, 0);
    }
}
