//! Zero-dependency x86_64 machine-code instruction encoder for `zakuro-aot`.

#![allow(dead_code, unused_variables)]

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

    // Stack & control flow
    pub fn push_r64(&mut self, r: u8) { unimplemented!() }
    pub fn pop_r64(&mut self, r: u8) { unimplemented!() }
    pub fn ret(&mut self) { unimplemented!() }
    pub fn nop(&mut self) { unimplemented!() }
    pub fn nop_n(&mut self, count: usize) { for _ in 0..count { self.bytes.push(0x90); } }
    pub fn int3(&mut self) { self.bytes.push(0xCC); }
    pub fn clc(&mut self) { self.bytes.push(0xF8); }
    pub fn stc(&mut self) { self.bytes.push(0xF9); }
    pub fn cmc(&mut self) { self.bytes.push(0xF5); }

    pub fn jmp_rel32(&mut self) -> usize { unimplemented!() }
    pub fn jmp_to(&mut self, target: usize) { unimplemented!() }
    pub fn jcc_rel32(&mut self, cond: u8) -> usize { unimplemented!() }
    pub fn jcc_to(&mut self, cond: u8, target: usize) { unimplemented!() }
    pub fn call_rel32(&mut self) -> usize { unimplemented!() }
    pub fn call_to(&mut self, target: usize) { unimplemented!() }
    pub fn jmp_r64(&mut self, r: u8) { unimplemented!() }
    pub fn call_r64(&mut self, r: u8) { unimplemented!() }
    pub fn jmp_mem64(&mut self, base: u8, disp: i32) { unimplemented!() }
    pub fn call_mem64(&mut self, base: u8, disp: i32) { unimplemented!() }

    // Moves & extensions (NEVER clobber flags!)
    pub fn mov_r64_r64(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn mov_r32_r32(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn mov_r32_imm32(&mut self, dst: u8, imm: u32) { unimplemented!() }
    pub fn mov_r64_imm64(&mut self, dst: u8, imm: u64) { unimplemented!() }
    pub fn movsx_r32_r8(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movsx_r32_r16(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movzx_r32_r8(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movzx_r32_r16(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movsxd_r64_r32(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn cmovcc_r32_r32(&mut self, cond: u8, dst: u8, src: u8) { unimplemented!() }
    pub fn cmovcc_r64_r64(&mut self, cond: u8, dst: u8, src: u8) { unimplemented!() }
    pub fn setcc_r8(&mut self, cond: u8, dst: u8) { unimplemented!() }

    // Memory [base + disp]
    pub fn ldr_r64_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn str_r64_mem(&mut self, src: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn ldr_r32_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn str_r32_mem(&mut self, src: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn ldr_u8_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn ldr_s8_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn ldr_u16_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn ldr_s16_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn str_r8_mem(&mut self, src: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn str_r16_mem(&mut self, src: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn str_imm32_mem(&mut self, base: u8, disp: i32, imm: u32) { unimplemented!() }
    pub fn str_imm8_mem(&mut self, base: u8, disp: i32, imm: u8) { unimplemented!() }
    pub fn lea_r64_mem(&mut self, dst: u8, base: u8, disp: i32) { unimplemented!() }

    // Memory [base + idx*scale + disp] (`scale` in {1, 2, 4, 8})
    pub fn ldr_r64_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn str_r64_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn ldr_r32_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn str_r32_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn ldr_u8_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn ldr_s8_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn ldr_u16_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn ldr_s16_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn str_r8_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn str_r16_sib(&mut self, src: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }
    pub fn lea_r64_sib(&mut self, dst: u8, base: u8, idx: u8, scale: u8, disp: i32) { unimplemented!() }

    // Integer ALU
    pub fn alu_r32_r32(&mut self, op: AluOp, dst: u8, src: u8) { unimplemented!() }
    pub fn alu_r32_imm32(&mut self, op: AluOp, dst: u8, imm: i32) { unimplemented!() }
    pub fn alu_r64_r64(&mut self, op: AluOp, dst: u8, src: u8) { unimplemented!() }
    pub fn alu_r64_imm32(&mut self, op: AluOp, dst: u8, imm: i32) { unimplemented!() }
    pub fn test_r32_r32(&mut self, a: u8, b: u8) { unimplemented!() }
    pub fn test_r64_r64(&mut self, a: u8, b: u8) { unimplemented!() }
    pub fn test_r32_imm32(&mut self, a: u8, imm: u32) { unimplemented!() }
    pub fn test_r8_imm8(&mut self, a: u8, imm: u8) { unimplemented!() }
    pub fn neg_r32(&mut self, r: u8) { unimplemented!() }
    pub fn not_r32(&mut self, r: u8) { unimplemented!() }
    pub fn imul_r32_r32(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn imul_r64_r64(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn mul_rax_r32(&mut self, src: u8) { unimplemented!() }
    pub fn imul_rax_r32(&mut self, src: u8) { unimplemented!() }
    pub fn mul_rax_r64(&mut self, src: u8) { unimplemented!() }
    pub fn imul_rax_r64(&mut self, src: u8) { unimplemented!() }
    pub fn shift_r32_imm(&mut self, op: ShiftOp, dst: u8, imm: u8) { unimplemented!() }
    pub fn shift_r32_cl(&mut self, op: ShiftOp, dst: u8) { unimplemented!() }
    pub fn shift_r64_imm(&mut self, op: ShiftOp, dst: u8, imm: u8) { unimplemented!() }
    pub fn shift_r64_cl(&mut self, op: ShiftOp, dst: u8) { unimplemented!() }
    pub fn bswap_r32(&mut self, r: u8) { unimplemented!() }
    pub fn bsr_r32(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn bt_r32_imm8(&mut self, r: u8, bit: u8) { unimplemented!() }
    pub fn bts_r32_imm8(&mut self, r: u8, bit: u8) { unimplemented!() }
    pub fn btr_r32_imm8(&mut self, r: u8, bit: u8) { unimplemented!() }

    // SSE2 scalar float / double
    pub fn movd_xmm_r32(&mut self, xmm: u8, r32: u8) { unimplemented!() }
    pub fn movd_r32_xmm(&mut self, r32: u8, xmm: u8) { unimplemented!() }
    pub fn movq_xmm_r64(&mut self, xmm: u8, r64: u8) { unimplemented!() }
    pub fn movq_r64_xmm(&mut self, r64: u8, xmm: u8) { unimplemented!() }
    pub fn movss_xmm_xmm(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movsd_xmm_xmm(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn movss_load(&mut self, xmm: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn movss_store(&mut self, base: u8, disp: i32, xmm: u8) { unimplemented!() }
    pub fn movsd_load(&mut self, xmm: u8, base: u8, disp: i32) { unimplemented!() }
    pub fn movsd_store(&mut self, base: u8, disp: i32, xmm: u8) { unimplemented!() }
    pub fn addss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn subss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn mulss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn divss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn sqrtss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn addsd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn subsd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn mulsd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn divsd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn sqrtsd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn xorps(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn andps(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn orps(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn xorpd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn andpd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn orpd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn ucomiss(&mut self, a: u8, b: u8) { unimplemented!() }
    pub fn ucomisd(&mut self, a: u8, b: u8) { unimplemented!() }
    pub fn cvtss2sd(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn cvtsd2ss(&mut self, dst: u8, src: u8) { unimplemented!() }
    pub fn cvtsi2ss_r32(&mut self, dst: u8, src32: u8) { unimplemented!() }
    pub fn cvtsi2sd_r32(&mut self, dst: u8, src32: u8) { unimplemented!() }
    pub fn cvtsi2ss_r64(&mut self, dst: u8, src64: u8) { unimplemented!() }
    pub fn cvtsi2sd_r64(&mut self, dst: u8, src64: u8) { unimplemented!() }
}
