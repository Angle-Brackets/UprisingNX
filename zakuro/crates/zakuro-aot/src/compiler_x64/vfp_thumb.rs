//! x86_64 AOT lowering for VFPv2 floating-point and Thumb-1 instructions.

#![allow(dead_code)]

use crate::discover::Mode;
use crate::runtime::offsets;
use crate::x64::{
    AluOp, ShiftOp, cc, RAX, RBX, RCX, RDX, RSI, R8, R10, R12, R13, R15, XMM0, XMM1, XMM2,
};

use super::{FnEmitter, SpanWord, StubId};

const RECOMP_DEPTH_LIMIT: i32 = 2048;

impl FnEmitter<'_> {
    pub(crate) fn vfp_data_processing(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        if coproc != 10 && coproc != 11 {
            return self.emit_interpret(a, op);
        }
        let double = coproc == 11;
        let (vd, vn, vm) = ((op >> 12) & 0xF, (op >> 16) & 0xF, op & 0xF);
        let (d_bit, n_bit, m_bit) = (
            (op & (1 << 22)) != 0,
            (op & (1 << 7)) != 0,
            (op & (1 << 5)) != 0,
        );
        let (rd, rn, rm) = if double {
            (dreg(vd, d_bit), dreg(vn, n_bit), dreg(vm, m_bit))
        } else {
            (sreg(vd, d_bit), sreg(vn, n_bit), sreg(vm, m_bit))
        };
        let negate = (op & (1 << 6)) != 0;
        let vop = match ((op >> 20) & 0b1011, negate) {
            (0b0000, false) => 0, // VMLA:  acc + a*b
            (0b0000, true) => 1,  // VMLS:  acc - a*b
            (0b0001, false) => 2, // VNMLS: -acc + a*b = (a*b) - acc
            (0b0001, true) => 3,  // VNMLA: -acc - a*b = -(a*b) - acc
            (0b0010, false) => 4, // VMUL:  a * b
            (0b0010, true) => 5,  // VNMUL: -(a * b)
            (0b0011, false) => 6, // VADD:  a + b
            (0b0011, true) => 7,  // VSUB:  a - b
            (0b1000, false) => 8, // VDIV:  a / b
            (0b1011, true) => return vfp_extension(self, a, op, double, rd, rm),
            _ => return self.emit_interpret(a, op),
        };

        vfp_maybe_vector(self, double, vop, rd, rn, rm, |em| {
            em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
            let disp = |r: u8| -> i32 { (r as i32) * if double { 8 } else { 4 } };
            emit_vfp_load(em, double, XMM0, RAX, disp(rn));
            emit_vfp_load(em, double, XMM1, RAX, disp(rm));
            match vop {
                4 => emit_vfp_mul(em, double, XMM0, XMM1),
                5 => {
                    emit_vfp_mul(em, double, XMM0, XMM1);
                    emit_vfp_negate(em, double, XMM0);
                }
                6 => emit_vfp_add(em, double, XMM0, XMM1),
                7 => emit_vfp_sub(em, double, XMM0, XMM1),
                8 => emit_vfp_div(em, double, XMM0, XMM1),
                0..=3 => {
                    // Non-FMA chained VFPv2 multiply-accumulate: round product first, then add/sub.
                    emit_vfp_mul(em, double, XMM0, XMM1);
                    if vop == 3 {
                        emit_vfp_negate(em, double, XMM0);
                    }
                    emit_vfp_load(em, double, XMM1, RAX, disp(rd));
                    match vop {
                        0 => {
                            emit_vfp_add(em, double, XMM1, XMM0);
                            emit_vfp_store(em, double, RAX, disp(rd), XMM1);
                        }
                        1 => {
                            emit_vfp_sub(em, double, XMM1, XMM0);
                            emit_vfp_store(em, double, RAX, disp(rd), XMM1);
                        }
                        _ => {
                            emit_vfp_sub(em, double, XMM0, XMM1);
                            emit_vfp_store(em, double, RAX, disp(rd), XMM0);
                        }
                    }
                    return;
                }
                _ => unreachable!(),
            }
            emit_vfp_store(em, double, RAX, disp(rd), XMM0);
        })
    }

    pub(crate) fn vfp_register_transfer(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        let load = (op & (1 << 20)) != 0;
        let rt = (op >> 12) & 0xF;
        match (coproc, (op >> 21) & 7) {
            (10 | 11, 7) if ((op >> 16) & 0xF) == 1 => {
                if !load {
                    // VMSR FPSCR, Rt — update *ctx.fpscr and sync MXCSR FTZ/DAZ/RC!
                    self.load_gpr(RAX, rt, a + 8);
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_FPSCR as i32);
                    self.buf.str_r32_mem(RAX, RCX, 0);

                    emit_stmxcsr_scratch(self);
                    self.buf.ldr_r32_mem(RCX, RBX, offsets::SCRATCH_FP as i32);
                    // Clear FTZ (15), RC (14:13), DAZ (6)
                    self.buf.alu_r32_imm32(AluOp::And, RCX, !0xE040u32 as i32);
                    // Set FTZ | DAZ (0x8040) when FPSCR.FZ (bit 24) is set
                    self.buf.mov_r32_imm32(RDX, 0x8040);
                    self.buf.mov_r32_imm32(RSI, 0);
                    self.buf.test_r32_imm32(RAX, 1 << 24);
                    self.buf.cmovcc_r32_r32(cc::Z, RDX, RSI);
                    self.buf.alu_r32_r32(AluOp::Or, RCX, RDX);
                    // Map ARM RMode FPSCR[23:22] (00=RN, 01=RP, 10=RM, 11=RZ)
                    // to x86 MXCSR RC[14:13]   (00=RN, 01=RM, 10=RP, 11=RZ) by swapping the 2 bits.
                    self.buf.mov_r32_r32(RDX, RAX);
                    self.buf.shift_r32_imm(ShiftOp::Shr, RDX, 22);
                    self.buf.alu_r32_imm32(AluOp::And, RDX, 3);
                    self.buf.mov_r32_r32(RSI, RDX);
                    self.buf.alu_r32_imm32(AluOp::And, RDX, 1);
                    self.buf.shift_r32_imm(ShiftOp::Shl, RDX, 1);
                    self.buf.shift_r32_imm(ShiftOp::Shr, RSI, 1);
                    self.buf.alu_r32_r32(AluOp::Or, RDX, RSI);
                    self.buf.shift_r32_imm(ShiftOp::Shl, RDX, 13);
                    self.buf.alu_r32_r32(AluOp::Or, RCX, RDX);

                    self.buf.str_r32_mem(RCX, RBX, offsets::SCRATCH_FP as i32);
                    emit_ldmxcsr_scratch(self);
                } else if rt == 15 {
                    // VMRS APSR_nzcv, FPSCR (`FMSTAT`)
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_FPSCR as i32);
                    self.buf.ldr_r32_mem(RAX, RCX, 0);
                    self.buf.alu_r32_imm32(AluOp::And, RAX, 0xF000_0000u32 as i32);
                    self.buf.mov_r32_r32(R15, RAX);
                } else {
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_FPSCR as i32);
                    self.buf.ldr_r32_mem(RAX, RCX, 0);
                    self.store_gpr(rt, RAX);
                }
                true
            }
            (10 | 11, 0) if rt != 15 => {
                let sn = sreg((op >> 16) & 0xF, (op & (1 << 7)) != 0);
                self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
                if load {
                    self.buf.ldr_r32_mem(RAX, RCX, (sn as i32) * 4);
                    self.store_gpr(rt, RAX);
                } else {
                    self.load_gpr(RAX, rt, a + 8);
                    self.buf.str_r32_mem(RAX, RCX, (sn as i32) * 4);
                }
                true
            }
            _ => self.emit_interpret(a, op),
        }
    }

    pub(crate) fn vfp_load_store(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        if coproc != 10 && coproc != 11 {
            return self.emit_interpret(a, op);
        }
        let double = coproc == 11;
        if ((op >> 21) & 0x7F) == 0b110_0010 {
            return vfp_move_pair(self, a, op, double);
        }
        let load = (op & (1 << 20)) != 0;
        let pre = (op & (1 << 24)) != 0;
        let up = (op & (1 << 23)) != 0;
        let writeback = (op & (1 << 21)) != 0;
        let rn = (op >> 16) & 0xF;
        if writeback && rn == 15 {
            return self.emit_interpret(a, op);
        }
        let offset = (op & 0xFF) * 4;
        let (vd, d_bit) = ((op >> 12) & 0xF, (op & (1 << 22)) != 0);
        let first = if double {
            dreg(vd, d_bit) as u32
        } else {
            sreg(vd, d_bit) as u32
        };
        let count = if pre && !writeback {
            1
        } else if double {
            (op & 0xFF) / 2
        } else {
            op & 0xFF
        };

        // PC-relative `.text` float/double literal pool folding (`VLDR Sd/Dd, [PC, ±#imm]`):
        if self.options.fold_text_literals && load && rn == 15 && pre && !writeback {
            let lit = if up {
                (a + 8).wrapping_add(offset)
            } else {
                (a + 8).wrapping_sub(offset)
            };
            let bytes = if double { 8u32 } else { 4u32 };
            if lit >= self.text.base && lit.saturating_add(bytes) <= self.text.end() {
                if !double {
                    if let Some(bits) = self.text.read32(lit) {
                        self.folded_spans.push((lit, lit + 4));
                        let sd = first as u8;
                        self.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
                        self.buf.str_imm32_mem(RAX, (sd as i32) * 4, bits);
                        return true;
                    }
                } else if let (Some(lo), Some(hi)) =
                    (self.text.read32(lit), self.text.read32(lit + 4))
                {
                    self.folded_spans.push((lit, lit + 8));
                    let s_lo = ((first * 2) & 31) as u8;
                    let s_hi = ((first * 2 + 1) & 31) as u8;
                    self.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
                    self.buf.str_imm32_mem(RAX, (s_lo as i32) * 4, lo);
                    self.buf.str_imm32_mem(RAX, (s_hi as i32) * 4, hi);
                    return true;
                }
            }
        }

        self.load_gpr(R8, rn, a + 8);
        let p_delta = match (up, pre && !writeback) {
            (true, true) => offset,
            (true, false) => 0,
            (false, _) => offset.wrapping_neg(),
        };
        if p_delta != 0 {
            self.buf.alu_r32_imm32(AluOp::Add, R8, p_delta as i32);
        }

        let words: Vec<SpanWord> = (0..count.max(1))
            .flat_map(|i| {
                if double {
                    vec![(first + i) * 2, (first + i) * 2 + 1]
                } else {
                    vec![first + i]
                }
            })
            .map(|w| SpanWord::Vfp((w & 31) as u8))
            .collect();
        emit_span(self, load, &words);

        if writeback {
            let fin_delta = if up { offset } else { offset.wrapping_neg() };
            let wb_delta = fin_delta.wrapping_sub(p_delta);
            if wb_delta != 0 {
                self.buf.alu_r32_imm32(AluOp::Add, R8, wb_delta as i32);
            }
            self.store_gpr(rn, R8);
        }
        true
    }

    pub(crate) fn lower_thumb(&mut self, a: u32, op: u32, second: u32) -> bool {
        match op >> 13 {
            0b000 if ((op >> 11) & 3) == 0b11 => {
                // ADD / SUB 3-reg or 3-bit imm
                let field = (op >> 6) & 7;
                let rn = (op >> 3) & 7;
                let rd = op & 7;
                let sub = (op & (1 << 9)) != 0;
                self.load_gpr(RAX, rn, a + 4);
                if (op & (1 << 10)) != 0 {
                    let op_kind = if sub { AluOp::Sub } else { AluOp::Add };
                    self.buf.alu_r32_imm32(op_kind, RAX, field as i32);
                } else {
                    self.load_gpr(RCX, field, a + 4);
                    let op_kind = if sub { AluOp::Sub } else { AluOp::Add };
                    self.buf.alu_r32_r32(op_kind, RAX, RCX);
                }
                self.store_gpr(rd, RAX);
                if sub {
                    self.set_nzcv_sub();
                } else {
                    self.set_nzcv_add();
                }
                true
            }
            0b000 => {
                // LSL / LSR / ASR imm5
                let amount = ((op >> 6) & 0x1F) as u8;
                let rm = (op >> 3) & 7;
                let rd = op & 7;
                self.load_gpr(RAX, rm, a + 4);
                let carry = match ((op >> 11) & 3, amount) {
                    (0, 0) => None,
                    (0, n) => {
                        self.buf.mov_r32_r32(RDX, RAX);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RDX, 32 - n);
                        self.buf.alu_r32_imm32(AluOp::And, RDX, 1);
                        self.buf.shift_r32_imm(ShiftOp::Shl, RAX, n);
                        Some(RDX)
                    }
                    (1, 0) => {
                        self.buf.mov_r32_r32(RDX, RAX);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RDX, 31);
                        self.buf.mov_r32_imm32(RAX, 0);
                        Some(RDX)
                    }
                    (1, n) => {
                        self.buf.mov_r32_r32(RDX, RAX);
                        if n > 1 {
                            self.buf.shift_r32_imm(ShiftOp::Shr, RDX, n - 1);
                        }
                        self.buf.alu_r32_imm32(AluOp::And, RDX, 1);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RAX, n);
                        Some(RDX)
                    }
                    (_, 0) => {
                        self.buf.mov_r32_r32(RDX, RAX);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RDX, 31);
                        self.buf.shift_r32_imm(ShiftOp::Sar, RAX, 31);
                        Some(RDX)
                    }
                    (_, n) => {
                        self.buf.mov_r32_r32(RDX, RAX);
                        if n > 1 {
                            self.buf.shift_r32_imm(ShiftOp::Shr, RDX, n - 1);
                        }
                        self.buf.alu_r32_imm32(AluOp::And, RDX, 1);
                        self.buf.shift_r32_imm(ShiftOp::Sar, RAX, n);
                        Some(RDX)
                    }
                };
                self.store_gpr(rd, RAX);
                self.buf.test_r32_r32(RAX, RAX);
                self.update_logical_flags(carry);
                true
            }
            0b001 => {
                // MOV / CMP / ADD / SUB imm8
                let rd = (op >> 8) & 7;
                let imm = op & 0xFF;
                match (op >> 11) & 3 {
                    0 => {
                        self.buf.mov_r32_imm32(RAX, imm);
                        self.store_gpr(rd, RAX);
                        self.buf.test_r32_r32(RAX, RAX);
                        self.update_logical_flags(None);
                    }
                    1 => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.buf.alu_r32_imm32(AluOp::Cmp, RAX, imm as i32);
                        self.set_nzcv_sub();
                    }
                    2 => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.buf.alu_r32_imm32(AluOp::Add, RAX, imm as i32);
                        self.store_gpr(rd, RAX);
                        self.set_nzcv_add();
                    }
                    _ => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.buf.alu_r32_imm32(AluOp::Sub, RAX, imm as i32);
                        self.store_gpr(rd, RAX);
                        self.set_nzcv_sub();
                    }
                }
                true
            }
            0b010 if (op & (1 << 12)) != 0 => {
                // Reg-offset load/store
                let rd = op & 7;
                let rb = (op >> 3) & 7;
                let ro = (op >> 6) & 7;
                self.load_gpr(R10, rb, a + 4);
                self.load_gpr(RAX, ro, a + 4);
                self.buf.alu_r32_r32(AluOp::Add, R10, RAX);
                let signed = (op & (1 << 9)) != 0;
                match ((op >> 10) & 3, signed) {
                    (0, true) => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.emit_write16(R10, RAX);
                    }
                    (1, true) => {
                        self.emit_read8(R10, RAX, true);
                        self.store_gpr(rd, RAX);
                    }
                    (2, true) => {
                        self.emit_read16(R10, RAX, false);
                        self.store_gpr(rd, RAX);
                    }
                    (_, true) => {
                        self.emit_read16(R10, RAX, true);
                        self.store_gpr(rd, RAX);
                    }
                    (0, false) => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.emit_write32(R10, RAX);
                    }
                    (1, false) => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.emit_write8(R10, RAX);
                    }
                    (2, false) => {
                        self.emit_read32(R10, RAX);
                        self.store_gpr(rd, RAX);
                    }
                    (_, false) => {
                        self.emit_read8(R10, RAX, false);
                        self.store_gpr(rd, RAX);
                    }
                }
                true
            }
            0b010 if (op & (1 << 11)) != 0 => {
                // LDR Rd, [PC, #imm]
                let rd = (op >> 8) & 7;
                let lit = ((a + 4) & !3).wrapping_add((op & 0xFF) * 4);
                if self.options.fold_text_literals
                    && lit >= self.text.base
                    && lit.saturating_add(4) <= self.text.end()
                {
                    if let Some(val) = self.text.read32(lit) {
                        self.folded_spans.push((lit, lit + 4));
                        self.buf.mov_r32_imm32(RAX, val);
                        self.store_gpr(rd, RAX);
                        return true;
                    }
                }
                self.buf.mov_r32_imm32(R10, lit);
                self.emit_read32(R10, RAX);
                self.store_gpr(rd, RAX);
                true
            }
            0b010 if (op & (1 << 10)) != 0 => thumb_high_register(self, a, op),
            0b010 => thumb_alu(self, a, op),
            0b011 => {
                let rd = op & 7;
                let rb = (op >> 3) & 7;
                let byte = (op & (1 << 12)) != 0;
                let off = ((op >> 6) & 0x1F) * if byte { 1 } else { 4 };
                self.load_gpr(R10, rb, a + 4);
                if off != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R10, off as i32);
                }
                match ((op & (1 << 11)) != 0, byte) {
                    (true, false) => {
                        self.emit_read32(R10, RAX);
                        self.store_gpr(rd, RAX);
                    }
                    (true, true) => {
                        self.emit_read8(R10, RAX, false);
                        self.store_gpr(rd, RAX);
                    }
                    (false, false) => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.emit_write32(R10, RAX);
                    }
                    (false, true) => {
                        self.load_gpr(RAX, rd, a + 4);
                        self.emit_write8(R10, RAX);
                    }
                }
                true
            }
            0b100 if (op & (1 << 12)) != 0 => {
                let rd = (op >> 8) & 7;
                let off = (op & 0xFF) * 4;
                self.load_gpr(R10, 13, a + 4);
                if off != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R10, off as i32);
                }
                if (op & (1 << 11)) != 0 {
                    self.emit_read32(R10, RAX);
                    self.store_gpr(rd, RAX);
                } else {
                    self.load_gpr(RAX, rd, a + 4);
                    self.emit_write32(R10, RAX);
                }
                true
            }
            0b100 => {
                let rd = op & 7;
                let rb = (op >> 3) & 7;
                let off = ((op >> 6) & 0x1F) * 2;
                self.load_gpr(R10, rb, a + 4);
                if off != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R10, off as i32);
                }
                if (op & (1 << 11)) != 0 {
                    self.emit_read16(R10, RAX, false);
                    self.store_gpr(rd, RAX);
                } else {
                    self.load_gpr(RAX, rd, a + 4);
                    self.emit_write16(R10, RAX);
                }
                true
            }
            0b101 if (op & (1 << 12)) != 0 => thumb_misc(self, a, op),
            0b101 => {
                let rd = (op >> 8) & 7;
                let off = (op & 0xFF) * 4;
                if (op & (1 << 11)) != 0 {
                    self.load_gpr(RAX, 13, a + 4);
                    if off != 0 {
                        self.buf.alu_r32_imm32(AluOp::Add, RAX, off as i32);
                    }
                } else {
                    self.buf.mov_r32_imm32(RAX, ((a + 4) & !3).wrapping_add(off));
                }
                self.store_gpr(rd, RAX);
                true
            }
            0b110 if (op & (1 << 12)) != 0 => match (op >> 8) & 0xF {
                0xF => self.emit_svc(a + 2, op & 0xFF),
                0xE => self.emit_interpret(a, op),
                c => {
                    let s8 = (((op & 0xFF) << 24) as i32 >> 24) as u32;
                    let target = (a + 4).wrapping_add(s8 << 1);
                    if self.mode == Mode::Thumb && self.labels.contains(&target) {
                        self.emit_cond_local_branch(c, target);
                    } else {
                        let skip = self.emit_cond_skip(c);
                        let _ = self.emit_jump(target, true);
                        self.patch_cond_skip(skip);
                    }
                    true
                }
            },
            0b110 => {
                // LDMIA / STMIA
                let rb = (op >> 8) & 7;
                let list = op & 0xFF;
                if list == 0 {
                    self.load_gpr(RAX, rb, a + 4);
                    self.buf.alu_r32_imm32(AluOp::Add, RAX, 0x40);
                    self.store_gpr(rb, RAX);
                    return true;
                }
                let load = (op & (1 << 11)) != 0;
                let span = list.count_ones() * 4;
                self.load_gpr(R8, rb, a + 4);
                let words: Vec<SpanWord> = (0..8)
                    .filter(|i| (list & (1 << i)) != 0)
                    .map(|r| SpanWord::Gpr(r as u8))
                    .collect();
                emit_span(self, load, &words);
                if !(load && (list & (1 << rb)) != 0) {
                    self.buf.alu_r32_imm32(AluOp::Add, R8, span as i32);
                    self.store_gpr(rb, R8);
                }
                true
            }
            _ => match op >> 11 {
                0b11100 => {
                    let s11 = (((op & 0x7FF) << 21) as i32 >> 21) as u32;
                    self.emit_jump((a + 4).wrapping_add(s11 << 1), true)
                }
                0b11110 if matches!(second >> 11, 0b11111 | 0b11101) => {
                    let high = ((((op & 0x7FF) << 21) as i32 >> 21) << 12) as u32;
                    let target = (a + 4).wrapping_add(high).wrapping_add((second & 0x7FF) << 1);
                    if (second >> 11) == 0b11111 {
                        self.emit_call((a + 4) | 1, target, true)
                    } else {
                        self.emit_call((a + 4) | 1, target & !3, false)
                    }
                }
                _ => self.emit_interpret(a, op),
            },
        }
    }
}

#[inline(always)]
fn sreg(vd: u32, extra: bool) -> u8 {
    (((vd << 1) | (extra as u32)) & 31) as u8
}

#[inline(always)]
fn dreg(vd: u32, extra: bool) -> u8 {
    ((((extra as u32) << 4) | vd) & 15) as u8
}

#[inline(always)]
fn emit_stmxcsr_scratch(em: &mut FnEmitter<'_>) {
    // stmxcsr dword ptr [rbx + SCRATCH_FP]
    em.buf.emit_u8(0x0F);
    em.buf.emit_u8(0xAE);
    em.buf.emit_u8(0x9B);
    em.buf.emit_u32((offsets::SCRATCH_FP as i32) as u32);
}

#[inline(always)]
fn emit_ldmxcsr_scratch(em: &mut FnEmitter<'_>) {
    // ldmxcsr dword ptr [rbx + SCRATCH_FP]
    em.buf.emit_u8(0x0F);
    em.buf.emit_u8(0xAE);
    em.buf.emit_u8(0x93);
    em.buf.emit_u32((offsets::SCRATCH_FP as i32) as u32);
}

#[inline(always)]
fn emit_vfp_load(em: &mut FnEmitter<'_>, double: bool, xmm: u8, base: u8, disp: i32) {
    if double {
        em.buf.movsd_load(xmm, base, disp);
    } else {
        em.buf.movss_load(xmm, base, disp);
    }
}

#[inline(always)]
fn emit_vfp_store(em: &mut FnEmitter<'_>, double: bool, base: u8, disp: i32, xmm: u8) {
    if double {
        em.buf.movsd_store(base, disp, xmm);
    } else {
        em.buf.movss_store(base, disp, xmm);
    }
}

#[inline(always)]
fn emit_vfp_add(em: &mut FnEmitter<'_>, double: bool, dst: u8, src: u8) {
    if double {
        em.buf.addsd(dst, src);
    } else {
        em.buf.addss(dst, src);
    }
}

#[inline(always)]
fn emit_vfp_sub(em: &mut FnEmitter<'_>, double: bool, dst: u8, src: u8) {
    if double {
        em.buf.subsd(dst, src);
    } else {
        em.buf.subss(dst, src);
    }
}

#[inline(always)]
fn emit_vfp_mul(em: &mut FnEmitter<'_>, double: bool, dst: u8, src: u8) {
    if double {
        em.buf.mulsd(dst, src);
    } else {
        em.buf.mulss(dst, src);
    }
}

#[inline(always)]
fn emit_vfp_div(em: &mut FnEmitter<'_>, double: bool, dst: u8, src: u8) {
    if double {
        em.buf.divsd(dst, src);
    } else {
        em.buf.divss(dst, src);
    }
}

#[inline(always)]
fn emit_vfp_negate(em: &mut FnEmitter<'_>, double: bool, xmm: u8) {
    if double {
        em.buf.mov_r64_imm64(RCX, 0x8000_0000_0000_0000);
        em.buf.movq_xmm_r64(XMM2, RCX);
        em.buf.xorpd(xmm, XMM2);
    } else {
        em.buf.mov_r32_imm32(RCX, 0x8000_0000);
        em.buf.movd_xmm_r32(XMM2, RCX);
        em.buf.xorps(xmm, XMM2);
    }
}

fn vfp_maybe_vector<F>(
    em: &mut FnEmitter<'_>,
    double: bool,
    op_code: u32,
    rd: u8,
    rn: u8,
    rm: u8,
    scalar_body: F,
) -> bool
where
    F: FnOnce(&mut FnEmitter<'_>),
{
    let bank = if double { 4 } else { 8 };
    if !em.options.emit_vfp_vectors || rd < bank {
        scalar_body(em);
        return true;
    }
    // Check `(*ctx.fpscr & 0x0007_0000) != 0` (`FPSCR.LEN` bits [18:16]):
    em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_FPSCR as i32);
    em.buf.ldr_r32_mem(RAX, RAX, 0);
    em.buf.test_r32_imm32(RAX, 0x0007_0000);
    let j_vec = em.buf.jcc_rel32(cc::NZ);

    scalar_body(em);
    let j_done = em.buf.jmp_rel32();

    let vec_pos = em.buf.pos();
    em.buf.patch_rel32(j_vec, vec_pos);
    let packed = ((double as u32) << 20)
        | ((op_code & 0xF) << 16)
        | (((rd as u32) & 31) << 10)
        | (((rn as u32) & 31) << 5)
        | ((rm as u32) & 31);
    em.buf.mov_r32_imm32(R10, packed);
    em.stub_call(StubId::VfpVector);

    let done_pos = em.buf.pos();
    em.buf.patch_rel32(j_done, done_pos);
    true
}

fn vfp_extension(em: &mut FnEmitter<'_>, a: u32, op: u32, double: bool, rd: u8, rm: u8) -> bool {
    let top = (op & (1 << 7)) != 0;
    match ((op >> 16) & 0xF, top) {
        (0b0000 | 0b0001, _) => {
            let second = ((op >> 16) & 1) != 0;
            let vop = 9 + (((second as u32) << 1) | (top as u32));
            vfp_maybe_vector(em, double, vop, rd, rd, rm, |em| {
                em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
                let disp = |r: u8| -> i32 { (r as i32) * if double { 8 } else { 4 } };
                match (second, top) {
                    (false, false) => {
                        // VMOV (raw bit copy without flushing subnormals)
                        if rd != rm {
                            if double {
                                em.buf.ldr_r64_mem(RCX, RAX, disp(rm));
                                em.buf.str_r64_mem(RCX, RAX, disp(rd));
                            } else {
                                em.buf.ldr_r32_mem(RCX, RAX, disp(rm));
                                em.buf.str_r32_mem(RCX, RAX, disp(rd));
                            }
                        }
                    }
                    (false, true) => {
                        // VABS (clear sign bit via andps/andpd)
                        emit_vfp_load(em, double, XMM0, RAX, disp(rm));
                        if double {
                            em.buf.mov_r64_imm64(RCX, 0x7FFF_FFFF_FFFF_FFFF);
                            em.buf.movq_xmm_r64(XMM2, RCX);
                            em.buf.andpd(XMM0, XMM2);
                        } else {
                            em.buf.mov_r32_imm32(RCX, 0x7FFF_FFFF);
                            em.buf.movd_xmm_r32(XMM2, RCX);
                            em.buf.andps(XMM0, XMM2);
                        }
                        emit_vfp_store(em, double, RAX, disp(rd), XMM0);
                    }
                    (true, false) => {
                        // VNEG (flip sign bit via xorps/xorpd)
                        emit_vfp_load(em, double, XMM0, RAX, disp(rm));
                        emit_vfp_negate(em, double, XMM0);
                        emit_vfp_store(em, double, RAX, disp(rd), XMM0);
                    }
                    (true, true) => {
                        // VSQRT
                        emit_vfp_load(em, double, XMM0, RAX, disp(rm));
                        if double {
                            em.buf.sqrtsd(XMM0, XMM0);
                        } else {
                            em.buf.sqrtss(XMM0, XMM0);
                        }
                        emit_vfp_store(em, double, RAX, disp(rd), XMM0);
                    }
                }
            })
        }
        (0b0100 | 0b0101, _) => {
            // VCMP / VCMPE (against register or 0.0)
            let cmp_zero = ((op >> 16) & 0xF) == 0b0101;
            em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
            let disp = |r: u8| -> i32 { (r as i32) * if double { 8 } else { 4 } };
            emit_vfp_load(em, double, XMM0, RAX, disp(rd));
            if cmp_zero {
                em.buf.xorps(XMM1, XMM1);
            } else {
                emit_vfp_load(em, double, XMM1, RAX, disp(rm));
            }
            if double {
                em.buf.ucomisd(XMM0, XMM1);
            } else {
                em.buf.ucomiss(XMM0, XMM1);
            }
            // Capture x86 EFLAGS into FPSCR[31:28] BEFORE any flag-clobbering op!
            // GT: 0b0010 (0x2), LT (CF=1): 0b1000 (0x8), EQ (ZF=1): 0b0110 (0x6), Unordered (PF=1): 0b0011 (0x3)
            em.buf.mov_r32_imm32(RAX, 0x2000_0000);
            em.buf.mov_r32_imm32(RCX, 0x8000_0000);
            em.buf.cmovcc_r32_r32(cc::B, RAX, RCX);
            em.buf.mov_r32_imm32(RCX, 0x6000_0000);
            em.buf.cmovcc_r32_r32(cc::E, RAX, RCX);
            em.buf.mov_r32_imm32(RCX, 0x3000_0000);
            em.buf.cmovcc_r32_r32(cc::P, RAX, RCX);

            em.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_FPSCR as i32);
            em.buf.ldr_r32_mem(RDX, RCX, 0);
            em.buf.alu_r32_imm32(AluOp::And, RDX, 0x0FFF_FFFF);
            em.buf.alu_r32_r32(AluOp::Or, RDX, RAX);
            em.buf.str_r32_mem(RDX, RCX, 0);
            true
        }
        (0b0111, true) => {
            // VCVT.F32.F64 / VCVT.F64.F32
            let (vd, vm) = ((op >> 12) & 0xF, op & 0xF);
            let (d_bit, m_bit) = ((op & (1 << 22)) != 0, (op & (1 << 5)) != 0);
            em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
            if double {
                let sd = sreg(vd, d_bit);
                em.buf.movsd_load(XMM0, RAX, (rm as i32) * 8);
                em.buf.cvtsd2ss(XMM0, XMM0);
                em.buf.movss_store(RAX, (sd as i32) * 4, XMM0);
            } else {
                let sm = sreg(vm, m_bit);
                let dd = dreg(vd, d_bit);
                em.buf.movss_load(XMM0, RAX, (sm as i32) * 4);
                em.buf.cvtss2sd(XMM0, XMM0);
                em.buf.movsd_store(RAX, (dd as i32) * 8, XMM0);
            }
            true
        }
        (0b1000, _) => {
            // VCVT from integer (`top` == signed)
            let src = sreg(op & 0xF, (op & (1 << 5)) != 0);
            em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
            // `ldr_r32_mem` zero-extends into 64-bit `RCX` (safe for unsigned `cvtsi2*_r64`).
            em.buf.ldr_r32_mem(RCX, RAX, (src as i32) * 4);
            if !double {
                if top {
                    em.buf.cvtsi2ss_r32(XMM0, RCX);
                } else {
                    em.buf.mov_r32_r32(RCX, RCX);
                    em.buf.cvtsi2ss_r64(XMM0, RCX);
                }
                em.buf.movss_store(RAX, (rd as i32) * 4, XMM0);
            } else {
                if top {
                    em.buf.cvtsi2sd_r32(XMM0, RCX);
                } else {
                    em.buf.mov_r32_r32(RCX, RCX);
                    em.buf.cvtsi2sd_r64(XMM0, RCX);
                }
                em.buf.movsd_store(RAX, (rd as i32) * 8, XMM0);
            }
            true
        }
        (0b1100 | 0b1101, _) => {
            // VCVT / VCVTR float -> integer (exact ARM rounding & saturation in `runtime.rs`)
            let signed = ((op >> 16) & 1) != 0;
            let round_zero = top;
            let sd = sreg((op >> 12) & 0xF, (op & (1 << 22)) != 0);
            let packed = ((double as u32) << 12)
                | ((signed as u32) << 11)
                | ((round_zero as u32) << 10)
                | (((sd as u32) & 31) << 5)
                | ((rm as u32) & 31);
            em.buf.mov_r32_imm32(R10, packed);
            em.stub_call(StubId::VfpToInt);
            true
        }
        _ => em.emit_interpret(a, op),
    }
}

fn vfp_move_pair(em: &mut FnEmitter<'_>, a: u32, op: u32, double: bool) -> bool {
    let load = (op & (1 << 20)) != 0;
    let rt = (op >> 12) & 0xF;
    let rt2 = (op >> 16) & 0xF;
    if rt == 15 || rt2 == 15 {
        return em.emit_interpret(a, op);
    }
    let (vm, m_bit) = (op & 0xF, (op & (1 << 5)) != 0);
    let (low, high) = if double {
        let dm = dreg(vm, m_bit);
        ((dm * 2) & 31, (dm * 2 + 1) & 31)
    } else {
        let sm = sreg(vm, m_bit);
        (sm, (sm + 1) & 31)
    };
    em.buf.ldr_r64_mem(RAX, RBX, offsets::CTX_VFP as i32);
    if load {
        em.buf.ldr_r32_mem(RCX, RAX, (low as i32) * 4);
        em.store_gpr(rt, RCX);
        em.buf.ldr_r32_mem(RCX, RAX, (high as i32) * 4);
        em.store_gpr(rt2, RCX);
    } else {
        em.load_gpr(RCX, rt, a + 8);
        em.buf.str_r32_mem(RCX, RAX, (low as i32) * 4);
        em.load_gpr(RCX, rt2, a + 8);
        em.buf.str_r32_mem(RCX, RAX, (high as i32) * 4);
    }
    true
}

/// Emits a slow-path word transfer at the guest address already loaded in `R10d`.
/// Preserves `R8` and `R9`.
fn emit_span_word_slow(em: &mut FnEmitter<'_>, load: bool, item: SpanWord) {
    match item {
        SpanWord::Gpr(r) if load => {
            em.emit_read32(R10, RAX);
            if r == 15 {
                em.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
            } else {
                em.store_gpr(r as u32, RAX);
            }
        }
        SpanWord::Gpr(r) => {
            if r == 15 {
                em.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
            } else {
                em.load_gpr(RAX, r as u32, 0);
            }
            em.emit_write32(R10, RAX);
        }
        SpanWord::Const(val) => {
            em.buf.mov_r32_imm32(RAX, val);
            em.emit_write32(R10, RAX);
        }
        SpanWord::Vfp(s) if load => {
            em.emit_read32(R10, RAX);
            em.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
            em.buf.str_r32_mem(RAX, RCX, (s as i32) * 4);
        }
        SpanWord::Vfp(s) => {
            em.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
            em.buf.ldr_r32_mem(RAX, RCX, (s as i32) * 4);
            em.emit_write32(R10, RAX);
        }
    }
}

/// Emits a fast single-page span transfer with per-word fallback for `VLDM`/`VSTM`/`LDMIA`/`STMIA`/`PUSH`/`POP`.
/// `R8d` must hold starting guest address `p`. `R8` is preserved across both fast and slow paths!
fn emit_span(em: &mut FnEmitter<'_>, load: bool, regs: &[SpanWord]) {
    if let [single] = regs {
        em.buf.mov_r32_r32(R10, R8);
        emit_span_word_slow(em, load, *single);
        return;
    }

    let bytes = (regs.len() * 4) as i32;
    let pages_reg = if load { R12 } else { R13 };

    // RCX = p & 0xFFF (zero-extended)
    em.buf.mov_r32_r32(RCX, R8);
    em.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFF);
    // Check if `(p & 0xFFF) + bytes - 1 >= 4096` (page boundary crossing)
    em.buf.lea_r64_mem(RDX, RCX, bytes - 1);
    em.buf.alu_r32_imm32(AluOp::Cmp, RDX, 4096);
    let j_slow1 = em.buf.jcc_rel32(cc::AE);

    // RDX = p >> 12 (zero-extended)
    em.buf.mov_r32_r32(RDX, R8);
    em.buf.shift_r32_imm(ShiftOp::Shr, RDX, 12);
    em.buf.ldr_r64_sib(RSI, pages_reg, RDX, 8, 0);
    em.buf.test_r64_r64(RSI, RSI);
    let j_slow2 = em.buf.jcc_rel32(cc::Z);
    em.buf.alu_r64_r64(AluOp::Add, RSI, RCX);

    let has_vfp = regs.iter().any(|w| matches!(w, SpanWord::Vfp(_)));
    if has_vfp {
        em.buf.ldr_r64_mem(RDX, RBX, offsets::CTX_VFP as i32);
    }

    for (i, item) in regs.iter().enumerate() {
        let off = (i * 4) as i32;
        match *item {
            SpanWord::Gpr(r) if load => {
                em.buf.ldr_r32_mem(RAX, RSI, off);
                if r == 15 {
                    em.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
                } else {
                    em.store_gpr(r as u32, RAX);
                }
            }
            SpanWord::Gpr(r) => {
                if r == 15 {
                    em.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
                } else {
                    em.load_gpr(RAX, r as u32, 0);
                }
                em.buf.str_r32_mem(RAX, RSI, off);
            }
            SpanWord::Const(val) => {
                em.buf.str_imm32_mem(RSI, off, val);
            }
            SpanWord::Vfp(s) if load => {
                em.buf.ldr_r32_mem(RAX, RSI, off);
                em.buf.str_r32_mem(RAX, RDX, (s as i32) * 4);
            }
            SpanWord::Vfp(s) => {
                em.buf.ldr_r32_mem(RAX, RDX, (s as i32) * 4);
                em.buf.str_r32_mem(RAX, RSI, off);
            }
        }
    }
    let j_done = em.buf.jmp_rel32();

    let slow_pos = em.buf.pos();
    em.buf.patch_rel32(j_slow1, slow_pos);
    em.buf.patch_rel32(j_slow2, slow_pos);
    for (i, item) in regs.iter().enumerate() {
        let off = (i * 4) as i32;
        em.buf.mov_r32_r32(R10, R8);
        if off != 0 {
            em.buf.alu_r32_imm32(AluOp::Add, R10, off);
        }
        emit_span_word_slow(em, load, *item);
    }
    let done_pos = em.buf.pos();
    em.buf.patch_rel32(j_done, done_pos);
}

fn thumb_alu(em: &mut FnEmitter<'_>, a: u32, op: u32) -> bool {
    let rd = op & 7;
    let rs = (op >> 3) & 7;
    match (op >> 6) & 0xF {
        0x0 => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.alu_r32_r32(AluOp::And, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.update_logical_flags(None);
        }
        0x1 => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.alu_r32_r32(AluOp::Xor, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.update_logical_flags(None);
        }
        kind @ (0x2 | 0x3 | 0x4 | 0x7) => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            let stub = match kind {
                0x2 => StubId::ShiftLslC,
                0x3 => StubId::ShiftLsrC,
                0x4 => StubId::ShiftAsrC,
                _ => StubId::ShiftRorC,
            };
            em.stub_call(stub);
            em.store_gpr(rd, RAX);
            em.buf.test_r32_r32(RAX, RAX);
            em.update_logical_flags(Some(RDX));
        }
        0x5 => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.bt_r32_imm8(R15, 29);
            em.buf.alu_r32_r32(AluOp::Adc, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.set_nzcv_add();
        }
        0x6 => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.bt_r32_imm8(R15, 29);
            em.buf.cmc();
            em.buf.alu_r32_r32(AluOp::Sbb, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.set_nzcv_sub();
        }
        0x8 => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.test_r32_r32(RAX, RCX);
            em.update_logical_flags(None);
        }
        0x9 => {
            em.load_gpr(RAX, rs, a + 4);
            em.buf.neg_r32(RAX);
            em.store_gpr(rd, RAX);
            em.set_nzcv_sub();
        }
        0xA => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.alu_r32_r32(AluOp::Cmp, RAX, RCX);
            em.set_nzcv_sub();
        }
        0xB => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.alu_r32_r32(AluOp::Add, RAX, RCX);
            em.set_nzcv_add();
        }
        0xC => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.update_logical_flags(None);
        }
        0xD => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.imul_r32_r32(RAX, RCX);
            em.store_gpr(rd, RAX);
            em.buf.test_r32_r32(RAX, RAX);
            em.update_logical_flags(None);
        }
        0xE => {
            em.load_gpr(RAX, rd, a + 4);
            em.load_gpr(RCX, rs, a + 4);
            em.buf.not_r32(RCX);
            em.buf.alu_r32_r32(AluOp::And, RAX, RCX);
            em.store_gpr(rd, RAX);
            em.update_logical_flags(None);
        }
        _ => {
            em.load_gpr(RAX, rs, a + 4);
            em.buf.not_r32(RAX);
            em.store_gpr(rd, RAX);
            em.buf.test_r32_r32(RAX, RAX);
            em.update_logical_flags(None);
        }
    }
    true
}

fn thumb_high_register(em: &mut FnEmitter<'_>, a: u32, op: u32) -> bool {
    let rd = (op & 7) | ((op >> 4) & 8);
    let rs = (op >> 3) & 0xF;
    em.load_gpr(RAX, rs, a + 4);
    match (op >> 8) & 3 {
        0 if rd == 15 => {
            em.buf.mov_r32_imm32(RCX, a + 4);
            em.buf.alu_r32_r32(AluOp::Add, RAX, RCX);
            // Non-interworking Thumb PC write keeps `ctx.thumb = 1` and aligns `(val | 1) & !1 == val & !1`.
            em.buf.alu_r32_imm32(AluOp::Or, RAX, 1);
            em.emit_jump_to_reg(RAX)
        }
        0 => {
            em.load_gpr(RCX, rd, a + 4);
            em.buf.alu_r32_r32(AluOp::Add, RCX, RAX);
            em.store_gpr(rd, RCX);
            true
        }
        1 => {
            em.load_gpr(RCX, rd, a + 4);
            em.buf.alu_r32_r32(AluOp::Cmp, RCX, RAX);
            em.set_nzcv_sub();
            true
        }
        2 if rd == 15 && rs == 14 => {
            em.buf.alu_r32_imm32(AluOp::Or, RAX, 1);
            em.emit_return_to_reg(RAX)
        }
        2 if rd == 15 => {
            em.buf.alu_r32_imm32(AluOp::Or, RAX, 1);
            em.emit_jump_to_reg(RAX)
        }
        2 => {
            em.store_gpr(rd, RAX);
            true
        }
        _ if (op & (1 << 7)) != 0 => {
            // Thumb BLX Rm (`RAX` holds target `Rm`)
            let next = a + 2;
            em.buf.mov_r32_imm32(RCX, next | 1);
            em.store_gpr(14, RCX);

            em.buf.mov_r32_r32(RCX, RAX);
            em.buf.alu_r32_imm32(AluOp::And, RCX, 1);
            em.buf.str_r8_mem(RCX, RBX, offsets::CTX_THUMB as i32);
            em.buf.mov_r32_r32(RDX, RAX);
            em.buf.alu_r32_imm32(AluOp::And, RAX, !3u32 as i32);
            em.buf.alu_r32_imm32(AluOp::And, RDX, !1u32 as i32);
            em.buf.test_r32_r32(RCX, RCX);
            em.buf.cmovcc_r32_r32(cc::NZ, RAX, RDX);
            em.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);

            // Depth check (`if ++ctx.depth > 2048`)
            em.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
            em.buf.alu_r32_imm32(AluOp::Add, RAX, 1);
            em.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
            em.buf.alu_r32_imm32(AluOp::Cmp, RAX, RECOMP_DEPTH_LIMIT);
            let j_ok = em.buf.jcc_rel32(cc::BE);
            em.buf.alu_r32_imm32(AluOp::Sub, RAX, 1);
            em.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
            em.buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, recomp_abi::EXIT_UNWIND);
            em.emit_epilogue_jump();
            let ok_pos = em.buf.pos();
            em.buf.patch_rel32(j_ok, ok_pos);

            em.stub_call(StubId::IndirectCall);
            em.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
            em.buf.alu_r32_imm32(AluOp::Sub, RAX, 1);
            em.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
            em.emit_check_unwind();

            // RETURNED_T(next): verify `r[15] == next` and `ctx.thumb == 1`
            em.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
            em.buf.mov_r32_imm32(RCX, next);
            em.buf.alu_r32_r32(AluOp::Sub, RAX, RCX);
            em.buf.ldr_u8_mem(RCX, RBX, offsets::CTX_THUMB as i32);
            em.buf.alu_r32_imm32(AluOp::Xor, RCX, 1);
            em.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
            let j_match = em.buf.jcc_rel32(cc::Z);
            em.stub_jump(StubId::IndirectJump);
            let match_pos = em.buf.pos();
            em.buf.patch_rel32(j_match, match_pos);
            true
        }
        _ if rs == 14 => em.emit_return_to_reg(RAX),
        _ => em.emit_jump_to_reg(RAX),
    }
}

fn thumb_misc(em: &mut FnEmitter<'_>, a: u32, op: u32) -> bool {
    let rd = op & 7;
    let rm = (op >> 3) & 7;
    match (op >> 8) & 0xF {
        0b0000 => {
            let off = ((op & 0x7F) * 4) as i32;
            em.load_gpr(RAX, 13, a + 4);
            if (op & (1 << 7)) != 0 {
                em.buf.alu_r32_imm32(AluOp::Sub, RAX, off);
            } else {
                em.buf.alu_r32_imm32(AluOp::Add, RAX, off);
            }
            em.store_gpr(13, RAX);
            true
        }
        0b0010 => {
            em.load_gpr(RAX, rm, a + 4);
            match (op >> 6) & 3 {
                0 => em.buf.movsx_r32_r16(RAX, RAX),
                1 => em.buf.movsx_r32_r8(RAX, RAX),
                2 => em.buf.movzx_r32_r16(RAX, RAX),
                _ => em.buf.movzx_r32_r8(RAX, RAX),
            }
            em.store_gpr(rd, RAX);
            true
        }
        0b1010 => {
            em.load_gpr(RAX, rm, a + 4);
            match (op >> 6) & 3 {
                0 => em.buf.bswap_r32(RAX),
                1 => {
                    em.buf.mov_r32_r32(RCX, RAX);
                    em.buf.alu_r32_imm32(AluOp::And, RAX, 0x00FF_00FF);
                    em.buf.shift_r32_imm(ShiftOp::Shl, RAX, 8);
                    em.buf.alu_r32_imm32(AluOp::And, RCX, 0xFF00_FF00u32 as i32);
                    em.buf.shift_r32_imm(ShiftOp::Shr, RCX, 8);
                    em.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                }
                3 => {
                    em.buf.mov_r32_r32(RCX, RAX);
                    em.buf.shift_r32_imm(ShiftOp::Shl, RAX, 8);
                    em.buf.alu_r32_imm32(AluOp::And, RAX, 0xFF00);
                    em.buf.shift_r32_imm(ShiftOp::Shr, RCX, 8);
                    em.buf.alu_r32_imm32(AluOp::And, RCX, 0xFF);
                    em.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                    em.buf.movsx_r32_r16(RAX, RAX);
                }
                _ => return em.emit_interpret(a, op),
            }
            em.store_gpr(rd, RAX);
            true
        }
        0b0110 | 0b1111 => true,
        0b0100 | 0b0101 | 0b1100 | 0b1101 => {
            let list = op & 0xFF;
            let extra = (op & (1 << 8)) != 0;
            let count = list.count_ones() + (extra as u32);
            if count == 0 {
                return em.emit_interpret(a, op);
            }
            let mut words: Vec<SpanWord> = (0..8)
                .filter(|i| (list & (1 << i)) != 0)
                .map(|r| SpanWord::Gpr(r as u8))
                .collect();
            if (op & (1 << 11)) != 0 {
                // POP
                em.load_gpr(R8, 13, a + 4);
                if extra {
                    words.push(SpanWord::Gpr(15));
                }
                emit_span(em, true, &words);
                em.buf.alu_r32_imm32(AluOp::Add, R8, (count * 4) as i32);
                em.store_gpr(13, R8);
                if extra {
                    em.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
                    return em.emit_return_to_reg(RAX);
                }
            } else {
                // PUSH
                em.load_gpr(R8, 13, a + 4);
                em.buf.alu_r32_imm32(AluOp::Sub, R8, (count * 4) as i32);
                if extra {
                    words.push(SpanWord::Gpr(14));
                }
                emit_span(em, false, &words);
                em.store_gpr(13, R8);
            }
            true
        }
        _ => em.emit_interpret(a, op),
    }
}
