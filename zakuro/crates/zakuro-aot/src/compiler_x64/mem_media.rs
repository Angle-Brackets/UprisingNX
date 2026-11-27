//! x86_64 AOT lowering for ARM memory transfers, swap/exclusive, media/SIMD, and dual-multiply ops.

#![allow(dead_code)]

use crate::runtime::offsets;
use crate::x64::{
    AluOp, ShiftOp, cc, R10, R11, R12, R13, R15, R8, R9, RAX, RBX, RCX, RDI, RDX, RSI,
};

use super::{FnEmitter, SpanWord, StubId};

impl FnEmitter<'_> {
    pub(crate) fn arm_single_transfer(&mut self, a: u32, op: u32) -> bool {
        let rn = (op >> 16) & 0xF;
        let rd = (op >> 12) & 0xF;
        let pre = (op & (1 << 24)) != 0;
        let up = (op & (1 << 23)) != 0;
        let byte = (op & (1 << 22)) != 0;
        let writeback = !pre || (op & (1 << 21)) != 0;
        let load = (op & (1 << 20)) != 0;
        if writeback && rn == 15 {
            return self.emit_interpret(a, op);
        }

        // PC-relative `.text` literal pool folding (`LDR Rd, [PC, ±#imm]` / `LDRB Rd, [PC, ±#imm]`):
        if self.options.fold_text_literals
            && load
            && rn == 15
            && pre
            && !writeback
            && (op & (1 << 25)) == 0
        {
            let imm = op & 0xFFF;
            let lit = if up { (a + 8).wrapping_add(imm) } else { (a + 8).wrapping_sub(imm) };
            if byte {
                if let Some(val) = self.text.read8(lit) {
                    self.folded_spans.push((lit, lit + 1));
                    if rd == 15 {
                        return self.emit_jump((val as u32) & !3, false);
                    }
                    self.buf.mov_r32_imm32(RAX, val as u32);
                    self.store_gpr(rd, RAX);
                    return true;
                }
            } else if lit >= self.text.base && lit.saturating_add(4) <= self.text.end() {
                if let Some(val) = self.text.read32(lit) {
                    self.folded_spans.push((lit, lit + 4));
                    if rd == 15 {
                        let thumb = (val & 1) != 0;
                        return self.emit_jump(val & if thumb { !1 } else { !3 }, thumb);
                    }
                    self.buf.mov_r32_imm32(RAX, val);
                    self.store_gpr(rd, RAX);
                    return true;
                }
            }
        }

        // Compute `base` in `r9d`, then `r8d` = effective address, `r9d` = writeback address `oa`
        // (`r8` and `r9` are preserved across `emit_read*`/`emit_write*`).
        self.load_gpr(R9, rn, a + 8);
        if (op & (1 << 25)) != 0 {
            self.load_gpr(RAX, op & 0xF, a + 8);
            let kind = (op >> 5) & 3;
            let amt = ((op >> 7) & 0x1F) as u8;
            match (kind, amt) {
                (0, 0) => {}
                (0, n) => self.buf.shift_r32_imm(ShiftOp::Shl, RAX, n),
                (1, 0) => self.buf.mov_r32_imm32(RAX, 0),
                (1, n) => self.buf.shift_r32_imm(ShiftOp::Shr, RAX, n),
                (2, 0) => self.buf.shift_r32_imm(ShiftOp::Sar, RAX, 31),
                (2, n) => self.buf.shift_r32_imm(ShiftOp::Sar, RAX, n),
                (_, 0) => {
                    // RRX: `(old_c << 31) | (m >> 1)` (`r15d[29]` is guest C).
                    self.buf.shift_r32_imm(ShiftOp::Shr, RAX, 1);
                    self.buf.mov_r32_r32(RCX, R15);
                    self.buf.alu_r32_imm32(AluOp::And, RCX, 0x2000_0000);
                    self.buf.shift_r32_imm(ShiftOp::Shl, RCX, 2);
                    self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                }
                (_, n) => self.buf.shift_r32_imm(ShiftOp::Ror, RAX, n),
            }
            if writeback && !pre {
                self.buf.mov_r32_r32(R8, R9);
                self.buf.alu_r32_r32(if up { AluOp::Add } else { AluOp::Sub }, R9, RAX);
            } else {
                self.buf.alu_r32_r32(if up { AluOp::Add } else { AluOp::Sub }, R9, RAX);
                self.buf.mov_r32_r32(R8, R9);
            }
        } else {
            let imm = op & 0xFFF;
            let delta = if up { imm as i32 } else { (imm as i32).wrapping_neg() };
            if writeback && !pre {
                self.buf.mov_r32_r32(R8, R9);
                if delta != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R9, delta);
                }
            } else {
                if delta != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R9, delta);
                }
                self.buf.mov_r32_r32(R8, R9);
            }
        }

        if load {
            if byte {
                self.emit_read8(R8, RAX, false);
            } else {
                self.emit_read32(R8, RAX);
            }
            if writeback && rn != rd {
                self.store_gpr(rn, R9);
            }
            if rd == 15 {
                let pop = rn == 13 && !pre && up && (op & (1 << 25)) == 0 && (op & 0xFFF) == 4;
                return if pop { self.emit_return_to_reg(RAX) } else { self.emit_jump_to_reg(RAX) };
            }
            self.store_gpr(rd, RAX);
        } else {
            self.load_gpr(RSI, rd, a + 12);
            if byte {
                self.emit_write8(R8, RSI);
            } else {
                self.emit_write32(R8, RSI);
            }
            if writeback {
                self.store_gpr(rn, R9);
            }
        }
        true
    }

    pub(crate) fn arm_extra_transfer(&mut self, a: u32, op: u32) -> bool {
        let rn = (op >> 16) & 0xF;
        let rd = (op >> 12) & 0xF;
        let pre = (op & (1 << 24)) != 0;
        let up = (op & (1 << 23)) != 0;
        let load = (op & (1 << 20)) != 0;
        let writeback = !pre || (op & (1 << 21)) != 0;
        let kind = (op >> 5) & 3;
        let pair = kind >= 2 && !load;
        if (writeback && rn == 15) || (load && rd == 15) || (pair && ((rd & 1) != 0 || rd == 14)) {
            return self.emit_interpret(a, op);
        }

        // Compute `r8d` = effective address and `r9d` = writeback address `oa`.
        self.load_gpr(R8, rn, a + 8);
        self.buf.mov_r32_r32(R9, R8);
        if (op & (1 << 22)) != 0 {
            let imm = ((op >> 4) & 0xF0) | (op & 0xF);
            let delta = if up { imm as i32 } else { (imm as i32).wrapping_neg() };
            if delta != 0 {
                self.buf.alu_r32_imm32(AluOp::Add, R9, delta);
            }
        } else {
            self.load_gpr(RAX, op & 0xF, a + 8);
            self.buf.alu_r32_r32(if up { AluOp::Add } else { AluOp::Sub }, R9, RAX);
        }
        if pre || !writeback {
            self.buf.mov_r32_r32(R8, R9);
        }

        match (kind, load) {
            (1, true) => {
                self.emit_read16(R8, RAX, false);
                if writeback && rn != rd {
                    self.store_gpr(rn, R9);
                }
                self.store_gpr(rd, RAX);
            }
            (1, false) => {
                self.load_gpr(RSI, rd, a + 8);
                self.emit_write16(R8, RSI);
                if writeback {
                    self.store_gpr(rn, R9);
                }
            }
            (2, true) => {
                self.emit_read8(R8, RAX, true);
                if writeback && rn != rd {
                    self.store_gpr(rn, R9);
                }
                self.store_gpr(rd, RAX);
            }
            (_, true) => {
                self.emit_read16(R8, RAX, true);
                if writeback && rn != rd {
                    self.store_gpr(rn, R9);
                }
                self.store_gpr(rd, RAX);
            }
            (2, false) => {
                // LDRD rd, rd+1: write back base before writing destination pair so `rn == rd` / `rd+1`
                // retains the loaded values.
                if writeback {
                    self.store_gpr(rn, R9);
                }
                self.emit_read32(R8, RAX);
                self.store_gpr(rd, RAX);
                self.buf.alu_r32_imm32(AluOp::Add, R8, 4);
                self.emit_read32(R8, RAX);
                self.store_gpr(rd + 1, RAX);
            }
            (_, false) => {
                // STRD rd, rd+1
                self.load_gpr(RSI, rd, a + 8);
                self.emit_write32(R8, RSI);
                self.buf.alu_r32_imm32(AluOp::Add, R8, 4);
                self.load_gpr(RSI, rd + 1, a + 8);
                self.emit_write32(R8, RSI);
                if writeback {
                    self.store_gpr(rn, R9);
                }
            }
        }
        true
    }

    /// Emits a fast single-page span transfer with per-word fallback for `LDM`/`STM`/`VPUSH`/`VPOP`.
    /// `r8d` must hold starting guest address `p`. Both `r8` and `r9` are preserved.
    pub(crate) fn emit_words_span(&mut self, load: bool, regs: &[SpanWord]) {
        if regs.is_empty() {
            return;
        }
        if let [single] = regs {
            match *single {
                SpanWord::Gpr(r) if load => {
                    self.emit_read32(R8, RAX);
                    self.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                }
                SpanWord::Gpr(r) => {
                    self.buf.ldr_r32_mem(RSI, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                    self.emit_write32(R8, RSI);
                }
                SpanWord::Const(val) => {
                    self.buf.mov_r32_imm32(RSI, val);
                    self.emit_write32(R8, RSI);
                }
                SpanWord::Vfp(s) if load => {
                    self.emit_read32(R8, RAX);
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
                    self.buf.str_r32_mem(RAX, RCX, (s as i32) * 4);
                }
                SpanWord::Vfp(s) => {
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
                    self.buf.ldr_r32_mem(RSI, RCX, (s as i32) * 4);
                    self.emit_write32(R8, RSI);
                }
            }
            return;
        }

        let bytes = (regs.len() * 4) as i32;
        let pages_reg = if load { R12 } else { R13 };

        self.buf.mov_r32_r32(RAX, R8);
        self.buf.alu_r32_imm32(AluOp::And, RAX, 0xFFF);
        self.buf.mov_r32_r32(RCX, RAX);
        self.buf.alu_r32_imm32(AluOp::Add, RCX, bytes - 1);
        self.buf.alu_r32_imm32(AluOp::Cmp, RCX, 0xFFF);
        let j_slow1 = self.buf.jcc_rel32(cc::A);

        self.buf.mov_r32_r32(RCX, R8);
        self.buf.shift_r32_imm(ShiftOp::Shr, RCX, 12);
        self.buf.ldr_r64_sib(RDX, pages_reg, RCX, 8, 0);
        self.buf.test_r64_r64(RDX, RDX);
        let j_slow2 = self.buf.jcc_rel32(cc::Z);

        self.buf.alu_r64_r64(AluOp::Add, RDX, RAX);
        let has_vfp = regs.iter().any(|w| matches!(w, SpanWord::Vfp(_)));
        if has_vfp {
            self.buf.ldr_r64_mem(RDI, RBX, offsets::CTX_VFP as i32);
        }

        for (i, item) in regs.iter().enumerate() {
            let off = (i * 4) as i32;
            match *item {
                SpanWord::Gpr(r) if load => {
                    self.buf.ldr_r32_mem(RAX, RDX, off);
                    self.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                }
                SpanWord::Gpr(r) => {
                    self.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                    self.buf.str_r32_mem(RAX, RDX, off);
                }
                SpanWord::Const(val) => {
                    self.buf.str_imm32_mem(RDX, off, val);
                }
                SpanWord::Vfp(s) if load => {
                    self.buf.ldr_r32_mem(RAX, RDX, off);
                    self.buf.str_r32_mem(RAX, RDI, (s as i32) * 4);
                }
                SpanWord::Vfp(s) => {
                    self.buf.ldr_r32_mem(RAX, RDI, (s as i32) * 4);
                    self.buf.str_r32_mem(RAX, RDX, off);
                }
            }
        }
        let j_done = self.buf.jmp_rel32();

        let slow_pos = self.buf.pos();
        self.buf.patch_rel32(j_slow1, slow_pos);
        self.buf.patch_rel32(j_slow2, slow_pos);
        for (i, item) in regs.iter().enumerate() {
            let off = (i * 4) as i32;
            self.buf.mov_r32_r32(R10, R8);
            if off != 0 {
                self.buf.alu_r32_imm32(AluOp::Add, R10, off);
            }
            match *item {
                SpanWord::Gpr(r) if load => {
                    self.emit_read32(R10, RAX);
                    self.buf.str_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                }
                SpanWord::Gpr(r) => {
                    self.buf.ldr_r32_mem(RSI, RBX, (offsets::CTX_R0 as i32) + (r as i32) * 4);
                    self.emit_write32(R10, RSI);
                }
                SpanWord::Const(val) => {
                    self.buf.mov_r32_imm32(RSI, val);
                    self.emit_write32(R10, RSI);
                }
                SpanWord::Vfp(s) if load => {
                    self.emit_read32(R10, RAX);
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
                    self.buf.str_r32_mem(RAX, RCX, (s as i32) * 4);
                }
                SpanWord::Vfp(s) => {
                    self.buf.ldr_r64_mem(RCX, RBX, offsets::CTX_VFP as i32);
                    self.buf.ldr_r32_mem(RSI, RCX, (s as i32) * 4);
                    self.emit_write32(R10, RSI);
                }
            }
        }
        let done_pos = self.buf.pos();
        self.buf.patch_rel32(j_done, done_pos);
    }

    pub(crate) fn arm_block_transfer(&mut self, a: u32, op: u32) -> bool {
        let rn = (op >> 16) & 0xF;
        let pre = (op & (1 << 24)) != 0;
        let up = (op & (1 << 23)) != 0;
        let writeback = (op & (1 << 21)) != 0;
        let load = (op & (1 << 20)) != 0;
        let list = op & 0xFFFF;
        if (op & (1 << 22)) != 0 || list == 0 || rn == 15 {
            return self.emit_interpret(a, op);
        }
        let span = list.count_ones() * 4;
        let p_delta: u32 = match (up, pre) {
            (true, false) => 0,
            (true, true) => 4,
            (false, false) => (span - 4).wrapping_neg(),
            (false, true) => span.wrapping_neg(),
        };
        let fin_delta: u32 = if up { span } else { span.wrapping_neg() };

        // `r8d` = starting guest address `p`.
        self.load_gpr(R8, rn, a + 8);
        if p_delta != 0 {
            self.buf.alu_r32_imm32(AluOp::Add, R8, p_delta as i32);
        }

        let registers: Vec<u32> = (0..16).filter(|i| (list & (1 << i)) != 0).collect();
        if load {
            if writeback && (list & (1 << rn)) == 0 {
                self.load_gpr(R9, rn, a + 8);
                if fin_delta != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R9, fin_delta as i32);
                }
                self.store_gpr(rn, R9);
            }
            let words: Vec<SpanWord> = registers.iter().map(|&r| SpanWord::Gpr(r as u8)).collect();
            self.emit_words_span(true, &words);
            if (list & (1 << 15)) != 0 {
                self.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
                return if rn == 13 {
                    self.emit_return_to_reg(RAX)
                } else {
                    self.emit_jump_to_reg(RAX)
                };
            }
        } else {
            let rn_in_non_lowest =
                writeback && (list & (1 << rn)) != 0 && (list & ((1 << rn) - 1)) != 0;
            if writeback {
                self.load_gpr(R9, rn, a + 8);
                if fin_delta != 0 {
                    self.buf.alu_r32_imm32(AluOp::Add, R9, fin_delta as i32);
                }
                if rn_in_non_lowest {
                    // Temporarily stash `fin` in `ctx.r[15]` (guest PC is static inside an AOT block).
                    self.buf.str_r32_mem(R9, RBX, (offsets::CTX_R0 as i32) + 15 * 4);
                }
            }
            let words: Vec<SpanWord> = registers
                .iter()
                .map(|&r| {
                    if r == 15 {
                        SpanWord::Const(a + 12)
                    } else if r == rn && rn_in_non_lowest {
                        SpanWord::Gpr(15)
                    } else {
                        SpanWord::Gpr(r as u8)
                    }
                })
                .collect();
            self.emit_words_span(false, &words);
            if writeback {
                self.store_gpr(rn, R9);
            }
        }
        true
    }

    pub(crate) fn arm_swap(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let rn = (op >> 16) & 0xF;
        let rm = op & 0xF;
        let byte = (op & (1 << 22)) != 0;

        self.load_gpr(R8, rn, a + 8);
        // Save `source` (`rm`) before writing `rd` in case `rd == rm` or `rd == rn`.
        self.load_gpr(R9, rm, a + 8);
        if byte {
            self.emit_read8(R8, RAX, false);
            self.store_gpr(rd, RAX);
            self.buf.mov_r32_r32(RSI, R9);
            self.emit_write8(R8, RSI);
        } else {
            self.emit_read32(R8, RAX);
            self.store_gpr(rd, RAX);
            self.buf.mov_r32_r32(RSI, R9);
            self.emit_write32(R8, RSI);
        }
        true
    }

    pub(crate) fn arm_exclusive(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        let rm = op & 0xF;
        let rn = (op >> 16) & 0xF;
        let size = (op >> 21) & 3;
        let load = (op & (1 << 20)) != 0;
        if rd == 15 || (size == 0b01 && (if load { rd } else { rm }) >= 14) {
            return self.emit_interpret(a, op);
        }

        self.load_gpr(R8, rn, a + 8);
        if load {
            self.buf.str_imm8_mem(RBX, offsets::CTX_EXCLUSIVE as i32, 1);
            self.buf.str_r32_mem(R8, RBX, offsets::CTX_EXCL_ADDR as i32);
            match size {
                0b00 => {
                    self.emit_read32(R8, RAX);
                    self.store_gpr(rd, RAX);
                }
                0b01 => {
                    self.emit_read32(R8, RAX);
                    self.store_gpr(rd, RAX);
                    self.buf.alu_r32_imm32(AluOp::Add, R8, 4);
                    self.emit_read32(R8, RAX);
                    self.store_gpr(rd + 1, RAX);
                }
                0b10 => {
                    self.emit_read8(R8, RAX, false);
                    self.store_gpr(rd, RAX);
                }
                _ => {
                    self.emit_read16(R8, RAX, false);
                    self.store_gpr(rd, RAX);
                }
            }
        } else {
            self.buf.ldr_u8_mem(RAX, RBX, offsets::CTX_EXCLUSIVE as i32);
            self.buf.test_r32_r32(RAX, RAX);
            let j_fail1 = self.buf.jcc_rel32(cc::Z);
            self.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_EXCL_ADDR as i32);
            self.buf.alu_r32_r32(AluOp::Cmp, RAX, R8);
            let j_fail2 = self.buf.jcc_rel32(cc::NE);

            match size {
                0b00 => {
                    self.load_gpr(RSI, rm, a + 8);
                    self.emit_write32(R8, RSI);
                }
                0b01 => {
                    self.load_gpr(RSI, rm, a + 8);
                    self.emit_write32(R8, RSI);
                    self.buf.alu_r32_imm32(AluOp::Add, R8, 4);
                    self.load_gpr(RSI, rm + 1, a + 8);
                    self.emit_write32(R8, RSI);
                }
                0b10 => {
                    self.load_gpr(RSI, rm, a + 8);
                    self.emit_write8(R8, RSI);
                }
                _ => {
                    self.load_gpr(RSI, rm, a + 8);
                    self.emit_write16(R8, RSI);
                }
            }
            self.buf.str_imm8_mem(RBX, offsets::CTX_EXCLUSIVE as i32, 0);
            self.buf.mov_r32_imm32(RAX, 0);
            self.store_gpr(rd, RAX);
            let j_done = self.buf.jmp_rel32();

            let fail_pos = self.buf.pos();
            self.buf.patch_rel32(j_fail1, fail_pos);
            self.buf.patch_rel32(j_fail2, fail_pos);
            // ARMv6K: executing STREX always clears the local exclusive monitor.
            self.buf.str_imm8_mem(RBX, offsets::CTX_EXCLUSIVE as i32, 0);
            self.buf.mov_r32_imm32(RAX, 1);
            self.store_gpr(rd, RAX);

            let done_pos = self.buf.pos();
            self.buf.patch_rel32(j_done, done_pos);
        }
        true
    }

    pub(crate) fn arm_media(&mut self, a: u32, op: u32) -> bool {
        let op1 = (op >> 20) & 0x1F;
        let op2 = (op >> 5) & 7;
        match op1 >> 3 {
            0b00 => self.arm_media_parallel(a, op, op1, op2),
            0b01 => self.arm_pack_saturate_extend(a, op, op1, op2),
            0b10 => self.arm_dual_multiply(a, op, op1),
            _ if op1 == 0b11000 && op2 == 0 => self.arm_usad8(a, op),
            _ => self.emit_interpret(a, op),
        }
    }

    fn arm_media_parallel(&mut self, a: u32, op: u32, op1: u32, op2: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 || (op1 & 3) == 0 || matches!(op2, 5 | 6) {
            return self.emit_interpret(a, op);
        }
        let rn = (op >> 16) & 0xF;
        let rm = op & 0xF;
        self.load_gpr(R10, rn, a + 8);
        self.load_gpr(R11, rm, a + 8);
        self.buf.mov_r32_imm32(RAX, (op1 << 3) | op2);
        self.stub_call(StubId::MediaParallel);
        self.store_gpr(rd, RAX);
        true
    }

    fn arm_pack_saturate_extend(&mut self, a: u32, op: u32, op1: u32, op2: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let rn = (op >> 16) & 0xF;
        let shift = ((op >> 7) & 0x1F) as u8;
        let unsigned = (op1 & 0b100) != 0;

        match (op1 & 7, op2) {
            (0b000, 0 | 2 | 4 | 6) if op1 == 0b01000 => {
                // PKHBT / PKHTB
                self.load_gpr(RAX, rn, a + 8);
                self.load_gpr(RCX, op & 0xF, a + 8);
                if (op & (1 << 6)) != 0 {
                    let s = if shift == 0 { 31 } else { shift };
                    self.buf.shift_r32_imm(ShiftOp::Sar, RCX, s);
                    self.buf.movzx_r32_r16(RCX, RCX);
                    self.buf.alu_r32_imm32(AluOp::And, RAX, 0xFFFF_0000_u32 as i32);
                    self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                } else {
                    self.buf.movzx_r32_r16(RAX, RAX);
                    if shift != 0 {
                        self.buf.shift_r32_imm(ShiftOp::Shl, RCX, shift);
                    }
                    self.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFFF_0000_u32 as i32);
                    self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                }
                self.store_gpr(rd, RAX);
            }
            (_, 0b011) => {
                // SXTB16/UXTB16/SXTB/SXTH/UXTB/UXTH (+ accumulate with Rn if Rn != 15)
                self.load_gpr(RAX, op & 0xF, a + 8);
                let rotate = (((op >> 10) & 3) * 8) as u8;
                if rotate != 0 {
                    self.buf.shift_r32_imm(ShiftOp::Ror, RAX, rotate);
                }
                let packed = match op1 & 7 {
                    0b000 => {
                        // SXTB16
                        self.buf.movsx_r32_r8(RCX, RAX);
                        self.buf.movzx_r32_r16(RCX, RCX);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RAX, 16);
                        self.buf.movsx_r32_r8(RAX, RAX);
                        self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 16);
                        self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                        true
                    }
                    0b010 => {
                        self.buf.movsx_r32_r8(RAX, RAX);
                        false
                    }
                    0b011 => {
                        self.buf.movsx_r32_r16(RAX, RAX);
                        false
                    }
                    0b100 => {
                        self.buf.alu_r32_imm32(AluOp::And, RAX, 0x00FF_00FF);
                        true
                    }
                    0b110 => {
                        self.buf.movzx_r32_r8(RAX, RAX);
                        false
                    }
                    0b111 => {
                        self.buf.movzx_r32_r16(RAX, RAX);
                        false
                    }
                    _ => return self.emit_interpret(a, op),
                };
                match (rn, packed) {
                    (15, _) => self.store_gpr(rd, RAX),
                    (_, true) => {
                        // Halfword-wise wrapping add `(rn + ext) & 0xFFFF | (((rn >> 16) + (ext >> 16)) << 16)`
                        self.load_gpr(RCX, rn, a + 8);
                        self.buf.mov_r32_r32(RDX, RCX);
                        self.buf.alu_r32_r32(AluOp::Add, RDX, RAX);
                        self.buf.movzx_r32_r16(RDX, RDX);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RCX, 16);
                        self.buf.shift_r32_imm(ShiftOp::Shr, RAX, 16);
                        self.buf.alu_r32_r32(AluOp::Add, RAX, RCX);
                        self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 16);
                        self.buf.alu_r32_r32(AluOp::Or, RAX, RDX);
                        self.store_gpr(rd, RAX);
                    }
                    (_, false) => {
                        self.load_gpr(RCX, rn, a + 8);
                        self.buf.alu_r32_r32(AluOp::Add, RAX, RCX);
                        self.store_gpr(rd, RAX);
                    }
                }
            }
            (0b000, 0b101) if op1 == 0b01000 => {
                // SEL Rd, Rn, Rm: expand GE[3:0] to 4-byte mask, then `((n ^ m) & mask) ^ m`.
                self.buf.ldr_u8_mem(RAX, RBX, offsets::CTX_GE as i32);
                self.buf.alu_r32_imm32(AluOp::And, RAX, 0xF);
                self.buf.mov_r32_imm32(RCX, 0x0020_4081);
                self.buf.imul_r32_r32(RAX, RCX);
                self.buf.alu_r32_imm32(AluOp::And, RAX, 0x0101_0101);
                self.buf.mov_r32_r32(RCX, RAX);
                self.buf.shift_r32_imm(ShiftOp::Shl, RCX, 8);
                self.buf.alu_r32_r32(AluOp::Sub, RCX, RAX);
                self.load_gpr(RAX, rn, a + 8);
                self.load_gpr(RDX, op & 0xF, a + 8);
                self.buf.alu_r32_r32(AluOp::Xor, RAX, RDX);
                self.buf.alu_r32_r32(AluOp::And, RAX, RCX);
                self.buf.alu_r32_r32(AluOp::Xor, RAX, RDX);
                self.store_gpr(rd, RAX);
            }
            (0b011, 0b001) if op1 == 0b01011 => {
                // REV
                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.bswap_r32(RAX);
                self.store_gpr(rd, RAX);
            }
            (0b011, 0b101) if op1 == 0b01011 => {
                // REV16
                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.bswap_r32(RAX);
                self.buf.shift_r32_imm(ShiftOp::Ror, RAX, 16);
                self.store_gpr(rd, RAX);
            }
            (0b111, 0b001) if op1 == 0b01111 => {
                // RBIT
                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.bswap_r32(RAX);
                for &(lo_mask, hi_mask, sh) in &[
                    (0x0F0F_0F0Fu32, 0xF0F0_F0F0u32, 4u8),
                    (0x3333_3333u32, 0xCCCC_CCCCu32, 2u8),
                    (0x5555_5555u32, 0xAAAA_AAAAu32, 1u8),
                ] {
                    self.buf.mov_r32_r32(RCX, RAX);
                    self.buf.alu_r32_imm32(AluOp::And, RAX, lo_mask as i32);
                    self.buf.shift_r32_imm(ShiftOp::Shl, RAX, sh);
                    self.buf.alu_r32_imm32(AluOp::And, RCX, hi_mask as i32);
                    self.buf.shift_r32_imm(ShiftOp::Shr, RCX, sh);
                    self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                }
                self.store_gpr(rd, RAX);
            }
            (0b111, 0b101) if op1 == 0b01111 => {
                // REVSH
                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.bswap_r32(RAX);
                self.buf.shift_r32_imm(ShiftOp::Sar, RAX, 16);
                self.store_gpr(rd, RAX);
            }
            (0b010 | 0b110, 0b001) => {
                // SSAT16 / USAT16
                let bits = ((op >> 16) & 0xF) + (!unsigned as u32);
                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.movsx_r32_r16(R8, RAX);
                self.buf.movsxd_r64_r32(R8, R8);
                self.emit_saturate(RAX, R8, bits, unsigned);
                self.buf.movzx_r32_r16(RAX, RAX);
                self.buf.str_r32_mem(RAX, RBX, offsets::SCRATCH_FP as i32);

                self.load_gpr(RAX, op & 0xF, a + 8);
                self.buf.shift_r32_imm(ShiftOp::Sar, RAX, 16);
                self.buf.movsxd_r64_r32(R8, RAX);
                self.emit_saturate(RAX, R8, bits, unsigned);
                self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 16);
                self.buf.ldr_r32_mem(RCX, RBX, offsets::SCRATCH_FP as i32);
                self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
                self.store_gpr(rd, RAX);
            }
            (_, o) if (o & 1) == 0 && matches!((op1 >> 2) & 0b111, 0b010 | 0b011) => {
                // SSAT / USAT
                let bits = ((op >> 16) & 0x1F) + (!unsigned as u32);
                self.load_gpr(RAX, op & 0xF, a + 8);
                if (op & (1 << 6)) != 0 {
                    let s = if shift == 0 { 31 } else { shift };
                    self.buf.shift_r32_imm(ShiftOp::Sar, RAX, s);
                } else if shift != 0 {
                    self.buf.shift_r32_imm(ShiftOp::Shl, RAX, shift);
                }
                self.buf.movsxd_r64_r32(R8, RAX);
                self.emit_saturate(RAX, R8, bits, unsigned);
                self.store_gpr(rd, RAX);
            }
            _ => return self.emit_interpret(a, op),
        }
        true
    }

    fn arm_dual_multiply(&mut self, a: u32, op: u32, op1: u32) -> bool {
        let rd = (op >> 16) & 0xF;
        let ra = (op >> 12) & 0xF;
        if rd == 15 || (op1 == 0b10100 && ra == 15) {
            return self.emit_interpret(a, op);
        }
        let rs = (op >> 8) & 0xF;
        let rm = op & 0xF;
        let bit5 = (op & (1 << 5)) != 0;
        let subtract = (op & (1 << 6)) != 0;

        match op1 {
            0b10000 | 0b10100 => {
                // SMLAD / SMUAD / SMLSD / SMUSD / SMLALD / SMLSLD
                self.load_gpr(RAX, rm, a + 8);
                self.load_gpr(RCX, rs, a + 8);
                if bit5 {
                    self.buf.shift_r32_imm(ShiftOp::Ror, RCX, 16);
                }
                // p0 = (i16)m * (i16)b in r8 (i64):
                self.buf.movsx_r32_r16(RDX, RAX);
                self.buf.movsx_r32_r16(RSI, RCX);
                self.buf.imul_r32_r32(RDX, RSI);
                self.buf.movsxd_r64_r32(R8, RDX);
                // p1 = (i16)(m >> 16) * (i16)(b >> 16) in rdx (i64):
                self.buf.shift_r32_imm(ShiftOp::Sar, RAX, 16);
                self.buf.shift_r32_imm(ShiftOp::Sar, RCX, 16);
                self.buf.imul_r32_r32(RAX, RCX);
                self.buf.movsxd_r64_r32(RDX, RAX);
                if subtract {
                    self.buf.alu_r64_r64(AluOp::Sub, R8, RDX);
                } else {
                    self.buf.alu_r64_r64(AluOp::Add, R8, RDX);
                }
                if op1 == 0b10000 {
                    if ra != 15 {
                        self.load_gpr(RAX, ra, a + 8);
                        self.buf.movsxd_r64_r32(RAX, RAX);
                        self.buf.alu_r64_r64(AluOp::Add, R8, RAX);
                    }
                    self.emit_accumulate_q(RAX, R8);
                    self.store_gpr(rd, RAX);
                } else {
                    self.load_gpr(RAX, ra, a + 8);
                    self.load_gpr(RDX, rd, a + 8);
                    self.buf.shift_r64_imm(ShiftOp::Shl, RDX, 32);
                    self.buf.alu_r64_r64(AluOp::Or, RAX, RDX);
                    self.buf.alu_r64_r64(AluOp::Add, RAX, R8);
                    self.store_gpr(ra, RAX);
                    self.buf.shift_r64_imm(ShiftOp::Shr, RAX, 32);
                    self.store_gpr(rd, RAX);
                }
            }
            0b10101 => {
                // SMMUL(R) / SMMLA(R) / SMMLS(R)
                self.load_gpr(RAX, rm, a + 8);
                self.buf.movsxd_r64_r32(RAX, RAX);
                self.load_gpr(RCX, rs, a + 8);
                self.buf.movsxd_r64_r32(RCX, RCX);
                self.buf.imul_r64_r64(RAX, RCX);
                if ra != 15 {
                    self.load_gpr(RCX, ra, a + 8);
                    self.buf.shift_r64_imm(ShiftOp::Shl, RCX, 32);
                    if subtract {
                        self.buf.alu_r64_r64(AluOp::Sub, RCX, RAX);
                        self.buf.mov_r64_r64(RAX, RCX);
                    } else {
                        self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                    }
                } else if subtract {
                    self.buf.mov_r32_imm32(RCX, 0);
                    self.buf.alu_r64_r64(AluOp::Sub, RCX, RAX);
                    self.buf.mov_r64_r64(RAX, RCX);
                }
                if bit5 {
                    self.buf.mov_r32_imm32(RCX, 0x8000_0000);
                    self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                }
                self.buf.shift_r64_imm(ShiftOp::Shr, RAX, 32);
                self.store_gpr(rd, RAX);
            }
            _ => return self.emit_interpret(a, op),
        }
        true
    }

    fn arm_usad8(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 16) & 0xF;
        let ra = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        self.load_gpr(RCX, op & 0xF, a + 8);
        self.load_gpr(RDX, (op >> 8) & 0xF, a + 8);
        self.buf.mov_r32_imm32(RAX, 0);
        for i in 0..4u8 {
            self.buf.mov_r32_r32(RSI, RCX);
            if i != 0 {
                self.buf.shift_r32_imm(ShiftOp::Shr, RSI, i * 8);
            }
            self.buf.alu_r32_imm32(AluOp::And, RSI, 0xFF);
            self.buf.mov_r32_r32(RDI, RDX);
            if i != 0 {
                self.buf.shift_r32_imm(ShiftOp::Shr, RDI, i * 8);
            }
            self.buf.alu_r32_imm32(AluOp::And, RDI, 0xFF);
            self.buf.alu_r32_r32(AluOp::Sub, RSI, RDI);
            self.buf.mov_r32_r32(RDI, RSI);
            self.buf.neg_r32(RSI);
            self.buf.cmovcc_r32_r32(cc::S, RSI, RDI);
            self.buf.alu_r32_r32(AluOp::Add, RAX, RSI);
        }
        if ra != 15 {
            self.load_gpr(RCX, ra, a + 8);
            self.buf.alu_r32_r32(AluOp::Add, RAX, RCX);
        }
        self.store_gpr(rd, RAX);
        true
    }
}
