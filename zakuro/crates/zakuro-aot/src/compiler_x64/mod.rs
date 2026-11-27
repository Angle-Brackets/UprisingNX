//! Direct ARMv6K + Thumb-1 + VFPv2 -> x86_64 Ahead-of-Time (AOT) compiler.
//!
//! Register allocation inside x86_64 AOT execution (`extern "sysv64"`):
//! - `rbx`:   `*mut AotExecContext` (`offset 0` is `recomp_abi::Context`, so `[rbx + CTX_R0 + i*4]` holds `r[0..15]`)
//! - `r12`:   `ctx.read_pages` (`*const *mut u8`)
//! - `r13`:   `ctx.write_pages` (`*const *mut u8`)
//! - `r14d`:  `ctx.budget` (`i32`)
//! - `r15d`:  Packed guest `NZCV` in bits `[31:28]` (`N=31, Z=30, C=29, V=28`), bits `[27:0] == 0`
//! - `rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11`: Scratch GPRs (caller-saved; `r8d`/`r9d`/`r10d` often hold fast-path args/addrs, `r11d` scratch)
//! - `xmm0..xmm7`: Scalar SSE2 float/double scratch (`vfp` array lives at `[rbx + CTX_VFP]`)
//! - `rsp`: 16-byte aligned inside every AOT function body (`push rbp; mov rbp, rsp` on entry).

#![allow(dead_code, unused_variables, unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use zakuro_fs::ExHeader;

use crate::discover::{Function, Image, Mode, Program, Segment};
use crate::runtime::{offsets, AotImage, CompileOptions};
use crate::x64::{AluOp, ShiftOp, X64Buf, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, cc, XMM0, XMM1, XMM2};

mod mem_media;
mod vfp_thumb;

const RECOMP_DEPTH_LIMIT: u16 = 2048;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StubId {
    SyncOut,
    SyncIn,
    SlowRead8,
    SlowRead16,
    SlowRead32,
    SlowWrite8,
    SlowWrite16,
    SlowWrite32,
    IndirectCall,
    IndirectJump,
    Interpret,
    MediaParallel,
    VfpVector,
    VfpToInt,
    ShiftLslC,
    ShiftLsrC,
    ShiftAsrC,
    ShiftRorC,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RelocKind {
    StubJmp(StubId),
    StubCall(StubId),
    DirectCall(u32),
    DirectTailJump(u32),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Reloc {
    pub local_disp_pos: u32,
    pub kind: RelocKind,
}

pub(crate) struct CompiledFunc {
    pub entry: u32,
    pub mode: Mode,
    pub buf: X64Buf,
    pub label_bytes: Vec<(u32, u32)>,
    pub relocs: Vec<Reloc>,
    pub spans: Vec<(u32, u32)>,
}

pub(crate) struct LocalPatch {
    pub disp_pos: usize,
    pub target_label: u32,
    pub cc: Option<u8>,
}

pub(crate) struct BudgetExit {
    pub disp_pos: usize,
    pub guest_addr: u32,
    pub count: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum SpanWord {
    Gpr(u8),
    Const(u32),
    Vfp(u8),
}

pub(crate) struct FnEmitter<'a> {
    pub text: &'a Segment,
    pub known_targets: &'a BTreeSet<u32>,
    pub labels: BTreeSet<u32>,
    pub mode: Mode,
    pub options: CompileOptions,
    pub buf: X64Buf,
    pub relocs: Vec<Reloc>,
    pub local_patches: Vec<LocalPatch>,
    pub epilogue_patches: Vec<usize>,
    pub depth_overflow_patches: Vec<usize>,
    pub budget_exits: Vec<BudgetExit>,
    pub interpreted_addrs: Vec<u32>,
    pub folded_spans: Vec<(u32, u32)>,
}

pub fn compile_title(
    program_id: u64,
    exheader: &ExHeader,
    code: &[u8],
    override_addrs: &[u32],
    hints: &[u32],
    options: CompileOptions,
    progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    let image = Image::from_exheader(exheader, code);
    let text_hash = image.text.hash();
    let program = image.into_program(override_addrs, hints);
    compile_program(program_id, text_hash, &program, override_addrs, options, progress)
}

pub fn compile_program(
    _program_id: u64,
    _text_hash: u64,
    _program: &Program,
    _override_addrs: &[u32],
    _options: CompileOptions,
    _progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    unimplemented!("x86_64 AOT compiler backend will be filled by Wave 1 workers")
}

impl FnEmitter<'_> {
    #[inline]
    pub(crate) fn load_gpr(&mut self, host_r32: u8, guest_reg: u32, pc_val: u32) {
        if guest_reg == 15 {
            self.buf.mov_r32_imm32(host_r32, pc_val);
        } else {
            self.buf.ldr_r32_mem(host_r32, RBX, (offsets::CTX_R0 as i32) + (guest_reg as i32) * 4);
        }
    }

    #[inline]
    pub(crate) fn store_gpr(&mut self, guest_reg: u32, host_r32: u8) {
        debug_assert!(guest_reg < 15);
        self.buf.str_r32_mem(host_r32, RBX, (offsets::CTX_R0 as i32) + (guest_reg as i32) * 4);
    }

    #[inline]
    pub(crate) fn stub_call(&mut self, id: StubId) {
        let disp = self.buf.call_rel32();
        self.relocs.push(Reloc { local_disp_pos: disp as u32, kind: RelocKind::StubCall(id) });
    }

    #[inline]
    pub(crate) fn stub_jump(&mut self, id: StubId) {
        let disp = self.buf.jmp_rel32();
        self.relocs.push(Reloc { local_disp_pos: disp as u32, kind: RelocKind::StubJmp(id) });
    }

    pub(crate) fn emit_cond_skip(&mut self, _arm_cond: u32) -> Option<usize> {
        unimplemented!()
    }

    /// Emits a conditional local jump (`arm_cond` 0..=13) to `target_label` inside the current function.
    pub(crate) fn emit_cond_local_branch(&mut self, _arm_cond: u32, _target_label: u32) {
        unimplemented!()
    }

    pub(crate) fn patch_cond_skip(&mut self, skip: Option<usize>) {
        if let Some(disp) = skip {
            self.buf.patch_rel32(disp, self.buf.pos());
        }
    }

    /// Captures live x86 `SF, ZF, CF, OF` after an `ADD`/`ADC` operation into `r15d[31:28]`
    /// (ARM `C = x86 CF`). Clobbers `rax, rcx, rdx, rsi`.
    pub(crate) fn set_nzcv_add(&mut self) {
        unimplemented!()
    }

    /// Captures live x86 `SF, ZF, CF, OF` after a `SUB`/`CMP`/`SBC`/`RSC`/`NEG` operation into
    /// `r15d[31:28]` (ARM `C = !x86 CF`). Clobbers `rax, rcx, rdx, rsi`.
    pub(crate) fn set_nzcv_sub(&mut self) {
        unimplemented!()
    }

    pub(crate) fn emit_epilogue_jump(&mut self) {
        let p = self.buf.jmp_rel32();
        self.epilogue_patches.push(p);
    }

    pub(crate) fn emit_check_unwind(&mut self) {
        unimplemented!()
    }

    pub(crate) fn emit_accumulate_q(&mut self, _rd: u8, _x_val: u8) {
        unimplemented!()
    }

    pub(crate) fn emit_saturate(&mut self, _rd: u8, _x_val: u8, _bits: u32, _unsigned: bool) {
        unimplemented!()
    }

    /// Memory fast/slow helpers.
    /// Contract:
    /// - Reads address from `addr_r32` (zero-extended; `addr_r32` may be any scratch reg, moved to `R10` if != `R10`).
    /// - Preserves `RBX`, `R12..R15`, **AND `R8` and `R9`** across BOTH fast and slow paths!
    /// - Clobbers `RAX, RCX, RDX, RSI, RDI, R10, R11` (and EFLAGS).
    /// - Output lands in `dst_r32` (never pass `R8`/`R9` as `dst_r32` unless intentional write; passing any of `RAX/RCX/RDX/RSI/RDI/R8/R9` works).
    pub(crate) fn emit_read8(&mut self, _addr_r32: u8, _dst_r32: u8, _signed: bool) {
        unimplemented!()
    }

    pub(crate) fn emit_read16(&mut self, _addr_r32: u8, _dst_r32: u8, _signed: bool) {
        unimplemented!()
    }

    pub(crate) fn emit_read32(&mut self, _addr_r32: u8, _dst_r32: u8) {
        unimplemented!()
    }

    /// `addr_r32` and `val_r32` (both 32-bit; `val_r32` must not be `RAX` if `addr_r32 != R10`—or helper handles safely; `R8` & `R9` are preserved!).
    pub(crate) fn emit_write8(&mut self, _addr_r32: u8, _val_r32: u8) {
        unimplemented!()
    }

    pub(crate) fn emit_write16(&mut self, _addr_r32: u8, _val_r32: u8) {
        unimplemented!()
    }

    pub(crate) fn emit_write32(&mut self, _addr_r32: u8, _val_r32: u8) {
        unimplemented!()
    }

    pub(crate) fn emit_call(&mut self, _link: u32, _target: u32, _callee_thumb: bool) -> bool {
        unimplemented!()
    }

    pub(crate) fn emit_jump(&mut self, _target: u32, _target_thumb: bool) -> bool {
        unimplemented!()
    }

    pub(crate) fn emit_return_to_reg(&mut self, _val_r32: u8) -> bool {
        unimplemented!()
    }

    pub(crate) fn emit_jump_to_reg(&mut self, _val_r32: u8) -> bool {
        unimplemented!()
    }

    pub(crate) fn emit_svc(&mut self, _next_pc: u32, _imm: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn emit_interpret(&mut self, _addr: u32, _op: u32) -> bool {
        unimplemented!()
    }

    /// Updates N and Z in `r15d` from live `SF, ZF` (after `test res, res` or `alu`) and sets C
    /// from `carry_r32` bit 0 if `Some(reg)` (or preserves old C if `None`), preserving V in `r15d`.
    pub(crate) fn update_logical_flags(&mut self, _carry_r32: Option<u8>) {
        unimplemented!()
    }
}
