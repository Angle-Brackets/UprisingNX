//! Direct ARMv6K + Thumb-1 + VFPv2 -> x86_64 Ahead-of-Time (AOT) compiler.
//!
//! Register allocation inside x86_64 AOT execution (`extern "sysv64"`):
//! - `rbx`:   `*mut AotExecContext` (`offset 0` is `recomp_abi::Context`, so `[rbx + CTX_R0 + i*4]` holds `r[0..15]`)
//! - `r12`:   `ctx.read_pages` (`*const *mut u8`)
//! - `r13`:   `ctx.write_pages` (`*const *mut u8`)
//! - `r14d`:  `ctx.budget` (`i32`)
//! - `r15d`:  Packed guest `NZCV` in bits `[31:28]` (`N=31, Z=30, C=29, V=28`), bits `[27:0] == 0`
//! - `rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11`: Scratch GPRs (caller-saved; `r8`/`r9` preserved across memory & shift helpers)
//! - `xmm0..xmm7`: Scalar SSE2 float/double scratch (`vfp` array lives at `[rbx + CTX_VFP]`)
//! - `rsp`: 16-byte aligned inside every AOT function body (`push rbp; mov rbp, rsp` on entry).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use recomp_abi::{EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND, NO_ORIGIN};
use zakuro_fs::ExHeader;

use crate::discover::{
    analyze, build_origins, decode_thumb, key, recompiles, Function, Image, Mode, Program, Segment,
};
use crate::runtime::{offsets, AotImage, CompileOptions};
use crate::x64::{
    cc, AluOp, ShiftOp, X64Buf, R10, R11, R12, R13, R14, R15, R8, R9, RAX, RBP, RBX, RCX, RDI, RDX,
    RSI, RSP,
};

mod mem_media;
mod vfp_thumb;

pub(crate) const RECOMP_DEPTH_LIMIT: u16 = 2048;

#[derive(Debug, Clone, Copy)]
pub(crate) enum StubId {
    #[allow(dead_code)]
    SyncOut,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    MediaParallel,
    #[allow(dead_code)]
    VfpVector,
    #[allow(dead_code)]
    VfpToInt,
    ShiftLslC,
    ShiftLsrC,
    ShiftAsrC,
    ShiftRorC,
}

struct RuntimeStubs {
    enter: usize,
    sync_out: usize,
    sync_in: usize,
    slow_read8: usize,
    slow_read16: usize,
    slow_read32: usize,
    slow_write8: usize,
    slow_write16: usize,
    slow_write32: usize,
    indirect_call: usize,
    indirect_jump: usize,
    interpret: usize,
    media_parallel: usize,
    vfp_vector: usize,
    vfp_to_int: usize,
    shift_lsl_c: usize,
    shift_lsr_c: usize,
    shift_asr_c: usize,
    shift_ror_c: usize,
}

impl RuntimeStubs {
    fn byte_of(&self, id: StubId) -> usize {
        match id {
            StubId::SyncOut => self.sync_out,
            StubId::SyncIn => self.sync_in,
            StubId::SlowRead8 => self.slow_read8,
            StubId::SlowRead16 => self.slow_read16,
            StubId::SlowRead32 => self.slow_read32,
            StubId::SlowWrite8 => self.slow_write8,
            StubId::SlowWrite16 => self.slow_write16,
            StubId::SlowWrite32 => self.slow_write32,
            StubId::IndirectCall => self.indirect_call,
            StubId::IndirectJump => self.indirect_jump,
            StubId::Interpret => self.interpret,
            StubId::MediaParallel => self.media_parallel,
            StubId::VfpVector => self.vfp_vector,
            StubId::VfpToInt => self.vfp_to_int,
            StubId::ShiftLslC => self.shift_lsl_c,
            StubId::ShiftLsrC => self.shift_lsr_c,
            StubId::ShiftAsrC => self.shift_asr_c,
            StubId::ShiftRorC => self.shift_ror_c,
        }
    }
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
    #[allow(dead_code)]
    pub cc: Option<u8>,
}

pub(crate) struct BudgetExit {
    pub disp_pos: usize,
    pub guest_addr: u32,
    pub count: u32,
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum SpanWord {
    Gpr(u8),
    Const(u32),
    Vfp(u8),
}

pub(crate) struct FnEmitter<'a> {
    #[allow(dead_code)]
    pub text: &'a Segment,
    pub known_targets: &'a BTreeSet<u32>,
    pub labels: BTreeSet<u32>,
    pub mode: Mode,
    #[allow(dead_code)]
    pub options: CompileOptions,
    pub buf: X64Buf,
    pub relocs: Vec<Reloc>,
    pub local_patches: Vec<LocalPatch>,
    pub epilogue_patches: Vec<usize>,
    pub depth_overflow_patches: Vec<(usize, Option<u32>)>,
    pub budget_exits: Vec<BudgetExit>,
    pub stale_check: Option<(usize, usize, u32, u32)>,
    pub interpreted_addrs: Vec<u32>,
    pub folded_spans: Vec<(u32, u32)>,
}

/// Compiles a 3DS title's decompressed `.code` image into a position-independent x86_64 `AotImage`.
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

/// Compiles a discovered [`Program`] into a position-independent x86_64 [`AotImage`].
pub fn compile_program(
    program_id: u64,
    text_hash: u64,
    program: &Program,
    override_addrs: &[u32],
    options: CompileOptions,
    progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    progress(0, 100);
    let analysis = analyze(program);
    progress(8, 100);

    let raw_functions: BTreeMap<u32, &Function> = analysis
        .functions
        .iter()
        .filter(|&(&entry, f)| recompiles(entry, f))
        .map(|(&e, f)| (e, f))
        .collect();

    // Global 1.0x instruction deduplication across shared tails / mid-function entries.
    let mut owner_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut global_labels: BTreeSet<u32> = BTreeSet::new();
    for (&entry, f) in &raw_functions {
        let f_key = key(entry, f.mode);
        owner_of.insert(f_key, f_key);
        for &lbl in &f.labels {
            global_labels.insert(key(lbl, f.mode));
        }
    }
    // Overridden functions claim their full instruction set first so raw body execution stays intact.
    for &ovr_key in override_addrs {
        let ovr_addr = ovr_key & !1;
        if let Some(f) = raw_functions.get(&ovr_addr) {
            let f_key = key(ovr_addr, f.mode);
            for &addr in &f.instructions {
                owner_of.entry(key(addr, f.mode)).or_insert(f_key);
            }
        }
    }
    for (&entry, f) in &raw_functions {
        let f_key = key(entry, f.mode);
        for &addr in &f.instructions {
            owner_of.entry(key(addr, f.mode)).or_insert(f_key);
        }
    }
    // Mark fallthroughs crossing an ownership boundary as explicit target labels.
    for f in raw_functions.values() {
        for &addr in &f.instructions {
            let a_key = key(addr, f.mode);
            let sz = if f.mode == Mode::Thumb {
                let op = program.text.read16(addr).unwrap_or(0);
                decode_thumb(op, program.text.read16(addr + 2), addr).1
            } else {
                4
            };
            let next = addr.wrapping_add(sz);
            let n_key = key(next, f.mode);
            if let (Some(&o1), Some(&o2)) = (owner_of.get(&a_key), owner_of.get(&n_key)) {
                if o1 != o2 {
                    global_labels.insert(n_key);
                }
            }
        }
    }

    let mut dedup_funcs: Vec<(u32, Function)> = Vec::with_capacity(raw_functions.len());
    let mut known_targets: BTreeSet<u32> = BTreeSet::new();
    for (&entry, f) in &raw_functions {
        let f_key = key(entry, f.mode);
        let instructions: Vec<u32> = f
            .instructions
            .iter()
            .copied()
            .filter(|&a| owner_of.get(&key(a, f.mode)) == Some(&f_key))
            .collect();
        if instructions.is_empty() {
            continue;
        }
        let labels: BTreeSet<u32> = instructions
            .iter()
            .copied()
            .filter(|&a| global_labels.contains(&key(a, f.mode)))
            .collect();
        if labels.contains(&entry) {
            known_targets.insert(f_key);
        }
        dedup_funcs.push((
            entry,
            Function { source: f.source, mode: f.mode, instructions, labels },
        ));
    }

    let total_funcs = dedup_funcs.len().max(1);
    let done_funcs = AtomicUsize::new(0);

    const CHUNK: usize = 256;
    let compile_slice = |(chunk_idx, slice): (usize, &[(u32, Function)])| -> Vec<CompiledFunc> {
        let base_idx = chunk_idx * CHUNK;
        let mut out = Vec::with_capacity(slice.len());
        for (i, (entry, func)) in slice.iter().enumerate() {
            out.push(compile_one_function(
                program,
                &known_targets,
                (base_idx + i) as u32,
                *entry,
                func,
                options,
            ));
        }
        let prev = done_funcs.fetch_add(slice.len(), Ordering::Relaxed) + slice.len();
        let pct = 10 + (prev * 82) / total_funcs;
        progress(pct.min(92), 100);
        out
    };
    let num_threads = std::thread::available_parallelism().map_or(3, |n| n.get().clamp(1, 16));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(2 * 1024 * 1024)
        .thread_name(|i| format!("aot-{i}"))
        .build();
    let compiled_chunks: Vec<Vec<CompiledFunc>> = match pool {
        Ok(pool) => pool.install(|| dedup_funcs.par_chunks(CHUNK).enumerate().map(compile_slice).collect()),
        Err(_) => dedup_funcs.chunks(CHUNK).enumerate().map(compile_slice).collect(),
    };
    let compiled_funcs: Vec<CompiledFunc> = compiled_chunks.into_iter().flatten().collect();

    let mut final_buf = X64Buf::with_capacity(
        4096 + compiled_funcs.iter().map(|f| f.buf.len() + 32).sum::<usize>(),
    );
    let stubs = emit_runtime_stubs(&mut final_buf);

    let mut override_map: BTreeMap<u32, usize> = BTreeMap::new();
    for (idx, &addr) in override_addrs.iter().take(8).enumerate() {
        override_map.insert(addr, idx);
    }

    let mut prologue_bytes: BTreeMap<u32, usize> = BTreeMap::new();
    let mut body_bytes: BTreeMap<u32, usize> = BTreeMap::new();
    let mut entries_map: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut func_relocs: Vec<(usize, Vec<Reloc>)> = Vec::with_capacity(compiled_funcs.len());
    let mut functions_spans: Vec<(u32, Vec<(u32, u32)>)> = Vec::with_capacity(compiled_funcs.len());

    for (func_idx, cf) in compiled_funcs.into_iter().enumerate() {
        let base_byte = final_buf.pos();
        final_buf.bytes.extend_from_slice(cf.buf.bytes());

        let f_key = key(cf.entry, cf.mode);
        let mut entry_local_byte: Option<usize> = None;
        for &(label_addr, local_byte) in &cf.label_bytes {
            let l_key = key(label_addr, cf.mode);
            let byte_off = (base_byte + local_byte as usize) as u32;
            let slot = entries_map.entry(l_key).or_insert((byte_off, func_idx as u32));
            if label_addr == cf.entry {
                *slot = (byte_off, func_idx as u32);
                entry_local_byte = Some(local_byte as usize);
            }
        }

        if let Some(entry_local) = entry_local_byte {
            let raw_prologue = base_byte;
            let raw_body = base_byte + entry_local;
            if let Some(&ovr_idx) = override_map.get(&f_key) {
                let (ovr_prologue, ovr_body, ovr_relocs) =
                    emit_override_wrapper(&mut final_buf, ovr_idx, raw_body as u32);
                prologue_bytes.insert(f_key, ovr_prologue);
                body_bytes.insert(f_key, ovr_body);
                entries_map.insert(f_key, (ovr_body as u32, NO_ORIGIN));
                func_relocs.push((0, ovr_relocs));
            } else {
                prologue_bytes.insert(f_key, raw_prologue);
                body_bytes.insert(f_key, raw_body);
            }
        }

        func_relocs.push((base_byte, cf.relocs));
        functions_spans.push((cf.entry, cf.spans));
    }

    for (base_byte, relocs) in func_relocs {
        for rel in relocs {
            let disp_pos = base_byte + rel.local_disp_pos as usize;
            let target_byte = match rel.kind {
                RelocKind::StubJmp(s) | RelocKind::StubCall(s) => stubs.byte_of(s),
                RelocKind::DirectCall(t_key) => prologue_bytes
                    .get(&t_key)
                    .copied()
                    .unwrap_or(stubs.indirect_call),
                RelocKind::DirectTailJump(t_key) => body_bytes
                    .get(&t_key)
                    .copied()
                    .unwrap_or(stubs.indirect_jump),
            };
            final_buf.patch_rel32(disp_pos, target_byte);
        }
    }

    progress(96, 100);

    let mut entries = Vec::with_capacity(entries_map.len());
    let mut entry_owners = Vec::with_capacity(entries_map.len());
    for (k, (byte_off, owner)) in entries_map {
        entries.push((k, byte_off));
        entry_owners.push(owner);
    }
    let origins = build_origins(&program.text, &functions_spans, &entry_owners);

    progress(100, 100);

    AotImage {
        program_id,
        text_hash,
        enter_offset: stubs.enter as u32,
        code: final_buf.into_bytes(),
        entries,
        origins,
    }
}

// ---------------------------------------------------------------------------
// Shared Runtime Stubs Emitter
// ---------------------------------------------------------------------------

#[inline(always)]
pub(crate) fn emit_stmxcsr_scratch_buf(buf: &mut X64Buf) {
    // stmxcsr dword ptr [rbx + SCRATCH_FP]
    buf.emit_u8(0x0F);
    buf.emit_u8(0xAE);
    buf.emit_u8(0x9B);
    buf.emit_u32((offsets::SCRATCH_FP as i32) as u32);
}

#[inline(always)]
pub(crate) fn emit_ldmxcsr_scratch_buf(buf: &mut X64Buf) {
    // ldmxcsr dword ptr [rbx + SCRATCH_FP]
    buf.emit_u8(0x0F);
    buf.emit_u8(0xAE);
    buf.emit_u8(0x93);
    buf.emit_u32((offsets::SCRATCH_FP as i32) as u32);
}

/// Syncs x86_64 MXCSR FTZ/DAZ/RC from guest `FPSCR` held in `eax` (clobbers `rcx`, `rdx`, `rsi`).
pub(crate) fn emit_sync_mxcsr_from_eax(buf: &mut X64Buf) {
    emit_stmxcsr_scratch_buf(buf);
    buf.ldr_r32_mem(RCX, RBX, offsets::SCRATCH_FP as i32);
    buf.alu_r32_imm32(AluOp::And, RCX, !0xE040u32 as i32);
    buf.mov_r32_imm32(RDX, 0x8040);
    buf.mov_r32_imm32(RSI, 0);
    buf.test_r32_imm32(RAX, 1 << 24);
    buf.cmovcc_r32_r32(cc::Z, RDX, RSI);
    buf.alu_r32_r32(AluOp::Or, RCX, RDX);
    // Map ARM RMode FPSCR[23:22] (00=RN, 01=+Inf, 10=-Inf, 11=RZ)
    // to x86 MXCSR RC[14:13]   (00=RN, 01=-Inf, 10=+Inf, 11=RZ) by swapping the 2 bits.
    buf.mov_r32_r32(RDX, RAX);
    buf.shift_r32_imm(ShiftOp::Shr, RDX, 22);
    buf.alu_r32_imm32(AluOp::And, RDX, 3);
    buf.mov_r32_r32(RSI, RDX);
    buf.alu_r32_imm32(AluOp::And, RDX, 1);
    buf.shift_r32_imm(ShiftOp::Shl, RDX, 1);
    buf.shift_r32_imm(ShiftOp::Shr, RSI, 1);
    buf.alu_r32_r32(AluOp::Or, RDX, RSI);
    buf.shift_r32_imm(ShiftOp::Shl, RDX, 13);
    buf.alu_r32_r32(AluOp::Or, RCX, RDX);
    buf.str_r32_mem(RCX, RBX, offsets::SCRATCH_FP as i32);
    emit_ldmxcsr_scratch_buf(buf);
}

/// Syncs x86_64 MXCSR FTZ/DAZ/RC from `*ctx.fpscr` (clobbers `rax`, `rcx`, `rdx`, `rsi`).
pub(crate) fn emit_sync_mxcsr_from_fpscr(buf: &mut X64Buf) {
    buf.ldr_r64_mem(RAX, RBX, offsets::CTX_FPSCR as i32);
    buf.ldr_r32_mem(RAX, RAX, 0);
    emit_sync_mxcsr_from_eax(buf);
}

#[inline(always)]
pub(crate) fn emit_save_host_mxcsr_buf(buf: &mut X64Buf) {
    emit_stmxcsr_scratch_buf(buf);
    buf.ldr_r32_mem(RAX, RBX, offsets::SCRATCH_FP as i32);
    buf.str_r32_mem(RAX, RBX, offsets::HOST_MXCSR as i32);
}

#[inline(always)]
pub(crate) fn emit_restore_host_mxcsr_buf(buf: &mut X64Buf) {
    // ldmxcsr dword ptr [rbx + HOST_MXCSR]
    buf.emit_u8(0x0F);
    buf.emit_u8(0xAE);
    buf.emit_u8(0x93);
    buf.emit_u32((offsets::HOST_MXCSR as i32) as u32);
}

fn emit_runtime_stubs(buf: &mut X64Buf) -> RuntimeStubs {
    // 1. sync_out: flushes r15d (NZCV) and r14d (budget) into AotExecContext.
    //    Uses ONLY RAX so RCX, RDX, RSI, RDI, R8, R9, R10, R11 are preserved.
    let sync_out = buf.pos();
    for (shift, off) in [
        (31u8, offsets::CTX_N),
        (30, offsets::CTX_Z),
        (29, offsets::CTX_C),
        (28, offsets::CTX_V),
    ] {
        buf.mov_r32_r32(RAX, R15);
        buf.shift_r32_imm(ShiftOp::Shr, RAX, shift);
        if shift < 31 {
            buf.alu_r32_imm32(AluOp::And, RAX, 1);
        }
        buf.str_r8_mem(RAX, RBX, off as i32);
    }
    buf.str_r32_mem(R14, RBX, offsets::CTX_BUDGET as i32);
    buf.ret();

    // 2. sync_in: reloads r15d (NZCV), r14d (budget), r12 (read_pages), r13 (write_pages).
    //    Uses ONLY RAX so caller-held registers remain intact.
    let sync_in = buf.pos();
    for (i, (shift, off)) in [
        (31u8, offsets::CTX_N),
        (30, offsets::CTX_Z),
        (29, offsets::CTX_C),
        (28, offsets::CTX_V),
    ]
    .into_iter()
    .enumerate()
    {
        buf.ldr_u8_mem(RAX, RBX, off as i32);
        buf.alu_r32_imm32(AluOp::And, RAX, 1);
        buf.shift_r32_imm(ShiftOp::Shl, RAX, shift);
        if i == 0 {
            buf.mov_r32_r32(R15, RAX);
        } else {
            buf.alu_r32_r32(AluOp::Or, R15, RAX);
        }
    }
    buf.ldr_r32_mem(R14, RBX, offsets::CTX_BUDGET as i32);
    buf.ldr_r64_mem(R12, RBX, offsets::CTX_READ_PAGES as i32);
    buf.ldr_r64_mem(R13, RBX, offsets::CTX_WRITE_PAGES as i32);
    buf.ret();

    // invoke_r11 helper: establishes matching [rbp] frame before jumping into target body.
    let invoke_r11 = buf.pos();
    buf.push_r64(RBP);
    buf.mov_r64_r64(RBP, RSP);
    buf.jmp_r64(R11);

    // 3. enter(rdi = ec, rsi = target_body_rx)
    let enter = buf.pos();
    buf.push_r64(RBP);
    buf.mov_r64_r64(RBP, RSP);
    buf.push_r64(RBX);
    buf.push_r64(R12);
    buf.push_r64(R13);
    buf.push_r64(R14);
    buf.push_r64(R15);
    buf.alu_r64_imm32(AluOp::Sub, RSP, 8); // 6 pushes (48B) + 8B -> RSP % 16 == 0
    buf.mov_r64_r64(RBX, RDI);
    buf.mov_r64_r64(R11, RSI);
    // Preserve caller host MXCSR in `ec.HOST_MXCSR` and sync MXCSR from `*ctx.fpscr`.
    emit_save_host_mxcsr_buf(buf);
    emit_sync_mxcsr_from_fpscr(buf);
    buf.call_to(sync_in);
    buf.call_to(invoke_r11);
    buf.call_to(sync_out);
    // Always restore caller host MXCSR on return (even after guest `VMSR FPSCR`).
    emit_restore_host_mxcsr_buf(buf);
    buf.alu_r64_imm32(AluOp::Add, RSP, 8);
    buf.pop_r64(R15);
    buf.pop_r64(R14);
    buf.pop_r64(R13);
    buf.pop_r64(R12);
    buf.pop_r64(RBX);
    buf.pop_r64(RBP);
    buf.ret();

    // 4. Slow memory read stubs (input: r10d = addr; output: eax = val; preserves R8, R9, RBX, R12..R15)
    let emit_slow_read = |buf: &mut X64Buf, fn_off: u16| -> usize {
        let start = buf.pos();
        buf.push_r64(RBP);
        buf.mov_r64_r64(RBP, RSP);
        buf.alu_r64_imm32(AluOp::Sub, RSP, 32);
        buf.str_r64_mem(R8, RSP, 0);
        buf.str_r64_mem(R9, RSP, 8);
        buf.call_to(sync_out);
        buf.mov_r64_r64(RDI, RBX);
        buf.mov_r32_r32(RSI, R10);
        emit_restore_host_mxcsr_buf(buf);
        buf.call_mem64(RBX, fn_off as i32);
        buf.str_r32_mem(RAX, RSP, 16);
        buf.call_to(sync_in);
        emit_sync_mxcsr_from_fpscr(buf);
        buf.ldr_r32_mem(RAX, RSP, 16);
        buf.ldr_r64_mem(R8, RSP, 0);
        buf.ldr_r64_mem(R9, RSP, 8);
        buf.alu_r64_imm32(AluOp::Add, RSP, 32);
        buf.pop_r64(RBP);
        buf.ret();
        start
    };
    let slow_read8 = emit_slow_read(buf, offsets::FN_SLOW_READ8);
    let slow_read16 = emit_slow_read(buf, offsets::FN_SLOW_READ16);
    let slow_read32 = emit_slow_read(buf, offsets::FN_SLOW_READ32);

    // 5. Slow write / interpret / VFP helper stubs (input: r10d = arg1, r11d = arg2; preserves R8, R9, RBX, R12..R15)
    let emit_slow_write = |buf: &mut X64Buf, fn_off: u16, use_host_mxcsr: bool| -> usize {
        let start = buf.pos();
        buf.push_r64(RBP);
        buf.mov_r64_r64(RBP, RSP);
        buf.alu_r64_imm32(AluOp::Sub, RSP, 16);
        buf.str_r64_mem(R8, RSP, 0);
        buf.str_r64_mem(R9, RSP, 8);
        buf.call_to(sync_out);
        buf.mov_r64_r64(RDI, RBX);
        buf.mov_r32_r32(RSI, R10);
        buf.mov_r32_r32(RDX, R11);
        if use_host_mxcsr {
            emit_restore_host_mxcsr_buf(buf);
        }
        buf.call_mem64(RBX, fn_off as i32);
        buf.call_to(sync_in);
        if use_host_mxcsr {
            emit_sync_mxcsr_from_fpscr(buf);
        }
        buf.ldr_r64_mem(R8, RSP, 0);
        buf.ldr_r64_mem(R9, RSP, 8);
        buf.alu_r64_imm32(AluOp::Add, RSP, 16);
        buf.pop_r64(RBP);
        buf.ret();
        start
    };
    let slow_write8 = emit_slow_write(buf, offsets::FN_SLOW_WRITE8, true);
    let slow_write16 = emit_slow_write(buf, offsets::FN_SLOW_WRITE16, true);
    let slow_write32 = emit_slow_write(buf, offsets::FN_SLOW_WRITE32, true);

    // 6. Indirect call & indirect jump (uses 8192-slot direct-mapped icache)
    let indirect_call = buf.pos();
    buf.push_r64(RBP);
    buf.mov_r64_r64(RBP, RSP);
    let indirect_jump = buf.pos();
    buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
    buf.ldr_u8_mem(RCX, RBX, offsets::CTX_THUMB as i32);
    buf.alu_r32_r32(AluOp::Or, RAX, RCX); // eax = key (zero-extended in rax)
    buf.mov_r32_imm32(RDX, 0x9E37_79B1);
    buf.imul_r32_r32(RDX, RAX);
    buf.shift_r32_imm(ShiftOp::Shr, RDX, 19);
    buf.shift_r32_imm(ShiftOp::Shl, RDX, 4); // 16 bytes/slot
    buf.ldr_r64_mem(RCX, RBX, offsets::ICACHE_PTR as i32);
    buf.alu_r64_r64(AluOp::Add, RCX, RDX);
    buf.ldr_r64_mem(RSI, RCX, 0);
    buf.ldr_r64_mem(R11, RCX, 8);
    buf.alu_r64_r64(AluOp::Cmp, RSI, RAX);
    let j_miss = buf.jcc_rel32(cc::NE);
    buf.test_r64_r64(R11, R11);
    let j_unw1 = buf.jcc_rel32(cc::Z);
    buf.jmp_r64(R11);

    let miss_pos = buf.pos();
    buf.patch_rel32(j_miss, miss_pos);
    buf.alu_r64_imm32(AluOp::Sub, RSP, 16);
    buf.str_r32_mem(RAX, RSP, 0);
    buf.call_to(sync_out);
    buf.mov_r64_r64(RDI, RBX);
    buf.ldr_r32_mem(RSI, RSP, 0);
    emit_restore_host_mxcsr_buf(buf);
    buf.call_mem64(RBX, offsets::FN_LOOKUP_BODY as i32);
    buf.str_r64_mem(RAX, RSP, 0);
    buf.call_to(sync_in);
    emit_sync_mxcsr_from_fpscr(buf);
    buf.ldr_r64_mem(R11, RSP, 0);
    buf.alu_r64_imm32(AluOp::Add, RSP, 16);
    buf.test_r64_r64(R11, R11);
    let j_unw2 = buf.jcc_rel32(cc::Z);
    buf.jmp_r64(R11);

    let unwind_pos = buf.pos();
    buf.patch_rel32(j_unw1, unwind_pos);
    buf.patch_rel32(j_unw2, unwind_pos);
    buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, EXIT_UNWIND);
    buf.pop_r64(RBP);
    buf.ret();

    // 7. Interpret stub (input: r10d = addr, r11d = op)
    let interpret = emit_slow_write(buf, offsets::FN_INTERPRET, true);

    // 8. Media parallel helper stub (input: eax = packed, r10d = a, r11d = b -> eax = res; preserves R8, R9)
    let media_parallel = buf.pos();
    buf.push_r64(RBP);
    buf.mov_r64_r64(RBP, RSP);
    buf.alu_r64_imm32(AluOp::Sub, RSP, 32);
    buf.str_r64_mem(R8, RSP, 0);
    buf.str_r64_mem(R9, RSP, 8);
    buf.str_r32_mem(RAX, RSP, 16);
    buf.call_to(sync_out);
    buf.mov_r64_r64(RDI, RBX);
    buf.ldr_r32_mem(RSI, RSP, 16);
    buf.mov_r32_r32(RDX, R10);
    buf.mov_r32_r32(RCX, R11);
    emit_restore_host_mxcsr_buf(buf);
    buf.call_mem64(RBX, offsets::FN_MEDIA_PARALLEL as i32);
    buf.str_r32_mem(RAX, RSP, 16);
    buf.call_to(sync_in);
    emit_sync_mxcsr_from_fpscr(buf);
    buf.ldr_r32_mem(RAX, RSP, 16);
    buf.ldr_r64_mem(R8, RSP, 0);
    buf.ldr_r64_mem(R9, RSP, 8);
    buf.alu_r64_imm32(AluOp::Add, RSP, 32);
    buf.pop_r64(RBP);
    buf.ret();

    // 9. VFP vector & float-to-int helper stubs (input: r10d = packed)
    let vfp_vector = emit_slow_write(buf, offsets::FN_VFP_VECTOR, false);
    let vfp_to_int = emit_slow_write(buf, offsets::FN_VFP_TO_INT, true);

    // 10. Pure-scratch register-shift-with-carry subroutines.
    //     Input: eax = val, ecx = raw Rs, r15d = NZCV<<28.
    //     Output: eax = shifted, edx = carry (0/1). Preserves R8, R9, R10, R11.
    let emit_shift_preamble = |buf: &mut X64Buf| -> usize {
        buf.mov_r32_r32(RDX, R15);
        buf.shift_r32_imm(ShiftOp::Shr, RDX, 29);
        buf.alu_r32_imm32(AluOp::And, RDX, 1);
        buf.alu_r32_imm32(AluOp::And, RCX, 0xFF);
        buf.jcc_rel32(cc::Z)
    };

    let shift_lsl_c = buf.pos();
    let j_lsl_z = emit_shift_preamble(buf);
    buf.alu_r32_imm32(AluOp::Cmp, RCX, 32);
    let j_lsl_gt = buf.jcc_rel32(cc::A);
    let j_lsl_eq = buf.jcc_rel32(cc::E);
    buf.shift_r32_cl(ShiftOp::Shl, RAX);
    buf.mov_r32_imm32(RDX, 0);
    buf.setcc_r8(cc::C, RDX);
    buf.ret();
    buf.patch_rel32(j_lsl_eq, buf.pos());
    buf.mov_r32_r32(RDX, RAX);
    buf.alu_r32_imm32(AluOp::And, RDX, 1);
    buf.mov_r32_imm32(RAX, 0);
    buf.ret();
    buf.patch_rel32(j_lsl_gt, buf.pos());
    buf.mov_r32_imm32(RAX, 0);
    buf.mov_r32_imm32(RDX, 0);
    buf.patch_rel32(j_lsl_z, buf.pos());
    buf.ret();

    let shift_lsr_c = buf.pos();
    let j_lsr_z = emit_shift_preamble(buf);
    buf.alu_r32_imm32(AluOp::Cmp, RCX, 32);
    let j_lsr_gt = buf.jcc_rel32(cc::A);
    let j_lsr_eq = buf.jcc_rel32(cc::E);
    buf.shift_r32_cl(ShiftOp::Shr, RAX);
    buf.mov_r32_imm32(RDX, 0);
    buf.setcc_r8(cc::C, RDX);
    buf.ret();
    buf.patch_rel32(j_lsr_eq, buf.pos());
    buf.mov_r32_r32(RDX, RAX);
    buf.shift_r32_imm(ShiftOp::Shr, RDX, 31);
    buf.mov_r32_imm32(RAX, 0);
    buf.ret();
    buf.patch_rel32(j_lsr_gt, buf.pos());
    buf.mov_r32_imm32(RAX, 0);
    buf.mov_r32_imm32(RDX, 0);
    buf.patch_rel32(j_lsr_z, buf.pos());
    buf.ret();

    let shift_asr_c = buf.pos();
    let j_asr_z = emit_shift_preamble(buf);
    buf.alu_r32_imm32(AluOp::Cmp, RCX, 32);
    let j_asr_ge = buf.jcc_rel32(cc::AE);
    buf.shift_r32_cl(ShiftOp::Sar, RAX);
    buf.mov_r32_imm32(RDX, 0);
    buf.setcc_r8(cc::C, RDX);
    buf.ret();
    buf.patch_rel32(j_asr_ge, buf.pos());
    buf.shift_r32_imm(ShiftOp::Sar, RAX, 31);
    buf.mov_r32_r32(RDX, RAX);
    buf.alu_r32_imm32(AluOp::And, RDX, 1);
    buf.patch_rel32(j_asr_z, buf.pos());
    buf.ret();

    let shift_ror_c = buf.pos();
    let j_ror_z = emit_shift_preamble(buf);
    buf.shift_r32_cl(ShiftOp::Ror, RAX);
    buf.mov_r32_r32(RDX, RAX);
    buf.shift_r32_imm(ShiftOp::Shr, RDX, 31);
    buf.patch_rel32(j_ror_z, buf.pos());
    buf.ret();

    RuntimeStubs {
        enter,
        sync_out,
        sync_in,
        slow_read8,
        slow_read16,
        slow_read32,
        slow_write8,
        slow_write16,
        slow_write32,
        indirect_call,
        indirect_jump,
        interpret,
        media_parallel,
        vfp_vector,
        vfp_to_int,
        shift_lsl_c,
        shift_lsr_c,
        shift_asr_c,
        shift_ror_c,
    }
}

fn emit_override_wrapper(
    buf: &mut X64Buf,
    ovr_idx: usize,
    raw_body_byte_offset: u32,
) -> (usize, usize, Vec<Reloc>) {
    let mut relocs = Vec::new();
    let prologue = buf.pos();
    buf.push_r64(RBP);
    buf.mov_r64_r64(RBP, RSP);
    let body = buf.pos();

    buf.ldr_r64_mem(RAX, RBX, (offsets::OVERRIDES_BASE + (ovr_idx as u16) * 8) as i32);
    buf.test_r64_r64(RAX, RAX);
    let j_null = buf.jcc_rel32(cc::Z);

    let s1 = buf.call_rel32();
    relocs.push(Reloc { local_disp_pos: s1 as u32, kind: RelocKind::StubCall(StubId::SyncOut) });

    buf.mov_r64_r64(RDI, RBX);
    buf.ldr_r64_mem(RSI, RBX, offsets::RX_BASE as i32);
    buf.mov_r32_imm32(RAX, raw_body_byte_offset);
    buf.alu_r64_r64(AluOp::Add, RSI, RAX);
    emit_restore_host_mxcsr_buf(buf);
    buf.call_mem64(RBX, (offsets::OVERRIDES_BASE + (ovr_idx as u16) * 8) as i32);

    let s2 = buf.call_rel32();
    relocs.push(Reloc { local_disp_pos: s2 as u32, kind: RelocKind::StubCall(StubId::SyncIn) });
    emit_sync_mxcsr_from_fpscr(buf);

    buf.pop_r64(RBP);
    buf.ret();

    // Null override slot fallback (e.g. modded `.code` without attached override):
    buf.patch_rel32(j_null, buf.pos());
    buf.ldr_r64_mem(RAX, RBX, offsets::RX_BASE as i32);
    buf.mov_r32_imm32(RCX, raw_body_byte_offset);
    buf.alu_r64_r64(AluOp::Add, RAX, RCX);
    buf.jmp_r64(RAX);

    (prologue, body, relocs)
}

// ---------------------------------------------------------------------------
// Per-Function Compiler
// ---------------------------------------------------------------------------

fn compile_one_function(
    program: &Program,
    known_targets: &BTreeSet<u32>,
    func_idx: u32,
    entry: u32,
    func: &Function,
    options: CompileOptions,
) -> CompiledFunc {
    let first_pass = emit_function_pass(program, known_targets, func_idx, entry, func, &func.labels, options);
    if first_pass.1.is_empty() {
        first_pass.0
    } else {
        let mut resumable = func.labels.clone();
        for addr in first_pass.1 {
            if let Ok(pos) = func.instructions.binary_search(&addr) {
                if let Some(&next) = func.instructions.get(pos + 1) {
                    resumable.insert(next);
                }
            }
        }
        emit_function_pass(program, known_targets, func_idx, entry, func, &resumable, options).0
    }
}

fn emit_function_pass(
    program: &Program,
    known_targets: &BTreeSet<u32>,
    func_idx: u32,
    entry: u32,
    func: &Function,
    labels: &BTreeSet<u32>,
    options: CompileOptions,
) -> (CompiledFunc, Vec<u32>) {
    let mut em = FnEmitter {
        text: &program.text,
        known_targets,
        labels: labels.clone(),
        mode: func.mode,
        options,
        buf: X64Buf::with_capacity(func.instructions.len() * 16 + 32),
        relocs: Vec::new(),
        local_patches: Vec::new(),
        epilogue_patches: Vec::new(),
        depth_overflow_patches: Vec::new(),
        budget_exits: Vec::new(),
        stale_check: None,
        interpreted_addrs: Vec::new(),
        folded_spans: Vec::new(),
    };

    let thumb = func.mode == Mode::Thumb;
    let insns = &func.instructions;
    if em.labels.contains(&entry) {
        em.buf.push_r64(RBP);
        em.buf.mov_r64_r64(RBP, RSP);
        if insns.first().copied() != Some(entry) {
            let p = em.buf.jmp_rel32();
            em.local_patches.push(LocalPatch { disp_pos: p, target_label: entry, cc: None });
        }
    }

    let mut label_map: BTreeMap<u32, u32> = BTreeMap::new();
    let mut spans: Vec<(u32, u32)> = Vec::new();

    for (i, &address) in insns.iter().enumerate() {
        if em.labels.contains(&address) {
            label_map.insert(address, em.buf.pos() as u32);
            if address == entry {
                em.buf.ldr_r64_mem(RAX, RBX, offsets::STALE_PTR as i32);
                em.buf.test_r64_r64(RAX, RAX);
                let j_stale = em.buf.jcc_rel32(cc::NZ);
                let resume_entry = em.buf.pos();
                em.stale_check = Some((j_stale, resume_entry, entry, func_idx));
            }
            let run = insns[i + 1..].iter().take_while(|a| !em.labels.contains(a)).count() as u32 + 1;
            em.emit_block_budget(address, run);
        }

        let (continues, size) = if thumb {
            let op = program.text.read16(address).unwrap_or(0) as u32;
            let second = program.text.read16(address + 2).unwrap_or(0) as u32;
            let sz = decode_thumb(op as u16, Some(second as u16), address).1;
            (em.lower_thumb(address, op, second), sz)
        } else {
            let op = program.text.read32(address).unwrap_or(0);
            (em.lower_arm(address, op), 4)
        };

        match spans.last_mut() {
            Some(last) if last.1 == address => last.1 = address + size,
            _ => spans.push((address, address + size)),
        }

        let next = address.wrapping_add(size);
        if continues && insns.get(i + 1) != Some(&next) {
            em.emit_jump(next, thumb);
        }
    }

    // Cold stale-function unwind stub (triggers when guest text patched a directly-called callee):
    if let Some((j_stale, resume_entry, entry_addr, f_idx)) = em.stale_check {
        em.buf.patch_rel32(j_stale, em.buf.pos());
        em.buf.ldr_u8_mem(RAX, RAX, f_idx as i32);
        em.buf.test_r32_r32(RAX, RAX);
        let j_ok = em.buf.jcc_rel32(cc::Z);
        em.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, entry_addr);
        em.buf.str_imm8_mem(RBX, offsets::CTX_THUMB as i32, thumb as u8);
        em.buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, EXIT_UNWIND);
        let j_epi = em.buf.jmp_rel32();
        em.epilogue_patches.push(j_epi);
        em.buf.patch_rel32(j_ok, resume_entry);
    }

    // Cold depth-overflow stub:
    if !em.depth_overflow_patches.is_empty() {
        let mut by_target: BTreeMap<Option<u32>, Vec<usize>> = BTreeMap::new();
        for (site, target) in std::mem::take(&mut em.depth_overflow_patches) {
            by_target.entry(target).or_default().push(site);
        }
        let mut tail_jumps = Vec::new();
        let none_sites = by_target.remove(&None);
        let count = by_target.len();
        for (idx, (target, sites)) in by_target.into_iter().enumerate() {
            let here = em.buf.pos();
            for site in sites {
                em.buf.patch_rel32(site, here);
            }
            if let Some(t) = target {
                em.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, t);
            }
            if idx + 1 < count {
                tail_jumps.push(em.buf.jmp_rel32());
            }
        }
        let depth_tail = em.buf.pos();
        if let Some(sites) = none_sites {
            for site in sites {
                em.buf.patch_rel32(site, depth_tail);
            }
        }
        for j in tail_jumps {
            em.buf.patch_rel32(j, depth_tail);
        }
        em.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        em.buf.alu_r32_imm32(AluOp::Sub, RAX, 1);
        em.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        em.buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, EXIT_UNWIND);
        let j_epi = em.buf.jmp_rel32();
        em.epilogue_patches.push(j_epi);
    }

    // Cold budget-exhausted stubs (falls straight into epilogue):
    if !em.budget_exits.is_empty() {
        let mut exit_jumps = Vec::with_capacity(em.budget_exits.len());
        let budget_exits = std::mem::take(&mut em.budget_exits);
        for be in budget_exits {
            em.buf.patch_rel32(be.disp_pos, em.buf.pos());
            em.buf.alu_r32_imm32(AluOp::Add, R14, be.count as i32);
            em.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, be.guest_addr);
            exit_jumps.push(em.buf.jmp_rel32());
        }
        let budget_out = em.buf.pos();
        for j in exit_jumps {
            em.buf.patch_rel32(j, budget_out);
        }
        em.buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, EXIT_BUDGET);
    }

    // Shared function epilogue:
    let epilogue = em.buf.pos();
    for p in &em.epilogue_patches {
        em.buf.patch_rel32(*p, epilogue);
    }
    em.buf.pop_r64(RBP);
    em.buf.ret();

    for lp in &em.local_patches {
        let target_byte = label_map[&lp.target_label] as usize;
        em.buf.patch_rel32(lp.disp_pos, target_byte);
    }

    spans.extend_from_slice(&em.folded_spans);
    let label_bytes: Vec<(u32, u32)> = label_map.into_iter().collect();
    let interpreted = std::mem::take(&mut em.interpreted_addrs);
    (
        CompiledFunc {
            entry,
            mode: func.mode,
            buf: em.buf,
            label_bytes,
            relocs: em.relocs,
            spans,
        },
        interpreted,
    )
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

    fn emit_block_budget(&mut self, guest_addr: u32, count: u32) {
        self.buf.alu_r32_imm32(AluOp::Sub, R14, count as i32);
        let disp_pos = self.buf.jcc_rel32(cc::L);
        self.budget_exits.push(BudgetExit { disp_pos, guest_addr, count });
    }

    /// Evaluates ARM condition `0..=13` from packed `r15d` (`NZCV` in `[31:28]`),
    /// returning the x86 condition code that is true when the ARM condition holds.
    /// Clobbers `rax` (for `8..=13`) and EFLAGS.
    fn emit_cond_test(&mut self, arm_cond: u32) -> u8 {
        match arm_cond {
            0 | 1 => {
                self.buf.bt_r32_imm8(R15, 30); // Z -> CF
                if arm_cond == 0 { cc::C } else { cc::NC }
            }
            2 | 3 => {
                self.buf.bt_r32_imm8(R15, 29); // C -> CF
                if arm_cond == 2 { cc::C } else { cc::NC }
            }
            4 | 5 => {
                self.buf.test_r32_r32(R15, R15); // N -> SF
                if arm_cond == 4 { cc::S } else { cc::NS }
            }
            6 | 7 => {
                self.buf.bt_r32_imm8(R15, 28); // V -> CF
                if arm_cond == 6 { cc::C } else { cc::NC }
            }
            8 | 9 => {
                // HI: C==1 && Z==0  <=>  (r15d & 0x6000_0000) == 0x2000_0000
                self.buf.mov_r32_r32(RAX, R15);
                self.buf.alu_r32_imm32(AluOp::And, RAX, 0x6000_0000);
                self.buf.alu_r32_imm32(AluOp::Cmp, RAX, 0x2000_0000);
                if arm_cond == 8 { cc::E } else { cc::NE }
            }
            10 | 11 => {
                // GE: N == V  <=>  ((r15d << 3) ^ r15d)[31] == 0
                self.buf.mov_r32_r32(RAX, R15);
                self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 3);
                self.buf.alu_r32_r32(AluOp::Xor, RAX, R15);
                if arm_cond == 10 { cc::NS } else { cc::S }
            }
            12 | 13 => {
                // GT: Z == 0 && N == V  <=>  (((r15d << 3) ^ r15d) & 0xC000_0000) == 0
                // (relies on r15d[27:0] == 0 so bit 30 of (r15d << 3) is 0)
                self.buf.mov_r32_r32(RAX, R15);
                self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 3);
                self.buf.alu_r32_r32(AluOp::Xor, RAX, R15);
                self.buf.alu_r32_imm32(AluOp::And, RAX, 0xC000_0000u32 as i32);
                if arm_cond == 12 { cc::E } else { cc::NE }
            }
            _ => unreachable!("invalid ARM condition {arm_cond}"),
        }
    }

    pub(crate) fn emit_cond_skip(&mut self, arm_cond: u32) -> Option<usize> {
        if arm_cond >= 14 {
            return None;
        }
        let cond_true = self.emit_cond_test(arm_cond);
        Some(self.buf.jcc_rel32(cc::invert(cond_true)))
    }

    /// Emits a conditional local jump (`arm_cond` 0..=13, or unconditional if >= 14) to `target_label`.
    pub(crate) fn emit_cond_local_branch(&mut self, arm_cond: u32, target_label: u32) {
        if arm_cond >= 14 {
            let p = self.buf.jmp_rel32();
            self.local_patches.push(LocalPatch { disp_pos: p, target_label, cc: None });
            return;
        }
        let cond_true = self.emit_cond_test(arm_cond);
        let p = self.buf.jcc_rel32(cond_true);
        self.local_patches.push(LocalPatch { disp_pos: p, target_label, cc: Some(cond_true) });
    }

    pub(crate) fn patch_cond_skip(&mut self, skip: Option<usize>) {
        if let Some(disp) = skip {
            self.buf.patch_rel32(disp, self.buf.pos());
        }
    }

    fn set_nzcv(&mut self, sub: bool) {
        // Read all four flags BEFORE any flag-modifying instruction!
        self.buf.setcc_r8(cc::S, RAX);
        self.buf.setcc_r8(cc::Z, RCX);
        self.buf.setcc_r8(if sub { cc::NC } else { cc::C }, RDX);
        self.buf.setcc_r8(cc::O, RSI);
        self.buf.movzx_r32_r8(RAX, RAX);
        self.buf.movzx_r32_r8(RCX, RCX);
        self.buf.movzx_r32_r8(RDX, RDX);
        self.buf.movzx_r32_r8(RSI, RSI);
        self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 3);
        self.buf.shift_r32_imm(ShiftOp::Shl, RCX, 2);
        self.buf.shift_r32_imm(ShiftOp::Shl, RDX, 1);
        self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
        self.buf.alu_r32_r32(AluOp::Or, RAX, RDX);
        self.buf.alu_r32_r32(AluOp::Or, RAX, RSI);
        self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 28);
        self.buf.mov_r32_r32(R15, RAX);
    }

    /// Captures live x86 `SF, ZF, CF, OF` after an `ADD`/`ADC` operation into `r15d[31:28]`
    /// (ARM `C = x86 CF`). Clobbers `rax, rcx, rdx, rsi`.
    pub(crate) fn set_nzcv_add(&mut self) {
        self.set_nzcv(false);
    }

    /// Captures live x86 `SF, ZF, CF, OF` after a `SUB`/`CMP`/`SBC`/`RSC`/`NEG` operation into
    /// `r15d[31:28]` (ARM `C = !x86 CF`). Clobbers `rax, rcx, rdx, rsi`.
    pub(crate) fn set_nzcv_sub(&mut self) {
        self.set_nzcv(true);
    }

    pub(crate) fn emit_epilogue_jump(&mut self) {
        let p = self.buf.jmp_rel32();
        self.epilogue_patches.push(p);
    }

    pub(crate) fn emit_check_unwind(&mut self) {
        self.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_EXIT as i32);
        self.buf.test_r32_r32(RAX, RAX);
        let p = self.buf.jcc_rel32(cc::NZ);
        self.epilogue_patches.push(p);
    }

    /// Checks if signed 64-bit value in `x_val` overflows `i32`, setting `ctx.q = 1` if so,
    /// and writes the low 32 bits into host `rd`. Never touches `R8`/`R9` unless `rd` is `R8`/`R9`.
    pub(crate) fn emit_accumulate_q(&mut self, rd: u8, x_val: u8) {
        let tmp = [R10, R11, RAX, RCX, RDX]
            .into_iter()
            .find(|&r| r != x_val && r != rd)
            .unwrap();
        self.buf.movsxd_r64_r32(tmp, x_val);
        self.buf.alu_r64_r64(AluOp::Cmp, tmp, x_val);
        let j_ok = self.buf.jcc_rel32(cc::E);
        self.buf.str_imm8_mem(RBX, offsets::CTX_Q as i32, 1);
        self.buf.patch_rel32(j_ok, self.buf.pos());
        if rd != x_val {
            self.buf.mov_r32_r32(rd, x_val);
        }
    }

    /// Saturates signed 64-bit value in `x_val` to `[-(1<<(bits-1)), (1<<(bits-1))-1]` (or
    /// unsigned `[0, (1<<bits)-1]`), setting `ctx.q = 1` on clamp, and writes host 32-bit `rd`.
    pub(crate) fn emit_saturate(&mut self, rd: u8, x_val: u8, bits: u32, unsigned: bool) {
        let (min_v, max_v): (i64, i64) = if unsigned {
            let max = if bits >= 32 { 0xFFFF_FFFF } else { (1i64 << bits) - 1 };
            (0, max)
        } else {
            (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
        };
        let tmp = [R10, R11, RAX, RCX, RDX, RSI, RDI]
            .into_iter()
            .find(|&r| r != x_val && r != rd)
            .unwrap();

        self.buf.mov_r64_imm64(tmp, max_v as u64);
        self.buf.alu_r64_r64(AluOp::Cmp, x_val, tmp);
        let j_le = self.buf.jcc_rel32(cc::LE);
        self.buf.str_imm8_mem(RBX, offsets::CTX_Q as i32, 1);
        self.buf.mov_r32_imm32(rd, max_v as u32);
        let j_done1 = self.buf.jmp_rel32();

        self.buf.patch_rel32(j_le, self.buf.pos());
        self.buf.mov_r64_imm64(tmp, min_v as u64);
        self.buf.alu_r64_r64(AluOp::Cmp, x_val, tmp);
        let j_ge = self.buf.jcc_rel32(cc::GE);
        self.buf.str_imm8_mem(RBX, offsets::CTX_Q as i32, 1);
        self.buf.mov_r32_imm32(rd, min_v as u32);
        let j_done2 = self.buf.jmp_rel32();

        self.buf.patch_rel32(j_ge, self.buf.pos());
        if rd != x_val {
            self.buf.mov_r32_r32(rd, x_val);
        }
        let done = self.buf.pos();
        self.buf.patch_rel32(j_done1, done);
        self.buf.patch_rel32(j_done2, done);
    }

    fn emit_read_impl(&mut self, addr_r32: u8, dst_r32: u8, size_bytes: u32, signed: bool) {
        if addr_r32 != R10 {
            self.buf.mov_r32_r32(R10, addr_r32);
        }
        let j_cross = if size_bytes > 1 {
            self.buf.mov_r32_r32(RCX, R10);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFF);
            self.buf.alu_r32_imm32(AluOp::Cmp, RCX, (0x1000 - size_bytes) as i32);
            Some(self.buf.jcc_rel32(cc::A))
        } else {
            None
        };
        self.buf.mov_r32_r32(RAX, R10);
        self.buf.shift_r32_imm(ShiftOp::Shr, RAX, 12);
        self.buf.ldr_r64_sib(RDX, R12, RAX, 8, 0);
        self.buf.test_r64_r64(RDX, RDX);
        let j_null = self.buf.jcc_rel32(cc::Z);
        if size_bytes == 1 {
            self.buf.mov_r32_r32(RCX, R10);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFF);
        }
        match (size_bytes, signed) {
            (1, true) => self.buf.ldr_s8_sib(dst_r32, RDX, RCX, 1, 0),
            (1, false) => self.buf.ldr_u8_sib(dst_r32, RDX, RCX, 1, 0),
            (2, true) => self.buf.ldr_s16_sib(dst_r32, RDX, RCX, 1, 0),
            (2, false) => self.buf.ldr_u16_sib(dst_r32, RDX, RCX, 1, 0),
            _ => self.buf.ldr_r32_sib(dst_r32, RDX, RCX, 1, 0),
        }
        let j_done = self.buf.jmp_rel32();

        let slow_pos = self.buf.pos();
        if let Some(j) = j_cross {
            self.buf.patch_rel32(j, slow_pos);
        }
        self.buf.patch_rel32(j_null, slow_pos);
        self.stub_call(match size_bytes {
            1 => StubId::SlowRead8,
            2 => StubId::SlowRead16,
            _ => StubId::SlowRead32,
        });
        match (size_bytes, signed) {
            (1, true) => self.buf.movsx_r32_r8(dst_r32, RAX),
            (2, true) => self.buf.movsx_r32_r16(dst_r32, RAX),
            _ if dst_r32 != RAX => self.buf.mov_r32_r32(dst_r32, RAX),
            _ => {}
        }
        self.buf.patch_rel32(j_done, self.buf.pos());
    }

    fn emit_write_impl(&mut self, addr_r32: u8, val_r32: u8, size_bytes: u32) {
        // Stage addr -> r10d, val -> r11d safely regardless of register aliasing.
        if addr_r32 == R11 && val_r32 == R10 {
            self.buf.mov_r32_r32(RAX, addr_r32);
            self.buf.mov_r32_r32(R11, val_r32);
            self.buf.mov_r32_r32(R10, RAX);
        } else if addr_r32 == R11 {
            self.buf.mov_r32_r32(R10, addr_r32);
            if val_r32 != R11 {
                self.buf.mov_r32_r32(R11, val_r32);
            }
        } else {
            if val_r32 != R11 {
                self.buf.mov_r32_r32(R11, val_r32);
            }
            if addr_r32 != R10 {
                self.buf.mov_r32_r32(R10, addr_r32);
            }
        }

        let j_cross = if size_bytes > 1 {
            self.buf.mov_r32_r32(RCX, R10);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFF);
            self.buf.alu_r32_imm32(AluOp::Cmp, RCX, (0x1000 - size_bytes) as i32);
            Some(self.buf.jcc_rel32(cc::A))
        } else {
            None
        };
        self.buf.mov_r32_r32(RAX, R10);
        self.buf.shift_r32_imm(ShiftOp::Shr, RAX, 12);
        self.buf.ldr_r64_sib(RDX, R13, RAX, 8, 0);
        self.buf.test_r64_r64(RDX, RDX);
        let j_null = self.buf.jcc_rel32(cc::Z);
        if size_bytes == 1 {
            self.buf.mov_r32_r32(RCX, R10);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xFFF);
        }
        match size_bytes {
            1 => self.buf.str_r8_sib(R11, RDX, RCX, 1, 0),
            2 => self.buf.str_r16_sib(R11, RDX, RCX, 1, 0),
            _ => self.buf.str_r32_sib(R11, RDX, RCX, 1, 0),
        }
        let j_done = self.buf.jmp_rel32();

        let slow_pos = self.buf.pos();
        if let Some(j) = j_cross {
            self.buf.patch_rel32(j, slow_pos);
        }
        self.buf.patch_rel32(j_null, slow_pos);
        self.stub_call(match size_bytes {
            1 => StubId::SlowWrite8,
            2 => StubId::SlowWrite16,
            _ => StubId::SlowWrite32,
        });
        self.buf.patch_rel32(j_done, self.buf.pos());
    }

    /// Memory fast/slow helpers.
    /// Contract:
    /// - Reads address from `addr_r32` (zero-extended; `addr_r32` may be any scratch reg, moved to `R10` if != `R10`).
    /// - Preserves `RBX`, `R12..R15`, **AND `R8` and `R9`** across BOTH fast and slow paths!
    /// - Clobbers `RAX, RCX, RDX, RSI, RDI, R10, R11` (and EFLAGS).
    /// - Output lands in `dst_r32` (passing any of `RAX/RCX/RDX/RSI/RDI/R8/R9/R10/R11` works).
    #[allow(dead_code)]
    pub(crate) fn emit_read8(&mut self, addr_r32: u8, dst_r32: u8, signed: bool) {
        self.emit_read_impl(addr_r32, dst_r32, 1, signed);
    }

    #[allow(dead_code)]
    pub(crate) fn emit_read16(&mut self, addr_r32: u8, dst_r32: u8, signed: bool) {
        self.emit_read_impl(addr_r32, dst_r32, 2, signed);
    }

    #[allow(dead_code)]
    pub(crate) fn emit_read32(&mut self, addr_r32: u8, dst_r32: u8) {
        self.emit_read_impl(addr_r32, dst_r32, 4, false);
    }

    #[allow(dead_code)]
    pub(crate) fn emit_write8(&mut self, addr_r32: u8, val_r32: u8) {
        self.emit_write_impl(addr_r32, val_r32, 1);
    }

    #[allow(dead_code)]
    pub(crate) fn emit_write16(&mut self, addr_r32: u8, val_r32: u8) {
        self.emit_write_impl(addr_r32, val_r32, 2);
    }

    #[allow(dead_code)]
    pub(crate) fn emit_write32(&mut self, addr_r32: u8, val_r32: u8) {
        self.emit_write_impl(addr_r32, val_r32, 4);
    }

    fn emit_depth_check(&mut self, unwind_pc: Option<u32>) {
        self.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        self.buf.alu_r32_imm32(AluOp::Add, RAX, 1);
        self.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        self.buf.alu_r32_imm32(AluOp::Cmp, RAX, RECOMP_DEPTH_LIMIT as i32);
        let p_depth = self.buf.jcc_rel32(cc::A);
        self.depth_overflow_patches.push((p_depth, unwind_pc));
    }

    fn emit_post_call_guard(&mut self, expected_pc: u32, caller_thumb: bool) {
        self.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        self.buf.alu_r32_imm32(AluOp::Sub, RAX, 1);
        self.buf.str_r32_mem(RAX, RBX, offsets::CTX_DEPTH as i32);
        self.emit_check_unwind();

        self.buf.ldr_r32_mem(RAX, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
        self.buf.alu_r32_imm32(AluOp::Xor, RAX, expected_pc as i32);
        self.buf.ldr_u8_mem(RCX, RBX, offsets::CTX_THUMB as i32);
        if caller_thumb {
            self.buf.alu_r32_imm32(AluOp::Xor, RCX, 1);
        }
        self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
        let j_ok = self.buf.jcc_rel32(cc::Z);
        self.stub_jump(StubId::IndirectJump);
        self.buf.patch_rel32(j_ok, self.buf.pos());
    }

    pub(crate) fn emit_call(&mut self, link: u32, target: u32, callee_thumb: bool) -> bool {
        let caller_thumb = (link & 1) != 0;
        let t_key = target | (callee_thumb as u32);
        let is_direct = self.known_targets.contains(&t_key);

        self.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 14 * 4) as i32, link);
        if !is_direct {
            self.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, target);
        }
        if callee_thumb != caller_thumb {
            self.buf.str_imm8_mem(RBX, offsets::CTX_THUMB as i32, callee_thumb as u8);
        }
        self.emit_depth_check(is_direct.then_some(target));

        if is_direct {
            let disp = self.buf.call_rel32();
            self.relocs.push(Reloc { local_disp_pos: disp as u32, kind: RelocKind::DirectCall(t_key) });
        } else {
            self.stub_call(StubId::IndirectCall);
        }

        self.emit_post_call_guard(link & !1, caller_thumb);
        true
    }

    pub(crate) fn emit_jump(&mut self, target: u32, target_thumb: bool) -> bool {
        let same_mode = (target_thumb && self.mode == Mode::Thumb) || (!target_thumb && self.mode == Mode::Arm);
        if same_mode && self.labels.contains(&target) {
            let p = self.buf.jmp_rel32();
            self.local_patches.push(LocalPatch { disp_pos: p, target_label: target, cc: None });
            return false;
        }
        if !same_mode {
            self.buf.str_imm8_mem(RBX, offsets::CTX_THUMB as i32, target_thumb as u8);
        }
        let t_key = target | (target_thumb as u32);
        if self.known_targets.contains(&t_key) {
            let disp = self.buf.jmp_rel32();
            self.relocs.push(Reloc { local_disp_pos: disp as u32, kind: RelocKind::DirectTailJump(t_key) });
        } else {
            self.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, target);
            self.stub_jump(StubId::IndirectJump);
        }
        false
    }

    /// `RETURN_TO(val_r32)`: updates `ctx.thumb = val & 1`, `ctx.r[15] = val & (thumb ? !1 : !3)`, returns.
    pub(crate) fn emit_return_to_reg(&mut self, val_r32: u8) -> bool {
        let tmp = if val_r32 == RAX { RCX } else { RAX };
        let from_thumb = self.mode == Mode::Thumb;
        self.buf.mov_r32_r32(tmp, val_r32);
        self.buf.test_r32_imm32(val_r32, 1);
        if !from_thumb {
            let j_thumb = self.buf.jcc_rel32(cc::NZ);
            self.buf.alu_r32_imm32(AluOp::And, tmp, !3u32 as i32);
            self.buf.str_r32_mem(tmp, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.emit_epilogue_jump();

            self.buf.patch_rel32(j_thumb, self.buf.pos());
            self.buf.str_imm8_mem(RBX, offsets::CTX_THUMB as i32, 1);
            self.buf.alu_r32_imm32(AluOp::And, tmp, !1u32 as i32);
            self.buf.str_r32_mem(tmp, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.emit_epilogue_jump();
        } else {
            let j_arm = self.buf.jcc_rel32(cc::Z);
            self.buf.alu_r32_imm32(AluOp::And, tmp, !1u32 as i32);
            self.buf.str_r32_mem(tmp, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.emit_epilogue_jump();

            self.buf.patch_rel32(j_arm, self.buf.pos());
            self.buf.str_imm8_mem(RBX, offsets::CTX_THUMB as i32, 0);
            self.buf.alu_r32_imm32(AluOp::And, tmp, !3u32 as i32);
            self.buf.str_r32_mem(tmp, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.emit_epilogue_jump();
        }
        false
    }

    fn write_interworking_pc(&mut self, val_r32: u8) {
        let tmp1 = if val_r32 == RAX { RDX } else { RAX };
        let tmp2 = if val_r32 == RCX { RSI } else { RCX };
        self.buf.mov_r32_r32(tmp1, val_r32);
        self.buf.mov_r32_r32(tmp2, val_r32);
        self.buf.alu_r32_imm32(AluOp::And, tmp1, 1);
        self.buf.str_r8_mem(tmp1, RBX, offsets::CTX_THUMB as i32);
        self.buf.shift_r32_imm(ShiftOp::Shl, tmp1, 1);
        self.buf.alu_r32_imm32(AluOp::Or, tmp1, !3u32 as i32);
        self.buf.alu_r32_r32(AluOp::And, tmp2, tmp1);
        self.buf.str_r32_mem(tmp2, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
    }

    /// `JUMP_TO(val_r32)`: interworking jump through a register (`BX Rm`, `LDR PC, ...`).
    pub(crate) fn emit_jump_to_reg(&mut self, val_r32: u8) -> bool {
        self.write_interworking_pc(val_r32);
        self.stub_jump(StubId::IndirectJump);
        false
    }

    #[allow(dead_code)]
    pub(crate) fn emit_svc(&mut self, next_pc: u32, imm: u32) -> bool {
        self.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 15 * 4) as i32, next_pc);
        self.buf.str_imm32_mem(RBX, offsets::CTX_SVC as i32, imm);
        self.buf.str_imm32_mem(RBX, offsets::CTX_EXIT as i32, EXIT_SVC);
        self.emit_epilogue_jump();
        false
    }

    pub(crate) fn emit_interpret(&mut self, addr: u32, op: u32) -> bool {
        self.interpreted_addrs.push(addr);
        self.buf.mov_r32_imm32(R10, addr);
        self.buf.mov_r32_imm32(R11, op);
        self.stub_call(StubId::Interpret);
        self.emit_check_unwind();
        true
    }

    /// Updates N and Z in `r15d` from live `SF, ZF` (after `test res, res` or `alu`) and sets C
    /// from `carry_r32` bit 0 if `Some(reg)` (or preserves old C if `None`), preserving V in `r15d`.
    pub(crate) fn update_logical_flags(&mut self, carry_r32: Option<u8>) {
        // Choose N/Z capture registers that do not alias `carry_r32`.
        let tmp_n = if carry_r32 == Some(RAX) { RSI } else { RAX };
        let tmp_z = if carry_r32 == Some(RCX) { RDI } else { RCX };

        self.buf.setcc_r8(cc::S, tmp_n);
        self.buf.setcc_r8(cc::Z, tmp_z);
        self.buf.movzx_r32_r8(tmp_n, tmp_n);
        self.buf.movzx_r32_r8(tmp_z, tmp_z);
        self.buf.shift_r32_imm(ShiftOp::Shl, tmp_n, 31);
        self.buf.shift_r32_imm(ShiftOp::Shl, tmp_z, 30);

        match carry_r32 {
            None => {
                self.buf.alu_r32_imm32(AluOp::And, R15, 0x3000_0000);
                self.buf.alu_r32_r32(AluOp::Or, R15, tmp_n);
                self.buf.alu_r32_r32(AluOp::Or, R15, tmp_z);
            }
            Some(c_reg) => {
                self.buf.alu_r32_imm32(AluOp::And, R15, 0x1000_0000);
                self.buf.alu_r32_r32(AluOp::Or, R15, tmp_n);
                self.buf.alu_r32_r32(AluOp::Or, R15, tmp_z);
                self.buf.mov_r32_r32(tmp_n, c_reg);
                self.buf.alu_r32_imm32(AluOp::And, tmp_n, 1);
                self.buf.shift_r32_imm(ShiftOp::Shl, tmp_n, 29);
                self.buf.alu_r32_r32(AluOp::Or, R15, tmp_n);
            }
        }
    }

    // -----------------------------------------------------------------------
    // ARMv6K Instruction Lowering (Core & ALU)
    // -----------------------------------------------------------------------

    fn lower_arm(&mut self, a: u32, op: u32) -> bool {
        let condition = op >> 28;
        if condition == 0xF {
            return self.arm_unconditional(a, op);
        }
        if condition == 0xE {
            return self.arm_body(a, op);
        }
        // Fast-path conditional branch `B<cond> label` inside the same function:
        if (op >> 25) & 7 == 0b101 && (op & (1 << 24)) == 0 {
            let off = (((op & 0x00FF_FFFF) << 8) as i32 >> 8) as u32;
            let target = a.wrapping_add(8).wrapping_add(off << 2);
            if self.mode == Mode::Arm && self.labels.contains(&target) {
                self.emit_cond_local_branch(condition, target);
                return true;
            }
        }
        let skip = self.emit_cond_skip(condition);
        let _ = self.arm_body(a, op);
        self.patch_cond_skip(skip);
        true
    }

    fn arm_unconditional(&mut self, a: u32, op: u32) -> bool {
        if op & 0xFE00_0000 == 0xFA00_0000 {
            let s24 = (((op & 0x00FF_FFFF) << 8) as i32 >> 8) as u32;
            let offset = (s24 << 2) | (((op >> 24) & 1) << 1);
            return self.emit_call(a + 4, a.wrapping_add(8).wrapping_add(offset) & !1, true);
        }
        if op & 0xFFF0_00F0 == 0xF570_0010 {
            self.buf.str_imm8_mem(RBX, offsets::CTX_EXCLUSIVE as i32, 0);
            return true;
        }
        if matches!((op >> 25) & 7, 0b010 | 0b011) {
            return true; // PLD / barriers
        }
        self.emit_interpret(a, op)
    }

    fn arm_body(&mut self, a: u32, op: u32) -> bool {
        match (op >> 25) & 7 {
            0b000 => self.arm_register_space(a, op),
            0b001 if (op >> 23) & 3 == 0b10 && (op & (1 << 20)) == 0 => {
                if (op & (1 << 21)) != 0 && ((op >> 16) & 0xF) == 0 {
                    true // NOP / yield / hints
                } else {
                    self.arm_status_register(a, op)
                }
            }
            0b001 => self.arm_data_processing(a, op),
            0b010 => self.arm_single_transfer(a, op),
            0b011 if (op & 0x10) != 0 => self.arm_media(a, op),
            0b011 => self.arm_single_transfer(a, op),
            0b100 => self.arm_block_transfer(a, op),
            0b101 => {
                let s24 = (((op & 0x00FF_FFFF) << 8) as i32 >> 8) as u32;
                let target = a.wrapping_add(8).wrapping_add(s24 << 2);
                if (op & (1 << 24)) != 0 {
                    self.emit_call(a + 4, target, false)
                } else {
                    self.emit_jump(target, false)
                }
            }
            0b110 => self.vfp_load_store(a, op),
            0b111 if (op & (1 << 24)) != 0 => self.emit_svc(a + 4, op & 0x00FF_FFFF),
            0b111 if (op & 0x10) != 0 && matches!((op >> 8) & 0xF, 10 | 11) => {
                self.vfp_register_transfer(a, op)
            }
            0b111 if (op & 0x10) != 0 => {
                let rd = (op >> 12) & 0xF;
                let data_cache = (op & 0x0FFF_0F10) == 0x0E07_0F10 && !matches!(op & 0xF, 5 | 7 | 13);
                if (op & 0x0FFF_0FFF) == 0x0E1D_0F70 && rd != 15 {
                    self.buf.ldr_r32_mem(RAX, RBX, offsets::CTX_TLS as i32);
                    self.store_gpr(rd, RAX);
                    true
                } else if data_cache {
                    true
                } else {
                    self.emit_interpret(a, op)
                }
            }
            _ => self.vfp_data_processing(a, op),
        }
    }

    fn arm_register_space(&mut self, a: u32, op: u32) -> bool {
        if (op & 0x90) == 0x90 {
            if ((op >> 5) & 3) != 0 {
                return self.arm_extra_transfer(a, op);
            }
            if (op & (1 << 24)) != 0 {
                return if (op & (1 << 23)) != 0 {
                    self.arm_exclusive(a, op)
                } else {
                    self.arm_swap(a, op)
                };
            }
            return match (op >> 21) & 7 {
                0b000 | 0b001 => self.arm_multiply(a, op),
                0b010 | 0b100..=0b111 => self.arm_multiply_long(a, op),
                _ => self.emit_interpret(a, op),
            };
        }
        if ((op >> 23) & 3) == 0b10 && (op & (1 << 20)) == 0 {
            return match (op >> 4) & 0xF {
                0b0000 => self.arm_status_register(a, op),
                0b0001 if ((op >> 21) & 3) == 0b11 => {
                    let rd = (op >> 12) & 0xF;
                    if rd == 15 {
                        return self.emit_interpret(a, op);
                    }
                    self.load_gpr(RAX, op & 0xF, a + 8);
                    self.buf.test_r32_r32(RAX, RAX);
                    let j_zero = self.buf.jcc_rel32(cc::Z);
                    self.buf.bsr_r32(RCX, RAX);
                    self.buf.alu_r32_imm32(AluOp::Xor, RCX, 31);
                    let j_done = self.buf.jmp_rel32();
                    self.buf.patch_rel32(j_zero, self.buf.pos());
                    self.buf.mov_r32_imm32(RCX, 32);
                    self.buf.patch_rel32(j_done, self.buf.pos());
                    self.store_gpr(rd, RCX);
                    true
                }
                0b0001 | 0b0011 => self.arm_branch_exchange(a, op),
                0b0101 => self.arm_saturating(a, op),
                kind if (kind & 0b1001) == 0b1000 => self.arm_halfword_multiply(a, op),
                _ => self.emit_interpret(a, op),
            };
        }
        self.arm_data_processing(a, op)
    }

    fn arm_status_register(&mut self, a: u32, op: u32) -> bool {
        if (op & (1 << 22)) != 0 {
            return self.emit_interpret(a, op);
        }
        if (op & (1 << 21)) == 0 {
            // MRS Rd, CPSR
            let rd = (op >> 12) & 0xF;
            if rd == 15 {
                return self.emit_interpret(a, op);
            }
            self.buf.mov_r32_r32(RAX, R15);
            self.buf.alu_r32_imm32(AluOp::Or, RAX, 0x1F);
            self.buf.ldr_u8_mem(RCX, RBX, offsets::CTX_Q as i32);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 1);
            self.buf.shift_r32_imm(ShiftOp::Shl, RCX, 27);
            self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
            self.buf.ldr_u8_mem(RCX, RBX, offsets::CTX_GE as i32);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xF);
            self.buf.shift_r32_imm(ShiftOp::Shl, RCX, 16);
            self.buf.alu_r32_r32(AluOp::Or, RAX, RCX);
            self.store_gpr(rd, RAX);
            return true;
        }
        let mask = (op >> 16) & 0xF;
        if (mask & 0b0011) != 0 {
            return self.emit_interpret(a, op);
        }
        if (op & (1 << 25)) != 0 {
            let imm = (op & 0xFF).rotate_right(((op >> 8) & 0xF) * 2);
            self.buf.mov_r32_imm32(RAX, imm);
        } else {
            self.load_gpr(RAX, op & 0xF, a + 8);
        }
        if (mask & 0b1000) != 0 {
            self.buf.mov_r32_r32(R15, RAX);
            self.buf.alu_r32_imm32(AluOp::And, R15, 0xF000_0000u32 as i32);
            self.buf.mov_r32_r32(RCX, RAX);
            self.buf.shift_r32_imm(ShiftOp::Shr, RCX, 27);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 1);
            self.buf.str_r8_mem(RCX, RBX, offsets::CTX_Q as i32);
        }
        if (mask & 0b0100) != 0 {
            self.buf.mov_r32_r32(RCX, RAX);
            self.buf.shift_r32_imm(ShiftOp::Shr, RCX, 16);
            self.buf.alu_r32_imm32(AluOp::And, RCX, 0xF);
            self.buf.str_r8_mem(RCX, RBX, offsets::CTX_GE as i32);
        }
        true
    }

    /// Evaluates ARM shifter operand into `(R8, Option<R9>)`.
    /// When `need_carry` is true, `None` means shifter carry == old `C`; `Some(R9)` holds `0` or `1`.
    fn arm_shifter(&mut self, a: u32, op: u32, need_carry: bool) -> (u8, Option<u8>) {
        if (op & (1 << 25)) != 0 {
            let rotate = ((op >> 8) & 0xF) * 2;
            let value = (op & 0xFF).rotate_right(rotate);
            self.buf.mov_r32_imm32(R8, value);
            let sc = if need_carry && rotate != 0 {
                self.buf.mov_r32_imm32(R9, value >> 31);
                Some(R9)
            } else {
                None
            };
            return (R8, sc);
        }
        let rm_idx = op & 0xF;
        let kind = (op >> 5) & 3;
        if (op & (1 << 4)) != 0 {
            // Shift by register `Rs` (`r15` reads as `a + 12`).
            let rs_idx = (op >> 8) & 0xF;
            self.load_gpr(RAX, rm_idx, a + 12);
            self.load_gpr(RCX, rs_idx, a + 12);
            let stub = match kind {
                0 => StubId::ShiftLslC,
                1 => StubId::ShiftLsrC,
                2 => StubId::ShiftAsrC,
                _ => StubId::ShiftRorC,
            };
            self.stub_call(stub);
            self.buf.mov_r32_r32(R8, RAX);
            if need_carry {
                self.buf.mov_r32_r32(R9, RDX);
                return (R8, Some(R9));
            }
            return (R8, None);
        }

        let amount = ((op >> 7) & 0x1F) as u8;
        self.load_gpr(R8, rm_idx, a + 8);
        let extract_bit = |buf: &mut X64Buf, bit: u8| {
            buf.mov_r32_r32(R9, R8);
            if bit > 0 {
                buf.shift_r32_imm(ShiftOp::Shr, R9, bit);
            }
            if bit < 31 {
                buf.alu_r32_imm32(AluOp::And, R9, 1);
            }
        };
        match (kind, amount) {
            (0, 0) => (R8, None),
            (0, n) => {
                if need_carry {
                    extract_bit(&mut self.buf, 32 - n);
                }
                self.buf.shift_r32_imm(ShiftOp::Shl, R8, n);
                (R8, need_carry.then_some(R9))
            }
            (1, 0) => {
                if need_carry {
                    extract_bit(&mut self.buf, 31);
                }
                self.buf.mov_r32_imm32(R8, 0);
                (R8, need_carry.then_some(R9))
            }
            (1, n) => {
                if need_carry {
                    extract_bit(&mut self.buf, n - 1);
                }
                self.buf.shift_r32_imm(ShiftOp::Shr, R8, n);
                (R8, need_carry.then_some(R9))
            }
            (2, 0) => {
                if need_carry {
                    extract_bit(&mut self.buf, 31);
                }
                self.buf.shift_r32_imm(ShiftOp::Sar, R8, 31);
                (R8, need_carry.then_some(R9))
            }
            (2, n) => {
                if need_carry {
                    extract_bit(&mut self.buf, n - 1);
                }
                self.buf.shift_r32_imm(ShiftOp::Sar, R8, n);
                (R8, need_carry.then_some(R9))
            }
            (_, 0) => {
                // RRX: `(old_c << 31) | (m >> 1)`, carry_out = `m & 1`
                if need_carry {
                    extract_bit(&mut self.buf, 0);
                }
                self.buf.shift_r32_imm(ShiftOp::Shr, R8, 1);
                self.buf.mov_r32_r32(RAX, R15);
                self.buf.alu_r32_imm32(AluOp::And, RAX, 0x2000_0000);
                self.buf.shift_r32_imm(ShiftOp::Shl, RAX, 2);
                self.buf.alu_r32_r32(AluOp::Or, R8, RAX);
                (R8, need_carry.then_some(R9))
            }
            (_, n) => {
                if need_carry {
                    extract_bit(&mut self.buf, n - 1);
                }
                self.buf.shift_r32_imm(ShiftOp::Ror, R8, n);
                (R8, need_carry.then_some(R9))
            }
        }
    }

    fn arm_data_processing(&mut self, a: u32, op: u32) -> bool {
        let opcode = (op >> 21) & 0xF;
        let set_flags = (op & (1 << 20)) != 0;
        let rn = (op >> 16) & 0xF;
        let rd = (op >> 12) & 0xF;
        let compare = (8..=11).contains(&opcode);
        if set_flags && rd == 15 && !compare {
            return self.emit_interpret(a, op);
        }
        let register_shift = (op & (1 << 25)) == 0 && (op & (1 << 4)) != 0;
        let logical = matches!(opcode, 0 | 1 | 8 | 9 | 12..=15);

        let (b_reg, sc_reg) = self.arm_shifter(a, op, set_flags && logical);
        if !matches!(opcode, 13 | 15) {
            self.load_gpr(R10, rn, if register_shift { a + 12 } else { a + 8 });
        }

        // Keep result in R10, which is preserved by set_nzcv_* and update_logical_flags.
        match opcode {
            0 => {
                self.buf.alu_r32_r32(AluOp::And, R10, b_reg);
                if set_flags {
                    self.update_logical_flags(sc_reg);
                }
            }
            1 | 9 => {
                self.buf.alu_r32_r32(AluOp::Xor, R10, b_reg);
                if set_flags {
                    self.update_logical_flags(sc_reg);
                }
            }
            2 | 10 => {
                if compare {
                    self.buf.alu_r32_r32(AluOp::Cmp, R10, b_reg);
                } else {
                    self.buf.alu_r32_r32(AluOp::Sub, R10, b_reg);
                }
                if set_flags {
                    self.set_nzcv_sub();
                }
            }
            3 => {
                self.buf.mov_r32_r32(R11, b_reg);
                self.buf.alu_r32_r32(AluOp::Sub, R11, R10);
                self.buf.mov_r32_r32(R10, R11);
                if set_flags {
                    self.set_nzcv_sub();
                }
            }
            4 | 11 => {
                if compare {
                    self.buf.mov_r32_r32(R11, R10);
                    self.buf.alu_r32_r32(AluOp::Add, R11, b_reg);
                } else {
                    self.buf.alu_r32_r32(AluOp::Add, R10, b_reg);
                }
                if set_flags {
                    self.set_nzcv_add();
                }
            }
            5 => {
                self.buf.bt_r32_imm8(R15, 29);
                self.buf.alu_r32_r32(AluOp::Adc, R10, b_reg);
                if set_flags {
                    self.set_nzcv_add();
                }
            }
            6 => {
                self.buf.bt_r32_imm8(R15, 29);
                self.buf.cmc();
                self.buf.alu_r32_r32(AluOp::Sbb, R10, b_reg);
                if set_flags {
                    self.set_nzcv_sub();
                }
            }
            7 => {
                self.buf.mov_r32_r32(R11, b_reg);
                self.buf.bt_r32_imm8(R15, 29);
                self.buf.cmc();
                self.buf.alu_r32_r32(AluOp::Sbb, R11, R10);
                self.buf.mov_r32_r32(R10, R11);
                if set_flags {
                    self.set_nzcv_sub();
                }
            }
            8 => {
                self.buf.test_r32_r32(R10, b_reg);
                if set_flags {
                    self.update_logical_flags(sc_reg);
                }
            }
            12 => {
                self.buf.alu_r32_r32(AluOp::Or, R10, b_reg);
                if set_flags {
                    self.update_logical_flags(sc_reg);
                }
            }
            13 => {
                self.buf.mov_r32_r32(R10, b_reg);
                if set_flags {
                    self.buf.test_r32_r32(R10, R10);
                    self.update_logical_flags(sc_reg);
                }
            }
            14 => {
                self.buf.not_r32(b_reg);
                self.buf.alu_r32_r32(AluOp::And, R10, b_reg);
                if set_flags {
                    self.update_logical_flags(sc_reg);
                }
            }
            _ => {
                self.buf.mov_r32_r32(R10, b_reg);
                self.buf.not_r32(R10);
                if set_flags {
                    self.buf.test_r32_r32(R10, R10);
                    self.update_logical_flags(sc_reg);
                }
            }
        }

        if compare {
            true
        } else if rd != 15 {
            self.store_gpr(rd, R10);
            true
        } else if opcode == 13 && (op & 0x0200_0FFF) == 14 {
            // `mov pc, lr` returns without mode switch.
            self.buf.alu_r32_imm32(AluOp::And, R10, !3u32 as i32);
            self.buf.str_r32_mem(R10, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.emit_epilogue_jump();
            false
        } else {
            self.buf.alu_r32_imm32(AluOp::And, R10, !3u32 as i32);
            self.buf.str_r32_mem(R10, RBX, (offsets::CTX_R0 + 15 * 4) as i32);
            self.stub_jump(StubId::IndirectJump);
            false
        }
    }

    fn arm_multiply(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 16) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        self.load_gpr(R10, op & 0xF, a + 8);
        self.load_gpr(RCX, (op >> 8) & 0xF, a + 8);
        self.buf.imul_r32_r32(R10, RCX);
        if (op & (1 << 21)) != 0 {
            self.load_gpr(RCX, (op >> 12) & 0xF, a + 8);
            self.buf.alu_r32_r32(AluOp::Add, R10, RCX);
        }
        if (op & (1 << 20)) != 0 {
            self.buf.test_r32_r32(R10, R10);
            self.update_logical_flags(None);
        }
        self.store_gpr(rd, R10);
        true
    }

    fn arm_multiply_long(&mut self, a: u32, op: u32) -> bool {
        let hi = (op >> 16) & 0xF;
        let lo = (op >> 12) & 0xF;
        if hi == 15 || lo == 15 {
            return self.emit_interpret(a, op);
        }
        self.load_gpr(RAX, op & 0xF, a + 8);
        self.load_gpr(RCX, (op >> 8) & 0xF, a + 8);
        let kind = (op >> 21) & 7;
        let signed = kind >= 0b110;
        if signed {
            self.buf.movsxd_r64_r32(RAX, RAX);
            self.buf.movsxd_r64_r32(RCX, RCX);
        }
        self.buf.imul_r64_r64(RAX, RCX);
        match kind {
            0b010 => {
                // UMAAL
                self.load_gpr(RCX, lo, a + 8);
                self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                self.load_gpr(RCX, hi, a + 8);
                self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
            }
            0b101 | 0b111 => {
                // UMLAL / SMLAL
                self.load_gpr(RCX, lo, a + 8);
                self.load_gpr(RDX, hi, a + 8);
                self.buf.shift_r64_imm(ShiftOp::Shl, RDX, 32);
                self.buf.alu_r64_r64(AluOp::Or, RCX, RDX);
                self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
            }
            _ => {}
        }
        self.store_gpr(lo, RAX);
        self.buf.mov_r64_r64(R10, RAX);
        self.buf.shift_r64_imm(ShiftOp::Shr, RAX, 32);
        self.store_gpr(hi, RAX);
        if kind != 0b010 && (op & (1 << 20)) != 0 {
            self.buf.test_r64_r64(R10, R10);
            self.update_logical_flags(None);
        }
        true
    }

    fn arm_halfword_multiply(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 16) & 0xF;
        let rn = (op >> 12) & 0xF;
        let kind = (op >> 21) & 3;
        if rd == 15 || (kind == 0b10 && rn == 15) {
            return self.emit_interpret(a, op);
        }
        let (x, y) = ((op >> 5) & 1 != 0, (op >> 6) & 1 != 0);
        let extract_half = |buf: &mut X64Buf, reg: u8, top: bool| {
            if top {
                buf.shift_r32_imm(ShiftOp::Sar, reg, 16);
            } else {
                buf.movsx_r32_r16(reg, reg);
            }
        };

        self.load_gpr(RAX, op & 0xF, a + 8);
        self.load_gpr(RCX, (op >> 8) & 0xF, a + 8);

        match kind {
            0b00 => {
                // SMLAxy
                extract_half(&mut self.buf, RAX, x);
                self.buf.movsxd_r64_r32(RAX, RAX);
                extract_half(&mut self.buf, RCX, y);
                self.buf.movsxd_r64_r32(RCX, RCX);
                self.buf.imul_r64_r64(RAX, RCX);
                self.load_gpr(RCX, rn, a + 8);
                self.buf.movsxd_r64_r32(RCX, RCX);
                self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                self.emit_accumulate_q(R10, RAX);
                self.store_gpr(rd, R10);
            }
            0b01 => {
                // SMLAWy (x=0) / SMULWy (x=1)
                self.buf.movsxd_r64_r32(RAX, RAX);
                extract_half(&mut self.buf, RCX, y);
                self.buf.movsxd_r64_r32(RCX, RCX);
                self.buf.imul_r64_r64(RAX, RCX);
                self.buf.shift_r64_imm(ShiftOp::Sar, RAX, 16);
                if x {
                    self.store_gpr(rd, RAX);
                } else {
                    self.load_gpr(RCX, rn, a + 8);
                    self.buf.movsxd_r64_r32(RCX, RCX);
                    self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                    self.emit_accumulate_q(R10, RAX);
                    self.store_gpr(rd, R10);
                }
            }
            0b10 => {
                // SMLALxy
                extract_half(&mut self.buf, RAX, x);
                self.buf.movsxd_r64_r32(RAX, RAX);
                extract_half(&mut self.buf, RCX, y);
                self.buf.movsxd_r64_r32(RCX, RCX);
                self.buf.imul_r64_r64(RAX, RCX);
                self.load_gpr(RCX, rn, a + 8);
                self.load_gpr(RDX, rd, a + 8);
                self.buf.shift_r64_imm(ShiftOp::Shl, RDX, 32);
                self.buf.alu_r64_r64(AluOp::Or, RCX, RDX);
                self.buf.alu_r64_r64(AluOp::Add, RAX, RCX);
                self.store_gpr(rn, RAX);
                self.buf.shift_r64_imm(ShiftOp::Shr, RAX, 32);
                self.store_gpr(rd, RAX);
            }
            _ => {
                // SMULxy
                extract_half(&mut self.buf, RAX, x);
                extract_half(&mut self.buf, RCX, y);
                self.buf.imul_r32_r32(RAX, RCX);
                self.store_gpr(rd, RAX);
            }
        }
        true
    }

    fn arm_saturating(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let kind = (op >> 21) & 3;
        self.load_gpr(R8, op & 0xF, a + 8);
        self.buf.movsxd_r64_r32(R8, R8);
        self.load_gpr(RAX, (op >> 16) & 0xF, a + 8);
        self.buf.movsxd_r64_r32(RAX, RAX);
        if kind >= 2 {
            // QDADD / QDSUB: first saturate `2 * Rn`
            self.buf.shift_r64_imm(ShiftOp::Shl, RAX, 1);
            self.emit_saturate(RAX, RAX, 32, false);
            self.buf.movsxd_r64_r32(RAX, RAX);
        }
        if (kind & 1) == 0 {
            self.buf.alu_r64_r64(AluOp::Add, R8, RAX);
        } else {
            self.buf.alu_r64_r64(AluOp::Sub, R8, RAX);
        }
        self.emit_saturate(R10, R8, 32, false);
        self.store_gpr(rd, R10);
        true
    }

    fn arm_branch_exchange(&mut self, a: u32, op: u32) -> bool {
        let rm = op & 0xF;
        if ((op >> 4) & 0xF) == 0b0011 {
            // BLX Rm
            self.load_gpr(R10, rm, a + 8);
            self.buf.str_imm32_mem(RBX, (offsets::CTX_R0 + 14 * 4) as i32, a + 4);
            self.write_interworking_pc(R10);
            self.emit_depth_check(None);
            self.stub_call(StubId::IndirectCall);
            self.emit_post_call_guard(a + 4, false);
            return true;
        }
        if rm == 14 {
            self.load_gpr(RAX, 14, a + 8);
            self.emit_return_to_reg(RAX)
        } else {
            self.load_gpr(RAX, rm, a + 8);
            self.emit_jump_to_reg(RAX)
        }
    }
}
