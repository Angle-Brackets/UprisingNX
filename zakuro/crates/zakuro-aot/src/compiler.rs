//! Direct ARMv6K + Thumb-1 + VFPv2 -> AArch64 Ahead-of-Time (AOT) compiler.
//!
//! Register allocation inside AOT execution:
//! - `w0..w14`:  Guest `r[0]..r[14]` (`w13` = `sp`, `w14` = `lr`)
//! - `w15`:      Guest `r[15]` (`pc` / branch target)
//! - `x16, x17`: Scratch GPRs (caller-saved)
//! - `x18`:      Reserved by Horizon OS / platform ABI — NEVER touched
//! - `x19`:      `*mut AotExecContext` (offset 0 is `recomp_abi::Context`)
//! - `x20`:      `ctx.read_pages` (`*const *mut u8`)
//! - `x21`:      `ctx.write_pages` (`*const *mut u8`)
//! - `w22`:      `ctx.budget` (`i32`)
//! - `w23`:      Guest `NZCV` in bits `[31:28]`, bits `[27:0] == 0`
//! - `w24`:      `ctx.depth` (`u32`, `<= 2048`)
//! - `x25, x26`: Scratch GPRs (`w26` carries slow-path read return / write value)
//! - `x27`:      Scratch GPR preserved across `slow_read*` / `slow_write*`
//! - `w28`:      `ctx.exit` (`0` = running, `1` = SVC, `2` = BUDGET, `3` = UNWIND)
//! - `x29, x30`: Host frame pointer & link register
//! - `s0..s31`:  Guest VFP `s0..s31` (`v0.s[0]..v31.s[0]`); `s30, s31` spilled to
//!               `[x19, #SCRATCH_FP]` when `d30/d31` (`f64`) or `v30/v31` (ASIMD) are needed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use recomp_abi::{EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND, NO_ORIGIN};
use zakuro_fs::ExHeader;

use crate::a64::{self, cond, CodeBuf, Shift, SP, ZR};
use crate::discover::{
    self, analyze, build_origins, decode_arm, decode_thumb, key, recompiles, Function, Image,
    Mode, Program, Segment,
};
use crate::hints_kiu::{builtin_hints, KIU_USA_PROGRAM_ID};
use crate::kiu_overrides::overrides_for;
use crate::runtime::{offsets, AotImage};

const RECOMP_DEPTH_LIMIT: u16 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    /// Fold PC-relative loads from immutable `.text` literal pools into immediate moves.
    pub fold_text_literals: bool,
    /// Emit `FPSCR.LEN != 0` short-vector checks on VFP arithmetic (`false` for KIU, which
    /// never uses short vectors).
    pub emit_vfp_vectors: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self { fold_text_literals: true, emit_vfp_vectors: true }
    }
}

impl CompileOptions {
    pub fn for_title(program_id: u64) -> Self {
        Self {
            fold_text_literals: true,
            emit_vfp_vectors: program_id != KIU_USA_PROGRAM_ID,
        }
    }
}

/// Compiles a 3DS title's decompressed `.code` image into a position-independent AArch64 `AotImage`.
pub fn compile_title(
    program_id: u64,
    exheader: &ExHeader,
    code: &[u8],
    extra_hints: &[u32],
    options: CompileOptions,
    progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    let image = Image::from_exheader(exheader, code);
    let text_hash = image.text.hash();
    let mut hints = builtin_hints(program_id, text_hash);
    hints.extend_from_slice(extra_hints);
    let ovr_list = overrides_for(program_id, text_hash);
    let ovr_addrs: Vec<u32> = ovr_list.iter().map(|&(a, _)| a).collect();

    let program = image.into_program(&ovr_addrs, &hints);
    compile_program(program_id, text_hash, &program, &ovr_addrs, options, progress)
}

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

    // Global 1.0x instruction deduplication:
    // Because every compiled function shares the exact same 16-byte `[x29, x30]` host stack frame
    // and pinned hardware register map, shared tails / mid-function entries never need to be
    // duplicated across functions — non-owners jump directly (`b label_body`) into the single owner!
    let mut owner_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut global_labels: BTreeSet<u32> = BTreeSet::new();
    for (&entry, f) in &raw_functions {
        let f_key = key(entry, f.mode);
        owner_of.insert(f_key, f_key);
        for &lbl in &f.labels {
            global_labels.insert(key(lbl, f.mode));
        }
    }
    // Overridden functions claim all of their instructions first so `raw(ctx)` inside an override
    // always owns and executes its complete original body.
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
    // Any fallthrough across an ownership boundary becomes an explicit label in the target owner:
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
        for &lbl in &labels {
            known_targets.insert(key(lbl, f.mode));
        }
        dedup_funcs.push((
            entry,
            Function { source: f.source, mode: f.mode, instructions, labels },
        ));
    }

    let total_funcs = dedup_funcs.len().max(1);
    let done_funcs = AtomicUsize::new(0);

    // Compile functions in parallel chunks using a temporary scoped pool (explicit 2 MiB stacks
    // and 3 cores on Horizon Switch, freeing all worker stacks as soon as bootstrap finishes).
    const CHUNK: usize = 256;
    let compile_slice = |slice: &[(u32, Function)]| -> Vec<CompiledFunc> {
        let mut out = Vec::with_capacity(slice.len());
        for (entry, func) in slice {
            out.push(compile_one_function(program, &known_targets, *entry, func, options));
        }
        let prev = done_funcs.fetch_add(slice.len(), Ordering::Relaxed) + slice.len();
        let pct = 10 + (prev * 82) / total_funcs;
        progress(pct.min(92), 100);
        out
    };
    let num_threads = if cfg!(target_os = "horizon") {
        3
    } else {
        std::thread::available_parallelism().map_or(3, |n| n.get().clamp(1, 16))
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(2 * 1024 * 1024)
        .thread_name(|i| format!("aot-{i}"))
        .build();
    let compiled_chunks: Vec<Vec<CompiledFunc>> = match pool {
        Ok(pool) => pool.install(|| dedup_funcs.par_chunks(CHUNK).map(compile_slice).collect()),
        Err(_) => dedup_funcs.chunks(CHUNK).map(compile_slice).collect(),
    };

    let compiled_funcs: Vec<CompiledFunc> = compiled_chunks.into_iter().flatten().collect();

    // Emit shared runtime stubs at word 0 of the final code buffer.
    let mut final_buf = CodeBuf::with_capacity(
        4096 + compiled_funcs.iter().map(|f| f.buf.words.len() + 16).sum::<usize>(),
    );
    let stubs = emit_runtime_stubs(&mut final_buf);

    // Map override addresses -> override index (0..8).
    let mut override_map: BTreeMap<u32, usize> = BTreeMap::new();
    for (idx, &addr) in override_addrs.iter().take(8).enumerate() {
        override_map.insert(addr, idx);
    }

    // Place every function in `final_buf`, recording:
    // - `prologue_words`: key(entry, mode) -> prologue word index (for DirectCall)
    // - `body_words`:     key(entry, mode) -> entry body word index (for DirectTailJump)
    // - `entries_map`:    key(label, mode) -> (body_byte_offset, owner_func_idx)
    let mut prologue_words: BTreeMap<u32, usize> = BTreeMap::new();
    let mut body_words: BTreeMap<u32, usize> = BTreeMap::new();
    let mut entries_map: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
    let mut func_relocs: Vec<(usize, Vec<Reloc>)> = Vec::with_capacity(compiled_funcs.len());
    let mut functions_spans: Vec<(u32, Vec<(u32, u32)>)> = Vec::with_capacity(compiled_funcs.len());

    for (func_idx, cf) in compiled_funcs.into_iter().enumerate() {
        let base_word = final_buf.pos();
        final_buf.words.extend_from_slice(&cf.buf.words);

        let f_key = key(cf.entry, cf.mode);

        // Register all resumable internal labels (preferring the function whose own entry == label).
        let mut entry_local_word: Option<usize> = None;
        for &(label_addr, local_word) in &cf.label_words {
            let l_key = key(label_addr, cf.mode);
            let byte_off = ((base_word + local_word as usize) * 4) as u32;
            let slot = entries_map.entry(l_key).or_insert((byte_off, func_idx as u32));
            if label_addr == cf.entry {
                *slot = (byte_off, func_idx as u32);
                entry_local_word = Some(local_word as usize);
            }
        }

        // Only register `base_word` as `f_key`'s prologue when `cf` actually owns `cf.entry`!
        // (If `cf.entry` was already owned by an earlier container and `cf` only holds leftover
        // blocks, `call_veneers` / `entries_map` will route calls/jumps to the container's label.)
        if let Some(entry_local) = entry_local_word {
            let raw_prologue = base_word;
            let raw_body = base_word + entry_local;
            if let Some(&ovr_idx) = override_map.get(&f_key) {
                let (ovr_prologue, ovr_body, ovr_relocs) =
                    emit_override_wrapper(&mut final_buf, ovr_idx, (raw_body * 4) as u32);
                prologue_words.insert(f_key, ovr_prologue);
                body_words.insert(f_key, ovr_body);
                entries_map.insert(f_key, ((ovr_body * 4) as u32, NO_ORIGIN));
                func_relocs.push((0, ovr_relocs));
            } else {
                prologue_words.insert(f_key, raw_prologue);
                body_words.insert(f_key, raw_body);
            }
        }

        func_relocs.push((base_word, cf.relocs));
        functions_spans.push((cf.entry, cf.spans));
    }

    // Also resolve calls targeting functions that were absorbed into a larger `container`!
    // Wait: for an absorbed function `sub_entry` inside `home_entry`, its code lives at
    // `entries_map[sub_key].0 / 4` (which is a label body inside `home`, WITHOUT a prologue right before it!).
    // So:
    // - `DirectTailJump(sub_key)` can jump straight to `sub_body_word`!
    // - `DirectCall(sub_key)`: if `sub_key` has no standalone prologue, emit a tiny 3-word call veneer
    //   (`stp x29, x30, [sp, #-16]!; mov x29, sp; b sub_body_word`) on demand!
    let mut call_veneers: BTreeMap<u32, usize> = BTreeMap::new();
    for (base_word, relocs) in &func_relocs {
        let _ = base_word;
        for rel in relocs {
            if let RelocKind::DirectCall(target_key) = rel.kind {
                if !prologue_words.contains_key(&target_key) && !call_veneers.contains_key(&target_key) {
                    if let Some(&(body_bytes, _)) = entries_map.get(&target_key) {
                        let target_body_word = (body_bytes / 4) as usize;
                        let veneer_word = final_buf.pos();
                        final_buf.emit(a64::stp64_pre(29, 30, SP, -16));
                        final_buf.emit(a64::mov_sp(29, SP));
                        let b_pos = final_buf.pos();
                        let delta = target_body_word as i32 - b_pos as i32;
                        final_buf.emit(a64::b_rel(delta));
                        call_veneers.insert(target_key, veneer_word);
                    }
                }
            }
        }
    }

    // Link all cross-function and runtime-stub `B` / `BL` relocations!
    for (base_word, relocs) in func_relocs {
        for rel in relocs {
            let at_word = base_word + rel.local_word as usize;
            let target_word = match rel.kind {
                RelocKind::StubB(s) | RelocKind::StubBl(s) => stubs.word_of(s),
                RelocKind::DirectCall(t_key) => prologue_words
                    .get(&t_key)
                    .or_else(|| call_veneers.get(&t_key))
                    .copied()
                    .unwrap_or(stubs.indirect_call),
                RelocKind::DirectTailJump(t_key) => body_words
                    .get(&t_key)
                    .copied()
                    .or_else(|| entries_map.get(&t_key).map(|&(b, _)| (b / 4) as usize))
                    .unwrap_or(stubs.indirect_jump),
            };
            let delta = target_word as i32 - at_word as i32;
            assert!(
                (-0x0200_0000..0x0200_0000).contains(&delta),
                "AOT branch displacement {delta} words exceeds ±128 MiB"
            );
            let insn = match rel.kind {
                RelocKind::StubBl(_) | RelocKind::DirectCall(_) => a64::bl_rel(delta),
                RelocKind::StubB(_) | RelocKind::DirectTailJump(_) => a64::b_rel(delta),
            };
            final_buf.patch(at_word, insn);
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
        enter_offset: (stubs.enter * 4) as u32,
        code: final_buf.words,
        entries,
        origins,
    }
}

// ---------------------------------------------------------------------------
// Shared Runtime Stubs Emitter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum StubId {
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
    shift_lsl_c: usize,
    shift_lsr_c: usize,
    shift_asr_c: usize,
    shift_ror_c: usize,
}

impl RuntimeStubs {
    fn word_of(&self, id: StubId) -> usize {
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
            StubId::ShiftLslC => self.shift_lsl_c,
            StubId::ShiftLsrC => self.shift_lsr_c,
            StubId::ShiftAsrC => self.shift_asr_c,
            StubId::ShiftRorC => self.shift_ror_c,
        }
    }
}

fn emit_runtime_stubs(buf: &mut CodeBuf) -> RuntimeStubs {
    // 1. sync_out (uses ONLY x16, x17 so x25, x26, x27 are preserved!)
    let sync_out = buf.pos();
    for i in 0..8u8 {
        buf.emit(a64::stp32_off(i * 2, i * 2 + 1, 19, (i as i16) * 8));
    }
    buf.emit(a64::ubfx32(16, 23, 31, 1));
    buf.emit(a64::strb_uoff(16, 19, offsets::CTX_N));
    buf.emit(a64::ubfx32(16, 23, 30, 1));
    buf.emit(a64::strb_uoff(16, 19, offsets::CTX_Z));
    buf.emit(a64::ubfx32(16, 23, 29, 1));
    buf.emit(a64::strb_uoff(16, 19, offsets::CTX_C));
    buf.emit(a64::ubfx32(16, 23, 28, 1));
    buf.emit(a64::strb_uoff(16, 19, offsets::CTX_V));
    buf.emit(a64::str32_uoff(22, 19, offsets::CTX_BUDGET));
    buf.emit(a64::str32_uoff(28, 19, offsets::CTX_EXIT));
    buf.emit(a64::str32_uoff(24, 19, offsets::CTX_DEPTH));
    buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_VFP));
    for i in 0..16u8 {
        buf.emit(a64::stp_s_off(i * 2, i * 2 + 1, 17, (i as i16) * 8));
    }
    buf.emit(a64::ret());

    // 2. sync_in (uses ONLY x16, x17 so x25, x26, x27 are preserved, and sets hardware NZCV = w23!)
    let sync_in = buf.pos();
    for i in 0..8u8 {
        buf.emit(a64::ldp32_off(i * 2, i * 2 + 1, 19, (i as i16) * 8));
    }
    buf.emit(a64::ldrb_uoff(23, 19, offsets::CTX_N));
    buf.emit(a64::lsl32_imm(23, 23, 31));
    buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_Z));
    buf.emit(a64::orr32_reg(23, 23, 16, Shift::Lsl, 30));
    buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_C));
    buf.emit(a64::orr32_reg(23, 23, 16, Shift::Lsl, 29));
    buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_V));
    buf.emit(a64::orr32_reg(23, 23, 16, Shift::Lsl, 28));
    buf.emit(a64::ldr32_uoff(22, 19, offsets::CTX_BUDGET));
    buf.emit(a64::ldr32_uoff(28, 19, offsets::CTX_EXIT));
    buf.emit(a64::ldr32_uoff(24, 19, offsets::CTX_DEPTH));
    buf.emit(a64::ldr64_uoff(20, 19, offsets::CTX_READ_PAGES));
    buf.emit(a64::ldr64_uoff(21, 19, offsets::CTX_WRITE_PAGES));
    buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_VFP));
    for i in 0..16u8 {
        buf.emit(a64::ldp_s_off(i * 2, i * 2 + 1, 17, (i as i16) * 8));
    }
    buf.emit(a64::msr_nzcv(23));
    buf.emit(a64::ret());

    // Helper for `bl target_word` within stubs:
    let bl_to = |buf: &mut CodeBuf, target: usize| {
        let p = buf.pos();
        buf.emit(a64::bl_rel(target as i32 - p as i32));
    };

    // invoke_x27 helper (`stp x29, x30, [sp, #-16]!; mov x29, sp; br x27`)
    let invoke_x27 = buf.pos();
    buf.emit(a64::stp64_pre(29, 30, SP, -16));
    buf.emit(a64::mov_sp(29, SP));
    buf.emit(a64::br(27));

    // 3. enter(ec: *mut AotExecContext, target_body_rx: *const u8)
    let enter = buf.pos();
    buf.emit(a64::stp64_pre(29, 30, SP, -160));
    buf.emit(a64::mov_sp(29, SP));
    buf.emit(a64::stp64_off(19, 20, SP, 16));
    buf.emit(a64::stp64_off(21, 22, SP, 32));
    buf.emit(a64::stp64_off(23, 24, SP, 48));
    buf.emit(a64::stp64_off(25, 26, SP, 64));
    buf.emit(a64::stp64_off(27, 28, SP, 80));
    buf.emit(a64::stp_d_off(8, 9, SP, 96));
    buf.emit(a64::stp_d_off(10, 11, SP, 112));
    buf.emit(a64::stp_d_off(12, 13, SP, 128));
    buf.emit(a64::stp_d_off(14, 15, SP, 144));
    buf.emit(a64::mov64(19, 0));
    buf.emit(a64::mov64(27, 1));
    bl_to(buf, sync_in);
    bl_to(buf, invoke_x27);
    bl_to(buf, sync_out);
    buf.emit(a64::ldp_d_off(14, 15, SP, 144));
    buf.emit(a64::ldp_d_off(12, 13, SP, 128));
    buf.emit(a64::ldp_d_off(10, 11, SP, 112));
    buf.emit(a64::ldp_d_off(8, 9, SP, 96));
    buf.emit(a64::ldp64_off(27, 28, SP, 80));
    buf.emit(a64::ldp64_off(25, 26, SP, 64));
    buf.emit(a64::ldp64_off(23, 24, SP, 48));
    buf.emit(a64::ldp64_off(21, 22, SP, 32));
    buf.emit(a64::ldp64_off(19, 20, SP, 16));
    buf.emit(a64::ldp64_post(29, 30, SP, 160));
    buf.emit(a64::ret());

    // 4. Slow memory read stubs (input: w16 = addr; output: w26 = val; preserves x27 & NZCV)
    let emit_slow_read = |buf: &mut CodeBuf, fn_off: u16| -> usize {
        let start = buf.pos();
        buf.emit(a64::stp64_pre(29, 30, SP, -32));
        buf.emit(a64::mov_sp(29, SP));
        buf.emit(a64::str32_uoff(16, SP, 16));
        bl_to(buf, sync_out);
        buf.emit(a64::mov64(0, 19));
        buf.emit(a64::ldr32_uoff(1, SP, 16));
        buf.emit(a64::ldr64_uoff(16, 19, fn_off));
        buf.emit(a64::blr(16));
        buf.emit(a64::str32_uoff(0, SP, 16));
        bl_to(buf, sync_in);
        buf.emit(a64::ldr32_uoff(26, SP, 16));
        buf.emit(a64::ldp64_post(29, 30, SP, 32));
        buf.emit(a64::ret());
        start
    };
    let slow_read8 = emit_slow_read(buf, offsets::FN_SLOW_READ8);
    let slow_read16 = emit_slow_read(buf, offsets::FN_SLOW_READ16);
    let slow_read32 = emit_slow_read(buf, offsets::FN_SLOW_READ32);

    // 5. Slow memory write stubs (input: w16 = addr, w26 = val; preserves x27 & NZCV)
    let emit_slow_write = |buf: &mut CodeBuf, fn_off: u16| -> usize {
        let start = buf.pos();
        buf.emit(a64::stp64_pre(29, 30, SP, -32));
        buf.emit(a64::mov_sp(29, SP));
        buf.emit(a64::stp32_off(16, 26, SP, 16));
        bl_to(buf, sync_out);
        buf.emit(a64::mov64(0, 19));
        buf.emit(a64::ldp32_off(1, 2, SP, 16));
        buf.emit(a64::ldr64_uoff(16, 19, fn_off));
        buf.emit(a64::blr(16));
        bl_to(buf, sync_in);
        buf.emit(a64::ldp64_post(29, 30, SP, 32));
        buf.emit(a64::ret());
        start
    };
    let slow_write8 = emit_slow_write(buf, offsets::FN_SLOW_WRITE8);
    let slow_write16 = emit_slow_write(buf, offsets::FN_SLOW_WRITE16);
    let slow_write32 = emit_slow_write(buf, offsets::FN_SLOW_WRITE32);

    // 6. Indirect call & indirect jump (uses 8192-slot direct-mapped icache!)
    let indirect_call = buf.pos();
    buf.emit(a64::stp64_pre(29, 30, SP, -16));
    buf.emit(a64::mov_sp(29, SP));
    let indirect_jump = buf.pos();
    buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_THUMB));
    buf.emit(a64::orr32_reg(16, 15, 16, Shift::Lsl, 0)); // w16 = key (zero-extended in x16)
    buf.mov_imm32(17, 0x9E37_79B1);
    buf.emit(a64::mul32(17, 16, 17));
    buf.emit(a64::lsr32_imm(17, 17, 19)); // 8192 slots -> top 13 bits
    buf.emit(a64::ldr64_uoff(25, 19, offsets::ICACHE_PTR));
    buf.emit(a64::add64_reg(25, 25, 17, Shift::Lsl, 4)); // 16 bytes/slot
    buf.emit(a64::ldp64_off(26, 27, 25, 0));
    buf.emit(a64::sub64_reg(26, 26, 16, Shift::Lsl, 0));
    let j_miss = buf.emit(a64::nop()); // cbnz64(26, miss)
    let j_unw1 = buf.emit(a64::nop()); // cbz64(27, unwind)
    buf.emit(a64::br(27));

    let miss_pos = buf.pos();
    buf.patch(j_miss, a64::cbnz64(26, miss_pos as i32 - j_miss as i32));
    buf.emit(a64::stp64_pre(16, ZR, SP, -16));
    bl_to(buf, sync_out);
    buf.emit(a64::mov64(0, 19));
    buf.emit(a64::ldp64_post(1, ZR, SP, 16));
    buf.emit(a64::ldr64_uoff(16, 19, offsets::FN_LOOKUP_BODY));
    buf.emit(a64::blr(16));
    buf.emit(a64::stp64_pre(0, ZR, SP, -16));
    bl_to(buf, sync_in);
    buf.emit(a64::ldp64_post(27, ZR, SP, 16));
    let j_unw2 = buf.emit(a64::nop()); // cbz64(27, unwind)
    buf.emit(a64::br(27));

    let unwind_pos = buf.pos();
    buf.patch(j_unw1, a64::cbz64(27, unwind_pos as i32 - j_unw1 as i32));
    buf.patch(j_unw2, a64::cbz64(27, unwind_pos as i32 - j_unw2 as i32));
    buf.mov_imm32(28, EXIT_UNWIND);
    buf.emit(a64::ldp64_post(29, 30, SP, 16));
    buf.emit(a64::ret());

    // 7. Interpret stub (input: w16 = addr, w26 = op)
    let interpret = emit_slow_write(buf, offsets::FN_INTERPRET);

    // 8. Media parallel helper stub (input: w16 = packed, w17 = a, w26 = b; output: w26 = res)
    let media_parallel = buf.pos();
    buf.emit(a64::stp64_pre(29, 30, SP, -32));
    buf.emit(a64::mov_sp(29, SP));
    buf.emit(a64::stp32_off(16, 17, SP, 16));
    buf.emit(a64::str32_uoff(26, SP, 24));
    bl_to(buf, sync_out);
    buf.emit(a64::mov64(0, 19));
    buf.emit(a64::ldp32_off(1, 2, SP, 16));
    buf.emit(a64::ldr32_uoff(3, SP, 24));
    buf.emit(a64::ldr64_uoff(16, 19, offsets::FN_MEDIA_PARALLEL));
    buf.emit(a64::blr(16));
    buf.emit(a64::str32_uoff(0, SP, 16));
    bl_to(buf, sync_in);
    buf.emit(a64::ldr32_uoff(26, SP, 16));
    buf.emit(a64::ldp64_post(29, 30, SP, 32));
    buf.emit(a64::ret());

    // 9. VFP short-vector helper stub (input: w16 = packed_vfp)
    let vfp_vector = emit_slow_write(buf, offsets::FN_VFP_VECTOR);

    // 10. Pure-scratch register-shift-with-carry subroutines
    // Input: w16 = val, w17 = raw Rs, w23 = NZCV<<28. Output: w16 = shifted, w25 = carry (0/1).
    let shift_lsl_c = buf.pos();
    buf.emit(a64::ubfx32(25, 23, 29, 1));
    buf.emit(a64::uxtb32(17, 17));
    let lsl_z = buf.emit(a64::nop());
    buf.emit(a64::cmp32_imm(17, 32));
    let lsl_hi = buf.emit(a64::nop());
    let lsl_eq = buf.emit(a64::nop());
    buf.mov_imm32(26, 32);
    buf.emit(a64::sub32_reg(26, 26, 17, Shift::Lsl, 0));
    buf.emit(a64::lsrv32(25, 16, 26));
    buf.emit(a64::and32_imm(25, 25, 1).unwrap());
    buf.emit(a64::lslv32(16, 16, 17));
    buf.emit(a64::ret());
    let p_eq = buf.pos();
    buf.patch(lsl_eq, a64::b_cond(cond::EQ, p_eq as i32 - lsl_eq as i32));
    buf.emit(a64::and32_imm(25, 16, 1).unwrap());
    buf.emit(a64::mov32(16, ZR));
    buf.emit(a64::ret());
    let p_hi = buf.pos();
    buf.patch(lsl_hi, a64::b_cond(cond::HI, p_hi as i32 - lsl_hi as i32));
    buf.emit(a64::mov32(25, ZR));
    buf.emit(a64::mov32(16, ZR));
    let p_ret = buf.pos();
    buf.patch(lsl_z, a64::cbz32(17, p_ret as i32 - lsl_z as i32));
    buf.emit(a64::ret());

    let shift_lsr_c = buf.pos();
    buf.emit(a64::ubfx32(25, 23, 29, 1));
    buf.emit(a64::uxtb32(17, 17));
    let lsr_z = buf.emit(a64::nop());
    buf.emit(a64::cmp32_imm(17, 32));
    let lsr_hi = buf.emit(a64::nop());
    let lsr_eq = buf.emit(a64::nop());
    buf.emit(a64::sub32_imm(26, 17, 1, false));
    buf.emit(a64::lsrv32(25, 16, 26));
    buf.emit(a64::and32_imm(25, 25, 1).unwrap());
    buf.emit(a64::lsrv32(16, 16, 17));
    buf.emit(a64::ret());
    let p_eq = buf.pos();
    buf.patch(lsr_eq, a64::b_cond(cond::EQ, p_eq as i32 - lsr_eq as i32));
    buf.emit(a64::lsr32_imm(25, 16, 31));
    buf.emit(a64::mov32(16, ZR));
    buf.emit(a64::ret());
    let p_hi = buf.pos();
    buf.patch(lsr_hi, a64::b_cond(cond::HI, p_hi as i32 - lsr_hi as i32));
    buf.emit(a64::mov32(25, ZR));
    buf.emit(a64::mov32(16, ZR));
    let p_ret = buf.pos();
    buf.patch(lsr_z, a64::cbz32(17, p_ret as i32 - lsr_z as i32));
    buf.emit(a64::ret());

    let shift_asr_c = buf.pos();
    buf.emit(a64::ubfx32(25, 23, 29, 1));
    buf.emit(a64::uxtb32(17, 17));
    let asr_z = buf.emit(a64::nop());
    buf.emit(a64::cmp32_imm(17, 32));
    let asr_hs = buf.emit(a64::nop());
    buf.emit(a64::sub32_imm(26, 17, 1, false));
    buf.emit(a64::lsrv32(25, 16, 26));
    buf.emit(a64::and32_imm(25, 25, 1).unwrap());
    buf.emit(a64::asrv32(16, 16, 17));
    buf.emit(a64::ret());
    let p_hs = buf.pos();
    buf.patch(asr_hs, a64::b_cond(cond::HS, p_hs as i32 - asr_hs as i32));
    buf.emit(a64::lsr32_imm(25, 16, 31));
    buf.emit(a64::asr32_imm(16, 16, 31));
    let p_ret = buf.pos();
    buf.patch(asr_z, a64::cbz32(17, p_ret as i32 - asr_z as i32));
    buf.emit(a64::ret());

    let shift_ror_c = buf.pos();
    buf.emit(a64::ubfx32(25, 23, 29, 1));
    buf.emit(a64::uxtb32(17, 17));
    let ror_z = buf.emit(a64::nop());
    buf.emit(a64::rorv32(16, 16, 17));
    buf.emit(a64::lsr32_imm(25, 16, 31));
    let p_ret = buf.pos();
    buf.patch(ror_z, a64::cbz32(17, p_ret as i32 - ror_z as i32));
    buf.emit(a64::ret());

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
        shift_lsl_c,
        shift_lsr_c,
        shift_asr_c,
        shift_ror_c,
    }
}

fn emit_override_wrapper(
    buf: &mut CodeBuf,
    ovr_idx: usize,
    raw_body_byte_offset: u32,
) -> (usize, usize, Vec<Reloc>) {
    let mut relocs = Vec::new();
    let prologue = buf.pos();
    buf.emit(a64::stp64_pre(29, 30, SP, -16));
    buf.emit(a64::mov_sp(29, SP));
    let body = buf.pos();

    let s1 = buf.emit(a64::nop());
    relocs.push(Reloc { local_word: s1 as u32, kind: RelocKind::StubBl(StubId::SyncOut) });

    buf.emit(a64::mov64(0, 19));
    buf.emit(a64::ldr64_uoff(1, 19, offsets::RX_BASE));
    buf.mov_imm32(16, raw_body_byte_offset);
    buf.emit(a64::add64_uxtw(1, 1, 16, 0));
    buf.emit(a64::ldr64_uoff(16, 19, offsets::OVERRIDES_BASE + (ovr_idx as u16) * 8));
    buf.emit(a64::blr(16));

    let s2 = buf.emit(a64::nop());
    relocs.push(Reloc { local_word: s2 as u32, kind: RelocKind::StubBl(StubId::SyncIn) });

    buf.emit(a64::ldp64_post(29, 30, SP, 16));
    buf.emit(a64::ret());
    (prologue, body, relocs)
}

// ---------------------------------------------------------------------------
// Per-Function Compiler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum RelocKind {
    StubB(StubId),
    StubBl(StubId),
    DirectCall(u32),
    DirectTailJump(u32),
}

#[derive(Debug, Clone, Copy)]
struct Reloc {
    local_word: u32,
    kind: RelocKind,
}

struct CompiledFunc {
    entry: u32,
    mode: Mode,
    buf: CodeBuf,
    label_words: Vec<(u32, u32)>,
    relocs: Vec<Reloc>,
    spans: Vec<(u32, u32)>,
}

struct LocalPatch {
    at_word: usize,
    target_label: u32,
    cond: Option<u8>,
}

struct BudgetExit {
    patch_word: usize,
    guest_addr: u32,
    count: u32,
}

struct FnEmitter<'a> {
    text: &'a Segment,
    known_targets: &'a BTreeSet<u32>,
    labels: BTreeSet<u32>,
    mode: Mode,
    options: CompileOptions,
    buf: CodeBuf,
    relocs: Vec<Reloc>,
    local_patches: Vec<LocalPatch>,
    epilogue_patches: Vec<usize>,
    depth_overflow_patches: Vec<usize>,
    budget_exits: Vec<BudgetExit>,
    interpreted_addrs: Vec<u32>,
    folded_spans: Vec<(u32, u32)>,
    /// True iff hardware `NZCV` is guaranteed to equal `w23` on all paths reaching `buf.pos()`.
    flags_live: bool,
}

fn compile_one_function(
    program: &Program,
    known_targets: &BTreeSet<u32>,
    entry: u32,
    func: &Function,
    options: CompileOptions,
) -> CompiledFunc {
    // Pass 1: find any instructions handed to `interpret()`, whose successor becomes a resumable label.
    let first_pass = emit_function_pass(program, known_targets, entry, func, &func.labels, options);
    let final_compiled = if first_pass.1.is_empty() {
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
        emit_function_pass(program, known_targets, entry, func, &resumable, options).0
    };
    final_compiled
}

fn emit_function_pass(
    program: &Program,
    known_targets: &BTreeSet<u32>,
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
        buf: CodeBuf::with_capacity(func.instructions.len() * 6 + 16),
        relocs: Vec::new(),
        local_patches: Vec::new(),
        epilogue_patches: Vec::new(),
        depth_overflow_patches: Vec::new(),
        budget_exits: Vec::new(),
        interpreted_addrs: Vec::new(),
        folded_spans: Vec::new(),
        flags_live: false,
    };

    let thumb = func.mode == Mode::Thumb;
    let insns = &func.instructions;
    // Standard prologue (`stp x29, x30, [sp, #-16]!; mov x29, sp`) is only needed when this
    // chunk owns its own `entry`; if `insns[0] != entry` (function has backward-branch blocks at
    // addresses below `entry`), jump from the prologue straight to `entry`!
    if em.labels.contains(&entry) {
        em.buf.emit(a64::stp64_pre(29, 30, SP, -16));
        em.buf.emit(a64::mov_sp(29, SP));
        if insns.first().copied() != Some(entry) {
            let p = em.buf.emit(a64::nop());
            em.local_patches.push(LocalPatch { at_word: p, target_label: entry, cond: None });
        }
    }

    let mut label_map: BTreeMap<u32, u32> = BTreeMap::new();
    let mut spans: Vec<(u32, u32)> = Vec::new();

    for (i, &address) in insns.iter().enumerate() {
        if em.labels.contains(&address) {
            em.flags_live = false;
            label_map.insert(address, em.buf.pos() as u32);
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

    // Cold depth-overflow stub (if this function contains any calls):
    if !em.depth_overflow_patches.is_empty() {
        let depth_out = em.buf.pos();
        for p in &em.depth_overflow_patches {
            let delta = depth_out as i32 - *p as i32;
            em.buf.patch(*p, a64::b_cond(cond::HI, delta));
        }
        em.buf.emit(a64::sub32_imm(24, 24, 1, false));
        em.buf.mov_imm32(28, EXIT_UNWIND);
        let j_epi = em.buf.emit(a64::nop());
        em.epilogue_patches.push(j_epi);
    }

    // Cold budget-exhausted stubs at the end of the function (falls straight into epilogue!):
    if !em.budget_exits.is_empty() {
        let mut exit_jumps = Vec::with_capacity(em.budget_exits.len());
        let budget_exits = std::mem::take(&mut em.budget_exits);
        for be in budget_exits {
            let here = em.buf.pos();
            let delta = here as i32 - be.patch_word as i32;
            em.buf.patch(be.patch_word, a64::b_cond(cond::LT, delta));
            em.buf.add_const32(22, 22, be.count, 16);
            em.buf.mov_imm32(15, be.guest_addr);
            exit_jumps.push(em.buf.emit(a64::nop()));
        }
        let budget_out = em.buf.pos();
        for j in exit_jumps {
            em.buf.patch(j, a64::b_rel(budget_out as i32 - j as i32));
        }
        em.buf.mov_imm32(28, EXIT_BUDGET);
    }

    // Shared function epilogue:
    let epilogue = em.buf.pos();
    for p in &em.epilogue_patches {
        let delta = epilogue as i32 - *p as i32;
        if (-0x40000..0x40000).contains(&delta) && (em.buf.words[*p] & 0xFF00_0000) == 0x3500_0000 {
            em.buf.patch(*p, a64::cbnz32(28, delta));
        } else {
            em.buf.patch(*p, a64::b_rel(delta));
        }
    }
    em.buf.emit(a64::ldp64_post(29, 30, SP, 16));
    em.buf.emit(a64::ret());

    // Patch intra-function label branches (emitting a 1-word veneer at end of function if any
    // conditional branch exceeded `b.cond`'s ±1 MiB range).
    for lp in &em.local_patches {
        let target_word = label_map[&lp.target_label] as usize;
        let delta = target_word as i32 - lp.at_word as i32;
        match lp.cond {
            None => em.buf.patch(lp.at_word, a64::b_rel(delta)),
            Some(c) if (-0x40000..0x40000).contains(&delta) => {
                em.buf.patch(lp.at_word, a64::b_cond(c, delta));
            }
            Some(c) => {
                let veneer = em.buf.pos();
                let d1 = veneer as i32 - lp.at_word as i32;
                em.buf.patch(lp.at_word, a64::b_cond(c, d1));
                let d2 = target_word as i32 - veneer as i32;
                em.buf.emit(a64::b_rel(d2));
            }
        }
    }

    spans.extend_from_slice(&em.folded_spans);
    let label_words: Vec<(u32, u32)> = label_map.into_iter().collect();
    let interpreted = std::mem::take(&mut em.interpreted_addrs);
    (
        CompiledFunc {
            entry,
            mode: func.mode,
            buf: em.buf,
            label_words,
            relocs: em.relocs,
            spans,
        },
        interpreted,
    )
}

impl FnEmitter<'_> {
    #[inline]
    fn ensure_hw_flags(&mut self) {
        if !self.flags_live {
            self.buf.emit(a64::msr_nzcv(23));
            self.flags_live = true;
        }
    }

    #[inline]
    fn stub_call(&mut self, id: StubId) {
        let at = self.buf.emit(a64::nop());
        self.relocs.push(Reloc { local_word: at as u32, kind: RelocKind::StubBl(id) });
    }

    #[inline]
    fn stub_jump(&mut self, id: StubId) {
        let at = self.buf.emit(a64::nop());
        self.relocs.push(Reloc { local_word: at as u32, kind: RelocKind::StubB(id) });
    }

    fn emit_epilogue_jump(&mut self) {
        let p = self.buf.emit(a64::nop());
        self.epilogue_patches.push(p);
    }

    fn emit_check_unwind(&mut self) {
        // Check `w28 != 0`: if within ±1 MiB of epilogue, patched directly as `cbnz w28, epilogue`;
        // to be 100% safe even in giant functions, `cbz w28, +2; b epilogue` is 2 words and unlimited!
        self.buf.emit(a64::cbz32(28, 2));
        self.emit_epilogue_jump();
    }

    fn emit_block_budget(&mut self, guest_addr: u32, count: u32) {
        if count <= 0xFFF {
            self.buf.emit(a64::subs32_imm(22, 22, count as u16, false));
        } else {
            self.buf.mov_imm32(16, count);
            self.buf.emit(a64::subs32_reg(22, 22, 16, Shift::Lsl, 0));
        }
        self.flags_live = false;
        // Guard against > ±1 MiB in ultra-huge functions: `b.ge +2; b cold_stub` if needed,
        // or direct `b.lt cold_stub` (40,000 words << 262,144 words!).
        let patch_word = self.buf.emit(a64::nop());
        self.budget_exits.push(BudgetExit { patch_word, guest_addr, count });
    }

    /// Reads guest `reg` (`0..=15`) into a register, materializing `pc_val` into `scratch` if `reg == 15`.
    fn get_reg(&mut self, reg: u32, pc_val: u32, scratch: u8) -> u8 {
        if reg == 15 {
            self.buf.mov_imm32(scratch, pc_val);
            scratch
        } else {
            reg as u8
        }
    }

    // -----------------------------------------------------------------------
    // Memory Access Emitters (Zero NZCV clobber, x27 always preserved!)
    // -----------------------------------------------------------------------

    fn emit_read8(&mut self, dst: u8, signed: bool) {
        // Input: w16 = guest address.
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 20, 17));
        self.buf.emit(a64::cbnz64(17, 4));
        self.stub_call(StubId::SlowRead8);
        if signed {
            self.buf.emit(a64::sxtb32(dst, 26));
        } else if dst != 26 {
            self.buf.emit(a64::mov32(dst, 26));
        } else {
            self.buf.emit(a64::nop());
        }
        self.buf.emit(a64::b_rel(3));
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        if signed {
            self.buf.emit(a64::ldrsb32_reg_uxtw(dst, 17, 25));
        } else {
            self.buf.emit(a64::ldrb_reg_uxtw(dst, 17, 25));
        }
    }

    fn emit_read16(&mut self, dst: u8, signed: bool) {
        // Input: w16 = guest address.
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 20, 17));
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::cbz64(17, 3));
        self.buf.emit(a64::add32_imm(26, 25, 1, false));
        self.buf.emit(a64::tbz(26, 12, 4));
        self.stub_call(StubId::SlowRead16);
        if signed {
            self.buf.emit(a64::sxth32(dst, 26));
        } else if dst != 26 {
            self.buf.emit(a64::mov32(dst, 26));
        } else {
            self.buf.emit(a64::nop());
        }
        self.buf.emit(a64::b_rel(2));
        if signed {
            self.buf.emit(a64::ldrsh32_reg_uxtw(dst, 17, 25));
        } else {
            self.buf.emit(a64::ldrh_reg_uxtw(dst, 17, 25));
        }
    }

    fn emit_read32_gpr(&mut self, dst: u8) {
        // Input: w16 = guest address.
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 20, 17));
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::cbz64(17, 3));
        self.buf.emit(a64::add32_imm(26, 25, 3, false));
        self.buf.emit(a64::tbz(26, 12, 4));
        self.stub_call(StubId::SlowRead32);
        if dst != 26 {
            self.buf.emit(a64::mov32(dst, 26));
        } else {
            self.buf.emit(a64::nop());
        }
        self.buf.emit(a64::b_rel(2));
        self.buf.emit(a64::ldr32_reg_uxtw(dst, 17, 25));
    }

    fn emit_read32_vfp(&mut self, sd: u8) {
        // Input: w16 = guest address.
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 20, 17));
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::cbz64(17, 3));
        self.buf.emit(a64::add32_imm(26, 25, 3, false));
        self.buf.emit(a64::tbz(26, 12, 4));
        self.stub_call(StubId::SlowRead32);
        self.buf.emit(a64::fmov_s_from_w(sd, 26));
        self.buf.emit(a64::b_rel(2));
        self.buf.emit(a64::ldr_s_reg_uxtw(sd, 17, 25));
    }

    fn emit_write8(&mut self, val_reg: u8) {
        // Input: w16 = guest address, val_reg = value.
        if val_reg != 26 {
            self.buf.emit(a64::mov32(26, val_reg));
        }
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 21, 17));
        self.buf.emit(a64::cbnz64(17, 3));
        self.stub_call(StubId::SlowWrite8);
        self.buf.emit(a64::b_rel(3));
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::strb_reg_uxtw(26, 17, 25));
    }

    fn emit_write16(&mut self, val_reg: u8) {
        if val_reg != 26 {
            self.buf.emit(a64::mov32(26, val_reg));
        }
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::add32_imm(17, 25, 1, false));
        self.buf.emit(a64::tbnz(17, 12, 4));
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 21, 17));
        self.buf.emit(a64::cbnz64(17, 3));
        self.stub_call(StubId::SlowWrite16);
        self.buf.emit(a64::b_rel(2));
        self.buf.emit(a64::strh_reg_uxtw(26, 17, 25));
    }

    fn emit_write32_gpr(&mut self, val_reg: u8) {
        if val_reg != 26 {
            self.buf.emit(a64::mov32(26, val_reg));
        }
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::add32_imm(17, 25, 3, false));
        self.buf.emit(a64::tbnz(17, 12, 4));
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 21, 17));
        self.buf.emit(a64::cbnz64(17, 3));
        self.stub_call(StubId::SlowWrite32);
        self.buf.emit(a64::b_rel(2));
        self.buf.emit(a64::str32_reg_uxtw(26, 17, 25));
    }

    fn emit_write32_vfp(&mut self, sn: u8) {
        self.buf.emit(a64::and32_imm(25, 16, 0xFFF).unwrap());
        self.buf.emit(a64::add32_imm(17, 25, 3, false));
        self.buf.emit(a64::tbnz(17, 12, 4));
        self.buf.emit(a64::lsr32_imm(17, 16, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, 21, 17));
        self.buf.emit(a64::cbnz64(17, 4));
        self.buf.emit(a64::fmov_w_from_s(26, sn));
        self.stub_call(StubId::SlowWrite32);
        self.buf.emit(a64::b_rel(2));
        self.buf.emit(a64::str_s_reg_uxtw(sn, 17, 25));
    }

    // -----------------------------------------------------------------------
    // Control Flow Emitters (`call`, `jump`, `return`, `svc`, `interpret`)
    // -----------------------------------------------------------------------

    fn emit_call(&mut self, link: u32, target: u32, callee_thumb: bool) -> bool {
        let caller_thumb = (link & 1) != 0;
        let t_key = target | (callee_thumb as u32);
        let is_direct = self.known_targets.contains(&t_key);

        self.buf.mov_imm32(14, link);
        if !is_direct {
            self.buf.mov_imm32(15, target);
        }
        if callee_thumb != caller_thumb {
            if callee_thumb {
                self.buf.mov_imm32(16, 1);
                self.buf.emit(a64::strb_uoff(16, 19, offsets::CTX_THUMB));
            } else {
                self.buf.emit(a64::strb_uoff(ZR, 19, offsets::CTX_THUMB));
            }
        }
        // Depth check: `if (++depth > 2048)` — note `> 2048`: when `depth` reaches `2049`,
        // `cmp w24, #2048; b.hi` (exact match to `++ctx->depth > RECOMP_DEPTH_LIMIT`!).
        self.buf.emit(a64::add32_imm(24, 24, 1, false));
        self.buf.emit(a64::cmp32_imm(24, RECOMP_DEPTH_LIMIT));
        let p_depth = self.buf.emit(a64::nop());
        self.depth_overflow_patches.push(p_depth);

        if is_direct {
            let at = self.buf.emit(a64::nop());
            self.relocs.push(Reloc { local_word: at as u32, kind: RelocKind::DirectCall(t_key) });
        } else {
            self.stub_call(StubId::IndirectCall);
        }
        self.flags_live = false;

        self.buf.emit(a64::sub32_imm(24, 24, 1, false));
        self.emit_check_unwind();

        // RETURNED(link & !1): verify `w15 == (link & !1)` and `ctx.thumb == caller_thumb`!
        let expected_pc = link & !1;
        self.buf.mov_imm32(16, expected_pc);
        self.buf.emit(a64::sub32_reg(16, 15, 16, Shift::Lsl, 0));
        self.buf.emit(a64::ldrb_uoff(17, 19, offsets::CTX_THUMB));
        if caller_thumb {
            self.buf.emit(a64::eor32_imm(17, 17, 1).unwrap());
        }
        self.buf.emit(a64::orr32_reg(16, 16, 17, Shift::Lsl, 0));
        self.buf.emit(a64::cbz32(16, 2));
        self.stub_jump(StubId::IndirectJump);
        true
    }

    fn emit_jump(&mut self, target: u32, target_thumb: bool) -> bool {
        let same_mode = (target_thumb && self.mode == Mode::Thumb) || (!target_thumb && self.mode == Mode::Arm);
        if same_mode && self.labels.contains(&target) {
            let p = self.buf.emit(a64::nop());
            self.local_patches.push(LocalPatch { at_word: p, target_label: target, cond: None });
            return false;
        }
        if !same_mode {
            if target_thumb {
                self.buf.mov_imm32(16, 1);
                self.buf.emit(a64::strb_uoff(16, 19, offsets::CTX_THUMB));
            } else {
                self.buf.emit(a64::strb_uoff(ZR, 19, offsets::CTX_THUMB));
            }
        }
        let t_key = target | (target_thumb as u32);
        if self.known_targets.contains(&t_key) {
            let at = self.buf.emit(a64::nop());
            self.relocs.push(Reloc { local_word: at as u32, kind: RelocKind::DirectTailJump(t_key) });
        } else {
            self.buf.mov_imm32(15, target);
            self.stub_jump(StubId::IndirectJump);
        }
        false
    }

    /// `RETURN_TO(val_reg)`: updates `ctx.thumb = val & 1`, `w15 = val & (thumb ? ~1 : ~3)`, returns.
    fn emit_return_to_reg(&mut self, val_reg: u8) -> bool {
        if val_reg != 16 {
            self.buf.emit(a64::mov32(16, val_reg));
        }
        let from_thumb = self.mode == Mode::Thumb;
        if !from_thumb {
            // Fast path when bit 0 == 0 (stays ARM):
            self.buf.emit(a64::tbz(16, 0, 5));
            self.buf.mov_imm32(17, 1);
            self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_THUMB));
            self.buf.emit(a64::and32_imm(15, 16, !1u32).unwrap());
            self.emit_epilogue_jump();
            self.buf.emit(a64::and32_imm(15, 16, !3u32).unwrap());
            self.emit_epilogue_jump();
        } else {
            // Fast path when bit 0 == 1 (stays Thumb):
            self.buf.emit(a64::tbnz(16, 0, 4));
            self.buf.emit(a64::strb_uoff(ZR, 19, offsets::CTX_THUMB));
            self.buf.emit(a64::and32_imm(15, 16, !3u32).unwrap());
            self.emit_epilogue_jump();
            self.buf.emit(a64::and32_imm(15, 16, !1u32).unwrap());
            self.emit_epilogue_jump();
        }
        false
    }

    /// `JUMP_TO(val_reg)`: interworking jump through a register (`BX Rm`, `LDR PC, ...`).
    fn emit_jump_to_reg(&mut self, val_reg: u8) -> bool {
        if val_reg != 16 {
            self.buf.emit(a64::mov32(16, val_reg));
        }
        self.buf.emit(a64::and32_imm(17, 16, 1).unwrap());
        self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_THUMB));
        self.buf.emit(a64::and32_imm(15, 16, !3u32).unwrap());
        self.buf.emit(a64::cbz32(17, 2));
        self.buf.emit(a64::and32_imm(15, 16, !1u32).unwrap());
        self.stub_jump(StubId::IndirectJump);
        false
    }

    fn emit_svc(&mut self, next_pc: u32, imm: u32) -> bool {
        self.buf.mov_imm32(15, next_pc);
        self.buf.mov_imm32(16, imm);
        self.buf.emit(a64::str32_uoff(16, 19, offsets::CTX_SVC));
        self.buf.mov_imm32(28, EXIT_SVC);
        self.emit_epilogue_jump();
        false
    }

    fn emit_interpret(&mut self, addr: u32, op: u32) -> bool {
        self.interpreted_addrs.push(addr);
        self.buf.mov_imm32(16, addr);
        self.buf.mov_imm32(26, op);
        self.stub_call(StubId::Interpret);
        self.flags_live = true; // `sync_in` executes `msr nzcv, x23`!
        self.emit_check_unwind();
        true
    }

    /// Updates only `N` and `Z` in `w23` from hardware `NZCV` (preserving `C` and `V`),
    /// or also updates `C` from `carry_reg` (`0` or `1`) when `Some(carry_reg)`.
    fn update_logical_flags(&mut self, carry_reg: Option<u8>) {
        self.buf.emit(a64::mrs_nzcv(16)); // x16[31:30] = N, Z; [29:28] = 0
        match carry_reg {
            None => {
                self.buf.emit(a64::and32_imm(23, 23, 0x3000_0000).unwrap());
                self.buf.emit(a64::orr32_reg(23, 23, 16, Shift::Lsl, 0));
            }
            Some(c_reg) => {
                self.buf.emit(a64::and32_imm(23, 23, 0x1000_0000).unwrap());
                self.buf.emit(a64::orr32_reg(23, 23, 16, Shift::Lsl, 0));
                self.buf.emit(a64::orr32_reg(23, 23, c_reg, Shift::Lsl, 29));
            }
        }
        self.flags_live = false;
    }

    /// Checks if signed 64-bit value in `x_val` overflows `i32`, setting `ctx.q = 1` if so.
    /// Clobbers `w17` and hardware `NZCV`.
    fn emit_accumulate_q(&mut self, rd: u8, x_val: u8) {
        self.buf.emit(a64::cmp64_sxtw(x_val, x_val));
        self.flags_live = false;
        self.buf.emit(a64::b_cond(cond::EQ, 3));
        self.buf.mov_imm32(17, 1);
        self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_Q));
        if rd != x_val {
            self.buf.emit(a64::mov32(rd, x_val));
        }
    }

    /// Saturates signed 64-bit value in `x_val` to `[-(1<<(bits-1)), (1<<(bits-1))-1]` (or
    /// unsigned `[0, (1<<bits)-1]`), setting `ctx.q = 1` on clamp, and writes `rd`.
    fn emit_saturate(&mut self, rd: u8, x_val: u8, bits: u32, unsigned: bool) {
        let (min_v, max_v): (i64, i64) = if unsigned {
            let max = if bits >= 32 { 0xFFFF_FFFF } else { (1i64 << bits) - 1 };
            (0, max)
        } else {
            (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
        };
        // Check > max_v:
        self.buf.mov_imm64(17, max_v as u64);
        self.buf.emit(a64::subs64_reg(ZR, x_val, 17, Shift::Lsl, 0));
        self.flags_live = false;
        let j_le = self.buf.emit(a64::nop());
        self.buf.mov_imm32(25, 1);
        self.buf.emit(a64::strb_uoff(25, 19, offsets::CTX_Q));
        self.buf.emit(a64::mov32(rd, 17));
        let j_done1 = self.buf.emit(a64::nop());

        // Check < min_v:
        let check_min = self.buf.pos();
        self.buf.patch(j_le, a64::b_cond(cond::LE, check_min as i32 - j_le as i32));
        self.buf.mov_imm64(17, min_v as u64);
        self.buf.emit(a64::subs64_reg(ZR, x_val, 17, Shift::Lsl, 0));
        let j_ge = self.buf.emit(a64::nop());
        self.buf.mov_imm32(25, 1);
        self.buf.emit(a64::strb_uoff(25, 19, offsets::CTX_Q));
        self.buf.emit(a64::mov32(rd, 17));
        let j_done2 = self.buf.emit(a64::nop());

        let in_range = self.buf.pos();
        self.buf.patch(j_ge, a64::b_cond(cond::GE, in_range as i32 - j_ge as i32));
        if rd != x_val {
            self.buf.emit(a64::mov32(rd, x_val));
        }
        let done = self.buf.pos();
        self.buf.patch(j_done1, a64::b_rel(done as i32 - j_done1 as i32));
        self.buf.patch(j_done2, a64::b_rel(done as i32 - j_done2 as i32));
    }

    // -----------------------------------------------------------------------
    // ARMv6K Instruction Lowering
    // -----------------------------------------------------------------------

    fn lower_arm(&mut self, a: u32, op: u32) -> bool {
        let condition = (op >> 28) as u8;
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
                self.ensure_hw_flags();
                let p = self.buf.emit(a64::nop());
                self.local_patches.push(LocalPatch { at_word: p, target_label: target, cond: Some(condition) });
                return true;
            }
        }
        self.ensure_hw_flags();
        let skip_patch = self.buf.emit(a64::nop());
        let _ = self.arm_body(a, op);
        let end = self.buf.pos();
        let delta = end as i32 - skip_patch as i32;
        self.buf.patch(skip_patch, a64::b_cond(cond::invert(condition), delta));
        // On skipped path `hw_nzcv == w23` held at `skip_patch`; on executed path it holds iff
        // `self.flags_live` is still true at `end`!
        true
    }

    fn arm_unconditional(&mut self, a: u32, op: u32) -> bool {
        if op & 0xFE00_0000 == 0xFA00_0000 {
            let s24 = (((op & 0x00FF_FFFF) << 8) as i32 >> 8) as u32;
            let offset = (s24 << 2) | (((op >> 24) & 1) << 1);
            return self.emit_call(a + 4, a.wrapping_add(8).wrapping_add(offset) & !1, true);
        }
        if op & 0xFFF0_00F0 == 0xF570_0010 {
            self.buf.emit(a64::strb_uoff(ZR, 19, offsets::CTX_EXCLUSIVE));
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
                    self.buf.emit(a64::ldr32_uoff(rd as u8, 19, offsets::CTX_TLS));
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
                    let rm = self.get_reg(op & 0xF, a + 8, 16);
                    self.buf.emit(a64::clz32(rd as u8, rm));
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
            let d = rd as u8;
            self.buf.mov_imm32(16, 0x1F);
            self.buf.emit(a64::orr32_reg(16, 16, 23, Shift::Lsl, 0));
            self.buf.emit(a64::ldrb_uoff(17, 19, offsets::CTX_Q));
            self.buf.emit(a64::orr32_reg(16, 16, 17, Shift::Lsl, 27));
            self.buf.emit(a64::ldrb_uoff(17, 19, offsets::CTX_GE));
            self.buf.emit(a64::and32_imm(17, 17, 0xF).unwrap());
            self.buf.emit(a64::orr32_reg(d, 16, 17, Shift::Lsl, 16));
            return true;
        }
        let mask = (op >> 16) & 0xF;
        if (mask & 0b0011) != 0 {
            return self.emit_interpret(a, op);
        }
        let val = if (op & (1 << 25)) != 0 {
            let imm = (op & 0xFF).rotate_right(((op >> 8) & 0xF) * 2);
            self.buf.mov_imm32(16, imm);
            16u8
        } else {
            self.get_reg(op & 0xF, a + 8, 16)
        };
        if (mask & 0b1000) != 0 {
            self.buf.emit(a64::and32_imm(23, val, 0xF000_0000).unwrap());
            self.buf.emit(a64::ubfx32(17, val, 27, 1));
            self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_Q));
            self.flags_live = false;
        }
        if (mask & 0b0100) != 0 {
            self.buf.emit(a64::ubfx32(17, val, 16, 4));
            self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_GE));
        }
        true
    }

    /// Evaluates ARM shifter operand into `(b_reg, Option<carry_reg>)`.
    /// When `need_carry` is true, `None` means `sc == old_c` (unchanged); `Some(reg)` holds `0` or `1`.
    fn arm_shifter(&mut self, a: u32, op: u32, need_carry: bool) -> (u8, Option<u8>) {
        if (op & (1 << 25)) != 0 {
            let rotate = ((op >> 8) & 0xF) * 2;
            let value = (op & 0xFF).rotate_right(rotate);
            self.buf.mov_imm32(16, value);
            let sc = if need_carry && rotate != 0 {
                self.buf.mov_imm32(25, value >> 31);
                Some(25)
            } else {
                None
            };
            return (16, sc);
        }
        let rm_idx = op & 0xF;
        let kind = (op >> 5) & 3;
        if (op & (1 << 4)) != 0 {
            // Shift by register `Rs` (`r15` reads as `a + 12`):
            let rs_idx = (op >> 8) & 0xF;
            let rm = self.get_reg(rm_idx, a + 12, 16);
            if rm != 16 {
                self.buf.emit(a64::mov32(16, rm));
            }
            let rs = self.get_reg(rs_idx, a + 12, 17);
            if rs != 17 {
                self.buf.emit(a64::mov32(17, rs));
            }
            let stub = match kind {
                0 => StubId::ShiftLslC,
                1 => StubId::ShiftLsrC,
                2 => StubId::ShiftAsrC,
                _ => StubId::ShiftRorC,
            };
            self.stub_call(stub);
            self.flags_live = false;
            return (16, if need_carry { Some(25) } else { None });
        }

        let amount = ((op >> 7) & 0x1F) as u8;
        let m = self.get_reg(rm_idx, a + 8, 16);
        match (kind, amount) {
            (0, 0) => (m, None),
            (0, n) => {
                if need_carry {
                    self.buf.emit(a64::ubfx32(25, m, 32 - n, 1));
                }
                self.buf.emit(a64::lsl32_imm(16, m, n));
                (16, need_carry.then_some(25))
            }
            (1, 0) => {
                if need_carry {
                    self.buf.emit(a64::lsr32_imm(25, m, 31));
                }
                self.buf.mov_imm32(16, 0);
                (16, need_carry.then_some(25))
            }
            (1, n) => {
                if need_carry {
                    self.buf.emit(a64::ubfx32(25, m, n - 1, 1));
                }
                self.buf.emit(a64::lsr32_imm(16, m, n));
                (16, need_carry.then_some(25))
            }
            (2, 0) => {
                if need_carry {
                    self.buf.emit(a64::lsr32_imm(25, m, 31));
                }
                self.buf.emit(a64::asr32_imm(16, m, 31));
                (16, need_carry.then_some(25))
            }
            (2, n) => {
                if need_carry {
                    self.buf.emit(a64::ubfx32(25, m, n - 1, 1));
                }
                self.buf.emit(a64::asr32_imm(16, m, n));
                (16, need_carry.then_some(25))
            }
            (_, 0) => {
                // RRX: `(old_c << 31) | (m >> 1)`, carry_out = `m & 1`
                if need_carry {
                    self.buf.emit(a64::and32_imm(25, m, 1).unwrap());
                }
                self.buf.emit(a64::lsr32_imm(16, m, 1));
                self.buf.emit(a64::ubfx32(17, 23, 29, 1));
                self.buf.emit(a64::orr32_reg(16, 16, 17, Shift::Lsl, 31));
                (16, need_carry.then_some(25))
            }
            (_, n) => {
                if need_carry {
                    self.buf.emit(a64::ubfx32(25, m, n - 1, 1));
                }
                self.buf.emit(a64::ror32_imm(16, m, n));
                (16, need_carry.then_some(25))
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
        let a_reg = if !matches!(opcode, 13 | 15) {
            self.get_reg(rn, if register_shift { a + 12 } else { a + 8 }, 26)
        } else {
            ZR
        };
        let out_reg: u8 = if compare { ZR } else if rd == 15 { 27 } else { rd as u8 };

        if !set_flags {
            match opcode {
                0 => { self.buf.emit(a64::and32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                1 => { self.buf.emit(a64::eor32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                2 => { self.buf.emit(a64::sub32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                3 => { self.buf.emit(a64::sub32_reg(out_reg, b_reg, a_reg, Shift::Lsl, 0)); }
                4 => { self.buf.emit(a64::add32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                5 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::adc32(out_reg, a_reg, b_reg));
                }
                6 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::sbc32(out_reg, a_reg, b_reg));
                }
                7 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::sbc32(out_reg, b_reg, a_reg));
                }
                12 => { self.buf.emit(a64::orr32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                13 => {
                    if out_reg != b_reg {
                        self.buf.emit(a64::mov32(out_reg, b_reg));
                    }
                }
                14 => { self.buf.emit(a64::bic32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0)); }
                _ => { self.buf.emit(a64::mvn32(out_reg, b_reg)); }
            }
        } else {
            match opcode {
                0 | 8 => {
                    self.buf.emit(a64::ands32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
                1 | 9 => {
                    self.buf.emit(a64::eor32_reg(17, a_reg, b_reg, Shift::Lsl, 0));
                    if !compare {
                        self.buf.emit(a64::mov32(out_reg, 17));
                    }
                    self.buf.emit(a64::ands32_reg(ZR, 17, 17, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
                2 | 10 => {
                    self.buf.emit(a64::subs32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                3 => {
                    self.buf.emit(a64::subs32_reg(out_reg, b_reg, a_reg, Shift::Lsl, 0));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                4 | 11 => {
                    self.buf.emit(a64::adds32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                5 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::adcs32(out_reg, a_reg, b_reg));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                6 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::sbcs32(out_reg, a_reg, b_reg));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                7 => {
                    self.ensure_hw_flags();
                    self.buf.emit(a64::sbcs32(out_reg, b_reg, a_reg));
                    self.buf.emit(a64::mrs_nzcv(23));
                    self.flags_live = true;
                }
                12 => {
                    self.buf.emit(a64::orr32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0));
                    self.buf.emit(a64::ands32_reg(ZR, out_reg, out_reg, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
                13 => {
                    if out_reg != b_reg {
                        self.buf.emit(a64::mov32(out_reg, b_reg));
                    }
                    self.buf.emit(a64::ands32_reg(ZR, out_reg, out_reg, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
                14 => {
                    self.buf.emit(a64::bics32_reg(out_reg, a_reg, b_reg, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
                _ => {
                    self.buf.emit(a64::mvn32(out_reg, b_reg));
                    self.buf.emit(a64::ands32_reg(ZR, out_reg, out_reg, Shift::Lsl, 0));
                    self.update_logical_flags(sc_reg);
                }
            }
        }

        if compare || rd != 15 {
            true
        } else if opcode == 13 && (op & 0x0200_0FFF) == 14 {
            // `mov pc, lr` returns without mode switch:
            self.buf.emit(a64::and32_imm(15, out_reg, !3u32).unwrap());
            self.emit_epilogue_jump();
            false
        } else {
            self.buf.emit(a64::and32_imm(15, out_reg, !3u32).unwrap());
            self.stub_jump(StubId::IndirectJump);
            false
        }
    }

    fn arm_single_transfer(&mut self, a: u32, op: u32) -> bool {
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
                    self.buf.mov_imm32(rd as u8, val as u32);
                    return true;
                }
            } else if lit >= self.text.base && lit.saturating_add(4) <= self.text.end() {
                if let Some(val) = self.text.read32(lit) {
                    self.folded_spans.push((lit, lit + 4));
                    if rd == 15 {
                        let thumb = (val & 1) != 0;
                        return self.emit_jump(val & if thumb { !1 } else { !3 }, thumb);
                    }
                    self.buf.mov_imm32(rd as u8, val);
                    return true;
                }
            }
        }

        // Compute `base` and `oa` (`w27 = oa` if writeback, `w16 = effective address`):
        let base_reg = self.get_reg(rn, a + 8, 25);
        if (op & (1 << 25)) != 0 {
            let m = self.get_reg(op & 0xF, a + 8, 17);
            let kind = (op >> 5) & 3;
            let amt = ((op >> 7) & 0x1F) as u8;
            match (kind, amt) {
                (0, n) => { self.buf.emit(a64::lsl32_imm(17, m, n)); }
                (1, 0) => { self.buf.mov_imm32(17, 0); }
                (1, n) => { self.buf.emit(a64::lsr32_imm(17, m, n)); }
                (2, 0) => { self.buf.emit(a64::asr32_imm(17, m, 31)); }
                (2, n) => { self.buf.emit(a64::asr32_imm(17, m, n)); }
                (_, 0) => {
                    self.buf.emit(a64::lsr32_imm(17, m, 1));
                    self.buf.emit(a64::ubfx32(26, 23, 29, 1));
                    self.buf.emit(a64::orr32_reg(17, 17, 26, Shift::Lsl, 31));
                }
                (_, n) => { self.buf.emit(a64::ror32_imm(17, m, n)); }
            }
            let oa_dst = if writeback { 27 } else { 16 };
            if up {
                self.buf.emit(a64::add32_reg(oa_dst, base_reg, 17, Shift::Lsl, 0));
            } else {
                self.buf.emit(a64::sub32_reg(oa_dst, base_reg, 17, Shift::Lsl, 0));
            }
        } else {
            let imm = op & 0xFFF;
            let delta = if up { imm } else { imm.wrapping_neg() };
            let oa_dst = if writeback { 27 } else { 16 };
            self.buf.add_const32(oa_dst, base_reg, delta, 17);
        }

        if writeback {
            if pre {
                self.buf.emit(a64::mov32(16, 27));
            } else {
                self.buf.emit(a64::mov32(16, base_reg));
            }
        }

        if load {
            let dst = if rd == 15 { 16 } else { rd as u8 };
            if byte {
                self.emit_read8(dst, false);
            } else {
                self.emit_read32_gpr(dst);
            }
            if writeback && rn != rd {
                self.buf.emit(a64::mov32(rn as u8, 27));
            }
            if rd == 15 {
                let pop = rn == 13 && !pre && up && (op & (1 << 25)) == 0 && (op & 0xFFF) == 4;
                return if pop { self.emit_return_to_reg(16) } else { self.emit_jump_to_reg(16) };
            }
        } else {
            let val_reg = self.get_reg(rd, a + 12, 26);
            if byte {
                self.emit_write8(val_reg);
            } else {
                self.emit_write32_gpr(val_reg);
            }
            if writeback {
                self.buf.emit(a64::mov32(rn as u8, 27));
            }
        }
        true
    }

    fn arm_extra_transfer(&mut self, a: u32, op: u32) -> bool {
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

        let base_reg = self.get_reg(rn, a + 8, 25);
        let oa_dst = if writeback || pair { 27 } else { 16 };
        if (op & (1 << 22)) != 0 {
            let imm = ((op >> 4) & 0xF0) | (op & 0xF);
            let delta = if up { imm } else { imm.wrapping_neg() };
            self.buf.add_const32(oa_dst, base_reg, delta, 17);
        } else {
            let m = self.get_reg(op & 0xF, a + 8, 17);
            if up {
                self.buf.emit(a64::add32_reg(oa_dst, base_reg, m, Shift::Lsl, 0));
            } else {
                self.buf.emit(a64::sub32_reg(oa_dst, base_reg, m, Shift::Lsl, 0));
            }
        }

        // For `pair` (`LDRD`/`STRD`), keep effective address in `w27` if `!writeback`, or if `writeback && !pre`,
        // handle `address` vs `oa` cleanly:
        match (kind, load) {
            (1, true) => {
                if writeback {
                    self.buf.emit(a64::mov32(16, if pre { 27 } else { base_reg }));
                }
                self.emit_read16(rd as u8, false);
                if writeback && rn != rd {
                    self.buf.emit(a64::mov32(rn as u8, 27));
                }
            }
            (1, false) => {
                if writeback {
                    self.buf.emit(a64::mov32(16, if pre { 27 } else { base_reg }));
                }
                let v = self.get_reg(rd, a + 8, 26);
                self.emit_write16(v);
                if writeback {
                    self.buf.emit(a64::mov32(rn as u8, 27));
                }
            }
            (2, true) => {
                if writeback {
                    self.buf.emit(a64::mov32(16, if pre { 27 } else { base_reg }));
                }
                self.emit_read8(rd as u8, true);
                if writeback && rn != rd {
                    self.buf.emit(a64::mov32(rn as u8, 27));
                }
            }
            (_, true) => {
                if writeback {
                    self.buf.emit(a64::mov32(16, if pre { 27 } else { base_reg }));
                }
                self.emit_read16(rd as u8, true);
                if writeback && rn != rd {
                    self.buf.emit(a64::mov32(rn as u8, 27));
                }
            }
            (2, false) => {
                // LDRD rd, rd+1:
                // Effective address in w16; save `oa` on stack or compute after if needed:
                // Notice: if `writeback && !pre`, `addr = base_reg`, `oa = w27`. Write `rn = w27`
                // AFTER saving `addr` in `w27`!
                if writeback && !pre {
                    self.buf.emit(a64::mov32(16, base_reg));
                    self.buf.emit(a64::mov32(rn as u8, 27));
                    self.buf.emit(a64::mov32(27, 16));
                } else {
                    if writeback {
                        self.buf.emit(a64::mov32(rn as u8, 27));
                    }
                    self.buf.emit(a64::mov32(16, 27));
                }
                self.emit_read32_gpr(rd as u8);
                self.buf.emit(a64::add32_imm(16, 27, 4, false));
                self.emit_read32_gpr((rd + 1) as u8);
            }
            (_, false) => {
                // STRD rd, rd+1:
                let eff_addr = if writeback && !pre { base_reg } else { 27 };
                // Save `oa` if `writeback && !pre`:
                if writeback && !pre {
                    // Temporarily store first word using `eff_addr`, keeping `oa` in `w27`!
                    self.buf.emit(a64::mov32(16, eff_addr));
                    self.emit_write32_gpr(rd as u8);
                    // Recompute `eff_addr + 4` = `oa - delta + 4`:
                    let imm = if (op & (1 << 22)) != 0 {
                        let raw = ((op >> 4) & 0xF0) | (op & 0xF);
                        if up { raw } else { raw.wrapping_neg() }
                    } else {
                        0 // handled below
                    };
                    if (op & (1 << 22)) != 0 {
                        self.buf.add_const32(16, 27, 4u32.wrapping_sub(imm), 17);
                    } else {
                        let m = self.get_reg(op & 0xF, a + 8, 17);
                        if up {
                            self.buf.emit(a64::sub32_reg(16, 27, m, Shift::Lsl, 0));
                        } else {
                            self.buf.emit(a64::add32_reg(16, 27, m, Shift::Lsl, 0));
                        }
                        self.buf.emit(a64::add32_imm(16, 16, 4, false));
                    }
                    self.emit_write32_gpr((rd + 1) as u8);
                    self.buf.emit(a64::mov32(rn as u8, 27));
                } else {
                    self.buf.emit(a64::mov32(16, 27));
                    self.emit_write32_gpr(rd as u8);
                    self.buf.emit(a64::add32_imm(16, 27, 4, false));
                    self.emit_write32_gpr((rd + 1) as u8);
                    if writeback {
                        self.buf.emit(a64::mov32(rn as u8, 27));
                    }
                }
            }
        }
        true
    }

    /// Emits a fast single-page span transfer with per-word fallback for `LDM`/`STM`/`VPUSH`/`VPOP`.
    /// `w27` must hold starting guest address `p`. `x27` is preserved!
    fn emit_words_span(&mut self, load: bool, regs: &[SpanWord]) {
        if let [single] = regs {
            self.buf.emit(a64::mov32(16, 27));
            match *single {
                SpanWord::Gpr(r) if load => self.emit_read32_gpr(r),
                SpanWord::Gpr(r) => self.emit_write32_gpr(r),
                SpanWord::Const(val) => {
                    self.buf.mov_imm32(26, val);
                    self.emit_write32_gpr(26);
                }
                SpanWord::Vfp(s) if load => self.emit_read32_vfp(s),
                SpanWord::Vfp(s) => self.emit_write32_vfp(s),
            }
            return;
        }

        let bytes = (regs.len() * 4) as u16;
        let pages_reg = if load { 20 } else { 21 };
        self.buf.emit(a64::and32_imm(25, 27, 0xFFF).unwrap());
        self.buf.emit(a64::add32_imm(17, 25, bytes - 1, false));
        let j_slow1 = self.buf.emit(a64::nop()); // tbnz w17, #12, slow
        self.buf.emit(a64::lsr32_imm(17, 27, 12));
        self.buf.emit(a64::ldr64_reg_lsl3(17, pages_reg, 17));
        let j_slow2 = self.buf.emit(a64::nop()); // cbz x17, slow
        self.buf.emit(a64::add64_uxtw(17, 17, 25, 0));

        for (i, item) in regs.iter().enumerate() {
            let off = (i * 4) as u16;
            match *item {
                SpanWord::Gpr(r) if load => { self.buf.emit(a64::ldr32_uoff(r, 17, off)); }
                SpanWord::Gpr(r) => { self.buf.emit(a64::str32_uoff(r, 17, off)); }
                SpanWord::Const(val) => {
                    self.buf.mov_imm32(16, val);
                    self.buf.emit(a64::str32_uoff(16, 17, off));
                }
                SpanWord::Vfp(s) if load => { self.buf.emit(a64::ldr_s_uoff(s, 17, off)); }
                SpanWord::Vfp(s) => { self.buf.emit(a64::str_s_uoff(s, 17, off)); }
            }
        }
        let j_done = self.buf.emit(a64::nop());

        let slow_pos = self.buf.pos();
        self.buf.patch(j_slow1, a64::tbnz(17, 12, slow_pos as i32 - j_slow1 as i32));
        self.buf.patch(j_slow2, a64::cbz64(17, slow_pos as i32 - j_slow2 as i32));
        for (i, item) in regs.iter().enumerate() {
            let off = (i * 4) as u16;
            if off == 0 {
                self.buf.emit(a64::mov32(16, 27));
            } else {
                self.buf.emit(a64::add32_imm(16, 27, off, false));
            }
            match *item {
                SpanWord::Gpr(r) if load => {
                    self.stub_call(StubId::SlowRead32);
                    if r != 26 {
                        self.buf.emit(a64::mov32(r, 26));
                    }
                }
                SpanWord::Gpr(r) => {
                    if r != 26 {
                        self.buf.emit(a64::mov32(26, r));
                    }
                    self.stub_call(StubId::SlowWrite32);
                }
                SpanWord::Const(val) => {
                    self.buf.mov_imm32(26, val);
                    self.stub_call(StubId::SlowWrite32);
                }
                SpanWord::Vfp(s) if load => {
                    self.stub_call(StubId::SlowRead32);
                    self.buf.emit(a64::fmov_s_from_w(s, 26));
                }
                SpanWord::Vfp(s) => {
                    self.buf.emit(a64::fmov_w_from_s(26, s));
                    self.stub_call(StubId::SlowWrite32);
                }
            }
        }
        let done_pos = self.buf.pos();
        self.buf.patch(j_done, a64::b_rel(done_pos as i32 - j_done as i32));
    }

    fn arm_block_transfer(&mut self, a: u32, op: u32) -> bool {
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

        // Put `p` in `w27`!
        self.buf.add_const32(27, rn as u8, p_delta, 16);

        let registers: Vec<u32> = (0..16).filter(|i| (list & (1 << i)) != 0).collect();
        if load {
            if writeback && (list & (1 << rn)) == 0 {
                self.buf.add_const32(rn as u8, rn as u8, fin_delta, 16);
            }
            let words: Vec<SpanWord> = registers
                .iter()
                .map(|&r| if r == 15 { SpanWord::Gpr(15) } else { SpanWord::Gpr(r as u8) })
                .collect();
            self.emit_words_span(true, &words);
            if (list & (1 << 15)) != 0 {
                return if rn == 13 { self.emit_return_to_reg(15) } else { self.emit_jump_to_reg(15) };
            }
        } else {
            // If `rn` is in `list`, writeback value `fin` is needed if `rn` is NOT the lowest register!
            let rn_in_non_lowest = writeback && (list & (1 << rn)) != 0 && (list & ((1 << rn) - 1)) != 0;
            if rn_in_non_lowest {
                // Temporarily put `fin` into `w15` (guest PC is unused until next jump/exit!)
                self.buf.add_const32(15, rn as u8, fin_delta, 16);
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
                if rn_in_non_lowest {
                    self.buf.emit(a64::mov32(rn as u8, 15));
                } else {
                    // Recompute `fin` from `p` (`w27`): `fin - p = fin_delta - p_delta`
                    self.buf.add_const32(rn as u8, 27, fin_delta.wrapping_sub(p_delta), 16);
                }
            }
        }
        true
    }

    fn arm_swap(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let rn = (op >> 16) & 0xF;
        let rm = op & 0xF;
        let byte = (op & (1 << 22)) != 0;
        let addr_r = self.get_reg(rn, a + 8, 27);
        if addr_r != 27 {
            self.buf.emit(a64::mov32(27, addr_r));
        }
        // Save `source` (`rm`) in `w15` in case `rd == rm`:
        let src_r = self.get_reg(rm, a + 8, 15);
        if src_r != 15 {
            self.buf.emit(a64::mov32(15, src_r));
        }
        self.buf.emit(a64::mov32(16, 27));
        if byte {
            self.emit_read8(rd as u8, false);
            self.buf.emit(a64::mov32(16, 27));
            self.emit_write8(15);
        } else {
            self.emit_read32_gpr(rd as u8);
            self.buf.emit(a64::mov32(16, 27));
            self.emit_write32_gpr(15);
        }
        true
    }

    fn arm_exclusive(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        let rm = op & 0xF;
        let rn = (op >> 16) & 0xF;
        let size = (op >> 21) & 3;
        let load = (op & (1 << 20)) != 0;
        if rd == 15 || (size == 0b01 && (if load { rd } else { rm }) >= 14) {
            return self.emit_interpret(a, op);
        }
        let addr_r = self.get_reg(rn, a + 8, 27);
        if addr_r != 27 {
            self.buf.emit(a64::mov32(27, addr_r));
        }
        if load {
            self.buf.mov_imm32(16, 1);
            self.buf.emit(a64::strb_uoff(16, 19, offsets::CTX_EXCLUSIVE));
            self.buf.emit(a64::str32_uoff(27, 19, offsets::CTX_EXCL_ADDR));
            self.buf.emit(a64::mov32(16, 27));
            match size {
                0b00 => self.emit_read32_gpr(rd as u8),
                0b01 => {
                    self.emit_read32_gpr(rd as u8);
                    self.buf.emit(a64::add32_imm(16, 27, 4, false));
                    self.emit_read32_gpr((rd + 1) as u8);
                }
                0b10 => self.emit_read8(rd as u8, false),
                _ => self.emit_read16(rd as u8, false),
            }
        } else {
            self.buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_EXCLUSIVE));
            let j_fail1 = self.buf.emit(a64::nop()); // cbz w16, fail
            self.buf.emit(a64::ldr32_uoff(17, 19, offsets::CTX_EXCL_ADDR));
            self.buf.emit(a64::sub32_reg(17, 17, 27, Shift::Lsl, 0));
            let j_fail2 = self.buf.emit(a64::nop()); // cbnz w17, fail

            self.buf.emit(a64::mov32(16, 27));
            match size {
                0b00 => self.emit_write32_gpr(rm as u8),
                0b01 => {
                    self.emit_write32_gpr(rm as u8);
                    self.buf.emit(a64::add32_imm(16, 27, 4, false));
                    self.emit_write32_gpr((rm + 1) as u8);
                }
                0b10 => self.emit_write8(rm as u8),
                _ => self.emit_write16(rm as u8),
            }
            self.buf.emit(a64::strb_uoff(ZR, 19, offsets::CTX_EXCLUSIVE));
            self.buf.mov_imm32(rd as u8, 0);
            let j_done = self.buf.emit(a64::nop());

            let fail_pos = self.buf.pos();
            self.buf.patch(j_fail1, a64::cbz32(16, fail_pos as i32 - j_fail1 as i32));
            self.buf.patch(j_fail2, a64::cbnz32(17, fail_pos as i32 - j_fail2 as i32));
            self.buf.mov_imm32(rd as u8, 1);

            let done_pos = self.buf.pos();
            self.buf.patch(j_done, a64::b_rel(done_pos as i32 - j_done as i32));
        }
        true
    }

    fn arm_multiply(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 16) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let s = self.get_reg((op >> 8) & 0xF, a + 8, 17);
        if (op & (1 << 21)) != 0 {
            let n = self.get_reg((op >> 12) & 0xF, a + 8, 25);
            self.buf.emit(a64::madd32(rd as u8, m, s, n));
        } else {
            self.buf.emit(a64::mul32(rd as u8, m, s));
        }
        if (op & (1 << 20)) != 0 {
            self.buf.emit(a64::ands32_reg(ZR, rd as u8, rd as u8, Shift::Lsl, 0));
            self.update_logical_flags(None);
        }
        true
    }

    fn arm_multiply_long(&mut self, a: u32, op: u32) -> bool {
        let hi = (op >> 16) & 0xF;
        let lo = (op >> 12) & 0xF;
        if hi == 15 || lo == 15 {
            return self.emit_interpret(a, op);
        }
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let s = self.get_reg((op >> 8) & 0xF, a + 8, 17);
        let kind = (op >> 21) & 7;
        match kind {
            0b010 => {
                // UMAAL
                self.buf.emit(a64::umull(25, m, s));
                self.buf.emit(a64::add64_uxtw(25, 25, lo as u8, 0));
                self.buf.emit(a64::add64_uxtw(25, 25, hi as u8, 0));
            }
            0b100 => {
                self.buf.emit(a64::umull(25, m, s));
            }
            0b101 => {
                self.buf.emit(a64::mov32(26, lo as u8));
                self.buf.emit(a64::bfi64(26, hi as u8, 32, 32));
                self.buf.emit(a64::umaddl(25, m, s, 26));
            }
            0b110 => {
                self.buf.emit(a64::smull(25, m, s));
            }
            _ => {
                self.buf.emit(a64::mov32(26, lo as u8));
                self.buf.emit(a64::bfi64(26, hi as u8, 32, 32));
                self.buf.emit(a64::smaddl(25, m, s, 26));
            }
        }
        self.buf.emit(a64::mov32(lo as u8, 25));
        self.buf.emit(a64::lsr64_imm(hi as u8, 25, 32));
        if kind != 0b010 && (op & (1 << 20)) != 0 {
            // 64-bit N and Z:
            self.buf.emit(a64::subs64_imm(ZR, 25, 0, false));
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
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let s = self.get_reg((op >> 8) & 0xF, a + 8, 17);

        let extract_half = |buf: &mut CodeBuf, dst: u8, src: u8, top: bool| {
            if top {
                buf.emit(a64::asr32_imm(dst, src, 16));
            } else {
                buf.emit(a64::sxth32(dst, src));
            }
        };

        match kind {
            0b00 => {
                // SMLAxy
                let n = self.get_reg(rn, a + 8, 25);
                self.buf.emit(a64::sxtw64(25, n));
                extract_half(&mut self.buf, 16, m, x);
                extract_half(&mut self.buf, 17, s, y);
                self.buf.emit(a64::smaddl(16, 16, 17, 25));
                self.emit_accumulate_q(rd as u8, 16);
            }
            0b01 => {
                // SMLAWy (x=0) / SMULWy (x=1)
                extract_half(&mut self.buf, 17, s, y);
                self.buf.emit(a64::smull(16, m, 17));
                self.buf.emit(a64::asr64_imm(16, 16, 16));
                if x {
                    self.buf.emit(a64::mov32(rd as u8, 16));
                } else {
                    let n = self.get_reg(rn, a + 8, 25);
                    self.buf.emit(a64::sxtw64(25, n));
                    self.buf.emit(a64::add64_reg(16, 16, 25, Shift::Lsl, 0));
                    self.emit_accumulate_q(rd as u8, 16);
                }
            }
            0b10 => {
                // SMLALxy
                extract_half(&mut self.buf, 16, m, x);
                extract_half(&mut self.buf, 17, s, y);
                self.buf.emit(a64::mov32(25, rn as u8));
                self.buf.emit(a64::bfi64(25, rd as u8, 32, 32));
                self.buf.emit(a64::smaddl(25, 16, 17, 25));
                self.buf.emit(a64::mov32(rn as u8, 25));
                self.buf.emit(a64::lsr64_imm(rd as u8, 25, 32));
            }
            _ => {
                // SMULxy
                extract_half(&mut self.buf, 16, m, x);
                extract_half(&mut self.buf, 17, s, y);
                self.buf.emit(a64::mul32(rd as u8, 16, 17));
            }
        }
        true
    }

    fn arm_saturating(&mut self, a: u32, op: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let n = self.get_reg((op >> 16) & 0xF, a + 8, 26);
        let kind = (op >> 21) & 3;
        // Save `m` sign-extended in `x27`:
        self.buf.emit(a64::sxtw64(27, m));
        self.buf.emit(a64::sxtw64(16, n));
        if kind >= 2 {
            // QDADD / QDSUB: first saturate `2 * n` into `w16`, then sign-extend `w16` to `x16`:
            self.buf.emit(a64::lsl64_imm(16, 16, 1));
            self.emit_saturate(16, 16, 32, false);
            self.buf.emit(a64::sxtw64(16, 16));
        }
        if (kind & 1) == 0 {
            self.buf.emit(a64::add64_reg(16, 27, 16, Shift::Lsl, 0));
        } else {
            self.buf.emit(a64::sub64_reg(16, 27, 16, Shift::Lsl, 0));
        }
        self.emit_saturate(rd as u8, 16, 32, false);
        true
    }

    fn arm_branch_exchange(&mut self, a: u32, op: u32) -> bool {
        let rm = op & 0xF;
        if ((op >> 4) & 0xF) == 0b0011 {
            // BLX Rm
            let v = self.get_reg(rm, a + 8, 16);
            if v != 16 {
                self.buf.emit(a64::mov32(16, v));
            }
            self.buf.mov_imm32(14, a + 4);
            self.buf.emit(a64::and32_imm(17, 16, 1).unwrap());
            self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_THUMB));
            self.buf.emit(a64::and32_imm(15, 16, !3u32).unwrap());
            self.buf.emit(a64::cbz32(17, 2));
            self.buf.emit(a64::and32_imm(15, 16, !1u32).unwrap());

            self.buf.emit(a64::add32_imm(24, 24, 1, false));
            self.buf.emit(a64::cmp32_imm(24, RECOMP_DEPTH_LIMIT));
            self.buf.emit(a64::b_cond(cond::LS, 4));
            self.buf.emit(a64::sub32_imm(24, 24, 1, false));
            self.buf.mov_imm32(28, EXIT_UNWIND);
            self.emit_epilogue_jump();

            self.stub_call(StubId::IndirectCall);
            self.flags_live = false;
            self.buf.emit(a64::sub32_imm(24, 24, 1, false));
            self.emit_check_unwind();

            // RETURNED(a + 4) (caller is ARM):
            self.buf.mov_imm32(16, a + 4);
            self.buf.emit(a64::sub32_reg(16, 15, 16, Shift::Lsl, 0));
            self.buf.emit(a64::ldrb_uoff(17, 19, offsets::CTX_THUMB));
            self.buf.emit(a64::orr32_reg(16, 16, 17, Shift::Lsl, 0));
            self.buf.emit(a64::cbz32(16, 2));
            self.stub_jump(StubId::IndirectJump);
            return true;
        }
        if rm == 14 {
            self.emit_return_to_reg(14)
        } else {
            let v = self.get_reg(rm, a + 8, 16);
            self.emit_jump_to_reg(v)
        }
    }

    fn arm_media(&mut self, a: u32, op: u32) -> bool {
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
        let n = self.get_reg((op >> 16) & 0xF, a + 8, 17);
        let m = self.get_reg(op & 0xF, a + 8, 26);

        // Fast inline ASIMD path for non-GE saturating/halving straight-lane ops
        // (`UQSUB8`, `UHADD8`, `UQADD8`, `QADD16`, `QSUB16`, etc. — hot in Moflex decoder!):
        let neon_insn = match (op1 & 7, op2) {
            // Signed saturating (`op1 & 7 == 0b010`):
            (0b010, 0) => Some(a64::sqadd_4h(30, 30, 31)),
            (0b010, 3) => Some(a64::sqsub_4h(30, 30, 31)),
            (0b010, 4) => Some(a64::sqadd_8b(30, 30, 31)),
            (0b010, 7) => Some(a64::sqsub_8b(30, 30, 31)),
            // Signed halving (`op1 & 7 == 0b011`):
            (0b011, 0) => Some(a64::shadd_4h(30, 30, 31)),
            (0b011, 3) => Some(a64::shsub_4h(30, 30, 31)),
            (0b011, 4) => Some(a64::shadd_8b(30, 30, 31)),
            (0b011, 7) => Some(a64::shsub_8b(30, 30, 31)),
            // Unsigned saturating (`op1 & 7 == 0b110`):
            (0b110, 0) => Some(a64::uqadd_4h(30, 30, 31)),
            (0b110, 3) => Some(a64::uqsub_4h(30, 30, 31)),
            (0b110, 4) => Some(a64::uqadd_8b(30, 30, 31)),
            (0b110, 7) => Some(a64::uqsub_8b(30, 30, 31)),
            // Unsigned halving (`op1 & 7 == 0b111`):
            (0b111, 0) => Some(a64::uhadd_4h(30, 30, 31)),
            (0b111, 3) => Some(a64::uhsub_4h(30, 30, 31)),
            (0b111, 4) => Some(a64::uhadd_8b(30, 30, 31)),
            (0b111, 7) => Some(a64::uhsub_8b(30, 30, 31)),
            _ => None,
        };
        if let Some(vec_op) = neon_insn {
            self.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
            self.buf.emit(a64::fmov_s_from_w(30, n));
            self.buf.emit(a64::fmov_s_from_w(31, m));
            self.buf.emit(vec_op);
            self.buf.emit(a64::fmov_w_from_s(rd as u8, 30));
            self.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
            return true;
        }

        // General / GE-setting / cross-lane parallel forms:
        let packed = (op1 << 3) | op2;
        if n != 17 {
            self.buf.emit(a64::mov32(17, n));
        }
        if m != 26 {
            self.buf.emit(a64::mov32(26, m));
        }
        self.buf.mov_imm32(16, packed);
        self.stub_call(StubId::MediaParallel);
        self.flags_live = true;
        if rd as u8 != 26 {
            self.buf.emit(a64::mov32(rd as u8, 26));
        }
        true
    }

    fn arm_pack_saturate_extend(&mut self, a: u32, op: u32, op1: u32, op2: u32) -> bool {
        let rd = (op >> 12) & 0xF;
        if rd == 15 {
            return self.emit_interpret(a, op);
        }
        let d = rd as u8;
        let rn = (op >> 16) & 0xF;
        let shift = ((op >> 7) & 0x1F) as u8;
        let unsigned = (op1 & 0b100) != 0;

        match (op1 & 7, op2) {
            (0b000, 0 | 2 | 4 | 6) if op1 == 0b01000 => {
                // PKHBT / PKHTB
                let n = self.get_reg(rn, a + 8, 16);
                let m = self.get_reg(op & 0xF, a + 8, 17);
                if (op & (1 << 6)) != 0 {
                    let s = if shift == 0 { 31 } else { shift };
                    self.buf.emit(a64::asr32_imm(17, m, s));
                    self.buf.emit(a64::uxth32(17, 17));
                    self.buf.emit(a64::and32_imm(16, n, 0xFFFF_0000).unwrap());
                    self.buf.emit(a64::orr32_reg(d, 16, 17, Shift::Lsl, 0));
                } else {
                    self.buf.emit(a64::uxth32(16, n));
                    self.buf.emit(a64::lsl32_imm(17, m, shift));
                    self.buf.emit(a64::and32_imm(17, 17, 0xFFFF_0000).unwrap());
                    self.buf.emit(a64::orr32_reg(d, 16, 17, Shift::Lsl, 0));
                }
            }
            (_, 0b011) => {
                // SXTB16/UXTB16/SXTB/SXTH/UXTB/UXTH (+ accumulate with Rn if Rn != 15)
                let m = self.get_reg(op & 0xF, a + 8, 16);
                let rotate = (((op >> 10) & 3) * 8) as u8;
                let x = if rotate == 0 {
                    m
                } else {
                    self.buf.emit(a64::ror32_imm(16, m, rotate));
                    16
                };
                let packed = match op1 & 7 {
                    0b000 => {
                        // SXTB16
                        self.buf.emit(a64::sxtb32(17, x));
                        self.buf.emit(a64::uxth32(17, 17));
                        self.buf.emit(a64::lsr32_imm(25, x, 16));
                        self.buf.emit(a64::sxtb32(25, 25));
                        self.buf.emit(a64::orr32_reg(16, 17, 25, Shift::Lsl, 16));
                        true
                    }
                    0b010 => {
                        self.buf.emit(a64::sxtb32(16, x));
                        false
                    }
                    0b011 => {
                        self.buf.emit(a64::sxth32(16, x));
                        false
                    }
                    0b100 => {
                        self.buf.emit(a64::and32_imm(16, x, 0x00FF_00FF).unwrap());
                        true
                    }
                    0b110 => {
                        self.buf.emit(a64::uxtb32(16, x));
                        false
                    }
                    0b111 => {
                        self.buf.emit(a64::uxth32(16, x));
                        false
                    }
                    _ => return self.emit_interpret(a, op),
                };
                match (rn, packed) {
                    (15, _) => {
                        if d != 16 {
                            self.buf.emit(a64::mov32(d, 16));
                        }
                    }
                    (_, true) => {
                        // Halfword-wise wrapping add `(rn + ext) & 0xFFFF | (((rn >> 16) + (ext >> 16)) << 16)`
                        self.buf.emit(a64::add32_reg(17, rn as u8, 16, Shift::Lsl, 0));
                        self.buf.emit(a64::uxth32(17, 17));
                        self.buf.emit(a64::lsr32_imm(25, rn as u8, 16));
                        self.buf.emit(a64::lsr32_imm(26, 16, 16));
                        self.buf.emit(a64::add32_reg(25, 25, 26, Shift::Lsl, 0));
                        self.buf.emit(a64::orr32_reg(d, 17, 25, Shift::Lsl, 16));
                    }
                    (_, false) => {
                        self.buf.emit(a64::add32_reg(d, rn as u8, 16, Shift::Lsl, 0));
                    }
                }
            }
            (0b000, 0b101) if op1 == 0b01000 => {
                // SEL Rd, Rn, Rm (9 branchless scalar insns!)
                let n = self.get_reg(rn, a + 8, 25);
                let m = self.get_reg(op & 0xF, a + 8, 26);
                self.buf.emit(a64::ldrb_uoff(16, 19, offsets::CTX_GE));
                self.buf.mov_imm32(17, 0x0020_4081);
                self.buf.emit(a64::mul32(16, 16, 17));
                self.buf.emit(a64::and32_imm(16, 16, 0x0101_0101).unwrap());
                self.buf.emit(a64::lsl32_imm(17, 16, 8));
                self.buf.emit(a64::sub32_reg(17, 17, 16, Shift::Lsl, 0)); // w17 = byte mask
                self.buf.emit(a64::and32_reg(16, n, 17, Shift::Lsl, 0));
                self.buf.emit(a64::bic32_reg(17, m, 17, Shift::Lsl, 0));
                self.buf.emit(a64::orr32_reg(d, 16, 17, Shift::Lsl, 0));
            }
            (0b011, 0b001) if op1 == 0b01011 => {
                let m = self.get_reg(op & 0xF, a + 8, 16);
                self.buf.emit(a64::rev32(d, m));
            }
            (0b011, 0b101) if op1 == 0b01011 => {
                let m = self.get_reg(op & 0xF, a + 8, 16);
                self.buf.emit(a64::rev16_32(d, m));
            }
            (0b111, 0b001) if op1 == 0b01111 => {
                let m = self.get_reg(op & 0xF, a + 8, 16);
                self.buf.emit(a64::rbit32(d, m));
            }
            (0b111, 0b101) if op1 == 0b01111 => {
                let m = self.get_reg(op & 0xF, a + 8, 16);
                self.buf.emit(a64::rev16_32(16, m));
                self.buf.emit(a64::sxth32(d, 16));
            }
            (0b010 | 0b110, 0b001) => {
                // SSAT16 / USAT16
                let bits = ((op >> 16) & 0xF) + (!unsigned as u32);
                let m = self.get_reg(op & 0xF, a + 8, 27);
                if m != 27 {
                    self.buf.emit(a64::mov32(27, m));
                }
                self.buf.emit(a64::sxth64(16, 27));
                self.emit_saturate(26, 16, bits, unsigned);
                self.buf.emit(a64::uxth32(26, 26));
                self.buf.emit(a64::asr32_imm(16, 27, 16));
                self.buf.emit(a64::sxtw64(16, 16));
                self.emit_saturate(16, 16, bits, unsigned);
                self.buf.emit(a64::orr32_reg(d, 26, 16, Shift::Lsl, 16));
            }
            (_, o) if (o & 1) == 0 && matches!((op1 >> 2) & 0b111, 0b010 | 0b011) => {
                // SSAT / USAT
                let bits = ((op >> 16) & 0x1F) + (!unsigned as u32);
                let m = self.get_reg(op & 0xF, a + 8, 16);
                if (op & (1 << 6)) != 0 {
                    let s = if shift == 0 { 31 } else { shift };
                    self.buf.emit(a64::asr32_imm(16, m, s));
                } else {
                    self.buf.emit(a64::lsl32_imm(16, m, shift));
                }
                self.buf.emit(a64::sxtw64(16, 16));
                self.emit_saturate(d, 16, bits, unsigned);
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
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let s = self.get_reg((op >> 8) & 0xF, a + 8, 17);
        let bit5 = (op & (1 << 5)) != 0;
        let subtract = (op & (1 << 6)) != 0;

        match op1 {
            0b10000 | 0b10100 => {
                // SMLAD / SMUAD / SMLSD / SMUSD / SMLALD / SMLSLD
                let b = if bit5 {
                    self.buf.emit(a64::ror32_imm(17, s, 16));
                    17
                } else {
                    s
                };
                // p0 = (i16)m * (i16)b in x25:
                self.buf.emit(a64::sxth32(25, m));
                self.buf.emit(a64::sxth32(26, b));
                self.buf.emit(a64::smull(25, 25, 26));
                // p1 = (i16)(m >> 16) * (i16)(b >> 16):
                self.buf.emit(a64::asr32_imm(26, m, 16));
                self.buf.emit(a64::asr32_imm(17, b, 16));
                if subtract {
                    self.buf.emit(a64::smsubl(25, 26, 17, 25));
                } else {
                    self.buf.emit(a64::smaddl(25, 26, 17, 25));
                }
                if op1 == 0b10000 {
                    if ra != 15 {
                        self.buf.emit(a64::sxtw64(16, ra as u8));
                        self.buf.emit(a64::add64_reg(25, 25, 16, Shift::Lsl, 0));
                    }
                    self.emit_accumulate_q(rd as u8, 25);
                } else {
                    self.buf.emit(a64::mov32(16, ra as u8));
                    self.buf.emit(a64::bfi64(16, rd as u8, 32, 32));
                    self.buf.emit(a64::add64_reg(25, 25, 16, Shift::Lsl, 0));
                    self.buf.emit(a64::mov32(ra as u8, 25));
                    self.buf.emit(a64::lsr64_imm(rd as u8, 25, 32));
                }
            }
            0b10101 => {
                // SMMUL(R) / SMMLA(R) / SMMLS(R)
                self.buf.emit(a64::smull(25, m, s));
                if ra != 15 {
                    self.buf.emit(a64::lsl64_imm(26, ra as u8, 32));
                    if subtract {
                        self.buf.emit(a64::sub64_reg(25, 26, 25, Shift::Lsl, 0));
                    } else {
                        self.buf.emit(a64::add64_reg(25, 26, 25, Shift::Lsl, 0));
                    }
                } else if subtract {
                    self.buf.emit(a64::sub64_reg(25, ZR, 25, Shift::Lsl, 0));
                }
                if bit5 {
                    self.buf.mov_imm32(16, 0x8000_0000);
                    self.buf.emit(a64::add64_uxtw(25, 25, 16, 0));
                }
                self.buf.emit(a64::lsr64_imm(rd as u8, 25, 32));
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
        let m = self.get_reg(op & 0xF, a + 8, 16);
        let s = self.get_reg((op >> 8) & 0xF, a + 8, 17);
        self.buf.mov_imm32(25, 0);
        for i in 0..4u8 {
            self.buf.emit(a64::ubfx32(26, m, i * 8, 8));
            self.buf.emit(a64::ubfx32(27, s, i * 8, 8));
            self.buf.emit(a64::subs32_reg(26, 26, 27, Shift::Lsl, 0));
            // abs(w26): if MI, negate!
            self.buf.emit(a64::b_cond(cond::PL, 2));
            self.buf.emit(a64::sub32_reg(26, ZR, 26, Shift::Lsl, 0));
            self.buf.emit(a64::add32_reg(25, 25, 26, Shift::Lsl, 0));
        }
        self.flags_live = false;
        if ra == 15 {
            self.buf.emit(a64::mov32(rd as u8, 25));
        } else {
            self.buf.emit(a64::add32_reg(rd as u8, 25, ra as u8, Shift::Lsl, 0));
        }
        true
    }

    // -----------------------------------------------------------------------
    // VFPv2 Lowering
    // -----------------------------------------------------------------------

    #[inline(always)]
    fn sreg(vd: u32, extra: bool) -> u8 {
        (((vd << 1) | (extra as u32)) & 31) as u8
    }

    #[inline(always)]
    fn dreg(vd: u32, extra: bool) -> u8 {
        ((((extra as u32) << 4) | vd) & 15) as u8
    }

    /// Loads guest `d_reg` (`{s[2*d_reg], s[2*d_reg+1]}`) into `d_dst` (`30` or `31`).
    /// Note: if `s_{2*d_reg}` or `s_{2*d_reg+1}` is `30` or `31` (`d_reg == 15`), reads from
    /// `[x19, #SCRATCH_FP]` where `s30, s31` were saved!
    fn emit_load_vfp_d(&mut self, d_dst: u8, d_src: u8) {
        let lo_s = (d_src * 2) & 31;
        let hi_s = (d_src * 2 + 1) & 31;
        if lo_s == 30 {
            self.buf.emit(a64::ldp32_off(16, 17, 19, offsets::SCRATCH_FP));
        } else {
            self.buf.emit(a64::fmov_w_from_s(16, lo_s));
            self.buf.emit(a64::fmov_w_from_s(17, hi_s));
        }
        self.buf.emit(a64::bfi64(16, 17, 32, 32));
        self.buf.emit(a64::fmov_d_from_x(d_dst, 16));
    }

    /// Writes `d_src` (`30` or `31`) back into guest `d_dst` (`{s[2*d_dst], s[2*d_dst+1]}`),
    /// updating `[x19, #SCRATCH_FP]` if `d_dst == 15` (`s30, s31`) so the subsequent restore loads it!
    fn emit_store_vfp_d(&mut self, d_dst: u8, d_src: u8) {
        let lo_s = (d_dst * 2) & 31;
        let hi_s = (d_dst * 2 + 1) & 31;
        self.buf.emit(a64::fmov_x_from_d(16, d_src));
        if lo_s == 30 {
            self.buf.emit(a64::lsr64_imm(17, 16, 32));
            self.buf.emit(a64::stp32_off(16, 17, 19, offsets::SCRATCH_FP));
        } else {
            self.buf.emit(a64::fmov_s_from_w(lo_s, 16));
            self.buf.emit(a64::lsr64_imm(16, 16, 32));
            self.buf.emit(a64::fmov_s_from_w(hi_s, 16));
        }
    }

    fn vfp_maybe_vector<F>(&mut self, double: bool, op_code: u32, rd: u8, rn: u8, rm: u8, scalar_body: F) -> bool
    where
        F: FnOnce(&mut Self),
    {
        let bank = if double { 4 } else { 8 };
        if !self.options.emit_vfp_vectors || rd < bank {
            scalar_body(self);
            return true;
        }
        // Check `(*ctx.fpscr & (7 << 16)) != 0`:
        self.buf.emit(a64::ldr64_uoff(16, 19, offsets::CTX_FPSCR));
        self.buf.emit(a64::ldr32_uoff(16, 16, 0));
        self.buf.emit(a64::ubfx32(16, 16, 16, 3));
        let j_vec = self.buf.emit(a64::nop());
        scalar_body(self);
        let j_done = self.buf.emit(a64::nop());

        let vec_pos = self.buf.pos();
        self.buf.patch(j_vec, a64::cbnz32(16, vec_pos as i32 - j_vec as i32));
        let packed = ((double as u32) << 20)
            | ((op_code & 0xF) << 16)
            | (((rd as u32) & 31) << 10)
            | (((rn as u32) & 31) << 5)
            | ((rm as u32) & 31);
        self.buf.mov_imm32(16, packed);
        self.stub_call(StubId::VfpVector);
        self.flags_live = false;

        let done_pos = self.buf.pos();
        self.buf.patch(j_done, a64::b_rel(done_pos as i32 - j_done as i32));
        true
    }

    fn vfp_data_processing(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        if coproc != 10 && coproc != 11 {
            return self.emit_interpret(a, op);
        }
        let double = coproc == 11;
        let (vd, vn, vm) = ((op >> 12) & 0xF, (op >> 16) & 0xF, op & 0xF);
        let (d_bit, n_bit, m_bit) = ((op & (1 << 22)) != 0, (op & (1 << 7)) != 0, (op & (1 << 5)) != 0);
        let (rd, rn, rm) = if double {
            (Self::dreg(vd, d_bit), Self::dreg(vn, n_bit), Self::dreg(vm, m_bit))
        } else {
            (Self::sreg(vd, d_bit), Self::sreg(vn, n_bit), Self::sreg(vm, m_bit))
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
            (0b1011, true) => return self.vfp_extension(a, op, double, rd, rm),
            _ => return self.emit_interpret(a, op),
        };

        self.vfp_maybe_vector(double, vop, rd, rn, rm, |em| {
            if !double {
                match vop {
                    4 => { em.buf.emit(a64::fmul_s(rd, rn, rm)); }
                    5 => { em.buf.emit(a64::fnmul_s(rd, rn, rm)); }
                    6 => { em.buf.emit(a64::fadd_s(rd, rn, rm)); }
                    7 => { em.buf.emit(a64::fsub_s(rd, rn, rm)); }
                    8 => { em.buf.emit(a64::fdiv_s(rd, rn, rm)); }
                    0..=3 => {
                        // Chained VFPv2 multiply-accumulate (rounds product FIRST, then adds/subs!).
                        // Pick scratch `s31` if `rd != 31`, else `s30`!
                        let tmp = if rd != 31 { 31 } else { 30 };
                        em.buf.emit(a64::str_s_uoff(tmp, 19, offsets::SCRATCH_FP as u16));
                        if vop == 3 {
                            em.buf.emit(a64::fnmul_s(tmp, rn, rm)); // tmp = -(a * b)
                            em.buf.emit(a64::fsub_s(rd, tmp, rd));  // rd  = -(a * b) - acc
                        } else {
                            em.buf.emit(a64::fmul_s(tmp, rn, rm));  // tmp = a * b
                            match vop {
                                0 => { em.buf.emit(a64::fadd_s(rd, rd, tmp)); }
                                1 => { em.buf.emit(a64::fsub_s(rd, rd, tmp)); }
                                _ => { em.buf.emit(a64::fsub_s(rd, tmp, rd)); } // (a * b) - acc
                            }
                        }
                        em.buf.emit(a64::ldr_s_uoff(tmp, 19, offsets::SCRATCH_FP as u16));
                    }
                    _ => unreachable!(),
                }
            } else {
                em.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                em.emit_load_vfp_d(30, rn);
                em.emit_load_vfp_d(31, rm);
                match vop {
                    4 => { em.buf.emit(a64::fmul_d(30, 30, 31)); }
                    5 => { em.buf.emit(a64::fnmul_d(30, 30, 31)); }
                    6 => { em.buf.emit(a64::fadd_d(30, 30, 31)); }
                    7 => { em.buf.emit(a64::fsub_d(30, 30, 31)); }
                    8 => { em.buf.emit(a64::fdiv_d(30, 30, 31)); }
                    0..=3 => {
                        if vop == 3 {
                            em.buf.emit(a64::fnmul_d(30, 30, 31));
                            em.emit_load_vfp_d(31, rd);
                            em.buf.emit(a64::fsub_d(30, 30, 31));
                        } else {
                            em.buf.emit(a64::fmul_d(30, 30, 31));
                            em.emit_load_vfp_d(31, rd);
                            match vop {
                                0 => { em.buf.emit(a64::fadd_d(30, 31, 30)); }
                                1 => { em.buf.emit(a64::fsub_d(30, 31, 30)); }
                                _ => { em.buf.emit(a64::fsub_d(30, 30, 31)); }
                            }
                        }
                    }
                    _ => unreachable!(),
                }
                em.emit_store_vfp_d(rd, 30);
                em.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
            }
        })
    }

    fn vfp_extension(&mut self, a: u32, op: u32, double: bool, rd: u8, rm: u8) -> bool {
        let top = (op & (1 << 7)) != 0;
        match ((op >> 16) & 0xF, top) {
            (0b0000 | 0b0001, _) => {
                let second = ((op >> 16) & 1) != 0;
                let vop = 9 + (((second as u32) << 1) | (top as u32));
                self.vfp_maybe_vector(double, vop, rd, rd, rm, |em| {
                    if !double {
                        match (second, top) {
                            (false, false) => {
                                if rd != rm {
                                    em.buf.emit(a64::fmov_s(rd, rm));
                                }
                            }
                            (false, true) => { em.buf.emit(a64::fabs_s(rd, rm)); }
                            (true, false) => { em.buf.emit(a64::fneg_s(rd, rm)); }
                            (true, true) => { em.buf.emit(a64::fsqrt_s(rd, rm)); }
                        }
                    } else if second && top {
                        em.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                        em.emit_load_vfp_d(30, rm);
                        em.buf.emit(a64::fsqrt_d(30, 30));
                        em.emit_store_vfp_d(rd, 30);
                        em.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                    } else {
                        // Raw bit VMOV/VABS/VNEG on `{s[2*m], s[2*m+1]}` without flushing subnormals!
                        let (dl, dh) = ((rd * 2) & 31, (rd * 2 + 1) & 31);
                        let (ml, mh) = ((rm * 2) & 31, (rm * 2 + 1) & 31);
                        em.buf.emit(a64::fmov_w_from_s(16, ml));
                        em.buf.emit(a64::fmov_w_from_s(17, mh));
                        match (second, top) {
                            (false, false) => {}
                            (false, true) => { em.buf.emit(a64::and32_imm(17, 17, 0x7FFF_FFFF).unwrap()); }
                            (true, false) => { em.buf.emit(a64::eor32_imm(17, 17, 0x8000_0000).unwrap()); }
                            _ => unreachable!(),
                        }
                        em.buf.emit(a64::fmov_s_from_w(dl, 16));
                        em.buf.emit(a64::fmov_s_from_w(dh, 17));
                    }
                })
            }
            (0b0100 | 0b0101, _) => {
                let cmp_zero = ((op >> 16) & 0xF) == 0b0101;
                if !double {
                    if cmp_zero {
                        self.buf.emit(a64::fcmp_s_zero(rd));
                    } else {
                        self.buf.emit(a64::fcmp_s(rd, rm));
                    }
                } else {
                    self.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                    self.emit_load_vfp_d(30, rd);
                    if cmp_zero {
                        self.buf.emit(a64::fcmp_d_zero(30));
                    } else {
                        self.emit_load_vfp_d(31, rm);
                        self.buf.emit(a64::fcmp_d(30, 31));
                    }
                    self.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                }
                self.buf.emit(a64::mrs_nzcv(16));
                self.flags_live = false;
                self.buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_FPSCR));
                self.buf.emit(a64::ldr32_uoff(25, 17, 0));
                self.buf.emit(a64::and32_imm(25, 25, 0x0FFF_FFFF).unwrap());
                self.buf.emit(a64::orr32_reg(25, 25, 16, Shift::Lsl, 0));
                self.buf.emit(a64::str32_uoff(25, 17, 0));
                true
            }
            (0b0111, true) => {
                // VCVT.F32.F64 / VCVT.F64.F32
                let (vd, vm) = ((op >> 12) & 0xF, op & 0xF);
                let (d_bit, m_bit) = ((op & (1 << 22)) != 0, (op & (1 << 5)) != 0);
                self.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                if double {
                    let sd = Self::sreg(vd, d_bit);
                    self.emit_load_vfp_d(30, rm);
                    if sd >= 30 {
                        self.buf.emit(a64::fcvt_s_from_d(30, 30));
                        let slot = offsets::SCRATCH_FP as u16 + ((sd - 30) as u16) * 4;
                        self.buf.emit(a64::str_s_uoff(30, 19, slot));
                    } else {
                        self.buf.emit(a64::fcvt_s_from_d(sd, 30));
                    }
                } else {
                    let sm = Self::sreg(vm, m_bit);
                    let dd = Self::dreg(vd, d_bit);
                    if sm >= 30 {
                        let slot = offsets::SCRATCH_FP as u16 + ((sm - 30) as u16) * 4;
                        self.buf.emit(a64::ldr_s_uoff(30, 19, slot));
                        self.buf.emit(a64::fcvt_d_from_s(30, 30));
                    } else {
                        self.buf.emit(a64::fcvt_d_from_s(30, sm));
                    }
                    self.emit_store_vfp_d(dd, 30);
                }
                self.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                true
            }
            (0b1000, _) => {
                // VCVT from integer (`top` == signed)
                let src = Self::sreg(op & 0xF, (op & (1 << 5)) != 0);
                if !double {
                    if top {
                        self.buf.emit(a64::scvtf_s_asimd(rd, src));
                    } else {
                        self.buf.emit(a64::ucvtf_s_asimd(rd, src));
                    }
                } else {
                    self.buf.emit(a64::fmov_w_from_s(16, src));
                    self.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                    if top {
                        self.buf.emit(a64::scvtf_d_from_w(30, 16));
                    } else {
                        self.buf.emit(a64::ucvtf_d_from_w(30, 16));
                    }
                    self.emit_store_vfp_d(rd, 30);
                    self.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                }
                true
            }
            (0b1100 | 0b1101, true) => {
                // VCVT to integer rounding toward zero
                let signed = ((op >> 16) & 1) != 0;
                let dst = Self::sreg((op >> 12) & 0xF, (op & (1 << 22)) != 0);
                if !double {
                    if signed {
                        self.buf.emit(a64::fcvtzs_s_asimd(dst, rm));
                    } else {
                        self.buf.emit(a64::fcvtzu_s_asimd(dst, rm));
                    }
                } else {
                    self.buf.emit(a64::stp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                    self.emit_load_vfp_d(30, rm);
                    if signed {
                        self.buf.emit(a64::fcvtzs_w_from_d(16, 30));
                    } else {
                        self.buf.emit(a64::fcvtzu_w_from_d(16, 30));
                    }
                    self.buf.emit(a64::ldp_s_off(30, 31, 19, offsets::SCRATCH_FP));
                    self.buf.emit(a64::fmov_s_from_w(dst, 16));
                }
                true
            }
            _ => self.emit_interpret(a, op),
        }
    }

    fn vfp_register_transfer(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        let load = (op & (1 << 20)) != 0;
        let rt = (op >> 12) & 0xF;
        match (coproc, (op >> 21) & 7) {
            (10 | 11, 7) if ((op >> 16) & 0xF) == 1 => {
                if !load {
                    // VMSR FPSCR, Rt — also sync hardware FPCR [25:22]!
                    let v = self.get_reg(rt, a + 8, 16);
                    self.buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_FPSCR));
                    self.buf.emit(a64::str32_uoff(v, 17, 0));
                    self.buf.emit(a64::mrs_fpcr(17));
                    self.buf.mov_imm32(25, 0x03C0_0000);
                    self.buf.emit(a64::bic64_reg(17, 17, 25, Shift::Lsl, 0));
                    self.buf.emit(a64::and32_reg(25, v, 25, Shift::Lsl, 0));
                    self.buf.emit(a64::orr64_reg(17, 17, 25, Shift::Lsl, 0));
                    self.buf.emit(a64::msr_fpcr(17));
                } else if rt == 15 {
                    // VMRS APSR_nzcv, FPSCR (`FMSTAT`)
                    self.buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_FPSCR));
                    self.buf.emit(a64::ldr32_uoff(16, 17, 0));
                    self.buf.emit(a64::and32_imm(23, 16, 0xF000_0000).unwrap());
                    self.buf.emit(a64::msr_nzcv(23));
                    self.flags_live = true;
                } else {
                    self.buf.emit(a64::ldr64_uoff(17, 19, offsets::CTX_FPSCR));
                    self.buf.emit(a64::ldr32_uoff(rt as u8, 17, 0));
                }
                true
            }
            (10 | 11, 0) if rt != 15 => {
                let sn = Self::sreg((op >> 16) & 0xF, (op & (1 << 7)) != 0);
                if load {
                    self.buf.emit(a64::fmov_w_from_s(rt as u8, sn));
                } else {
                    self.buf.emit(a64::fmov_s_from_w(sn, rt as u8));
                }
                true
            }
            _ => self.emit_interpret(a, op),
        }
    }

    fn vfp_load_store(&mut self, a: u32, op: u32) -> bool {
        let coproc = (op >> 8) & 0xF;
        if coproc != 10 && coproc != 11 {
            return self.emit_interpret(a, op);
        }
        let double = coproc == 11;
        if ((op >> 21) & 0x7F) == 0b110_0010 {
            return self.vfp_move_pair(a, op, double);
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
        let first = if double { Self::dreg(vd, d_bit) as u32 } else { Self::sreg(vd, d_bit) as u32 };
        let count = if pre && !writeback {
            1
        } else if double {
            (op & 0xFF) / 2
        } else {
            op & 0xFF
        };

        // PC-relative `.text` float/double literal pool folding (`VLDR Sd/Dd, [PC, ±#imm]`):
        if self.options.fold_text_literals && load && rn == 15 && pre && !writeback {
            let lit = if up { (a + 8).wrapping_add(offset) } else { (a + 8).wrapping_sub(offset) };
            let bytes = if double { 8u32 } else { 4u32 };
            if lit >= self.text.base && lit.saturating_add(bytes) <= self.text.end() {
                if !double {
                    if let Some(bits) = self.text.read32(lit) {
                        self.folded_spans.push((lit, lit + 4));
                        let sd = first as u8;
                        if bits == 0 {
                            self.buf.emit(a64::fmov_s_from_w(sd, ZR));
                        } else {
                            self.buf.mov_imm32(16, bits);
                            self.buf.emit(a64::fmov_s_from_w(sd, 16));
                        }
                        return true;
                    }
                } else if let (Some(lo), Some(hi)) = (self.text.read32(lit), self.text.read32(lit + 4)) {
                    self.folded_spans.push((lit, lit + 8));
                    let s_lo = ((first * 2) & 31) as u8;
                    let s_hi = ((first * 2 + 1) & 31) as u8;
                    if lo == 0 {
                        self.buf.emit(a64::fmov_s_from_w(s_lo, ZR));
                    } else {
                        self.buf.mov_imm32(16, lo);
                        self.buf.emit(a64::fmov_s_from_w(s_lo, 16));
                    }
                    if hi == 0 {
                        self.buf.emit(a64::fmov_s_from_w(s_hi, ZR));
                    } else {
                        self.buf.mov_imm32(16, hi);
                        self.buf.emit(a64::fmov_s_from_w(s_hi, 16));
                    }
                    return true;
                }
            }
        }

        let base_reg = self.get_reg(rn, a + 8, 25);
        let p_delta = match (up, pre && !writeback) {
            (true, true) => offset,
            (true, false) => 0,
            (false, _) => offset.wrapping_neg(),
        };
        self.buf.add_const32(27, base_reg, p_delta, 16);

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
        self.emit_words_span(load, &words);

        if writeback {
            let fin_delta = if up { offset } else { offset.wrapping_neg() };
            self.buf.add_const32(rn as u8, 27, fin_delta.wrapping_sub(p_delta), 16);
        }
        true
    }

    fn vfp_move_pair(&mut self, a: u32, op: u32, double: bool) -> bool {
        let load = (op & (1 << 20)) != 0;
        let rt = (op >> 12) & 0xF;
        let rt2 = (op >> 16) & 0xF;
        if rt == 15 || rt2 == 15 {
            return self.emit_interpret(a, op);
        }
        let (vm, m_bit) = (op & 0xF, (op & (1 << 5)) != 0);
        let (low, high) = if double {
            let dm = Self::dreg(vm, m_bit);
            ((dm * 2) & 31, (dm * 2 + 1) & 31)
        } else {
            let sm = Self::sreg(vm, m_bit);
            (sm, (sm + 1) & 31)
        };
        if load {
            self.buf.emit(a64::fmov_w_from_s(rt as u8, low));
            self.buf.emit(a64::fmov_w_from_s(rt2 as u8, high));
        } else {
            self.buf.emit(a64::fmov_s_from_w(low, rt as u8));
            self.buf.emit(a64::fmov_s_from_w(high, rt2 as u8));
        }
        true
    }

    // -----------------------------------------------------------------------
    // Thumb-1 Instruction Lowering
    // -----------------------------------------------------------------------

    fn lower_thumb(&mut self, a: u32, op: u32, second: u32) -> bool {
        match op >> 13 {
            0b000 if ((op >> 11) & 3) == 0b11 => {
                // ADD / SUB 3-reg or 3-bit imm
                let field = ((op >> 6) & 7) as u8;
                let rn = ((op >> 3) & 7) as u8;
                let rd = (op & 7) as u8;
                let sub = (op & (1 << 9)) != 0;
                if (op & (1 << 10)) != 0 {
                    if sub {
                        self.buf.emit(a64::subs32_imm(rd, rn, field as u16, false));
                    } else {
                        self.buf.emit(a64::adds32_imm(rd, rn, field as u16, false));
                    }
                } else if sub {
                    self.buf.emit(a64::subs32_reg(rd, rn, field, Shift::Lsl, 0));
                } else {
                    self.buf.emit(a64::adds32_reg(rd, rn, field, Shift::Lsl, 0));
                }
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
                true
            }
            0b000 => {
                // LSL / LSR / ASR imm5
                let amount = ((op >> 6) & 0x1F) as u8;
                let rm = ((op >> 3) & 7) as u8;
                let rd = (op & 7) as u8;
                let carry = match ((op >> 11) & 3, amount) {
                    (0, 0) => {
                        if rd != rm {
                            self.buf.emit(a64::mov32(rd, rm));
                        }
                        None
                    }
                    (0, n) => {
                        self.buf.emit(a64::ubfx32(25, rm, 32 - n, 1));
                        self.buf.emit(a64::lsl32_imm(rd, rm, n));
                        Some(25)
                    }
                    (1, 0) => {
                        self.buf.emit(a64::lsr32_imm(25, rm, 31));
                        self.buf.mov_imm32(rd, 0);
                        Some(25)
                    }
                    (1, n) => {
                        self.buf.emit(a64::ubfx32(25, rm, n - 1, 1));
                        self.buf.emit(a64::lsr32_imm(rd, rm, n));
                        Some(25)
                    }
                    (_, 0) => {
                        self.buf.emit(a64::lsr32_imm(25, rm, 31));
                        self.buf.emit(a64::asr32_imm(rd, rm, 31));
                        Some(25)
                    }
                    (_, n) => {
                        self.buf.emit(a64::ubfx32(25, rm, n - 1, 1));
                        self.buf.emit(a64::asr32_imm(rd, rm, n));
                        Some(25)
                    }
                };
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(carry);
                true
            }
            0b001 => {
                // MOV / CMP / ADD / SUB imm8
                let rd = ((op >> 8) & 7) as u8;
                let imm = (op & 0xFF) as u16;
                match (op >> 11) & 3 {
                    0 => {
                        self.buf.mov_imm32(rd, imm as u32);
                        self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                        self.update_logical_flags(None);
                    }
                    1 => {
                        self.buf.emit(a64::subs32_imm(ZR, rd, imm, false));
                        self.buf.emit(a64::mrs_nzcv(23));
                        self.flags_live = true;
                    }
                    2 => {
                        self.buf.emit(a64::adds32_imm(rd, rd, imm, false));
                        self.buf.emit(a64::mrs_nzcv(23));
                        self.flags_live = true;
                    }
                    _ => {
                        self.buf.emit(a64::subs32_imm(rd, rd, imm, false));
                        self.buf.emit(a64::mrs_nzcv(23));
                        self.flags_live = true;
                    }
                }
                true
            }
            0b010 if (op & (1 << 12)) != 0 => {
                // Reg-offset load/store
                let rd = (op & 7) as u8;
                let rb = ((op >> 3) & 7) as u8;
                let ro = ((op >> 6) & 7) as u8;
                self.buf.emit(a64::add32_reg(16, rb, ro, Shift::Lsl, 0));
                let signed = (op & (1 << 9)) != 0;
                match ((op >> 10) & 3, signed) {
                    (0, true) => self.emit_write16(rd),
                    (1, true) => self.emit_read8(rd, true),
                    (2, true) => self.emit_read16(rd, false),
                    (_, true) => self.emit_read16(rd, true),
                    (0, false) => self.emit_write32_gpr(rd),
                    (1, false) => self.emit_write8(rd),
                    (2, false) => self.emit_read32_gpr(rd),
                    (_, false) => self.emit_read8(rd, false),
                }
                true
            }
            0b010 if (op & (1 << 11)) != 0 => {
                // LDR Rd, [PC, #imm]
                let rd = ((op >> 8) & 7) as u8;
                let lit = ((a + 4) & !3).wrapping_add((op & 0xFF) * 4);
                if self.options.fold_text_literals
                    && lit >= self.text.base
                    && lit.saturating_add(4) <= self.text.end()
                {
                    if let Some(val) = self.text.read32(lit) {
                        self.folded_spans.push((lit, lit + 4));
                        self.buf.mov_imm32(rd, val);
                        return true;
                    }
                }
                self.buf.mov_imm32(16, lit);
                self.emit_read32_gpr(rd);
                true
            }
            0b010 if (op & (1 << 10)) != 0 => self.thumb_high_register(a, op),
            0b010 => self.thumb_alu(op),
            0b011 => {
                let rd = (op & 7) as u8;
                let rb = ((op >> 3) & 7) as u8;
                let byte = (op & (1 << 12)) != 0;
                let off = (((op >> 6) & 0x1F) * if byte { 1 } else { 4 }) as u16;
                self.buf.emit(a64::add32_imm(16, rb, off, false));
                match ((op & (1 << 11)) != 0, byte) {
                    (true, false) => self.emit_read32_gpr(rd),
                    (true, true) => self.emit_read8(rd, false),
                    (false, false) => self.emit_write32_gpr(rd),
                    (false, true) => self.emit_write8(rd),
                }
                true
            }
            0b100 if (op & (1 << 12)) != 0 => {
                let rd = ((op >> 8) & 7) as u8;
                let off = ((op & 0xFF) * 4) as u16;
                self.buf.emit(a64::add32_imm(16, 13, off, false));
                if (op & (1 << 11)) != 0 {
                    self.emit_read32_gpr(rd);
                } else {
                    self.emit_write32_gpr(rd);
                }
                true
            }
            0b100 => {
                let rd = (op & 7) as u8;
                let rb = ((op >> 3) & 7) as u8;
                let off = (((op >> 6) & 0x1F) * 2) as u16;
                self.buf.emit(a64::add32_imm(16, rb, off, false));
                if (op & (1 << 11)) != 0 {
                    self.emit_read16(rd, false);
                } else {
                    self.emit_write16(rd);
                }
                true
            }
            0b101 if (op & (1 << 12)) != 0 => self.thumb_misc(a, op),
            0b101 => {
                let rd = ((op >> 8) & 7) as u8;
                let off = (op & 0xFF) * 4;
                if (op & (1 << 11)) != 0 {
                    self.buf.emit(a64::add32_imm(rd, 13, off as u16, false));
                } else {
                    self.buf.mov_imm32(rd, ((a + 4) & !3).wrapping_add(off));
                }
                true
            }
            0b110 if (op & (1 << 12)) != 0 => {
                match (op >> 8) & 0xF {
                    0xF => self.emit_svc(a + 2, op & 0xFF),
                    0xE => self.emit_interpret(a, op),
                    c => {
                        let s8 = (((op & 0xFF) << 24) as i32 >> 24) as u32;
                        let target = (a + 4).wrapping_add(s8 << 1);
                        self.ensure_hw_flags();
                        if self.mode == Mode::Thumb && self.labels.contains(&target) {
                            let p = self.buf.emit(a64::nop());
                            self.local_patches.push(LocalPatch { at_word: p, target_label: target, cond: Some(c as u8) });
                        } else {
                            let skip = self.buf.emit(a64::nop());
                            let _ = self.emit_jump(target, true);
                            let end = self.buf.pos();
                            self.buf.patch(skip, a64::b_cond(cond::invert(c as u8), end as i32 - skip as i32));
                        }
                        true
                    }
                }
            }
            0b110 => {
                // LDMIA / STMIA
                let rb = ((op >> 8) & 7) as u8;
                let list = op & 0xFF;
                if list == 0 {
                    self.buf.emit(a64::add32_imm(rb, rb, 0x40, false));
                    return true;
                }
                let load = (op & (1 << 11)) != 0;
                let span = list.count_ones() * 4;
                self.buf.emit(a64::mov32(27, rb));
                let words: Vec<SpanWord> = (0..8).filter(|i| (list & (1 << i)) != 0).map(|r| SpanWord::Gpr(r as u8)).collect();
                self.emit_words_span(load, &words);
                if !(load && (list & (1 << rb)) != 0) {
                    self.buf.emit(a64::add32_imm(rb, 27, span as u16, false));
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

    fn thumb_alu(&mut self, op: u32) -> bool {
        let rd = (op & 7) as u8;
        let rs = ((op >> 3) & 7) as u8;
        match (op >> 6) & 0xF {
            0x0 => {
                self.buf.emit(a64::ands32_reg(rd, rd, rs, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            0x1 => {
                self.buf.emit(a64::eor32_reg(rd, rd, rs, Shift::Lsl, 0));
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            kind @ (0x2 | 0x3 | 0x4 | 0x7) => {
                self.buf.emit(a64::mov32(16, rd));
                self.buf.emit(a64::mov32(17, rs));
                let stub = match kind {
                    0x2 => StubId::ShiftLslC,
                    0x3 => StubId::ShiftLsrC,
                    0x4 => StubId::ShiftAsrC,
                    _ => StubId::ShiftRorC,
                };
                self.stub_call(stub);
                self.buf.emit(a64::mov32(rd, 16));
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(Some(25));
            }
            0x5 => {
                self.ensure_hw_flags();
                self.buf.emit(a64::adcs32(rd, rd, rs));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
            }
            0x6 => {
                self.ensure_hw_flags();
                self.buf.emit(a64::sbcs32(rd, rd, rs));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
            }
            0x8 => {
                self.buf.emit(a64::ands32_reg(ZR, rd, rs, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            0x9 => {
                self.buf.emit(a64::subs32_reg(rd, ZR, rs, Shift::Lsl, 0));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
            }
            0xA => {
                self.buf.emit(a64::subs32_reg(ZR, rd, rs, Shift::Lsl, 0));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
            }
            0xB => {
                self.buf.emit(a64::adds32_reg(ZR, rd, rs, Shift::Lsl, 0));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
            }
            0xC => {
                self.buf.emit(a64::orr32_reg(rd, rd, rs, Shift::Lsl, 0));
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            0xD => {
                self.buf.emit(a64::mul32(rd, rd, rs));
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            0xE => {
                self.buf.emit(a64::bics32_reg(rd, rd, rs, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
            _ => {
                self.buf.emit(a64::mvn32(rd, rs));
                self.buf.emit(a64::ands32_reg(ZR, rd, rd, Shift::Lsl, 0));
                self.update_logical_flags(None);
            }
        }
        true
    }

    fn thumb_high_register(&mut self, a: u32, op: u32) -> bool {
        let rd = (op & 7) | ((op >> 4) & 8);
        let rs = (op >> 3) & 0xF;
        let src = self.get_reg(rs, a + 4, 16);
        match (op >> 8) & 3 {
            0 if rd == 15 => {
                let d_val = self.get_reg(rd, a + 4, 17);
                self.buf.emit(a64::add32_reg(15, d_val, src, Shift::Lsl, 0));
                self.buf.emit(a64::and32_imm(15, 15, !1u32).unwrap());
                self.stub_jump(StubId::IndirectJump);
                false
            }
            0 => {
                self.buf.emit(a64::add32_reg(rd as u8, rd as u8, src, Shift::Lsl, 0));
                true
            }
            1 => {
                let d_val = self.get_reg(rd, a + 4, 17);
                self.buf.emit(a64::subs32_reg(ZR, d_val, src, Shift::Lsl, 0));
                self.buf.emit(a64::mrs_nzcv(23));
                self.flags_live = true;
                true
            }
            2 if rd == 15 && rs == 14 => {
                self.buf.emit(a64::and32_imm(15, 14, !1u32).unwrap());
                self.emit_epilogue_jump();
                false
            }
            2 if rd == 15 => {
                self.buf.emit(a64::and32_imm(15, src, !1u32).unwrap());
                self.stub_jump(StubId::IndirectJump);
                false
            }
            2 => {
                if rd as u8 != src {
                    self.buf.emit(a64::mov32(rd as u8, src));
                }
                true
            }
            _ if (op & (1 << 7)) != 0 => {
                // Thumb BLX Rm
                if src != 16 {
                    self.buf.emit(a64::mov32(16, src));
                }
                let next = a + 2;
                self.buf.mov_imm32(14, next | 1);
                self.buf.emit(a64::and32_imm(17, 16, 1).unwrap());
                self.buf.emit(a64::strb_uoff(17, 19, offsets::CTX_THUMB));
                self.buf.emit(a64::and32_imm(15, 16, !3u32).unwrap());
                self.buf.emit(a64::cbz32(17, 2));
                self.buf.emit(a64::and32_imm(15, 16, !1u32).unwrap());

                self.buf.emit(a64::add32_imm(24, 24, 1, false));
                self.buf.emit(a64::cmp32_imm(24, RECOMP_DEPTH_LIMIT));
                self.buf.emit(a64::b_cond(cond::LS, 4));
                self.buf.emit(a64::sub32_imm(24, 24, 1, false));
                self.buf.mov_imm32(28, EXIT_UNWIND);
                self.emit_epilogue_jump();

                self.stub_call(StubId::IndirectCall);
                self.flags_live = false;
                self.buf.emit(a64::sub32_imm(24, 24, 1, false));
                self.emit_check_unwind();

                // RETURNED_T(next):
                self.buf.mov_imm32(16, next);
                self.buf.emit(a64::sub32_reg(16, 15, 16, Shift::Lsl, 0));
                self.buf.emit(a64::ldrb_uoff(17, 19, offsets::CTX_THUMB));
                self.buf.emit(a64::eor32_imm(17, 17, 1).unwrap());
                self.buf.emit(a64::orr32_reg(16, 16, 17, Shift::Lsl, 0));
                self.buf.emit(a64::cbz32(16, 2));
                self.stub_jump(StubId::IndirectJump);
                true
            }
            _ if rs == 14 => self.emit_return_to_reg(14),
            _ => self.emit_jump_to_reg(src),
        }
    }

    fn thumb_misc(&mut self, a: u32, op: u32) -> bool {
        let rd = (op & 7) as u8;
        let rm = ((op >> 3) & 7) as u8;
        match (op >> 8) & 0xF {
            0b0000 => {
                let off = ((op & 0x7F) * 4) as u16;
                if (op & (1 << 7)) != 0 {
                    self.buf.emit(a64::sub32_imm(13, 13, off, false));
                } else {
                    self.buf.emit(a64::add32_imm(13, 13, off, false));
                }
                true
            }
            0b0010 => {
                match (op >> 6) & 3 {
                    0 => { self.buf.emit(a64::sxth32(rd, rm)); }
                    1 => { self.buf.emit(a64::sxtb32(rd, rm)); }
                    2 => { self.buf.emit(a64::uxth32(rd, rm)); }
                    _ => { self.buf.emit(a64::uxtb32(rd, rm)); }
                }
                true
            }
            0b1010 => {
                match (op >> 6) & 3 {
                    0 => { self.buf.emit(a64::rev32(rd, rm)); }
                    1 => { self.buf.emit(a64::rev16_32(rd, rm)); }
                    3 => {
                        self.buf.emit(a64::rev16_32(16, rm));
                        self.buf.emit(a64::sxth32(rd, 16));
                    }
                    _ => return self.emit_interpret(a, op),
                }
                true
            }
            0b0110 | 0b1111 => true,
            0b0100 | 0b0101 | 0b1100 | 0b1101 => {
                let list = op & 0xFF;
                let extra = (op & (1 << 8)) != 0;
                let count = list.count_ones() + (extra as u32);
                if count == 0 {
                    return self.emit_interpret(a, op);
                }
                let mut words: Vec<SpanWord> =
                    (0..8).filter(|i| (list & (1 << i)) != 0).map(|r| SpanWord::Gpr(r as u8)).collect();
                if (op & (1 << 11)) != 0 {
                    // POP
                    self.buf.emit(a64::mov32(27, 13));
                    if extra {
                        words.push(SpanWord::Gpr(15));
                    }
                    self.emit_words_span(true, &words);
                    self.buf.emit(a64::add32_imm(13, 27, (count * 4) as u16, false));
                    if extra {
                        return self.emit_return_to_reg(15);
                    }
                } else {
                    // PUSH
                    self.buf.emit(a64::sub32_imm(27, 13, (count * 4) as u16, false));
                    if extra {
                        words.push(SpanWord::Gpr(14));
                    }
                    self.emit_words_span(false, &words);
                    self.buf.emit(a64::mov32(13, 27));
                }
                true
            }
            _ => self.emit_interpret(a, op),
        }
    }
}

#[derive(Clone, Copy)]
enum SpanWord {
    Gpr(u8),
    Const(u32),
    Vfp(u8),
}

// Keep unused warning quiet on helper re-exports
const _: fn(u32, u32) -> discover::Instruction = decode_arm;

#[cfg(test)]
mod size_tests {
    use super::*;

    #[test]
    fn inspect_kiu_aot_size_and_speed() {
        let rom_path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../roms/Kid Icarus - Uprising (USA) (En,Fr,Es) (Rev 1).cci"
        ));
        if !rom_path.exists() {
            return;
        }
        let title = zakuro_fs::Title::load(rom_path).unwrap();
        let code = title.code().unwrap();
        let t0 = std::time::Instant::now();
        let image = compile_title(
            title.program_id(),
            &title.exheader,
            &code,
            &[],
            CompileOptions::for_title(title.program_id()),
            |_, _| {},
        );
        let elapsed = t0.elapsed();
        let mib = image.code_bytes().len() as f64 / (1024.0 * 1024.0);
        println!(
            "KIU AOT compiled on host in {elapsed:.2?}: {mib:.2} MiB ({} words, {} entries, {} functions, {} pieces)",
            image.code.len(),
            image.entries.len(),
            image.origins.functions.len(),
            image.origins.pieces.len(),
        );
        assert!(mib < 96.0, "expected <= 96 MiB, got {mib:.2} MiB");
    }
}
