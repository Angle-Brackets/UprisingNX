//! Static x86_64 machine-code disassembly & ABI invariant auditor + full-ROM
//! entry-point differential sweep for `zakuro-aot`.

#![cfg(target_arch = "x86_64")]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use iced_x86::{
    Code, Decoder, DecoderOptions, FastFormatter, FlowControl, Instruction,
    InstructionInfoFactory, InstructionInfoOptions, Mnemonic, OpAccess, OpKind, Register,
    RflagsBits,
};
use rayon::prelude::*;
use recomp_abi::{Context, Host, EXIT_BUDGET, EXIT_NONE, EXIT_SVC, EXIT_UNWIND};
use zakuro_aot::discover::{OriginData, Program, Segment, Source};
use zakuro_aot::runtime::{offsets, ExecBuffer};
use zakuro_aot::{compile_program, compile_title, AotImage, CompileOptions, LoadedAot};
use zakuro_cpu::{Bus, Cpu, Exit};

const KIU_OVERRIDES: [u32; 5] = [
    0x006E_A040,
    0x00A3_EDCC,
    0x006D_269C,
    0x006D_2ABC,
    0x0074_62A4,
];

#[derive(Debug, Clone, Copy)]
pub struct AuditStats {
    pub total_bytes: usize,
    pub total_insns: u64,
    pub total_entries: usize,
    pub total_branches: u64,
    pub total_host_calls: u64,
    pub stub_bytes: usize,
}

struct BitSet {
    words: Vec<u64>,
    len: usize,
}

impl BitSet {
    fn new(len: usize) -> Self {
        Self { words: vec![0u64; (len + 64) / 64], len }
    }

    #[inline]
    fn set(&mut self, idx: usize) {
        assert!(idx < self.len, "bitset index {idx} out of bounds (len={})", self.len);
        self.words[idx >> 6] |= 1u64 << (idx & 63);
    }

    #[inline]
    fn get(&self, idx: usize) -> bool {
        idx < self.len && ((self.words[idx >> 6] >> (idx & 63)) & 1) != 0
    }
}

fn fmt_insn(instr: &Instruction) -> String {
    let mut out = String::new();
    let mut fmt = FastFormatter::new();
    fmt.format(instr, &mut out);
    format!("0x{:08X}: {out}", instr.ip())
}

fn dump_window(code: &[u8], focus_ip: u64) -> String {
    let start = focus_ip.saturating_sub(32) as usize;
    let end = (focus_ip as usize + 32).min(code.len());
    let mut decoder = Decoder::with_ip(64, &code[start..end], start as u64, DecoderOptions::NONE);
    let mut fmt = FastFormatter::new();
    let mut lines = Vec::new();
    while decoder.can_decode() {
        let instr = decoder.decode();
        let mut s = String::new();
        fmt.format(&instr, &mut s);
        let mark = if instr.ip() == focus_ip { " <--" } else { "" };
        lines.push(format!("  0x{:08X}: {s}{mark}", instr.ip()));
    }
    lines.join("\n")
}

#[derive(Clone, Copy)]
struct StubLayout {
    sync_in_start: u64,
    invoke_r11_start: u64,
    enter_start: u64,
    enter_end: u64,
    stub_bytes: usize,
}

fn locate_stubs(code: &[u8], enter_offset: u32, min_entry_off: Option<usize>) -> StubLayout {
    let enter_start = enter_offset as u64;
    assert!(
        (enter_start as usize) < code.len(),
        "enter_offset {enter_offset} >= code.len() {}",
        code.len()
    );

    let mut decoder = Decoder::with_ip(64, code, 0, DecoderOptions::NONE);
    let mut ret_count_pre_enter = 0usize;
    let mut sync_in_start = None;
    let mut invoke_r11_start = None;
    let mut enter_end = None;
    let mut post_enter_rets = 0usize;
    let mut stub_bytes = None;

    while decoder.can_decode() {
        let instr = decoder.decode();
        let ip = instr.ip();
        let next_ip = instr.next_ip();

        if ip < enter_start {
            if instr.flow_control() == FlowControl::Return {
                ret_count_pre_enter += 1;
                if ret_count_pre_enter == 1 {
                    sync_in_start = Some(next_ip);
                } else if ret_count_pre_enter == 2 {
                    invoke_r11_start = Some(next_ip);
                }
            }
        } else if enter_end.is_none() {
            if instr.flow_control() == FlowControl::Return {
                enter_end = Some(next_ip);
            }
        } else if instr.flow_control() == FlowControl::Return {
            post_enter_rets += 1;
            // After `enter`: 11 frame stubs (`slow_read8..32`, `slow_write8..32`,
            // `indirect_call/jump`, `interpret`, `media_parallel`, `vfp_vector`, `vfp_to_int`)
            // + 4 shift stubs (`shift_lsl_c`:3 + `shift_lsr_c`:3 + `shift_asr_c`:2 + `shift_ror_c`:1 = 9)
            // = 20 `ret` instructions.
            if post_enter_rets == 20 {
                stub_bytes = Some(next_ip as usize);
                break;
            }
        }
    }

    let sync_in_start = sync_in_start.expect("failed to locate sync_in stub end");
    let invoke_r11_start = invoke_r11_start.expect("failed to locate invoke_r11 stub start");
    let enter_end = enter_end.expect("failed to locate enter stub end");
    let stub_bytes = stub_bytes.expect("failed to locate end of runtime stubs");

    assert!(
        sync_in_start < invoke_r11_start && invoke_r11_start < enter_start && enter_start < enter_end,
        "invalid stub ordering: sync_in={sync_in_start:#x} invoke_r11={invoke_r11_start:#x} enter=[{enter_start:#x},{enter_end:#x})"
    );
    assert!(
        stub_bytes <= code.len(),
        "stub_bytes {stub_bytes} exceeds code.len() {}",
        code.len()
    );
    if let Some(min_off) = min_entry_off {
        assert!(
            stub_bytes <= min_off,
            "stub_bytes {stub_bytes:#x} exceeds earliest entry offset {min_off:#x}"
        );
    }
    if stub_bytes < code.len() {
        let mut d = Decoder::with_ip(64, &code[stub_bytes..], stub_bytes as u64, DecoderOptions::NONE);
        let i0 = d.decode();
        let i1 = d.decode();
        assert!(
            i0.mnemonic() == Mnemonic::Push
                && i0.op0_kind() == OpKind::Register
                && i0.op0_register() == Register::RBP
                && i1.mnemonic() == Mnemonic::Mov
                && i1.op0_kind() == OpKind::Register
                && i1.op0_register() == Register::RBP
                && i1.op1_kind() == OpKind::Register
                && i1.op1_register() == Register::RSP,
            "expected first compiled function prologue `push rbp; mov rbp, rsp` at stub_bytes={stub_bytes:#x}, got:\n{}\n{}",
            fmt_insn(&i0),
            fmt_insn(&i1),
        );
    }

    StubLayout { sync_in_start, invoke_r11_start, enter_start, enter_end, stub_bytes }
}

fn verify_stub_stack_alignment(code: &[u8], layout: &StubLayout) {
    // Decode all stub instructions into a compact offset-indexed table so we can
    // propagate `rsp % 16` along control-flow edges in `0..layout.stub_bytes`.
    let stub_len = layout.stub_bytes;
    let mut decoder = Decoder::with_ip(64, &code[..stub_len], 0, DecoderOptions::NONE);
    let mut insns: Vec<Instruction> = Vec::new();
    let mut idx_at = vec![u32::MAX; stub_len];
    while decoder.can_decode() {
        let instr = decoder.decode();
        let ip = instr.ip() as usize;
        idx_at[ip] = insns.len() as u32;
        insns.push(instr);
    }

    let mut rsp_mod16: Vec<Option<u8>> = vec![None; insns.len()];
    let mut queue: VecDeque<(usize, u8)> = VecDeque::new();
    let mut seed_ip = |ip: usize, mod16: u8| {
        if ip < stub_len {
            let idx = idx_at[ip];
            assert_ne!(idx, u32::MAX, "stub seed 0x{ip:X} not at instruction start");
            queue.push_back((idx as usize, mod16));
        }
    };

    // Function/call-style stub entries start with `RSP % 16 == 8` (caller `call` pushed 8 bytes):
    seed_ip(0, 8);
    seed_ip(layout.sync_in_start as usize, 8);
    seed_ip(layout.invoke_r11_start as usize, 8);
    seed_ip(layout.enter_start as usize, 8);

    // After `enter_end`, each of the 11 frame stubs starts right after the previous `ret`.
    // Also detect the 4 shift stubs (which have multiple `ret`s each, all at `mod16 == 8`).
    let mut post_enter_rets = 0usize;
    let mut next_is_frame_stub = true;
    for instr in &insns {
        let ip = instr.ip() as usize;
        if ip < layout.enter_end as usize {
            continue;
        }
        if next_is_frame_stub {
            seed_ip(ip, 8);
            next_is_frame_stub = false;
        }
        if instr.flow_control() == FlowControl::Return {
            post_enter_rets += 1;
            if post_enter_rets < 11 {
                next_is_frame_stub = true;
            } else if matches!(post_enter_rets, 11 | 14 | 17 | 19) {
                // Entry of shift_lsl_c (after #11), shift_lsr_c (#14), shift_asr_c (#17), shift_ror_c (#19)
                seed_ip(instr.next_ip() as usize, 8);
            }
        }
    }

    // `indirect_jump` (immediately after `indirect_call`'s `push rbp; mov rbp, rsp`) is
    // entered by tail-jumps with `RSP % 16 == 0`.
    let mut found_indirect_jump = false;
    for w in insns.windows(3) {
        if w[0].ip() >= layout.enter_end
            && w[0].mnemonic() == Mnemonic::Push
            && w[0].op0_register() == Register::RBP
            && w[1].mnemonic() == Mnemonic::Mov
            && w[1].op0_register() == Register::RBP
            && w[1].op1_register() == Register::RSP
            && w[2].mnemonic() == Mnemonic::Mov
            && w[2].op0_register() == Register::EAX
        {
            // Confirm `indirect_call` -> `indirect_jump`: the next instruction reads `[rbx + CTX_R15]`
            if w[2].op1_kind() == OpKind::Memory && w[2].memory_base() == Register::RBX {
                seed_ip(w[2].ip() as usize, 0);
                found_indirect_jump = true;
                break;
            }
        }
    }
    assert!(found_indirect_jump, "failed to locate indirect_jump stub entry");

    while let Some((idx, cur_mod16)) = queue.pop_front() {
        if let Some(prev) = rsp_mod16[idx] {
            assert_eq!(
                prev, cur_mod16,
                "inconsistent stack alignment in stub at {}: {prev} vs {cur_mod16}",
                fmt_insn(&insns[idx])
            );
            continue;
        }
        rsp_mod16[idx] = Some(cur_mod16);
        let instr = &insns[idx];

        let mut next_mod16 = cur_mod16;
        match instr.mnemonic() {
            Mnemonic::Push => next_mod16 = (next_mod16 + 8) & 15,
            Mnemonic::Pop => next_mod16 = (next_mod16 + 8) & 15,
            Mnemonic::Sub
                if instr.op0_kind() == OpKind::Register
                    && instr.op0_register() == Register::RSP
                    && matches!(instr.op1_kind(), OpKind::Immediate8to64 | OpKind::Immediate32to64) =>
            {
                let imm = instr.immediate(1);
                next_mod16 = ((next_mod16 as i64 - imm as i64).rem_euclid(16)) as u8;
            }
            Mnemonic::Add
                if instr.op0_kind() == OpKind::Register
                    && instr.op0_register() == Register::RSP
                    && matches!(instr.op1_kind(), OpKind::Immediate8to64 | OpKind::Immediate32to64) =>
            {
                let imm = instr.immediate(1);
                next_mod16 = ((next_mod16 as i64 + imm as i64).rem_euclid(16)) as u8;
            }
            _ => {}
        }

        match instr.flow_control() {
            FlowControl::Return => {
                assert_eq!(
                    cur_mod16, 8,
                    "stub return with unaligned RSP%16={cur_mod16} at {}",
                    fmt_insn(instr)
                );
            }
            FlowControl::Call | FlowControl::IndirectCall => {
                assert_eq!(
                    cur_mod16, 0,
                    "stub call with unaligned RSP%16={cur_mod16} (must be 0 before call) at {}",
                    fmt_insn(instr)
                );
                if (instr.next_ip() as usize) < stub_len {
                    let next_idx = idx_at[instr.next_ip() as usize] as usize;
                    queue.push_back((next_idx, next_mod16));
                }
            }
            FlowControl::UnconditionalBranch => {
                let target = instr.near_branch64() as usize;
                assert!(target < stub_len);
                queue.push_back((idx_at[target] as usize, next_mod16));
            }
            FlowControl::ConditionalBranch => {
                let target = instr.near_branch64() as usize;
                assert!(target < stub_len);
                queue.push_back((idx_at[target] as usize, next_mod16));
                if (instr.next_ip() as usize) < stub_len {
                    queue.push_back((idx_at[instr.next_ip() as usize] as usize, next_mod16));
                }
            }
            FlowControl::IndirectBranch => {
                assert_eq!(
                    cur_mod16, 0,
                    "stub indirect tail jump with RSP%16={cur_mod16} (expected 0) at {}",
                    fmt_insn(instr)
                );
            }
            FlowControl::Next => {
                if (instr.next_ip() as usize) < stub_len {
                    queue.push_back((idx_at[instr.next_ip() as usize] as usize, next_mod16));
                }
            }
            other => panic!("unexpected stub FlowControl {other:?} at {}", fmt_insn(instr)),
        }
    }

    for (i, m) in rsp_mod16.iter().enumerate() {
        assert!(m.is_some(), "unreachable stub instruction at {}", fmt_insn(&insns[i]));
    }
}

pub fn audit_aot_image(image: &AotImage) -> AuditStats {
    let code = image.code_bytes();
    assert!(!code.is_empty(), "AOT image code buffer is empty");

    let mut insn_starts = BitSet::new(code.len() + 1);
    let mut branch_targets = BitSet::new(code.len() + 1);
    let mut total_insns = 0u64;
    let mut total_branches = 0u64;

    // Pass 1: Linear decode, instruction validity, branch-target capture.
    {
        let mut decoder = Decoder::with_ip(64, code, 0, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instr = decoder.decode();
            let ip = instr.ip() as usize;
            assert!(
                !instr.is_invalid()
                    && instr.code() != Code::INVALID
                    && instr.mnemonic() != Mnemonic::Ud2
                    && instr.mnemonic() != Mnemonic::Int3,
                "invalid/trap instruction decoded at 0x{ip:08X}:\n{}",
                dump_window(code, ip as u64)
            );
            insn_starts.set(ip);
            total_insns += 1;

            match instr.flow_control() {
                FlowControl::Next | FlowControl::Return => {}
                FlowControl::ConditionalBranch
                | FlowControl::UnconditionalBranch
                | FlowControl::Call => {
                    total_branches += 1;
                    assert_eq!(
                        instr.op0_kind(),
                        OpKind::NearBranch64,
                        "expected NearBranch64 operand at {}",
                        fmt_insn(&instr)
                    );
                    let target = instr.near_branch64() as usize;
                    assert!(
                        target < code.len(),
                        "out-of-bounds branch target 0x{target:08X} (code len 0x{:X}) at {}",
                        code.len(),
                        fmt_insn(&instr)
                    );
                    branch_targets.set(target);
                }
                FlowControl::IndirectBranch | FlowControl::IndirectCall => {
                    total_branches += 1;
                }
                other => {
                    panic!("unexpected FlowControl {other:?} at {}", fmt_insn(&instr));
                }
            }
        }
        assert_eq!(
            decoder.position(),
            code.len(),
            "decoder stopped early at {} of {} bytes",
            decoder.position(),
            code.len()
        );
    }

    assert!(
        (image.enter_offset as usize) < code.len(),
        "enter_offset {} out of bounds ({})",
        image.enter_offset,
        code.len()
    );
    branch_targets.set(image.enter_offset as usize);

    let mut min_entry_off: Option<usize> = None;
    for &(guest_key, byte_off) in &image.entries {
        let off = byte_off as usize;
        assert!(
            off < code.len(),
            "entry 0x{guest_key:08X} byte offset {off} >= code.len() {}",
            code.len()
        );
        branch_targets.set(off);
        min_entry_off = Some(min_entry_off.map_or(off, |m| m.min(off)));
    }

    for (word_idx, (&bt, &is)) in branch_targets
        .words
        .iter()
        .zip(insn_starts.words.iter())
        .enumerate()
    {
        let bad = bt & !is;
        if bad != 0 {
            let bit = bad.trailing_zeros() as usize;
            let off = (word_idx << 6) + bit;
            panic!(
                "branch/entry target 0x{off:08X} does NOT land on an instruction boundary!\n{}",
                dump_window(code, off as u64)
            );
        }
    }

    let layout = locate_stubs(code, image.enter_offset, min_entry_off);
    verify_stub_stack_alignment(code, &layout);

    // Pass 2: Register pinning, stack frame integrity, HOST_MXCSR restore, and EFLAGS dataflow.
    let mut total_host_calls = 0u64;
    let mut live_flags: u32 = 0;
    let mut prev_instr: Option<Instruction> = None;
    let mut info_factory = InstructionInfoFactory::new();

    // Check `enter` prologue/epilogue callee-saved restore and HOST_MXCSR save/restore.
    let mut enter_saved_host_mxcsr = false;
    let mut enter_restored_host_mxcsr = false;
    let mut enter_pops = Vec::new();
    let mut pending_mov_rbp_rsp = false;
    let mut pending_ret_after_pop_rbp = false;

    let mut decoder = Decoder::with_ip(64, code, 0, DecoderOptions::NONE);
    while decoder.can_decode() {
        let instr = decoder.decode();
        let ip = instr.ip();
        let in_enter = ip >= layout.enter_start && ip < layout.enter_end;
        let in_sync_in = ip >= layout.sync_in_start && ip < layout.invoke_r11_start;
        let in_stubs = (ip as usize) < layout.stub_bytes;

        if ip == 0 || ip == image.enter_offset as u64 || branch_targets.get(ip as usize) {
            live_flags = 0;
        }

        // Verify paired prologue/epilogue invariants in compiled code.
        let mut was_push_rbp = false;
        if !in_stubs {
            if pending_mov_rbp_rsp {
                assert!(
                    instr.mnemonic() == Mnemonic::Mov
                        && instr.op0_kind() == OpKind::Register
                        && instr.op0_register() == Register::RBP
                        && instr.op1_kind() == OpKind::Register
                        && instr.op1_register() == Register::RSP,
                    "compiled `push rbp` not immediately followed by `mov rbp, rsp` at {}\n{}",
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
                pending_mov_rbp_rsp = false;
                was_push_rbp = true;
            }
            if pending_ret_after_pop_rbp {
                assert!(
                    instr.flow_control() == FlowControl::Return,
                    "compiled `pop rbp` not immediately followed by `ret` at {}\n{}",
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
                pending_ret_after_pop_rbp = false;
            } else if instr.flow_control() == FlowControl::Return {
                panic!(
                    "compiled `ret` not immediately preceded by `pop rbp` at {}\n{}",
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
            }
        }

        // EFLAGS read check before updating live_flags.
        let rflags_read = instr.rflags_read();
        if (rflags_read & !live_flags) != 0 {
            panic!(
                "EFLAGS hazard at {}: reads 0x{rflags_read:X} while live_flags=0x{live_flags:X} (missing=0x{:X})\n{}",
                fmt_insn(&instr),
                rflags_read & !live_flags,
                dump_window(code, ip)
            );
        }

        // HOST_MXCSR check on `call qword ptr [rbx + disp]`
        if instr.mnemonic() == Mnemonic::Call && instr.op0_kind() == OpKind::Memory {
            assert!(
                instr.memory_base() == Register::RBX && instr.memory_index() == Register::None,
                "unexpected indirect memory call target at {}\n{}",
                fmt_insn(&instr),
                dump_window(code, ip)
            );
            total_host_calls += 1;
            let disp = instr.memory_displacement64();
            if disp != offsets::FN_VFP_VECTOR as u64 {
                let prev = prev_instr.as_ref().unwrap_or_else(|| {
                    panic!("host call without preceding instruction at {}", fmt_insn(&instr))
                });
                let ok_ldmxcsr = prev.mnemonic() == Mnemonic::Ldmxcsr
                    && prev.op0_kind() == OpKind::Memory
                    && prev.memory_base() == Register::RBX
                    && prev.memory_index() == Register::None
                    && prev.memory_displacement64() == offsets::HOST_MXCSR as u64;
                assert!(
                    ok_ldmxcsr,
                    "host call [rbx + {disp}] not preceded by `ldmxcsr dword ptr [rbx + {}]`!\n  prev: {}\n  call: {}\n{}",
                    offsets::HOST_MXCSR,
                    fmt_insn(prev),
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
            }
        }

        // Track `enter` invariants.
        if in_enter {
            if instr.mnemonic() == Mnemonic::Mov
                && instr.op0_kind() == OpKind::Memory
                && instr.memory_base() == Register::RBX
                && instr.memory_displacement64() == offsets::HOST_MXCSR as u64
            {
                enter_saved_host_mxcsr = true;
            }
            if instr.mnemonic() == Mnemonic::Ldmxcsr
                && instr.op0_kind() == OpKind::Memory
                && instr.memory_base() == Register::RBX
                && instr.memory_displacement64() == offsets::HOST_MXCSR as u64
            {
                enter_restored_host_mxcsr = true;
            }
            if instr.mnemonic() == Mnemonic::Pop && instr.op0_kind() == OpKind::Register {
                enter_pops.push(instr.op0_register());
            }
        }

        // Pinned register & stack register write checks.
        let info = info_factory.info_options(&instr, InstructionInfoOptions::NO_MEMORY_USAGE);
        for reg_info in info.used_registers() {
            if !matches!(
                reg_info.access(),
                OpAccess::Write
                    | OpAccess::CondWrite
                    | OpAccess::ReadWrite
                    | OpAccess::ReadCondWrite
            ) {
                continue;
            }
            let full = reg_info.register().full_register();
            if full == Register::RBX && !in_enter {
                panic!(
                    "RBX written outside `enter` at {}\n{}",
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
            }
            if (full == Register::R12 || full == Register::R13) && !in_enter && !in_sync_in {
                panic!(
                    "{full:?} written outside `enter`/`sync_in` at {}\n{}",
                    fmt_insn(&instr),
                    dump_window(code, ip)
                );
            }
            if !in_stubs {
                if full == Register::R14
                    && !matches!(instr.mnemonic(), Mnemonic::Add | Mnemonic::Sub)
                {
                    panic!(
                        "R14 (budget) written by non-Add/Sub at {}\n{}",
                        fmt_insn(&instr),
                        dump_window(code, ip)
                    );
                }
                if full == Register::R15
                    && !matches!(instr.mnemonic(), Mnemonic::Mov | Mnemonic::And | Mnemonic::Or)
                {
                    panic!(
                        "R15 (NZCV) written by unexpected op at {}\n{}",
                        fmt_insn(&instr),
                        dump_window(code, ip)
                    );
                }
                if full == Register::RSP || full == Register::RBP {
                    let is_push_rbp = instr.mnemonic() == Mnemonic::Push
                        && instr.op0_kind() == OpKind::Register
                        && instr.op0_register() == Register::RBP;
                    let is_mov_rbp_rsp = instr.mnemonic() == Mnemonic::Mov
                        && instr.op0_kind() == OpKind::Register
                        && instr.op0_register() == Register::RBP
                        && instr.op1_kind() == OpKind::Register
                        && instr.op1_register() == Register::RSP;
                    let is_pop_rbp = instr.mnemonic() == Mnemonic::Pop
                        && instr.op0_kind() == OpKind::Register
                        && instr.op0_register() == Register::RBP;
                    let is_ret = instr.flow_control() == FlowControl::Return;
                    let is_call_rsp = full == Register::RSP
                        && matches!(
                            instr.flow_control(),
                            FlowControl::Call | FlowControl::IndirectCall
                        );

                    if is_push_rbp {
                        pending_mov_rbp_rsp = true;
                    } else if is_pop_rbp {
                        pending_ret_after_pop_rbp = true;
                    } else if !(is_mov_rbp_rsp && was_push_rbp) && !is_ret && !is_call_rsp {
                        panic!(
                            "compiled instruction illegally writes {full:?} at {}\n{}",
                            fmt_insn(&instr),
                            dump_window(code, ip)
                        );
                    }
                }
            }
        }

        // Update live_flags for next linear instruction.
        let mut defined =
            instr.rflags_written() | instr.rflags_cleared() | instr.rflags_set();
        let mut undef = instr.rflags_undefined();
        if in_stubs
            && matches!(instr.mnemonic(), Mnemonic::Shl | Mnemonic::Shr | Mnemonic::Sar)
            && instr.op_count() >= 2
            && instr.op1_kind() == OpKind::Register
            && instr.op1_register() == Register::CL
        {
            // Stub shift helpers explicitly guard `1 <= cl <= 31` before executing
            // `shl/shr/sar eax, cl`, so x86 always defines CF dynamically here.
            defined |= RflagsBits::CF;
            undef &= !RflagsBits::CF;
        }
        live_flags = (live_flags & !undef) | defined;

        if matches!(
            instr.flow_control(),
            FlowControl::Call
                | FlowControl::IndirectCall
                | FlowControl::IndirectBranch
                | FlowControl::UnconditionalBranch
                | FlowControl::Return
        ) {
            live_flags = 0;
        }

        prev_instr = Some(instr);
    }

    assert!(!pending_mov_rbp_rsp, "trailing `push rbp` at end of buffer");
    assert!(!pending_ret_after_pop_rbp, "trailing `pop rbp` at end of buffer");
    assert!(enter_saved_host_mxcsr, "`enter` did not save HOST_MXCSR");
    assert!(enter_restored_host_mxcsr, "`enter` did not restore HOST_MXCSR");
    assert_eq!(
        enter_pops,
        vec![
            Register::R15,
            Register::R14,
            Register::R13,
            Register::R12,
            Register::RBX,
            Register::RBP,
        ],
        "`enter` did not restore callee-saved registers in expected order"
    );

    AuditStats {
        total_bytes: code.len(),
        total_insns,
        total_entries: image.entries.len(),
        total_branches,
        total_host_calls,
        stub_bytes: layout.stub_bytes,
    }
}

// ---------------------------------------------------------------------------
// Synthetic ARM + Thumb + VFP + DSP + Overrides + Slowpaths Audit Test
// ---------------------------------------------------------------------------

#[allow(dead_code)]
mod synth {
    pub const EQ: u32 = 0;
    pub const NE: u32 = 1;
    pub const AL: u32 = 14;

    pub const LSL: u32 = 0;
    pub const LSR: u32 = 1;
    pub const ASR: u32 = 2;
    pub const ROR: u32 = 3;

    pub const OP_ADD: u32 = 4;
    pub const OP_ADC: u32 = 5;
    pub const OP_SBC: u32 = 6;
    pub const OP_RSB: u32 = 3;
    pub const OP_MOV: u32 = 13;

    pub const fn dp_imm(cond: u32, op: u32, s: bool, rn: u32, rd: u32, imm8: u32, rot4: u32) -> u32 {
        (cond << 28) | (1 << 25) | (op << 21) | ((s as u32) << 20) | (rn << 16) | (rd << 12) | (rot4 << 8) | (imm8 & 0xFF)
    }

    pub const fn dp_reg_imm(cond: u32, op: u32, s: bool, rn: u32, rd: u32, rm: u32, st: u32, imm5: u32) -> u32 {
        (cond << 28) | (op << 21) | ((s as u32) << 20) | (rn << 16) | (rd << 12) | ((imm5 & 31) << 7) | ((st & 3) << 5) | rm
    }

    pub const fn dp_reg_reg(cond: u32, op: u32, s: bool, rn: u32, rd: u32, rm: u32, st: u32, rs: u32) -> u32 {
        (cond << 28) | (op << 21) | ((s as u32) << 20) | (rn << 16) | (rd << 12) | ((rs & 15) << 8) | ((st & 3) << 5) | 0x10 | rm
    }

    pub const fn b_cond(cond: u32, cur: u32, target: u32) -> u32 {
        let off = (target.wrapping_sub(cur.wrapping_add(8)) as i32) >> 2;
        (cond << 28) | 0x0A00_0000 | ((off as u32) & 0x00FF_FFFF)
    }

    pub const fn bl_cond(cond: u32, cur: u32, target: u32) -> u32 {
        let off = (target.wrapping_sub(cur.wrapping_add(8)) as i32) >> 2;
        (cond << 28) | 0x0B00_0000 | ((off as u32) & 0x00FF_FFFF)
    }

    pub const fn blx_thumb(cur: u32, target_thumb: u32) -> u32 {
        let diff = target_thumb.wrapping_sub(cur.wrapping_add(8)) as i32;
        let h = ((diff >> 1) & 1) as u32;
        let imm24 = ((diff >> 2) as u32) & 0x00FF_FFFF;
        0xFA00_0000 | (h << 24) | imm24
    }

    pub const fn bx(cond: u32, rm: u32) -> u32 { (cond << 28) | 0x012F_FF10 | rm }
    pub const fn blx_reg(cond: u32, rm: u32) -> u32 { (cond << 28) | 0x012F_FF30 | rm }
    pub const fn clz(cond: u32, rd: u32, rm: u32) -> u32 { (cond << 28) | 0x016F_0F10 | (rd << 12) | rm }
    pub const fn mrs_cpsr(cond: u32, rd: u32) -> u32 { (cond << 28) | 0x010F_0000 | (rd << 12) }
    pub const fn msr_cpsr_f_imm(cond: u32, imm8: u32) -> u32 { (cond << 28) | 0x0328_F000 | (imm8 & 0xFF) }

    pub const fn ldr_pc_rel(cond: u32, rd: u32, byte: bool, cur: u32, lit: u32) -> u32 {
        let up = lit >= cur + 8;
        let off = if up { lit - (cur + 8) } else { (cur + 8) - lit };
        (cond << 28) | (1 << 26) | (1 << 24) | ((up as u32) << 23) | ((byte as u32) << 22) | (1 << 20) | (15 << 16) | (rd << 12) | (off & 0xFFF)
    }

    pub const fn vldr_pc_rel(cond: u32, double: bool, rd: u32, cur: u32, lit: u32) -> u32 {
        let (vd, db) = if double { (rd & 0xF, (rd >> 4) & 1) } else { ((rd >> 1) & 0xF, rd & 1) };
        let up = lit >= cur + 8;
        let words = (if up { lit - (cur + 8) } else { (cur + 8) - lit }) / 4;
        (cond << 28) | 0x0D1F_0A00 | ((double as u32) << 8) | (db << 22) | ((up as u32) << 23) | (vd << 12) | (words & 0xFF)
    }

    pub const fn vfp_dp(cond: u32, double: bool, opc1: u32, neg: bool, rd: u32, rn: u32, rm: u32) -> u32 {
        let (vd, db) = if double { (rd & 0xF, (rd >> 4) & 1) } else { ((rd >> 1) & 0xF, rd & 1) };
        let (vn, nb) = if double { (rn & 0xF, (rn >> 4) & 1) } else { ((rn >> 1) & 0xF, rn & 1) };
        let (vm, mb) = if double { (rm & 0xF, (rm >> 4) & 1) } else { ((rm >> 1) & 0xF, rm & 1) };
        (cond << 28) | 0x0E00_0A00 | ((double as u32) << 8) | ((opc1 & 0xB) << 20) | (db << 22) | (vn << 16) | (vd << 12) | (nb << 7) | ((neg as u32) << 6) | (mb << 5) | vm
    }

    pub const fn vfp_ext(cond: u32, double: bool, opc2: u32, top: bool, vd: u32, db: u32, vm: u32, mb: u32) -> u32 {
        (cond << 28) | 0x0EB0_0A40 | ((double as u32) << 8) | (db << 22) | ((opc2 & 0xF) << 16) | (vd << 12) | ((top as u32) << 7) | (mb << 5) | vm
    }

    pub struct Builder {
        pub base: u32,
        pub bytes: Vec<u8>,
    }

    impl Builder {
        pub fn new(base: u32) -> Self { Self { base, bytes: Vec::new() } }
        pub fn addr(&self) -> u32 { self.base + self.bytes.len() as u32 }
        pub fn align4(&mut self) { while self.bytes.len() % 4 != 0 { self.bytes.extend_from_slice(&0x46C0u16.to_le_bytes()); } }
        pub fn arm(&mut self, w: u32) -> u32 { let a = self.addr(); self.bytes.extend_from_slice(&w.to_le_bytes()); a }
        pub fn patch_arm(&mut self, addr: u32, w: u32) {
            let off = (addr - self.base) as usize;
            self.bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        pub fn thumb(&mut self, hw: u16) -> u32 { let a = self.addr(); self.bytes.extend_from_slice(&hw.to_le_bytes()); a }
        pub fn patch_thumb(&mut self, addr: u32, hw: u16) {
            let off = (addr - self.base) as usize;
            self.bytes[off..off + 2].copy_from_slice(&hw.to_le_bytes());
        }
        pub fn thumb_pair(&mut self, pair: [u16; 2]) -> u32 { let a = self.addr(); self.thumb(pair[0]); self.thumb(pair[1]); a }
    }
}

#[test]
fn audit_synthetic_x64_image_with_overrides_vfp_thumb_and_slowpaths() {
    use synth::*;

    const SYNTH_BASE: u32 = 0x0010_0000;
    let mut b = Builder::new(SYNTH_BASE);

    // 1. Overridden ARM function (`arm_ovr`)
    let arm_ovr = b.addr();
    b.arm(dp_imm(AL, OP_ADD, true, 0, 0, 1, 0)); // adds r0, r0, #1
    b.arm(bx(AL, 14));

    // 2. Thumb callee (`thumb_fn`) + Thumb overridden leaf (`thumb_ovr`)
    let thumb_ovr = b.addr();
    b.thumb(0x3007); // adds r0, #7
    b.thumb(0x4770); // bx lr

    let thumb_fn = b.addr();
    b.thumb(0xB510); // push {r4, lr}
    b.thumb(0x242A); // movs r4, #42
    b.thumb(0x1900); // adds r0, r0, r4
    b.thumb(0x4160); // adcs r0, r4
    b.thumb(0x41A0); // sbcs r0, r4
    b.thumb(0x4264); // negs r4, r4
    b.thumb(0x40A0); // lsls r0, r4 (ShiftLslC)
    b.thumb(0x40E0); // lsrs r0, r4 (ShiftLsrC)
    b.thumb(0x4120); // asrs r0, r4 (ShiftAsrC)
    b.thumb(0x41E0); // rors r0, r4 (ShiftRorC)
    b.thumb(0x4360); // muls r0, r4
    b.thumb(0x4480 | (8 & 7) | (1 << 7)); // add r8, r0
    b.thumb(0x4501); // cmp r1, r0
    let j_t_cond = b.thumb(0);
    b.thumb(0x3001); // adds r0, #1
    let t_after_cond = b.addr();
    let diff_t = ((t_after_cond.wrapping_sub(j_t_cond + 4)) as i32) >> 1;
    b.patch_thumb(j_t_cond, 0xD000 | ((diff_t as u16) & 0xFF)); // beq t_after_cond
    b.thumb(0x4798); // blx r3 (IndirectCall)
    // BL to thumb_ovr
    let diff_bl = thumb_ovr.wrapping_sub(b.addr() + 4) as i32;
    b.thumb_pair([
        0xF000 | (((diff_bl >> 12) as u16) & 0x7FF),
        0xF800 | (((diff_bl >> 1) as u16) & 0x7FF),
    ]);
    // BLX to arm_ovr
    let pc_al = (b.addr() + 4) & !3;
    let diff_blx = arm_ovr.wrapping_sub(pc_al) as i32;
    b.thumb_pair([
        0xF000 | (((diff_blx >> 12) as u16) & 0x7FF),
        0xE800 | (((diff_blx >> 1) as u16) & 0x7FE),
    ]);
    b.thumb(0xBD10); // pop {r4, pc}
    b.align4();

    // 3. ARM tail-jump caller into `arm_ovr`
    let arm_tail_ovr = b.addr();
    let j_tail = b.arm(0);
    b.patch_arm(j_tail, b_cond(AL, j_tail, arm_ovr));

    // 4. Main ARM function covering ALU, conditional skips/branches, memory, DSP/media,
    //    VFP scalar + short vector, literal folding, DirectCall/IndirectCall, Interpret, SVC.
    let arm_main = b.addr();
    let j_ldr_lit = b.arm(0);
    let j_vldr_s = b.arm(0);
    let j_vldr_d = b.arm(0);
    b.arm(dp_reg_imm(AL, OP_ADD, true, 0, 4, 1, LSL, 3)); // adds r4, r0, r1, lsl #3
    b.arm(dp_reg_imm(EQ, OP_ADC, true, 4, 5, 2, ROR, 0)); // adceqs r5, r4, r2, ror #0 (RRX)
    b.arm(dp_reg_reg(NE, OP_SBC, true, 5, 6, 1, LSL, 2)); // sbcnes r6, r5, r1, lsl r2
    b.arm(dp_reg_reg(AL, OP_MOV, true, 0, 7, 1, ASR, 2)); // movs r7, r1, asr r2
    b.arm(dp_imm(AL, OP_RSB, true, 7, 7, 1, 0)); // rsbs r7, r7, #1
    b.arm(clz(AL, 8, 7));
    b.arm(mrs_cpsr(AL, 9));
    b.arm(msr_cpsr_f_imm(AL, 0xF0));
    let j_cond_br = b.arm(0);
    b.arm(dp_imm(AL, OP_ADD, false, 0, 0, 5, 0));
    let arm_cond_target = b.addr();
    b.patch_arm(j_cond_br, b_cond(EQ, j_cond_br, arm_cond_target));

    // Memory ops (LDR/STR/LDRB/STRH/LDRD/STRD/LDMIA/STMDB/LDREX/STREX/CLREX)
    b.arm(0xE580_2000); // str r2, [r0]
    b.arm(0xE590_3000); // ldr r3, [r0]
    b.arm(0xE1C0_20B2); // strh r2, [r0, #2]
    b.arm(0xE1D0_40F2); // ldrsh r4, [r0, #2]
    b.arm(0xE1C0_20F0); // strd r2, [r0]
    b.arm(0xE1C0_40D0); // ldrd r4, [r0]
    b.arm(0xE92D_4030); // push {r4, r5, lr}
    b.arm(0xE8BD_4030); // pop {r4, r5, lr}
    b.arm(0xE190_A99F); // ldrex r10, [r0]
    b.arm(0xE180_B993); // strex r11, r3, [r0]
    b.arm(0xF57F_F01F); // clrex
    b.arm(0xEE1D_0F70); // mrc p15, 0, r0, c13, c0, 3

    // DSP / Media / Parallel SIMD (exercises MediaParallel stub + inline saturation/byte ops)
    b.arm(0xE100_4051); // qadd r4, r1, r0
    b.arm(0xE140_5051); // qdadd r5, r1, r0
    b.arm(0xE107_1280); // smlabb r7, r0, r2, r1
    b.arm(0xE0C6_5291); // smlal r5, r6, r1, r2
    b.arm(0xE709_1210); // smlad r9, r0, r2, r1
    b.arm(0xE75A_1230); // smmlar r10, r0, r2, r1
    b.arm(0xE650_4F12); // uadd8 r4, r0, r2 (MediaParallel)
    b.arm(0xE680_5FB2); // sel r5, r0, r2
    b.arm(0xE6BF_9F30); // rev r9, r0
    b.arm(0xE6FF_AF30); // rbit r10, r0
    b.arm(0xE6ED_C110); // usat r12, #13, r0, lsl #2
    b.arm(0xE781_2210); // usada8 r1, r0, r2, r2

    // VFP scalar & short-vector (s8..s15 / d4..d7 -> VfpVector stub when emit_vfp_vectors=true)
    b.arm(vfp_dp(AL, false, 0b0011, false, 2, 0, 1)); // vadd.f32 s2, s0, s1 (scalar)
    b.arm(vfp_dp(AL, false, 0b0011, false, 8, 9, 10)); // vadd.f32 s8, s9, s10 (vector check!)
    b.arm(vfp_dp(AL, true, 0b0011, true, 4, 5, 6)); // vsub.f64 d4, d5, d6 (vector check!)
    b.arm(vfp_dp(AL, false, 0b0000, false, 12, 13, 14)); // vmla.f32 s12, s13, s14 (vector check!)
    b.arm(vfp_ext(AL, false, 0b0001, true, 16 >> 1, 16 & 1, 17 >> 1, 17 & 1)); // vsqrt.f32 s16, s17
    b.arm(0xEEB4_3AC2); // vcmpe.f32 s6, s5
    b.arm(0xEEF1_FA10); // fmstat
    b.arm(0xEEFD_7AC2); // vcvt.s32.f32 s15, s5 (VfpToInt)
    b.arm(0xEEB8_7AC0); // vcvt.f32.s32 s14, s0
    b.arm(0xEE00_0A10); // vmov s0, r0
    b.arm(0xEE10_0A10); // vmov r0, s0
    b.arm(0xED2D_0A04); // vpush {s0-s3}
    b.arm(0xECBD_0A04); // vpop {s0-s3}
    b.arm(0xEEE1_0A10); // vmsr fpscr, r0

    // DirectCall (to overridden ARM + to Thumb), IndirectCall, Interpret fallback, conditional SVC, return
    let j_call_ovr = b.arm(0);
    b.patch_arm(j_call_ovr, bl_cond(AL, j_call_ovr, arm_ovr));
    let j_call_thumb = b.arm(0);
    b.patch_arm(j_call_thumb, blx_thumb(j_call_thumb, thumb_fn));
    b.arm(blx_reg(AL, 9)); // IndirectCall
    b.arm(0xE7F0_00F0); // UDF -> Interpret stub
    b.arm(0x0F00_0021); // svceq #0x21
    b.arm(0xE1A0_F00E); // mov pc, lr

    // Literal pool after function return (folded when fold_text_literals = true)
    let lit_word = b.arm(0x1234_5678);
    let lit_f32 = b.arm(3.25_f32.to_bits());
    let lit_f64_lo = b.arm(0x0000_0000);
    let _lit_f64_hi = b.arm(0x4004_0000);
    b.patch_arm(j_ldr_lit, ldr_pc_rel(AL, 0, false, j_ldr_lit, lit_word));
    b.patch_arm(j_vldr_s, vldr_pc_rel(AL, false, 0, j_vldr_s, lit_f32));
    b.patch_arm(j_vldr_d, vldr_pc_rel(AL, true, 1, j_vldr_d, lit_f64_lo));

    let program = Program {
        text: Segment { base: SYNTH_BASE, bytes: b.bytes },
        seeds: vec![
            (arm_main, Source::Entry),
            (arm_tail_ovr, Source::Entry),
            (arm_ovr, Source::Override),
            (thumb_fn | 1, Source::Call),
            (thumb_ovr | 1, Source::Override),
        ],
        slots: None,
    };

    let options = CompileOptions { fold_text_literals: true, emit_vfp_vectors: true };
    let overrides = [arm_ovr, thumb_ovr | 1];
    let image = compile_program(0xCAFE_BABE, 0x1234_5678, &program, &overrides, options, |_, _| {});
    let stats = audit_aot_image(&image);
    println!("Synthetic x64 AOT audit passed: {stats:?}");
    assert!(stats.total_insns > 100);
    assert!(stats.total_branches > 10);
    assert!(stats.total_host_calls >= 10);
}

// ---------------------------------------------------------------------------
// Full-ROM Static Disassembly & ABI Audit Test
// ---------------------------------------------------------------------------

fn find_local_rom() -> Option<PathBuf> {
    std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../roms"))
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("cci") || e.eq_ignore_ascii_case("3ds")))
}

#[test]
fn audit_full_kiu_rom_x64_disassembly_and_abi_invariants() {
    let Some(rom_path) = find_local_rom() else {
        eprintln!("No test ROM in roms/, skipping full-ROM x64 static audit");
        return;
    };

    let title = zakuro_fs::Title::load(&rom_path).expect("load test ROM");
    let code = title.code().expect("decompress code.bin");

    for (label, options) in [
        ("default (emit_vfp_vectors=true)", CompileOptions::default()),
        ("kiu_prod (emit_vfp_vectors=false)", CompileOptions { fold_text_literals: true, emit_vfp_vectors: false }),
    ] {
        let t_compile = Instant::now();
        let image = compile_title(
            title.program_id(),
            &title.exheader,
            &code,
            &KIU_OVERRIDES,
            &[],
            options,
            |_, _| {},
        );
        let compile_elapsed = t_compile.elapsed();

        let t_audit = Instant::now();
        let stats = audit_aot_image(&image);
        let audit_elapsed = t_audit.elapsed();

        println!(
            "Full KIU ROM x64 static audit [{label}] PASSED in {:.2?} (compile {:.2?}): {stats:?}",
            audit_elapsed, compile_elapsed
        );
        assert!(stats.total_bytes > 50 * 1024 * 1024, "expected full KIU AOT image > 50 MiB");
        assert!(stats.total_insns > 1_000_000);
        assert!(stats.total_entries > 100_000);
        assert_eq!(
            stats.total_host_calls,
            11 + KIU_OVERRIDES.len() as u64,
            "expected 11 runtime stub host calls + {} override wrapper host calls",
            KIU_OVERRIDES.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Full-ROM Parallel All-Entries Differential Sweep vs zakuro-cpu
// ---------------------------------------------------------------------------

const PAGE_SIZE: usize = 0x1000;
const RETURN: u32 = 0xFFFF_FFF0;
const STACK_TOP: u32 = 0x1000_0000;
const STACK_SIZE: u32 = 0x4_0000;
const HEAP: u32 = 0x0800_0000;
const HEAP_SIZE: u32 = 0x4_0000;
const TLS: u32 = 0x1FF8_2000;

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

// SAFETY: `read_pages` and `write_pages` point exclusively into `self.regions` buffers
// that are never reallocated after `TestMemory::new`.
unsafe impl Send for TestMemory {}

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

    fn copy_from_regions(&mut self, src_regions: &[Region]) {
        for (dst, src) in self.regions.iter_mut().zip(src_regions) {
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
        let was_thumb = ctx.thumb != 0;
        load_ctx(ctx, &mut run.cpu);
        run.cpu.regs[15] = address;
        let exit = run.cpu.step(&mut run.memory);
        store_ctx(&run.cpu, ctx);
        run.fallbacks += 1;
        let step = if was_thumb { 2 } else { 4 };
        if exit.is_some() || (ctx.thumb != 0) != was_thumb || ctx.r[15] != address.wrapping_add(step) {
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

fn read_host_mxcsr() -> u32 {
    let mut mxcsr = 0u32;
    unsafe {
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
    }
    mxcsr
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
                let mxcsr_before = read_host_mxcsr();
                aot.enter(&mut *ctx, target_rx);
                let mxcsr_after = read_host_mxcsr();
                assert_eq!(
                    mxcsr_before, mxcsr_after,
                    "host MXCSR leaked across aot.enter (before={mxcsr_before:#010X}, after={mxcsr_after:#010X})"
                );
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

struct WorkerState {
    aot: LoadedAot,
    a: Run,
    b: Run,
}

#[ignore]
#[test]
fn full_rom_all_entries_differential_sweep() {
    let Some(rom_path) = find_local_rom() else {
        eprintln!("No test ROM in roms/, skipping full-ROM differential sweep");
        return;
    };

    let t0 = Instant::now();
    let title = zakuro_fs::Title::load(&rom_path).expect("load test ROM");
    let code = title.code().expect("decompress code.bin");
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

    let text_end = title.exheader.text.address + title.exheader.text.size;
    let shared_exec = Arc::new(ExecBuffer::exec_copy(pure_image.code_bytes()).expect("exec ROM AOT"));

    let make_worker_aot = |image: &AotImage, exec: &Arc<ExecBuffer>| -> LoadedAot {
        let exec_ref = Arc::clone(exec);
        let rx_ptr = exec_ref.rx_ptr();
        let len = exec_ref.len();
        let exec_view = unsafe {
            ExecBuffer::from_external(rx_ptr, len, Some(Box::new(move || drop(exec_ref))))
        };
        LoadedAot::new(
            image.program_id,
            image.text_hash,
            image.enter_offset,
            exec_view,
            image.entries.clone(),
            OriginData {
                pieces: Vec::new(),
                functions: image.origins.functions.clone(),
                runs: Vec::new(),
                owners: image.origins.owners.clone(),
            },
        )
    };

    // Collect ALL word-aligned entries in `.text` where `aot.has_code(addr)`.
    let probe_aot = make_worker_aot(&pure_image, &shared_exec);
    let mut func_starts = Vec::new();
    let mut addr = title.exheader.text.address;
    while addr < text_end {
        if probe_aot.has_code(addr) {
            func_starts.push(addr);
        }
        addr += 4;
    }
    drop(probe_aot);
    println!(
        "Full-ROM differential sweep: compiled in {:.2?}, sweeping ALL {} word-aligned `.text` entries across {} Rayon threads",
        t0.elapsed(),
        func_starts.len(),
        rayon::current_num_threads(),
    );
    assert!(func_starts.len() > 100_000, "expected >100k entries, got {}", func_starts.len());

    // Build pristine memory template matching `verify.rs`.
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
    let mut data_bytes = data_seg.bytes;
    data_bytes.resize(data_bytes.len() + title.exheader.bss_size as usize, 0);
    let mut pristine_regions = vec![
        Region { base: text_seg.base, bytes: text_seg.bytes, writable: false, direct_page: true },
        Region { base: ro_seg.base, bytes: ro_seg.bytes, writable: false, direct_page: true },
        Region { base: data_seg.base, bytes: data_bytes, writable: true, direct_page: true },
        Region { base: HEAP, bytes: heap, writable: true, direct_page: true },
        Region { base: STACK_TOP - STACK_SIZE, bytes: vec![0; STACK_SIZE as usize], writable: true, direct_page: true },
        Region { base: TLS, bytes: vec![0; PAGE_SIZE], writable: true, direct_page: true },
    ];
    for r in &mut pristine_regions {
        r.bytes.resize(r.bytes.len().next_multiple_of(PAGE_SIZE), 0);
    }

    let pool: Mutex<Vec<WorkerState>> = Mutex::new(
        (0..rayon::current_num_threads().max(1))
            .map(|_| WorkerState {
                aot: make_worker_aot(&pure_image, &shared_exec),
                a: Run {
                    memory: TestMemory::new(pristine_regions.clone()),
                    cpu: Cpu::new(),
                    pending: None,
                    interpreted: 0,
                    fallbacks: 0,
                },
                b: Run {
                    memory: TestMemory::new(pristine_regions.clone()),
                    cpu: Cpu::new(),
                    pending: None,
                    interpreted: 0,
                    fallbacks: 0,
                },
            })
            .collect(),
    );

    let tested = AtomicUsize::new(0);
    let returned = AtomicUsize::new(0);
    let both_limit = AtomicUsize::new(0);
    let one_limit = AtomicUsize::new(0);
    let mismatches = AtomicUsize::new(0);
    let total_fallbacks = AtomicUsize::new(0);
    let sample_mismatches: Mutex<Vec<String>> = Mutex::new(Vec::new());

    let indexed: Vec<(usize, u32)> = func_starts.iter().copied().enumerate().collect();
    indexed.par_chunks(128).for_each(|chunk| {
        let mut ws = loop {
            if let Some(w) = pool.lock().unwrap().pop() {
                break w;
            }
            std::thread::yield_now();
        };

        let mut local_tested = 0usize;
        let mut local_returned = 0usize;
        let mut local_both_limit = 0usize;
        let mut local_one_limit = 0usize;
        let mut local_mismatches = 0usize;
        let mut local_fallbacks = 0usize;

        for &(idx, fn_addr) in chunk {
            ws.a.memory.copy_from_regions(&pristine_regions);
            ws.b.memory.copy_from_regions(&pristine_regions);
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
            ws.a.cpu = cpu.clone();
            ws.b.cpu = cpu.clone();
            ws.b.interpreted = 0;
            ws.b.fallbacks = 0;
            ws.b.pending = None;

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
                read_pages: ws.b.memory.read_pages.as_ptr(),
                write_pages: ws.b.memory.write_pages.as_ptr(),
                vfp: ws.b.cpu.vfp.regs.as_mut_ptr(),
                fpscr: &mut ws.b.cpu.vfp.fpscr,
                host: &HOST,
                user: &mut ws.b as *mut Run as *mut c_void,
            };
            store_ctx(&cpu, &mut ctx);

            let exp = run_interpreted(&mut ws.a, 5_000);
            let slice = if idx % 2 == 0 { 5_000 } else { 1 + (rnd() % 32) as i32 };
            let got = unsafe { run_aot(&ws.aot, &mut ws.b, &mut ctx, slice, 5_000) };
            local_tested += 1;
            local_fallbacks += ws.b.fallbacks as usize;

            if exp == Stop::Limit || got == Stop::Limit {
                if exp == Stop::Limit && got == Stop::Limit {
                    local_both_limit += 1;
                } else {
                    local_one_limit += 1;
                }
                continue;
            }
            let mut diffs = compare_runs(&ws.a, &ws.b, &ctx);
            if exp != got {
                diffs.insert(0, format!("stop: interp={exp:?} aot={got:?}"));
            }
            if !diffs.is_empty() {
                local_mismatches += 1;
                let mut guard = sample_mismatches.lock().unwrap();
                if guard.len() < 25 {
                    guard.push(format!("0x{fn_addr:08X}: {}", diffs.join(", ")));
                }
            } else if exp == Stop::Returned {
                local_returned += 1;
            }
        }

        pool.lock().unwrap().push(ws);

        tested.fetch_add(local_tested, Ordering::Relaxed);
        returned.fetch_add(local_returned, Ordering::Relaxed);
        both_limit.fetch_add(local_both_limit, Ordering::Relaxed);
        one_limit.fetch_add(local_one_limit, Ordering::Relaxed);
        mismatches.fetch_add(local_mismatches, Ordering::Relaxed);
        total_fallbacks.fetch_add(local_fallbacks, Ordering::Relaxed);
    });

    let total_tested = tested.load(Ordering::Relaxed);
    let total_returned = returned.load(Ordering::Relaxed);
    let total_both_limit = both_limit.load(Ordering::Relaxed);
    let total_one_limit = one_limit.load(Ordering::Relaxed);
    let total_mismatches = mismatches.load(Ordering::Relaxed);
    let fallbacks = total_fallbacks.load(Ordering::Relaxed);

    for m in sample_mismatches.lock().unwrap().iter() {
        println!("MISMATCH: {m}");
    }
    println!(
        "Full-ROM all-entries differential sweep in {:.2?}: tested={total_tested}, returned={total_returned}, both_limit={total_both_limit}, one_limit={total_one_limit}, fallbacks={fallbacks}, mismatches={total_mismatches}",
        t0.elapsed(),
    );
    assert_eq!(
        total_mismatches, 0,
        "{total_mismatches} of {total_tested} ROM entries mismatched zakuro-cpu!"
    );
}
