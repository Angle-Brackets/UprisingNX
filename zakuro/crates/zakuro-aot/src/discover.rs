//! Control-flow discovery, function container deduplication, and code-hash
//! piece segmentation for 3DS ARMv6K + Thumb-1 executables.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use recomp_abi::{CodeHash, Piece, Run, NO_ORIGIN};
use zakuro_fs::{CodeSetInfo, ExHeader};

const PAGE_SIZE: usize = 0x1000;
pub const ALWAYS: u32 = 0xE;
const PC: u32 = 15;
const LR: u32 = 14;
const SP: u32 = 13;
const HUGE: usize = 10_000;
const FAR: u32 = 0x4000;
pub const OWN_FUNCTION: usize = 128;

#[derive(Clone)]
pub struct Segment {
    pub base: u32,
    pub bytes: Vec<u8>,
}

impl Segment {
    pub fn end(&self) -> u32 {
        self.base.wrapping_add(self.bytes.len() as u32)
    }

    pub fn contains(&self, address: u32) -> bool {
        address >= self.base && address < self.end()
    }

    #[inline]
    pub fn read32(&self, address: u32) -> Option<u32> {
        let offset = address.checked_sub(self.base)? as usize;
        let bytes = self.bytes.get(offset..offset + 4)?;
        Some(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    #[inline]
    pub fn read16(&self, address: u32) -> Option<u16> {
        let offset = address.checked_sub(self.base)? as usize;
        let bytes = self.bytes.get(offset..offset + 2)?;
        Some(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    #[inline]
    pub fn read8(&self, address: u32) -> Option<u8> {
        let offset = address.checked_sub(self.base)? as usize;
        self.bytes.get(offset).copied()
    }

    pub fn hash(&self) -> u64 {
        let mut h = CodeHash::default();
        h.add(&self.bytes);
        h.value()
    }

    pub fn words(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.bytes
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .map(|(i, word)| (self.base + i as u32 * 4, u32::from_le_bytes(*word)))
    }
}

pub struct Image {
    pub entry: u32,
    pub text: Segment,
    pub rodata: Segment,
    pub data: Segment,
}

impl Image {
    pub fn from_exheader(header: &ExHeader, code: &[u8]) -> Image {
        let segment = |info: CodeSetInfo, offset: usize| Segment {
            base: info.address,
            bytes: code.get(offset..offset + info.size as usize).unwrap_or_default().to_vec(),
        };
        let rodata_offset = header.text.num_pages as usize * PAGE_SIZE;
        let data_offset = rodata_offset + header.rodata.num_pages as usize * PAGE_SIZE;
        Image {
            entry: header.text.address,
            text: segment(header.text, 0),
            rodata: segment(header.rodata, rodata_offset),
            data: segment(header.data, data_offset),
        }
    }

    pub fn into_program(self, overrides: &[u32], hints: &[u32]) -> Program {
        let entry = std::iter::once((self.entry, Source::Entry));
        let ovr = overrides.iter().map(|&a| (a, Source::Override));
        let hnt = hints.iter().map(|&a| (a, Source::Hint));
        let pointers = [&self.rodata, &self.data]
            .into_iter()
            .flat_map(Segment::words)
            .filter(|&(_, value)| self.text.contains(value & !1))
            .map(|(_, value)| (value, Source::Pointer));
        let seeds = entry.chain(ovr).chain(hnt).chain(pointers).collect();
        Program { text: self.text, seeds, slots: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    Entry,
    Call,
    Pointer,
    Scan,
    Override,
    Hint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Arm,
    Thumb,
}

impl Mode {
    fn other(self) -> Mode {
        match self {
            Mode::Arm => Mode::Thumb,
            Mode::Thumb => Mode::Arm,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub source: Source,
    pub mode: Mode,
    pub instructions: Vec<u32>,
    pub labels: BTreeSet<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Byte {
    Unknown,
    Code,
    Literal,
}

pub struct Analysis {
    pub functions: BTreeMap<u32, Function>,
    pub map: Vec<Byte>,
    pub indirect_sites: usize,
    pub jump_tables: usize,
    pub svc_sites: usize,
    pub dead_ends: usize,
}

pub struct Program {
    pub text: Segment,
    pub seeds: Vec<(u32, Source)>,
    pub slots: Option<BTreeSet<u32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    Branch { target: u32, link: bool },
    CallOtherMode { target: u32 },
    BranchRegister { register: u32, link: bool },
    Return,
    JumpTable,
    AddressTable { index: u32 },
    IndirectJump,
    LoadPcLiteral { literal: u32 },
    LiteralLoad { literal: u32, size: u32 },
    Svc,
    Undefined,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub condition: u32,
    pub flow: Flow,
}

impl Instruction {
    pub fn is_conditional(self) -> bool {
        self.condition != ALWAYS
    }
}

fn sign_extend_24(value: u32) -> i32 {
    ((value << 8) as i32) >> 8
}

fn sign_extend(value: u32, bits: u32) -> i32 {
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

fn compare_immediate(word: u32) -> Option<(u32, u32)> {
    if word & 0x0FF0_F000 != 0x0350_0000 {
        return None;
    }
    let rotate = ((word >> 8) & 0xF) * 2;
    Some(((word >> 16) & 0xF, (word & 0xFF).rotate_right(rotate)))
}

pub fn decode_arm(word: u32, address: u32) -> Instruction {
    let condition = word >> 28;
    let pc = address.wrapping_add(8);
    let rd = (word >> 12) & 0xF;
    let rn = (word >> 16) & 0xF;

    let flow = if condition == 0xF {
        if word & 0x0E00_0000 == 0x0A00_0000 {
            let half = (word >> 24) & 1;
            let offset = (sign_extend_24(word & 0x00FF_FFFF) << 2) + (half << 1) as i32;
            Flow::CallOtherMode { target: pc.wrapping_add(offset as u32) }
        } else {
            Flow::Other
        }
    } else if word & 0x0E00_0000 == 0x0A00_0000 {
        let offset = sign_extend_24(word & 0x00FF_FFFF) << 2;
        Flow::Branch { target: pc.wrapping_add(offset as u32), link: word & (1 << 24) != 0 }
    } else if word & 0x0FFF_FFD0 == 0x012F_FF10 {
        let register = word & 0xF;
        let link = word & (1 << 5) != 0;
        if register == LR && !link {
            Flow::Return
        } else {
            Flow::BranchRegister { register, link }
        }
    } else if word & 0x0F00_0000 == 0x0F00_0000 {
        Flow::Svc
    } else if word & 0x0C00_0000 == 0x0400_0000 {
        arm_single_transfer(word, pc, rd, rn)
    } else if word & 0x0E00_0000 == 0x0800_0000 {
        let load = word & (1 << 20) != 0;
        if load && word & (1 << 15) != 0 {
            if rn == SP { Flow::Return } else { Flow::IndirectJump }
        } else {
            Flow::Other
        }
    } else if word & 0x0F30_0E00 == 0x0D10_0A00 && rn == PC {
        let offset = (word & 0xFF) * 4;
        let literal = if word & (1 << 23) != 0 { (pc & !3) + offset } else { (pc & !3) - offset };
        let size = if word & (1 << 8) != 0 { 8 } else { 4 };
        Flow::LiteralLoad { literal, size }
    } else if word & 0x0C00_0000 == 0 {
        arm_data_processing(word, rd, rn)
    } else {
        Flow::Other
    };
    Instruction { condition, flow }
}

fn arm_single_transfer(word: u32, pc: u32, rd: u32, rn: u32) -> Flow {
    let load = word & (1 << 20) != 0;
    let register_offset = word & (1 << 25) != 0;
    if !load {
        return Flow::Other;
    }
    if rd == PC && rn == PC && register_offset && word & 0x0170_0FF0 == 0x0110_0100 {
        return Flow::AddressTable { index: word & 0xF };
    }
    if rn == PC && !register_offset && word & (1 << 24) != 0 {
        let offset = word & 0xFFF;
        let literal = if word & (1 << 23) != 0 { pc + offset } else { pc - offset };
        let size = if word & (1 << 22) != 0 { 1 } else { 4 };
        return if rd == PC { Flow::LoadPcLiteral { literal } } else { Flow::LiteralLoad { literal, size } };
    }
    if rd == PC {
        if rn == SP && word & (1 << 24) == 0 { Flow::Return } else { Flow::IndirectJump }
    } else {
        Flow::Other
    }
}

fn arm_data_processing(word: u32, rd: u32, rn: u32) -> Flow {
    let immediate = word & (1 << 25) != 0;
    if !immediate && word & 0x90 == 0x90 {
        return Flow::Other;
    }
    let opcode = (word >> 21) & 0xF;
    let sets_flags = word & (1 << 20) != 0;
    if (8..=11).contains(&opcode) || rd != PC {
        return Flow::Other;
    }
    let rm = word & 0xF;
    let shift = (word >> 4) & 0xFF;
    match opcode {
        0xD if !immediate && rm == LR && shift == 0 => Flow::Return,
        0x4 if !immediate && rn == PC && shift == 0x10 && !sets_flags => Flow::JumpTable,
        _ => Flow::IndirectJump,
    }
}

pub fn decode_thumb(halfword: u16, next: Option<u16>, address: u32) -> (Instruction, u32) {
    let pc = address.wrapping_add(4);
    let always = |flow| Instruction { condition: ALWAYS, flow };
    let word = halfword as u32;

    match halfword >> 11 {
        0b11010 | 0b11011 => {
            let condition = (word >> 8) & 0xF;
            let flow = match condition {
                0xF => Flow::Svc,
                0xE => Flow::Undefined,
                _ => Flow::Branch { target: pc.wrapping_add((sign_extend(word & 0xFF, 8) << 1) as u32), link: false },
            };
            (Instruction { condition, flow }, 2)
        }
        0b11100 => {
            let target = pc.wrapping_add((sign_extend(word & 0x7FF, 11) << 1) as u32);
            (always(Flow::Branch { target, link: false }), 2)
        }
        0b11110 => {
            let high = sign_extend(word & 0x7FF, 11) << 12;
            let Some(second) = next else { return (always(Flow::Undefined), 2) };
            let low = ((second & 0x7FF) as i32) << 1;
            let target = pc.wrapping_add((high + low) as u32);
            match second >> 11 {
                0b11111 => (always(Flow::Branch { target, link: true }), 4),
                0b11101 => (always(Flow::CallOtherMode { target: target & !3 }), 4),
                _ => (always(Flow::Undefined), 2),
            }
        }
        0b11111 | 0b11101 => (always(Flow::Undefined), 2),
        0b01001 => {
            let literal = (pc & !3) + (word & 0xFF) * 4;
            (always(Flow::LiteralLoad { literal, size: 4 }), 2)
        }
        _ => (always(thumb_other(halfword)), 2),
    }
}

fn thumb_other(halfword: u16) -> Flow {
    if halfword & 0xFF00 == 0xBD00 {
        return Flow::Return;
    }
    if halfword & 0xFF07 == 0x4700 {
        let register = ((halfword >> 3) & 0xF) as u32;
        let link = halfword & 0x80 != 0;
        return if register == LR && !link { Flow::Return } else { Flow::BranchRegister { register, link } };
    }
    if halfword & 0xFC87 == 0x4487 {
        let source = ((halfword >> 3) & 0xF) as u32;
        let mov = halfword & 0x0300 == 0x0200;
        return if mov && source == LR { Flow::Return } else { Flow::IndirectJump };
    }
    Flow::Other
}

struct Discovery<'a> {
    text: &'a Segment,
    slots: Option<&'a BTreeSet<u32>>,
    analysis: Analysis,
    entries: Option<BTreeSet<u32>>,
    queue: VecDeque<(u32, Mode, Source)>,
    guesses: VecDeque<(u32, Mode, Source)>,
}

pub fn analyze(program: &Program) -> Analysis {
    let mut discovery = Discovery {
        text: &program.text,
        slots: program.slots.as_ref(),
        analysis: Analysis {
            functions: BTreeMap::new(),
            map: vec![Byte::Unknown; program.text.bytes.len()],
            indirect_sites: 0,
            jump_tables: 0,
            svc_sites: 0,
            dead_ends: 0,
        },
        entries: None,
        queue: VecDeque::new(),
        guesses: VecDeque::new(),
    };
    for &slot in program.slots.iter().flatten() {
        discovery.mark(slot, 4, Byte::Literal);
    }
    for &(address, source) in &program.seeds {
        discovery.code_pointer(address, source);
    }
    discovery.run();
    loop {
        let starts = discovery.gap_starts();
        if starts.is_empty() {
            break;
        }
        discovery.queue.extend(starts.into_iter().map(|address| (address, Mode::Arm, Source::Scan)));
        discovery.run();
    }
    discovery.split_huge();
    discovery.analysis
}

impl Discovery<'_> {
    fn split_huge(&mut self) {
        let huge: Vec<(u32, Mode)> = self
            .analysis
            .functions
            .iter()
            .filter(|(_, f)| f.instructions.len() > HUGE)
            .map(|(&entry, f)| (entry, f.mode))
            .collect();
        if huge.is_empty() {
            return;
        }
        let counts = (
            self.analysis.indirect_sites,
            self.analysis.jump_tables,
            self.analysis.svc_sites,
            self.analysis.dead_ends,
        );
        self.entries = Some(self.analysis.functions.keys().copied().collect());
        let mut huge = huge;
        for _ in 0..4 {
            for &(entry, mode) in &huge {
                let (instructions, labels) = self.explore(entry, mode);
                if let Some(function) = self.analysis.functions.get_mut(&entry) {
                    function.instructions = instructions;
                    function.labels = labels;
                }
            }
            self.run();
            let still: Vec<(u32, Mode)> = self
                .analysis
                .functions
                .iter()
                .filter(|(_, f)| f.instructions.len() > HUGE)
                .map(|(&entry, f)| (entry, f.mode))
                .collect();
            if still.is_empty() || still == huge {
                break;
            }
            huge = still;
        }
        self.entries = None;
        (
            self.analysis.indirect_sites,
            self.analysis.jump_tables,
            self.analysis.svc_sites,
            self.analysis.dead_ends,
        ) = counts;
    }

    fn run(&mut self) {
        while let Some((entry, mode, source)) = self.queue.pop_front().or_else(|| self.guesses.pop_front()) {
            if !self.analysis.functions.contains_key(&entry) {
                let (instructions, labels) = self.explore(entry, mode);
                self.analysis.functions.insert(entry, Function { source, mode, instructions, labels });
            }
        }
    }

    fn mark(&mut self, address: u32, size: u32, kind: Byte) {
        for byte in address..address + size {
            if let Some(offset) = byte.checked_sub(self.text.base) {
                if let Some(slot) = self.analysis.map.get_mut(offset as usize) {
                    if *slot != Byte::Code {
                        *slot = kind;
                    }
                }
            }
        }
    }

    fn byte(&self, address: u32) -> Byte {
        self.analysis.map[(address - self.text.base) as usize]
    }

    fn gap_starts(&self) -> Vec<u32> {
        let mut starts = Vec::new();
        let mut in_gap = false;
        let mut address = self.text.base.next_multiple_of(4);
        while let Some(word) = self.text.read32(address) {
            let offset = (address - self.text.base) as usize;
            let unknown = self.analysis.map[offset..offset + 4].iter().all(|&byte| byte == Byte::Unknown);
            if unknown && !self.analysis.functions.contains_key(&address) {
                let instruction = decode_arm(word, address);
                let push = word & 0xFFFF_0000 == 0xE92D_0000 && word & 0xFFFF != 0;
                let starts_gap = !in_gap && instruction.condition == ALWAYS && instruction.flow != Flow::Undefined;
                if push || starts_gap {
                    starts.push(address);
                }
            }
            in_gap = unknown;
            address += 4;
        }
        starts
    }

    fn may_point(&self, address: u32) -> bool {
        self.slots.is_none_or(|slots| slots.contains(&address))
    }

    fn code_pointer(&mut self, value: u32, source: Source) {
        let (address, mode) = if value & 1 != 0 {
            (value & !1, Mode::Thumb)
        } else if value & 3 == 0 {
            (value, Mode::Arm)
        } else {
            return;
        };
        if self.text.contains(address) && !self.analysis.functions.contains_key(&address) {
            if source == Source::Pointer {
                self.guesses.push_back((address, mode, source));
            } else {
                self.queue.push_back((address, mode, source));
            }
        }
    }

    fn call(&mut self, target: u32, mode: Mode) {
        if self.text.contains(target) && !self.analysis.functions.contains_key(&target) {
            self.queue.push_back((target, mode, Source::Call));
        }
    }

    fn decode(&self, address: u32, mode: Mode) -> Option<(Instruction, u32)> {
        match mode {
            Mode::Arm => Some((decode_arm(self.text.read32(address)?, address), 4)),
            Mode::Thumb => {
                let halfword = self.text.read16(address)?;
                Some(decode_thumb(halfword, self.text.read16(address + 2), address))
            }
        }
    }

    fn explore(&mut self, entry: u32, mode: Mode) -> (Vec<u32>, BTreeSet<u32>) {
        let mut blocks = vec![entry];
        let mut seen = BTreeSet::new();
        let mut code = BTreeSet::new();
        let mut labels = BTreeSet::new();

        while let Some(start) = blocks.pop() {
            labels.insert(start);
            let mut address = start;
            while self.text.contains(address) && seen.insert(address) {
                if self.byte(address) == Byte::Literal {
                    break;
                }
                let Some((instruction, size)) = self.decode(address, mode) else { break };
                if instruction.flow == Flow::Undefined {
                    self.analysis.dead_ends += 1;
                    break;
                }
                self.mark(address, size, Byte::Code);
                code.insert(address);
                let conditional = instruction.is_conditional();

                let returns = matches!(
                    instruction.flow,
                    Flow::Branch { link: true, .. }
                        | Flow::CallOtherMode { .. }
                        | Flow::BranchRegister { link: true, .. }
                        | Flow::Svc
                );
                if returns {
                    labels.insert(address + size);
                }
                match instruction.flow {
                    Flow::Branch { target, link: true } => self.call(target, mode),
                    Flow::CallOtherMode { target } => self.call(target, mode.other()),
                    Flow::Branch { target, link: false } => {
                        let far = target < entry || target.abs_diff(address) > FAR;
                        let tail_call = target != entry
                            && self.entries.as_ref().is_some_and(|entries| entries.contains(&target) || far);
                        if tail_call && self.text.contains(target) {
                            if let Some(entries) = self.entries.as_mut() {
                                entries.insert(target);
                            }
                            self.call(target, mode);
                        }
                        if self.text.contains(target) && !tail_call {
                            blocks.push(target);
                        }
                        if !conditional {
                            break;
                        }
                    }
                    Flow::BranchRegister { link: true, .. } => self.analysis.indirect_sites += 1,
                    Flow::BranchRegister { link: false, .. } | Flow::IndirectJump => {
                        self.analysis.indirect_sites += 1;
                        if !conditional {
                            break;
                        }
                    }
                    Flow::Return => {
                        if !conditional {
                            break;
                        }
                    }
                    Flow::JumpTable => {
                        self.branch_table(address, &mut blocks);
                        break;
                    }
                    Flow::AddressTable { index } => {
                        self.address_table(address, index, &mut blocks);
                        if !conditional {
                            break;
                        }
                    }
                    Flow::LoadPcLiteral { literal } => {
                        self.mark(literal, 4, Byte::Literal);
                        if let Some(target) = self.text.read32(literal).filter(|_| self.may_point(literal)) {
                            self.code_pointer(target, Source::Pointer);
                        }
                        if !conditional {
                            break;
                        }
                    }
                    Flow::LiteralLoad { literal, size } => {
                        if self.text.contains(literal) && self.byte(literal) != Byte::Code {
                            self.mark(literal, size, Byte::Literal);
                            if size == 4 && self.may_point(literal) {
                                if let Some(value) = self.text.read32(literal) {
                                    self.code_pointer(value, Source::Pointer);
                                }
                            }
                        }
                    }
                    Flow::Svc => self.analysis.svc_sites += 1,
                    Flow::Undefined | Flow::Other => {}
                }
                address += size;
            }
        }
        labels.retain(|label| code.contains(label));
        (code.into_iter().collect(), labels)
    }

    fn branch_table(&mut self, address: u32, blocks: &mut Vec<u32>) {
        self.analysis.jump_tables += 1;
        let mut entry = address + 4;
        while let Some(word) = self.text.read32(entry) {
            match decode_arm(word, entry) {
                Instruction { condition: ALWAYS, flow: Flow::Branch { link: false, .. } } => {
                    blocks.push(entry);
                    entry += 4;
                }
                _ => break,
            }
        }
    }

    fn address_table(&mut self, address: u32, index: u32, blocks: &mut Vec<u32>) {
        self.analysis.jump_tables += 1;
        let entries = address
            .checked_sub(4)
            .and_then(|previous| self.text.read32(previous))
            .and_then(compare_immediate)
            .filter(|&(register, _)| register == index)
            .map(|(_, limit)| limit as usize + 1);
        let table = address + 8;
        for i in 0..entries.unwrap_or(256) {
            let slot = table + i as u32 * 4;
            let Some(target) = self.text.read32(slot) else { break };
            if !self.text.contains(target & !1) || !self.may_point(slot) {
                break;
            }
            self.mark(slot, 4, Byte::Literal);
            if target & 3 == 0 {
                blocks.push(target);
            } else {
                self.code_pointer(target, Source::Pointer);
            }
        }
    }
}

pub fn recompiles(entry: u32, function: &Function) -> bool {
    function.labels.contains(&entry)
}

pub fn key(address: u32, mode: Mode) -> u32 {
    address | (mode == Mode::Thumb) as u32
}

/// Finds functions completely enclosed inside larger functions of the same mode.
pub fn containers(functions: &BTreeMap<u32, &Function>) -> BTreeMap<u32, u32> {
    let mut owners: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&entry, function) in functions {
        for &label in &function.labels {
            owners.entry(label).or_default().push(entry);
        }
    }
    let rank = |entry: u32| (functions[&entry].instructions.len(), std::cmp::Reverse(entry));
    let mut parent = BTreeMap::new();
    for (&entry, function) in functions {
        if function.instructions.len() <= OWN_FUNCTION {
            continue;
        }
        let contains = |other: u32| {
            let code = &functions[&other].instructions;
            function.instructions.iter().all(|address| code.binary_search(address).is_ok())
        };
        let best = owners[&entry]
            .iter()
            .copied()
            .filter(|&other| functions[&other].mode == function.mode && rank(other) > rank(entry))
            .filter(|&other| contains(other))
            .max_by_key(|&other| rank(other));
        if let Some(best) = best {
            parent.insert(entry, best);
        }
    }
    parent
        .keys()
        .map(|&entry| {
            let mut container = parent[&entry];
            while let Some(&next) = parent.get(&container) {
                container = next;
            }
            (entry, container)
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct FuncOrigin {
    pub first: u32,
    pub count: u32,
    pub start: u32,
}

/// Serialized origin tables for `check()` mod/cheat staleness detection.
#[derive(Clone, Default)]
pub struct OriginData {
    pub pieces: Vec<Piece>,
    pub functions: Vec<FuncOrigin>,
    pub runs: Vec<Run>,
    pub owners: Vec<u32>,
}

pub fn build_origins(
    text: &Segment,
    functions_spans: &[(u32, Vec<(u32, u32)>)],
    entry_owners: &[u32],
) -> OriginData {
    let merged_spans: Vec<Vec<(u32, u32)>> = functions_spans
        .iter()
        .map(|(_, raw)| coalesce_spans(raw))
        .collect();
    let piece_ranges = cut_pieces(merged_spans.iter().flatten().copied());
    if piece_ranges.is_empty() {
        return OriginData {
            pieces: Vec::new(),
            functions: functions_spans
                .iter()
                .map(|&(start, _)| FuncOrigin { first: 0, count: 0, start })
                .collect(),
            runs: Vec::new(),
            owners: entry_owners.to_vec(),
        };
    }

    let at = |address: u32| address.wrapping_sub(text.base) as usize;
    let pieces: Vec<Piece> = piece_ranges
        .iter()
        .map(|&(start, end)| {
            let mut hash = CodeHash::default();
            hash.add(text.bytes.get(at(start)..at(end)).unwrap_or_default());
            Piece { start, end, hash: hash.value() }
        })
        .collect();

    let mut functions = Vec::with_capacity(functions_spans.len());
    let mut runs = Vec::new();
    for (spans, &(entry, _)) in merged_spans.iter().zip(functions_spans.iter()) {
        functions.push(FuncOrigin { first: runs.len() as u32, count: spans.len() as u32, start: entry });
        for &(start, end) in spans {
            let first = piece_ranges.partition_point(|&(piece, _)| piece < start) as u32;
            let last = piece_ranges.partition_point(|&(piece, _)| piece < end) as u32;
            runs.push(Run { first, end: last });
        }
    }

    let _ = NO_ORIGIN;
    OriginData { pieces, functions, runs, owners: entry_owners.to_vec() }
}

fn coalesce_spans(raw: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted: Vec<(u32, u32)> = raw.iter().copied().filter(|&(s, e)| s < e).collect();
    sorted.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (s, e) in sorted {
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

fn cut_pieces(spans: impl Iterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let mut spans: Vec<(u32, u32)> = spans.collect();
    spans.sort_unstable();
    let cuts: BTreeSet<u32> = spans.iter().flat_map(|&(start, end)| [start, end]).collect();
    let mut pieces = Vec::new();
    let mut covered: Option<(u32, u32)> = None;
    let cut = |(start, end): (u32, u32), pieces: &mut Vec<(u32, u32)>| {
        let mut from = start;
        for &at in cuts.range(start + 1..end) {
            pieces.push((from, at));
            from = at;
        }
        pieces.push((from, end));
    };
    for (start, end) in spans {
        match covered {
            Some((from, to)) if start <= to => covered = Some((from, to.max(end))),
            _ => {
                if let Some(stretch) = covered {
                    cut(stretch, &mut pieces);
                }
                covered = Some((start, end));
            }
        }
    }
    if let Some(stretch) = covered {
        cut(stretch, &mut pieces);
    }
    pieces
}
