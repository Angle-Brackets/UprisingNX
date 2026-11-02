//! the vertex and geometry shaders run over several vertices or primitives
//! at once, one lane each, so
//! that every instruction works through all of them together with vector
//! instructions. a lane computes exactly what the interpreter computes for
//! its vertex, operation for operation, and the rare operations run lane by
//! lane through the interpreter's own arithmetic. where the vertices branch
//! apart, the ones going the other way carry on in a copy of the batch.

use wide::f32x8;

use super::isa::OpCode;
use super::{Block, Op, Operand, Program, ShaderUnit, Vec4};
use super::{FLOAT_UNIFORMS, INPUT_REGISTERS, OUTPUT_REGISTERS, PROGRAM_SIZE, TEMP_REGISTERS};

/// how many vertices a batch shades together.
pub const LANES: usize = 8;

/// one component of a register, for every lane.
type Lanes = f32x8;
/// a register for every lane, a row of lanes per component.
type Wide = [Lanes; 4];

const ZERO: Wide = [f32x8::ZERO; 4];

/// super::TEMP_START in every lane.
const TEMP_START: Wide = [f32x8::ZERO, f32x8::ZERO, f32x8::ZERO, f32x8::ONE];

/// the float uniforms in every lane, put there once for all of a draw's
/// batches instead of at every read.
pub(super) struct Uniforms([Wide; FLOAT_UNIFORMS]);

impl Uniforms {
    pub(super) fn new(unit: &ShaderUnit) -> Uniforms {
        Uniforms(unit.float_uniforms.map(|uniform| uniform.map(f32x8::splat)))
    }

    /// a uniform, none past the last one.
    #[inline(always)]
    fn at(&self, index: i32) -> &Wide {
        match usize::try_from(index).ok().and_then(|index| self.0.get(index)) {
            Some(value) => value,
            None => &ZERO,
        }
    }
}

/// a batch of vertices' registers.
#[derive(Clone)]
pub(super) struct Batch {
    input: [Wide; INPUT_REGISTERS],
    output: [Wide; OUTPUT_REGISTERS],
    temp: [Wide; TEMP_REGISTERS],
    /// a0 and a1 for every lane, the loop counter is the same for all.
    address: [[i32; LANES]; 2],
    loop_counter: i32,
    condition: [[bool; LANES]; 2],
}

impl Batch {
    pub(super) fn new() -> Self {
        Self {
            input: [ZERO; INPUT_REGISTERS],
            output: [ZERO; OUTPUT_REGISTERS],
            temp: [TEMP_START; TEMP_REGISTERS],
            address: [[0; LANES]; 2],
            loop_counter: 0,
            condition: [[false; LANES]; 2],
        }
    }

    #[inline(always)]
    pub(super) fn reset(&mut self, program: &Program, inputs: &[[Vec4; INPUT_REGISTERS]]) {
        for register in (0..OUTPUT_REGISTERS).filter(|r| program.outputs & (1 << r) != 0) {
            self.output[register] = ZERO;
        }
        for register in (0..TEMP_REGISTERS).filter(|r| program.temps & (1 << r) != 0) {
            self.temp[register] = TEMP_START;
        }
        self.address = [[0; LANES]; 2];
        self.loop_counter = 0;
        self.condition = [[false; LANES]; 2];
        let last = inputs.len() - 1;
        for register in (0..INPUT_REGISTERS).filter(|register| program.inputs & (1 << register) != 0) {
            let r0 = std::array::from_fn(|lane| inputs[lane.min(last)][register][0]);
            let r1 = std::array::from_fn(|lane| inputs[lane.min(last)][register][1]);
            let r2 = std::array::from_fn(|lane| inputs[lane.min(last)][register][2]);
            let r3 = std::array::from_fn(|lane| inputs[lane.min(last)][register][3]);
            self.input[register] = [f32x8::from(r0), f32x8::from(r1), f32x8::from(r2), f32x8::from(r3)];
        }
    }
}

/// every lane, one bit each.
const ALL: u8 = u8::MAX;

/// the output registers of lanes that went their own way at a branch.
type Fork = (u8, [Wide; OUTPUT_REGISTERS]);

/// one emitted triangle, as each vertex's output registers.
pub(super) type Triangle = [[Vec4; OUTPUT_REGISTERS]; 3];

/// what the lanes of a geometry shader emit, each its own vertices and
/// triangles, the way the interpreter's emitter keeps them for one.
pub(super) struct Emitters {
    slot: [usize; LANES],
    completes: [bool; LANES],
    slots: [Triangle; LANES],
    pub(super) triangles: [Vec<Triangle>; LANES],
}

impl Emitters {
    pub(super) fn new() -> Emitters {
        Emitters {
            slot: [0; LANES],
            completes: [false; LANES],
            slots: [[[[0.0; 4]; OUTPUT_REGISTERS]; 3]; LANES],
            triangles: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// shades up to LANES vertices, writing their output registers.
pub(super) fn run(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    outputs: &mut [[Vec4; OUTPUT_REGISTERS]],
    batch: &mut Batch,
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
) {
    debug_assert!(!inputs.is_empty() && inputs.len() <= LANES);
    batch.reset(program, inputs);
    blocks.clear();
    forks.clear();
    let finished = execute(unit, uniforms, program, batch, blocks, forks, unit.entry_point, 0, ALL, None);
    if forks.is_empty() && finished == ALL {
        for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
            let c0 = batch.output[register][0].to_array();
            let c1 = batch.output[register][1].to_array();
            let c2 = batch.output[register][2].to_array();
            let c3 = batch.output[register][3].to_array();
            for (lane, output) in outputs.iter_mut().enumerate() {
                output[register] = [c0[lane], c1[lane], c2[lane], c3[lane]];
            }
        }
        return;
    }
    forks.push((finished, batch.output));
    // each lane's outputs from the copy it finished in. the registers
    // nothing writes stay zero, as they come
    if forks.len() == 1 && forks[0].0 == ALL {
        let registers = &forks[0].1;
        for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
            for component in 0..4 {
                let values = registers[register][component].to_array();
                for (lane, output) in outputs.iter_mut().enumerate() {
                    output[register][component] = values[lane];
                }
            }
        }
    } else {
        for (lanes, registers) in forks.iter() {
            for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
                for component in 0..4 {
                    let values = registers[register][component].to_array();
                    for (lane, output) in outputs.iter_mut().enumerate() {
                        if lanes & (1 << lane) != 0 {
                            output[register][component] = values[lane];
                        }
                    }
                }
            }
        }
    }
}

/// runs the geometry shader over up to LANES primitives' inputs, each
/// lane's triangles going in its emitter. lanes past the last primitive
/// repeat it and what they emit is theirs to drop.
pub(super) fn run_geometry(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    inputs: &[[Vec4; INPUT_REGISTERS]],
    batch: &mut Batch,
    emitters: &mut Emitters,
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
) {
    debug_assert!(!inputs.is_empty() && inputs.len() <= LANES);
    batch.reset(program, inputs);
    blocks.clear();
    forks.clear();
    execute(unit, uniforms, program, batch, blocks, forks, unit.entry_point, 0, ALL, Some(emitters));
}

/// each active lane's output registers into its emitter's slot, a
/// triangle out when the slot completes one.
fn emit(batch: &Batch, program: &Program, active: u8, emitters: &mut Emitters) {
    for register in (0..OUTPUT_REGISTERS).filter(|register| program.outputs & (1 << register) != 0) {
        for component in 0..4 {
            let values = batch.output[register][component].to_array();
            for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
                let slot = emitters.slot[lane];
                emitters.slots[lane][slot][register][component] = values[lane];
            }
        }
    }
    for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
        if emitters.completes[lane] {
            let triangle = emitters.slots[lane];
            emitters.triangles[lane].push(triangle);
        }
    }
}

/// runs the program over the batch from pc on, for the lanes in active,
/// which all go the same way. at a branch they part at, the ones going the
/// other way carry on in a copy of the batch, whose outputs go in forks.
/// the lanes still here at the end come back.
#[allow(clippy::too_many_arguments)]
fn execute(
    unit: &ShaderUnit,
    uniforms: &Uniforms,
    program: &Program,
    batch: &mut Batch,
    blocks: &mut Vec<Block>,
    forks: &mut Vec<Fork>,
    mut pc: u32,
    mut budget: u32,
    mut active: u8,
    mut emitters: Option<&mut Emitters>,
) -> u8 {
    // where uniforms go to be read like registers
    let mut scratch = [ZERO; 3];

    loop {
        if !blocks.is_empty() {
            while let Some(top) = blocks.last_mut() {
                if pc != top.end {
                    break;
                }
                batch.loop_counter = batch.loop_counter.wrapping_add(top.increment);
                if top.repeat == 0 {
                    pc = top.return_address;
                    blocks.pop();
                } else {
                    top.repeat -= 1;
                    pc = top.start;
                }
            }
        }
        // a program that never ends is a decoding bug, the interpreter caps
        // it and so does this
        if pc as usize >= PROGRAM_SIZE || budget >= 0x10000 {
            break;
        }
        budget += 1;

        let op = &program.ops[pc as usize];
        match op.opcode {
            OpCode::End => break,
            OpCode::Nop => pc += 1,
            // what a vertex shader runs has nothing to emit to
            OpCode::SetEmit => {
                if let Some(emitters) = emitters.as_deref_mut() {
                    let (slot, completes) = (op.instruction.emit_vertex().min(2), op.instruction.emit_primitive());
                    for lane in (0..LANES).filter(|lane| active & (1 << lane) != 0) {
                        emitters.slot[lane] = slot;
                        emitters.completes[lane] = completes;
                    }
                }
                pc += 1;
            }
            OpCode::Emit => {
                if let Some(emitters) = emitters.as_deref_mut() {
                    emit(batch, program, active, emitters);
                }
                pc += 1;
            }
            OpCode::Mad | OpCode::MadI => {
                let [sa, sb, sc] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                let c = operand(unit, uniforms, batch, &op.sources[2], sc);
                let mask = op.mask;
                if let Some(slot) = slot_mut(batch, op.destination) {
                    if mask == 0b1111 {
                        slot[0] = multiply(a.r0, b.r0) + c.r0;
                        slot[1] = multiply(a.r1, b.r1) + c.r1;
                        slot[2] = multiply(a.r2, b.r2) + c.r2;
                        slot[3] = multiply(a.r3, b.r3) + c.r3;
                    } else {
                        if mask & 0b1000 != 0 { slot[0] = multiply(a.r0, b.r0) + c.r0; }
                        if mask & 0b0100 != 0 { slot[1] = multiply(a.r1, b.r1) + c.r1; }
                        if mask & 0b0010 != 0 { slot[2] = multiply(a.r2, b.r2) + c.r2; }
                        if mask & 0b0001 != 0 { slot[3] = multiply(a.r3, b.r3) + c.r3; }
                    }
                }
                pc += 1;
            }
            OpCode::Dp4 => {
                let mask = op.mask;
                let [sa, sb, _] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                if let Some(slot) = slot_mut(batch, op.destination) {
                    let d = dot(&a, &b, 4);
                    if mask & 0b1000 != 0 { slot[0] = d; }
                    if mask & 0b0100 != 0 { slot[1] = d; }
                    if mask & 0b0010 != 0 { slot[2] = d; }
                    if mask & 0b0001 != 0 { slot[3] = d; }
                }
                pc += 1;
            }
            OpCode::Dp3 => {
                let mask = op.mask;
                let [sa, sb, _] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                if let Some(slot) = slot_mut(batch, op.destination) {
                    let d = dot(&a, &b, 3);
                    if mask & 0b1000 != 0 { slot[0] = d; }
                    if mask & 0b0100 != 0 { slot[1] = d; }
                    if mask & 0b0010 != 0 { slot[2] = d; }
                    if mask & 0b0001 != 0 { slot[3] = d; }
                }
                pc += 1;
            }
            OpCode::Mul => {
                let mask = op.mask;
                let [sa, sb, _] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                if let Some(slot) = slot_mut(batch, op.destination) {
                    if mask == 0b1111 {
                        slot[0] = multiply(a.r0, b.r0);
                        slot[1] = multiply(a.r1, b.r1);
                        slot[2] = multiply(a.r2, b.r2);
                        slot[3] = multiply(a.r3, b.r3);
                    } else {
                        if mask & 0b1000 != 0 { slot[0] = multiply(a.r0, b.r0); }
                        if mask & 0b0100 != 0 { slot[1] = multiply(a.r1, b.r1); }
                        if mask & 0b0010 != 0 { slot[2] = multiply(a.r2, b.r2); }
                        if mask & 0b0001 != 0 { slot[3] = multiply(a.r3, b.r3); }
                    }
                }
                pc += 1;
            }
            OpCode::Add => {
                let mask = op.mask;
                let [sa, sb, _] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                let b = operand(unit, uniforms, batch, &op.sources[1], sb);
                if let Some(slot) = slot_mut(batch, op.destination) {
                    if mask == 0b1111 {
                        slot[0] = a.r0 + b.r0;
                        slot[1] = a.r1 + b.r1;
                        slot[2] = a.r2 + b.r2;
                        slot[3] = a.r3 + b.r3;
                    } else {
                        if mask & 0b1000 != 0 { slot[0] = a.r0 + b.r0; }
                        if mask & 0b0100 != 0 { slot[1] = a.r1 + b.r1; }
                        if mask & 0b0010 != 0 { slot[2] = a.r2 + b.r2; }
                        if mask & 0b0001 != 0 { slot[3] = a.r3 + b.r3; }
                    }
                }
                pc += 1;
            }
            OpCode::Mov => {
                let mask = op.mask;
                let [sa, _, _] = &mut scratch;
                let a = operand(unit, uniforms, batch, &op.sources[0], sa);
                if let Some(slot) = slot_mut(batch, op.destination) {
                    if mask == 0b1111 {
                        slot[0] = a.r0;
                        slot[1] = a.r1;
                        slot[2] = a.r2;
                        slot[3] = a.r3;
                    } else {
                        if mask & 0b1000 != 0 { slot[0] = a.r0; }
                        if mask & 0b0100 != 0 { slot[1] = a.r1; }
                        if mask & 0b0010 != 0 { slot[2] = a.r2; }
                        if mask & 0b0001 != 0 { slot[3] = a.r3; }
                    }
                }
                pc += 1;
            }
            OpCode::Loop => {
                let instruction = op.instruction;
                let integers = unit.int_uniforms[instruction.integer_index() as usize & 3];
                let (last, _) = instruction.flow_target();
                batch.loop_counter = integers[1] as i32;
                blocks.push(Block {
                    end: last + 1,
                    return_address: last + 1,
                    repeat: integers[0] as u32,
                    increment: integers[2] as i8 as i32,
                    start: pc + 1,
                    is_loop: true,
                });
                pc += 1;
            }
            OpCode::Call
            | OpCode::CallU
            | OpCode::CallC
            | OpCode::JmpC
            | OpCode::JmpU
            | OpCode::IfC
            | OpCode::IfU
            | OpCode::Break
            | OpCode::BreakC => {
                let taken = condition(unit, batch, op) & active;
                let others = active & !taken;
                if taken != 0 && others != 0 {
                    // the lanes that do not take it carry on in a copy
                    let mut copy = batch.clone();
                    let mut copy_blocks = blocks.clone();
                    let next = branch(op, pc, false, &mut copy_blocks);
                    let finished =
                        execute(unit, uniforms, program, &mut copy, &mut copy_blocks, forks, next, budget, others, emitters.as_deref_mut());
                    forks.push((finished, copy.output));
                    active = taken;
                }
                pc = branch(op, pc, taken != 0, blocks);
            }
            _ => {
                arithmetic(unit, uniforms, batch, op, &mut scratch);
                pc += 1;
            }
        }
    }
    active
}

/// where a flow instruction goes, taken or not, entering or leaving the
/// blocks that takes.
fn branch(op: &Op, pc: u32, taken: bool, blocks: &mut Vec<Block>) -> u32 {
    let (destination, count) = op.instruction.flow_target();
    let mut enter = |start: u32, count: u32, return_address: u32| {
        blocks.push(Block { end: start + count, return_address, repeat: 0, increment: 0, start, is_loop: false });
        start
    };
    match op.opcode {
        OpCode::Call | OpCode::CallU | OpCode::CallC if taken => enter(destination, count, pc + 1),
        OpCode::JmpC | OpCode::JmpU if taken => destination,
        OpCode::IfC | OpCode::IfU if taken => enter(pc + 1, destination - (pc + 1).min(destination), destination + count),
        OpCode::IfC | OpCode::IfU => enter(destination, count, destination + count),
        // leave the innermost loop, and any if or call inside it
        OpCode::Break | OpCode::BreakC if taken => {
            let mut next = pc + 1;
            while let Some(block) = blocks.pop() {
                if block.is_loop {
                    next = block.return_address;
                    break;
                }
            }
            next
        }
        _ => pc + 1,
    }
}

/// the shader's multiply, zero rather than NaN for zero times infinity.
/// a NaN is rare, so the lanes are only looked at when there is one.
#[inline(always)]
fn multiply(a: Lanes, b: Lanes) -> Lanes {
    let product = a * b;
    let nan = product.is_nan();
    if !nan.any() {
        return product;
    }
    let fixed = nan & !a.is_nan() & !b.is_nan();
    fixed.select(f32x8::ZERO, product)
}

/// every lane of a row through f, the interpreter's arithmetic one value
/// at a time, for the operations too rare to be worth doing wide.
#[inline(always)]
fn each(row: Lanes, f: impl Fn(f32) -> f32) -> Lanes {
    f32x8::from(row.to_array().map(f))
}

/// a float uniform, none past the last one.
#[inline(always)]
fn uniform(unit: &ShaderUnit, index: i32) -> Vec4 {
    if (0..FLOAT_UNIFORMS as i32).contains(&index) {
        unit.float_uniforms[index as usize]
    } else {
        super::ZERO
    }
}

/// a source as an instruction reads it, pre-swizzled and pre-negated into registers.
struct Source {
    r0: Lanes,
    r1: Lanes,
    r2: Lanes,
    r3: Lanes,
}

impl Source {
    #[inline(always)]
    fn row(&self, component: usize) -> Lanes {
        match component {
            0 => self.r0,
            1 => self.r1,
            2 => self.r2,
            _ => self.r3,
        }
    }
}

#[inline(always)]
fn slot_mut(batch: &mut Batch, register: u32) -> Option<&mut Wide> {
    match register {
        0x00..=0x0F => Some(&mut batch.output[register as usize]),
        0x10..=0x1F => Some(&mut batch.temp[(register - 0x10) as usize]),
        _ => None,
    }
}

/// a source's value in every lane, swizzled and negated as its descriptor
/// says. a uniform is the same in every lane unless an address register
/// offsets it, which puts it together in scratch.
#[inline(always)]
fn operand<'a>(unit: &ShaderUnit, uniforms: &'a Uniforms, batch: &'a Batch, operand: &Operand, scratch: &'a mut Wide) -> Source {
    let register = operand.register;
    let value: &'a Wide = match register {
        0x00..=0x0F => &batch.input[register as usize],
        0x10..=0x1F => &batch.temp[(register - 0x10) as usize],
        _ => {
            let base = register as i32 - 0x20;
            match operand.index {
                0 => uniforms.at(base),
                3 => uniforms.at(base.wrapping_add(batch.loop_counter)),
                index => {
                    let offsets = &batch.address[index as usize - 1];
                    let picked: [Vec4; LANES] = std::array::from_fn(|lane| uniform(unit, base.wrapping_add(offsets[lane])));
                    *scratch = std::array::from_fn(|component| f32x8::from(picked.map(|value| value[component])));
                    scratch
                }
            }
        }
    };
    let s = operand.swizzle;
    if s == [0, 1, 2, 3] && !operand.negate {
        return Source { r0: value[0], r1: value[1], r2: value[2], r3: value[3] };
    }
    let (mut r0, mut r1, mut r2, mut r3) = (
        value[s[0] as usize],
        value[s[1] as usize],
        value[s[2] as usize],
        value[s[3] as usize],
    );
    if operand.negate {
        r0 = -r0;
        r1 = -r1;
        r2 = -r2;
        r3 = -r3;
    }
    Source { r0, r1, r2, r3 }
}

#[inline(always)]
fn write(batch: &mut Batch, op: &Op, value: &Wide) {
    let Some(slot) = slot_mut(batch, op.destination) else { return };
    if op.mask == 0b1111 {
        *slot = *value;
        return;
    }
    // the mask's most significant bit selects x
    if op.mask & 0b1000 != 0 { slot[0] = value[0]; }
    if op.mask & 0b0100 != 0 { slot[1] = value[1]; }
    if op.mask & 0b0010 != 0 { slot[2] = value[2]; }
    if op.mask & 0b0001 != 0 { slot[3] = value[3]; }
}

/// a dot product over the first count components, in every lane.
#[inline(always)]
fn dot(a: &Source, b: &Source, count: usize) -> Lanes {
    let mut sum = multiply(a.r0, b.r0);
    if count > 1 {
        sum += multiply(a.r1, b.r1);
    }
    if count > 2 {
        sum += multiply(a.r2, b.r2);
    }
    if count > 3 {
        sum += multiply(a.r3, b.r3);
    }
    sum
}

/// true or false in every lane, as one and zero.
#[inline(always)]
fn select(mask: Lanes) -> Lanes {
    mask.select(f32x8::ONE, f32x8::ZERO)
}

#[inline(always)]
fn arithmetic(unit: &ShaderUnit, uniforms: &Uniforms, batch: &mut Batch, op: &Op, scratch: &mut [Wide; 3]) {
    let mask = op.mask;
    let [sa, sb, _] = scratch;
    let a = operand(unit, uniforms, batch, &op.sources[0], sa);
    let b = operand(unit, uniforms, batch, &op.sources[1], sb);
    match op.opcode {
        OpCode::Add => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                if mask == 0b1111 {
                    slot[0] = a.r0 + b.r0;
                    slot[1] = a.r1 + b.r1;
                    slot[2] = a.r2 + b.r2;
                    slot[3] = a.r3 + b.r3;
                } else {
                    if mask & 0b1000 != 0 { slot[0] = a.r0 + b.r0; }
                    if mask & 0b0100 != 0 { slot[1] = a.r1 + b.r1; }
                    if mask & 0b0010 != 0 { slot[2] = a.r2 + b.r2; }
                    if mask & 0b0001 != 0 { slot[3] = a.r3 + b.r3; }
                }
            }
        }
        OpCode::Mul => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                if mask == 0b1111 {
                    slot[0] = multiply(a.r0, b.r0);
                    slot[1] = multiply(a.r1, b.r1);
                    slot[2] = multiply(a.r2, b.r2);
                    slot[3] = multiply(a.r3, b.r3);
                } else {
                    if mask & 0b1000 != 0 { slot[0] = multiply(a.r0, b.r0); }
                    if mask & 0b0100 != 0 { slot[1] = multiply(a.r1, b.r1); }
                    if mask & 0b0010 != 0 { slot[2] = multiply(a.r2, b.r2); }
                    if mask & 0b0001 != 0 { slot[3] = multiply(a.r3, b.r3); }
                }
            }
        }
        OpCode::Mov => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                if mask == 0b1111 {
                    slot[0] = a.r0;
                    slot[1] = a.r1;
                    slot[2] = a.r2;
                    slot[3] = a.r3;
                } else {
                    if mask & 0b1000 != 0 { slot[0] = a.r0; }
                    if mask & 0b0100 != 0 { slot[1] = a.r1; }
                    if mask & 0b0010 != 0 { slot[2] = a.r2; }
                    if mask & 0b0001 != 0 { slot[3] = a.r3; }
                }
            }
        }
        OpCode::Dp3 => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                let d = dot(&a, &b, 3);
                if mask & 0b1000 != 0 { slot[0] = d; }
                if mask & 0b0100 != 0 { slot[1] = d; }
                if mask & 0b0010 != 0 { slot[2] = d; }
                if mask & 0b0001 != 0 { slot[3] = d; }
            }
        }
        OpCode::Dp4 => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                let d = dot(&a, &b, 4);
                if mask & 0b1000 != 0 { slot[0] = d; }
                if mask & 0b0100 != 0 { slot[1] = d; }
                if mask & 0b0010 != 0 { slot[2] = d; }
                if mask & 0b0001 != 0 { slot[3] = d; }
            }
        }
        OpCode::Dph | OpCode::DphI => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                let d = dot(&a, &b, 3) + b.r3;
                if mask & 0b1000 != 0 { slot[0] = d; }
                if mask & 0b0100 != 0 { slot[1] = d; }
                if mask & 0b0010 != 0 { slot[2] = d; }
                if mask & 0b0001 != 0 { slot[3] = d; }
            }
        }
        OpCode::Max => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                if mask & 0b1000 != 0 { slot[0] = a.r0.simd_gt(b.r0).select(a.r0, b.r0); }
                if mask & 0b0100 != 0 { slot[1] = a.r1.simd_gt(b.r1).select(a.r1, b.r1); }
                if mask & 0b0010 != 0 { slot[2] = a.r2.simd_gt(b.r2).select(a.r2, b.r2); }
                if mask & 0b0001 != 0 { slot[3] = a.r3.simd_gt(b.r3).select(a.r3, b.r3); }
            }
        }
        OpCode::Min => {
            if let Some(slot) = slot_mut(batch, op.destination) {
                if mask & 0b1000 != 0 { slot[0] = a.r0.simd_lt(b.r0).select(a.r0, b.r0); }
                if mask & 0b0100 != 0 { slot[1] = a.r1.simd_lt(b.r1).select(a.r1, b.r1); }
                if mask & 0b0010 != 0 { slot[2] = a.r2.simd_lt(b.r2).select(a.r2, b.r2); }
                if mask & 0b0001 != 0 { slot[3] = a.r3.simd_lt(b.r3).select(a.r3, b.r3); }
            }
        }
        OpCode::Flr => {
            let res = [each(a.r0, f32::floor), each(a.r1, f32::floor), each(a.r2, f32::floor), each(a.r3, f32::floor)];
            write(batch, op, &res);
        }
        OpCode::Rcp => {
            let res = [f32x8::ONE / a.r0; 4];
            write(batch, op, &res);
        }
        OpCode::Rsq => {
            let res = [f32x8::ONE / a.r0.sqrt(); 4];
            write(batch, op, &res);
        }
        OpCode::Ex2 => {
            let res = [each(a.r0, f32::exp2); 4];
            write(batch, op, &res);
        }
        OpCode::Lg2 => {
            let res = [each(a.r0, f32::log2); 4];
            write(batch, op, &res);
        }
        OpCode::Sge | OpCode::SgeI => {
            let res = [select(a.r0.simd_ge(b.r0)), select(a.r1.simd_ge(b.r1)), select(a.r2.simd_ge(b.r2)), select(a.r3.simd_ge(b.r3))];
            write(batch, op, &res);
        }
        OpCode::Slt | OpCode::SltI => {
            let res = [select(a.r0.simd_lt(b.r0)), select(a.r1.simd_lt(b.r1)), select(a.r2.simd_lt(b.r2)), select(a.r3.simd_lt(b.r3))];
            write(batch, op, &res);
        }
        OpCode::Dst | OpCode::DstI => {
            let res = [f32x8::ONE, multiply(a.r1, b.r1), a.r2, b.r3];
            write(batch, op, &res);
        }
        OpCode::LitP => {
            let (x, y, w) = (a.r0, a.r1, a.r3);
            let conditions = [x.to_array().map(|x| x >= 0.0), w.to_array().map(|w| w >= 0.0)];
            let result = [each(x, |x| x.max(0.0)), each(y, |y| y.clamp(-127.9961, 127.9961)), f32x8::ZERO, each(w, |w| w.max(0.0))];
            batch.condition = conditions;
            write(batch, op, &result);
        }
        OpCode::Mova => {
            let (x, y) = (a.r0.to_array(), a.r1.to_array());
            if op.mask & 0b1000 != 0 {
                batch.address[0] = x.map(|x| x as i32);
            }
            if op.mask & 0b0100 != 0 {
                batch.address[1] = y.map(|y| y as i32);
            }
        }
        OpCode::Cmp => {
            let modes = op.instruction.compare_modes();
            let conditions: [[bool; LANES]; 2] = std::array::from_fn(|component| {
                let (x, y) = (a.row(component).to_array(), b.row(component).to_array());
                std::array::from_fn(|lane| super::compare(modes[component], x[lane], y[lane]))
            });
            batch.condition = conditions;
        }
        other => {
            log::debug!("unimplemented shader opcode {other:?}");
        }
    }
}

/// the lanes a flow instruction goes its way in, one bit each.
fn condition(unit: &ShaderUnit, batch: &Batch, op: &Op) -> u8 {
    let instruction = op.instruction;
    let all = |taken: bool| if taken { ALL } else { 0 };
    match op.opcode {
        OpCode::Call | OpCode::Break => ALL,
        OpCode::CallU | OpCode::IfU => all(unit.bool_uniforms & (1 << instruction.bool_index()) != 0),
        // jmpu can test for either value, the low bit of its count inverts it
        OpCode::JmpU => {
            let set = unit.bool_uniforms & (1 << instruction.bool_index()) != 0;
            all(set == (instruction.flow_target().1 & 1 == 0))
        }
        _ => {
            let reference = instruction.condition_reference();
            let mut lanes = 0;
            for lane in 0..LANES {
                let x = batch.condition[0][lane] == reference[0];
                let y = batch.condition[1][lane] == reference[1];
                let taken = match instruction.condition_op() {
                    0 => x || y,
                    1 => x && y,
                    2 => x,
                    _ => y,
                };
                lanes |= (taken as u8) << lane;
            }
            lanes
        }
    }
}
