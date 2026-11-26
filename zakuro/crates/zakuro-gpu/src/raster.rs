//! software rasterization of PICA200 draw calls.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::registers::*;
use crate::lighting::{Lighting, Tables};

#[cfg(feature = "vulkan")]
pub(crate) mod hardware;
use crate::texture::TextureFormat;
use crate::shader::{self, ShaderUnit, Vec4};
use crate::{format::ColorFormat, GpuMemory};

/// reads a float24-encoded viewport register.
fn read_float24(registers: &[u32], index: usize) -> f32 {
    crate::shader::isa::decode_float24(registers[index])
}

/// decodes a GPUREG_*_LOC register into a physical address.
fn loc_register(registers: &[u32], index: usize) -> u32 {
    (registers[index] & 0x0FFF_FFFF) << 3
}

struct AttributeLoader {
    offset: u32,
    /// for each of the loader's up to twelve components, the attribute
    /// (0-11) it holds, or 12-15 for four to sixteen bytes of padding.
    components: u64,
    stride: u32,
    component_count: u32,
}

fn read_loaders(registers: &[u32]) -> [AttributeLoader; REG_ATTRIBUTE_LOADER_COUNT] {
    std::array::from_fn(|i| {
        let base = REG_ATTRIBUTE_LOADER + i * REG_ATTRIBUTE_LOADER_STRIDE;
        let word1 = registers[base + 1];
        let word2 = registers[base + 2];
        // the third word holds components 8-11 in its low half, the bytes
        // per vertex in bits [23:16] and the component count in [31:28].
        AttributeLoader {
            offset: registers[base] & 0x0FFF_FFFF,
            components: word1 as u64 | (((word2 & 0xFFFF) as u64) << 32),
            // bits 16-23, not 24-31. wasted hours on this shit
            stride: (word2 >> 16) & 0xFF,
            component_count: (word2 >> 28) & 0xF,
        }
    })
}

/// (type, component_count) for attribute slot, from the combined 64-bit
/// format register.
fn attribute_format(registers: &[u32], slot: u32) -> (u32, u32) {
    let combined =
        registers[REG_ATTRIBUTE_FORMAT_LOW] as u64 | ((registers[REG_ATTRIBUTE_FORMAT_HIGH] as u64) << 32);
    let nibble = ((combined >> (slot * 4)) & 0xF) as u32;
    (nibble & 0x3, (nibble >> 2) + 1)
}

/// byte size of one component of the given attribute type (0=byte, 1=ubyte,
/// 2=short, 3=float).
fn component_size(ty: u32) -> u32 {
    match ty {
        0 | 1 => 1,
        2 => 2,
        _ => 4,
    }
}

/// one component of an attribute, from its bytes, the way LoaderPlan::place
/// reads them.
#[cfg(test)]
fn component(bytes: &[u8], ty: u32) -> f32 {
    match ty {
        0 => bytes[0] as i8 as f32,
        1 => bytes[0] as f32,
        2 => i16::from_le_bytes([bytes[0], bytes[1]]) as f32,
        _ => f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
    }
}

/// where an attribute sits in a loader's vertex.
struct Field {
    id: usize,
    offset: u32,
    ty: u32,
    count: u32,
}

/// one attribute array, as a vertex of it is laid out.
struct LoaderLayout {
    /// the loader, 0 to 11.
    index: usize,
    offset: u32,
    stride: u32,
    /// the bytes of a vertex the fields take.
    size: u32,
    fields: Vec<Field>,
}

/// how the attribute arrays lay out a vertex, worked out once for a draw
/// rather than for every vertex.
struct VertexLayout {
    loaders: Vec<LoaderLayout>,
}

impl VertexLayout {
    fn read(registers: &[u32]) -> VertexLayout {
        let loaders = read_loaders(registers)
            .iter()
            .enumerate()
            .filter(|(_, loader)| loader.component_count > 0)
            .map(|(index, loader)| {
                let mut offset = 0u32;
                let mut fields = Vec::new();
                for slot in 0..loader.component_count.min(12) {
                    let id = ((loader.components >> (slot * 4)) & 0xF) as usize;
                    if id >= 12 {
                        // padding, aligned to a word, then 4, 8, 12 or 16 bytes.
                        offset = offset.next_multiple_of(4) + (id as u32 - 11) * 4;
                        continue;
                    }
                    let (ty, count) = attribute_format(registers, id as u32);
                    let size = component_size(ty);
                    // each attribute starts aligned to its component size.
                    offset = offset.next_multiple_of(size);
                    fields.push(Field { id, offset, ty, count });
                    offset += size * count;
                }
                LoaderLayout { index, offset: loader.offset, stride: loader.stride, size: offset, fields }
            })
            .collect();
        VertexLayout { loaders }
    }
}

/// a draw's vertices as the shader takes them, worked out once for the
/// draw: what each input register holds when no array feeds it, and the
/// fields the arrays do give, each going straight to its register.
struct InputPlan {
    template: [Vec4; shader::INPUT_REGISTERS],
    /// for each input register, the fixed attribute it takes, if one does.
    fixed: [Option<usize>; shader::INPUT_REGISTERS],
    loaders: Vec<LoaderPlan>,
}

/// the fields one array gives, and where a vertex of it is.
struct LoaderPlan {
    /// the loader, 0 to 11, whose register holds the array's offset.
    index: usize,
    offset: u32,
    stride: u32,
    size: u32,
    fields: Vec<PlacedField>,
}

/// a field of an array's vertex, and the input register it lands in.
struct PlacedField {
    offset: u32,
    ty: u32,
    count: u32,
    register: usize,
}

impl InputPlan {
    /// what fetch_vertex works out for every vertex, the same way: an
    /// attribute an array gives is read, one flagged fixed takes the value
    /// the command list set, the rest read (0, 0, 0, 1), and the unit's
    /// permutation places them, a later attribute over an earlier one in
    /// the same register.
    fn new(registers: &[u32], layout: &VertexLayout, fixed: &[Vec4; 16]) -> InputPlan {
        // the array field that gives each attribute, the last when several do
        let mut source = [None; 12];
        for (l, loader) in layout.loaders.iter().enumerate() {
            for (f, field) in loader.fields.iter().enumerate() {
                source[field.id] = Some((l, f));
            }
        }
        let count = (((registers[REG_VS_NUM_INPUT_ATTRIBUTES] & 0xF) + 1) as usize).min(12);
        let map = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] as u64
            | ((registers[REG_VS_BLOCK + SHADER_INPUT_MAP_HIGH] as u64) << 32);
        let mut owner = [None; shader::INPUT_REGISTERS];
        for id in 0..count {
            owner[((map >> (id * 4)) & 0xF) as usize] = Some(id);
        }
        let fixed_mask = (registers[REG_ATTRIBUTE_FORMAT_HIGH] >> 16) & 0xFFF;
        let mut template = [shader::ZERO; shader::INPUT_REGISTERS];
        let mut fixed_registers = [None; shader::INPUT_REGISTERS];
        let mut loaders: Vec<LoaderPlan> = layout
            .loaders
            .iter()
            .map(|loader| LoaderPlan {
                index: loader.index,
                offset: loader.offset,
                stride: loader.stride,
                size: loader.size,
                fields: Vec::new(),
            })
            .collect();
        for (register, id) in owner.iter().enumerate() {
            let Some(id) = *id else { continue };
            match source[id] {
                Some((l, f)) => {
                    let field = &layout.loaders[l].fields[f];
                    loaders[l].fields.push(PlacedField { offset: field.offset, ty: field.ty, count: field.count, register });
                }
                None if fixed_mask & (1 << id) != 0 => fixed_registers[register] = Some(id),
                None => template[register] = [0.0, 0.0, 0.0, 1.0],
            }
        }
        loaders.retain(|loader| !loader.fields.is_empty());
        let mut plan = InputPlan { template, fixed: fixed_registers, loaders };
        plan.refresh(registers, fixed);
        plan
    }

    /// what a draw changes without changing how its vertices are laid
    /// out, where its arrays start and the values of its fixed attributes.
    fn refresh(&mut self, registers: &[u32], fixed: &[Vec4; 16]) {
        for loader in &mut self.loaders {
            loader.offset = registers[REG_ATTRIBUTE_LOADER + loader.index * REG_ATTRIBUTE_LOADER_STRIDE] & 0x0FFF_FFFF;
        }
        for (register, id) in self.fixed.iter().enumerate() {
            if let Some(id) = *id {
                self.template[register] = fixed[id];
            }
        }
    }

    /// one vertex's input registers.
    #[cfg(test)]
    fn fetch<M: GpuMemory>(&self, memory: &mut M, base: u32, vertex_index: u32) -> [Vec4; shader::INPUT_REGISTERS] {
        let mut input = self.template;
        self.fetch_into(memory, base, vertex_index, &mut input);
        input
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn fetch_into<M: GpuMemory>(&self, memory: &mut M, base: u32, vertex_index: u32, input: &mut [Vec4; shader::INPUT_REGISTERS]) {
        for loader in &self.loaders {
            // base is physical, and a loader's offset can carry it from one
            // region into another, titles point the base at the start of VRAM
            // and reach vertex data in FCRAM through the offset.
            let vertex = memory.translate(base + loader.offset + vertex_index * loader.stride);
            let size = loader.size as usize;
            match memory.slice(vertex, size) {
                Some(bytes) => loader.place(bytes, input),
                None => {
                    // twelve fields of sixteen bytes, and their alignment, fit
                    let mut bytes = [0u8; 256];
                    memory.read(vertex, &mut bytes[..size]);
                    loader.place(&bytes[..size], input);
                }
            }
        }
    }
}

impl LoaderPlan {
    /// puts the fields of a vertex's bytes in their registers. components
    /// an array does not provide read as (0, 0, 0, 1), a two-component
    /// texture coordinate arrives as (u, v, 0, 1).
    #[inline(always)]
    fn place(&self, bytes: &[u8], input: &mut [Vec4; shader::INPUT_REGISTERS]) {
        for field in &self.fields {
            let (at, count) = (field.offset as usize, field.count as usize);
            let mut value = [0.0f32, 0.0, 0.0, 1.0];
            let src = &bytes[at..];
            match field.ty {
                0 => {
                    if src.len() >= count {
                        for i in 0..count {
                            value[i] = src[i] as i8 as f32;
                        }
                    }
                }
                1 => {
                    if src.len() >= count {
                        for i in 0..count {
                            value[i] = src[i] as f32;
                        }
                    }
                }
                2 => {
                    if src.len() >= count * 2 {
                        match count {
                            1 => {
                                value[0] = i16::from_le_bytes([src[0], src[1]]) as f32;
                            }
                            2 => {
                                value[0] = i16::from_le_bytes([src[0], src[1]]) as f32;
                                value[1] = i16::from_le_bytes([src[2], src[3]]) as f32;
                            }
                            3 => {
                                value[0] = i16::from_le_bytes([src[0], src[1]]) as f32;
                                value[1] = i16::from_le_bytes([src[2], src[3]]) as f32;
                                value[2] = i16::from_le_bytes([src[4], src[5]]) as f32;
                            }
                            4 => {
                                value[0] = i16::from_le_bytes([src[0], src[1]]) as f32;
                                value[1] = i16::from_le_bytes([src[2], src[3]]) as f32;
                                value[2] = i16::from_le_bytes([src[4], src[5]]) as f32;
                                value[3] = i16::from_le_bytes([src[6], src[7]]) as f32;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    if src.len() >= count * 4 {
                        match count {
                            1 => {
                                value[0] = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                            }
                            2 => {
                                value[0] = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                                value[1] = f32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                            }
                            3 => {
                                value[0] = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                                value[1] = f32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                                value[2] = f32::from_le_bytes([src[8], src[9], src[10], src[11]]);
                            }
                            4 => {
                                value[0] = f32::from_le_bytes([src[0], src[1], src[2], src[3]]);
                                value[1] = f32::from_le_bytes([src[4], src[5], src[6], src[7]]);
                                value[2] = f32::from_le_bytes([src[8], src[9], src[10], src[11]]);
                                value[3] = f32::from_le_bytes([src[12], src[13], src[14], src[15]]);
                            }
                            _ => {}
                        }
                    }
                }
            }
            input[field.register] = value;
        }
    }
}

/// the words a draw's vertex layout comes from: the attribute formats,
/// each loader's components, stride and count, and how the vertex shader
/// takes its inputs. the loaders' offsets are not among them, titles go
/// from one mesh to the next by changing those alone.
const PLAN_KEY: usize = 2 + REG_ATTRIBUTE_LOADER_COUNT * 2 + 3;
/// the vertex layouts kept, more than a busy frame draws with.
const PLANS: usize = 32;

/// the input plans of the vertex layouts draws used last. working one out
/// takes longer than the rest of a small draw.
#[derive(Default)]
struct Plans {
    keys: Vec<[u32; PLAN_KEY]>,
    plans: Vec<InputPlan>,
    /// the plan the last draw used, and the one a new layout replaces next.
    current: usize,
    next: usize,
}

impl Plans {
    /// the plan for the layout the registers set, brought up to the draw.
    fn update(&mut self, registers: &[u32], fixed: &[Vec4; 16]) -> &InputPlan {
        let mut key = [0; PLAN_KEY];
        key[0] = registers[REG_ATTRIBUTE_FORMAT_LOW];
        key[1] = registers[REG_ATTRIBUTE_FORMAT_HIGH];
        for i in 0..REG_ATTRIBUTE_LOADER_COUNT {
            let base = REG_ATTRIBUTE_LOADER + i * REG_ATTRIBUTE_LOADER_STRIDE;
            key[2 + i * 2] = registers[base + 1];
            key[3 + i * 2] = registers[base + 2];
        }
        key[PLAN_KEY - 3] = registers[REG_VS_NUM_INPUT_ATTRIBUTES];
        key[PLAN_KEY - 2] = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW];
        key[PLAN_KEY - 1] = registers[REG_VS_BLOCK + SHADER_INPUT_MAP_HIGH];
        let found = if self.keys.get(self.current) == Some(&key) {
            Some(self.current)
        } else {
            self.keys.iter().position(|kept| *kept == key)
        };
        self.current = match found {
            Some(i) => {
                self.plans[i].refresh(registers, fixed);
                i
            }
            None => {
                let plan = InputPlan::new(registers, &VertexLayout::read(registers), fixed);
                if self.keys.len() < PLANS {
                    self.keys.push(key);
                    self.plans.push(plan);
                    self.keys.len() - 1
                } else {
                    let i = self.next;
                    self.next = (i + 1) % PLANS;
                    self.keys[i] = key;
                    self.plans[i] = plan;
                    i
                }
            }
        };
        &self.plans[self.current]
    }

    /// the plan update last gave.
    fn current(&self) -> &InputPlan {
        &self.plans[self.current]
    }
}

/// fetches one vertex's attributes and lays them out as shader input
#[cfg(test)]
fn fetch_vertex<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    // physical address of the attribute arrays.
    base: u32,
    layout: &VertexLayout,
    fixed: &[Vec4; 16],
    vertex_index: u32,
) -> [Vec4; shader::INPUT_REGISTERS] {
    InputPlan::new(registers, layout, fixed).fetch(memory, base, vertex_index)
}

/// places attributes in a shader unit's input registers (v0-v15), as the
/// unit's attribute permutation (in the block at block) assigns them.
fn map_inputs(registers: &[u32], block: usize, attributes: &[Vec4]) -> [Vec4; shader::INPUT_REGISTERS] {
    let map = registers[block + SHADER_INPUT_MAP_LOW] as u64
        | ((registers[block + SHADER_INPUT_MAP_HIGH] as u64) << 32);
    let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
    for (id, value) in attributes.iter().enumerate() {
        let register = ((map >> (id * 4)) & 0xF) as usize;
        input[register] = *value;
    }
    input
}

/// one vertex after shading, clip-space position plus varyings.
#[derive(Clone, Copy, Default)]
struct Vertex {
    clip: Vec4,
    color: Vec4,
    /// texture coordinate sets 0-2, each (u, v), and w of set 0 for
    /// 2D projection.
    texcoords: [[f32; 2]; 3],
    texcoord0_w: f32,
    /// the surface's orientation and the vector to the viewer, which
    /// fragment lighting works from.
    quaternion: Vec4,
    view: [f32; 3],
}

impl Vertex {
    /// the vertex a fraction t of the way from self to other, which is
    /// how clipping makes new vertices where an edge crosses a plane.
    fn lerp(&self, other: &Vertex, t: f32) -> Vertex {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Vertex {
            clip: std::array::from_fn(|i| mix(self.clip[i], other.clip[i])),
            color: std::array::from_fn(|i| mix(self.color[i], other.color[i])),
            texcoords: std::array::from_fn(|set| {
                std::array::from_fn(|i| mix(self.texcoords[set][i], other.texcoords[set][i]))
            }),
            texcoord0_w: mix(self.texcoord0_w, other.texcoord0_w),
            quaternion: std::array::from_fn(|i| mix(self.quaternion[i], other.quaternion[i])),
            view: std::array::from_fn(|i| mix(self.view[i], other.view[i])),
        }
    }
}

/// where each varying the rasterizer needs lives in the shader's output
/// registers, as GPUREG_SH_OUTMAP_O* describes it.
#[derive(Debug, Clone, Copy)]
struct OutputMap {
    /// (register, component) for each of x, y, z, w.
    position: [(usize, usize); 4],
    color: [Option<(usize, usize)>; 4],
    /// (u, v) of texture coordinate sets 0, 1 and 2, and w of set 0.
    texcoords: [[Option<(usize, usize)>; 2]; 3],
    texcoord0_w: Option<(usize, usize)>,
    quaternion: [Option<(usize, usize)>; 4],
    view: [Option<(usize, usize)>; 3],
}

impl Default for OutputMap {
    fn default() -> Self {
        // the layout picasso/nihstro assign when a shader does not say
        // otherwise, used only when the registers describe nothing at all.
        OutputMap {
            position: [(0, 0), (0, 1), (0, 2), (0, 3)],
            color: [
                Some((2, 0)),
                Some((2, 1)),
                Some((2, 2)),
                Some((2, 3)),
            ],
            texcoords: [[Some((3, 0)), Some((3, 1))], [None; 2], [None; 2]],
            texcoord0_w: None,
            quaternion: [None; 4],
            view: [None; 3],
        }
    }
}

fn read_output_map(registers: &[u32]) -> OutputMap {
    let total = (registers[REG_SHADER_OUTPUT_TOTAL] & 0x7).min(7) as usize;
    if total == 0 {
        return OutputMap::default();
    }

    let mut map = OutputMap {
        position: [(0, 0), (0, 1), (0, 2), (0, 3)],
        color: [None; 4],
        texcoords: [[None; 2]; 3],
        texcoord0_w: None,
        quaternion: [None; 4],
        view: [None; 3],
    };
    let mut saw_position = false;

    for register in 0..total {
        let word = registers[REG_SHADER_OUTPUT_MAP + register];
        for component in 0..4 {
            // each byte names the semantic that output component carries.
            let semantic = ((word >> (component * 8)) & 0x1F) as usize;
            let slot = (register, component);
            match semantic {
                0..=3 => {
                    map.position[semantic] = slot;
                    saw_position = true;
                }
                4..=7 => map.quaternion[semantic - 4] = Some(slot),
                8..=11 => map.color[semantic - 8] = Some(slot),
                12..=13 => map.texcoords[0][semantic - 12] = Some(slot),
                14..=15 => map.texcoords[1][semantic - 14] = Some(slot),
                16 => map.texcoord0_w = Some(slot),
                18..=20 => map.view[semantic - 18] = Some(slot),
                22..=23 => map.texcoords[2][semantic - 22] = Some(slot),
                _ => {}
            }
        }
    }

    if !saw_position {
        return OutputMap::default();
    }
    map
}

/// runs the vertex shader over one vertex's inputs and returns the
/// attributes it outputs, packed as the rest of the pipeline sees them.
/// a stage's output attributes are the output registers its mask enables, in
/// order, attribute 2 is the third enabled register, which is not necessarily
/// o2.
fn pack_outputs(outputs: &[Vec4; shader::OUTPUT_REGISTERS], mask: u32) -> [Vec4; 16] {
    // a mask of zero is a stage nobody configured, take the registers as
    // they are rather than dropping everything.
    if mask & 0xFFFF == 0 {
        return *outputs;
    }
    let mut packed = [shader::ZERO; 16];
    let enabled = (0..shader::OUTPUT_REGISTERS).filter(|register| mask & (1 << register) != 0);
    for (slot, register) in enabled.enumerate() {
        packed[slot] = outputs[register];
    }
    packed
}

/// where each of a vertex's 22 values comes from in a stage's output
/// registers, worked out once a draw from the output map and the stage's
/// output mask: what packing the outputs and picking the varyings out of
/// them gives, without packing every vertex.
#[derive(Clone, Copy)]
struct VertexPlan {
    /// the output register times four plus the component, or PLAN_ZERO or
    /// PLAN_ONE, in the order of clip, color, texcoords, quaternion, view
    /// and texcoord0_w.
    from: [u8; 22],
}

/// a value that no output register gives.
const PLAN_ZERO: u8 = 64;
/// a color that no output register gives: a shader that writes no color
/// means "use white", not "use whatever happened to be in that register",
/// an untextured draw should come out lit rather than black, and a textured
/// one unmodulated.
const PLAN_ONE: u8 = 65;

impl VertexPlan {
    fn new(map: &OutputMap, mask: u32) -> VertexPlan {
        // the output register behind each attribute, as pack_outputs packs
        // them: a mask of zero takes the registers as they are
        let mut registers = [None; shader::OUTPUT_REGISTERS];
        let enabled = (0..shader::OUTPUT_REGISTERS).filter(|register| mask & 0xFFFF == 0 || mask & (1 << register) != 0);
        for (slot, register) in enabled.enumerate() {
            registers[slot] = Some(register);
        }
        // an attribute past the enabled registers is packed as zero
        let read = |(attribute, component): (usize, usize)| registers[attribute].map_or(PLAN_ZERO, |register| (register * 4 + component) as u8);
        let or = |slot: Option<(usize, usize)>, missing: u8| slot.map_or(missing, read);
        let mut from = [PLAN_ZERO; 22];
        for i in 0..4 {
            from[i] = read(map.position[i]);
            from[4 + i] = or(map.color[i], PLAN_ONE);
            from[14 + i] = or(map.quaternion[i], PLAN_ZERO);
        }
        for (set, coordinates) in map.texcoords.iter().enumerate() {
            for (i, &slot) in coordinates.iter().enumerate() {
                from[8 + set * 2 + i] = or(slot, PLAN_ZERO);
            }
        }
        for (i, &slot) in map.view.iter().enumerate() {
            from[18 + i] = or(slot, PLAN_ZERO);
        }
        from[21] = or(map.texcoord0_w, PLAN_ONE);
        VertexPlan { from }
    }

    /// picks the rasterizer's varyings out of a vertex's output registers.
    #[inline(always)]
    fn vertex(&self, outputs: &[Vec4; shader::OUTPUT_REGISTERS]) -> Vertex {
        #[inline(always)]
        fn get(outputs: &[Vec4; shader::OUTPUT_REGISTERS], idx: u8) -> f32 {
            if idx < 64 {
                outputs[(idx >> 2) as usize][(idx & 3) as usize]
            } else if idx == PLAN_ONE {
                1.0
            } else {
                0.0
            }
        }
        let f = &self.from;
        Vertex {
            clip: [get(outputs, f[0]), get(outputs, f[1]), get(outputs, f[2]), get(outputs, f[3])],
            color: [get(outputs, f[4]), get(outputs, f[5]), get(outputs, f[6]), get(outputs, f[7])],
            texcoords: [[get(outputs, f[8]), get(outputs, f[9])], [get(outputs, f[10]), get(outputs, f[11])], [get(outputs, f[12]), get(outputs, f[13])]],
            texcoord0_w: get(outputs, f[21]),
            quaternion: [get(outputs, f[14]), get(outputs, f[15]), get(outputs, f[16]), get(outputs, f[17])],
            view: [get(outputs, f[18]), get(outputs, f[19]), get(outputs, f[20])],
        }
    }
}

/// takes shader inputs through the vertex shader, over threads when there
/// are enough of them, and the geometry shader when one is enabled,
/// producing the vertices the rasterizer assembles. order says which input
/// each vertex comes from when an index buffer repeats them, and comes back
/// to say which of the vertices made each one is, none for the geometry
/// shader's, which come in order.
/// vertices kept for a draw that may be reused once more.
struct Reuse {
    vertices: std::sync::Arc<Vec<Vertex>>,
    samples: [Vertex; 3],
    age: u8,
}

fn reuse_pixels() -> f32 {
    static N: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("ZAKURO_REUSE_PX").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0))
}

/// whether three vertices shaded now sit where, and look as, those the
/// draw was last shaded with, to within a pixel and a little colour.
fn barely_moved(now: &[Vertex; 3], before: &[Vertex; 3], half_width: f32, half_height: f32) -> bool {
    let tolerance = reuse_pixels();
    now.iter().zip(before).all(|(a, b)| {
        let (wa, wb) = (a.clip[3], b.clip[3]);
        if !(wa >= 1e-3 && wb >= 1e-3) || (wa - wb).abs() > 0.01 * wa {
            return false;
        }
        let dx = (a.clip[0] / wa - b.clip[0] / wb) * half_width;
        let dy = (a.clip[1] / wa - b.clip[1] / wb) * half_height;
        let colour = a.color.iter().zip(&b.color).all(|(x, y)| (x - y).abs() < 0.02);
        let uv = a.texcoords[0].iter().zip(&b.texcoords[0]).all(|(x, y)| (x - y).abs() < 0.002);
        dx.abs() < tolerance && dy.abs() < tolerance && colour && uv
    })
}

/// the vertices of an earlier frame's shading of this draw, when three of
/// its vertices shaded now are where those were; at most once in a row.
fn try_reuse(
    resources: &mut Resources,
    registers: &[u32],
    unit: &ShaderUnit,
    signature: u64,
    picks: &[[Vec4; shader::INPUT_REGISTERS]; 3],
) -> Option<std::sync::Arc<Vec<Vertex>>> {
    let entry = resources.vreuse.get_mut(&signature).filter(|entry| entry.age < 1)?;
    let plan = VertexPlan::new(&read_output_map(registers), registers[REG_VS_OUTPUT_MASK]);
    let now = shade_with(unit, picks, |outputs| plan.vertex(outputs));
    let now: [Vertex; 3] = [now[0], now[1], now[2]];
    let (half_width, half_height) = (read_float24(registers, REG_VIEWPORT_WIDTH), read_float24(registers, REG_VIEWPORT_HEIGHT));
    if barely_moved(&now, &entry.samples, half_width, half_height) {
        entry.age += 1;
        Some(entry.vertices.clone())
    } else {
        None
    }
}

fn vcache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZAKURO_NO_VCACHE").is_none())
}

/// what tells a draw from the others from one frame to the next, cheaply:
/// its program, size and a few of its vertices' positions.
fn draw_signature(unit: &ShaderUnit, inputs: &[[Vec4; shader::INPUT_REGISTERS]]) -> u64 {
    let last = inputs.len() - 1;
    draw_signature_of(unit, inputs.len(), &[inputs[0], inputs[inputs.len() / 2], inputs[last]])
}

/// the same from the size and the three vertices (the first, middle and last).
fn draw_signature_of(unit: &ShaderUnit, count: usize, picks: &[[Vec4; shader::INPUT_REGISTERS]; 3]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ unit.entry_point as u64;
    let mut mix = |value: u32| {
        hash = (hash ^ value as u64).wrapping_mul(0x100000001b3);
    };
    mix(count as u32);
    for pick in picks {
        for value in &pick[0] {
            mix(value.to_bits());
        }
    }
    hash
}

/// whether a draw is worth hashing: it hit the last few times, or it is
/// time to look again (three in a row every thirty-second time, since what
/// repeats does so on the next frame).
fn vcache_worth(scores: &mut std::collections::HashMap<u64, (u8, u8)>, signature: u64) -> bool {
    let entry = scores.entry(signature).or_insert((0, 0));
    entry.1 = entry.1.wrapping_add(1);
    entry.0 > 0 || entry.1 % 32 < 3
}

fn vcache_note(scores: &mut std::collections::HashMap<u64, (u8, u8)>, signature: u64, hit: bool) {
    if let Some(entry) = scores.get_mut(&signature) {
        entry.0 = if hit { 3 } else { entry.0.saturating_sub(1) };
    }
    if scores.len() > 4096 {
        scores.clear();
    }
}

/// a hash of everything shading a draw's vertices reads: program, uniforms,
/// how outputs are packed, and the first four input registers of each vertex.
fn shading_key(registers: &[u32], unit: &ShaderUnit, inputs: &[[Vec4; shader::INPUT_REGISTERS]]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    let mut mix = |value: u32| {
        hash = (hash ^ value as u64).wrapping_mul(0x100000001b3);
    };
    mix(unit.entry_point);
    mix(inputs.len() as u32);
    mix(registers[REG_VS_OUTPUT_MASK]);
    for word in unit.program.iter().step_by(1) {
        mix(*word);
    }
    for word in unit.descriptors.iter() {
        mix(*word);
    }
    for uniform in unit.float_uniforms.iter() {
        for value in uniform {
            mix(value.to_bits());
        }
    }
    for int in unit.int_uniforms {
        mix(u32::from_le_bytes(int));
    }
    mix(unit.bool_uniforms as u32);
    for from in VertexPlan::new(&read_output_map(registers), registers[REG_VS_OUTPUT_MASK]).from {
        mix(from as u32);
    }
    for input in inputs {
        for register in &input[..4] {
            for value in register {
                mix(value.to_bits());
            }
        }
    }
    hash
}

/// ZAKURO_GPU_GEOMETRY=0 turns off geometry-shader draws as slots on the
/// host GPU, each slot running both programs; on by default.
#[cfg(feature = "vulkan")]
pub(crate) fn gpu_geometry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(test) || *ON.get_or_init(|| std::env::var("ZAKURO_GPU_GEOMETRY").map_or(true, |value| value != "0"))
}

fn cpu_shade_log() -> bool {
    static WHY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WHY.get_or_init(|| std::env::var_os("ZAKURO_CPU_SHADE_LOG").is_some())
}

fn process_vertices<'a>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    inputs: &[[Vec4; shader::INPUT_REGISTERS]],
    order: Option<&'a [usize]>,
) -> (Vec<Vertex>, Option<&'a [usize]>) {
    let _timer = crate::stats::time(&crate::stats::SHADING);
    crate::stats::fell_back(crate::stats::FALL_CPU);
    if log::log_enabled!(log::Level::Trace) {
        if let Some(input) = inputs.first() {
            let used_uniforms = vertex_shader.float_uniforms.iter().filter(|u| **u != shader::ZERO).count();
            log::trace!(
                "vertex shader: entry {}, {used_uniforms} non-zero uniforms, output mask 0x{:X}, \
                 first vertex inputs {:?}",
                vertex_shader.entry_point,
                registers[REG_VS_OUTPUT_MASK],
                &input[..8],
            );
        }
    }
    let map = read_output_map(registers);
    // ZAKURO_CPU_SHADE_LOG=1: each draw that takes the CPU over a
    // millisecond to shade
    let started = cpu_shade_log().then(std::time::Instant::now);
    let slow = |stage: &str, made: usize, vs_time: Option<std::time::Duration>| {
        if let Some(elapsed) = started.map(|s| s.elapsed()).filter(|e| e.as_micros() >= 1000) {
            log::info!(
                target: "zakuro_gpu::diag",
                "slow CPU shading: {:.2} ms ({stage}, vertex shader {:.2} ms), {} inputs, {} in order, {made} out, vs entry {}, gs entry {}, gs config 0x{:X}, gs input config 0x{:X}",
                elapsed.as_secs_f64() * 1e3,
                vs_time.map_or(0.0, |t| t.as_secs_f64() * 1e3),
                inputs.len(),
                order.map_or(0, <[usize]>::len),
                vertex_shader.entry_point,
                geometry_shader.entry_point,
                registers[REG_GS_CONFIG],
                registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG],
            );
        }
    };
    if registers[REG_GEOSTAGE_CONFIG] & 0x3 == 2 {
        // the results in draw order, which repeats vertices an index buffer
        // names more than once
        let shaded = shade(registers, vertex_shader, inputs);
        let vs_time = started.map(|s| s.elapsed());
        let vertices = match order {
            Some(order) => geometry_stage(registers, geometry_shader, &map, order.iter().map(|&i| shaded[i])),
            None => geometry_stage(registers, geometry_shader, &map, shaded.into_iter()),
        };
        slow("geometry", vertices.len(), vs_time);
        if started.is_some_and(|s| s.elapsed().as_micros() >= 1000) {
            if let Some(report) = shader::geometry_report(geometry_shader) {
                log::info!(target: "zakuro_gpu::diag", "slow geometry program: {report}");
            }
        }
        return (vertices, None);
    }
    // each vertex made once, by the threads that shade it
    let plan = VertexPlan::new(&map, registers[REG_VS_OUTPUT_MASK]);
    let vertices = shade_with(vertex_shader, inputs, |outputs| plan.vertex(outputs));
    slow("vertex", vertices.len(), None);
    (vertices, order)
}

/// a draw's vertices, as shader inputs or already through the shaders.
#[derive(Clone, Copy)]
enum Vertices<'a> {
    Unshaded {
        vertex_shader: &'a ShaderUnit,
        geometry_shader: &'a ShaderUnit,
        inputs: &'a [[Vec4; shader::INPUT_REGISTERS]],
        /// which input each vertex is, when an index buffer repeats them.
        order: Option<&'a [usize]>,
    },
    /// vertex arrays for the GPU to decode and shade.
    #[cfg(feature = "vulkan")]
    Raw {
        vertex_shader: &'a ShaderUnit,
        raw: &'a hardware::RawInputs,
        /// each vertex of the draw, counted from the first of the arrays.
        vertices: &'a [u32],
    },
    Shaded(&'a [Vertex]),
    /// vertices shaded once for each input, with the order an index buffer
    /// takes them in, as the reuse of an earlier frame's has them.
    ShadedOrdered { shaded: &'a [Vertex], order: Option<&'a [usize]> },
}

impl Vertices<'_> {
    fn len(&self) -> usize {
        match self {
            Vertices::Unshaded { inputs, order, .. } => order.map_or(inputs.len(), <[usize]>::len),
            #[cfg(feature = "vulkan")]
            Vertices::Raw { vertices, .. } => vertices.len(),
            Vertices::Shaded(shaded) => shaded.len(),
            Vertices::ShadedOrdered { shaded, order } => order.map_or(shaded.len(), <[usize]>::len),
        }
    }
}

/// where the GPU's vertex shader finds each varying, per semantic
/// component of position, color, texture coordinates 0, 1 and 2, the
/// quaternion and the view vector, the output register times four plus
/// the component, the way to_vertex reads them.
#[cfg(feature = "vulkan")]
fn output_semantics(registers: &[u32]) -> [u32; 24] {
    semantics_for(registers, registers[REG_VS_OUTPUT_MASK])
}

/// output_semantics for the stage whose output mask is given, the
/// geometry shader's for a draw with one.
#[cfg(feature = "vulkan")]
fn semantics_for(registers: &[u32], mask: u32) -> [u32; 24] {
    let map = read_output_map(registers);
    let mask = mask & 0xFFFF;
    // the output register behind each attribute, as pack_outputs packs them
    let mut enabled = [0; shader::OUTPUT_REGISTERS];
    let mut count = 0;
    for register in (0..shader::OUTPUT_REGISTERS).filter(|r| mask == 0 || mask & (1 << r) != 0) {
        enabled[count] = register;
        count += 1;
    }
    let enabled = &enabled[..count];
    let slot = |slot: Option<(usize, usize)>| match slot {
        None => hardware::MISSING,
        Some((attribute, component)) => {
            enabled.get(attribute).map_or(hardware::ZERO, |&register| (register * 4 + component) as u32)
        }
    };
    let [t0, t1, t2] = map.texcoords;
    let slots = map
        .position
        .map(Some)
        .into_iter()
        .chain(map.color)
        .chain(t0)
        .chain(t1)
        .chain(t2)
        .chain(map.quaternion)
        .chain(map.view)
        .chain([map.texcoord0_w]);
    let mut semantics = [hardware::MISSING; 24];
    for (semantic, found) in semantics.iter_mut().zip(slots) {
        *semantic = slot(found);
    }
    semantics
}

/// the output registers that place a vertex, its position, and the normal
/// and the view the lighting takes, and those of its textures'
/// coordinates, one bit each.
pub(crate) fn placing_outputs(registers: &[u32]) -> (u16, u16) {
    let map = read_output_map(registers);
    let mask = registers[REG_VS_OUTPUT_MASK] & 0xFFFF;
    // the output register behind an attribute, as pack_outputs packs them
    let register = |attribute: usize| (0..shader::OUTPUT_REGISTERS).filter(|r| mask == 0 || mask & (1 << r) != 0).nth(attribute);
    let outputs = |slots: &mut dyn Iterator<Item = (usize, usize)>| {
        slots.filter_map(|(attribute, _)| register(attribute)).fold(0, |outputs, register| outputs | 1 << register)
    };
    let placing = outputs(&mut map.position.into_iter().chain(map.quaternion.into_iter().flatten()).chain(map.view.into_iter().flatten()));
    let coordinates = outputs(&mut map.texcoords.into_iter().flatten().flatten().chain(map.texcoord0_w));
    (placing, coordinates)
}

/// the semantics output_semantics worked out for the last draw, and the
/// registers they came from, the output map and mask, which draws mostly
/// keep.
#[cfg(feature = "vulkan")]
#[derive(Default)]
struct LastSemantics {
    read: [u32; 9],
    semantics: Option<[u32; 24]>,
}

#[cfg(feature = "vulkan")]
impl LastSemantics {
    fn get(&mut self, registers: &[u32]) -> [u32; 24] {
        let mut read = [0; 9];
        read[0] = registers[REG_SHADER_OUTPUT_TOTAL];
        read[1..8].copy_from_slice(&registers[REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END]);
        read[8] = registers[REG_VS_OUTPUT_MASK];
        match self.semantics {
            Some(semantics) if self.read == read => semantics,
            _ => {
                let semantics = output_semantics(registers);
                *self = LastSemantics { read, semantics: Some(semantics) };
                semantics
            }
        }
    }
}

fn shader_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let wanted = std::env::var("ZAKURO_SHADER_THREADS").ok().and_then(|v| v.parse::<usize>().ok());
        let threads = if let Some(wanted) = wanted {
            wanted.max(1)
        } else if cfg!(target_os = "horizon") {
            2
        } else {
            std::thread::available_parallelism().map_or(2, |c| (c.get() / 2).clamp(1, 4))
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(256 * 1024)
            .thread_name(|i| format!("shader worker {i}"))
            .start_handler(|i| crate::thread_role("shader", i))
            .build()
            .expect("shader pool")
    })
}

/// vertices a draw needs before shading them is worth splitting over
/// threads.
#[cfg(target_os = "horizon")]
const PARALLEL_VERTICES: usize = 48;
#[cfg(not(target_os = "horizon"))]
const PARALLEL_VERTICES: usize = 32;
/// the fewest vertices each thread takes at a time, whole batches of the
/// shader's.
const PARALLEL_CHUNK: usize = 16;

/// vertices a draw needs before shading goes over threads, PARALLEL_VERTICES
/// unless ZAKURO_PARALLEL_VERTICES says another number, to try them on a device.
fn parallel_vertices() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("ZAKURO_PARALLEL_VERTICES").ok().and_then(|v| v.parse().ok()).unwrap_or(PARALLEL_VERTICES))
}

fn shade(registers: &[u32], unit: &ShaderUnit, inputs: &[[Vec4; shader::INPUT_REGISTERS]]) -> Vec<[Vec4; 16]> {
    let mask = registers[REG_VS_OUTPUT_MASK];
    shade_with(unit, inputs, |outputs| pack_outputs(outputs, mask))
}

/// runs the vertex shader over inputs, over threads when there are enough
/// of them, and what to do with each one's outputs on the thread that made
/// them, in the inputs' order.
fn shade_with<T: Send + Default + Clone>(
    unit: &ShaderUnit,
    inputs: &[[Vec4; shader::INPUT_REGISTERS]],
    then: impl Fn(&[Vec4; shader::OUTPUT_REGISTERS]) -> T + Sync,
) -> Vec<T> {
    if inputs.len() < parallel_vertices() || crate::shade_queue::IN_JOB.with(|flag| flag.get()) {
        return shader::run_vertices_mapped(unit, inputs, then);
    }
    let pool = shader_pool();
    // one share for each worker and one for this thread, which works too
    // instead of waiting for them to wake up and finish
    let shares = pool.current_num_threads() + 1;
    let chunk = inputs.len().div_ceil(shares).next_multiple_of(8).max(PARALLEL_CHUNK);
    let mut outputs: Vec<T> = vec![T::default(); inputs.len()];
    {
        let mut work = outputs.chunks_mut(chunk).zip(inputs.chunks(chunk));
        let first = work.next();
        pool.in_place_scope(|scope| {
            for (out_chunk, in_chunk) in work {
                let then = &then;
                scope.spawn(move |_| shader::run_vertices_into(unit, in_chunk, out_chunk, then));
            }
            if let Some((out_chunk, in_chunk)) = first {
                shader::run_vertices_into(unit, in_chunk, out_chunk, &then);
            }
        });
    }
    outputs
}

fn geometry_stage(
    registers: &[u32],
    unit: &ShaderUnit,
    map: &OutputMap,
    vertex_outputs: impl Iterator<Item = [Vec4; 16]>,
) -> Vec<Vertex> {
    let mode = registers[REG_GS_CONFIG] & 0xFF;
    if mode == 2 {
        return fixed_primitives(registers, unit, map, vertex_outputs);
    }
    if mode != 0 {
        // once, a warning a draw costs a frame its time
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| log::warn!("geometry shader mode {mode} is not implemented"));
        return Vec::new();
    }
    let per_vertex = ((registers[REG_VS_OUTPUT_TOTAL] & 0xF) + 1) as usize;
    let per_invocation = ((registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG] & 0xF) + 1) as usize;
    let output_mask = registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK];

    // every invocation's inputs, then all of them through the shader in
    // batches
    let count = vertex_outputs.size_hint().0;
    let mut pending = [shader::ZERO; 32];
    let mut pending_len = 0;
    let mut invocations = Vec::with_capacity((count * per_vertex) / per_invocation);
    for outputs in vertex_outputs {
        pending[pending_len..pending_len + per_vertex].copy_from_slice(&outputs[..per_vertex]);
        pending_len += per_vertex;
        if pending_len < per_invocation {
            continue;
        }
        invocations.push(map_inputs(registers, REG_GS_BLOCK, &pending[..per_invocation]));
        pending_len = 0;
    }
    // many primitives go over threads, like many vertices do
    let emitted: Vec<Vec<_>> = if cfg!(target_os = "horizon") || invocations.len() < PARALLEL_VERTICES / 4 {
        shader::run_geometry_many(unit, &invocations)
    } else {
        invocations.par_chunks(PARALLEL_CHUNK / 4).flat_map_iter(|chunk| shader::run_geometry_many(unit, chunk)).collect()
    };

    let plan = VertexPlan::new(map, output_mask);
    let mut vertices = Vec::with_capacity(invocations.len() * 6);
    for (input, triangles) in invocations.iter().zip(&emitted) {
        if vertices.is_empty() && log::log_enabled!(log::Level::Trace) {
            let used_uniforms = unit.float_uniforms.iter().filter(|u| **u != shader::ZERO).count();
            log::trace!(
                "geometry shader: entry {} com mode 0x{:X}, {per_vertex} per vertex, \
                 {per_invocation} per invocation, input map 0x{:08X}, output mask 0x{:X}, \
                 {used_uniforms} non-zero uniforms, bools 0x{:X}, ints {:?}, inputs {:?}, \
                 first output {:?}, {} triangles",
                unit.entry_point,
                registers[REG_VS_COM_MODE],
                registers[REG_GS_BLOCK + SHADER_INPUT_MAP_LOW],
                output_mask,
                unit.bool_uniforms,
                unit.int_uniforms,
                &input[..per_invocation.min(4)],
                triangles.first().map(|t| &t[0][..4]),
                triangles.len(),
            );
        }
        for triangle in triangles {
            for outputs in triangle {
                vertices.push(plan.vertex(outputs));
            }
        }
    }
    vertices
}

/// the geometry shader's fixed primitive mode: each primitive's vertices,
/// a fixed number of them in draw order, go into its float uniforms from a
/// start index on, the vertex shader's outputs one after another, and the
/// program runs once for each primitive with no inputs of its own.
fn fixed_primitives(
    registers: &[u32],
    unit: &ShaderUnit,
    map: &OutputMap,
    vertex_outputs: impl Iterator<Item = [Vec4; 16]>,
) -> Vec<Vertex> {
    let config = registers[REG_GS_CONFIG];
    let per_vertex = ((registers[REG_VS_OUTPUT_TOTAL] & 0xF) + 1) as usize;
    let fixed = (((config >> 8) & 0xF) + 1) as usize;
    let start = ((config >> 16) & 0xFF) as usize;
    let per_primitive = per_vertex * fixed;
    if start + per_primitive > shader::FLOAT_UNIFORMS {
        return Vec::new();
    }
    let attributes: Vec<Vec4> = vertex_outputs.flat_map(|outputs| outputs.into_iter().take(per_vertex)).collect();
    let primitives: Vec<&[Vec4]> = attributes.chunks_exact(per_primitive).collect();
    let plan = VertexPlan::new(map, registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK]);
    // the compiled program runs a share of the primitives through one
    // context, with their vertices put in its uniforms, and the batched
    // runner the ones it can't, one run a primitive
    let run = |chunk: &[&[Vec4]]| -> Vec<Vertex> {
        if let Some(triangles) = shader::run_geometry_fixed(unit, start, chunk) {
            return triangles.iter().flatten().map(|outputs| plan.vertex(outputs)).collect();
        }
        let mut patched = unit.clone();
        chunk
            .iter()
            .flat_map(|primitive| {
                patched.float_uniforms[start..start + per_primitive].copy_from_slice(primitive);
                for i in start..start + per_primitive {
                    patched.wide_uniforms[i] = patched.float_uniforms[i].map(wide::f32x8::splat);
                }
                shader::run_geometry_many(&patched, &[[shader::ZERO; shader::INPUT_REGISTERS]]).into_iter().flatten()
            })
            .flatten()
            .map(|outputs| plan.vertex(&outputs))
            .collect()
    };
    // a primitive makes many triangles, so a few of them are worth sharing
    // out, a share each for the workers and this thread
    if primitives.len() < 8 || crate::shade_queue::IN_JOB.with(|flag| flag.get()) {
        return run(&primitives);
    }
    let pool = shader_pool();
    let shares = pool.current_num_threads() + 1;
    let chunk = primitives.len().div_ceil(shares);
    let mut made: Vec<Vec<Vertex>> = vec![Vec::new(); primitives.len().div_ceil(chunk)];
    {
        let mut work = made.iter_mut().zip(primitives.chunks(chunk));
        let first = work.next();
        pool.in_place_scope(|scope| {
            for (out, share) in work {
                let run = &run;
                scope.spawn(move |_| *out = run(share));
            }
            if let Some((out, share)) = first {
                *out = run(share);
            }
        });
    }
    made.concat()
}

/// screen-space vertex, with every varying pre-divided by w so the
/// rasterizer only has to interpolate linearly and divide once per pixel.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Screen {
    pub(crate) x: f32,
    pub(crate) y: f32,
    /// z/w, which is linear in screen space, the depth map turns it into
    /// the value the depth buffer holds.
    pub(crate) z: f32,
    pub(crate) inv_w: f32,
    #[allow(dead_code)]
    pub(crate) clip_x: f32,
    #[allow(dead_code)]
    pub(crate) clip_y: f32,
    #[allow(dead_code)]
    pub(crate) w: f32,
    pub(crate) color: Vec4,
    pub(crate) texcoords: [[f32; 2]; 3],
    pub(crate) texcoord0_w: f32,
    pub(crate) quaternion: Vec4,
    pub(crate) view: [f32; 3],
}

fn to_screen(vertex: &Vertex, viewport: (f32, f32, f32, f32)) -> Option<Screen> {
    let (vx, vy, vw, vh) = viewport;
    let w = vertex.clip[3];
    if w.abs() < 1e-8 {
        return None;
    }
    let inv_w = 1.0 / w;
    let ndc_x = vertex.clip[0] * inv_w;
    let ndc_y = vertex.clip[1] * inv_w;

    #[cfg(target_os = "horizon")]
    let (sx, sy) = (vx + (ndc_x * 0.5 + 0.5) * vw, vy + (ndc_y * 0.5 + 0.5) * vh);
    #[cfg(not(target_os = "horizon"))]
    let snap = |c: f32| (c * 16.0).round() * (1.0 / 16.0);
    #[cfg(not(target_os = "horizon"))]
    let (sx, sy) = (snap(vx + (ndc_x * 0.5 + 0.5) * vw), snap(vy + (ndc_y * 0.5 + 0.5) * vh));
    Some(Screen {
        x: sx,
        y: sy,
        z: vertex.clip[2] * inv_w,
        inv_w,
        clip_x: vertex.clip[0],
        clip_y: vertex.clip[1],
        w,
        color: vertex.color,
        texcoords: vertex.texcoords,
        texcoord0_w: vertex.texcoord0_w,
        quaternion: vertex.quaternion,
        view: vertex.view,
    })
}

#[inline(always)]
fn plane_dist(v: &Vertex, plane_idx: usize) -> f32 {
    match plane_idx {
        0 => v.clip[3] - 1e-5,
        1 => -v.clip[2],
        _ => v.clip[2] + v.clip[3],
    }
}

/// whether a vertex is inside the volume the PICA draws.
#[inline(always)]
fn inside(vertex: &Vertex) -> bool {
    let [_, _, z, w] = vertex.clip;
    w >= 1e-5 && z <= 0.0 && z + w >= 0.0
}

/// a vertex with z on an end of the range the PICA draws when it misses
/// one only by rounding. the PICA works positions out in 24-bit floats,
/// which land on 0 and -w, singles can miss them by a little, and 2D drawn
/// on the near or far plane, a title's video say, lost triangles to it.
#[inline(always)]
fn held_in_range(mut vertex: Vertex) -> Vertex {
    let [_, _, z, w] = vertex.clip;
    let z_over_w = z / w;
    if z_over_w > 0.0 && z_over_w < 1e-8 {
        vertex.clip[2] = 0.0;
    } else if z_over_w < -1.0 && z_over_w > -1.00001 {
        vertex.clip[2] = -w;
    }
    vertex
}

struct ClippedPoly {
    vertices: [Vertex; 16],
    len: usize,
}

impl ClippedPoly {
    #[inline(always)]
    fn as_slice(&self) -> &[Vertex] {
        &self.vertices[..self.len]
    }
}

/// clips a triangle to the volume the PICA draws, in place on the stack without heap allocation.
#[inline(always)]
fn clip_triangle(triangle: [Vertex; 3]) -> ClippedPoly {
    let triangle = triangle.map(held_in_range);
    let mut poly = ClippedPoly {
        vertices: [Vertex::default(); 16],
        len: 3,
    };
    poly.vertices[0] = triangle[0];
    poly.vertices[1] = triangle[1];
    poly.vertices[2] = triangle[2];

    if triangle.iter().all(inside) {
        return poly;
    }

    let mut scratch = [Vertex::default(); 16];
    for plane_idx in 0..3 {
        let mut clipped_len = 0;
        let p_len = poly.len;
        for i in 0..p_len {
            let current = &poly.vertices[i];
            let next = &poly.vertices[(i + 1) % p_len];
            let d0 = plane_dist(current, plane_idx);
            let d1 = plane_dist(next, plane_idx);
            if d0 >= 0.0 && clipped_len < 16 {
                scratch[clipped_len] = *current;
                clipped_len += 1;
            }
            if (d0 >= 0.0) != (d1 >= 0.0) && clipped_len < 16 {
                scratch[clipped_len] = current.lerp(next, d0 / (d0 - d1));
                clipped_len += 1;
            }
        }
        poly.len = clipped_len;
        poly.vertices[..clipped_len].copy_from_slice(&scratch[..clipped_len]);
        if poly.len < 3 {
            poly.len = 0;
            return poly;
        }
    }
    poly
}

/// a texture that is rows of a buffer the GPU drew and still holds, which
/// the GPU copies instead of guest memory getting the buffer back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DrawnTexture {
    pub(crate) addr: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: ColorFormat,
}

/// whether a draw reading its own target samples it directly; on by default
/// (ZAKURO_FEEDBACK_LOOP=0 falls back to a snapshot copy).
pub(crate) fn feedback_loop() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ZAKURO_FEEDBACK_LOOP")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(!cfg!(test))
    })
}

/// how a texture minifies: the magnification filter's bit says it by default
/// (as it always has), ZAKURO_FILTER=split reads the minification bit as the
/// console does, =any is linear when either asks, =linear forces both (a test).
fn filter_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("ZAKURO_FILTER").as_deref() {
        Ok("split") => 0,
        Ok("any") => 2,
        Ok("linear") => 3,
        _ => 1,
    })
}

/// texture unit 0's configuration and backing data for one draw call.
struct BoundTexture {
    /// the texture decoded, row by row from the top, none when it is drawn
    /// unless it reaches past the rows drawn.
    texels: Arc<[[u8; 4]]>,
    drawn: Option<DrawnTexture>,
    /// bilinear rather than point sampling when magnifying.
    linear: bool,
    /// bilinear rather than point sampling when minifying.
    min_linear: bool,
    /// bilinear rather than point sampling between mipmap levels.
    mip_linear: bool,
    /// smallest mipmap level present (0 when only the full size is used).
    max_level: u8,
    /// signed 5.8 fixed-point LOD bias from GPUREG_TEXUNITi_LOD.
    lod_bias: i16,
    wrap_s: Wrap,
    wrap_t: Wrap,
    width: u32,
    height: u32,
    border: [f32; 4],
    /// the texture pack's picture for it, which the host's GPU draws with
    /// in its place once it has been read.
    replacement: Option<Arc<crate::pack::Material>>,
}

/// what a draw does for frame interpolation, see crate::interpolation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Pass {
    /// the title's draw, which also goes into the twins of the buffers it
    /// draws into or reads, where they have any.
    #[default]
    Real,
    /// the title's draw, which a draw of its own in between follows, its
    /// buffers get twins before it.
    Twinned,
    /// the draw in between, into the twins alone.
    Twin,
}

/// what draws keep from one to the next.
#[derive(Default)]
pub struct Resources {
    /// vertices shaded last time for draws whose program, uniforms and
    /// inputs were exactly these, by a hash of them, to reuse when nothing
    /// changed (a still camera).
    vcache: std::collections::HashMap<u64, std::sync::Arc<Vec<Vertex>>>,
    /// the generation before it, for draws that came back after a few others.
    vcache_old: std::collections::HashMap<u64, std::sync::Arc<Vec<Vertex>>>,
    /// by draw signature, the vertices last shaded for it and three of them to
    /// compare against, for reusing them when the draw barely moved.
    vreuse: std::collections::HashMap<u64, Reuse>,
    /// per draw signature, how many recent lookups it hit (to three) and a counter.
    vcache_score: std::collections::HashMap<u64, (u8, u8)>,
    /// the draw whose vertices a helper is shading, which is drawn once the
    /// next one has gone to shade too, see finish_pending.
    pending: std::collections::VecDeque<Pending>,
    pub textures: TextureCache,
    pub light_tables: Tables,
    pub proctex_tables: crate::proctex::Tables,
    pub fog_table: crate::fog::Table,
    scratch: Scratch,
    /// the bytes of the last memory fill, the next one's buffer.
    pub(crate) fill: Vec<u8>,
    /// the words of the last command list, the next one's buffer.
    pub(crate) words: Vec<u32>,
    /// the last draw's triangles as the GPU takes them, the next one's
    /// buffer.
    triangles: Vec<u32>,
    /// a geometry draw's inputs in draw order and its slots' indices, the
    /// next one's buffers, and the slots each geometry program needs, by
    /// its fingerprint and entry, none for one the CPU runs.
    #[cfg(feature = "vulkan")]
    geometry_inputs: Vec<[Vec4; shader::INPUT_REGISTERS]>,
    #[cfg(feature = "vulkan")]
    geometry_indices: Vec<u32>,
    #[cfg(feature = "vulkan")]
    geometry_slots: std::collections::HashMap<(u64, u32), Option<u32>>,
    placed: Vec<Screen>,
    slots: Vec<u32>,
    shaded_indices: Vec<u32>,
    /// the last draw's texture environment and lighting, which the next
    /// mostly keeps.
    tex_env: crate::tev::LastTexEnv,
    lighting: crate::lighting::LastLighting,
    /// where the last draw's varyings were in the output registers.
    #[cfg(feature = "vulkan")]
    semantics: LastSemantics,
    /// the host GPU, when draws go to it rather than to the software path.
    #[cfg(feature = "vulkan")]
    pub(crate) hardware: Option<hardware::Hardware>,
    /// dialogue portraits drawn again on the top screen, see mirror_portrait.
    pub(crate) portraits: Portraits,
}

/// what mirror_portrait needs to put the bottom screen's portraits on the
/// top screen.
#[derive(Default)]
pub(crate) struct Portraits {
    /// where the last transfer to the top screen read from.
    pub(crate) top_source: Option<u32>,
    /// the top screen's color buffer registers, address, format and size,
    /// from the last draw into the buffer that transfer read.
    pub(crate) top: Option<[u32; 3]>,
    /// the right corner's portrait and the left one's.
    sides: [Side; 2],
    /// the battle display's draws the bottom screen made this frame that lie
    /// in an island, see ISLANDS.
    hud: Vec<PortraitDraw>,
    /// the last battle display draws flushed, reused if a subtitle flush
    /// follows an end-of-frame transfer flush before the bottom screen runs.
    last_hud: Vec<PortraitDraw>,
    /// the selected land-battle power's name when shown centered at the bottom
    /// of the top screen (`PowerLabelMode::Bottom`), drawn with subtitle shadow.
    power_label: Vec<PortraitDraw>,
    last_power_label: Vec<PortraitDraw>,
    /// vertical bottom-screen bounds (`[low_x, high_x]` in buffer x) of the
    /// active speech bubble above the power bar on this frame, if drawn.
    power_bubble_x: Option<[f32; 2]>,
    /// the dialogue box's text the bottom screen drew this frame, for the
    /// top screen when the title draws no subtitles there itself (How to
    /// Play), and the last of it, as with the battle display.
    dialogue: Vec<PortraitDraw>,
    last_dialogue: Vec<PortraitDraw>,
    /// flushes since the top screen last drew subtitles of its own.
    top_subtitle_quiet: u32,
    /// flushes in a row the bottom screen drew the dialogue box's text, which
    /// only shows on the top screen once it has stayed a few, a flash of the
    /// text buffer while the game rewrites it does not.
    dialogue_held: u32,
    /// flushes in a row since the bottom screen last drew that text.
    dialogue_missing: u32,
    /// remaining flushes keeping tutorial portrait/subtitle layout active while
    /// bottom-screen How-to-Play / S14 / S15 dialogue is playing (`0` in normal battles).
    tutorial_dialogue_hold: u8,
    /// whether the bottom screen has drawn since the last portrait flush.
    bottom_seen: bool,
    /// the title screen's logo and copyright draws redirected from the bottom
    /// screen onto the top screen.
    title: Vec<PortraitDraw>,
    /// whether the bottom screen is currently rendering the title screen pass
    /// (started by the 128x256 Etc1 sky backdrop).
    title_bottom_drawing: bool,
    /// whether the current bottom-screen pass has already counted a title frame.
    title_bottom_counted: bool,
    /// whether the title screen's bottom-screen logo pass was seen since the
    /// last top-screen transfer.
    title_sky_seen: bool,
    /// whether the title screen is currently active.
    pub(crate) title_sky_active: bool,
    /// consecutive top-screen transfers without a title-screen logo draw.
    title_sky_missed: u8,
    /// whether the title screen's sizzle-reel attract video is playing.
    title_sizzle_active: bool,
    /// whether the bottom-screen title sky was seen since the last flush during the sizzle reel.
    title_sizzle_seen: bool,
    /// consecutive flushes without bottom-screen title sky during the sizzle reel.
    title_sizzle_missed: u8,
    /// number of title-screen guest frames elapsed since the opening fly-in started.
    title_frames: u32,
    /// whether a complete bottom-screen title logo pass has been captured.
    title_captured: bool,
    /// warmup passes completed for the current title screen visit (0..=2).
    title_warmups: u8,
    /// fade-in opacity (0.0..=1.0) for the title logo on the top screen,
    /// fading in when the Triumphal trumpets sound after Pit's fly-in.
    title_alpha: f32,
    /// whether the title screen was seen since the frontend last asked.
    pub(crate) title_seen: bool,
    /// first 256x16 Etc1 card header texture address seen on the top screen
    /// this frame (Fuse Weapons draws green A + yellow B).
    fuse_first_header: u32,
    /// whether both Fuse Weapons card headers were seen since last flush.
    fuse_frame_seen: bool,
    /// whether the 256x128 Etc1A4 result-weapon portrait was seen since last flush.
    fuse_weapon_seen: bool,
    /// whether the 64x64 Rgb565 workbench preview element was seen since last flush
    /// (drawn on valid workbench previews, omitted during the full-screen reveal).
    fuse_preview_seen: bool,
    /// consecutive top-screen transfers without Fuse Weapons card headers.
    fuse_missed: u8,
    /// whether the game is currently rendering the Fuse Weapons dual-screen UI.
    pub(crate) fuse_active: bool,
    /// whether the full-screen weapon fusion reveal overlay is playing.
    pub(crate) fuse_reveal: bool,
    /// state machine step (0..=3) matching Fiend's Cauldron's 3-layer mural pass
    /// (`128x64 Etc1` -> `64x64 La4` -> `256x256 L4`) on the top screen.
    cauldron_mural_step: u8,
    /// whether Fiend's Cauldron's top-screen mural was seen since last flush.
    cauldron_top_seen: bool,
    /// whether Fiend's Cauldron's bottom-screen slider cursor was drawn since last flush.
    cauldron_cursor_seen: bool,
    /// whether Fiend's Cauldron's 3D cauldron pot (`128x128 Etc1`) was drawn on
    /// the bottom screen since last flush (stays true through the heart-dump launch).
    cauldron_pot_seen: bool,
    /// consecutive top-screen transfers without Fiend's Cauldron draws.
    cauldron_missed: u8,
    /// whether the game is currently rendering Fiend's Cauldron (prep or heart-dump).
    pub(crate) cauldron_active: bool,
    /// whether the interactive Fiend's Cauldron HUD (pouch & intensity card) is active.
    pub(crate) cauldron_hud_active: bool,
    /// horizontal clip-space delta parking the top-screen Intensity card at x = 255.
    cauldron_card_dx: f32,
    /// live bottom-screen slider cursor center X (`0.0..320.0`).
    pub(crate) cauldron_cursor_x: f32,
    /// whether `cauldron_curr_tier` has been seeded on entry to Fiend's Cauldron.
    cauldron_tier_init: bool,
    /// previous mural tier (`0..=9`) cross-dissolving out when Intensity changes.
    cauldron_prev_tier: u8,
    /// current mural tier (`0..=9`) cross-dissolving in when Intensity changes.
    cauldron_curr_tier: u8,
    /// `0.0..=1.0` cross-dissolve progress from `cauldron_prev_tier` to `cauldron_curr_tier`.
    cauldron_mural_fade: f32,
    /// left-eye top-screen buffer address used to step `cauldron_mural_fade` once per frame.
    cauldron_fade_buf: u32,
    /// smoothly lerped room tint RGB (`[0.0..1.0; 3]`: icy slate-blue `< 2.0` vs crimson `>= 2.0`).
    pub(crate) cauldron_room_rgb: [f32; 3],
    /// state machine step (0..=3) matching Chapter Select's top-screen draw sequence
    /// (`512x256 Etc1` page -> `128x128 Etc1A4` rose crest -> `256x64 Etc1` stage art).
    chapter_step: u8,
    /// whether Chapter Select's top-screen stage preview was seen since last flush.
    chapter_top_seen: bool,
    /// consecutive top-screen transfers without Chapter Select draws.
    chapter_missed: u8,
    /// whether the game is currently rendering the Chapter Select UI.
    pub(crate) chapter_active: bool,
    /// rightmost X (`0.0..400.0`) of header glyph vertices (`sy <= 60`) in the current frame.
    chapter_frame_title_right: f32,
    /// leftmost & rightmost X of `"Chapter XX"` (`sx <= 106, sy <= 54`) in the current frame.
    chapter_frame_num_bounds: (f32, f32),
    /// latched `[left, right]` X of `"Chapter XX"` so the laurel wreath hugs single- and double-digit chapters.
    chapter_num_bounds: (f32, f32),
    /// latched rightmost X of the Chapter Select title text, for centering in the presenter.
    pub(crate) chapter_title_right: f32,
    /// state machine step (`0..=4`) matching Mode Select's top-screen draw sequence.
    mode_top_step: u8,
    /// whether Mode Select's top-screen wave & header were seen since last flush.
    mode_top_seen: bool,
    /// whether the most recent top-screen transfer was Mode Select (gates bottom activation).
    mode_top_latched: bool,
    /// whether Mode Select's `256x256 Rgba4` bottom-screen pills were seen since last flush.
    mode_bot_seen: bool,
    /// suppresses `mode_active` if a sub-menu shared the top paper without Mode Select pills.
    mode_rejected: bool,
    /// consecutive top-screen transfers without Mode Select draws.
    mode_missed: u8,
    /// whether the game is currently rendering the Mode Select UI.
    pub(crate) mode_active: bool,
    /// state machine step (`0..=3`) matching Solo Hub's top-screen sanctuary UI sequence.
    solo_top_step: u8,
    /// whether Solo Hub's top-screen sanctuary UI was seen since last flush.
    solo_top_seen: bool,
    /// consecutive top-screen transfers without Solo Hub draws.
    solo_missed: u8,
    /// whether the game is currently rendering the Solo Hub UI.
    pub(crate) solo_active: bool,
    /// state machine step (`0..=4`) matching File Select's top-screen draw sequence.
    file_top_step: u8,
    /// whether File Select's top-screen lined paper & speech bubble were seen since last flush.
    file_top_seen: bool,
    /// consecutive top-screen transfers without File Select draws.
    file_missed: u8,
    /// whether the game is currently rendering the File Select UI.
    pub(crate) file_active: bool,
    /// whether File Select's bottom `#00FFFF` chroma canvas has been painted this frame.
    pub(crate) file_bot_painted: bool,
    /// state machine step (`0..=2`) matching Weapon Select / Category Select's top pass.
    weapon_step: u8,
    /// state machine step (`0..=2`) matching Gear Hub's top pass (`256x256 L4` -> `256x256 Etc1`).
    gear_step: u8,
    /// state machine step (`0..=2`) matching Power Configurator's top pass (`256x256 Etc1` -> `128x64 L4`).
    power_step: u8,
    /// whether Power Configurator's `64x64 Etc1` board tiles were seen on the bottom screen.
    power_bot_etc1: bool,
    /// whether Power Configurator's bottom workbench (or Gear Hub entry) is armed, excluding Vault's Power Portrait.
    power_bot_latched: bool,
    /// whether `~Power Set X~` summary overlay (`256x32 Etc1A4`) is active on the current top pass.
    power_set_overlay: bool,
    /// horizontal slide offset (`0` at rest) of `~Power Set X~` pills during slide-in/out.
    power_set_shift: f32,
    /// whether `~Power Set X~` has 8+ powers populating its second pill column.
    power_set_two_cols: bool,
    /// sub-screen (`1` = Weapon List, `2` = Category, `3` = Gear Hub, `4` = Power Configurator) seen since flush.
    weapon_sub_seen: u8,
    /// active sub-screen (`1` = Weapon List, `2` = Category, `3` = Gear Hub, `4` = Power Configurator).
    pub(crate) weapon_sub: u8,
    /// consecutive top-screen transfers without Weapon / Category / Gear Hub draws.
    weapon_missed: u8,
    /// whether the game is currently rendering Gear Hub, Weapon Category Select, or Weapon List.
    pub(crate) weapon_active: bool,
    /// whether Weapon Select's bottom `#FF00FF` chroma canvas has been painted this frame.
    pub(crate) weapon_bot_painted: bool,
    /// whether Weapon List's bottom modal dimmer (`Try It! / Equip It! / Back`) has drawn this frame.
    weapon_bot_modal: bool,
    /// whether `Try It! / Equip It! / Back` modal banner is active on Weapon List's top screen.
    weapon_modal_top: bool,
    /// horizontal slide delta keeping `Ranged / Melee / Value` text aligned with the Stars Plaque.
    weapon_plaque_dx: f32,
    /// whether battle-entry field resets have already been applied during this battle.
    battle_cleared: bool,
    /// whether the 3DS frontend-menu top-right battery icon was seen outside of battle.
    pub(crate) frontend_menu_seen: bool,
    /// whether the user toggled `R-Stick` to view raw `LargeTop` instead of custom menu UI.
    pub(crate) menu_override: bool,
    /// whether any textured draw targeted the bottom screen since the last flush.
    bottom_tex_seen: bool,
    /// consecutive top-screen transfers without any textured bottom-screen draw.
    bottom_tex_missed: u8,
    /// whether the bottom screen has no textured UI content (logos, movies, 3D fusion reveal).
    pub(crate) bottom_blank: bool,
    /// whether the game is in play, a battle, where its bottom screen's
    /// portraits are dialogue; in menus the same kind of picture is not.
    pub(crate) in_play: bool,
    /// whether the game is actively in a battle (flying, on foot, or a boss
    /// dialogue cutscene), rather than in menus with the bottom screen hidden.
    pub(crate) in_battle: bool,
    /// whether the current battle session is still active across quiet flight swoops.
    pub(crate) active_battle: bool,
    /// whether KIU's in-battle Pause menu is currently open (`InfoManager+0xC0` bit 16).
    pub(crate) pause_active: bool,
    /// whether the bottom screen drew a large dialogue portrait since the
    /// frontend last asked, see note_big_portrait: a conversation in a
    /// battle, before the player has control, that the game's overrides say
    /// nothing of.
    pub(crate) big_portrait: bool,
    /// which of the battle display's islands the bottom screen drew this
    /// frame while the game was thought to be out of play, see
    /// note_battle_display.
    hud_islands: u8,
    /// whether this frame's portraits already went into the top screen,
    /// under the subtitles; else they go in before it goes out.
    pub(crate) flushed: bool,
    /// while the portraits or a shadow are being drawn, which are not the
    /// game's draws to look at.
    adding: bool,
    /// the lines of text the title last wrote into its text buffers, paired
    /// with each 512x128 buffer's physical address, see note_text.
    text_lines: Vec<TextLine>,
    text_line_addrs: Vec<u32>,
}

impl Portraits {
    fn text_lines_for(&self, tex_addr: u32) -> Vec<TextLine> {
        if self.text_line_addrs.len() != self.text_lines.len() || tex_addr == 0 {
            return self.text_lines.clone();
        }
        self.text_lines
            .iter()
            .zip(&self.text_line_addrs)
            .filter(|(_, &addr)| addr == 0 || addr == tex_addr)
            .map(|(&line, _)| line)
            .collect()
    }
}

/// a line of text in the title's 512x128 text buffer, in its pixels, rows
/// up from its bottom: the rows its letters take and how far across they go.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TextLine {
    bottom: f32,
    top: f32,
    left: f32,
    right: f32,
}

/// a draw into Kid Icarus Uprising's 512x128 text buffer, where it writes
/// the letters its subtitles and dialogue box show.
fn is_text_buffer(registers: &[u32]) -> bool {
    read_float24(registers, REG_VIEWPORT_WIDTH) == 256.0 && read_float24(registers, REG_VIEWPORT_HEIGHT) == 64.0
}

/// keeps where the title writes letters into its text buffer, line by
/// line: a letter from its A4 font joins the line its rows overlap, and a
/// draw of anything else, the clearing that comes before new text, forgets
/// the lines it covers.
fn note_text(registers: &[u32], resources: &mut Resources, vertices: Vertices) {
    let Some(vertices) = shaded_in_order(registers, vertices) else { return };
    let viewport = (0.0, 0.0, 512.0, 128.0);
    let base = TEXTURE_UNIT_BASES[0];
    let letters = registers[REG_TEXTURE_CONFIG] & 1 != 0 && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::A4;
    let boxes = vertices.chunks_exact(3).filter_map(|triangle| {
        let corners: Vec<Screen> = triangle.iter().filter_map(|vertex| to_screen(vertex, viewport)).collect();
        (corners.len() == 3).then(|| {
            let (xs, ys) = (corners.iter().map(|c| c.x), corners.iter().map(|c| c.y));
            TextLine {
                bottom: ys.clone().fold(f32::MAX, f32::min),
                top: ys.fold(f32::MIN, f32::max),
                left: xs.clone().fold(f32::MAX, f32::min),
                right: xs.fold(f32::MIN, f32::max),
            }
        })
    });
    let target_addr = loc_register(registers, REG_COLOR_BUFFER_ADDRESS);
    let Portraits { text_lines: lines, text_line_addrs: addrs, .. } = &mut resources.portraits;
    if addrs.len() != lines.len() {
        addrs.resize(lines.len(), 0);
    }
    for found in boxes {
        if !letters {
            let mut i = 0;
            while i < lines.len() {
                let same_buf = addrs[i] == 0 || target_addr == 0 || addrs[i] == target_addr;
                if same_buf && !(lines[i].top <= found.bottom || lines[i].bottom >= found.top) {
                    lines.remove(i);
                    addrs.remove(i);
                } else {
                    i += 1;
                }
            }
            continue;
        }
        let overlap = |line: &TextLine| line.top.min(found.top) - line.bottom.max(found.bottom);
        let existing = lines.iter().enumerate().find_map(|(i, line)| {
            let same_buf = addrs[i] == 0 || target_addr == 0 || addrs[i] == target_addr;
            (same_buf && overlap(line) * 2.0 >= found.top - found.bottom).then_some(i)
        });
        match existing {
            Some(i) => {
                let line = &mut lines[i];
                *line = TextLine {
                    bottom: line.bottom.min(found.bottom),
                    top: line.top.max(found.top),
                    left: line.left.min(found.left),
                    right: line.right.max(found.right),
                };
                if addrs[i] == 0 {
                    addrs[i] = target_addr;
                }
            }
            None => {
                if lines.len() >= 64 {
                    lines.remove(0);
                    addrs.remove(0);
                }
                lines.push(found);
                addrs.push(target_addr);
            }
        }
    }
}

/// the subtitles' vertices with each line of text in the middle of the top
/// screen: every quad of the draw, the text and the title's own shadow, cut
/// Checks whether a top-screen 512x128 RGBA8 text draw is actually KIU's
/// bottom-strip battle dialogue subtitle box rather than another 512x128 text
/// draw (such as Chapter End bonus labels like "No Damage" / "Melee Finisher"
/// or Pause screen notices).
///
/// KIU's 2-line battle subtitle quad (`each_subtitle_line_goes_to_the_middle_of_the_screen`)
/// occupies `bx = 15.1..56.0` (`sy = 184..225`), `by = 97.6..361.0` (`sx = 39..302`),
/// while 1-line and short dialogue quads sit in the same bottom strip with
/// smaller height (~14..24 px) and/or narrower width (anchored near `high_by ≈ 361`),
/// with 1 or 2 identical-UV quads (text + shadow).
fn is_top_subtitle_quad(vertices: &[Vertex]) -> bool {
    if vertices.len() < 6 || !vertices.len().is_multiple_of(6) {
        return false;
    }
    let uv_bounds = |quad: &[Vertex]| -> (f32, f32, f32, f32) {
        let mut u0 = f32::MAX;
        let mut u1 = f32::MIN;
        let mut v0 = f32::MAX;
        let mut v1 = f32::MIN;
        for v in quad {
            u0 = u0.min(v.texcoords[0][0]);
            u1 = u1.max(v.texcoords[0][0]);
            v0 = v0.min(v.texcoords[0][1]);
            v1 = v1.max(v.texcoords[0][1]);
        }
        (u0, v0, u1, v1)
    };
    let (mut low_bx, mut high_bx) = (f32::MAX, f32::MIN);
    let (mut low_by, mut high_by) = (f32::MAX, f32::MIN);
    for v in &vertices[..6] {
        let w = v.clip[3];
        if !w.is_finite() || w.abs() < 1e-6 {
            return false;
        }
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        low_bx = low_bx.min(bx);
        high_bx = high_bx.max(bx);
        low_by = low_by.min(by);
        high_by = high_by.max(by);
    }
    let h = high_bx - low_bx;
    let w = high_by - low_by;
    if !(low_bx >= 2.0 && high_bx <= 72.0 && (10.0..=68.0).contains(&h)) {
        return false;
    }
    if !(w >= 8.0 && low_by >= 2.0 && high_by >= 200.0 && high_by <= 398.0) {
        return false;
    }
    let first_uv = uv_bounds(&vertices[..6]);
    for quad in vertices[6..].chunks_exact(6) {
        let uv = uv_bounds(quad);
        if (uv.0 - first_uv.0).abs() > 1e-3
            || (uv.1 - first_uv.1).abs() > 1e-3
            || (uv.2 - first_uv.2).abs() > 1e-3
            || (uv.3 - first_uv.3).abs() > 1e-3
        {
            return false;
        }
    }
    true
}

/// into one slice per line across the text buffer's rows, each slid along
/// the screen by what its line needs. none when the draw is not quads
/// across the lines the buffer holds.
#[allow(dead_code)]
fn centered_subtitle(vertices: &[Vertex], lines: &[TextLine]) -> Option<Vec<Vertex>> {
    if lines.is_empty() || vertices.is_empty() || !vertices.len().is_multiple_of(6) {
        return None;
    }
    let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
    // the quads' corners, by their texture coordinates, low and high u and v
    let corners = |quad: &[Vertex]| -> Option<[[Vertex; 2]; 2]> {
        let (us, vs) = (quad.iter().map(|v| v.texcoords[0][0]), quad.iter().map(|v| v.texcoords[0][1]));
        let (u0, u1) = (us.clone().fold(f32::MAX, f32::min), us.fold(f32::MIN, f32::max));
        let (v0, v1) = (vs.clone().fold(f32::MAX, f32::min), vs.fold(f32::MIN, f32::max));
        let at = |u: f32, v: f32| quad.iter().find(|x| near(x.texcoords[0][0], u) && near(x.texcoords[0][1], v)).copied();
        Some([[at(u0, v0)?, at(u0, v1)?], [at(u1, v0)?, at(u1, v1)?]])
    };
    let first = corners(&vertices[..6])?;
    let [u0, v0] = first[0][0].texcoords[0];
    let [u1, v1] = first[1][1].texcoords[0];
    for quad in vertices[6..].chunks_exact(6) {
        let edges = corners(quad)?;
        let [qu0, qv0] = edges[0][0].texcoords[0];
        let [qu1, qv1] = edges[1][1].texcoords[0];
        if (qu0 - u0).abs() > 1e-3
            || (qv0 - v0).abs() > 1e-3
            || (qu1 - u1).abs() > 1e-3
            || (qv1 - v1).abs() > 1e-3
        {
            return None;
        }
    }
    // the lines the quads show, low rows first, in texture terms
    let mut shown: Vec<TextLine> = lines
        .iter()
        .map(|line| TextLine { bottom: line.bottom / 128.0, top: line.top / 128.0, left: line.left / 512.0, right: line.right / 512.0 })
        .filter(|line| (line.bottom + line.top) / 2.0 > v0 && (line.bottom + line.top) / 2.0 < v1)
        .collect();
    if shown.is_empty() {
        return None;
    }
    shown.sort_by(|a, b| a.bottom.total_cmp(&b.bottom));
    // each line's slide, in clip-space units of the top screen's 400 across,
    // which a vertex's clip y holds: its middle goes to the screen's middle
    let ndc_y = |vertex: &Vertex| vertex.clip[1] / vertex.clip[3];
    let slides: Vec<f32> = shown
        .iter()
        .map(|line| {
            let middle = ((line.left + line.right) / 2.0 - u0) / (u1 - u0);
            let low = ndc_y(&first[0][0]) + (ndc_y(&first[1][0]) - ndc_y(&first[0][0])) * middle;
            -low
        })
        .collect();
    // where each slice starts and ends in v: the quad's ends, and halfway
    // between one line and the next
    let mut cuts = vec![v0];
    cuts.extend(shown.windows(2).map(|pair| (pair[0].top + pair[1].bottom) / 2.0));
    cuts.push(v1);
    let mut out = Vec::with_capacity(vertices.len() * shown.len());
    for quad in vertices.chunks_exact(6) {
        let edges = corners(quad)?;
        for (slice, slide) in slides.iter().enumerate() {
            let (from, to) = ((cuts[slice] - v0) / (v1 - v0), (cuts[slice + 1] - v0) / (v1 - v0));
            for vertex in quad {
                let side = &edges[!near(vertex.texcoords[0][0], edges[0][0].texcoords[0][0]) as usize];
                let high = !near(vertex.texcoords[0][1], side[0].texcoords[0][1]);
                let mut moved = side[0].lerp(&side[1], if high { to } else { from });
                moved.clip[1] += slide * moved.clip[3];
                out.push(moved);
            }
        }
    }
    Some(out)
}

/// a corner of the top screen and the portraits fading in and out there.
#[derive(Default)]
struct Side {
    /// the draws of the portrait the game drew this frame.
    incoming: Vec<PortraitDraw>,
    /// the portrait shown, and the one before it while it fades out.
    shown: Option<Faded>,
    leaving: Option<Faded>,
}

/// a portrait's draws, moved into the top screen's buffer, and how much of
/// it shows.
struct Faded {
    draws: Vec<PortraitDraw>,
    alpha: f32,
}

/// one draw of a portrait, its registers pointed at the top screen and its
/// vertices already shaded, so it can be drawn again after the game stops.
#[derive(Clone)]
struct PortraitDraw {
    registers: Vec<u32>,
    vertices: Vec<Vertex>,
}

impl PortraitDraw {
    /// the picture it shows, its texture's address.
    fn picture(&self) -> u32 {
        self.registers[TEXTURE_UNIT_BASES[0] + 4]
    }
}

/// whether ZAKURO_PORTRAITS asks for the bottom screen's portraits on the
/// top screen. On by default unless ZAKURO_PORTRAITS=0.
pub(crate) fn portraits_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZAKURO_PORTRAITS").map_or(true, |value| value != "0"))
}

/// whether a draw targets the bottom screen's 240x320 framebuffer.
fn is_bottom(registers: &[u32]) -> bool {
    let dimensions = registers[REG_FRAMEBUFFER_DIMENSIONS];
    let height = ((dimensions >> 12) & 0x3FF) + 1;
    height == 320 || read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
}

/// whether a bottom-screen draw is left undrawn while in play: the bottom
/// screen is not shown then, so only what is mirrored onto the top screen is
/// needed, the portraits and the small pieces of the battle display. whole
/// bottom-screen pictures and meshes that no one sees are skipped, which
/// saves decoding, vertex setup and shading for them.
fn skip_bottom_in_play(
    registers: &[u32],
    in_battle: bool,
    on_title: bool,
    title_ready: bool,
    vertex_count: u32,
) -> bool {
    if !on_bottom(registers) {
        return true;
    }
    let portrait = is_portrait(registers);
    if portrait {
        return false;
    }
    if on_title {
        return is_title_bottom_sky(registers) || title_ready;
    }
    if !in_battle {
        return true;
    }
    let base = TEXTURE_UNIT_BASES[0];
    if TextureFormat::from_raw(registers[base + 13]) == TextureFormat::A4 {
        return true;
    }
    // HUD batches reach 66 vertices for multi-digit kill/heart counters + corner
    // cloud, 54 for the 9-slice item pickup card, and 114 for the power tray with
    // its animated name bubble, while heavy 3D bottom-screen meshes stay skipped.
    vertex_count > 144
}

impl Portraits {
    fn reset_title(&mut self) {
        self.title_bottom_drawing = false;
        self.title_bottom_counted = false;
        self.title_sky_seen = false;
        self.title_sky_active = false;
        self.title_sky_missed = 0;
        self.title_sizzle_active = false;
        self.title_sizzle_seen = false;
        self.title_sizzle_missed = 0;
        self.title_frames = 0;
        self.title_captured = false;
        self.title_warmups = 0;
        self.title_alpha = 0.0;
        self.title_seen = false;
        self.title.clear();
    }
}

/// whether the GPU is currently drawing the title screen (not a battle).
fn is_on_title(portraits: &Portraits) -> bool {
    !portraits.in_battle
        && (portraits.title_bottom_drawing
            || portraits.title_sky_active
            || portraits.title_sky_seen)
}

/// whether the title screen (or its sizzle reel) is shown as single-screen
/// `TopOnly`, redirecting the logo onto the top screen and skipping bottom draws.
pub(crate) fn is_capturing_title(portraits: &Portraits) -> bool {
    (is_on_title(portraits) || portraits.title_sizzle_active)
        && !portraits.menu_override
        && !portraits.fuse_active
        && !portraits.cauldron_active
        && !portraits.chapter_active
        && !portraits.mode_active
        && !portraits.solo_active
        && !portraits.file_active
        && !portraits.weapon_active
}

/// Kid Icarus Uprising's real-time title screen sky layer on the top screen
/// (512x512 La4 cloud mask).
fn is_title_sky(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && registers[base + 1] == (512 << 16 | 512)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::La4
}

/// the bottom screen's 128x256 Etc1 sky backdrop on the title screen, which
/// is skipped when compositing the title logo onto the top screen.
fn is_title_bottom_sky(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    on_bottom(registers)
        && (registers[base + 1] == (128 << 16 | 256) || registers[base + 1] == (256 << 16 | 128))
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Etc1
}

/// Kid Icarus Uprising's Fuse Weapons A/B card header banners on the top
/// screen (two distinct 256x16 Etc1 textures drawn per frame).
fn is_fuse_card_header(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && registers[base + 1] == (256 << 16 | 16)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Etc1
}

/// Kid Icarus Uprising's 256x128 Etc1A4 result-weapon portrait on the top screen.
fn is_fuse_result_weapon(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && registers[base + 1] == (256 << 16 | 128)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Etc1A4
}

/// Kid Icarus Uprising's 64x64 Rgb565 workbench preview texture, drawn whenever
/// a result weapon is previewed on the Fuse Weapons workbench and omitted while
/// the full-screen golden fusion reveal plays.
fn is_fuse_small_preview(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && registers[base + 1] == (64 << 16 | 64)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Rgb565
}

/// whether a top-screen draw during Fuse Weapons needs per-vertex inspection to
/// pack the Result Card (`Ranged / Melee / Value` + alpha-composited weapon
/// preview) onto a single continuous background strip (`x = 254..400, y = 0..45`).
fn needs_fuse_top_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return false;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (256 << 16 | 128) && matches!(fmt, TextureFormat::Etc1A4 | TextureFormat::La4))
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        || (dims == (64 << 16 | 64)
            && matches!(fmt, TextureFormat::L8 | TextureFormat::Etc1 | TextureFormat::Rgb565))
}

/// packs the Fuse Weapons result stats (`Ranged / Melee / Value` + up to 6
/// stars) on the left and the alpha-masked result weapon + drop shadow + live
/// radial glow/sparkles on the right of `Screen::Top`'s continuous upper-right
/// filigree strip (`y = 0..45`), while omitting the 3DS battery icon so the
/// Result Card background is completely seamless.
fn adjust_fuse_top(registers: &[u32], mut shaded: Vec<Vertex>) -> Option<Vec<Vertex>> {
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    // 1. Relocate the 256x128 Etc1A4 result weapon & its drop shadow cleanly
    //    into the right half of the y = 0..52 strip at 1:1 square pixels.
    if dims == (256 << 16 | 128) && fmt == TextureFormat::Etc1A4 {
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            let nx = 369.0 + (sx - 323.0) * 0.42;
            let ny = 26.0 + (sy - 123.5) * 0.42;
            set_top_xy(v, nx, ny);
        }
        return Some(shaded);
    }

    // 2. Scale only the glyph/star triangles belonging to Ranged / Melee / Value
    //    (x >= 252, y <= 56) so up to 6 stars fit cleanly beside the weapon at
    //    matching card font size, leaving Card A/B and banner text untouched.
    if dims == (256 << 16 | 256)
        && fmt == TextureFormat::A4
        && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
    {
        for tri in shaded.chunks_exact_mut(3) {
            let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
            if pts.iter().all(|&[x, y]| x >= 252.0 && y <= 56.0) {
                for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                    let nx = 255.5 + (sx - 255.5) * 0.62;
                    let ny = 12.0 + (sy - 5.0) * 0.62;
                    set_top_xy(v, nx, ny);
                }
            }
        }
        return Some(shaded);
    }

    // Compute screen bounding box for the remaining small sprite checks.
    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_top_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 3. Omit the 3DS battery icon in the top-right corner ([382..400, 0..13]).
    if dims == (256 << 16 | 128)
        && fmt == TextureFormat::La4
        && shaded.len() == 6
        && left >= 378.0
        && right <= 402.0
        && upper >= -2.0
        && lower <= 16.0
    {
        return None;
    }

    // 4. Relocate the radial sunburst aura (64x64 L8) and twinkling sparkle
    //    particles (64x64 Etc1 / Rgb565) with the exact same transform as the
    //    weapon so they blend radially onto the single continuous background
    //    with zero box seam.
    if dims == (64 << 16 | 64) {
        let relocate = match fmt {
            TextureFormat::L8 => left >= 165.0,
            TextureFormat::Etc1 | TextureFormat::Rgb565 => left >= 200.0,
            _ => false,
        };
        if relocate {
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                let nx = 369.0 + (sx - 323.0) * 0.42;
                let ny = 26.0 + (sy - 123.5) * 0.42;
                set_top_xy(v, nx, ny);
            }
            return Some(shaded);
        }
    }

    Some(shaded)
}

/// whether a draw is Fiend's Cauldron's 10-panel parchment paper strip (`128x64 Etc1`, 300 vertices).
fn is_cauldron_mural_paper(registers: &[u32], vertex_count: usize) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && vertex_count == 300
        && registers[base + 1] == (128 << 16 | 64)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Etc1
}

/// whether a top-screen draw during Fiend's Cauldron needs per-vertex inspection
/// to separate the full un-occluded Parchment Mural (`y = 25..161`, crossfaded
/// smoothly across tier swaps) from the parked `Total Hearts` pouch (`x = 16..108,
/// y = 165..237`) and `[Intensity X.X]` card (`x = 186..324, y = 166..240`).
fn needs_cauldron_top_adjust(registers: &[u32]) -> bool {
    registers[REG_TEXTURE_CONFIG] & 1 != 0
}

fn copy_regs(registers: &[u32]) -> [u32; 0x300] {
    let mut out = [0u32; 0x300];
    let n = registers.len().min(0x300);
    out[..n].copy_from_slice(&registers[..n]);
    out
}

fn rasterize_cauldron_faded<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut vertices: Vec<Vertex>,
    alpha: f32,
) -> Option<u32> {
    let alpha = alpha.clamp(0.0, 1.0);
    if alpha <= 0.004 {
        return Some(0);
    }
    if alpha >= 0.996 {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&vertices));
        resources.portraits.adding = false;
        return res;
    }
    let mut regs_buf = copy_regs(registers);
    let regs = &mut regs_buf[..registers.len()];
    let alpha_u8 = ((alpha * 255.0).round() as u32).min(255);
    if regs[REG_COLOR_OPERATION] & 0x100 == 0 {
        regs[REG_COLOR_OPERATION] = (regs[REG_COLOR_OPERATION] & !0xFF) | 0x00E4_0100;
        regs[REG_BLEND_FUNC] = 0x01DC_0000;
        regs[REG_BLEND_COLOR] = alpha_u8 << 24;
    } else {
        let stage5 = TEV_STAGES[5];
        if (regs[stage5] & 0xF) == 0xF
            && ((regs[stage5] >> 16) & 0xF) == 0xF
            && (regs[stage5 + 1] & 0xF_00FF) == 0
            && regs[stage5 + 2] == 0
        {
            regs[stage5..stage5 + 5].copy_from_slice(&[
                0x00EF_000F,
                0,
                0x0001_0000,
                alpha_u8 << 24,
                0,
            ]);
        } else {
            for v in &mut vertices {
                v.color[3] *= alpha;
            }
        }
    }
    resources.portraits.adding = true;
    let res = rasterize(regs, memory, resources, Vertices::Shaded(&vertices));
    resources.portraits.adding = false;
    res
}

fn draw_cauldron_top<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    const MURAL_PITCH: f32 = 356.6287;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };
    let map_mural_y = |sy: f32| -> f32 { 25.0 + (sy - 44.0) * (136.0 / 155.0) };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_top_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 1. Omit off-screen helper quads and the 3DS battery icon ([382..400, 0..13]).
    if upper > 250.0 {
        return Some(0);
    }
    if dims == (256 << 16 | 128)
        && fmt == TextureFormat::La4
        && shaded.len() == 6
        && left >= 378.0
        && right <= 402.0
        && upper >= -2.0
        && lower <= 16.0
    {
        return Some(0);
    }

    // 2. Mural Layer 1: 128x64 Etc1 parchment paper (`v == 300`, all 10 panels).
    //    Closes the top-center pouch U-notch (`x = 148..252, y = 44..98`) and
    //    smoothly cross-dissolves `prev_tier -> curr_tier` when Intensity swaps!
    if dims == (128 << 16 | 64) && fmt == TextureFormat::Etc1 && shaded.len() == 300 {
        let target = ((21.02 - left) / MURAL_PITCH).round().clamp(0.0, 9.0) as u8;
        let buf_addr = registers[REG_COLOR_BUFFER_ADDRESS];
        let p = &mut resources.portraits;
        if !p.cauldron_tier_init {
            p.cauldron_tier_init = true;
            p.cauldron_curr_tier = target;
            p.cauldron_prev_tier = target;
            p.cauldron_mural_fade = 1.0;
            p.cauldron_fade_buf = buf_addr;
            p.cauldron_room_rgb = if target < 2 { [0.08, 0.09, 0.15] } else { [0.14, 0.03, 0.00] };
        } else if p.cauldron_fade_buf == 0 || p.cauldron_fade_buf == buf_addr {
            p.cauldron_fade_buf = buf_addr;
            if target != p.cauldron_curr_tier {
                if p.cauldron_mural_fade >= 0.30 {
                    p.cauldron_prev_tier = p.cauldron_curr_tier;
                }
                p.cauldron_curr_tier = target;
                p.cauldron_mural_fade = 0.06;
            } else if p.cauldron_mural_fade < 1.0 {
                p.cauldron_mural_fade = (p.cauldron_mural_fade + 0.055).min(1.0);
            }
        }
        let (prev, curr, fade) = (p.cauldron_prev_tier, p.cauldron_curr_tier, p.cauldron_mural_fade);
        let s_fade = fade * fade * (3.0 - 2.0 * fade);

        let build_paper = |mut verts: Vec<Vertex>, dx: f32| -> Vec<Vertex> {
            for v in &mut verts {
                let [mut sx, mut sy] = to_top_xy(v);
                sx += dx;
                if (136.0..=264.0).contains(&sx) && (75.0..=118.0).contains(&sy) {
                    sy = 43.8;
                }
                set_top_xy(v, sx, map_mural_y(sy));
            }
            verts
        };

        if s_fade < 0.996 && prev != curr {
            let prev_dx = (curr as i32 - prev as i32) as f32 * MURAL_PITCH;
            let prev_verts = build_paper(shaded.clone(), prev_dx);
            let _ = rasterize_cauldron_faded(registers, memory, resources, prev_verts, 1.0);
            let curr_verts = build_paper(shaded, 0.0);
            return rasterize_cauldron_faded(registers, memory, resources, curr_verts, s_fade);
        }
        let curr_verts = build_paper(shaded, 0.0);
        return rasterize_cauldron_faded(registers, memory, resources, curr_verts, 1.0);
    }

    // 3. Mural Layer 2: 64x64 La4 ornate border (`v == 960`), dropping the center
    //    U-bracket quads and bridging the top border bar seamlessly across x = 148..252.
    if dims == (64 << 16 | 64) && fmt == TextureFormat::La4 && shaded.len() >= 480 {
        let (prev, curr, fade) = (
            resources.portraits.cauldron_prev_tier,
            resources.portraits.cauldron_curr_tier,
            resources.portraits.cauldron_mural_fade,
        );
        let s_fade = fade * fade * (3.0 - 2.0 * fade);

        let build_border = |verts: &[Vertex], dx: f32| -> Vec<Vertex> {
            let mut out = Vec::with_capacity(verts.len());
            for quad in verts.chunks_exact(6) {
                let mut q_pts = [[0.0f32; 2]; 6];
                let (mut ql, mut qu, mut qr, mut qd) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                for (i, v) in quad.iter().enumerate() {
                    let [sx, sy] = to_top_xy(v);
                    let sx = sx + dx;
                    q_pts[i] = [sx, sy];
                    ql = ql.min(sx);
                    qu = qu.min(sy);
                    qr = qr.max(sx);
                    qd = qd.max(sy);
                }
                if ql >= 15.0 && qr <= 385.0 {
                    // Omit the U-bracket framing the old center Pouch
                    if ql >= 136.0 && qr <= 264.0 && qd > 56.0 && qu < 118.0 {
                        continue;
                    }
                    // Stretch left & right top horizontal border bars to meet at x = 200
                    if qu >= 33.0 && qd <= 58.0 {
                        if (qr - 148.0).abs() < 18.0 {
                            for pt in &mut q_pts {
                                if pt[0] > qr - 3.0 {
                                    pt[0] = 200.5;
                                }
                            }
                        } else if (ql - 252.0).abs() < 18.0 {
                            for pt in &mut q_pts {
                                if pt[0] < ql + 3.0 {
                                    pt[0] = 199.5;
                                }
                            }
                        }
                    }
                }
                for (v, [sx, sy]) in quad.iter().copied().zip(q_pts) {
                    let mut nv = v;
                    set_top_xy(&mut nv, sx, map_mural_y(sy));
                    out.push(nv);
                }
            }
            out
        };

        if s_fade < 0.996 && prev != curr {
            let prev_dx = (curr as i32 - prev as i32) as f32 * MURAL_PITCH;
            let prev_verts = build_border(&shaded, prev_dx);
            let _ = rasterize_cauldron_faded(registers, memory, resources, prev_verts, 1.0 - s_fade);
            let curr_verts = build_border(&shaded, 0.0);
            return rasterize_cauldron_faded(registers, memory, resources, curr_verts, s_fade);
        }
        let curr_verts = build_border(&shaded, 0.0);
        return rasterize_cauldron_faded(registers, memory, resources, curr_verts, 1.0);
    }

    // 4. Mural Layer 3: 256x256 L4 Reapers, Treasure Chests, Hearts & Winged Pit (`v == 2166`).
    if dims == (256 << 16 | 256) && fmt == TextureFormat::L4 && shaded.len() >= 600 {
        let (prev, curr, fade) = (
            resources.portraits.cauldron_prev_tier,
            resources.portraits.cauldron_curr_tier,
            resources.portraits.cauldron_mural_fade,
        );
        let s_fade = fade * fade * (3.0 - 2.0 * fade);

        let build_ink = |mut verts: Vec<Vertex>, dx: f32| -> Vec<Vertex> {
            for v in &mut verts {
                let [sx, sy] = to_top_xy(v);
                set_top_xy(v, sx + dx, map_mural_y(sy));
            }
            verts
        };

        if s_fade < 0.996 && prev != curr {
            let prev_dx = (curr as i32 - prev as i32) as f32 * MURAL_PITCH;
            let prev_verts = build_ink(shaded.clone(), prev_dx);
            let _ = rasterize_cauldron_faded(registers, memory, resources, prev_verts, 1.0 - s_fade);
            let curr_verts = build_ink(shaded, 0.0);
            return rasterize_cauldron_faded(registers, memory, resources, curr_verts, s_fade);
        }
        let curr_verts = build_ink(shaded, 0.0);
        return rasterize_cauldron_faded(registers, memory, resources, curr_verts, 1.0);
    }

    // 5. Outer room wall (`v == 120`, `upper < 0`) & bottom vignette (`v == 120`, `upper > 40`):
    //    on the outer wall pass (right after the 3 mural draws and right BEFORE the Pouch,
    //    Intensity card, and Top Header draw!), record smoothly-lerped `cauldron_room_rgb`
    //    and wipe `y = 0..25` & `y = 161..240` to `#000000` so overhanging Reaper scythes
    //    never bleed behind the parked Header, Pouch, or Intensity card.
    if dims == (256 << 16 | 256) && fmt == TextureFormat::L4 && shaded.len() == 120 {
        if upper < 0.0 {
            if let Some(&v0) = shaded.first() {
                let buf_addr = registers[REG_COLOR_BUFFER_ADDRESS];
                if resources.portraits.cauldron_fade_buf == 0 || resources.portraits.cauldron_fade_buf == buf_addr {
                    for c in 0..3 {
                        resources.portraits.cauldron_room_rgb[c] +=
                            (v0.color[c] - resources.portraits.cauldron_room_rgb[c]) * 0.11;
                    }
                }
                let mut regs_buf = copy_regs(registers);
                let regs = &mut regs_buf[..registers.len()];
                regs[REG_FACE_CULLING] = 0;
                regs[REG_ALPHA_TEST] &= !1;
                regs[REG_DEPTH_COLOR_MASK] = (regs[REG_DEPTH_COLOR_MASK] & !0xFF) | 0x0F00;
                regs[REG_PRIMITIVE_CONFIG] &= !(3 << 8);
                regs[REG_COLOR_OPERATION] = (regs[REG_COLOR_OPERATION] & !0xFF) | 0x00E4_0100;
                regs[REG_BLEND_FUNC] = 0x0101_0000;
                for (stage, &tbase) in TEV_STAGES.iter().enumerate() {
                    let src = if stage == 0 { 0xE | (0xE << 16) } else { 0xF | (0xF << 16) };
                    regs[tbase..tbase + 5].copy_from_slice(&[src, 0, 0, 0xFF00_0000, 0]);
                }
                let mk = |nx: f32, ny: f32| {
                    let mut v = v0;
                    v.color = [0.0, 0.0, 0.0, 1.0];
                    set_top_xy(&mut v, nx, ny);
                    v
                };
                let wipe = [
                    mk(-4.0, -2.0), mk(404.0, -2.0), mk(404.0, 25.0),
                    mk(-4.0, -2.0), mk(404.0, 25.0), mk(-4.0, 25.0),
                    mk(-4.0, 161.0), mk(404.0, 161.0), mk(404.0, 242.0),
                    mk(-4.0, 161.0), mk(404.0, 242.0), mk(-4.0, 242.0),
                ];
                resources.portraits.adding = true;
                let res = rasterize(regs, memory, resources, Vertices::Shaded(&wipe));
                resources.portraits.adding = false;
                return res;
            }
        }
        return Some(0);
    }

    // 6. Park all 6 draws of the `[Intensity X.X]` card (`156 <= upper <= 242`) at
    //    `x_center = 255.0` (`[186.4..323.6, 165.9..239.5]`) below the mural.
    if (156.0..=242.0).contains(&upper) && lower <= 252.0 && !(left < -20.0 && right > 420.0) {
        if dims == (64 << 16 | 64) && fmt == TextureFormat::La4 && (168.0..185.0).contains(&upper) {
            resources.portraits.cauldron_card_dx = 255.0 - (left + right) * 0.5;
        }
        let dx = resources.portraits.cauldron_card_dx;
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            set_top_xy(v, sx + dx, sy - 6.0);
        }
        return rasterize_cauldron_faded(registers, memory, resources, shaded, 1.0);
    }

    // 7. Shift all 4 chunks of the top instruction header (`256x256 A4`, `topo == 0`,
    //    `lower <= 28`) uniformly up by 3px so they sit on a single crisp baseline in `y = 6..21`.
    if dims == (256 << 16 | 256)
        && fmt == TextureFormat::A4
        && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
        && lower <= 28.0
    {
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            set_top_xy(v, sx, sy - 3.0);
        }
        return rasterize_cauldron_faded(registers, memory, resources, shaded, 1.0);
    }

    // 8. Park the `Total Hearts` pouch + highlight + `"Total"` tag + red digits
    //    (`150 <= left .. right <= 252`, `12 <= upper .. lower <= 105`) cleanly
    //    at `[16.0..102.4, 166.0..233.8]` (`0.80x` uniform) below the mural!
    if left >= 150.0 && right <= 252.0 && upper >= 12.0 && lower <= 105.0 {
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            set_top_xy(v, 16.0 + (sx - 146.0) * 0.80, 166.0 + (sy - 21.0) * 0.80);
        }
        return rasterize_cauldron_faded(registers, memory, resources, shaded, 1.0);
    }

    rasterize_cauldron_faded(registers, memory, resources, shaded, 1.0)
}

/// whether a bottom-screen draw during Fiend's Cauldron needs per-vertex inspection
/// to record `cauldron_cursor_x` and omit the circular Skull dial body (keeping
/// only the native vertical line + upward blue pointer arrow under `0..9`).
fn needs_cauldron_bot_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return true;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (256 << 16 | 128) && fmt == TextureFormat::Rgba4)
        || (dims == (64 << 16 | 64) && fmt == TextureFormat::Rgb565)
        || (dims == (128 << 16 | 128) && fmt == TextureFormat::L4)
}

fn adjust_cauldron_bot(
    registers: &[u32],
    shaded: Vec<Vertex>,
    portraits: &mut Portraits,
) -> Option<Vec<Vertex>> {
    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_bot_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 0. While the Parchment Mural is behind the Cauldron, omit the untextured
    //    full-screen tint quad (`v == 6`) and the two `128x128 L4` floor spotlight
    //    ellipses (`v == 6`, `lower > 150`) so `bot_screen` is pure `#000000`
    //    outside the Cauldron, Pan, Flames, Hearts, and UI!
    if portraits.cauldron_hud_active {
        if !textured
            && !portraits.cauldron_pot_seen
            && shaded.len() == 6
            && (right - left) > 300.0
            && (lower - upper) > 220.0
        {
            return None;
        }
        if textured
            && dims == (128 << 16 | 128)
            && fmt == TextureFormat::L4
            && shaded.len() == 6
            && lower > 150.0
        {
            return None;
        }
    }
    if !textured {
        return Some(shaded);
    }

    // 1. Omit twinkling sparkle sprite on the Skull dial (`64x64 Rgb565`, y in 30..90).
    if dims == (64 << 16 | 64)
        && fmt == TextureFormat::Rgb565
        && shaded.len() == 6
        && upper >= 28.0
        && lower <= 92.0
    {
        return None;
    }

    if dims == (256 << 16 | 128) && fmt == TextureFormat::Rgba4 {
        // 2. Skull dial + slider rail (`v == 48`, width ~44, lower ~79):
        //    record exact `cursor_x` + mark interactive HUD active, keeping only
        //    any thin vertical rail above the 0..9 bar (`sy <= 16.0`).
        let width = right - left;
        if (36..=60).contains(&shaded.len())
            && (38.0..=52.0).contains(&width)
            && (68.0..=90.0).contains(&lower)
        {
            let cx = (left + right) * 0.5;
            portraits.cauldron_cursor_x = cx;
            portraits.cauldron_cursor_seen = true;
            portraits.cauldron_hud_active = true;
            let mut kept = Vec::with_capacity(6);
            for tri in shaded.chunks_exact(3) {
                let pts = [to_bot_xy(&tri[0]), to_bot_xy(&tri[1]), to_bot_xy(&tri[2])];
                if pts.iter().all(|&[x, y]| (x - cx).abs() <= 6.0 && y <= 16.5) {
                    kept.extend_from_slice(tri);
                }
            }
            return (!kept.is_empty()).then_some(kept);
        }

        // 3. Omit any Skull dial ring, ears, or release-ripple quads in `y = 31.5..96.0`.
        if upper >= 31.5 && lower <= 96.0 {
            return None;
        }
    }

    Some(shaded)
}

/// horizontal clip-space scale widening the 320px bottom screen across the full
/// `x = 18.67..408.0` (`389.33 x 127.0`) 16:9 Chapter Select lower deck while
/// keeping every Chapter Card, laurel, and glyph at 1:1 unstretched aspect.
pub const CHAPTER_BOT_KX: f32 = 0.450;

fn needs_chapter_top_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return false;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        || (dims == (256 << 16 | 128) && fmt == TextureFormat::La4)
        || (dims == (32 << 16 | 32) && fmt == TextureFormat::La4)
}

fn adjust_chapter_top(
    registers: &[u32],
    mut shaded: Vec<Vertex>,
    portraits: &mut Portraits,
) -> Option<Vec<Vertex>> {
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    // Pull the Laurel Wreath (`32x32 La4`, 2 quads: wreath + drop-shadow) up and
    // inward around the live `"Chapter X"` / `"Chapter XX"` bounds so it hugs the
    // text snugly without drooping across the underline bar.
    if dims == (32 << 16 | 32) && fmt == TextureFormat::La4 && shaded.len() == 12 {
        let mut left = f32::MAX;
        for v in &shaded {
            left = left.min(to_top_xy(v)[0]);
        }
        let (nl, nr) = if portraits.chapter_num_bounds.1 > portraits.chapter_num_bounds.0 + 20.0 {
            portraits.chapter_num_bounds
        } else {
            (30.1, 93.2)
        };
        let num_cx = (nl + nr) * 0.5;
        let num_w = nr - nl;
        let kx = ((num_w + 15.5) / 94.9).clamp(0.72, 0.90);
        const KY: f32 = 0.82;
        let is_shadow = left > 17.5;
        let (src_cx, src_cy) = if is_shadow { (66.8, 49.0) } else { (61.45, 47.2) };
        let (dst_cx, dst_cy) = if is_shadow {
            (num_cx + 1.6, 42.6 + 1.5)
        } else {
            (num_cx, 42.6)
        };
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            set_top_xy(v, dst_cx + (sx - src_cx) * kx, dst_cy + (sy - src_cy) * KY);
        }
        return Some(shaded);
    }

    if dims == (256 << 16 | 256) && fmt == TextureFormat::A4 {
        for v in &shaded {
            let [sx, sy] = to_top_xy(v);
            if (24.0..=60.0).contains(&sy) && (25.0..=395.0).contains(&sx) {
                portraits.chapter_frame_title_right = portraits.chapter_frame_title_right.max(sx);
                if (28.0..=54.0).contains(&sy) && sx <= 106.0 {
                    let (l, r) = portraits.chapter_frame_num_bounds;
                    portraits.chapter_frame_num_bounds = (
                        if l == 0.0 { sx } else { l.min(sx) },
                        r.max(sx),
                    );
                }
            }
        }
        return Some(shaded);
    }

    if dims == (256 << 16 | 128) && fmt == TextureFormat::La4 && shaded.len() == 6 {
        let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for v in &shaded {
            let [sx, sy] = to_top_xy(v);
            left = left.min(sx);
            upper = upper.min(sy);
            right = right.max(sx);
            lower = lower.max(sy);
        }
        if left >= 378.0 && right <= 402.0 && upper >= -2.0 && lower <= 16.0 {
            portraits.frontend_menu_seen = true;
            return None;
        }
    }

    Some(shaded)
}

fn adjust_chapter_bot(registers: &[u32], mut shaded: Vec<Vertex>) -> Option<Vec<Vertex>> {
    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };
    let set_bot_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((320.0 - nx) / 160.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_bot_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 1. Omit the 3DS book-unroll page cylinder, gutter shadow, and dark sanctuary-floor
    //    backing (`256x256 Etc1`, `height > 200`) across the entire transition & idle state
    //    so all Chapter Cards appear cleanly without a 0.45x-compressed black floor slab!
    if !textured && (lower - upper) > 180.0 && (right - left) < 310.0 {
        return None;
    }
    if textured
        && dims == (256 << 16 | 128)
        && fmt == TextureFormat::Etc1
        && shaded.len() > 200
        && (right - left) <= 1200.0
    {
        return None;
    }
    if textured
        && dims == (256 << 16 | 256)
        && fmt == TextureFormat::Etc1
        && (lower - upper) > 200.0
    {
        return None;
    }

    // Keep the full-screen white paper backdrop quad (`v == 66`, `-1365..1580`) intact.
    if textured
        && dims == (256 << 16 | 128)
        && fmt == TextureFormat::Etc1
        && (right - left) > 1200.0
    {
        return Some(shaded);
    }

    // 2. Bottom navigation rail (`lower >= 194` and not an off-screen y > 1000 sprite):
    //    - Back arrow (`right <= 65`): pin left at 1:1 aspect
    //    - X : Gear (`left >= 215`): pin right (clear of Rose Crest) at 1:1 aspect
    //    - Scrollbar rail (`16x64 La4`, wide): span cleanly between Back and X : Gear
    //    - Scrollbar `[ || ]` thumb (`16x64 La4`, narrow): track rail position at 1:1 aspect
    if lower >= 194.0 && upper < 500.0 {
        if right <= 65.0 && upper >= 190.0 {
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, 3.0 + (sx - 3.0) * CHAPTER_BOT_KX, sy);
            }
            return Some(shaded);
        }
        if left >= 215.0 && upper >= 190.0 {
            const GEAR_SCALE: f32 = 1.32;
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(
                    v,
                    280.0 - (286.0 - sx) * (CHAPTER_BOT_KX * GEAR_SCALE),
                    195.0 + (sy - 195.0) * GEAR_SCALE,
                );
            }
            return Some(shaded);
        }
        if textured && dims == (16 << 16 | 64) && fmt == TextureFormat::La4 {
            const RAIL_OLD_L: f32 = 80.1;
            const RAIL_OLD_R: f32 = 243.9;
            const RAIL_NEW_L: f32 = 44.0;
            const RAIL_NEW_R: f32 = 242.0;
            if (right - left) > 90.0 {
                for v in &mut shaded {
                    let [sx, sy] = to_bot_xy(v);
                    let t = (sx - RAIL_OLD_L) / (RAIL_OLD_R - RAIL_OLD_L);
                    set_bot_xy(v, RAIL_NEW_L + t * (RAIL_NEW_R - RAIL_NEW_L), sy);
                }
            } else {
                let cx = (left + right) * 0.5;
                let t = ((cx - RAIL_OLD_L) / (RAIL_OLD_R - RAIL_OLD_L)).clamp(0.0, 1.0);
                let cx_new = RAIL_NEW_L + t * (RAIL_NEW_R - RAIL_NEW_L);
                for v in &mut shaded {
                    let [sx, sy] = to_bot_xy(v);
                    set_bot_xy(v, cx_new + (sx - cx) * CHAPTER_BOT_KX, sy);
                }
            }
            return Some(shaded);
        }
    }

    // 3. All 25 Chapter Card thumbnails, black ink frames, laurels, & pink rose-thorn
    //    selection border: widen horizontal frustum by `1 / CHAPTER_BOT_KX` (~711 logical
    //    px across) so ~7-8 unstretched 1:1 Chapter Cards fill the full 16:9 width!
    for v in &mut shaded {
        let [sx, sy] = to_bot_xy(v);
        set_bot_xy(v, 160.0 + (sx - 160.0) * CHAPTER_BOT_KX, sy);
    }
    Some(shaded)
}

/// vertical & horizontal 1:1 scale factor lifting Mode Select's `320x240` bottom-screen
/// interlocking button layout cleanly over the Top Screen's lined parchment & wings.
pub const MODE_BOT_SCALE: f32 = 0.81;
/// horizontal left origin in `426.67`-wide 16:9 space so left ink leaves (`0..68`)
/// and right angel wings (`334..426`) frame the interlocking Mode Select buttons.
pub const MODE_BOT_X_ORIGIN: f32 = 74.0;

fn needs_mode_top_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return false;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (32 << 16 | 8) && fmt == TextureFormat::Rgb565)
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::La4)
        || fmt == TextureFormat::L4
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        || (dims == (256 << 16 | 128) && fmt == TextureFormat::La4)
}

fn adjust_mode_top(
    registers: &[u32],
    mut shaded: Vec<Vertex>,
    portraits: &mut Portraits,
) -> Option<Vec<Vertex>> {
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        return Some(shaded);
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    // 1. Omit the redundant top-left header underline (`32x8 Rgb565`), header title
    //    (`256x256 La4`), and small mode silhouette icon (non-`256x256` `L4` on Solo,
    //    or `66x66` `256x256 L4` header silhouette at `[238..305, 21..88]` on Vault),
    //    keeping 100% of the `256x256 L4` wings, vines, leaves, and undulating black wave!
    if (dims == (32 << 16 | 8) && fmt == TextureFormat::Rgb565)
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::La4)
        || (dims != (256 << 16 | 256) && fmt == TextureFormat::L4)
    {
        return None;
    }
    if dims == (256 << 16 | 256) && fmt == TextureFormat::L4 && shaded.len() <= 60 {
        let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for v in &shaded {
            let [sx, sy] = to_top_xy(v);
            left = left.min(sx);
            upper = upper.min(sy);
            right = right.max(sx);
            lower = lower.max(sy);
        }
        if (right - left) <= 85.0
            && (lower - upper) <= 85.0
            && left >= 210.0
            && right <= 355.0
            && lower <= 105.0
        {
            return None;
        }
    }

    // 2. Omit the 3DS battery icon (`[382..400, 0..13]`).
    if dims == (256 << 16 | 128) && fmt == TextureFormat::La4 && shaded.len() == 6 {
        let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for v in &shaded {
            let [sx, sy] = to_top_xy(v);
            left = left.min(sx);
            upper = upper.min(sy);
            right = right.max(sx);
            lower = lower.max(sy);
        }
        if left >= 378.0 && right <= 402.0 && upper >= -2.0 && lower <= 16.0 {
            portraits.frontend_menu_seen = true;
            return None;
        }
    }

    // 3. Keep the bottom description blurb (`256x256 A4`) at exact 1:1 unstretched aspect!
    if dims == (256 << 16 | 256) && fmt == TextureFormat::A4 {
        const SX: f32 = 400.0 / 426.66666;
        for tri in shaded.chunks_exact_mut(3) {
            let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
            let cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0;
            if cy >= 200.0 {
                for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                    set_top_xy(v, 200.0 + (sx - 200.0) * SX, sy - 2.0);
                }
            }
        }
    }

    Some(shaded)
}

fn draw_mode_bot<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0
        || shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4)
    {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };
    let set_wide_xy = |v: &mut Vertex, wx: f32, wy: f32| {
        let nx = wx * 0.75;
        let w = v.clip[3];
        v.clip[0] = ((240.0 - wy) / 120.0 - 1.0) * w;
        v.clip[1] = ((320.0 - nx) / 160.0 - 1.0) * w;
    };

    // 1. Replace Mode Select's `128x256 L4` pastel backdrop with pure `#0000FF` (`0, 0, 1`)
    //    so the Top Screen's lined parchment, wings, and vines shine through behind the buttons!
    if dims == (128 << 16 | 256) && fmt == TextureFormat::L4 {
        let mut chroma_buf = copy_regs(registers);
        let chroma_regs = &mut chroma_buf[..registers.len()];
        chroma_regs[REG_TEXTURE_CONFIG] &= !0x7;
        for (stage, &stage_base) in TEV_STAGES.iter().enumerate() {
            let vals = match stage {
                0 => [0x000E_000E, 0, 0, 0xFFFF_0000, 0],
                _ => [0x000F_000F, 0, 0, 0, 0],
            };
            chroma_regs[stage_base..stage_base + 5].copy_from_slice(&vals);
        }
        resources.portraits.adding = true;
        let res = rasterize(chroma_regs, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 2. Lift all 8 interlocking buttons (`Solo`, `StreetPass`, `Vault`, `Together`,
    //    `Back`, `Options`, `AR`, `Manual`) at exact 1:1 unstretched aspect ratio!
    for v in &mut shaded {
        let [sx, sy] = to_bot_xy(v);
        set_wide_xy(
            v,
            MODE_BOT_X_ORIGIN + sx * MODE_BOT_SCALE,
            2.0 + sy * MODE_BOT_SCALE,
        );
    }
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        res
}

/// `[orig_cx, orig_cy, dst_cx, dst_cy, scale]` in `320x240` -> `426.67x240` widescreen
/// coordinates for Solo Hub's 7 repurposed bottom-screen UI elements (Option B Perimeter).
pub const SOLO_PERIMETER_SLOTS: [[f32; 5]; 7] = [
    // 0: Move Out! (foot of stairs, above description blurb)
    [133.0, 62.0, 213.33333, 165.0, 0.44],
    // 1: Treasure Hunt (top-right corner)
    [282.0, 40.0, 368.0, 38.0, 0.84],
    // 2: Practice Range (mid-left edge)
    [34.0, 145.0, 54.0, 124.0, 0.80],
    // 3: Arms Altar (mid-right edge)
    [256.0, 128.0, 366.0, 121.0, 0.76],
    // 4: Idol Toss (bottom-left-center)
    [130.0, 204.0, 76.0, 196.0, 0.76],
    // 5: Gear (bottom-right)
    [265.0, 208.0, 368.0, 210.0, 0.74],
    // 6: Back (bottom-left corner)
    [21.0, 216.0, 22.0, 218.0, 0.86],
];

fn needs_solo_top_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return false;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (128 << 16 | 128) && fmt == TextureFormat::L4)
        || (dims == (64 << 16 | 64) && fmt == TextureFormat::Rgba4)
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        || (dims == (256 << 16 | 128) && fmt == TextureFormat::La4)
}

fn adjust_solo_top(
    registers: &[u32],
    mut shaded: Vec<Vertex>,
    portraits: &mut Portraits,
) -> Option<Vec<Vertex>> {
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        return Some(shaded);
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_top_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 1. Omit the 3DS battery icon (`[382..400, 0..13]`).
    if dims == (256 << 16 | 128)
        && fmt == TextureFormat::La4
        && shaded.len() == 6
        && left >= 378.0
        && right <= 402.0
        && upper >= -2.0
        && lower <= 16.0
    {
        portraits.frontend_menu_seen = true;
        return None;
    }

    // 2. Omit the giant lower-right black swirl (`128x128 L4`, `[274..420, 106..245]`)
    //    so the live 3D Sanctuary wall & floor stay unobstructed.
    if dims == (128 << 16 | 128)
        && fmt == TextureFormat::L4
        && shaded.len() == 12
        && left >= 250.0
        && lower >= 220.0
    {
        return None;
    }

    // Shared 1:1 unstretched transform moving the Total Hearts pouch + digits
    // from lower-right (`357.0, 177.0`) to top-left (`46.0, 42.0` on `400x240`).
    const POUCH_SY: f32 = 0.78;
    const POUCH_SX: f32 = POUCH_SY * (400.0 / 426.66666);

    // 3. Relocate Total Hearts white pouch + drop shadow (`64x64 Rgba4`, `v == 6`,
    //    `[314..402, 131..223]`), leaving the bottom bar vignette (`v == 12`) untouched.
    if dims == (64 << 16 | 64)
        && fmt == TextureFormat::Rgba4
        && shaded.len() == 6
        && left >= 300.0
        && upper >= 120.0
    {
        for v in &mut shaded {
            let [sx, sy] = to_top_xy(v);
            set_top_xy(
                v,
                46.0 + (sx - 357.0) * POUCH_SX,
                42.0 + (sy - 177.0) * POUCH_SY,
            );
        }
        return Some(shaded);
    }

    // 4. `256x256 A4` text batches:
    //    - Hearts-pouch digits (`x >= 308, y in 180..=206.5`) follow the pouch to top-left!
    //    - Bottom-center description blurbs (`y >= 207`) scale neatly at 1:1 aspect in the
    //      center gap (`x_center = 200, y_center = 225.5`) between `Idol Toss` and `Gear`.
    if dims == (256 << 16 | 256) && fmt == TextureFormat::A4 {
        const BLURB_SY: f32 = 0.80;
        const BLURB_SX: f32 = BLURB_SY * (400.0 / 426.66666);
        for tri in shaded.chunks_exact_mut(3) {
            let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
            if pts.iter().all(|&[x, y]| x >= 308.0 && (180.0..=206.5).contains(&y)) {
                for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                    set_top_xy(
                        v,
                        46.0 + (sx - 357.0) * POUCH_SX,
                        42.0 + (sy - 177.0) * POUCH_SY,
                    );
                }
            } else if pts.iter().all(|&[_, y]| y >= 207.0) {
                for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                    set_top_xy(
                        v,
                        200.0 + (sx - 200.0) * BLURB_SX,
                        225.5 + (sy - 221.0) * BLURB_SY,
                    );
                }
            }
        }
        return Some(shaded);
    }

    Some(shaded)
}

fn draw_solo_bot<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };
    let set_wide_xy = |v: &mut Vertex, wx: f32, wy: f32| {
        let nx = wx * 0.75;
        let w = v.clip[3];
        v.clip[0] = ((240.0 - wy) / 120.0 - 1.0) * w;
        v.clip[1] = ((320.0 - nx) / 160.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_bot_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 1. Replace the single full-screen `256x256 Etc1` floor quad (`[-3.8..323.8, -2.9..242.9]`)
    //    with a pure `#FF00FF` (`1.0, 0.0, 1.0`) chroma canvas so `present_hub.frag` keys out
    //    every dark outline & `L4` drop shadow analytically with zero fringe.
    if textured
        && dims == (256 << 16 | 256)
        && fmt == TextureFormat::Etc1
        && shaded.len() == 6
        && left <= 2.0
        && right >= 318.0
        && upper <= 2.0
        && lower >= 238.0
    {
        let mut chroma_regs = registers.to_vec();
        chroma_regs[REG_TEXTURE_CONFIG] &= !0x7;
        for (stage, &stage_base) in TEV_STAGES.iter().enumerate() {
            let vals = match stage {
                0 => [0x000E_000E, 0, 0, 0xFFFF_00FF, 0],
                _ => [0x000F_000F, 0, 0, 0, 0],
            };
            chroma_regs[stage_base..stage_base + 5].copy_from_slice(&vals);
        }
        resources.portraits.adding = true;
        let res = rasterize(&chroma_regs, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 2. Classify any full-screen fade quad (`> 300 x 220`) and pass it through unmodified
    //    so `present_hub.frag` normalizes black/white transitions cleanly.
    if (right - left) >= 300.0 && (lower - upper) >= 220.0 {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 3. Repurpose each of KIU's 7 station draw groups into its 1:1 Perimeter Frame slot!
    let cx = (left + right) * 0.5;
    let cy = (upper + lower) * 0.5;
    let slot_idx = if cx < 70.0 && cy >= 195.0 {
        6 // Back
    } else if cx < 70.0 && cy >= 80.0 {
        2 // Practice Range
    } else if cx < 195.0 && cy >= 175.0 {
        4 // Idol Toss
    } else if cx < 195.0 {
        0 // Move Out!
    } else if cy <= 68.0 {
        1 // Treasure Hunt
    } else if cy <= 172.0 {
        3 // Arms Altar
    } else {
        5 // Gear
    };

    let [ox, oy, dx, dy, scale] = SOLO_PERIMETER_SLOTS[slot_idx];
    for v in &mut shaded {
        let [sx, sy] = to_bot_xy(v);
        set_wide_xy(v, dx + (sx - ox) * scale, dy + (sy - oy) * scale);
    }

        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        res
}

/// 1:1 unstretched scale factor for File Select's 3 leather-bound Tomes (`I / II / III`)
/// and connected speech-bubble pointer tail in `426.67x240` widescreen space (Option A).
pub const FILE_BOT_SCALE: f32 = 0.66;
/// vertical destination center (in `240p` space) for Tome shelf center `(160.0, 95.0)`.
pub const FILE_BOT_TOMES_Y: f32 = 178.0;

fn needs_file_top_adjust(registers: &[u32]) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return false;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    (dims == (64 << 16 | 8) && fmt == TextureFormat::Rgb565)
        || (dims == (512 << 16 | 256) && fmt == TextureFormat::L4)
        || (dims == (16 << 16 | 32) && fmt == TextureFormat::La4)
        || (dims == (64 << 16 | 64) && fmt == TextureFormat::Rgba4)
        || (dims == (128 << 16 | 128) && matches!(fmt, TextureFormat::Rgba8 | TextureFormat::Etc1A4))
        || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        || (dims == (256 << 16 | 128) && fmt == TextureFormat::La4)
}

fn adjust_file_top(
    registers: &[u32],
    mut shaded: Vec<Vertex>,
    portraits: &mut Portraits,
) -> Option<Vec<Vertex>> {
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        return Some(shaded);
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_top_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_top_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 1. Omit the 3DS battery icon (`[382..400, 0..13]`).
    if dims == (256 << 16 | 128)
        && fmt == TextureFormat::La4
        && shaded.len() == 6
        && left >= 378.0
        && right <= 402.0
        && upper >= -2.0
        && lower <= 16.0
    {
        portraits.frontend_menu_seen = true;
        return None;
    }

    // 2. `64x8 Rgb565`: keep full-screen diagonal pastel rainbow washes (`v <= 300`),
    //    and scale only the `v == 804` pink/blue spiral behind the Speech Bubble!
    if dims == (64 << 16 | 8) && fmt == TextureFormat::Rgb565 && shaded.len() <= 300 {
        return Some(shaded);
    }

    // 3. `512x256 L4` ink art:
    //    - Omit the 3DS bezel-bridge vertical stem (`v == 6`, `lower > 300.0 && left < 265.0`)
    //      so KIU's sharp bottom-screen pointer tail connects directly into Tome I / II / III!
    //    - Tuck the undulating bottom wave (`v >= 48 && right - left > 500.0`) slightly lower
    //      (`+10 px`) so it frames the bottom edge cleanly under the 3 Tomes.
    //    - Keep outer corner vines & wings (`upper < -40 || left > 350 || upper > 140`) as-is.
    if dims == (512 << 16 | 256) && fmt == TextureFormat::L4 {
        if (shaded.len() == 6 && lower > 300.0 && left < 265.0)
            || (left > 350.0 && upper < 100.0)
        {
            return None;
        }
        if shaded.len() >= 48 && (right - left) > 500.0 {
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_top_xy(v, sx, sy + 10.0);
            }
            return Some(shaded);
        }
        if upper < -40.0 || upper > 140.0 {
            return Some(shaded);
        }
    }

    // 4. Scale & center the entire Speech-Bubble Summary Group (`1:1` unstretched aspect!)
    //    into the upper half (`y ≈ 6..126`), centered directly above the 3 Tomes!
    const BUBBLE_SY: f32 = 0.68;
    const BUBBLE_SX: f32 = BUBBLE_SY * (400.0 / 426.66666);
    for v in &mut shaded {
        let [sx, sy] = to_top_xy(v);
        set_top_xy(
            v,
            200.0 + (sx - 200.0) * BUBBLE_SX,
            68.0 + (sy - 96.0) * BUBBLE_SY,
        );
    }
    Some(shaded)
}

fn draw_file_bot<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    if !resources.portraits.file_bot_painted {
        resources.portraits.file_bot_painted = true;
        let mut clear_buf = copy_regs(registers);
        let clear_regs = &mut clear_buf[..registers.len()];
        clear_regs[REG_PRIMITIVE_CONFIG] &= !0x300;
        clear_regs[REG_FACE_CULLING] = 0;
        clear_regs[0x0065] = 0;
        clear_regs[REG_TEXTURE_CONFIG] &= !0x7;
        clear_regs[REG_ALPHA_TEST] = 0;
        clear_regs[REG_STENCIL_TEST] = 0;
        clear_regs[REG_COLOR_OPERATION] = 0x00E4_0100;
        clear_regs[REG_BLEND_FUNC] = 0x0101_0000;
        clear_regs[REG_COLOR_BUFFER_WRITE] = 0xF;
        clear_regs[REG_DEPTH_COLOR_MASK] = 0xF << 8;
        for (stage, &stage_base) in TEV_STAGES.iter().enumerate() {
            let vals = match stage {
                0 => [0x000E_000E, 0, 0, 0xFFFF_FF00, 0],
                _ => [0x000F_000F, 0, 0, 0, 0],
            };
            clear_regs[stage_base..stage_base + 5].copy_from_slice(&vals);
        }
        let cover: [Vertex; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]].map(|[x, y]| Vertex {
            clip: [x, y, 0.0, 1.0],
            color: [0.0, 1.0, 1.0, 1.0],
            quaternion: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        });
        resources.portraits.adding = true;
        let _ = rasterize(clear_regs, memory, resources, Vertices::Shaded(&cover));
        resources.portraits.adding = false;
    }

    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);

    // 1. Omit KIU's bottom-screen pastel background draws (`v=2280` starburst + `128x256 L4`
    //    vignette) now that the `#00FFFF` chroma canvas sits cleanly underneath the 3D Tomes!
    if (!textured && shaded.len() > 1000)
        || (textured && dims == (128 << 16 | 256) && fmt == TextureFormat::L4)
    {
        return Some(0);
    }
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };
    let set_wide_xy = |v: &mut Vertex, wx: f32, wy: f32| {
        let nx = wx * 0.75;
        let w = v.clip[3];
        v.clip[0] = ((240.0 - wy) / 120.0 - 1.0) * w;
        v.clip[1] = ((320.0 - nx) / 160.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_bot_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // 2. Pass full-screen untextured fade quads (`[0..320, 0..240]`) through unmodified so
    //    `present_hub.frag` normalizes black/white transitions seamlessly.
    if !textured && shaded.len() == 6 && (right - left) >= 300.0 && (lower - upper) >= 220.0 {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 3. Omit the obsolete 3DS `[•HOME go to HOME Menu]` sticker (`32x64 L4` pill + `A4` text).
    if textured
        && ((dims == (32 << 16 | 64) && fmt == TextureFormat::L4)
            || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4 && upper >= 188.0))
    {
        return Some(0);
    }

    // 4. Omit draws parked off-screen (`upper > 350` or `right < -50`).
    if upper > 350.0 || right < -50.0 {
        return Some(0);
    }

    // 5. Bottom-left `Back` button (`64x32 La4`, `[3.2, 197.9 .. 50.7, 239.5]`).
    if textured && dims == (64 << 16 | 32) && fmt == TextureFormat::La4 {
        for v in &mut shaded {
            let [sx, sy] = to_bot_xy(v);
            set_wide_xy(v, 26.0 + (sx - 27.0) * 0.85, 219.5 + (sy - 218.7) * 0.85);
        }
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 6. Bottom-right `How to Play` badge (`128x64 L4` banner/text + `32x32 Rgba4` book icon).
    if textured
        && ((dims == (128 << 16 | 64) && fmt == TextureFormat::L4)
            || (dims == (32 << 16 | 32) && fmt == TextureFormat::Rgba4 && upper >= 180.0))
    {
        for v in &mut shaded {
            let [sx, sy] = to_bot_xy(v);
            set_wide_xy(v, 370.0 + (sx - 258.0) * 0.78, 221.0 + (sy - 220.0) * 0.78);
        }
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 7. Sub-prompt modal (`128x64 Etc1A4`, e.g. `YES / NO` confirmation): fit cleanly in
    //    the lower half (`wy ≈ 123..235`) when active.
    if textured && dims == (128 << 16 | 64) && fmt == TextureFormat::Etc1A4 {
        for v in &mut shaded {
            let [sx, sy] = to_bot_xy(v);
            set_wide_xy(v, 213.33333 + (sx - 160.0) * 0.50, 179.0 + (sy - 120.0) * 0.50);
        }
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    // 8. 3D Leather-Bound Tomes (`I / II / III`), selection border glow, Roman numerals,
    //    player names, & connected speech-bubble pointer tail (`512x256 L4` `v == 9`):
    for v in &mut shaded {
        let [sx, sy] = to_bot_xy(v);
        set_wide_xy(
            v,
            213.33333 + (sx - 160.0) * FILE_BOT_SCALE,
            FILE_BOT_TOMES_Y + (sy - 95.0) * FILE_BOT_SCALE,
        );
    }

        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        res
}

fn needs_weapon_top_adjust(registers: &[u32], sub: u8, vertex_count: usize) -> bool {
    if registers[REG_TEXTURE_CONFIG] & 1 == 0 {
        return sub == 4 && vertex_count == 6;
    }
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    match sub {
        1 => {
            (dims == (256 << 16 | 128) && matches!(fmt, TextureFormat::La4 | TextureFormat::Etc1A4))
                || (dims == (64 << 16 | 64) && fmt == TextureFormat::L4)
                || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        }
        2 => {
            (dims == (512 << 16 | 256) && fmt == TextureFormat::Etc1 && vertex_count == 12)
                || (dims == (128 << 16 | 128) && fmt == TextureFormat::L4)
                || (dims == (256 << 16 | 128) && matches!(fmt, TextureFormat::Etc1A4 | TextureFormat::La4))
                || (dims == (256 << 16 | 256) && fmt == TextureFormat::A4)
        }
        3 => {
            (dims == (256 << 16 | 256) && matches!(fmt, TextureFormat::Etc1 | TextureFormat::A4))
                || (dims == (128 << 16 | 128) && fmt == TextureFormat::La4)
                || (dims == (256 << 16 | 128) && matches!(fmt, TextureFormat::Etc1A4 | TextureFormat::La4))
                || (dims == (64 << 16 | 64) && matches!(fmt, TextureFormat::Etc1 | TextureFormat::La4))
                || (dims == (32 << 16 | 32) && matches!(fmt, TextureFormat::L8 | TextureFormat::La4))
                || (dims == (8 << 16 | 8) && fmt == TextureFormat::La4)
        }
        4 => {
            (dims == (256 << 16 | 128) && fmt == TextureFormat::La4 && vertex_count == 6)
                || (dims == (256 << 16 | 256) && matches!(fmt, TextureFormat::Etc1 | TextureFormat::A4))
                || (dims == (64 << 16 | 64) && fmt == TextureFormat::La4)
                || (vertex_count == 54 && matches!(fmt, TextureFormat::Etc1 | TextureFormat::Etc1A4))
                || (dims == (32 << 16 | 32) && matches!(fmt, TextureFormat::La4 | TextureFormat::Etc1))
                || (dims == (8 << 16 | 8) && fmt == TextureFormat::La4)
                || (dims == (16 << 16 | 16) && fmt == TextureFormat::Etc1A4)
                || (dims == (256 << 16 | 32) && fmt == TextureFormat::Etc1A4)
        }
        _ => false,
    }
}

fn draw_weapon_top<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let mut sub = resources.portraits.weapon_sub;

    let to_top_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
        [400.0 - by, 240.0 - bx]
    };
    let set_wide_top = |v: &mut Vertex, wx: f32, wy: f32| {
        let nx = wx * 0.9375;
        let w = v.clip[3];
        v.clip[0] = ((240.0 - wy) / 120.0 - 1.0) * w;
        v.clip[1] = ((400.0 - nx) / 200.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_top_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // Omit the 3DS battery icon (`[382..400, 0..13]`) across all three screens.
    if dims == (256 << 16 | 128)
        && fmt == TextureFormat::La4
        && shaded.len() == 6
        && left >= 378.0
        && right <= 402.0
        && upper >= -2.0
        && lower <= 16.0
    {
        resources.portraits.frontend_menu_seen = true;
        return Some(0);
    }

    let mut disable_cull = false;

    if sub == 1 {
        // Weapon List (Option B: Armory Left | Right-Aligned Inspector):
        const PLAQUE_S: f32 = 0.76;
        const MOD_S: f32 = 0.75;
        const MOD_Y0: f32 = 79.0;

        if dims == (256 << 16 | 128) && fmt == TextureFormat::La4 {
            if shaded.len() == 12 {
                // `Try It! / Equip It! / Back` modal header banner (`[-44.7..378.6, 12.3..58.8]`):
                if lower < -50.0 {
                    resources.portraits.weapon_modal_top = false;
                } else {
                    resources.portraits.weapon_modal_top = true;
                    disable_cull = true;
                    for v in &mut shaded {
                        let [sx, sy] = to_top_xy(v);
                        set_wide_top(
                            v,
                            426.66666 - sx * 0.44,
                            28.0 + (sy - 12.3) * 0.72,
                        );
                    }
                }
            } else if shaded.len() == 18 && right < 180.0 {
                // `Ranged / Melee / Value` cream Stars Plaque: mirror horizontally and right-align
                // flush to `wx = 426.67` so its curved black border faces left!
                resources.portraits.weapon_plaque_dx = (134.5 - right).max(0.0) * PLAQUE_S;
                disable_cull = true;
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        426.66666 - sx * PLAQUE_S,
                        24.0 + (sy - 24.0) * PLAQUE_S,
                    );
                }
            } else if shaded.len() == 18 {
                // 6 staggered Modifier pills + curved black-thorn spine, right-aligned at `wx = 426.67`:
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        426.66666 + (sx - 400.0) * MOD_S,
                        MOD_Y0 + (sy - 24.6) * MOD_S,
                    );
                }
            }
        } else if dims == (64 << 16 | 64) && fmt == TextureFormat::L4 && shaded.len() == 6 {
            // Circular Weapon-Type icon: placed to the LEFT of the mirrored Stars Plaque!
            if upper >= 100.0 {
                const MED_S: f32 = 0.66;
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        266.0 + (sx - 301.55) * MED_S,
                        51.5 + (sy - 204.6) * MED_S,
                    );
                }
            } else {
                // Modal banner weapon-type icon (`[3.4..45.0, 14.3..55.9]`)
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        278.0 + (sx - 24.2) * 0.72,
                        44.8 + (sy - 35.1) * 0.72,
                    );
                }
            }
        } else if dims == (256 << 16 | 128) && fmt == TextureFormat::Etc1A4 && shaded.len() == 6 {
            if !resources.portraits.weapon_modal_top {
                const WEP_S: f32 = 0.68;
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        340.0 + (sx - 108.0) * WEP_S,
                        202.0 + (sy - 166.7) * WEP_S,
                    );
                }
            } else {
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        340.0 + (sx - 200.0) * 0.78,
                        145.0 + (sy - 145.1) * 0.78,
                    );
                }
            }
        } else if dims == (256 << 16 | 256)
            && fmt == TextureFormat::A4
            && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
        {
            let modal = resources.portraits.weapon_modal_top;
            for tri in shaded.chunks_exact_mut(3) {
                let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                let cx = (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0;
                let cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0;
                if cy < 23.0 {
                    // Header weapon name (`"Beam Claws"`), centered over Right Inspector:
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(v, 340.0 + (sx - 156.0) * 0.92, sy);
                    }
                } else if modal {
                    if (50.0..=350.0).contains(&cx) && cy < 52.0 {
                        for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                            set_wide_top(v, 352.0 + (sx - 200.0) * 0.78, 44.8 + (sy - 35.0) * 0.78);
                        }
                    } else if cx < 50.0 {
                        for (v, [_, sy]) in tri.iter_mut().zip(pts) {
                            set_wide_top(v, 500.0, sy);
                        }
                    } else {
                        for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                            set_wide_top(
                                v,
                                426.66666 + (sx - 400.0) * MOD_S,
                                MOD_Y0 + (sy - 24.6) * MOD_S,
                            );
                        }
                    }
                } else if cx < 168.0 {
                    // `Ranged / Melee / Value` + stars inside the mirrored right-aligned Stars Plaque:
                    let plaque_dx = resources.portraits.weapon_plaque_dx;
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(
                            v,
                            326.0 + plaque_dx + (sx - 6.2) * PLAQUE_S,
                            24.0 + (sy - 24.0) * PLAQUE_S,
                        );
                    }
                } else {
                    // 6 Modifier pill labels:
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(
                            v,
                            426.66666 + (sx - 400.0) * MOD_S,
                            MOD_Y0 + (sy - 24.6) * MOD_S,
                        );
                    }
                }
            }
        }
    } else if sub == 2 {
        // Weapon Category Select (`"Select a weapon category."`):
        if dims == (128 << 16 | 128) && fmt == TextureFormat::L4 {
            // Hovering `Back` renders a giant L4 arrow watermark across the right showcase: omit it.
            if left > 20.0 {
                return Some(0);
            }
            // Category emblem card (`[-37.9..138.9, 92.7..213.1]`): mirror & right-align
            // at `wx = 426.67` so its flat edge docks right and its gradient fades left!
            disable_cull = true;
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    426.66666 - sx * 0.80,
                    118.0 + (sy - 92.7) * 0.74,
                );
            }
        } else if dims == (512 << 16 | 256) && fmt == TextureFormat::Etc1 && shaded.len() == 12 {
            // Category footer bar (`[-34.4..361.2, 210.0..245.6]`): mirror & right-align across
            // `wx = 58..426.67` so full category descriptions fit with dark contrast!
            disable_cull = true;
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(v, 426.66666 - sx * 1.02, sy);
            }
        } else if dims == (256 << 16 | 128) && fmt == TextureFormat::Etc1A4 && shaded.len() == 6 {
            // Equipped 3D Weapon showcase at upper-right:
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    340.0 + (sx - 252.0) * 0.76,
                    82.0 + (sy - 116.8) * 0.76,
                );
            }
        } else if dims == (256 << 16 | 256)
            && fmt == TextureFormat::A4
            && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
        {
            for tri in shaded.chunks_exact_mut(3) {
                let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                let cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0;
                if cy < 24.0 {
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(v, 213.33333 + (sx - 196.0) * 0.94, sy);
                    }
                } else if cy >= 200.0 {
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(v, 246.0 + (sx - 172.5) * 0.90, sy);
                    }
                } else {
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(
                            v,
                            378.0 + (sx - 59.6) * 0.74,
                            118.0 + (sy - 92.7) * 0.74,
                        );
                    }
                }
            }
        }
    } else if sub == 3 {
        // Gear Hub (`"Arm yourself for battle."` — `WEAPONS / POWERS`):
        // If Power Configurator's `64x64 La4` (`v == 54`) board frame appears, switch directly to `sub = 4`.
        if dims == (64 << 16 | 64) && fmt == TextureFormat::La4 && shaded.len() >= 30 {
            resources.portraits.weapon_sub = 4;
            resources.portraits.weapon_sub_seen = 4;
            sub = 4;
        }
    }

    if sub == 3 {
        if dims == (256 << 16 | 256) && fmt == TextureFormat::Etc1 && shaded.len() == 6 {
            // Pit's rainbow winged silhouette on the Right Showcase:
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    338.0 + (sx - 129.0) * 0.74,
                    136.0 + (sy - 131.0) * 0.74,
                );
            }
        } else if dims == (128 << 16 | 128) && fmt == TextureFormat::La4 {
            // Omit Back-button hover arrow watermark so it never overwrites `"WEAPONS"` / `"POWERS"`.
            if left < 200.0 || upper > 110.0 {
                return Some(0);
            }
            // `"WEAPONS"` / `"POWERS"` serif display title at upper-right:
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    342.0 + (sx - 318.0) * 0.84,
                    58.0 + (sy - 86.0) * 0.84,
                );
            }
        } else if dims == (256 << 16 | 128) && fmt == TextureFormat::Etc1A4 && shaded.len() == 6 {
            // Equipped 3D Weapon showcase at lower-right:
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    340.0 + (sx - 149.0) * 0.62,
                    158.0 + (sy - 157.0) * 0.62,
                );
            }
        } else if (dims == (64 << 16 | 64) && fmt == TextureFormat::Etc1)
            || (dims == (32 << 16 | 32) && matches!(fmt, TextureFormat::L8 | TextureFormat::La4))
            || (dims == (8 << 16 | 8) && fmt == TextureFormat::La4)
        {
            // Equipped Power Board preview (`POWERS` highlighted) at lower-right:
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    340.0 + (sx - 180.0) * 0.74,
                    152.0 + (sy - 139.0) * 0.74,
                );
            }
        } else if dims == (256 << 16 | 256)
            && fmt == TextureFormat::A4
            && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
        {
            for tri in shaded.chunks_exact_mut(3) {
                let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                let cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0;
                if cy < 24.0 {
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(v, 213.33333 + (sx - 145.0) * 0.94, sy);
                    }
                }
            }
        }
    } else if sub == 4 {
        // Power Configurator (Option A: 0.80x Workbench Left | Power Inspector / Set Summary Right):
        const S_BOX: f32 = 0.58;
        const S_ROT: f32 = 0.78;
        const S_DESC_X: f32 = 0.44;
        const S_DESC_Y: f32 = 0.58;

        if textured && dims == (256 << 16 | 256) && fmt == TextureFormat::Etc1 && shaded.len() >= 100 {
            // Start of Power Configurator's top pass: lined cream parchment + gold seal + header ribbon.
            resources.portraits.power_set_overlay = false;
            resources.portraits.power_set_two_cols = false;
        } else if !textured && shaded.len() == 6 {
            // Translucent white description card (`[-18.8..418.8, 163.8..208.5]`):
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    256.0 + (sx + 18.8) * (168.0 / 437.6),
                    144.0 + (sy - 163.8) * (40.0 / 44.7),
                );
            }
        } else if textured && dims == (256 << 16 | 32) && fmt == TextureFormat::Etc1A4 {
            if shaded.len() >= 30 {
                // `~Power Set X~` sky-blue summary backing sheet (`[-49.5..449.5, 9.9..233.8]`):
                if right <= 0.0 {
                    resources.portraits.power_set_overlay = false;
                    return Some(0);
                }
                resources.portraits.power_set_overlay = true;
                let pill_shift = ((-49.5 - left) * 0.45276).max(0.0);
                resources.portraits.power_set_shift = pill_shift;
                let x_slide = pill_shift * 0.78;
                let span_x = (right - left).max(1.0);
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        255.0 + x_slide + (sx - left) * (170.0 / span_x),
                        36.0 + (sy - 9.9) * (200.0 / 223.9),
                    );
                }
            } else {
                // Equipped power pill cards inside `~Power Set X~`:
                if !resources.portraits.power_set_overlay || left > 500.0 {
                    return Some(0);
                }
                let shift = resources.portraits.power_set_shift;
                if left + shift >= 195.0 {
                    resources.portraits.power_set_two_cols = true;
                }
                let s_set = if resources.portraits.power_set_two_cols { 0.40 } else { 0.78 };
                let x_slide = shift * 0.78;
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        257.0 + (sx + shift) * s_set + x_slide,
                        44.0 + (sy - 23.0) * s_set,
                    );
                }
            }
        } else if resources.portraits.power_set_overlay {
            // Mini icons (`32x32 Etc1`, `16x16 Etc1A4`), mini shapes (`32x32 La4`, `8x8 La4`),
            // and `256x256 A4` labels belonging to the `~Power Set X~` summary overlay:
            if left > 500.0 {
                return Some(0);
            }
            let shift = resources.portraits.power_set_shift;
            let s_set = if resources.portraits.power_set_two_cols { 0.40 } else { 0.78 };
            let x_slide = shift * 0.78;
            if textured
                && dims == (256 << 16 | 256)
                && fmt == TextureFormat::A4
                && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
            {
                for tri in shaded.chunks_exact_mut(3) {
                    let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                    let cx = (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0;
                    if cx + shift < -10.0 || cx > 500.0 {
                        for v in tri.iter_mut() {
                            v.clip = [0.0; 4];
                        }
                    } else {
                        for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                            set_wide_top(
                                v,
                                257.0 + (sx + shift) * s_set + x_slide,
                                44.0 + (sy - 23.0) * s_set,
                            );
                        }
                    }
                }
            } else {
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        257.0 + (sx + shift) * s_set + x_slide,
                        44.0 + (sy - 23.0) * s_set,
                    );
                }
            }
        } else if textured && dims == (64 << 16 | 64) && fmt == TextureFormat::La4 {
            // `6x6` Shape Preview outer frame (`cx < 175`); omit the oversized outer pink box (`cx >= 175`)
            // around the Power Icon since the Power Icon already carries its own bold rounded border:
            resources.portraits.power_set_two_cols = false;
            if left > 500.0 || right < 10.0 {
                return Some(0);
            }
            for tri in shaded.chunks_exact_mut(3) {
                let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                let cx = (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0;
                if cx < 175.0 {
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(
                            v,
                            302.0 + (sx - 103.1) * S_BOX,
                            76.0 + (sy - 100.0) * S_BOX,
                        );
                    }
                } else {
                    for v in tri.iter_mut() {
                        v.clip = [0.0; 4];
                    }
                }
            }
        } else if textured
            && shaded.len() == 54
            && matches!(fmt, TextureFormat::Etc1 | TextureFormat::Etc1A4)
        {
            // Selected Power Icon (`[219.2..330.5, 44.1..155.4]`):
            if left > 500.0 || right < 10.0 {
                return Some(0);
            }
            for v in &mut shaded {
                let [sx, sy] = to_top_xy(v);
                set_wide_top(
                    v,
                    382.0 + (sx - 274.85) * S_BOX,
                    76.0 + (sy - 99.75) * S_BOX,
                );
            }
        } else if textured && dims == (32 << 16 | 32) && fmt == TextureFormat::La4 {
            if left > 500.0 || right < 10.0 {
                return Some(0);
            }
            if upper >= 205.0 {
                // `[L]` shoulder badge of `L/R: Rotate` (`[67.5..84.7, 219.5..236.7]`, `cy ≈ 228`):
                if upper > 231.0 {
                    return Some(0);
                }
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        343.5 + (sx - 137.0) * S_ROT,
                        126.0 + (sy - 227.5) * S_ROT,
                    );
                }
            } else {
                // `6x6` tetromino footprint preview (`[44.8..161.4, 41.7..158.3]`):
                for v in &mut shaded {
                    let [sx, sy] = to_top_xy(v);
                    set_wide_top(
                        v,
                        302.0 + (sx - 103.1) * S_BOX,
                        76.0 + (sy - 100.0) * S_BOX,
                    );
                }
            }
        } else if textured
            && dims == (256 << 16 | 256)
            && fmt == TextureFormat::A4
            && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
        {
            for tri in shaded.chunks_exact_mut(3) {
                let pts = [to_top_xy(&tri[0]), to_top_xy(&tri[1]), to_top_xy(&tri[2])];
                let cx = (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0;
                let cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0;
                if cx < -12.0 || cx > 412.0 || cy > 240.0 {
                    for v in tri.iter_mut() {
                        v.clip = [0.0; 4];
                    }
                } else if cy < 28.0 {
                    // Header (`Power Name   Lv. X   Uses: Y`) at 1:1 square aspect:
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(v, sx, sy);
                    }
                } else if cy >= 212.0 {
                    // `L/R: Rotate` footer text (`217.9..237.1`, rest `cy ≈ 227.5`):
                    if cy > 234.0 {
                        for v in tri.iter_mut() {
                            v.clip = [0.0; 4];
                        }
                    } else {
                        for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                            set_wide_top(
                                v,
                                343.5 + (sx - 137.0) * S_ROT,
                                126.0 + (sy - 227.5) * S_ROT,
                            );
                        }
                    }
                } else {
                    // Power Description text (`173.9..200.9`), centered inside Right Inspector card:
                    for (v, [sx, sy]) in tri.iter_mut().zip(pts) {
                        set_wide_top(
                            v,
                            340.0 + (sx - 200.0) * S_DESC_X,
                            164.0 + (sy - 187.4) * S_DESC_Y,
                        );
                    }
                }
            }
        }
    }

    resources.portraits.adding = true;
    let res = if disable_cull {
        let mut regs_buf = copy_regs(registers);
        let regs = &mut regs_buf[..registers.len()];
        regs[REG_FACE_CULLING] = 0;
        rasterize(regs, memory, resources, Vertices::Shaded(&shaded))
    } else {
        rasterize(registers, memory, resources, Vertices::Shaded(&shaded))
    };
    resources.portraits.adding = false;
    res
}

/// Paints opaque `#FF00FF` exterior/scroll-clip mask geometry on bottom chroma menus.
fn rasterize_magenta_mask<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    shaded: &mut [Vertex],
) -> Option<u32> {
    let mut mask_buf = copy_regs(registers);
    let mask_regs = &mut mask_buf[..registers.len()];
    mask_regs[REG_PRIMITIVE_CONFIG] &= !0x300;
    mask_regs[REG_FACE_CULLING] = 0;
    mask_regs[0x0065] = 0;
    mask_regs[REG_TEXTURE_CONFIG] &= !0x7;
    mask_regs[REG_ALPHA_TEST] = 0;
    mask_regs[REG_STENCIL_TEST] = 0;
    mask_regs[REG_COLOR_OPERATION] = 0x00E4_0100;
    mask_regs[REG_BLEND_FUNC] = 0x0101_0000;
    mask_regs[REG_COLOR_BUFFER_WRITE] = 0xF;
    mask_regs[REG_DEPTH_COLOR_MASK] = 0xF << 8;
    for (stage, &stage_base) in TEV_STAGES.iter().enumerate() {
        let vals = match stage {
            0 => [0x000E_000E, 0, 0, 0xFFFF_00FF, 0],
            _ => [0x000F_000F, 0, 0, 0, 0],
        };
        mask_regs[stage_base..stage_base + 5].copy_from_slice(&vals);
    }
    for v in shaded.iter_mut() {
        v.color = [1.0, 0.0, 1.0, 1.0];
    }
    resources.portraits.adding = true;
    let res = rasterize(mask_regs, memory, resources, Vertices::Shaded(shaded));
    resources.portraits.adding = false;
    res
}

fn draw_weapon_bot<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    mut shaded: Vec<Vertex>,
) -> Option<u32> {
    if !resources.portraits.weapon_bot_painted {
        resources.portraits.weapon_bot_painted = true;
        resources.portraits.weapon_bot_modal = false;
        let mut cover: [Vertex; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]].map(|[x, y]| Vertex {
            clip: [x, y, 0.0, 1.0],
            color: [1.0, 0.0, 1.0, 1.0],
            quaternion: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        });
        let _ = rasterize_magenta_mask(registers, memory, resources, &mut cover);
    }

    if shaded.iter().any(|v| !v.clip[3].is_finite() || v.clip[3] <= 1e-4) {
        resources.portraits.adding = true;
        let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
        resources.portraits.adding = false;
        return res;
    }

    let textured = registers[REG_TEXTURE_CONFIG] & 1 != 0;
    let base = TEXTURE_UNIT_BASES[0];
    let dims = registers[base + 1];
    let fmt = TextureFormat::from_raw(registers[base + 13]);
    let sub = resources.portraits.weapon_sub;

    let to_bot_xy = |v: &Vertex| -> [f32; 2] {
        let w = v.clip[3];
        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
        let by = (v.clip[1] / w * 0.5 + 0.5) * 320.0;
        [320.0 - by, 240.0 - bx]
    };
    let set_bot_xy = |v: &mut Vertex, nx: f32, ny: f32| {
        let w = v.clip[3];
        v.clip[0] = ((240.0 - ny) / 120.0 - 1.0) * w;
        v.clip[1] = ((320.0 - nx) / 160.0 - 1.0) * w;
    };

    let (mut left, mut upper, mut right, mut lower) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for v in &shaded {
        let [sx, sy] = to_bot_xy(v);
        left = left.min(sx);
        upper = upper.min(sy);
        right = right.max(sx);
        lower = lower.max(sy);
    }

    // Omit draws parked far off-screen (e.g. closed modal at `x > 1900`).
    if left > 450.0 || right < -150.0 || upper > 350.0 || lower < -50.0 {
        return Some(0);
    }

    // On Weapon List (`sub == 1`), `64x8 Rgb565` (`v == 12`) is KIU's top & bottom scroll-clip
    // mask over scrolled weapon rows — paint as `#FF00FF`; omit `64x8 Rgb565` wallpaper otherwise.
    if textured && dims == (64 << 16 | 8) && fmt == TextureFormat::Rgb565 {
        if sub == 1 && shaded.len() >= 12 {
            return rasterize_magenta_mask(registers, memory, resources, &mut shaded);
        }
        return Some(0);
    }
    if sub == 4 {
        // Power Configurator (Option A):
        // 1. `64x64 Etc1` (`v == 6`, width > 300): clamp KIU's dark charcoal damask strictly
        //    inside the Power List well (`[146.5..307.5, 25.5..200.0]`) so sparse lists stay
        //    neatly framed while the outer border around Tabs/Back/OK stays `#FF00FF`.
        // 2. `64x64 Etc1` (`v == 12`, width > 300): KIU's top & bottom scroll-clip mask quads
        //    drawn over scrolled-off list rows — paint as `#FF00FF` so scrolled-off rows are
        //    masked cleanly before `Autofill` / `Category` / `Back` / `OK (START)` draw on top!
        if textured && dims == (64 << 16 | 64) && fmt == TextureFormat::Etc1 && (right - left) > 300.0 {
            if shaded.len() == 6 {
                for v in &mut shaded {
                    let [sx, sy] = to_bot_xy(v);
                    set_bot_xy(v, sx.clamp(146.5, 307.5), sy.clamp(25.5, 200.0));
                }
            } else {
                return rasterize_magenta_mask(registers, memory, resources, &mut shaded);
            }
        } else if !textured && (right - left) > 340.0 {
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, sx.clamp(0.0, 312.0), sy.clamp(0.0, 202.0));
            }
        } else if upper >= 200.5 && (right - left) < 300.0 {
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, sx, sy - 3.5);
            }
        }
    } else if sub == 2 {
        if (!textured && shaded.len() >= 18 && (right - left) > 400.0)
            || (textured && dims == (128 << 16 | 128) && fmt == TextureFormat::L4)
            || (textured && dims == (8 << 16 | 16) && fmt == TextureFormat::L4)
        {
            return Some(0);
        }
        // Mirror the 3 dark-green thorny frame quads (`128x64 L4`, `64x256 L4`, `256x32 L4`)
        // horizontally so the flat edge docks flush-left (`x = 0`) and the thorny vine edge
        // wraps organically around the right of the 2x5 Category buttons!
        if textured
            && fmt == TextureFormat::L4
            && (dims == (128 << 16 | 64) || dims == (64 << 16 | 256) || dims == (256 << 16 | 32))
        {
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, 320.0 - sx, sy);
            }
            let mut regs_buf = copy_regs(registers);
            let regs = &mut regs_buf[..registers.len()];
            regs[REG_FACE_CULLING] = 0;
            resources.portraits.adding = true;
            let res = rasterize(regs, memory, resources, Vertices::Shaded(&shaded));
            resources.portraits.adding = false;
            return res;
        }
        // Pass full-screen fade quads through unmodified; keep `Back` (`64x32 La4`) at `(sx + 4, sy)`
        // in bottom-left; shift the 10 Category plaques, icons, names, & highlight left by `-48`.
        let dx = if !textured && (right - left) >= 315.0 && (lower - upper) >= 235.0 {
            0.0
        } else if textured && dims == (64 << 16 | 32) && fmt == TextureFormat::La4 {
            4.0
        } else {
            -48.0
        };
        for v in &mut shaded {
            let [sx, sy] = to_bot_xy(v);
            set_bot_xy(v, sx + dx, sy);
        }
    } else if sub == 1 {
        if textured && dims == (256 << 16 | 128) && fmt == TextureFormat::L4 {
            // Clamp the dark well (`v == 24`) and header/side/bottom frame (`v == 39`) strictly
            // to `[3.0..315.0, 0.0..194.5]` so `[Back]` and `[< ALL Page >]` float cleanly on `#FF00FF`.
            let y_min = if shaded.len() <= 24 { 29.0 } else { 0.0 };
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, sx.clamp(3.0, 315.0), sy.clamp(y_min, 194.5));
            }
        } else if !textured && (right - left) > 340.0 {
            if shaded.len() == 6 {
                resources.portraits.weapon_bot_modal = true;
            }
            for v in &mut shaded {
                let [sx, sy] = to_bot_xy(v);
                set_bot_xy(v, sx.clamp(3.0, 315.0), sy.clamp(0.0, 194.5));
            }
        } else if textured
            && dims == (256 << 16 | 256)
            && fmt == TextureFormat::A4
            && (registers[REG_PRIMITIVE_CONFIG] >> 8 & 3) == 0
            && !resources.portraits.weapon_bot_modal
            && !(lower <= 24.0 || (upper >= 208.0 && lower <= 227.0))
        {
            // Clip partially scrolled weapon-row names & values (`"Viridi Palm 282"`) strictly inside
            // `[30.0..192.5]` while preserving `"Equipped"`, `"Strength"`, `"1/33"`, and modal text.
            let mut kept = Vec::with_capacity(shaded.len());
            for tri in shaded.chunks_exact(3) {
                let ys = [to_bot_xy(&tri[0])[1], to_bot_xy(&tri[1])[1], to_bot_xy(&tri[2])[1]];
                if ys.into_iter().all(|y| (30.0..=192.5).contains(&y)) {
                    kept.extend_from_slice(tri);
                }
            }
            if kept.is_empty() {
                return Some(0);
            }
            shaded = kept;
        }
    }

    resources.portraits.adding = true;
    let res = rasterize(registers, memory, resources, Vertices::Shaded(&shaded));
    resources.portraits.adding = false;
    res
}

/// tracks when the title screen's top-screen sky or bottom-screen logo pass is
/// being drawn, as well as Fuse Weapons' A/B card headers.
fn note_title_draw(registers: &[u32], portraits: &mut Portraits) {
    if portraits.in_battle {
        if portraits.battle_cleared {
            return;
        }
        portraits.battle_cleared = true;
        portraits.reset_title();
        portraits.fuse_first_header = 0;
        portraits.fuse_frame_seen = false;
        portraits.fuse_weapon_seen = false;
        portraits.fuse_preview_seen = false;
        portraits.fuse_active = false;
        portraits.fuse_reveal = false;
        portraits.fuse_missed = 0;
        portraits.cauldron_mural_step = 0;
        portraits.cauldron_top_seen = false;
        portraits.cauldron_cursor_seen = false;
        portraits.cauldron_pot_seen = false;
        portraits.cauldron_active = false;
        portraits.cauldron_hud_active = false;
        portraits.cauldron_tier_init = false;
        portraits.cauldron_fade_buf = 0;
        portraits.cauldron_missed = 0;
        portraits.chapter_step = 0;
        portraits.chapter_top_seen = false;
        portraits.chapter_active = false;
        portraits.chapter_missed = 0;
        portraits.chapter_frame_title_right = 0.0;
        portraits.chapter_frame_num_bounds = (0.0, 0.0);
        portraits.mode_top_step = 0;
        portraits.mode_top_seen = false;
        portraits.mode_top_latched = false;
        portraits.mode_bot_seen = false;
        portraits.mode_rejected = false;
        portraits.mode_active = false;
        portraits.mode_missed = 0;
        portraits.solo_top_step = 0;
        portraits.solo_top_seen = false;
        portraits.solo_active = false;
        portraits.solo_missed = 0;
        portraits.file_top_step = 0;
        portraits.file_top_seen = false;
        portraits.file_active = false;
        portraits.file_missed = 0;
        portraits.weapon_step = 0;
        portraits.gear_step = 0;
        portraits.power_step = 0;
        portraits.power_bot_etc1 = false;
        portraits.power_bot_latched = false;
        portraits.power_set_overlay = false;
        portraits.power_set_shift = 0.0;
        portraits.power_set_two_cols = false;
        portraits.weapon_sub_seen = 0;
        portraits.weapon_sub = 0;
        portraits.weapon_active = false;
        portraits.weapon_bot_modal = false;
        portraits.weapon_missed = 0;
        portraits.weapon_modal_top = false;
        return;
    }
    portraits.battle_cleared = false;
    if read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
        && read_float24(registers, REG_VIEWPORT_HEIGHT) == 200.0
    {
        portraits.title_bottom_drawing = false;
        portraits.title_bottom_counted = false;
        if !portraits.fuse_active && registers[REG_TEXTURE_CONFIG] & 1 != 0 {
            let base = TEXTURE_UNIT_BASES[0];
            let dims = registers[base + 1];
            let fmt = TextureFormat::from_raw(registers[base + 13]);
            if dims == (64 << 16 | 64) && fmt == TextureFormat::Etc1 {
                portraits.file_top_step = 1;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
            } else if portraits.file_top_step == 1
                && dims == (8 << 16 | 16)
                && fmt == TextureFormat::L4
            {
                portraits.file_top_step = 2;
            } else if portraits.file_top_step == 2
                && dims == (64 << 16 | 8)
                && fmt == TextureFormat::Rgb565
            {
                portraits.file_top_step = 3;
            } else if portraits.file_top_step == 3
                && dims == (64 << 16 | 8)
                && fmt == TextureFormat::Rgb565
            {
                // Multiple `64x8 Rgb565` rainbow/spiral draws precede `512x256 L4`.
            } else if portraits.file_top_step == 3
                && dims == (512 << 16 | 256)
                && fmt == TextureFormat::L4
            {
                portraits.file_top_step = 4;
                portraits.file_top_seen = true;
                portraits.file_active = true;
                portraits.file_missed = 0;
                portraits.reset_title();
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.solo_active = false;
                portraits.weapon_active = false;
                portraits.weapon_modal_top = false;
                portraits.power_set_overlay = false;
                portraits.power_set_two_cols = false;
                portraits.power_bot_latched = false;
                portraits.frontend_menu_seen = true;
            } else {
                portraits.file_top_step = 0;
            }
        } else if registers[REG_TEXTURE_CONFIG] & 1 != 0 {
            portraits.file_top_step = 0;
        }
        if !portraits.title_sky_active && !portraits.fuse_active && registers[REG_TEXTURE_CONFIG] & 1 != 0 {
            let base = TEXTURE_UNIT_BASES[0];
            let dims = registers[base + 1];
            let fmt = TextureFormat::from_raw(registers[base + 13]);
            if dims == (128 << 16 | 64) && fmt == TextureFormat::Etc1 {
                portraits.cauldron_mural_step = 1;
            } else if portraits.cauldron_mural_step == 1
                && dims == (64 << 16 | 64)
                && fmt == TextureFormat::La4
            {
                portraits.cauldron_mural_step = 2;
            } else if portraits.cauldron_mural_step == 2
                && dims == (256 << 16 | 256)
                && fmt == TextureFormat::L4
            {
                portraits.cauldron_mural_step = 3;
                portraits.cauldron_top_seen = true;
                portraits.cauldron_active = true;
                portraits.cauldron_hud_active = true;
                portraits.cauldron_missed = 0;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.solo_active = false;
                portraits.file_active = false;
                portraits.weapon_active = false;
                portraits.weapon_modal_top = false;
                portraits.power_set_overlay = false;
                portraits.power_set_two_cols = false;
            } else {
                portraits.cauldron_mural_step = 0;
            }

            if dims == (512 << 16 | 256) && fmt == TextureFormat::Etc1 {
                portraits.chapter_step = 1;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
            } else if portraits.chapter_step == 1
                && dims == (128 << 16 | 128)
                && fmt == TextureFormat::Etc1A4
            {
                // Do NOT activate `chapter_active` yet: Arms Altar's Redeem Hearts
                // preview also draws `512x256 Etc1` followed by a `128x128 Etc1A4`
                // weapon texture. Wait for Chapter Select's `256x64 Etc1` stage banner.
                portraits.chapter_step = 2;
            } else if portraits.chapter_step == 2
                && dims == (32 << 16 | 32)
                && fmt == TextureFormat::La4
            {
                // Optional cleared-chapter laurel wreath before the 256x64 stage preview.
            } else if portraits.chapter_step == 2
                && dims == (256 << 16 | 64)
                && fmt == TextureFormat::Etc1
            {
                portraits.chapter_step = 3;
                portraits.chapter_top_seen = true;
                portraits.chapter_active = true;
                portraits.chapter_missed = 0;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.solo_active = false;
                portraits.weapon_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.file_active = false;
            } else {
                portraits.chapter_step = 0;
            }

            // Mode Select's top screen draws:
            //   1. `256x32 Etc1` lined rainbow parchment (shared across all 8 mode tiles)
            //   2. 0..=2 per-mode background silhouette draws (`256x128 Etc1` on Solo,
            //      `128x128 Etc1` on Vault, `128x64 Etc1` on Options, `32x32 Rgb565` on
            //      StreetPass, `128x128 L4` / `256x128 La4` on Together / AR / Manual)
            //   3. One or more `256x256 L4` wing/vine flourishes
            //   4. `32x8 Rgb565` header underline
            if dims == (256 << 16 | 32) && fmt == TextureFormat::Etc1 {
                portraits.mode_top_step = 1;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
            } else if matches!(portraits.mode_top_step, 1 | 4 | 5)
                && dims == (256 << 16 | 256)
                && fmt == TextureFormat::L4
            {
                portraits.mode_top_step = 2;
            } else if portraits.mode_top_step == 1 {
                portraits.mode_top_step = 4;
            } else if portraits.mode_top_step == 4 {
                portraits.mode_top_step = 5;
            } else if portraits.mode_top_step == 2
                && dims == (256 << 16 | 256)
                && fmt == TextureFormat::L4
            {
                // Multiple `256x256 L4` wing/vine flourish draws precede the header underline.
            } else if portraits.mode_top_step == 2
                && dims == (32 << 16 | 8)
                && fmt == TextureFormat::Rgb565
            {
                portraits.mode_top_step = 3;
                portraits.mode_top_seen = true;
                portraits.mode_top_latched = true;
                if !portraits.mode_rejected {
                    portraits.mode_active = true;
                    portraits.mode_missed = 0;
                    portraits.cauldron_active = false;
                    portraits.cauldron_hud_active = false;
                    portraits.chapter_active = false;
                    portraits.solo_active = false;
                    portraits.file_active = false;
                    portraits.weapon_active = false;
                    portraits.weapon_modal_top = false;
                }
                portraits.frontend_menu_seen = true;
            } else {
                portraits.mode_top_step = 0;
            }

            // Solo Hub's top screen (both Palutena's Temple and Viridi's Sanctuary) draws:
            //   - Viridi's Sanctuary: `16x32 Rgb565` -> `64x64 Rgba4` (bottom vignette)
            //     -> `128x128 L4` (lower-right black swirl) -> `64x64 Rgba4` (Total Hearts pouch)
            //   - Palutena's Temple: `64x64 Rgba4` (bottom vignette) -> `64x64 Rgba4` (Total Hearts pouch)
            // Matching the 2D HUD pair (`64x64 Rgba4 -> [128x128 L4] -> 64x64 Rgba4`) handles
            // both domains independently of 3D room meshes and camera frustum culling, while
            // keeping the Viridi `16x32 Rgb565` fast path so its swirl is hidden on frame 1.
            if dims == (16 << 16 | 32) && fmt == TextureFormat::Rgb565 {
                portraits.solo_top_step = 1;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
            } else if portraits.solo_top_step == 1
                && dims == (64 << 16 | 64)
                && fmt == TextureFormat::Rgba4
            {
                portraits.solo_top_step = 2;
            } else if portraits.solo_top_step == 2
                && dims == (128 << 16 | 128)
                && fmt == TextureFormat::L4
            {
                portraits.solo_top_step = 3;
                portraits.solo_top_seen = true;
                portraits.solo_active = true;
                portraits.solo_missed = 0;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.file_active = false;
                portraits.weapon_active = false;
                portraits.weapon_modal_top = false;
                portraits.frontend_menu_seen = true;
            } else if matches!(portraits.solo_top_step, 2 | 4 | 5)
                && dims == (64 << 16 | 64)
                && fmt == TextureFormat::Rgba4
            {
                portraits.solo_top_step = 3;
                portraits.solo_top_seen = true;
                portraits.solo_active = true;
                portraits.solo_missed = 0;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.file_active = false;
                portraits.weapon_active = false;
                portraits.weapon_modal_top = false;
                portraits.frontend_menu_seen = true;
            } else if portraits.solo_top_step == 4
                && dims == (128 << 16 | 128)
                && fmt == TextureFormat::L4
            {
                portraits.solo_top_step = 5;
            } else if dims == (64 << 16 | 64)
                && fmt == TextureFormat::Rgba4
                && !portraits.file_top_seen
                && !portraits.cauldron_top_seen
            {
                portraits.solo_top_step = 4;
            } else {
                portraits.solo_top_step = 0;
            }

            // Weapon List (`512x256 Etc1` -> `256x128 La4`) & Weapon Category (`512x256 Etc1` -> `128x128 L4`):
            if dims == (512 << 16 | 256) && fmt == TextureFormat::Etc1 {
                portraits.weapon_step = 1;
                portraits.weapon_modal_top = false;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
            } else if portraits.weapon_step == 1
                && dims == (256 << 16 | 128)
                && fmt == TextureFormat::La4
            {
                portraits.weapon_step = 2;
                portraits.weapon_sub_seen = 1;
                portraits.weapon_sub = 1;
                portraits.weapon_active = true;
                portraits.weapon_missed = 0;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.solo_active = false;
                portraits.file_active = false;
                portraits.power_set_overlay = false;
                portraits.power_set_two_cols = false;
                portraits.frontend_menu_seen = true;
            } else if portraits.weapon_step == 1
                && dims == (128 << 16 | 128)
                && fmt == TextureFormat::L4
            {
                portraits.weapon_step = 2;
                portraits.weapon_sub_seen = 2;
                portraits.weapon_sub = 2;
                portraits.weapon_active = true;
                portraits.weapon_missed = 0;
                portraits.weapon_modal_top = false;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.solo_active = false;
                portraits.file_active = false;
                portraits.power_set_overlay = false;
                portraits.power_set_two_cols = false;
                portraits.frontend_menu_seen = true;
            } else {
                portraits.weapon_step = 0;
            }

            // Gear Hub (`256x256 L4` -> `256x256 Etc1` Pit rainbow silhouette):
            if dims == (256 << 16 | 256) && fmt == TextureFormat::L4 {
                portraits.gear_step = 1;
            } else if portraits.gear_step == 1
                && dims == (256 << 16 | 256)
                && fmt == TextureFormat::Etc1
            {
                portraits.gear_step = 2;
                portraits.power_bot_latched = true;
                portraits.weapon_sub_seen = 3;
                portraits.weapon_sub = 3;
                portraits.weapon_active = true;
                portraits.weapon_missed = 0;
                portraits.weapon_modal_top = false;
                portraits.cauldron_active = false;
                portraits.cauldron_hud_active = false;
                portraits.chapter_active = false;
                portraits.mode_active = false;
                portraits.mode_top_latched = false;
                portraits.solo_active = false;
                portraits.file_active = false;
                portraits.power_set_overlay = false;
                portraits.power_set_two_cols = false;
                portraits.frontend_menu_seen = true;
            } else {
                portraits.gear_step = 0;
            }

            // Power Configurator (`256x256 Etc1` -> `128x64 L4`, armed via Gear Hub or bottom `6x6` + Set Tabs):
            if dims == (256 << 16 | 256) && fmt == TextureFormat::Etc1 {
                portraits.power_step = 1;
            } else if portraits.power_step == 1
                && dims == (128 << 16 | 64)
                && fmt == TextureFormat::L4
            {
                portraits.power_step = 2;
                if portraits.power_bot_latched {
                    if portraits.weapon_sub != 4 {
                        portraits.power_set_overlay = false;
                        portraits.power_set_two_cols = false;
                    }
                    portraits.weapon_sub_seen = 4;
                    portraits.weapon_sub = 4;
                    portraits.weapon_active = true;
                    portraits.weapon_missed = 0;
                    portraits.weapon_modal_top = false;
                    portraits.cauldron_active = false;
                    portraits.cauldron_hud_active = false;
                    portraits.chapter_active = false;
                    portraits.mode_active = false;
                    portraits.mode_top_latched = false;
                    portraits.solo_active = false;
                    portraits.file_active = false;
                    portraits.frontend_menu_seen = true;
                }
            } else {
                portraits.power_step = 0;
            }
        } else {
            portraits.cauldron_mural_step = 0;
            portraits.chapter_step = 0;
            portraits.mode_top_step = 0;
            portraits.solo_top_step = 0;
            portraits.weapon_step = 0;
            portraits.gear_step = 0;
            portraits.power_step = 0;
        }
        if is_title_sky(registers) {
            portraits.title_sky_seen = true;
            portraits.title_sky_active = true;
            portraits.title_sky_missed = 0;
            portraits.title_sizzle_active = false;
            portraits.title_sizzle_seen = false;
            portraits.title_sizzle_missed = 0;
            portraits.title_seen = true;
            portraits.mode_active = false;
            portraits.mode_top_latched = false;
            portraits.solo_active = false;
            portraits.file_active = false;
            portraits.top = Some([
                registers[REG_COLOR_BUFFER_ADDRESS],
                registers[REG_COLOR_BUFFER_FORMAT],
                registers[REG_FRAMEBUFFER_DIMENSIONS],
            ]);
        } else if !portraits.title_sky_active && is_fuse_card_header(registers) {
            let addr = registers[TEXTURE_UNIT_BASES[0] + 4];
            if addr != 0 {
                if portraits.fuse_first_header == 0 {
                    portraits.fuse_first_header = addr;
                } else if portraits.fuse_first_header != addr {
                    portraits.fuse_frame_seen = true;
                    if !portraits.fuse_reveal {
                        portraits.fuse_active = true;
                        portraits.fuse_missed = 0;
                    }
                }
            }
        } else if !portraits.title_sky_active && portraits.fuse_frame_seen {
            if is_fuse_result_weapon(registers) {
                portraits.fuse_weapon_seen = true;
            } else if is_fuse_small_preview(registers) {
                portraits.fuse_preview_seen = true;
                portraits.fuse_active = true;
                portraits.fuse_reveal = false;
                portraits.fuse_missed = 0;
            }
        }
    } else if portraits.title_sky_active && on_bottom(registers) && is_title_bottom_sky(registers) {
        portraits.title_bottom_drawing = true;
        portraits.title_seen = true;
        portraits.title_frames = portraits.title_frames.saturating_add(1);
        if portraits.title_frames <= 16 || !portraits.title_captured {
            portraits.title.clear();
        }
    } else if portraits.title_sizzle_active && on_bottom(registers) && is_title_bottom_sky(registers) {
        portraits.title_sizzle_seen = true;
        portraits.title_seen = true;
    } else if on_bottom(registers) {
        let base = TEXTURE_UNIT_BASES[0];
        let dims = registers[base + 1];
        let fmt = TextureFormat::from_raw(registers[base + 13]);
        if portraits.cauldron_active && dims == (128 << 16 | 128) && fmt == TextureFormat::Etc1 {
            portraits.cauldron_pot_seen = true;
        }
        if !portraits.in_battle
            && (portraits.mode_top_seen || portraits.mode_top_latched || portraits.mode_active)
            && dims == (256 << 16 | 256)
            && fmt == TextureFormat::Rgba4
        {
            portraits.mode_bot_seen = true;
            portraits.mode_rejected = false;
            portraits.mode_active = true;
            portraits.mode_missed = 0;
            portraits.solo_active = false;
            portraits.power_bot_etc1 = false;
            portraits.power_bot_latched = false;
        }
        if !portraits.in_battle && !portraits.mode_active {
            if dims == (64 << 16 | 64) && fmt == TextureFormat::Etc1 {
                portraits.power_bot_etc1 = true;
            } else if portraits.power_bot_etc1
                && dims == (256 << 16 | 64)
                && fmt == TextureFormat::Rgba4
            {
                portraits.power_bot_latched = true;
            }
        }
    } else if !is_bottom(registers) {
        portraits.title_bottom_drawing = false;
        portraits.title_bottom_counted = false;
    }
}

/// a dialogue portrait as Kid Icarus Uprising draws it, an Etc1 256x256
/// picture on the whole bottom screen using its two-stage TEV pedestal-cut
/// combiner (`0x0CA == 0x0002_0001 && 0x0D2 == 0x0009_0001`).
fn is_portrait(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    on_bottom(registers)
        && registers[base + 1] == (256 << 16 | 256)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Etc1
        && (registers[REG_COLOR_OPERATION] & 0x100 != 0 || registers[REG_ALPHA_TEST] & 1 != 0)
        && registers[0x0CA] == 0x0002_0001
        && registers[0x0D2] == 0x0009_0001
}

/// a textured draw over the whole bottom screen's 240x320, the way the
/// title draws its pictures there.
fn on_bottom(registers: &[u32]) -> bool {
    read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
        && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
        && registers[REG_TEXTURE_CONFIG] & 1 != 0
}

/// Kid Icarus Uprising's subtitles: its 512x128 RGBA8 text buffer drawn on
/// the top screen.
fn is_subtitle(registers: &[u32]) -> bool {
    let base = TEXTURE_UNIT_BASES[0];
    registers[REG_TEXTURE_CONFIG] & 1 != 0
        && registers[base + 1] == (512 << 16 | 128)
        && TextureFormat::from_raw(registers[base + 13]) == TextureFormat::Rgba8
}

/// draws the subtitles' text again first, black and partly see-through, a
/// pixel down and right of it, a shadow that keeps it readable over bright
/// scenes and the portraits.
fn draw_subtitle_shadow<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, vertices: Vertices) {
    /// how dark the shadow is, out of 255.
    const SHADOW_ALPHA: u32 = 0xC0;
    let mut shadow_buf = copy_regs(registers);
    let shadow = &mut shadow_buf[..registers.len()];
    // the first stage: constant black, with the text's alpha times the
    // constant's; the rest pass it on
    for (stage, &base) in TEV_STAGES.iter().enumerate() {
        let (source, combiner) = match stage {
            0 => (0xE | 3 << 16 | 0xE << 20, 1 << 16),
            _ => (0xF | 0xF << 16, 0),
        };
        shadow[base..base + 5].copy_from_slice(&[source, 0, combiner, SHADOW_ALPHA << 24, 0]);
    }
    // the top screen's buffer runs across its rows from the screen's
    // bottom, so a row lower is x - 1 and a column right y + 1
    let xy = registers[REG_VIEWPORT_XY];
    let shifted = |value: u32, by: i32| ((((value & 0x3FF) << 22) as i32 >> 22) + by) as u32 & 0x3FF;
    shadow[REG_VIEWPORT_XY] = shifted(xy, -1) | shifted(xy >> 16, 1) << 16;
    rasterize(shadow, memory, resources, vertices);
}

/// draws a crisp 4-way dark outline plus drop shadow under the bottom-centered
/// power name so white text stays legible even over bright explosions.
fn draw_power_label_outline<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    resources: &mut Resources,
    vertices: &[Vertex],
) {
    let fade = vertices.first().map_or(1.0, |v| v.color[3]).clamp(0.0, 1.0);
    if fade <= 0.01 {
        return;
    }
    let mut shadow_buf = copy_regs(registers);
    let shadow = &mut shadow_buf[..registers.len()];
    let mut shifted = Vec::with_capacity(vertices.len());
    // Shift vertices in clip space rather than REG_VIEWPORT_XY: +1 on viewport_x
    // makes viewport_right = 241 > buffer_width (240) and changes target stride!
    for (dx, dy, base_alpha) in [(-1.0f32, 0.0f32, 145.0), (1.0, 0.0, 145.0), (0.0, -1.0, 145.0), (0.0, 1.0, 145.0), (-1.0, 1.0, 195.0)] {
        let alpha = ((base_alpha * fade).round() as u32).min(255);
        for (stage, &base) in TEV_STAGES.iter().enumerate() {
            let (source, combiner) = match stage {
                0 => (0xE | 3 << 16 | 0xE << 20, 1 << 16),
                _ => (0xF | 0xF << 16, 0),
            };
            shadow[base..base + 5].copy_from_slice(&[source, 0, combiner, alpha << 24, 0]);
        }
        shifted.clear();
        shifted.extend(vertices.iter().copied().map(|mut v| {
            let w = v.clip[3];
            v.clip[0] += (dx / 120.0) * w;
            v.clip[1] += (dy / 200.0) * w;
            v
        }));
        rasterize(shadow, memory, resources, Vertices::Shaded(&shifted));
    }
}

/// the texture combiner's six stages, each five registers from there.
const TEV_STAGES: [usize; 6] = [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8];

/// remembers the top screen's color buffer, either a 240x400 top-screen draw
/// into VRAM or the buffer the last transfer to the top screen read from.
fn note_top_buffer(registers: &[u32], resources: &mut Resources) {
    let dimensions = registers[REG_FRAMEBUFFER_DIMENSIONS];
    let (width, height) = (dimensions & 0x7FF, ((dimensions >> 12) & 0x3FF) + 1);
    let format = ColorFormat::from_color_buffer((registers[REG_COLOR_BUFFER_FORMAT] >> 16) & 7);
    let start = loc_register(registers, REG_COLOR_BUFFER_ADDRESS);
    let in_vram_top = (0x1800_0000..0x1860_0000).contains(&start) && width >= 240 && height >= 400;
    let matches_source = resources.portraits.top_source.is_some_and(|source| {
        (start..start + width * height * format.bytes_per_pixel() as u32).contains(&source)
    });
    if in_vram_top || matches_source {
        resources.portraits.top = Some([
            registers[REG_COLOR_BUFFER_ADDRESS],
            registers[REG_COLOR_BUFFER_FORMAT],
            dimensions,
        ]);
    }
}

/// a draw's vertices in the order it takes them, shaded on the CPU.
fn shaded_in_order(registers: &[u32], vertices: Vertices) -> Option<Vec<Vertex>> {
    match vertices {
        Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } => {
            let (shaded, order) = process_vertices(registers, vertex_shader, geometry_shader, inputs, order);
            Some(match order {
                Some(order) => order.iter().map(|&i| shaded[i]).collect(),
                None => shaded,
            })
        }
        Vertices::Shaded(shaded) => Some(shaded.to_vec()),
        Vertices::ShadedOrdered { shaded, order } => Some(match order {
            Some(order) => order.iter().map(|&i| shaded[i]).collect(),
            None => shaded.to_vec(),
        }),
        #[cfg(feature = "vulkan")]
        Vertices::Raw { .. } => None,
    }
}

/// the part of a draw on the bottom screen, lowest and highest corners, x
/// across its 240 rows from the screen's bottom edge, y along them from its
/// left edge; none when none of it is on the screen.
fn bottom_box(vertices: &[Vertex]) -> Option<([f32; 2], [f32; 2])> {
    let viewport = (0.0, 0.0, 240.0, 320.0);
    let (mut low, mut high) = ([f32::MAX; 2], [f32::MIN; 2]);
    for screen in vertices.iter().filter(|vertex| inside(vertex)).filter_map(|vertex| to_screen(vertex, viewport)) {
        low = [low[0].min(screen.x), low[1].min(screen.y)];
        high = [high[0].max(screen.x), high[1].max(screen.y)];
    }
    let (low, high) = ([low[0].max(0.0), low[1].max(0.0)], [high[0].min(240.0), high[1].min(320.0)]);
    (low[0] < high[0] && low[1] < high[1]).then_some((low, high))
}

/// a bottom-screen draw's registers pointed at the top screen's buffer, the
/// whole bottom screen scaled by scale with its corner at x, y there, and
/// no depth, the top screen's buffer has none of the bottom one's.
fn onto_top(registers: &[u32], [address, format, dimensions]: [u32; 3], scale: f32, x: f32, y: f32) -> Vec<u32> {
    let mut patched = registers.to_vec();
    patched[REG_COLOR_BUFFER_ADDRESS] = address;
    patched[REG_COLOR_BUFFER_FORMAT] = format;
    patched[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
    patched[REG_VIEWPORT_WIDTH] = float24(120.0 * scale);
    patched[REG_VIEWPORT_HEIGHT] = float24(160.0 * scale);
    patched[REG_VIEWPORT_XY] = (x.round() as i32 as u32 & 0x3FF) | ((y.round() as i32 as u32 & 0x3FF) << 16);
    patched[REG_DEPTH_COLOR_MASK] &= !(1 | 1 << 12);
    patched[REG_DEPTH_STENCIL_WRITE] = 0;
    patched[REG_DEPTH_BUFFER_FORMAT] &= !3;
    patched
}

/// notes a portrait that fills much of the bottom screen, as a character's
/// does in a conversation, where the chapter select's pictures of chapters,
/// as alike in shape, are small.
fn note_big_portrait(registers: &[u32], resources: &mut Resources, vertices: &Vertices) {
    let vertices: Vec<Vertex> = match vertices {
        Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } => {
            let (shaded, order) = process_vertices(registers, vertex_shader, geometry_shader, inputs, *order);
            match order {
                Some(order) => order.iter().map(|&i| shaded[i]).collect(),
                None => shaded,
            }
        }
        Vertices::Shaded(shaded) => shaded.to_vec(),
        Vertices::ShadedOrdered { .. } => return,
        #[cfg(feature = "vulkan")]
        Vertices::Raw { .. } => return,
    };
    let Some((low, high)) = bottom_box(&vertices) else { return };
    let share = (high[0] - low[0]) * (high[1] - low[1]) / (240.0 * 320.0);
    if share >= 0.25 {
        resources.portraits.big_portrait = true;
    }
}

/// ZAKURO_BATTLE_DETECT=1 turns on note_battle_display, not yet checked on
/// the console against the menus.
fn battle_detection() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZAKURO_BATTLE_DETECT").is_some())
}

/// a land battle's bottom screen, which the game's overrides say nothing of
/// while the player stands still, draws the counters in the corner and the
/// items in the lower left, the two islands of the battle display together,
/// as no menu does. seen in one frame they say it is a battle.
fn note_battle_display(registers: &[u32], resources: &mut Resources, vertices: &Vertices) {
    let vertices: Vec<Vertex> = match vertices {
        Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } => {
            let (shaded, order) = process_vertices(registers, vertex_shader, geometry_shader, inputs, *order);
            match order {
                Some(order) => order.iter().map(|&i| shaded[i]).collect(),
                None => shaded,
            }
        }
        Vertices::Shaded(shaded) => shaded.to_vec(),
        Vertices::ShadedOrdered { .. } => return,
        #[cfg(feature = "vulkan")]
        Vertices::Raw { .. } => return,
    };
    let Some((low, high)) = bottom_box(&vertices) else { return };
    let [left, upper, right, lower] = [320.0 - high[1], 240.0 - high[0], 320.0 - low[1], 240.0 - low[0]];
    const SLACK: f32 = 2.0;
    for (index, island) in ISLANDS.iter().enumerate() {
        let [x0, y0, x1, y1] = island.from;
        if left >= x0 - SLACK && upper >= y0 - SLACK && right <= x1 + SLACK && lower <= y1 + SLACK {
            resources.portraits.hud_islands |= 1 << index;
        }
    }
    if resources.portraits.hud_islands == 0b11 {
        resources.portraits.big_portrait = true;
    }
}

/// takes a bottom-screen portrait for the top screen's buffer, smaller, in
/// the lower corner on its side, beside the subtitles. flush_portraits
/// draws it there.
fn mirror_portrait(registers: &[u32], resources: &mut Resources, vertices: Vertices) {
    /// how big the portrait is on the top screen, and how far from its side:
    /// none, the bottom screen's side edge cuts some, and that cut goes on
    /// the top screen's side edge so it does not show either.
    const SCALE: f32 = 0.4;
    const TUT_SCALE: f32 = 0.26;
    const MARGIN: f32 = 0.0;
    let Some(top) = resources.portraits.top else { return };
    let Some(vertices) = shaded_in_order(registers, vertices) else { return };
    // the screen's edge cuts it, so that edge goes on the top screen's
    // bottom edge and the cut does not show
    let Some((low, high)) = bottom_box(&vertices) else { return };
    // a conversation's portrait keeps the frontend in play while it is up
    if (high[0] - low[0]) * (high[1] - low[1]) / (240.0 * 320.0) >= 0.25 {
        resources.portraits.big_portrait = true;
    }
    // its lowest x on the bottom edge, and its y against the end of its own
    // side of the top screen's 400; y runs leftward from the screen's right
    // edge on both screens, so a low y is on the right.
    // During tutorial movies (`tutorial_dialogue_hold > 0`), both portraits sit
    // on the same bottom level framing the squeezed subtitle text (`x_screen = 0..51`
    // for Palutena, `53..193` for text, `194..245` for Pit), leaving `[246..400]`
    // completely clear for the 3D Switch control cards.
    let tutorial_mode = resources.portraits.tutorial_dialogue_hold > 0;
    let scale = if tutorial_mode { TUT_SCALE } else { SCALE };
    let right = (low[1] + high[1]) / 2.0 < 160.0;
    let x = -low[0] * scale;
    let y = match (tutorial_mode, right) {
        (true, true) => 155.0 - low[1] * scale,
        (false, true) => MARGIN - low[1] * scale,
        (_, false) => 400.0 - MARGIN - high[1] * scale,
    };
    let mut registers = onto_top(registers, top, scale, x, y);
    registers[TEXTURE_UNIT_BASES[0] + 2] |= 0x6;
    if registers[0x0CA] == 0x0002_0001 && registers[0x0D2] == 0x0009_0001 {
        // clean_portrait_texels decodes the Etc1 portrait's alpha into A and
        // unpremultiplies its dark border, so stage 2 modulates by tex0.a.
        registers[0x0D0] = (registers[0x0D0] & 0xFFFF) | 0x0F03_0000;
        registers[0x0D2] = (registers[0x0D2] & 0xFFFF) | 0x0001_0000;
    }
    // the title's pictures stop at the bottom screen's top edge, some cut
    // there, Medusa's snakes; on the top screen that edge fades out
    let mut vertices = faded_top(registers[REG_PRIMITIVE_CONFIG] >> 8 & 3, &vertices);
    if tutorial_mode && right && low[1] <= 4.0 {
        const WING_FADE: f32 = 20.0;
        let cut_y = low[1] + WING_FADE;
        vertices = split_triangles(&vertices, |v| (cut_y / 160.0 - 1.0) * v.clip[3] - v.clip[1]);
        for vertex in &mut vertices {
            let col = (vertex.clip[1] / vertex.clip[3] * 0.5 + 0.5) * 320.0;
            vertex.color[3] *= ((col - low[1]) / WING_FADE).clamp(0.0, 1.0);
        }
    }
    registers[REG_PRIMITIVE_CONFIG] &= !(3 << 8);
    resources.portraits.sides[!right as usize].incoming.push(PortraitDraw { registers, vertices });
}

/// how many of the bottom screen's rows below its top edge a portrait fades
/// out over on the top screen.
const PORTRAIT_FADE: f32 = 24.0;

/// a bottom-screen draw's triangles, as a list, fading out over the last
/// PORTRAIT_FADE rows before the bottom screen's top edge, by their alpha:
/// each triangle is cut along the band's lower edge, so the band has
/// corners of its own and its alpha runs from whole to none across it alone.
// ponytail: fades by the vertices' alpha, so only a draw that blends by it
// fades, as Kid Icarus Uprising's portraits do.
fn faded_top(topology: u32, vertices: &[Vertex]) -> Vec<Vertex> {
    // where an edge crosses the band's lower edge, in clip space, so the
    // cut is where it is however far the vertex is
    let edge = (240.0 - PORTRAIT_FADE) / 120.0 - 1.0;
    let below = |vertex: &Vertex| edge * vertex.clip[3] - vertex.clip[0];
    let mut faded = Vec::with_capacity(vertices.len() * 2);
    for (a, b, c) in assemble(topology, vertices.len()) {
        let triangle = [vertices[a], vertices[b], vertices[c]];
        // the part below the band's edge, then the part in it
        for side in [1.0, -1.0] {
            let mut polygon = [triangle[0]; 4];
            let mut poly_len = 0usize;
            for (i, current) in triangle.iter().enumerate() {
                let next = &triangle[(i + 1) % 3];
                let (d0, d1) = (side * below(current), side * below(next));
                if d0 >= 0.0 {
                    polygon[poly_len] = *current;
                    poly_len += 1;
                }
                if d0 * d1 < 0.0 {
                    polygon[poly_len] = current.lerp(next, d0 / (d0 - d1));
                    poly_len += 1;
                }
            }
            for i in 1..poly_len.saturating_sub(1) {
                faded.extend([polygon[0], polygon[i], polygon[i + 1]]);
            }
        }
    }
    for vertex in &mut faded {
        let row = (vertex.clip[0] / vertex.clip[3] * 0.5 + 0.5) * 240.0;
        vertex.color[3] *= ((240.0 - row) / PORTRAIT_FADE).clamp(0.0, 1.0);
    }
    faded
}

/// cuts a triangle list along a clip-space plane (`plane(v) == 0`) so alpha
/// ramps across a narrow bezel band stay confined to that band.
fn split_triangles(triangles: &[Vertex], plane: impl Fn(&Vertex) -> f32) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(triangles.len() + 12);
    for tri in triangles.chunks_exact(3) {
        for side in [1.0, -1.0] {
            let mut polygon = [tri[0]; 4];
            let mut poly_len = 0usize;
            for i in 0..3 {
                let current = &tri[i];
                let next = &tri[(i + 1) % 3];
                let (d0, d1) = (side * plane(current), side * plane(next));
                if d0 >= 0.0 {
                    polygon[poly_len] = *current;
                    poly_len += 1;
                }
                if d0 * d1 < 0.0 {
                    polygon[poly_len] = current.lerp(next, d0 / (d0 - d1));
                    poly_len += 1;
                }
            }
            for i in 1..poly_len.saturating_sub(1) {
                out.extend([polygon[0], polygon[i], polygon[i + 1]]);
            }
        }
    }
    out
}

/// a part of the bottom screen's battle display shown on the top screen:
/// where it is on the bottom screen, left, top, right and bottom from its
/// top left, where its top left goes on the top screen, and its scale.
#[derive(Clone, Copy)]
struct Island {
    from: [f32; 4],
    to: [f32; 2],
    scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HudKind {
    Counters,
    ItemCard,
    Buffs,
    Powers,
}

/// Kid Icarus Uprising's persistent battle display islands: the counters
/// (intensity, monsters defeated, hearts) beside SCORE on the top bar, and the
/// powers (air-battle orbs, land-battle power selector bar & callout) below
/// Pit's health bar.
// ponytail: fixed places for this one title, a per-title table when another
// game needs islands.
const ISLANDS: [Island; 2] = [
    Island { from: [196.0, 0.0, 320.0, 52.0], to: [232.0, 3.0], scale: 0.56 },
    Island { from: [0.0, 135.0, 215.0, 240.0], to: [8.0, 24.0], scale: 0.46 },
];

/// Land-battle item collection toast card (9-slice La4 backing, 64x64 item icon,
/// crimson title + black description), placed compactly along the mid-right edge
/// well clear of SCORE above and character portraits/subtitles below.
const ITEM_CARD_ISLAND: Island = Island {
    from: [23.0, 106.0, 301.0, 203.0],
    to: [269.0, 74.0],
    scale: 0.44,
};

/// Active buff / held-item status icons on the right edge of the bottom screen,
/// tucked neatly under SCORE and above the item pickup card.
const BUFFS_ISLAND: Island = Island {
    from: [280.0, 52.0, 320.0, 160.0],
    to: [374.0, 40.0],
    scale: 0.50,
};

/// Where the selected land-battle power's name is displayed on the top screen.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerLabelMode {
    /// Centered near the bottom of the top screen in crisp white shadowed text.
    Bottom = 0,
    /// Inside the speech bubble directly above the power selector bar.
    AboveBar = 1,
    /// Hidden (power icons only).
    Off = 2,
}

static POWER_LABEL_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(255);

pub fn set_power_label_mode(mode: PowerLabelMode) {
    POWER_LABEL_MODE.store(mode as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn power_label_mode() -> PowerLabelMode {
    match POWER_LABEL_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => PowerLabelMode::Bottom,
        1 => PowerLabelMode::AboveBar,
        2 => PowerLabelMode::Off,
        _ => match std::env::var("ZAKURO_POWER_LABEL").ok().as_deref().map(str::trim) {
            Some("bar" | "above" | "above_bar") => PowerLabelMode::AboveBar,
            Some("off" | "0" | "none") => PowerLabelMode::Off,
            _ => PowerLabelMode::Bottom,
        },
    }
}

/// whether a bottom-screen draw from KIU's 512x128 RGBA8 text buffer is the
/// top-of-bottom-screen dialogue box (used in How to Play) rather than a
/// battle HUD label (power name callout or item pickup card text).
fn is_bottom_dialogue_box(vertices: &[Vertex]) -> bool {
    let Some((low, high)) = bottom_box(vertices) else { return false };
    let (upper, lower) = (240.0 - high[0], 240.0 - low[0]);
    upper < 64.0 && lower <= 96.0
}

/// whether a bottom-screen draw from KIU's 512x128 RGBA8 text buffer lies
/// inside the active speech bubble above the power selector bar.
fn is_power_label_text(portraits: &Portraits, vertices: &[Vertex]) -> bool {
    let Some([b_low, b_high]) = portraits.power_bubble_x else { return false };
    let Some((low, high)) = bottom_box(vertices) else { return false };
    let [left, _upper, right, _lower] = [320.0 - high[1], 240.0 - high[0], 320.0 - low[1], 240.0 - low[0]];
    left <= 96.0 && right <= 225.0 && low[0] >= b_low - 3.0 && high[0] <= b_high + 3.0
}

/// redirects the selected power's name onto the bottom center of the top screen
/// at full readable scale in white shadowed text (`PowerLabelMode::Bottom`).
fn mirror_power_label(registers: &[u32], resources: &mut Resources, vertices: &[Vertex]) {
    let Some(top) = resources.portraits.top else { return };
    let Some((low, high)) = bottom_box(vertices) else { return };
    const SCALE: f32 = 1.28;
    let pop_slide = ((low[0] - 61.5) * 0.35).clamp(-6.0, 2.0);
    let target_low_x = 12.0 + pop_slide;
    let mid_y = (low[1] + high[1]) * 0.5;
    let mut moved: Vec<Vertex> = vertices
        .iter()
        .copied()
        .map(|mut vertex| {
            let w = vertex.clip[3];
            let below = [(vertex.clip[0] / w * 0.5 + 0.5) * 240.0, (vertex.clip[1] / w * 0.5 + 0.5) * 320.0];
            let above = [
                target_low_x + (below[0] - low[0]) * SCALE,
                200.0 + (below[1] - mid_y) * SCALE,
            ];
            vertex.clip[0] = (above[0] / 120.0 - 1.0) * w;
            vertex.clip[1] = (above[1] / 200.0 - 1.0) * w;
            // KIU tints the bottom-screen speech-bubble text black; on the top
            // screen's open scene it draws crisp white over draw_subtitle_shadow.
            vertex.color = [1.0, 1.0, 1.0, (vertex.color[3] * 0.92).clamp(0.0, 1.0)];
            vertex
        })
        .collect();
    let tex_addr = loc_register(registers, TEXTURE_UNIT_BASES[0] + 4);
    let buf_lines = resources.portraits.text_lines_for(tex_addr);
    if let Some(centered) = centered_subtitle(&moved, &buf_lines) {
        moved = centered;
    }
    // Trim the padded bottom 23% of the text quad so ascenders from an adjacent
    // packed row in the 512x128 scratch buffer (e.g. Practice Range stats) never
    // poke into the bottom edge below g/p/y descenders.
    let quad_h = (high[0] - low[0]) * SCALE;
    if quad_h >= 12.0 && moved.len().is_multiple_of(3) {
        let cut_x = target_low_x + quad_h * 0.23;
        let split = split_triangles(&moved, |v| (cut_x / 120.0 - 1.0) * v.clip[3] - v.clip[0]);
        moved = split
            .chunks_exact(3)
            .filter(|tri| {
                let mean_x: f32 = tri
                    .iter()
                    .map(|v| (v.clip[0] / v.clip[3] * 0.5 + 0.5) * 240.0)
                    .sum::<f32>()
                    / 3.0;
                mean_x >= cut_x - 0.02
            })
            .flatten()
            .copied()
            .collect();
    }
    let mut registers = onto_top(registers, top, 1.0, 0.0, 0.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(200.0);
    registers[TEXTURE_UNIT_BASES[0] + 2] |= 0x6;
    resources.portraits.power_label.push(PortraitDraw { registers, vertices: moved });
}

/// takes a bottom-screen draw that lies in one of the islands for the top
/// screen's buffer, where that island goes, for flush_portraits to draw.
fn mirror_hud(registers: &[u32], resources: &mut Resources, vertices: Vertices) {
    let Some(top) = resources.portraits.top else { return };
    let Some(vertices) = shaded_in_order(registers, vertices) else { return };
    let Some((low, high)) = bottom_box(&vertices) else { return };
    // in screen terms, left, top, right and bottom, y down: the buffer's
    // rows run up from the screen's bottom edge and leftward from its right
    let [left, upper, right, lower] = [320.0 - high[1], 240.0 - high[0], 320.0 - low[1], 240.0 - low[0]];
    const SLACK: f32 = 2.0;
    let within = |[x0, y0, x1, y1]: [f32; 4]| {
        left >= x0 - SLACK && upper >= y0 - SLACK && right <= x1 + SLACK && lower <= y1 + SLACK
    };
    let tex_base = TEXTURE_UNIT_BASES[0];
    let tex_fmt = TextureFormat::from_raw(registers[tex_base + 13]);
    let tex_dims = registers[tex_base + 1];
    let label_mode = power_label_mode();
    let is_item_card = left >= 20.0 - SLACK
        && upper >= 96.0 - SLACK
        && right <= 306.0 + SLACK
        && lower <= 208.0 + SLACK
        && (right >= 230.0
            || (is_subtitle(registers) && left >= 90.0 && upper < 155.0)
            || (matches!(tex_fmt, TextureFormat::Etc1 | TextureFormat::Etc1A4)
                && tex_dims == (64 << 16 | 64)
                && upper <= 150.0));
    let (kind, mut island) = if within(ISLANDS[0].from) {
        (HudKind::Counters, ISLANDS[0])
    } else if is_item_card {
        (HudKind::ItemCard, ITEM_CARD_ISLAND)
    } else if within(BUFFS_ISLAND.from) {
        (HudKind::Buffs, BUFFS_ISLAND)
    } else if within(ISLANDS[1].from)
        && (!is_subtitle(registers) || is_power_label_text(&resources.portraits, &vertices))
    {
        (HudKind::Powers, ISLANDS[1])
    } else {
        return;
    };
    if kind == HudKind::Powers && label_mode != PowerLabelMode::AboveBar {
        // without the speech bubble above the slots, dock the power bar snugly
        // right beneath Pit's HP wing
        island.to[1] = 12.0;
    }

    let topology = registers[REG_PRIMITIVE_CONFIG] >> 8 & 3;
    let mut triangulated = false;
    let mut vertices = vertices;
    if tex_fmt == TextureFormat::La4 {
        match kind {
            HudKind::Counters => {
                let mut cloud = Vec::new();
                let mut glyphs = Vec::new();
                for (a, b, c) in assemble(topology, vertices.len()) {
                    let tri = [vertices[a], vertices[b], vertices[c]];
                    let white = tri
                        .iter()
                        .all(|v| v.color[0] > 0.9 && v.color[1] > 0.9 && v.color[2] > 0.9);
                    let xs = tri.map(|v| (v.clip[0] / v.clip[3] * 0.5 + 0.5) * 240.0);
                    let ys = tri.map(|v| (v.clip[1] / v.clip[3] * 0.5 + 0.5) * 320.0);
                    let h = xs.into_iter().fold(f32::MIN, f32::max)
                        - xs.into_iter().fold(f32::MAX, f32::min);
                    let w = ys.into_iter().fold(f32::MIN, f32::max)
                        - ys.into_iter().fold(f32::MAX, f32::min);
                    if white && (w > 18.0 || h > 18.0) {
                        cloud.extend(tri);
                    } else {
                        glyphs.extend(tri);
                    }
                }
                if !cloud.is_empty() {
                    let mut cloud = split_triangles(&cloud, |v| {
                        (233.0 / 120.0 - 1.0) * v.clip[3] - v.clip[0]
                    });
                    cloud = split_triangles(&cloud, |v| {
                        (7.0 / 160.0 - 1.0) * v.clip[3] - v.clip[1]
                    });
                    for v in &mut cloud {
                        let bx = (v.clip[0] / v.clip[3] * 0.5 + 0.5) * 240.0;
                        let by = (v.clip[1] / v.clip[3] * 0.5 + 0.5) * 320.0;
                        let fx = ((240.0 - bx) / 7.0).clamp(0.0, 1.0);
                        let fy = (by / 7.0).clamp(0.0, 1.0);
                        v.color[3] *= 0.62 * fx * fy;
                    }
                    cloud.extend(glyphs);
                    vertices = cloud;
                    triangulated = true;
                }
            }
            HudKind::Powers
                if vertices.len() >= 24
                    && vertices
                        .first()
                        .is_some_and(|v| v.color[0] > 0.9 && v.color[1] > 0.9 && v.color[2] > 0.9) =>
            {
                let mut tris: Vec<Vertex> = assemble(topology, vertices.len())
                    .flat_map(|(a, b, c)| [vertices[a], vertices[b], vertices[c]])
                    .collect();
                for edge_x in [6.0, 42.0, 48.0] {
                    tris = split_triangles(&tris, |v| {
                        (edge_x / 120.0 - 1.0) * v.clip[3] - v.clip[0]
                    });
                }
                tris = split_triangles(&tris, |v| {
                    (314.0 / 160.0 - 1.0) * v.clip[3] - v.clip[1]
                });
                let (mut b_low, mut b_high) = (f32::MAX, f32::MIN);
                let mut tray_only = Vec::with_capacity(tris.len());
                for tri in tris.chunks_exact(3) {
                    let bxs = [tri[0], tri[1], tri[2]].map(|v| (v.clip[0] / v.clip[3] * 0.5 + 0.5) * 240.0);
                    let mean_bx = (bxs[0] + bxs[1] + bxs[2]) / 3.0;
                    if mean_bx > 42.05 {
                        b_low = b_low.min(bxs[0].min(bxs[1]).min(bxs[2]));
                        b_high = b_high.max(bxs[0].max(bxs[1]).max(bxs[2]));
                    } else {
                        tray_only.extend_from_slice(tri);
                    }
                }
                if b_low < b_high {
                    resources.portraits.power_bubble_x = Some([b_low, b_high]);
                }
                if label_mode != PowerLabelMode::AboveBar {
                    tris = tray_only;
                    if tris.is_empty() {
                        return;
                    }
                }
                for v in &mut tris {
                    let bx = (v.clip[0] / v.clip[3] * 0.5 + 0.5) * 240.0;
                    let by = (v.clip[1] / v.clip[3] * 0.5 + 0.5) * 320.0;
                    let fx = (bx / 6.0).clamp(0.0, 1.0);
                    let fy = ((320.0 - by) / 6.0).clamp(0.0, 1.0);
                    let callout_t = ((bx - 42.0) / 6.0).clamp(0.0, 1.0);
                    let panel_alpha = 0.60 + 0.28 * callout_t;
                    v.color[3] *= panel_alpha * fx * fy;
                }
                vertices = tris;
                triangulated = true;
            }
            HudKind::ItemCard if vertices.len() >= 36 => {
                for v in &mut vertices {
                    v.color[3] *= 0.70;
                }
            }
            _ => {}
        }
    }

    // the island's top left on the bottom screen lands on its place on the
    // top screen; in buffer terms x runs up from the screen's bottom edge
    // and y leftward from its right edge, 320 wide below and 400 above
    let [x0, y0, x1, y1] = island.from;
    let [to_x, to_y] = island.to;
    let (width, height) = ((x1 - x0) * island.scale, (y1 - y0) * island.scale);
    let x = (240.0 - to_y - height) - (240.0 - y1) * island.scale;
    let y = (400.0 - to_x - width) - (320.0 - x1) * island.scale;
    // the vertices move rather than the viewport, which would reach past
    // the top screen's buffer for an island moved up from the bottom, and
    // the GPU draws such a viewport wrong; it covers the whole top screen
    for vertex in &mut vertices {
        let w = vertex.clip[3];
        let below = [(vertex.clip[0] / w * 0.5 + 0.5) * 240.0, (vertex.clip[1] / w * 0.5 + 0.5) * 320.0];
        let above = [x + below[0] * island.scale, y + below[1] * island.scale];
        vertex.clip[0] = (above[0] / 120.0 - 1.0) * w;
        vertex.clip[1] = (above[1] / 200.0 - 1.0) * w;
    }
    let mut registers = onto_top(registers, top, 1.0, 0.0, 0.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(200.0);
    registers[tex_base + 2] |= 0x6;
    if triangulated {
        registers[REG_PRIMITIVE_CONFIG] &= !(3 << 8);
    }
    let hud_alpha: f32 = match kind {
        HudKind::Counters | HudKind::Powers => 0.88,
        HudKind::Buffs => 0.86,
        HudKind::ItemCard => 0.92,
    };
    if (registers[0x0F8] & 0x000F_000F) == 0x000F_000F && registers[0x0FA] == 0 {
        if registers[REG_COLOR_OPERATION] & 0x100 == 0 {
            registers[REG_COLOR_OPERATION] = (registers[REG_COLOR_OPERATION] & !0xFF) | 0x00E4_0100;
            registers[REG_BLEND_FUNC] = 0x0176_0000;
        }
        registers[0x0F8] = 0x00EF_000F;
        registers[0x0F9] = 0x0000_0000;
        registers[0x0FA] = 0x0001_0000;
        registers[0x0FB] = ((hud_alpha * 255.0).round() as u32) << 24;
        registers[0x0FC] = 0x0000_0000;
    }
    resources.portraits.hud.push(PortraitDraw { registers, vertices });
}

/// takes the bottom screen's dialogue box text, the title's text buffer
/// drawn there, for the top screen: centered across it, just above its
/// bottom edge, where subtitles go. for flush_portraits to draw.
fn mirror_dialogue(registers: &[u32], resources: &mut Resources, vertices: Vertices) {
    let Some(top) = resources.portraits.top else { return };
    let Some(vertices) = shaded_in_order(registers, vertices) else { return };
    let Some((low, high)) = bottom_box(&vertices) else { return };
    // buffer terms: x up from the screen's bottom edge, y along it leftward
    // from the right edge, 320 wide below and 400 above.
    // Squeeze the bottom-screen dialogue (`0.50x`) neatly between Palutena
    // (`x_screen = 0..51`) and Pit (`x_screen = 194..245`) along the same bottom
    // row (`x_screen = 51..195`), keeping `[246..400]` clear for Switch cards.
    const ABOVE_BOTTOM: f32 = 8.0;
    const SCALE: f32 = 0.50;
    let mid_y = (low[1] + high[1]) * 0.5;
    let half_w = (high[1] - low[1]) * 0.5 * SCALE;
    let center_y_buf = 277.0f32.clamp((205.0 + half_w).min(277.0), (349.0 - half_w).max(277.0));
    let mut vertices = vertices;
    for vertex in &mut vertices {
        let w = vertex.clip[3];
        let below = [(vertex.clip[0] / w * 0.5 + 0.5) * 240.0, (vertex.clip[1] / w * 0.5 + 0.5) * 320.0];
        let above = [
            ABOVE_BOTTOM + (below[0] - low[0]) * SCALE,
            center_y_buf + (below[1] - mid_y) * SCALE,
        ];
        vertex.clip[0] = (above[0] / 120.0 - 1.0) * w;
        vertex.clip[1] = (above[1] / 200.0 - 1.0) * w;
    }
    let mut registers = onto_top(registers, top, 1.0, 0.0, 0.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(200.0);
    registers[TEXTURE_UNIT_BASES[0] + 2] |= 0x6;
    resources.portraits.dialogue.push(PortraitDraw { registers, vertices });
}

/// redirects the bottom screen's Kid Icarus: Uprising logo and copyright
/// notice onto the top screen during the title screen, dropping the
/// bottom-screen sky backdrop and the "Touch!" call-to-action.
fn mirror_title(registers: &[u32], resources: &mut Resources, vertices: Vertices) {
    if is_title_bottom_sky(registers) {
        return;
    }
    let top = resources.portraits.top.unwrap_or([
        0x1808_0000 >> 3,
        registers[REG_COLOR_BUFFER_FORMAT],
        256 | (511 << 12),
    ]);
    let Some(vertices) = shaded_in_order(registers, vertices) else { return };
    let Some((low, high)) = bottom_box(&vertices) else { return };
    // in screen terms, left, top, right and bottom, y down (320x240 bottom screen)
    let [left, upper, right, lower] = [320.0 - high[1], 240.0 - high[0], 320.0 - low[1], 240.0 - low[0]];
    // drop full-screen backdrops
    if (right - left) >= 316.0 && (lower - upper) >= 236.0 {
        return;
    }
    // copyright notice at bottom-left (y >= 200, left < 10): keep at bottom-left of 400x240 top screen;
    // logo elements (y < 140): center horizontally (+40 px) and shift down slightly (-32 px in buffer x);
    // drop the "Touch!" prompt and its wing ornaments (y >= 140, left >= 40)
    let is_copyright = upper >= 200.0 && left < 10.0 && right <= 115.0;
    let is_logo = upper < 140.0;
    if !is_logo && !is_copyright {
        return;
    }
    let (dx, dy) = if is_copyright { (0.0, 80.0) } else { (-32.0, 40.0) };
    let mut vertices = vertices;
    for vertex in &mut vertices {
        let w = vertex.clip[3];
        let below = [(vertex.clip[0] / w * 0.5 + 0.5) * 240.0, (vertex.clip[1] / w * 0.5 + 0.5) * 320.0];
        let above = [below[0] + dx, below[1] + dy];
        vertex.clip[0] = (above[0] / 120.0 - 1.0) * w;
        vertex.clip[1] = (above[1] / 200.0 - 1.0) * w;
    }
    let mut registers = onto_top(registers, top, 1.0, 0.0, 0.0);
    registers[REG_VIEWPORT_HEIGHT] = float24(200.0);
    resources.portraits.title.push(PortraitDraw { registers, vertices });
}

/// draws a redirected title-screen element onto the top screen with fade-in alpha
/// and optional sizzle-reel corner transition (`corner_t` in `0.0..=1.0`).
fn draw_title_faded<M: GpuMemory>(
    memory: &mut M,
    resources: &mut Resources,
    draw: &PortraitDraw,
    alpha: f32,
    corner_t: f32,
    warmup_only: bool,
) {
    let is_copyright = draw
        .vertices
        .first()
        .is_some_and(|v| v.clip[0] < -0.7 * v.clip[3]);
    let s = corner_t * corner_t * (3.0 - 2.0 * corner_t);
    let effective_alpha = if warmup_only {
        0.0
    } else if is_copyright {
        alpha * (1.0 - s)
    } else {
        alpha
    };
    if !warmup_only && effective_alpha <= 0.004 {
        return;
    }
    let mut registers = draw.registers.clone();
    if let Some([address, format, dimensions]) = resources.portraits.top {
        registers[REG_COLOR_BUFFER_ADDRESS] = address;
        registers[REG_COLOR_BUFFER_FORMAT] = format;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
    }
    registers[REG_COLOR_OPERATION] = (registers[REG_COLOR_OPERATION] & !0xFF) | 0x00E4_0100;
    registers[REG_BLEND_FUNC] = 0x0176_0000;
    registers[0x0F8] = 0x00EF_000F;
    registers[0x0F9] = 0x0000_0000;
    registers[0x0FA] = 0x0001_0000;
    registers[0x0FB] = ((effective_alpha * 255.0).round() as u32) << 24;
    registers[0x0FC] = 0x0000_0000;
    resources.portraits.adding = true;
    if !warmup_only && !is_copyright && s > 0.0 {
        let scale = 1.0 + (0.42 - 1.0) * s;
        let cx = 125.85 + (201.35 - 125.85) * s;
        let cy = 192.75 + (333.80 - 192.75) * s;
        let mut transformed = draw.vertices.clone();
        for vertex in &mut transformed {
            let w = vertex.clip[3];
            let c0 = (vertex.clip[0] / w * 0.5 + 0.5) * 240.0;
            let c1 = (vertex.clip[1] / w * 0.5 + 0.5) * 400.0;
            let above0 = cx + (c0 - 125.85) * scale;
            let above1 = cy + (c1 - 192.75) * scale;
            vertex.clip[0] = (above0 / 120.0 - 1.0) * w;
            vertex.clip[1] = (above1 / 200.0 - 1.0) * w;
        }
        rasterize(&registers, memory, resources, Vertices::Shaded(&transformed));
    } else {
        rasterize(&registers, memory, resources, Vertices::Shaded(&draw.vertices));
    }
    resources.portraits.adding = false;
}

/// how much of a portrait shows over what is behind it once it has faded in,
/// slightly less than all so the scene can be seen through it.
const PORTRAIT_ALPHA: f32 = 0.85;

/// draws the portraits the bottom screen took last frame into the top
/// screen's buffer, fading in a new one over the one before, and out one the
/// game stopped drawing. whether it drew any. once a frame: under the
/// subtitles when there are any, else just before the top screen goes out.
pub(crate) fn flush_portraits<M: GpuMemory>(memory: &mut M, resources: &mut Resources) -> bool {
    // how much of a portrait fades in or out each frame, 8 frames in all
    let step = 1.0 / 8.0;
    let bottom_seen = std::mem::take(&mut resources.portraits.bottom_seen);
    if bottom_seen {
        resources.portraits.power_bubble_x = None;
    }
    if std::mem::take(&mut resources.portraits.bottom_tex_seen) {
        resources.portraits.bottom_tex_missed = 0;
        resources.portraits.bottom_blank = false;
    } else {
        resources.portraits.bottom_tex_missed = resources.portraits.bottom_tex_missed.saturating_add(1);
        if resources.portraits.bottom_tex_missed > 2 {
            resources.portraits.bottom_blank = true;
        }
    }
    resources.portraits.hud_islands = 0;
    resources.portraits.fuse_first_header = 0;
    let fuse_headers = std::mem::take(&mut resources.portraits.fuse_frame_seen);
    let fuse_weapon = std::mem::take(&mut resources.portraits.fuse_weapon_seen);
    let fuse_preview = std::mem::take(&mut resources.portraits.fuse_preview_seen);
    if resources.portraits.in_battle {
        resources.portraits.fuse_active = false;
        resources.portraits.fuse_reveal = false;
        resources.portraits.fuse_missed = 0;
    } else if fuse_headers {
        let reveal = fuse_weapon && !fuse_preview;
        resources.portraits.fuse_active = !reveal;
        resources.portraits.fuse_reveal = reveal;
        resources.portraits.fuse_missed = 0;
    } else if resources.portraits.fuse_active || resources.portraits.fuse_reveal {
        resources.portraits.fuse_missed = resources.portraits.fuse_missed.saturating_add(1);
        if resources.portraits.fuse_missed > 1 {
            resources.portraits.fuse_active = false;
            resources.portraits.fuse_reveal = false;
        }
    }
    resources.portraits.cauldron_mural_step = 0;
    let cauldron_top = std::mem::take(&mut resources.portraits.cauldron_top_seen);
    let cauldron_cursor = std::mem::take(&mut resources.portraits.cauldron_cursor_seen);
    let cauldron_pot = std::mem::take(&mut resources.portraits.cauldron_pot_seen);
    if resources.portraits.in_battle || resources.portraits.fuse_active || resources.portraits.title_sky_active {
        resources.portraits.cauldron_active = false;
        resources.portraits.cauldron_hud_active = false;
        resources.portraits.cauldron_tier_init = false;
        resources.portraits.cauldron_fade_buf = 0;
        resources.portraits.cauldron_missed = 0;
    } else if cauldron_top && cauldron_cursor {
        resources.portraits.cauldron_active = true;
        resources.portraits.cauldron_hud_active = true;
        resources.portraits.cauldron_missed = 0;
    } else if resources.portraits.cauldron_active && (cauldron_pot || cauldron_top) {
        // Keep FiendsCauldron active through the heart-dump launch animation
        // while hiding the interactive pouch & intensity cursor thumb!
        resources.portraits.cauldron_hud_active = false;
        resources.portraits.cauldron_missed = 0;
    } else if resources.portraits.cauldron_active {
        resources.portraits.cauldron_missed = resources.portraits.cauldron_missed.saturating_add(1);
        if resources.portraits.cauldron_missed > 2 {
            resources.portraits.cauldron_active = false;
            resources.portraits.cauldron_hud_active = false;
            resources.portraits.cauldron_tier_init = false;
            resources.portraits.cauldron_fade_buf = 0;
        }
    }
    resources.portraits.chapter_step = 0;
    let chapter_top = std::mem::take(&mut resources.portraits.chapter_top_seen);
    let frame_title_r = std::mem::take(&mut resources.portraits.chapter_frame_title_right);
    if frame_title_r > 0.0 {
        resources.portraits.chapter_title_right = frame_title_r;
    }
    let frame_num = std::mem::take(&mut resources.portraits.chapter_frame_num_bounds);
    if frame_num.1 > frame_num.0 + 20.0 {
        resources.portraits.chapter_num_bounds = frame_num;
    }
    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.title_sky_active
    {
        resources.portraits.chapter_active = false;
        resources.portraits.chapter_missed = 0;
    } else if chapter_top {
        resources.portraits.chapter_active = true;
        resources.portraits.chapter_missed = 0;
    } else if resources.portraits.chapter_active {
        resources.portraits.chapter_missed = resources.portraits.chapter_missed.saturating_add(1);
        if resources.portraits.chapter_missed > 2 {
            resources.portraits.chapter_active = false;
        }
    }
    resources.portraits.mode_top_step = 0;
    resources.portraits.solo_top_step = 0;
    resources.portraits.file_top_step = 0;
    resources.portraits.weapon_step = 0;
    resources.portraits.gear_step = 0;
    let mode_top = std::mem::take(&mut resources.portraits.mode_top_seen);
    let mode_bot = std::mem::take(&mut resources.portraits.mode_bot_seen);
    let solo_top = std::mem::take(&mut resources.portraits.solo_top_seen);
    let file_top = std::mem::take(&mut resources.portraits.file_top_seen);
    let weapon_sub_seen = std::mem::take(&mut resources.portraits.weapon_sub_seen);
    if mode_top {
        resources.portraits.mode_top_latched = true;
    }

    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active
        || resources.portraits.title_sky_active
        || solo_top
        || file_top
    {
        resources.portraits.mode_active = false;
        resources.portraits.mode_top_latched = false;
        resources.portraits.mode_rejected = false;
        resources.portraits.mode_missed = 0;
    } else if mode_bot {
        resources.portraits.mode_active = true;
        resources.portraits.mode_rejected = false;
        resources.portraits.mode_missed = 0;
    } else if mode_top {
        // Wait up to 2 top transfers for `BOT`'s `256x256 Rgba4` pills to confirm ModeSelect;
        // if `mode_top` persists without `mode_bot`, reject so sub-menus stay untouched.
        if resources.portraits.mode_active {
            resources.portraits.mode_missed = resources.portraits.mode_missed.saturating_add(1);
            if resources.portraits.mode_missed > 2 {
                resources.portraits.mode_active = false;
                resources.portraits.mode_rejected = true;
            }
        }
    } else if resources.portraits.mode_active || resources.portraits.mode_top_latched || resources.portraits.mode_rejected {
        resources.portraits.mode_missed = resources.portraits.mode_missed.saturating_add(1);
        if resources.portraits.mode_missed > 2 {
            resources.portraits.mode_active = false;
            resources.portraits.mode_top_latched = false;
            resources.portraits.mode_rejected = false;
        }
    }

    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active
        || resources.portraits.title_sky_active
        || resources.portraits.mode_active
        || file_top
    {
        resources.portraits.solo_active = false;
        resources.portraits.solo_missed = 0;
    } else if solo_top {
        resources.portraits.solo_active = true;
        resources.portraits.solo_missed = 0;
    } else if resources.portraits.solo_active {
        resources.portraits.solo_missed = resources.portraits.solo_missed.saturating_add(1);
        if resources.portraits.solo_missed > 2 {
            resources.portraits.solo_active = false;
        }
    }

    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active
        || resources.portraits.title_sky_active
        || resources.portraits.mode_active
        || resources.portraits.solo_active
    {
        resources.portraits.file_active = false;
        resources.portraits.file_missed = 0;
    } else if file_top {
        resources.portraits.file_active = true;
        resources.portraits.file_missed = 0;
    } else if resources.portraits.file_active {
        resources.portraits.file_missed = resources.portraits.file_missed.saturating_add(1);
        if resources.portraits.file_missed > 2 {
            resources.portraits.file_active = false;
        }
    }

    resources.portraits.power_bot_etc1 = false;
    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active
        || resources.portraits.title_sky_active
        || resources.portraits.mode_active
        || resources.portraits.solo_active
        || resources.portraits.file_active
    {
        resources.portraits.weapon_active = false;
        resources.portraits.weapon_sub = 0;
        resources.portraits.weapon_missed = 0;
        resources.portraits.weapon_modal_top = false;
        resources.portraits.power_bot_latched = false;
        resources.portraits.power_set_overlay = false;
    } else if weapon_sub_seen != 0 {
        resources.portraits.weapon_active = true;
        resources.portraits.weapon_sub = weapon_sub_seen;
        resources.portraits.weapon_missed = 0;
    } else if resources.portraits.weapon_active {
        resources.portraits.weapon_missed = resources.portraits.weapon_missed.saturating_add(1);
        if resources.portraits.weapon_missed > 2 {
            resources.portraits.weapon_active = false;
            resources.portraits.weapon_sub = 0;
            resources.portraits.weapon_modal_top = false;
            resources.portraits.power_bot_latched = false;
            resources.portraits.power_set_overlay = false;
        }
    }

    let title_sky_this_frame = std::mem::take(&mut resources.portraits.title_sky_seen);
    let title_sizzle_this_frame = std::mem::take(&mut resources.portraits.title_sizzle_seen);
    const TITLE_LOGO_DELAY_FRAMES: u32 = 248;
    const TITLE_LOGO_FADE_FRAMES: f32 = 36.0;
    const TITLE_SIZZLE_FADE_START: u32 = 335;
    const TITLE_SIZZLE_FADE_FRAMES: f32 = 16.0;
    if resources.portraits.in_battle
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active
        || resources.portraits.mode_active
        || resources.portraits.solo_active
        || resources.portraits.file_active
        || resources.portraits.weapon_active
    {
        resources.portraits.reset_title();
    } else if title_sky_this_frame {
        resources.portraits.title_sky_missed = 0;
        resources.portraits.title_sizzle_active = false;
        resources.portraits.title_sizzle_missed = 0;
    } else if resources.portraits.title_sky_active {
        resources.portraits.title_sky_missed = resources.portraits.title_sky_missed.saturating_add(1);
        if resources.portraits.title_sky_missed > 1 {
            let enter_sizzle = resources.portraits.title_frames >= TITLE_LOGO_DELAY_FRAMES;
            resources.portraits.title_sky_active = false;
            resources.portraits.title_alpha = 0.0;
            resources.portraits.title_frames = 0;
            resources.portraits.title_captured = false;
            resources.portraits.title_warmups = 0;
            resources.portraits.title.clear();
            if enter_sizzle {
                resources.portraits.title_sizzle_active = true;
                resources.portraits.title_sizzle_missed = 0;
            }
        }
    } else {
        resources.portraits.title.clear();
        if resources.portraits.title_sizzle_active {
            if title_sizzle_this_frame {
                resources.portraits.title_sizzle_missed = 0;
            } else {
                resources.portraits.title_sizzle_missed =
                    resources.portraits.title_sizzle_missed.saturating_add(1);
                if resources.portraits.title_sizzle_missed > 3 {
                    resources.portraits.title_sizzle_active = false;
                }
            }
        }
    }
    let mut drew_title = false;
    // Pit flies in across the top screen for the first ~248 title-screen guest
    // frames (~8.3s) and disappears in a starburst at the top-right as the
    // Triumphal trumpets sound. Only composite onto frames that actually rendered
    // the 3D title sky (`title_sky_this_frame`) and when `LargeTop` override is off.
    if resources.portraits.title_sky_active
        && title_sky_this_frame
        && !resources.portraits.menu_override
        && !resources.portraits.title.is_empty()
    {
        if resources.portraits.title_frames >= 16 && resources.portraits.title.len() >= 2 {
            resources.portraits.title_captured = true;
        }
        if resources.portraits.title_frames >= 16
            && (resources.portraits.title_warmups as usize) < resources.portraits.title.len()
        {
            if prewarm_title_draws_async(memory, &resources.portraits.title) {
                let idx = resources.portraits.title_warmups as usize;
                resources.portraits.title_warmups += 1;
                let draw = resources.portraits.title[idx].clone();
                draw_title_faded(memory, resources, &draw, 0.0, 0.0, true);
            }
        } else if resources.portraits.title_frames >= TITLE_LOGO_DELAY_FRAMES {
            let fade_in = ((resources.portraits.title_frames - TITLE_LOGO_DELAY_FRAMES + 1) as f32
                / TITLE_LOGO_FADE_FRAMES)
                .clamp(0.0, 1.0);
            let fade_out = if resources.portraits.title_frames >= TITLE_SIZZLE_FADE_START {
                (1.0 - (resources.portraits.title_frames - TITLE_SIZZLE_FADE_START + 1) as f32
                    / TITLE_SIZZLE_FADE_FRAMES)
                    .clamp(0.0, 1.0)
            } else {
                1.0
            };
            let alpha = fade_in * fade_out;
            resources.portraits.title_alpha = alpha;
            if alpha > 0.004 {
                let title = resources.portraits.title.clone();
                for draw in &title {
                    draw_title_faded(memory, resources, draw, alpha, 0.0, false);
                    drew_title = true;
                }
            }
        }
    }
    if ((resources.portraits.in_play
        || resources.portraits.fuse_active
        || resources.portraits.cauldron_active
        || resources.portraits.chapter_active)
        && !resources.portraits.in_battle)
        || resources.portraits.pause_active
    {
        resources.portraits.sides = Default::default();
        resources.portraits.hud.clear();
        resources.portraits.last_hud.clear();
        resources.portraits.power_label.clear();
        resources.portraits.last_power_label.clear();
        resources.portraits.power_bubble_x = None;
        resources.portraits.dialogue.clear();
        resources.portraits.last_dialogue.clear();
        resources.portraits.tutorial_dialogue_hold = 0;
        return drew_title;
    }
    let mut sides = std::mem::take(&mut resources.portraits.sides);
    let mut drew = false;
    for side in &mut sides {
        let incoming = std::mem::take(&mut side.incoming);
        if bottom_seen || !incoming.is_empty() {
            match (incoming.is_empty(), &mut side.shown) {
                // the same picture, where the game has it now
                (false, Some(shown)) if shown.draws.first().map(PortraitDraw::picture) == incoming.first().map(PortraitDraw::picture) => {
                    shown.draws = incoming;
                    shown.alpha = (shown.alpha + step).min(PORTRAIT_ALPHA);
                }
                // a new one, over the one before
                (false, _) => {
                    side.leaving = side.shown.take().or(side.leaving.take());
                    side.shown = Some(Faded { draws: incoming, alpha: step });
                }
                // the game stopped drawing it
                (true, Some(shown)) => shown.alpha -= step,
                (true, None) => {}
            }
            if let Some(leaving) = &mut side.leaving {
                leaving.alpha -= step;
            }
            for faded in [&mut side.leaving, &mut side.shown] {
                if faded.as_ref().is_some_and(|faded| faded.alpha <= 0.0) {
                    *faded = None;
                }
            }
        }
        for faded in [&side.leaving, &side.shown].into_iter().flatten() {
            for draw in &faded.draws {
                draw_faded(memory, resources, draw, faded.alpha);
                drew = true;
            }
        }
    }
    resources.portraits.sides = sides;
    // the battle display as the bottom screen had it, no fading, it is
    // there whenever the game draws it
    let hud = if bottom_seen || !resources.portraits.hud.is_empty() {
        let taken = std::mem::take(&mut resources.portraits.hud);
        resources.portraits.last_hud = taken.clone();
        taken
    } else {
        resources.portraits.last_hud.clone()
    };
    let top_target = resources.portraits.top;
    resources.portraits.adding = true;
    for mut draw in hud {
        if let Some([address, format, dimensions]) = top_target {
            draw.registers[REG_COLOR_BUFFER_ADDRESS] = address;
            draw.registers[REG_COLOR_BUFFER_FORMAT] = format;
            draw.registers[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
        }
        rasterize(&draw.registers, memory, resources, Vertices::Shaded(&draw.vertices));
        drew = true;
    }
    let power_label = if bottom_seen || !resources.portraits.power_label.is_empty() {
        let taken = std::mem::take(&mut resources.portraits.power_label);
        resources.portraits.last_power_label = taken.clone();
        taken
    } else {
        resources.portraits.last_power_label.clone()
    };
    // the dialogue box's text, over the portraits, with the subtitles' shadow
    let subtitle_up = resources.portraits.top_subtitle_quiet <= 12;
    resources.portraits.top_subtitle_quiet = resources.portraits.top_subtitle_quiet.saturating_add(1);
    if !resources.portraits.in_battle {
        resources.portraits.tutorial_dialogue_hold = 0;
    } else {
        resources.portraits.tutorial_dialogue_hold = resources.portraits.tutorial_dialogue_hold.saturating_sub(1);
    }
    /// how many flushes the text must stay before it shows.
    const DIALOGUE_HOLD: u32 = 6;
    let dialogue = if bottom_seen || !resources.portraits.dialogue.is_empty() {
        let taken = std::mem::take(&mut resources.portraits.dialogue);
        let portraits = &mut resources.portraits;
        if taken.is_empty() {
            // the game draws its bottom screen every other refresh at times,
            // only a few flushes in a row without the text end it
            portraits.dialogue_missing = portraits.dialogue_missing.saturating_add(1);
            if portraits.dialogue_missing > 4 {
                portraits.dialogue_held = 0;
            }
        } else {
            portraits.dialogue_missing = 0;
            portraits.dialogue_held = portraits.dialogue_held.saturating_add(1);
        }
        if !taken.is_empty() || resources.portraits.dialogue_missing > 4 {
            resources.portraits.last_dialogue = taken;
        }
        resources.portraits.last_dialogue.clone()
    } else {
        resources.portraits.last_dialogue.clone()
    };
    let dialogue = if resources.portraits.dialogue_held >= DIALOGUE_HOLD { dialogue } else { Vec::new() };
    let lift_power_label = subtitle_up || !dialogue.is_empty();
    if power_label_mode() == PowerLabelMode::Bottom {
        for mut draw in power_label {
            if let Some([address, format, dimensions]) = top_target {
                draw.registers[REG_COLOR_BUFFER_ADDRESS] = address;
                draw.registers[REG_COLOR_BUFFER_FORMAT] = format;
                draw.registers[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
            }
            if lift_power_label {
                for vertex in &mut draw.vertices {
                    vertex.clip[0] += (44.0 / 120.0) * vertex.clip[3];
                }
            }
            draw_power_label_outline(&draw.registers, memory, resources, &draw.vertices);
            rasterize(&draw.registers, memory, resources, Vertices::Shaded(&draw.vertices));
            drew = true;
        }
    }
    for mut draw in dialogue {
        if let Some([address, format, dimensions]) = top_target {
            draw.registers[REG_COLOR_BUFFER_ADDRESS] = address;
            draw.registers[REG_COLOR_BUFFER_FORMAT] = format;
            draw.registers[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
        }
        draw_subtitle_shadow(&draw.registers, memory, resources, Vertices::Shaded(&draw.vertices));
        rasterize(&draw.registers, memory, resources, Vertices::Shaded(&draw.vertices));
        drew = true;
    }
    resources.portraits.adding = false;
    drew
}

/// draws a portrait with only alpha of it showing over what is there.
fn draw_faded<M: GpuMemory>(memory: &mut M, resources: &mut Resources, draw: &PortraitDraw, alpha: f32) {
    let mut regs_buf = copy_regs(&draw.registers);
    let registers = &mut regs_buf[..draw.registers.len()];
    if let Some([address, format, dimensions]) = resources.portraits.top {
        registers[REG_COLOR_BUFFER_ADDRESS] = address;
        registers[REG_COLOR_BUFFER_FORMAT] = format;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = dimensions;
    }
    if alpha >= 1.0 {
        resources.portraits.adding = true;
        rasterize(registers, memory, resources, Vertices::Shaded(&draw.vertices));
        resources.portraits.adding = false;
        return;
    }
    let mut vertices = draw.vertices.clone();
    if registers[REG_COLOR_OPERATION] & 0x100 == 0 {
        // blends it with the constant alpha, color source * alpha plus the
        // buffer * (1 - alpha), the buffer's alpha kept
        registers[REG_COLOR_OPERATION] |= 0x100;
        registers[REG_BLEND_FUNC] = 12 << 16 | 13 << 20 | 1 << 28;
        registers[REG_BLEND_COLOR] = ((alpha * 255.0).round() as u32) << 24;
    } else {
        // it blends already, by its own alpha as the game draws it
        for vertex in &mut vertices {
            vertex.color[3] *= alpha;
        }
    }
    resources.portraits.adding = true;
    rasterize(registers, memory, resources, Vertices::Shaded(&vertices));
    resources.portraits.adding = false;
}

const REG_BLEND_COLOR: usize = 0x103;

/// the inverse of decode_float24, for normal values.
fn float24(value: f32) -> u32 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xFF) - 127 + 63;
    ((bits >> 31) << 23) | (exponent << 16) | ((bits >> 7) & 0xFFFF)
}

impl Resources {
    /// lets go of the textures decoded, what the last draws left to the
    /// next and what the host's GPU drew, memory having been put back as a
    /// saved state had it.
    pub(crate) fn forget(&mut self) {
        self.textures.forget();
        self.tex_env = Default::default();
        self.lighting = Default::default();
        #[cfg(feature = "vulkan")]
        {
            self.semantics = Default::default();
            if let Some(hardware) = self.hardware.as_mut() {
                hardware.forget();
            }
        }
    }

    /// what the next draws do for frame interpolation.
    pub(crate) fn set_pass(&mut self, pass: Pass) {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.hardware.as_mut() {
            hardware.pass = pass;
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = pass;
    }

    /// whether draws go into twins, frame interpolation drawing the
    /// pictures in between.
    pub(crate) fn twins(&self) -> bool {
        #[cfg(feature = "vulkan")]
        if let Some(hardware) = self.hardware.as_ref() {
            return hardware.twins();
        }
        false
    }
}

/// the buffers a draw fills, kept from one draw to the next. allocating
/// them for each draw costs more than filling them, far more on Windows,
/// which hands a large block out fresh from the system every time, its
/// pages faulting in again as they are first written.
#[derive(Default)]
struct Scratch {
    bytes: Vec<u8>,
    /// for each vertex index, the stamp of the last draw that named it and
    /// its place among that draw's vertices. a draw stamps the entries it
    /// uses instead of clearing a table as long as its largest index.
    stamps: Vec<u32>,
    places: Vec<u32>,
    stamp: u32,
    unique: Vec<u32>,
    order: Vec<usize>,
    inputs: Vec<[Vec4; shader::INPUT_REGISTERS]>,
    /// each vertex of a draw counted from the first one it uses.
    relative: Vec<u32>,
    plans: Plans,
    /// the arrays a draw hands the GPU to decode.
    #[cfg(feature = "vulkan")]
    raw: hardware::RawInputs,
}

impl Scratch {
    /// the vertices the indices name, each once, in the order they first
    /// come, and each index's place among them.
    fn unique_indices(&mut self, count: usize, mut get_index: impl FnMut(usize) -> u32) {
        if self.stamps.is_empty() {
            self.stamps.resize(4096, 0);
            self.places.resize(4096, 0);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            // the stamps came around, none of the old ones may match
            self.stamps.fill(0);
            self.stamp = 1;
        }
        self.unique.clear();
        self.unique.reserve(count);
        self.order.clear();
        self.order.reserve(count);
        for i in 0..count {
            let index = get_index(i);
            let idx = index as usize;
            if idx >= self.stamps.len() {
                let new_len = (idx + 1).max(self.stamps.len() * 2).max(256);
                self.stamps.resize(new_len, 0);
                self.places.resize(new_len, 0);
            }
            if self.stamps[idx] != self.stamp {
                self.stamps[idx] = self.stamp;
                self.places[idx] = self.unique.len() as u32;
                self.unique.push(index);
            }
            self.order.push(self.places[idx] as usize);
        }
    }
}

/// textures decoded to RGBA, kept across draws for as long as the bytes
/// they came from stay the same. decoding a texel for every sample, four
/// of them when filtering, costs far more than looking one up.
#[derive(Default)]
pub struct TextureCache {
    /// by address, format and size.
    entries: HashMap<TextureKey, Decoded>,
    texels: usize,
    /// counts command lists. nothing but drawing changes memory while one
    /// runs, so a texture checked during it is not checked again unless a
    /// draw writes over it.
    list: u64,
    /// where the draws of this list wrote, as address and length.
    written: Vec<(u32, u32)>,
    /// the texture pack whose pictures replace textures, and where a
    /// texture is laid out to hash it as Citra did.
    pack: Option<Arc<crate::pack::Pack>>,
    rows: Vec<u8>,
}

type TextureKey = (u32, TextureFormat, u32, u32);

/// a texture's texels, and the texture pack's picture replacing it.
type Found = (Arc<[[u8; 4]]>, Option<Arc<crate::pack::Material>>);

struct Decoded {
    /// the hash of the bytes it was decoded from, and the bytes, so that a
    /// change decodes again only the tiles it touched.
    hash: u64,
    bytes: Vec<u8>,
    texels: Arc<[[u8; 4]]>,
    /// the list it was last checked in, and how many writes that list had
    /// made by then.
    checked: (u64, usize),
    replacement: Option<Arc<crate::pack::Material>>,
}

/// how many texels the cache holds before it starts over, 256 MiB of them.
const CACHED_TEXELS: usize = 64 * 1024 * 1024;

impl TextureCache {
    /// replaces textures with the pack's pictures from now on, or with none.
    pub fn set_pack(&mut self, pack: Option<Arc<crate::pack::Pack>>) {
        self.pack = pack;
        self.entries.clear();
        self.texels = 0;
    }

    /// lets go of every texture, memory having been put back as a saved
    /// state had it.
    pub fn forget(&mut self) {
        self.entries.clear();
        self.texels = 0;
    }

    /// a new command list starts, and memory may have changed since the last.
    pub fn begin_list(&mut self) {
        self.list += 1;
        self.written.clear();
    }

    /// a draw writes to size bytes at addr.
    fn wrote(&mut self, addr: u32, size: u32) {
        if self.written.last() != Some(&(addr, size)) {
            self.written.push((addr, size));
        }
    }

    /// the texture, when it was checked during this list and nothing drew
    /// over it since.
    fn checked(&self, key: TextureKey, size: u32) -> Option<Found> {
        let decoded = self.entries.get(&key).filter(|d| d.bytes.len() == size as usize)?;
        let (list, writes) = decoded.checked;
        let addr = key.0 & !3;
        let overwritten = self.written[writes.min(self.written.len())..]
            .iter()
            .any(|&(start, length)| start < addr.saturating_add(size) && addr < start.saturating_add(length));
        (list == self.list && !overwritten).then(|| (decoded.texels.clone(), decoded.replacement.clone()))
    }

    fn decoded(&mut self, key: TextureKey, data: &[u8]) -> Found {
        let _timer = crate::stats::time(&crate::stats::TEXTURES);
        let (addr, format, width, height) = key;
        let height = height & 0xFFFF;
        let portrait = addr & 1 != 0;
        let title_sprite = addr & 2 != 0;
        let bpp = format.bits_per_pixel() as usize;
        let base_size = (width as usize * height as usize * bpp).div_ceil(8);
        let base_count = (width * height) as usize;
        let checked = (self.list, self.written.len());
        if let Some(decoded) = self.entries.get_mut(&key) {
            if decoded.bytes == data {
                decoded.checked = checked;
                return (decoded.texels.clone(), decoded.replacement.clone());
            }
            let hash = fingerprint(data);
            if !portrait && !title_sprite && data.len() == base_size && decoded.hash != hash && decoded.bytes.len() == data.len() && width % 8 == 0 && height % 8 == 0 {
                // a title changing a few sprites of an atlas rewrites a
                // handful of its 8x8 tiles, those alone are decoded again,
                // into a copy, as the old texels may still be uploading
                let mut copy: Arc<[[u8; 4]]> = Arc::from(&decoded.texels[..]);
                let texels = Arc::get_mut(&mut copy).expect("a copy no one else holds");
                let tile = data.len() / (width * height / 64) as usize;
                let tiles = decoded.bytes.chunks(tile).zip(data.chunks(tile)).enumerate();
                for (index, (_, new)) in tiles.filter(|(_, (old, new))| old != new) {
                    let (tile_x, tile_y) = (index as u32 % (width / 8) * 8, index as u32 / (width / 8) * 8);
                    if crate::texture::decode_tile(new, format, texels, width, tile_x, tile_y) {
                        continue;
                    }
                    for y in tile_y..tile_y + 8 {
                        for x in tile_x..tile_x + 8 {
                            texels[(y * width + x) as usize] = crate::texture::sample_texel(data, format, width, x, y);
                        }
                    }
                }
                decoded.bytes.copy_from_slice(data);
                decoded.texels = copy;
                decoded.hash = hash;
                decoded.replacement = picture_for(self.pack.as_deref(), &data[..base_size], &decoded.texels[..base_count], key, &mut self.rows);
            }
            if decoded.hash == hash && decoded.bytes.len() == data.len() {
                decoded.checked = checked;
                return (decoded.texels.clone(), decoded.replacement.clone());
            }
        }
        let hash = fingerprint(data);
        let mut decoded_texels = crate::texture::decode_texture(&data[..base_size], format, width, height);
        let replacement = if title_sprite {
            None
        } else {
            picture_for(self.pack.as_deref(), &data[..base_size], &decoded_texels[..base_count], key, &mut self.rows)
        };
        let texels: Arc<[[u8; 4]]> = if portrait {
            clean_portrait_texels(&mut decoded_texels, width, height);
            Arc::from(decoded_texels)
        } else if title_sprite {
            let cache = title_cache();
            let cache_key = (width, height, hash);
            let cached = cache
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get(&cache_key)
                .cloned();
            if let Some(cached) = cached {
                cached
            } else {
                let upscaled: Arc<[[u8; 4]]> =
                    Arc::from(upscale_title_texels(&decoded_texels, width, height));
                cache
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert(cache_key, upscaled.clone());
                upscaled
            }
        } else {
            if data.len() > base_size {
                let mut offset = base_size;
                let mut level = 1u32;
                while offset < data.len() {
                    let (w, h) = (width >> level, height >> level);
                    let level_size = (w as usize * h as usize * bpp).div_ceil(8);
                    if w < 8 || h < 8 || offset + level_size > data.len() {
                        break;
                    }
                    decoded_texels.extend_from_slice(&crate::texture::decode_texture(
                        &data[offset..offset + level_size],
                        format,
                        w,
                        h,
                    ));
                    offset += level_size;
                    level += 1;
                }
            }
            Arc::from(decoded_texels)
        };
        let count = texels.len();
        if self.texels + count > CACHED_TEXELS {
            self.entries.clear();
            self.texels = 0;
        }
        let decoded = Decoded { hash, bytes: data.to_vec(), texels: texels.clone(), checked, replacement: replacement.clone() };
        if let Some(old) = self.entries.insert(key, decoded) {
            self.texels -= old.texels.len();
        }
        self.texels += count;
        (texels, replacement)
    }
}

fn title_cache() -> &'static std::sync::Mutex<HashMap<(u32, u32, u64), Arc<[[u8; 4]]>>> {
    static TITLE_CACHE: std::sync::OnceLock<
        std::sync::Mutex<HashMap<(u32, u32, u64), Arc<[[u8; 4]]>>>,
    > = std::sync::OnceLock::new();
    TITLE_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// kicks off background bicubic upscaling of the title screen's RGBA8 sprites
/// on a worker thread as soon as the title draws are captured, returning `true`
/// once all of them are present in `title_cache()` so the GPU thread never
/// stalls on CPU upscaling.
fn prewarm_title_draws_async<M: GpuMemory>(memory: &mut M, draws: &[PortraitDraw]) -> bool {
    static IN_FLIGHT: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<(u32, u32, u64)>>> =
        std::sync::OnceLock::new();
    let in_flight = IN_FLIGHT.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    let base = TEXTURE_UNIT_BASES[0];
    let mut all_ready = true;
    let mut to_spawn: Vec<(u32, u32, u64, Vec<u8>)> = Vec::new();
    for draw in draws {
        if draw.registers[REG_TEXTURE_CONFIG] & 1 == 0 {
            continue;
        }
        let format = crate::texture::TextureFormat::from_raw(draw.registers[base + 13]);
        if format != TextureFormat::Rgba8 {
            continue;
        }
        let dimensions = draw.registers[base + 1];
        let height = dimensions & 0x7FF;
        let width = (dimensions >> 16) & 0x7FF;
        if width == 0 || height == 0 {
            continue;
        }
        let address = loc_register(&draw.registers, base + 4);
        if address == 0 {
            continue;
        }
        let addr = memory.translate(address);
        let size = (width as usize) * (height as usize) * 4;
        let bytes = match memory.slice(addr, size) {
            Some(slice) => slice.to_vec(),
            None => {
                let mut buf = vec![0u8; size];
                memory.read(addr, &mut buf);
                buf
            }
        };
        let hash = fingerprint(&bytes);
        let key = (width, height, hash);
        let ready = title_cache()
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .contains_key(&key);
        if !ready {
            all_ready = false;
            let mut guard = in_flight.lock().unwrap_or_else(|p| p.into_inner());
            if guard.insert(key) {
                to_spawn.push((width, height, hash, bytes));
            }
        }
    }
    if !to_spawn.is_empty() {
        std::thread::spawn(move || {
            for (width, height, hash, bytes) in to_spawn {
                let decoded = crate::texture::decode_texture(&bytes, TextureFormat::Rgba8, width, height);
                let upscaled: Arc<[[u8; 4]]> = Arc::from(upscale_title_texels(&decoded, width, height));
                title_cache()
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .insert((width, height, hash), upscaled);
            }
        });
    }
    all_ready
}

/// upscales a 3DS title-screen RGBA8 sprite (the Kid Icarus: Uprising logo and
/// copyright notice) 2x in memory using clamped Catmull-Rom bicubic
/// interpolation in premultiplied-alpha space followed by an unsharp mask, and
/// builds a 3-level sharpened mipmap pyramid so both the full-size centered
/// logo and the 42%-scaled sizzle-reel corner logo sample cleanly without
/// blur or dark alpha fringes.
fn upscale_title_texels(base: &[[u8; 4]], width: u32, height: u32) -> Vec<[u8; 4]> {
    let (w, h) = (width as usize, height as usize);
    let mut pm = vec![[0.0f32; 4]; w * h];
    for (dst, &[r, g, b, a]) in pm.iter_mut().zip(base.iter()) {
        let af = a as f32 / 255.0;
        *dst = [r as f32 * af, g as f32 * af, b as f32 * af, a as f32];
    }
    let sample_pm = |x: i32, y: i32| -> [f32; 4] {
        let cx = x.clamp(0, w as i32 - 1) as usize;
        let cy = y.clamp(0, h as i32 - 1) as usize;
        pm[cy * w + cx]
    };
    let catmull_rom = |t: f32| -> [f32; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ]
    };
    let (out_w, out_h) = (w * 2, h * 2);
    let mut up_pm = vec![[0.0f32; 4]; out_w * out_h];
    for oy in 0..out_h {
        let sy = (oy as f32 + 0.5) * 0.5 - 0.5;
        let iy = sy.floor() as i32;
        let wy = catmull_rom(sy - iy as f32);
        for ox in 0..out_w {
            let sx = (ox as f32 + 0.5) * 0.5 - 0.5;
            let ix = sx.floor() as i32;
            // Fast-path: if the 2x2 clamping neighborhood is completely transparent,
            // the clamped result is guaranteed to be [0.0; 4].
            if sample_pm(ix, iy)[3] == 0.0
                && sample_pm(ix + 1, iy)[3] == 0.0
                && sample_pm(ix, iy + 1)[3] == 0.0
                && sample_pm(ix + 1, iy + 1)[3] == 0.0
            {
                continue;
            }
            let wx = catmull_rom(sx - ix as f32);
            let mut sum = [0.0f32; 4];
            let mut lo = [f32::MAX; 4];
            let mut hi = [f32::MIN; 4];
            for dy in -1..=2 {
                let row_w = wy[(dy + 1) as usize];
                for dx in -1..=2 {
                    let p = sample_pm(ix + dx, iy + dy);
                    let weight = row_w * wx[(dx + 1) as usize];
                    for c in 0..4 {
                        sum[c] += p[c] * weight;
                        if (0..=1).contains(&dx) && (0..=1).contains(&dy) {
                            lo[c] = lo[c].min(p[c]);
                            hi[c] = hi[c].max(p[c]);
                        }
                    }
                }
            }
            for c in 0..4 {
                sum[c] = sum[c].clamp(lo[c], hi[c]);
            }
            up_pm[oy * out_w + ox] = sum;
        }
    }
    let sharpen_and_unpremult = |buf: &[[f32; 4]], bw: usize, bh: usize, strength: f32, wide_strength: f32| -> Vec<[u8; 4]> {
        let mut out = vec![[0u8; 4]; bw * bh];
        for y in 0..bh {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(bh - 1);
            let ym2 = y.saturating_sub(2);
            let yp2 = (y + 2).min(bh - 1);
            for x in 0..bw {
                let center = buf[y * bw + x];
                if center[3] == 0.0 {
                    continue;
                }
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(bw - 1);
                let xm2 = x.saturating_sub(2);
                let xp2 = (x + 2).min(bw - 1);
                let mut blur = [0.0f32; 4];
                let mut lo = center;
                let mut hi = center;
                let neighbors = [
                    (xm, ym, 1.0), (x, ym, 2.0), (xp, ym, 1.0),
                    (xm, y,  2.0), (x, y,  4.0), (xp, y,  2.0),
                    (xm, yp, 1.0), (x, yp, 2.0), (xp, yp, 1.0),
                ];
                for &(nx, ny, w_k) in &neighbors {
                    let p = buf[ny * bw + nx];
                    for c in 0..4 {
                        blur[c] += p[c] * (w_k / 16.0);
                        lo[c] = lo[c].min(p[c]);
                        hi[c] = hi[c].max(p[c]);
                    }
                }
                let mut wide_blur = [0.0f32; 4];
                if wide_strength > 0.0 {
                    let wide_neighbors = [
                        (xm2, ym2, 1.0), (x, ym2, 2.0), (xp2, ym2, 1.0),
                        (xm2, y,   2.0), (x, y,   4.0), (xp2, y,   2.0),
                        (xm2, yp2, 1.0), (x, yp2, 2.0), (xp2, yp2, 1.0),
                    ];
                    for &(nx, ny, w_k) in &wide_neighbors {
                        let p = buf[ny * bw + nx];
                        for c in 0..4 {
                            wide_blur[c] += p[c] * (w_k / 16.0);
                            lo[c] = lo[c].min(p[c]);
                            hi[c] = hi[c].max(p[c]);
                        }
                    }
                }
                let mut s = [0.0f32; 4];
                for c in 0..4 {
                    let delta = strength * (center[c] - blur[c]) + wide_strength * (center[c] - wide_blur[c]);
                    let headroom = (hi[c] - lo[c]) * 0.08;
                    s[c] = (center[c] + delta).clamp((lo[c] - headroom).max(0.0), (hi[c] + headroom).min(255.0));
                }
                let a = s[3].round() as u8;
                if a <= 2 {
                    out[y * bw + x] = [0, 0, 0, 0];
                } else {
                    let af = s[3] / 255.0;
                    let r = (s[0] / af).round().clamp(0.0, 255.0) as u8;
                    let g = (s[1] / af).round().clamp(0.0, 255.0) as u8;
                    let b = (s[2] / af).round().clamp(0.0, 255.0) as u8;
                    out[y * bw + x] = [r, g, b, a];
                }
            }
        }
        out
    };
    let mut total_texels = sharpen_and_unpremult(&up_pm, out_w, out_h, 1.35, 0.65);
    let mut cur_pm = up_pm;
    let (mut cur_w, mut cur_h) = (out_w, out_h);
    for level in 1..=3 {
        let (nw, nh) = (cur_w >> 1, cur_h >> 1);
        if nw < 8 || nh < 8 {
            break;
        }
        let mut next_pm = vec![[0.0f32; 4]; nw * nh];
        for y in 0..nh {
            for x in 0..nw {
                let p00 = cur_pm[(y * 2) * cur_w + (x * 2)];
                let p10 = cur_pm[(y * 2) * cur_w + (x * 2 + 1)];
                let p01 = cur_pm[(y * 2 + 1) * cur_w + (x * 2)];
                let p11 = cur_pm[(y * 2 + 1) * cur_w + (x * 2 + 1)];
                next_pm[y * nw + x] = std::array::from_fn(|c| 0.25 * (p00[c] + p10[c] + p01[c] + p11[c]));
            }
        }
        let mip_strength = if level == 1 { 0.65 } else { 0.35 };
        total_texels.extend_from_slice(&sharpen_and_unpremult(&next_pm, nw, nh, mip_strength, 0.0));
        cur_pm = next_pm;
        cur_w = nw;
        cur_h = nh;
    }
    total_texels
}

/// Kid Icarus Uprising's Etc1 portraits encode transparency by shifting the
/// character's color up by a 1/4 pedestal and taking alpha from 2 * (R + G),
/// with the outer edge anti-aliased against black. Unpremultiplying boundary
/// texels and writing true alpha into A keeps the dark border from showing
/// when scaled and filtered over the bright top screen.
fn clean_portrait_texels(texels: &mut [[u8; 4]], width: u32, height: u32) {
    let (w, h) = (width as usize, height as usize);
    let raw = texels.to_vec();
    for y in 0..h {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(h - 1);
        let by0 = y.saturating_sub(2);
        let by1 = (y + 2).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(w - 1);
            let bx0 = x.saturating_sub(2);
            let bx1 = (x + 2).min(w - 1);
            let [r, g, b, _] = raw[y * w + x];
            let s = r as u32 + g as u32;
            let mut s_min = u32::MAX;
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let [nr, ng, _, _] = raw[ny * w + nx];
                    s_min = s_min.min(nr as u32 + ng as u32);
                }
            }
            if s_min <= 32 {
                let mut s_max = 0u32;
                let mut best = [r, g, b];
                for ny in by0..=by1 {
                    for nx in bx0..=bx1 {
                        let [nr, ng, nb, _] = raw[ny * w + nx];
                        let ns = nr as u32 + ng as u32;
                        if ns > s_max {
                            s_max = ns;
                            best = [nr, ng, nb];
                        }
                    }
                }
                texels[y * w + x] = if s <= 32 || s_max <= 96 {
                    [best[0], best[1], best[2], 0]
                } else {
                    let alpha = ((s - 16) as f32 / (s_max - 16) as f32).clamp(0.05, 1.0);
                    let unpremult = |c: u8| (c as f32 / alpha).round().clamp(0.0, 255.0) as u8;
                    [unpremult(r), unpremult(g), unpremult(b), (alpha * 255.0).round() as u8]
                };
            } else {
                texels[y * w + x] = [r, g, b, (s * 2).min(255) as u8];
            }
        }
    }
}

/// the pack's picture for a texture of these bytes, whenever they change.
fn picture_for(
    pack: Option<&crate::pack::Pack>,
    data: &[u8],
    texels: &[[u8; 4]],
    key: TextureKey,
    rows: &mut Vec<u8>,
) -> Option<Arc<crate::pack::Material>> {
    let (_, format, width, height) = key;
    let height = height & 0xFFFF;
    if crate::pack::recording() {
        crate::pack::record(data, texels, format, width, height);
    }
    pack?.find(data, format, width, height, rows)
}

/// a quick hash of a texture's bytes, to notice when they change. four
/// running hashes side by side, which the CPU works on at once.
fn fingerprint(bytes: &[u8]) -> u64 {
    const K: u64 = 0x517C_C1B7_2722_0A95;
    let step = |hash: u64, word: u64| (hash.rotate_left(5) ^ word).wrapping_mul(K);
    let mut lanes = [bytes.len() as u64, 1, 2, 3];
    let (blocks, rest) = bytes.as_chunks::<32>();
    for block in blocks {
        for (lane, word) in lanes.iter_mut().zip(block.as_chunks::<8>().0) {
            *lane = step(*lane, u64::from_le_bytes(*word));
        }
    }
    let hash = lanes.into_iter().fold(0, step);
    rest.iter().fold(hash, |hash, &byte| step(hash, byte as u64))
}

/// how a texture coordinate outside 0..1 is brought back inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Wrap {
    ClampToEdge,
    /// outside the texture, the unit's border color.
    ClampToBorder,
    Repeat,
    MirroredRepeat,
}

impl Wrap {
    fn from_raw(value: u32) -> Wrap {
        // three bits, of which only the low two are documented, the upper
        // four encodings behave like these, as Citra established.
        match value & 0x7 {
            0 | 4 => Wrap::ClampToEdge,
            1 | 5 => Wrap::ClampToBorder,
            3 => Wrap::MirroredRepeat,
            _ => Wrap::Repeat,
        }
    }

    /// maps an integer texel coordinate into 0..size.
    fn apply(self, coordinate: i32, size: u32) -> u32 {
        let size = size as i32;
        let wrapped = match self {
            Wrap::ClampToEdge | Wrap::ClampToBorder => coordinate.clamp(0, size - 1),
            Wrap::Repeat => coordinate.rem_euclid(size),
            Wrap::MirroredRepeat => {
                let period = coordinate.rem_euclid(size * 2);
                if period < size {
                    period
                } else {
                    size * 2 - 1 - period
                }
            }
        };
        wrapped as u32
    }
}

impl BoundTexture {
    fn texel(&self, x: i32, y: i32) -> [f32; 4] {
        let outside = |wrap: Wrap, coordinate: i32, size: u32| {
            wrap == Wrap::ClampToBorder && !(0..size as i32).contains(&coordinate)
        };
        if outside(self.wrap_s, x, self.width) || outside(self.wrap_t, y, self.height) {
            return self.border;
        }
        let x = self.wrap_s.apply(x, self.width);
        let y = self.wrap_t.apply(y, self.height);
        self.texels[(y * self.width + x) as usize].map(|c| c as f32 / 255.0)
    }

    /// samples at (u, v), filtering the way the unit is configured.
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        // PICA texture coordinates have v=0 at the bottom row.
        let fx = u * self.width as f32;
        let fy = (1.0 - v) * self.height as f32;

        if !self.linear {
            return self.texel(fx.floor() as i32, fy.floor() as i32);
        }

        // texel centers sit at half-integer positions.
        let fx = fx - 0.5;
        let fy = fy - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as i32, y0 as i32);

        let a = self.texel(x0, y0);
        let b = self.texel(x0 + 1, y0);
        let c = self.texel(x0, y0 + 1);
        let d = self.texel(x0 + 1, y0 + 1);
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * tx;
            let bottom = c[i] + (d[i] - c[i]) * tx;
            top + (bottom - top) * ty
        })
    }
}

/// first register of each texture unit's block.
const TEXTURE_UNIT_BASES: [usize; 3] = [0x081, 0x091, 0x099];

/// what a unit that is on but points at no texture samples: transparent
/// black. the PICA gives the last color it fetched there, which drawing
/// can't follow, and transparent comes closest. Pokémon X and Y draw the
/// male player's cheeks with no texture, squares that an alpha test drops
/// instead of showing them black.
fn empty_texture() -> BoundTexture {
    static TEXELS: std::sync::OnceLock<Arc<[[u8; 4]]>> = std::sync::OnceLock::new();
    BoundTexture {
        texels: TEXELS.get_or_init(|| Arc::from([[0u8; 4]].as_slice())).clone(),
        drawn: None,
        linear: false,
        min_linear: false,
        mip_linear: false,
        max_level: 0,
        lod_bias: 0,
        wrap_s: Wrap::Repeat,
        wrap_t: Wrap::Repeat,
        width: 1,
        height: 1,
        border: [0.0; 4],
        replacement: None,
    }
}

/// reads the configuration of one texture unit and copies its image out of
/// guest memory, or None when the unit is disabled or unusable.
fn bind_texture<M: GpuMemory>(
    registers: &[u32],
    memory: &mut M,
    cache: &mut TextureCache,
    unit: usize,
    drawn: Option<DrawnTexture>,
    past: bool,
) -> Option<BoundTexture> {
    // one bit per unit in GPUREG_TEXUNIT_CONFIG.
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }

    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let height = dimensions & 0x7FF;
    let width = (dimensions >> 16) & 0x7FF;
    if width == 0 || height == 0 {
        return Some(empty_texture());
    }

    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = crate::texture::TextureFormat::from_raw(registers[format_register]);
    let address = loc_register(registers, base + 4);
    if address == 0 {
        // a unit pointed at nothing.
        return Some(empty_texture());
    }

    let addr = memory.translate(address);
    log::trace!("texture unit {unit}: 0x{address:08X} -> 0x{addr:08X}, {width}x{height} {format:?}");
    let portrait = unit == 0 && format == TextureFormat::Etc1 && registers[0x0CA] == 0x0002_0001 && registers[0x0D0] >> 16 == 0x0F03;
    let title_sprite = unit == 0
        && format == TextureFormat::Rgba8
        && registers[0x0F8] == 0x00EF_000F
        && registers[0x0FA] == 0x0001_0000;
    let lod = registers[base + 3];
    let req_max_level = if portrait || title_sprite || drawn.is_some() { 0 } else { (lod >> 16) & 0xF };
    let mut max_level = 0u32;
    while max_level < req_max_level && (width >> (max_level + 1)) >= 8 && (height >> (max_level + 1)) >= 8 {
        max_level += 1;
    }
    let bpp = format.bits_per_pixel() as u64;
    let mut total_bits = 0u64;
    for level in 0..=max_level {
        total_bits += (width >> level) as u64 * (height >> level) as u64 * bpp;
    }
    let size = total_bits.div_ceil(8) as usize;
    let lod_bias = (((lod & 0x1FFF) as i32) << 19 >> 19) as i16;

    let key = (
        addr | (portrait as u32) | ((title_sprite as u32) << 1),
        format,
        width,
        height | (max_level << 16),
    );
    let (texels, replacement) = if drawn.is_some() && !past {
        (Arc::default(), None)
    } else if let Some(found) = cache.checked(key, size as u32) {
        found
    } else {
        match memory.slice(addr, size) {
            Some(data) => cache.decoded(key, data),
            None => {
                let mut data = vec![0u8; size];
                memory.read(addr, &mut data);
                cache.decoded(key, &data)
            }
        }
    };

    // filter mode in bit 1 (magnification), 2 (minification), and 24 (mipmap),
    // the wrap modes for T and S in bits 8-10 and 12-14.
    let config = registers[base + 2];
    let (out_width, out_height, out_max_level, out_lod_bias) = if title_sprite {
        (width * 2, height * 2, 3u8, -192i16)
    } else {
        (width, height, max_level as u8, lod_bias)
    };
    let mipmapped = out_max_level > 0;
    Some(BoundTexture {
        texels,
        drawn,
        linear: config & 0x2 != 0 || mipmapped || filter_mode() == 3,
        min_linear: mipmapped || match filter_mode() {
            3 => true,
            1 => config & 0x2 != 0,
            0 => config & 0x4 != 0,
            _ => config & 0x6 != 0,
        },
        mip_linear: config & (1 << 24) != 0 || mipmapped || filter_mode() == 3,
        max_level: out_max_level,
        lod_bias: out_lod_bias,
        wrap_t: if title_sprite { Wrap::ClampToEdge } else { Wrap::from_raw(config >> 8) },
        wrap_s: if title_sprite { Wrap::ClampToEdge } else { Wrap::from_raw(config >> 12) },
        width: out_width,
        height: out_height,
        // the border color register comes first in each unit's block, RGBA8
        border: registers[base].to_le_bytes().map(|c| c as f32 / 255.0),
        // a buffer drawn into is the GPU's, not a pack's picture
        replacement: replacement.filter(|_| drawn.is_none()),
    })
}

/// a texture unit's texture as rows of a color buffer, when it is on and in
/// a format a color buffer has.
#[cfg(feature = "vulkan")]
fn drawn_texture<M: GpuMemory>(registers: &[u32], memory: &M, unit: usize) -> Option<DrawnTexture> {
    use crate::texture::TextureFormat;
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }
    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let (height, width) = (dimensions & 0x7FF, (dimensions >> 16) & 0x7FF);
    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = match TextureFormat::from_raw(registers[format_register]) {
        TextureFormat::Rgba8 => ColorFormat::Rgba8,
        TextureFormat::Rgb8 => ColorFormat::Rgb8,
        TextureFormat::Rgba5551 => ColorFormat::Rgb5A1,
        TextureFormat::Rgb565 => ColorFormat::Rgb565,
        TextureFormat::Rgba4 => ColorFormat::Rgba4,
        _ => return None,
    };
    let address = loc_register(registers, base + 4);
    (width != 0 && height != 0 && address != 0).then(|| DrawnTexture { addr: memory.translate(address), width, height, format })
}

/// with ZAKURO_LOG_DRAWS=1, logs each texture a draw reads and the buffer it
/// draws into, the first time the pair shows up and again whenever it comes
/// back after two seconds unseen. the id is the texture's hash as Citra
/// names it, the name ZAKURO_DUMP_TEXTURES saves it under, to find which
/// draws are which picture. it hashes up to 256 KiB a draw, so drawing slows.
fn log_draw<M: GpuMemory>(registers: &[u32], memory: &mut M, target: u32, buffer: (u32, u32), viewport: (f32, f32, f32, f32)) {
    use std::hash::{Hash, Hasher};
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};
    static ON: OnceLock<bool> = OnceLock::new();
    static SEEN: Mutex<Option<HashMap<u64, Instant>>> = Mutex::new(None);
    if !*ON.get_or_init(|| std::env::var("ZAKURO_LOG_DRAWS").is_ok_and(|value| value != "0")) {
        return;
    }
    for unit in 0..3 {
        let Some((addr, len)) = texture_range(registers, memory, unit) else { continue };
        let base = TEXTURE_UNIT_BASES[unit];
        let dimensions = registers[base + 1];
        let (height, width) = (dimensions & 0x7FF, (dimensions >> 16) & 0x7FF);
        let format = TextureFormat::from_raw(registers[if unit == 0 { base + 13 } else { base + 5 }]);
        let mut bytes = vec![0; len.min(256 << 10) as usize];
        memory.read(addr, &mut bytes);
        // the name ZAKURO_DUMP_TEXTURES gives the picture, so the two match
        let Some(id) = crate::pack::hash(crate::pack::Hashing::Rows, &bytes, format, width, height, &mut Vec::new()) else { continue };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (id, target).hash(&mut hasher);
        let key = hasher.finish();
        let now = Instant::now();
        let mut seen = SEEN.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let last = seen.get_or_insert_with(HashMap::new).insert(key, now);
        if last.is_some_and(|last| now - last < Duration::from_secs(2)) {
            continue;
        }
        let (x, y, w, h) = viewport;
        log::info!(
            "draw tex{unit} {id:016X} {width}x{height} {format:?} at 0x{addr:08X} into 0x{target:08X} {}x{} viewport {x},{y} {w}x{h}",
            buffer.0,
            buffer.1
        );
    }
}

/// the guest memory a texture unit reads, when it is on.
fn texture_range<M: GpuMemory>(registers: &[u32], memory: &M, unit: usize) -> Option<(u32, u32)> {
    if registers[REG_TEXTURE_CONFIG] & (1 << unit) == 0 {
        return None;
    }
    let base = TEXTURE_UNIT_BASES[unit];
    let dimensions = registers[base + 1];
    let (height, width) = (dimensions & 0x7FF, (dimensions >> 16) & 0x7FF);
    let format_register = if unit == 0 { base + 13 } else { base + 5 };
    let format = crate::texture::TextureFormat::from_raw(registers[format_register]);
    let address = loc_register(registers, base + 4);
    if width == 0 || height == 0 || address == 0 {
        return None;
    }
    let bits = width as u64 * height as u64 * format.bits_per_pixel() as u64;
    Some((memory.translate(address), bits.div_ceil(8) as u32))
}

/// where and how depth testing reads/writes, or None when disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compare {
    Never,
    Always,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

impl Compare {
    fn from_raw(value: u32) -> Compare {
        match value & 0x7 {
            0 => Compare::Never,
            1 => Compare::Always,
            2 => Compare::Equal,
            3 => Compare::NotEqual,
            4 => Compare::Less,
            5 => Compare::LessOrEqual,
            6 => Compare::Greater,
            _ => Compare::GreaterOrEqual,
        }
    }

    fn passes<T: PartialOrd>(self, value: T, reference: T) -> bool {
        match self {
            Compare::Never => false,
            Compare::Always => true,
            Compare::Equal => value == reference,
            Compare::NotEqual => value != reference,
            Compare::Less => value < reference,
            Compare::LessOrEqual => value <= reference,
            Compare::Greater => value > reference,
            Compare::GreaterOrEqual => value >= reference,
        }
    }
}

/// discards fragments whose alpha fails a comparison.
#[derive(Debug, Clone, Copy)]
struct AlphaTest {
    function: Compare,
    reference: u8,
}

impl AlphaTest {
    fn read(registers: &[u32]) -> Option<AlphaTest> {
        let config = registers[REG_ALPHA_TEST];
        (config & 1 != 0).then(|| AlphaTest {
            function: Compare::from_raw(config >> 4),
            reference: ((config >> 8) & 0xFF) as u8,
        })
    }

    fn passes(&self, alpha: u8) -> bool {
        self.function.passes(alpha, self.reference)
    }
}

/// what a stencil update does to the stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementSaturate,
    DecrementSaturate,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOp {
    fn from_raw(value: u32) -> StencilOp {
        match value & 0x7 {
            0 => StencilOp::Keep,
            1 => StencilOp::Zero,
            2 => StencilOp::Replace,
            3 => StencilOp::IncrementSaturate,
            4 => StencilOp::DecrementSaturate,
            5 => StencilOp::Invert,
            6 => StencilOp::IncrementWrap,
            _ => StencilOp::DecrementWrap,
        }
    }

    fn apply(self, value: u8, reference: u8) -> u8 {
        match self {
            StencilOp::Keep => value,
            StencilOp::Zero => 0,
            StencilOp::Replace => reference,
            StencilOp::IncrementSaturate => value.saturating_add(1),
            StencilOp::DecrementSaturate => value.saturating_sub(1),
            StencilOp::Invert => !value,
            StencilOp::IncrementWrap => value.wrapping_add(1),
            StencilOp::DecrementWrap => value.wrapping_sub(1),
        }
    }
}

/// GPUREG_STENCIL_TEST and GPUREG_STENCIL_OP.
#[derive(Debug, Clone, Copy)]
struct Stencil {
    function: Compare,
    reference: u8,
    /// bits the comparison looks at.
    compare_mask: u8,
    /// bits an update may change.
    write_mask: u8,
    /// applied when the stencil test fails, when it passes but the depth
    /// test fails, and when both pass.
    fail: StencilOp,
    depth_fail: StencilOp,
    pass: StencilOp,
}

impl Stencil {
    fn read(registers: &[u32]) -> Option<Stencil> {
        let test = registers[REG_STENCIL_TEST];
        if test & 1 == 0 {
            return None;
        }
        let op = registers[REG_STENCIL_OP];
        Some(Stencil {
            function: Compare::from_raw(test >> 4),
            write_mask: (test >> 8) as u8,
            reference: (test >> 16) as u8,
            compare_mask: (test >> 24) as u8,
            fail: StencilOp::from_raw(op),
            depth_fail: StencilOp::from_raw(op >> 4),
            pass: StencilOp::from_raw(op >> 8),
        })
    }

    /// the reference, not the stored value, is on the left of the comparison.
    fn passes(&self, stored: u8) -> bool {
        self.function
            .passes(self.reference & self.compare_mask, stored & self.compare_mask)
    }

    fn update(&self, stored: u8, op: StencilOp) -> u8 {
        let new = op.apply(stored, self.reference);
        (stored & !self.write_mask) | (new & self.write_mask)
    }
}

/// the depth/stencil buffer, and what a draw does with it.
struct DepthStencil {
    addr: u32,
    /// bytes per sample, 2 for D16, 3 for D24, 4 for D24S8, which keeps
    /// the stencil in the top byte.
    bytes: u32,
    test: Option<Compare>,
    write_depth: bool,
    stencil: Option<Stencil>,
    write_stencil: bool,
}

impl DepthStencil {
    fn read<M: GpuMemory>(registers: &[u32], memory: &M) -> Option<DepthStencil> {
        // GPUREG_DEPTH_COLOR_MASK, bit 0 enables the depth test, bits 4-6
        // pick the comparison, bit 12 enables writing depth back.
        let mask = registers[REG_DEPTH_COLOR_MASK];
        let writable = registers[REG_DEPTH_STENCIL_WRITE] != 0;
        let format = registers[REG_DEPTH_BUFFER_FORMAT] & 0x3;
        let test = (mask & 1 != 0).then(|| Compare::from_raw(mask >> 4));
        let write_depth = writable && mask & (1 << 12) != 0;
        let stencil = if format == 3 { Stencil::read(registers) } else { None };
        if test.is_none() && !write_depth && stencil.is_none() {
            return None;
        }
        Some(DepthStencil {
            addr: memory.translate(loc_register(registers, REG_DEPTH_BUFFER_ADDRESS)),
            bytes: match format {
                0 => 2,
                2 => 3,
                _ => 4,
            },
            test,
            write_depth,
            stencil,
            write_stencil: writable,
        })
    }

    /// the stored depth, as 0..1, and stencil.
    fn read_sample(&self, surface: &mut Band, index: u32) -> (f32, u8) {
        let mut raw = [0u8; 4];
        raw[..self.bytes as usize].copy_from_slice(surface.at(index));
        let value = u32::from_le_bytes(raw);
        match self.bytes {
            2 => (value as f32 / 65535.0, 0),
            _ => ((value & 0x00FF_FFFF) as f32 / 16_777_215.0, (value >> 24) as u8),
        }
    }

    fn write_depth(&self, surface: &mut Band, index: u32, depth: f32) {
        let sample = surface.at(index);
        if self.bytes == 2 {
            sample.copy_from_slice(&((depth * 65535.0) as u16).to_le_bytes());
        } else {
            // only the low three bytes, the stencil shares the word.
            sample[..3].copy_from_slice(&((depth * 16_777_215.0) as u32).to_le_bytes()[..3]);
        }
    }

    fn write_stencil(&self, surface: &mut Band, index: u32, value: u8) {
        if self.write_stencil && self.bytes == 4 {
            surface.at(index)[3] = value;
        }
    }
}

/// how clip-space depth becomes the value in the depth buffer, z/w scaled and
/// offset by GPUREG_DEPTHMAP_SCALE and _OFFSET.
#[derive(Debug, Clone, Copy)]
struct DepthMap {
    scale: f32,
    offset: f32,
    /// a w-buffer multiplies the mapped value by w.
    w_buffer: bool,
}

impl DepthMap {
    fn read(registers: &[u32]) -> DepthMap {
        DepthMap {
            scale: read_float24(registers, REG_VIEWPORT_DEPTH_RANGE),
            offset: read_float24(registers, REG_VIEWPORT_DEPTH_NEAR),
            w_buffer: registers[REG_DEPTHMAP_ENABLE] & 1 == 0,
        }
    }
}

/// the color buffer a draw renders into.
/// a copy of the rows of a buffer a draw can reach, so that its pixels do
/// not each go through guest memory. it goes back when the draw is done.
struct Surface {
    base: u32,
    bytes: u32,
    /// where the copy starts inside the buffer, in bytes.
    first: u32,
    data: Vec<u8>,
}

impl Surface {
    /// copies buffer rows rows of a tiled buffer, rounded out to whole
    /// rows of tiles, which is what keeps the copy one piece of memory.
    fn load<M: GpuMemory>(memory: &mut M, base: u32, bytes: u32, width: u32, height: u32, rows: Range<u32>) -> Surface {
        let tile_row = width / 8 * 64 * bytes;
        let total = height.div_ceil(8) * tile_row;
        // a width that is not a whole number of tiles spills into the
        // next row of tiles.
        let spill = if width.is_multiple_of(8) { 0 } else { 64 * bytes };
        let first = rows.start / 8 * tile_row;
        let end = (rows.end.div_ceil(8) * tile_row + spill).min(total).max(first);
        let mut data = vec![0u8; (end - first) as usize];
        memory.read(base + first, &mut data);
        Surface { base, bytes, first, data }
    }

    fn store<M: GpuMemory>(&self, memory: &mut M) {
        memory.write(self.base + self.first, &self.data);
    }

    fn whole(&mut self) -> Band<'_> {
        Band { bytes: self.bytes, first: self.first, data: &mut self.data }
    }

    /// the copy cut into one band per row of tiles, top of the buffer first.
    fn tile_rows(&mut self, width: u32) -> Vec<Band<'_>> {
        let tile_row = (width / 8 * 64 * self.bytes) as usize;
        let (bytes, first) = (self.bytes, self.first);
        self.data
            .chunks_mut(tile_row)
            .enumerate()
            .map(|(i, data)| Band { bytes, first: first + (i * tile_row) as u32, data })
            .collect()
    }
}

/// a stretch of a surface's rows, the part one thread fills.
struct Band<'a> {
    bytes: u32,
    /// where the stretch starts inside the buffer, in bytes.
    first: u32,
    data: &'a mut [u8],
}

impl Band<'_> {
    /// the bytes of the pixel at a tiled index.
    fn at(&mut self, index: u32) -> &mut [u8] {
        let offset = (index * self.bytes - self.first) as usize;
        &mut self.data[offset..offset + self.bytes as usize]
    }
}

struct ColorTarget {
    addr: u32,
    /// the pixels a draw may touch, in window coordinates, the viewport,
    /// cut to the buffer.
    left: i32,
    bottom: i32,
    right: i32,
    top: i32,
    /// the buffer's size.
    buffer_width: u32,
    buffer_height: u32,
    format: ColorFormat,
    /// which of red, green, blue and alpha the draw may change.
    write: [bool; 4],
}

/// what happened to the fragments of a draw, for tracing why a draw that
/// covers the screen leaves nothing on it.
#[derive(Debug, Default, Clone, Copy)]
struct FillStats {
    written: u32,
    depth_failed: u32,
    stencil_failed: u32,
    alpha_failed: u32,
}

impl FillStats {
    fn add(&mut self, other: FillStats) {
        self.written += other.written;
        self.depth_failed += other.depth_failed;
        self.stencil_failed += other.stencil_failed;
        self.alpha_failed += other.alpha_failed;
    }
}

/// everything about a draw that stays the same from triangle to triangle.
struct DrawState<'a> {
    target: ColorTarget,
    depth_map: DepthMap,
    depth_stencil: Option<DepthStencil>,
    tex_env: crate::tev::TexEnv,
    alpha_test: Option<AlphaTest>,
    blend: Option<crate::blend::Blend>,
    logic_op: Option<crate::blend::LogicOp>,
    textures: &'a [Option<BoundTexture>; 3],
    /// texture unit 0 divides its coordinates by w of coordinate set 0.
    texture0_projection: bool,
    /// texture unit 2 can read coordinate set 1 instead of its own.
    texture2_uses_coord1: bool,
    lighting: Option<Lighting>,
    tables: &'a Tables,
    /// texture 3, made up from its coordinates, and its tables.
    proctex: Option<crate::proctex::ProcTex<'a>>,
    proctex_tables: &'a crate::proctex::Tables,
    fog: Option<crate::fog::Fog<'a>>,
    fog_table: &'a crate::fog::Table,
}

#[cfg(feature = "vulkan")]
impl DrawState<'_> {
    /// the draw as the GPU takes it.
    fn hardware<'a>(&'a self, registers: &'a [u32], geometry: hardware::Geometry<'a>) -> hardware::Draw<'a> {
        let target = &self.target;
        hardware::Draw {
            registers,
            target: target.addr,
            format: target.format,
            width: target.buffer_width,
            height: target.buffer_height,
            scissor: [target.left, target.bottom, target.right, target.top],
            depth: self.depth_stencil.as_ref().map(|d| (d.addr, d.bytes)),
            depth_map: self.depth_map,
            geometry,
            textures: self.textures,
            lighting: self.lighting.as_ref(),
            tables: self.tables,
            proctex: self.proctex.is_some(),
            proctex_tables: self.proctex_tables,
            fog: self.fog.is_some(),
            fog_table: self.fog_table,
        }
    }
}

/// rasterizes one triangle, texturing and the combiners, then the alpha,
/// stencil and depth tests in the order the hardware runs them, then
/// blending and the color write.
fn fill_triangle(
    color_surface: Option<&mut Band>,
    mut depth_surface: Option<&mut Band>,
    rows: Range<i32>,
    [a, b, c]: [Screen; 3],
    state: &DrawState,
    stats: &mut FillStats,
) {
    let target = &state.target;
    let min_x = (a.x.min(b.x).min(c.x).floor() as i32).max(target.left);
    let max_x = (a.x.max(b.x).max(c.x).ceil() as i32).min(target.right);
    let min_y = (a.y.min(b.y).min(c.y).floor() as i32).max(target.bottom).max(rows.start);
    let max_y = (a.y.max(b.y).max(c.y).ceil() as i32).min(target.top).min(rows.end);
    if min_x >= max_x || min_y >= max_y {
        return;
    }

    // q and -q are the same orientation, but halfway between them is not,
    // so the quaternions all go to the side of the first one.
    let flip = |mut vertex: Screen| {
        let q = vertex.quaternion;
        if (0..4).map(|i| q[i] * a.quaternion[i]).sum::<f32>() < 0.0 {
            vertex.quaternion = q.map(|c| -c);
        }
        vertex
    };
    let (b, c) = (flip(b), flip(c));

    // wind every triangle the same way, so that "inside" is the positive
    // side of all three edges.
    let mut area = Edge::new(&a, &b).at(fixed(c.x), fixed(c.y));
    let (b, c) = if area < 0 {
        area = -area;
        (c, b)
    } else {
        (b, c)
    };
    // zero area means the triangle is degenerate.
    if area == 0 {
        return;
    }
    let inverse_area = 1.0 / area as f32;
    let edges = [Edge::new(&b, &c), Edge::new(&c, &a), Edge::new(&a, &b)];

    let bpp = target.format.bytes_per_pixel();
    let mut color_surface = color_surface;
    let writes_color = color_surface.is_some();
    let partial_write = writes_color && !target.write.iter().all(|&w| w);
    let any_texture = state.textures.iter().any(Option::is_some) || state.proctex.is_some();
    let mut pixel = [0u8; 4];

    for y in min_y..max_y {
        for x in min_x..max_x {
            // the pixel's center, in sixteenths
            let (px, py) = (x as i64 * 16 + 8, y as i64 * 16 + 8);

            // barycentric weights via the edge functions.
            let values = edges.map(|edge| edge.at(px, py));
            if !edges.iter().zip(values).all(|(edge, value)| edge.covers(value)) {
                continue;
            }
            let [w0, w1, w2] = values.map(|value| value as f32 * inverse_area);

            // perspective-correct interpolation, interpolate attribute/w and
            // 1/w linearly in screen space, then divide.
            let inv_w = w0 * a.inv_w + w1 * b.inv_w + w2 * c.inv_w;
            if inv_w.abs() < 1e-8 {
                continue;
            }
            let w = 1.0 / inv_w;

            let mut depth = (w0 * a.z + w1 * b.z + w2 * c.z) * state.depth_map.scale
                + state.depth_map.offset;
            if state.depth_map.w_buffer {
                depth *= w;
            }
            let depth = depth.clamp(0.0, 1.0);

            let row = target.buffer_height - 1 - y as u32;
            let index = crate::format::morton_offset(x as u32, row, target.buffer_width, 1);

            // with no stencil to update, a fragment that fails the depth
            // test is gone whatever its color, so skip the shading.
            let stored = state
                .depth_stencil
                .as_ref()
                .zip(depth_surface.as_deref_mut())
                .map(|(buffer, surface)| buffer.read_sample(surface, index));
            if let (Some(buffer), Some((stored_depth, _))) = (&state.depth_stencil, stored) {
                if buffer.stencil.is_none()
                    && buffer.test.is_some_and(|test| !test.passes(depth, stored_depth))
                {
                    stats.depth_failed += 1;
                    continue;
                }
            }

            let (aw, bw, cw) = (w0 * a.inv_w, w1 * b.inv_w, w2 * c.inv_w);
            let interpolate = |va: f32, vb: f32, vc: f32| (aw * va + bw * vb + cw * vc) * w;
            let color: Vec4 = std::array::from_fn(|i| {
                interpolate(a.color[i], b.color[i], c.color[i]).clamp(0.0, 1.0)
            });

            // sample the bound textures, then let the texture environment
            // decide what the fragment's color actually is, the samples and
            // the vertex color are only its inputs.
            let mut samples = [[0.0f32, 0.0, 0.0, 1.0]; 4];
            if any_texture {
                let mut uv0 = [interpolate(a.texcoords[0][0], b.texcoords[0][0], c.texcoords[0][0]), interpolate(a.texcoords[0][1], b.texcoords[0][1], c.texcoords[0][1])];
                if state.texture0_projection {
                    let tw = interpolate(a.texcoord0_w, b.texcoord0_w, c.texcoord0_w);
                    if tw != 0.0 {
                        uv0 = [uv0[0] / tw, uv0[1] / tw];
                    }
                }
                for (unit, texture) in state.textures.iter().enumerate() {
                    let Some(texture) = texture else { continue };
                    let set = match unit {
                        2 if state.texture2_uses_coord1 => 1,
                        unit => unit,
                    };
                    let [u, v] = if set == 0 {
                        uv0
                    } else {
                        [interpolate(a.texcoords[set][0], b.texcoords[set][0], c.texcoords[set][0]), interpolate(a.texcoords[set][1], b.texcoords[set][1], c.texcoords[set][1])]
                    };
                    samples[unit] = texture.sample(u, v);
                }
                if let Some(proctex) = &state.proctex {
                    let set = proctex.coordinates;
                    let [u, v] = if set == 0 {
                        uv0
                    } else {
                        [interpolate(a.texcoords[set][0], b.texcoords[set][0], c.texcoords[set][0]), interpolate(a.texcoords[set][1], b.texcoords[set][1], c.texcoords[set][1])]
                    };
                    samples[3] = proctex.sample(u, v);
                }
            }

            let fragment = state.lighting.as_ref().map(|lighting| {
                let quaternion = std::array::from_fn(|i| {
                    interpolate(a.quaternion[i], b.quaternion[i], c.quaternion[i])
                });
                let view = std::array::from_fn(|i| interpolate(a.view[i], b.view[i], c.view[i]));
                lighting.shade(state.tables, quaternion, view, &samples)
            });
            let combined = state.tex_env.apply(color, samples, fragment);
            let mut rgba = combined.map(|c| (c * 255.0) as u8);

            if let Some(test) = state.alpha_test {
                if !test.passes(rgba[3]) {
                    stats.alpha_failed += 1;
                    continue;
                }
            }
            if let Some(fog) = &state.fog {
                fog.apply(&mut rgba, depth);
            }

            if let (Some(buffer), Some((stored_depth, stored_stencil)), Some(surface)) =
                (&state.depth_stencil, stored, depth_surface.as_deref_mut())
            {
                if let Some(stencil) = &buffer.stencil {
                    if !stencil.passes(stored_stencil) {
                        buffer.write_stencil(surface, index, stencil.update(stored_stencil, stencil.fail));
                        stats.stencil_failed += 1;
                        continue;
                    }
                }
                let depth_passes = buffer.test.is_none_or(|test| test.passes(depth, stored_depth));
                if let Some(stencil) = &buffer.stencil {
                    let op = if depth_passes { stencil.pass } else { stencil.depth_fail };
                    buffer.write_stencil(surface, index, stencil.update(stored_stencil, op));
                }
                if !depth_passes {
                    stats.depth_failed += 1;
                    continue;
                }
                if buffer.write_depth {
                    buffer.write_depth(surface, index, depth);
                }
            }

            let Some(surface) = color_surface.as_deref_mut() else { continue };

            // combine with what is already in the buffer, the way the output
            // merger is configured to, and keep the channels the draw may not
            // change.
            if state.blend.is_some() || state.logic_op.is_some() || partial_write {
                pixel[..bpp].copy_from_slice(surface.at(index));
                let existing = target.format.decode(&pixel[..bpp]);
                if let Some(blend) = &state.blend {
                    let blended = blend.apply(
                        rgba.map(|c| c as f32 / 255.0),
                        existing.map(|c| c as f32 / 255.0),
                    );
                    rgba = blended.map(|c| (c * 255.0) as u8);
                } else if let Some(op) = state.logic_op {
                    rgba = std::array::from_fn(|channel| op.apply(rgba[channel], existing[channel]));
                }
                for channel in 0..4 {
                    if !target.write[channel] {
                        rgba[channel] = existing[channel];
                    }
                }
            }

            target.format.encode(rgba, &mut pixel[..bpp]);
            surface.at(index).copy_from_slice(&pixel[..bpp]);
            stats.written += 1;
        }
    }
}

/// a row of tiles, its color and depth and the window rows it holds.
type Piece<'a> = (Option<Band<'a>>, Option<Band<'a>>, Range<i32>);

/// pixels a draw has to cover before it is worth splitting over threads.
const PARALLEL_PIXELS: f32 = 4096.0;

/// fills the triangles on the copies of the buffers, rows being the window
/// rows they can reach. a big draw is cut into rows of tiles that threads
/// take on, which never touch the same pixel.
fn fill(
    color: &mut Option<Surface>,
    depth: &mut Option<Surface>,
    triangles: &[[Screen; 3]],
    rows: Range<i32>,
    state: &DrawState,
    stats: &mut FillStats,
) {
    let target = &state.target;
    let area: f32 = triangles
        .iter()
        .map(|t| {
            let (x, y) = (t.map(|v| v.x), t.map(|v| v.y));
            let span = |c: [f32; 3]| c.iter().copied().fold(f32::MIN, f32::max) - c.iter().copied().fold(f32::MAX, f32::min);
            span(x) * span(y)
        })
        .sum();
    // a width that is not a whole number of tiles spills across rows of
    // tiles, so those stay on one thread.
    if area < PARALLEL_PIXELS || !target.buffer_width.is_multiple_of(8) {
        let mut color_band = color.as_mut().map(Surface::whole);
        let mut depth_band = depth.as_mut().map(Surface::whole);
        for &triangle in triangles {
            fill_triangle(color_band.as_mut(), depth_band.as_mut(), rows.clone(), triangle, state, stats);
        }
        return;
    }

    let width = target.buffer_width;
    let height = target.buffer_height as i32;
    let first_tile_row = color.as_ref().or(depth.as_ref()).map_or(0, |s| s.first / (width / 8 * 64 * s.bytes)) as i32;
    let colors = color.as_mut().map(|surface| surface.tile_rows(width));
    let depths = depth.as_mut().map(|surface| surface.tile_rows(width));
    let count = colors.as_ref().or(depths.as_ref()).map_or(0, Vec::len);
    let mut colors = colors.map(|bands| bands.into_iter().map(Some).collect::<Vec<_>>());
    let mut depths = depths.map(|bands| bands.into_iter().map(Some).collect::<Vec<_>>());
    let pieces: Vec<Piece> = (0..count)
        .map(|i| {
            // tile row t holds buffer rows 8t to 8t+7, window rows counting
            // from the other end
            let t = first_tile_row + i as i32;
            let band_rows = (height - 8 * t - 8).max(rows.start)..(height - 8 * t).min(rows.end);
            let color_band = colors.as_mut().and_then(|bands| bands[i].take());
            let depth_band = depths.as_mut().and_then(|bands| bands[i].take());
            (color_band, depth_band, band_rows)
        })
        .filter(|piece| !piece.2.is_empty())
        .collect();
    let fill_piece = |(mut color_band, mut depth_band, band_rows): Piece| {
        let mut stats = FillStats::default();
        for &triangle in triangles {
            fill_triangle(color_band.as_mut(), depth_band.as_mut(), band_rows.clone(), triangle, state, &mut stats);
        }
        stats
    };
    #[cfg(target_os = "horizon")]
    let total = pieces.into_iter().map(fill_piece).fold(FillStats::default(), |mut all, stats| {
        all.add(stats);
        all
    });
    #[cfg(not(target_os = "horizon"))]
    let total = pieces.into_par_iter().map(fill_piece).reduce(FillStats::default, |mut all, stats| {
        all.add(stats);
        all
    });
    stats.add(total);
}

/// a screen coordinate in sixteenths of a pixel, the steps the PICA places
/// vertices on.
fn fixed(c: f32) -> i64 {
    (c * 16.0).round() as i64
}

/// one edge of a triangle, as a function that is zero on the edge and positive
/// on the triangle's side of it, in whole sixteenths so that a pixel exactly on
/// the edge is known to be.
#[derive(Debug, Clone, Copy)]
struct Edge {
    x: i64,
    y: i64,
    dx: i64,
    dy: i64,
    owns_ties: bool,
}

impl Edge {
    fn new(from: &Screen, to: &Screen) -> Edge {
        let (x, y) = (fixed(from.x), fixed(from.y));
        let (dx, dy) = (fixed(to.x) - x, fixed(to.y) - y);
        // a pixel centered on an edge goes to the triangle on its right, or
        // above it when the edge is flat, as on the PICA
        Edge { x, y, dx, dy, owns_ties: dy < 0 || (dy == 0 && dx > 0) }
    }

    /// the function at a point in sixteenths.
    #[inline]
    fn at(&self, px: i64, py: i64) -> i64 {
        self.dx * (py - self.y) - self.dy * (px - self.x)
    }

    #[inline]
    fn covers(&self, value: i64) -> bool {
        value > 0 || (value == 0 && self.owns_ties)
    }
}

/// executes a draw call, fetches vertices, shades them, assembles triangles
/// and fills them. Returns the number of vertices processed.
/// a draw whose vertices go through the vertex shader on a helper thread, and
/// the registers it is drawn with, copied, as the next draws change them.
struct Pending {
    registers: Vec<u32>,
    /// the vertices shaded once each, found in the cache or from a helper.
    shaded: std::sync::mpsc::Receiver<std::sync::Arc<Vec<Vertex>>>,
    order: Option<Vec<usize>>,
    /// where a helper's vertices go in the caches, when they did not come from them.
    cache_key: Option<u64>,
    signature: Option<u64>,
}

/// whether draws shade ahead, ZAKURO_SHADE_AHEAD=0 for one at a time.
fn shade_ahead_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZAKURO_SHADE_AHEAD").is_ok_and(|value| value != "0"))
}

/// whether a draw has anything of its own to do before its vertices are
/// shaded, the portraits and text of the bottom screen and the subtitles,
/// which look at them.
fn looks_at_vertices(registers: &[u32]) -> bool {
    let top = read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0 && read_float24(registers, REG_VIEWPORT_HEIGHT) == 200.0;
    if top {
        is_subtitle(registers)
    } else {
        is_text_buffer(registers) || on_bottom(registers) || is_bottom(registers)
    }
}

/// draws the draw that waits for its vertices, once they are shaded.
pub(crate) fn finish_pending<M: GpuMemory>(memory: &mut M, resources: &mut Resources) {
    while finish_oldest(memory, resources) {}
}

/// draws the oldest of the draws that wait for their vertices. whether there was one.
fn finish_oldest<M: GpuMemory>(memory: &mut M, resources: &mut Resources) -> bool {
    let Some(pending) = resources.pending.pop_front() else { return false };
    let waited = std::time::Instant::now();
    let shaded = pending.shaded.recv().unwrap_or_default();
    crate::stats::SHADE_WAIT.fetch_add(waited.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    if let Some(key) = pending.cache_key {
        if resources.vcache.len() >= 40 {
            resources.vcache_old = std::mem::take(&mut resources.vcache);
        }
        resources.vcache.insert(key, shaded.clone());
    }
    if let (Some(signature), false) = (pending.signature, shaded.is_empty()) {
        let last = shaded.len() - 1;
        let samples = [shaded[0], shaded[shaded.len() / 2], shaded[last]];
        if resources.vreuse.len() > 256 {
            resources.vreuse.clear();
        }
        resources.vreuse.insert(signature, Reuse { vertices: shaded.clone(), samples, age: 0 });
    }
    let order = pending.order.as_deref();
    rasterize_now(&pending.registers, memory, resources, Vertices::ShadedOrdered { shaded: &shaded, order });
    true
}

/// how many vertices a shading job ahead takes at most, ZAKURO_SHADE_CHUNK.
fn shade_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("ZAKURO_SHADE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256).max(16))
}

/// how many draws may wait for their vertices at once, ZAKURO_SHADE_DEPTH.
fn shade_depth() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("ZAKURO_SHADE_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1))
}

pub fn draw<M: GpuMemory>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    fixed_attributes: &[Vec4; 16],
    memory: &mut M,
    resources: &mut Resources,
    indexed: bool,
    ahead: bool,
) -> u32 {
    let _timer = crate::stats::time(&crate::stats::DRAWING);
    crate::stats::DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let vertex_count = registers[REG_VERTEX_COUNT];
    let first_vertex = registers[REG_VERTEX_OFFSET];
    if vertex_count == 0 || vertex_count > 0x1_0000 {
        return 0;
    }
    {
        // ZAKURO_DRAW_LOG: what the draws are made of, every 2000 draws
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("ZAKURO_DRAW_LOG").is_some()) {
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static N: AtomicU64 = AtomicU64::new(0);
            static TOTAL: AtomicU64 = AtomicU64::new(0);
            static BIG: AtomicU64 = AtomicU64::new(0);
            static BIG_TOTAL: AtomicU64 = AtomicU64::new(0);
            static TINY: AtomicU64 = AtomicU64::new(0);
            TOTAL.fetch_add(vertex_count as u64, Relaxed);
            if vertex_count >= 500 {
                BIG.fetch_add(1, Relaxed);
                BIG_TOTAL.fetch_add(vertex_count as u64, Relaxed);
            }
            if vertex_count <= 12 {
                TINY.fetch_add(1, Relaxed);
            }
            if N.fetch_add(1, Relaxed) % 2000 == 1999 {
                log::info!(
                    target: "zakuro_gpu::diag",
                    "2000 draws: {} vertices, {} draws of 500+ vertices hold {} of them, {} draws of 12 or fewer",
                    TOTAL.swap(0, Relaxed), BIG.swap(0, Relaxed), BIG_TOTAL.swap(0, Relaxed), TINY.swap(0, Relaxed)
                );
            }
        }
    }
    note_title_draw(registers, &mut resources.portraits);
    let on_title = is_on_title(&resources.portraits);
    let capturing_title = is_capturing_title(&resources.portraits);
    let title_ready = resources.portraits.title_sizzle_active
        || resources.portraits.title_frames < 16
        || resources.portraits.title_captured;
    let bottom = is_bottom(registers);
    if bottom {
        resources.portraits.bottom_seen = true;
        if on_bottom(registers) {
            resources.portraits.bottom_tex_seen = true;
            resources.portraits.bottom_tex_missed = 0;
            resources.portraits.bottom_blank = false;
        }
    }
    if bottom
        && (resources.portraits.in_play || capturing_title)
        && !resources.portraits.fuse_active
        && !resources.portraits.cauldron_active
        && !resources.portraits.chapter_active
        && !resources.portraits.mode_active
        && !resources.portraits.solo_active
        && !resources.portraits.file_active
        && !resources.portraits.weapon_active
        && skip_bottom_in_play(
            registers,
            resources.portraits.in_battle || resources.portraits.active_battle,
            on_title || resources.portraits.title_sizzle_active,
            title_ready,
            vertex_count,
        )
    {
        return vertex_count;
    }

    let attribute_base = loc_register(registers, REG_ATTRIBUTE_BASE);

    // resolve each of the vertex_count draw indices to an actual vertex
    // array index, sequential for DrawArrays, looked up in the index buffer
    // for DrawElements.
    let index_config = registers[REG_INDEX_ARRAY];
    let index_short = index_config & 0x8000_0000 != 0;
    let index_base = memory.translate(attribute_base + (index_config & 0x0FFF_FFFF));


    // an index buffer names most vertices several times, fetch and shade
    // each of them once.
    let mut scratch = std::mem::take(&mut resources.scratch);
    if indexed {
        // the whole index buffer at once, a read per index costs far more
        let size = if index_short { 2 } else { 1 };
        let len = (vertex_count * size) as usize;
        let bytes = &mut scratch.bytes;
        bytes.clear();
        bytes.reserve(len);
        match memory.slice(index_base, len) {
            Some(slice) => bytes.extend_from_slice(slice),
            None => {
                bytes.resize(len, 0);
                memory.read(index_base, bytes);
            }
        }
    }
    let plan = scratch.plans.update(registers, fixed_attributes);
    log::trace!(
        "array draw: {vertex_count} vertices from 0x{attribute_base:08X}, indexed {indexed}, formats \
         0x{:08X}{:08X}, arrays {:?}",
        registers[REG_ATTRIBUTE_FORMAT_HIGH],
        registers[REG_ATTRIBUTE_FORMAT_LOW],
        plan.loaders.iter().map(|l| (l.offset, l.stride, l.size)).collect::<Vec<_>>(),
    );
    // the GPU decodes the arrays itself when it shades the draw, from the
    // first vertex the draw uses to the last
    #[cfg(feature = "vulkan")]
    if resources.hardware.as_ref().is_some_and(|hardware| hardware.shades()) && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2 {
        let (first, span) = if indexed {
            // over the index buffer's own bytes or halfwords, which the
            // vector units compare many at a time, they have no unsigned
            // comparison of whole words
            let (low, high) = if index_short {
                let halves = scratch.bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b));
                let (low, high) = halves.fold((u16::MAX, 0), |(low, high), index| (low.min(index), high.max(index)));
                (low as u32, high as u32)
            } else {
                let (low, high) = scratch.bytes.iter().fold((u8::MAX, 0), |(low, high), &index| (low.min(index), high.max(index)));
                (low as u32, high as u32)
            };
            (low, high - low + 1)
        } else {
            (first_vertex, vertex_count)
        };
        static NO_RAW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_raw = *NO_RAW.get_or_init(|| std::env::var_os("ZAKURO_NO_RAW_INPUTS").is_some());
        let allow_bottom = !is_bottom(registers)
            || (!resources.portraits.in_play
                && !(on_bottom(registers) && is_portrait(registers))
                && !(on_bottom(registers) && vertex_count <= 16 && battle_detection()));
        if !no_raw && allow_bottom && raw_inputs(plan, memory, attribute_base, first, span, &mut scratch.raw) {
            scratch.relative.clear();
            if indexed {
                // straight from the index buffer, the draw needs its indices
                // as words only when the CPU decodes its arrays
                if index_short {
                    let halves = scratch.bytes.as_chunks::<2>().0.iter();
                    scratch.relative.extend(halves.map(|b| u16::from_le_bytes(*b) as u32 - first));
                } else {
                    scratch.relative.extend(scratch.bytes.iter().map(|&b| b as u32 - first));
                }
            } else {
                scratch.relative.extend(0..vertex_count);
            }
            let vertices = Vertices::Raw { vertex_shader, raw: &scratch.raw, vertices: &scratch.relative };
            if rasterize(registers, memory, resources, vertices).is_some() {
                resources.scratch = scratch;
                return vertex_count;
            }
        }
    }
    if indexed {
        let bytes = std::mem::take(&mut scratch.bytes);
        if index_short {
            let chunks = bytes.as_chunks::<2>().0;
            scratch.unique_indices(chunks.len(), |i| u16::from_le_bytes(chunks[i]) as u32);
        } else {
            scratch.unique_indices(bytes.len(), |i| bytes[i] as u32);
        }
        scratch.bytes = bytes;
    } else {
        scratch.unique.clear();
        scratch.unique.extend((0..vertex_count).map(|i| first_vertex + i));
    }
    {
        static ON2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON2.get_or_init(|| std::env::var_os("ZAKURO_DRAW_LOG").is_some()) {
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static UNIQUE: AtomicU64 = AtomicU64::new(0);
            static DRAWS: AtomicU64 = AtomicU64::new(0);
            UNIQUE.fetch_add(scratch.unique.len() as u64, Relaxed);
            if DRAWS.fetch_add(1, Relaxed) % 2000 == 1999 {
                log::info!(target: "zakuro_gpu::diag", "2000 draws: {} unique vertices shaded", UNIQUE.swap(0, Relaxed));
            }
        }
    }
    let plan = scratch.plans.current();
    // a draw that barely moved since it was shaded is not even gathered:
    // three of its vertices say so, and the rest are the last frame's
    #[cfg(feature = "vulkan")]
    let gpu_shades = resources.hardware.as_ref().is_some_and(|hardware| hardware.shades());
    #[cfg(not(feature = "vulkan"))]
    let gpu_shades = false;
    if scratch.unique.len() >= 128
        && vcache_enabled()
        && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2
        && !looks_at_vertices(registers)
        && !gpu_shades
    {
        let n = scratch.unique.len();
        let mut picks = [plan.template; 3];
        for (input, at) in picks.iter_mut().zip([0, n / 2, n - 1]) {
            let vertex_index = scratch.unique[at];
            for loader in &plan.loaders {
                let size = loader.size as usize;
                let vertex = memory.translate(attribute_base + loader.offset + vertex_index * loader.stride);
                match memory.slice(vertex, size) {
                    Some(bytes) => loader.place(bytes, input),
                    None => {
                        let mut bytes = [0u8; 256];
                        memory.read(vertex, &mut bytes[..size]);
                        loader.place(&bytes[..size], input);
                    }
                }
            }
        }
        let signature = draw_signature_of(vertex_shader, n, &picks);
        if let Some(vertices) = try_reuse(resources, registers, vertex_shader, signature, &picks) {
            let order = indexed.then_some(&scratch.order[..]);
            rasterize(registers, memory, resources, Vertices::ShadedOrdered { shaded: &vertices, order });
            resources.scratch = scratch;
            return vertex_count;
        }
    }
    scratch.inputs.clear();
    scratch.inputs.resize(scratch.unique.len(), plan.template);
    let max_index = scratch.unique.iter().copied().max().unwrap_or(0);
    for loader in &plan.loaders {
        let array_vaddr = memory.translate(attribute_base + loader.offset);
        let array_bytes = (max_index as usize + 1) * loader.stride as usize;
        if let Some(slice) = memory.slice(array_vaddr, array_bytes) {
            let size = loader.size as usize;
            let stride = loader.stride as usize;
            for (i, &vertex_index) in scratch.unique.iter().enumerate() {
                let at = vertex_index as usize * stride;
                if at + size <= slice.len() {
                    loader.place(&slice[at..at + size], &mut scratch.inputs[i]);
                }
            }
        } else {
            for (i, &vertex_index) in scratch.unique.iter().enumerate() {
                let vertex = memory.translate(attribute_base + loader.offset + vertex_index * loader.stride);
                let size = loader.size as usize;
                match memory.slice(vertex, size) {
                    Some(bytes) => loader.place(bytes, &mut scratch.inputs[i]),
                    None => {
                        let mut bytes = [0u8; 256];
                        memory.read(vertex, &mut bytes[..size]);
                        loader.place(&bytes[..size], &mut scratch.inputs[i]);
                    }
                }
            }
        }
    }
    let order = indexed.then_some(&scratch.order[..]);
    #[cfg(feature = "vulkan")]
    let ahead = ahead
        && shade_ahead_wanted()
        && resources.hardware.as_ref().is_some_and(|hardware| !hardware.shades())
        && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2
        && !looks_at_vertices(registers);
    #[cfg(not(feature = "vulkan"))]
    let ahead = false;
    if ahead {
        // the vertices go to a helper with what shades them, as the unit is
        // by now, while the draw before them is drawn here
        let inputs = &scratch.inputs;
        // the exact cache first, as rasterize_now does
        let signature = (vcache_enabled() && inputs.len() >= 128).then(|| draw_signature(vertex_shader, inputs));
        let cache_key = signature
            .filter(|&signature| vcache_worth(&mut resources.vcache_score, signature))
            .map(|_| shading_key(registers, vertex_shader, inputs));
        let found = cache_key.and_then(|key| resources.vcache.get(&key).or_else(|| resources.vcache_old.get(&key))).cloned();
        if let (Some(signature), Some(_)) = (signature, cache_key) {
            vcache_note(&mut resources.vcache_score, signature, found.is_some());
        }
        let (copied, order) = (registers.to_vec(), order.map(<[usize]>::to_vec));
        let (send, shaded) = std::sync::mpsc::channel();
        let pending = match found {
            Some(vertices) => {
                let _ = send.send(vertices);
                Pending { registers: copied, shaded, order, cache_key: None, signature: None }
            }
            None => {
                // in pieces, so the helpers share a big draw between them
                let unit = std::sync::Arc::new(vertex_shader.snapshot());
                let inputs = std::sync::Arc::new(inputs.clone());
                let shading_registers = std::sync::Arc::new(copied.clone());
                let chunk = shade_chunk();
                let pieces = inputs.len().div_ceil(chunk).max(1);
                let results = std::sync::Arc::new(std::sync::Mutex::new(vec![Vec::new(); pieces]));
                let left = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(pieces));
                for piece in 0..pieces {
                    let (unit, inputs, shading_registers) = (unit.clone(), inputs.clone(), shading_registers.clone());
                    let (results, left, send) = (results.clone(), left.clone(), send.clone());
                    crate::shade_queue::submit(Box::new(move || {
                        let range = piece * chunk..((piece + 1) * chunk).min(inputs.len());
                        crate::shade_queue::IN_JOB.with(|flag| flag.set(true));
                        let (shaded, _) = process_vertices(&shading_registers, &unit, &unit, &inputs[range], None);
                        crate::shade_queue::IN_JOB.with(|flag| flag.set(false));
                        results.lock().unwrap_or_else(|p| p.into_inner())[piece] = shaded;
                        if left.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                            let parts = std::mem::take(&mut *results.lock().unwrap_or_else(|p| p.into_inner()));
                            let _ = send.send(std::sync::Arc::new(parts.concat()));
                        }
                    }));
                }
                Pending { registers: copied, shaded, order, cache_key, signature }
            }
        };
        resources.scratch = scratch;
        // the draws before this one are drawn while it and the one before
        // it are shaded, up to as many as may wait
        resources.pending.push_back(pending);
        while resources.pending.len() > shade_depth() {
            finish_oldest(memory, resources);
        }
        return vertex_count;
    }
    let vertices = Vertices::Unshaded { vertex_shader, geometry_shader, inputs: &scratch.inputs, order };
    rasterize(registers, memory, resources, vertices);
    resources.scratch = scratch;
    vertex_count
}

/// draws vertices a title sent one attribute at a time through the fixed
/// attribute registers ("immediate mode") instead of from arrays in memory.
pub fn draw_immediate<M: GpuMemory>(
    registers: &[u32],
    vertex_shader: &ShaderUnit,
    geometry_shader: &ShaderUnit,
    memory: &mut M,
    resources: &mut Resources,
    vertices: &[[Vec4; 16]],
) -> u32 {
    note_title_draw(registers, &mut resources.portraits);
    let on_title = is_on_title(&resources.portraits);
    let capturing_title = is_capturing_title(&resources.portraits);
    let title_ready = resources.portraits.title_sizzle_active
        || resources.portraits.title_frames < 16
        || resources.portraits.title_captured;
    let bottom = is_bottom(registers);
    if bottom {
        resources.portraits.bottom_seen = true;
        if on_bottom(registers) {
            resources.portraits.bottom_tex_seen = true;
            resources.portraits.bottom_tex_missed = 0;
            resources.portraits.bottom_blank = false;
        }
    }
    if bottom
        && (resources.portraits.in_play || capturing_title)
        && !resources.portraits.fuse_active
        && !resources.portraits.cauldron_active
        && !resources.portraits.chapter_active
        && !resources.portraits.mode_active
        && !resources.portraits.solo_active
        && !resources.portraits.file_active
        && !resources.portraits.weapon_active
        && skip_bottom_in_play(
            registers,
            resources.portraits.in_battle || resources.portraits.active_battle,
            on_title || resources.portraits.title_sizzle_active,
            title_ready,
            vertices.len() as u32,
        )
    {
        return vertices.len() as u32;
    }
    // immediate mode sizes its vertices by GPUREG_VSH_NUM_ATTR.
    let count = ((registers[REG_VS_ATTRIBUTE_COUNT] & 0xF) + 1) as usize;
    log::trace!("immediate draw: {} vertices of {count} attributes", vertices.len());
    let inputs: Vec<_> = vertices
        .iter()
        .map(|attributes| map_inputs(registers, REG_VS_BLOCK, &attributes[..count]))
        .collect();
    let vertices = Vertices::Unshaded { vertex_shader, geometry_shader, inputs: &inputs, order: None };
    rasterize(registers, memory, resources, vertices).unwrap_or(0)
}

/// puts a draw's arrays for the GPU to decode in raw, from the first
/// vertex the draw uses on. false when they cost more to hand over than to
/// decode on the CPU or do not lie in one piece of memory.
#[cfg(feature = "vulkan")]
fn raw_inputs<M: GpuMemory>(
    plan: &InputPlan,
    memory: &mut M,
    base: u32,
    first: u32,
    span: u32,
    raw: &mut hardware::RawInputs,
) -> bool {
    // past this the vertices between the ones a draw uses cost too much
    const LIMIT: u64 = 4 << 20;
    raw.arrays.clear();
    raw.fields = [None; shader::INPUT_REGISTERS];
    raw.defaults = plan.template;
    let mut total = 0u64;
    for loader in &plan.loaders {
        let start = base.wrapping_add(loader.offset).wrapping_add(first.wrapping_mul(loader.stride));
        let len = (span - 1) * loader.stride + loader.size;
        total += len as u64;
        if total > LIMIT {
            return false;
        }
        // a vertex's address is translated on its own on the CPU, which
        // comes to the same where the array is in one piece
        let addr = memory.translate(start);
        if memory.slice(addr, len as usize).is_none() {
            return false;
        }
        for field in &loader.fields {
            // a field fills its components over (0, 0, 0, 1)
            raw.defaults[field.register] = [0.0, 0.0, 0.0, 1.0];
            raw.fields[field.register] = Some(hardware::RawField {
                array: raw.arrays.len(),
                stride: loader.stride,
                offset: field.offset,
                ty: field.ty,
                count: field.count,
            });
        }
        raw.arrays.push((addr, len));
    }
    true
}

/// the vertices of each triangle out of count of them, as assemble puts
/// them, each the vertex it names, into indices.
fn assemble_into(topology: u32, count: usize, vertex: impl Fn(usize) -> u32, indices: &mut Vec<u32>) {
    indices.clear();
    // a strip or a fan a triangle at a time, a flattened iterator has no
    // length to reserve by and pushes a vertex at a time
    if matches!(topology, 1 | 2) {
        indices.reserve(count.saturating_sub(2) * 3);
    }
    match topology {
        1 => {
            for i in 2..count {
                let (a, b) = if i % 2 == 0 { (i - 2, i - 1) } else { (i - 1, i - 2) };
                indices.extend([vertex(a), vertex(b), vertex(i)]);
            }
        }
        2 => {
            for i in 2..count {
                indices.extend([vertex(0), vertex(i - 1), vertex(i)]);
            }
        }
        _ => indices.extend((0..count / 3 * 3).map(vertex)),
    }
}

/// the triangles of a draw of count vertices as the GPU takes them, three
/// vertices each, numbered as the raw arrays number them, or by their place
/// among the vertices order gives, or by their place in the draw. a list of
/// raw vertices is its own triangles, which go as they are rather than
/// copied.
#[cfg(feature = "vulkan")]
fn triangle_indices<'a>(
    topology: u32,
    count: usize,
    raw: Option<&'a [u32]>,
    order: Option<&[usize]>,
    assembled: &'a mut Vec<u32>,
) -> &'a [u32] {
    match raw {
        Some(vertices) if !matches!(topology, 1 | 2) => &vertices[..count / 3 * 3],
        Some(vertices) => {
            assemble_into(topology, count, |i| vertices[i], assembled);
            assembled
        }
        None => {
            assemble_into(topology, count, |i| order.map_or(i, |order| order[i]) as u32, assembled);
            assembled
        }
    }
}

/// the vertices of each triangle out of count of them, for a list, a strip
/// (1) or a fan (2).
fn assemble(topology: u32, count: usize) -> impl Iterator<Item = (usize, usize, usize)> {
    let triangles = match topology {
        1 | 2 => count.saturating_sub(2),
        _ => count / 3,
    };
    (0..triangles).map(move |i| match topology {
        1 => {
            if (i + 2) % 2 == 0 {
                (i, i + 1, i + 2)
            } else {
                (i + 1, i, i + 2)
            }
        }
        2 => (0, i + 1, i + 2),
        _ => (i * 3, i * 3 + 1, i * 3 + 2),
    })
}

/// assembles shaded vertices into triangles the way the primitive configuration
/// says, and fills them with the current back-end state.
/// the triangles drawn, none for arrays the GPU could not decode, which the
/// CPU decodes then.
/// draws vertices, after the draw before them, which may still wait for its.
fn rasterize<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, vertices: Vertices) -> Option<u32> {
    finish_pending(memory, resources);
    rasterize_now(registers, memory, resources, vertices)
}

fn rasterize_now<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, vertices: Vertices) -> Option<u32> {
    if portraits_on() && !resources.portraits.adding {
        if read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0 && read_float24(registers, REG_VIEWPORT_HEIGHT) == 200.0 {
            note_top_buffer(registers, resources);
            if resources.portraits.fuse_active
                && !resources.portraits.menu_override
                && needs_fuse_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_fuse_top(registers, shaded) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if !resources.portraits.in_battle
                && !resources.portraits.title_sky_active
                && !resources.portraits.fuse_active
                && is_cauldron_mural_paper(registers, vertices.len())
            {
                resources.portraits.cauldron_active = true;
                resources.portraits.cauldron_hud_active = true;
            }
            if resources.portraits.cauldron_hud_active
                && !resources.portraits.menu_override
                && needs_cauldron_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_cauldron_top(registers, memory, resources, shaded);
                }
            }
            if resources.portraits.chapter_active
                && !resources.portraits.menu_override
                && needs_chapter_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_chapter_top(registers, shaded, &mut resources.portraits) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.mode_active
                && !resources.portraits.menu_override
                && needs_mode_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_mode_top(registers, shaded, &mut resources.portraits) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.solo_active
                && !resources.portraits.menu_override
                && needs_solo_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_solo_top(registers, shaded, &mut resources.portraits) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.file_active
                && !resources.portraits.menu_override
                && needs_file_top_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_file_top(registers, shaded, &mut resources.portraits) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.weapon_active
                && !resources.portraits.menu_override
                && needs_weapon_top_adjust(registers, resources.portraits.weapon_sub, vertices.len())
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_weapon_top(registers, memory, resources, shaded);
                }
            } else if !resources.portraits.in_battle
                && !resources.portraits.frontend_menu_seen
                && vertices.len() == 6
                && registers[REG_TEXTURE_CONFIG] & 1 != 0
                && registers[TEXTURE_UNIT_BASES[0] + 1] == (256 << 16 | 128)
                && TextureFormat::from_raw(registers[TEXTURE_UNIT_BASES[0] + 13]) == TextureFormat::La4
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices.clone()) {
                    let (mut l, mut u, mut r, mut d) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                    for v in &shaded {
                        let w = v.clip[3];
                        let bx = (v.clip[0] / w * 0.5 + 0.5) * 240.0;
                        let by = (v.clip[1] / w * 0.5 + 0.5) * 400.0;
                        let (sx, sy) = (400.0 - by, 240.0 - bx);
                        l = l.min(sx);
                        u = u.min(sy);
                        r = r.max(sx);
                        d = d.max(sy);
                    }
                    if l >= 378.0 && r <= 402.0 && u >= -2.0 && d <= 16.0 {
                        resources.portraits.frontend_menu_seen = true;
                    }
                }
            }
            if !resources.portraits.in_battle && is_title_sky(registers) {
                resources.portraits.title_sky_seen = true;
                resources.portraits.title_seen = true;
                resources.portraits.top = Some([
                    registers[REG_COLOR_BUFFER_ADDRESS],
                    registers[REG_COLOR_BUFFER_FORMAT],
                    registers[REG_FRAMEBUFFER_DIMENSIONS],
                ]);
            }
            if resources.portraits.in_play
                && resources.portraits.in_battle
                && !resources.portraits.pause_active
                && is_subtitle(registers)
            {
                // the CPU decodes its arrays, so we can verify it is KIU's bottom-strip
                // dialogue subtitle box before centering (leaving Chapter End score/bonus
                // labels and other 512x128 text draws untouched).
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                let shaded = shaded_in_order(registers, vertices);
                if shaded.as_deref().is_some_and(is_top_subtitle_quad) {
                    resources.portraits.top_subtitle_quiet = 0;
                    resources.portraits.tutorial_dialogue_hold = 0;
                    resources.portraits.adding = true;
                    // the portraits go under the subtitles, in the buffer that
                    // goes out, once a frame
                    let top = resources.portraits.top.map(|[address, ..]| address);
                    if top == Some(registers[REG_COLOR_BUFFER_ADDRESS]) && !resources.portraits.flushed {
                        resources.portraits.flushed = true;
                        flush_portraits(memory, resources);
                    }
                    let tex_addr = loc_register(registers, TEXTURE_UNIT_BASES[0] + 4);
                    let buf_lines = resources.portraits.text_lines_for(tex_addr);
                    let centered = shaded.as_deref().and_then(|v| centered_subtitle(v, &buf_lines));
                    let draw_vertices = match &centered {
                        Some(v) => Vertices::Shaded(v),
                        None => vertices,
                    };
                    draw_subtitle_shadow(registers, memory, resources, draw_vertices);
                    rasterize(registers, memory, resources, draw_vertices);
                    resources.portraits.adding = false;
                    return Some(0);
                }
            }
        } else if is_text_buffer(registers) {
            // Track 512x128 A4 text line bounds unconditionally (zero visual side
            // effects) so the first subtitle of a chapter/cutscene or after a pause
            // always has its glyph bounds ready before `in_battle` wakes up.
            #[cfg(feature = "vulkan")]
            if matches!(vertices, Vertices::Raw { .. }) {
                return None;
            }
            note_text(registers, resources, vertices);
        } else if on_bottom(registers) || is_bottom(registers) {
            if resources.portraits.cauldron_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
                && needs_cauldron_bot_adjust(registers)
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_cauldron_bot(registers, shaded, &mut resources.portraits) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.chapter_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    if let Some(adjusted) = adjust_chapter_bot(registers, shaded) {
                        resources.portraits.adding = true;
                        let res = rasterize(registers, memory, resources, Vertices::Shaded(&adjusted));
                        resources.portraits.adding = false;
                        return res;
                    }
                    return Some(0);
                }
            }
            if resources.portraits.mode_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_mode_bot(registers, memory, resources, shaded);
                }
            }
            if resources.portraits.solo_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_solo_bot(registers, memory, resources, shaded);
                }
            }
            if resources.portraits.file_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_file_bot(registers, memory, resources, shaded);
                }
            }
            if resources.portraits.weapon_active
                && !resources.portraits.menu_override
                && read_float24(registers, REG_VIEWPORT_WIDTH) == 120.0
                && read_float24(registers, REG_VIEWPORT_HEIGHT) == 160.0
            {
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                if let Some(shaded) = shaded_in_order(registers, vertices) {
                    return draw_weapon_bot(registers, memory, resources, shaded);
                }
            }
            let portrait = on_bottom(registers) && is_portrait(registers);
            let on_title = is_on_title(&resources.portraits);
            let capturing_title = is_capturing_title(&resources.portraits);
            // the CPU decodes its arrays then, and its vertices say where it is.
            // out of play only a portrait's are looked at, to see whether it is
            // a big one, a conversation
            let special_menu = resources.portraits.fuse_active
                || resources.portraits.cauldron_active
                || resources.portraits.chapter_active
                || resources.portraits.mode_active
                || resources.portraits.solo_active
                || resources.portraits.file_active
                || resources.portraits.weapon_active;
            #[cfg(feature = "vulkan")]
            if ((resources.portraits.in_play
                && !special_menu
                && (resources.portraits.in_battle || resources.portraits.active_battle))
                || (capturing_title && on_title)
                || portrait)
                && matches!(vertices, Vertices::Raw { .. })
            {
                return None;
            }
            if !special_menu && (!resources.portraits.in_play || !resources.portraits.in_battle) && portrait {
                note_big_portrait(registers, resources, &vertices);
            }
            if !special_menu && !resources.portraits.in_play && !portrait && on_bottom(registers) && vertices.len() <= 16 && battle_detection() {
                // a sprite of the battle display, looked at on the CPU
                #[cfg(feature = "vulkan")]
                if matches!(vertices, Vertices::Raw { .. }) {
                    return None;
                }
                note_battle_display(registers, resources, &vertices);
            }
            if capturing_title {
                if on_title && on_bottom(registers) {
                    mirror_title(registers, resources, vertices);
                }
                return Some(0);
            }
            // in play the bottom screen is not shown, so what it draws goes
            // onto the top screen or nowhere; in a menu it is drawn as it is
            if resources.portraits.in_play && !special_menu {
                if on_bottom(registers) {
                    if portrait {
                        mirror_portrait(registers, resources, vertices);
                    } else if is_subtitle(registers) {
                        if let Some(shaded) = shaded_in_order(registers, vertices) {
                            if resources.portraits.top_subtitle_quiet > 30
                                && is_bottom_dialogue_box(&shaded)
                            {
                                resources.portraits.tutorial_dialogue_hold = 120;
                                mirror_dialogue(registers, resources, Vertices::Shaded(&shaded));
                            } else if (resources.portraits.in_battle || resources.portraits.active_battle)
                                && is_power_label_text(&resources.portraits, &shaded)
                            {
                                match power_label_mode() {
                                    PowerLabelMode::Bottom => {
                                        mirror_power_label(registers, resources, &shaded);
                                    }
                                    PowerLabelMode::AboveBar => {
                                        mirror_hud(registers, resources, Vertices::Shaded(&shaded));
                                    }
                                    PowerLabelMode::Off => {}
                                }
                            } else if resources.portraits.in_battle || resources.portraits.active_battle {
                                mirror_hud(registers, resources, Vertices::Shaded(&shaded));
                            }
                        }
                    } else if resources.portraits.in_battle || resources.portraits.active_battle {
                        mirror_hud(registers, resources, vertices);
                    } else if on_title {
                        mirror_title(registers, resources, vertices);
                    }
                }
                return Some(0);
            }
        }
    }
    let vertex_count = vertices.len();
    // the offset is two signed 10-bit fields.
    let signed10 = |value: u32| (((value & 0x3FF) << 22) as i32 >> 22) as f32;
    let viewport_x = signed10(registers[REG_VIEWPORT_XY]);
    let viewport_y = signed10(registers[REG_VIEWPORT_XY] >> 16);
    // the viewport registers hold *half* the extent, because what the hardware
    // actually wants is the scale factor that maps clip space (-1..1) onto the
    // target.
    let viewport_width = read_float24(registers, REG_VIEWPORT_WIDTH) * 2.0;
    let viewport_height = read_float24(registers, REG_VIEWPORT_HEIGHT) * 2.0;
    if viewport_width <= 0.0 || viewport_height <= 0.0 {
        return Some(0);
    }
    let viewport = (viewport_x, viewport_y, viewport_width, viewport_height);

    let color_buffer_raw = loc_register(registers, REG_COLOR_BUFFER_ADDRESS);
    let target_addr = memory.translate(color_buffer_raw);
    // the format sits in bits 16-18 of the register, not the low bits, the low
    // half holds how many bytes a pixel takes.
    let target_format = ColorFormat::from_color_buffer((registers[REG_COLOR_BUFFER_FORMAT] >> 16) & 7);
    // the buffer's real size, which the tile addressing uses, it is often
    // padded past the viewport, and tiling at the viewport's width instead
    // reads back as the image repeating down the screen.
    let dimensions = registers[REG_FRAMEBUFFER_DIMENSIONS];
    let viewport_right = (viewport_x + viewport_width).round() as i32;
    let viewport_top = (viewport_y + viewport_height).round() as i32;
    let buffer_width = (dimensions & 0x7FF).max(viewport_right.max(1) as u32);
    let buffer_height = (((dimensions >> 12) & 0x3FF) + 1).max(viewport_top.max(1) as u32);
    let (target_width, target_height) = (buffer_width, buffer_height);

    // draws are far more frequent than fills or transfers (thousands per
    // second versus dozens), so this stays at trace level to keep debug
    // usable for spotting the rarer GX commands.
    if log::log_enabled!(log::Level::Trace) {
        log::trace!(
            "draw: {vertex_count} vertices, topology reg 0x{:X}, viewport \
             {viewport_width}x{viewport_height} @({viewport_x},{viewport_y}), color buffer raw \
             0x{color_buffer_raw:08X} -> 0x{target_addr:08X} {target_format:?}, \
             texture0_addr=0x{:08X} texture0_config=0x{:08X} texture0_format=0x{:X}",
            registers[REG_PRIMITIVE_CONFIG],
            registers[REG_TEXTURE0_ADDRESS],
            registers[REG_TEXTURE_CONFIG],
            registers[REG_TEXTURE0_FORMAT],
        );
    }

    // color writes need the buffer to allow them at all, then each
    // channel's bit in GPUREG_DEPTH_COLOR_MASK.
    let color_writable = registers[REG_COLOR_BUFFER_WRITE] != 0;
    let color_mask = registers[REG_DEPTH_COLOR_MASK] >> 8;
    // a texture can be a buffer the GPU drew into, guest memory has to have
    // it before it is read
    // one the GPU still holds it copies there
    #[allow(unused_mut)]
    let mut drawn: [Option<DrawnTexture>; 3] = [None; 3];
    // the rest of one reaching past the rows drawn is read as memory holds
    // it, as of what was written back last
    #[allow(unused_mut)]
    let mut past = [false; 3];
    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        for (unit, drawn) in drawn.iter_mut().enumerate() {
            *drawn = drawn_texture(registers, memory, unit).filter(|texture| hardware.holds(texture));
            if let Some(texture) = drawn {
                past[unit] = !hardware.covers(texture);
                continue;
            }
            if let Some((addr, len)) = texture_range(registers, memory, unit) {
                if let Err(error) = hardware.prepare_read(memory, addr, len) {
                    log::error!("the GPU could not write back a buffer, {error}");
                }
            }
        }
    }
    let textures: [Option<BoundTexture>; 3] =
        std::array::from_fn(|unit| bind_texture(registers, memory, &mut resources.textures, unit, drawn[unit], past[unit]));
    log_draw(registers, memory, target_addr, (buffer_width, buffer_height), viewport);
    let tex_env = resources.tex_env.read(registers);
    // the lighting only matters to a draw whose combiners read it.
    let lighting = if tex_env.reads_lighting() { resources.lighting.read(registers) } else { None };
    let state = DrawState {
        target: ColorTarget {
            addr: target_addr,
            // drawing stays inside the viewport.
            left: (viewport_x.round() as i32).max(0),
            bottom: (viewport_y.round() as i32).max(0),
            right: viewport_right.min(buffer_width as i32),
            top: viewport_top.min(buffer_height as i32),
            buffer_width,
            buffer_height,
            format: target_format,
            write: std::array::from_fn(|channel| color_writable && color_mask & (1 << channel) != 0),
        },
        depth_map: DepthMap::read(registers),
        depth_stencil: DepthStencil::read(registers, memory),
        tex_env,
        alpha_test: AlphaTest::read(registers),
        blend: crate::blend::Blend::read(registers),
        logic_op: crate::blend::LogicOp::read(registers),
        textures: &textures,
        texture0_projection: (registers[TEXTURE_UNIT_BASES[0] + 2] >> 28) & 0x7 == 3,
        texture2_uses_coord1: registers[REG_TEXTURE_CONFIG] & (1 << 13) != 0,
        lighting,
        tables: &resources.light_tables,
        proctex: crate::proctex::ProcTex::read(registers, &resources.proctex_tables),
        proctex_tables: &resources.proctex_tables,
        fog: crate::fog::Fog::read(registers, &resources.fog_table),
        fog_table: &resources.fog_table,
    };
    // what the draw writes, which the textures of later draws in the list
    // may be
    let pixels = buffer_width * buffer_height;
    resources.textures.wrote(target_addr, pixels * target_format.bytes_per_pixel() as u32);
    if let Some(depth) = &state.depth_stencil {
        resources.textures.wrote(depth.addr, pixels * depth.bytes);
    }

    // GPUREG_PRIMITIVE_CONFIG bits [9:8], 0 = triangle list, 1 = strip,
    // 2 = fan, 3 = whatever the geometry shader emitted, which is a list.
    let topology = (registers[REG_PRIMITIVE_CONFIG] >> 8) & 0x3;

    // GPUREG_FACECULLING_CONFIG, 0 keeps everything, 1 keeps triangles
    // wound clockwise and 2 counter-clockwise, as seen with y pointing up.
    let cull_mode = registers[REG_FACE_CULLING] & 0x3;

    // the GPU runs the vertex shader itself, unless a geometry shader
    // comes after it
    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        let target = &state.target;
        let whole = target.buffer_width.is_multiple_of(8) && target.buffer_height.is_multiple_of(8);
        let unshaded = match vertices {
            Vertices::Unshaded { vertex_shader, inputs, order, .. } => {
                Some((vertex_shader, hardware::Inputs::Decoded(inputs), order, None))
            }
            Vertices::Raw { vertex_shader, raw, vertices } => Some((vertex_shader, hardware::Inputs::Raw(raw), None, Some(vertices))),
            Vertices::Shaded(_) | Vertices::ShadedOrdered { .. } => None,
        };
        // ZAKURO_CPU_SHADE_LOG=1: why draws the GPU could shade are shaded
        // on the CPU, and how many vertices
        if hardware.shades() && cpu_shade_log() {
            use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
            static COUNTS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
            let reason = match () {
                _ if unshaded.is_none() => Some(0),
                _ if registers[REG_GEOSTAGE_CONFIG] & 0x3 == 2 => Some(1),
                _ if !whole => Some(2),
                _ => None,
            };
            if let Some(reason) = reason {
                COUNTS[reason * 2].fetch_add(1, Relaxed);
                COUNTS[reason * 2 + 1].fetch_add(vertex_count as u64, Relaxed);
                let total = COUNTS[0].load(Relaxed) + COUNTS[2].load(Relaxed) + COUNTS[4].load(Relaxed);
                if total % 3000 == 0 {
                    let c: Vec<u64> = COUNTS.iter().map(|count| count.swap(0, Relaxed)).collect();
                    log::info!(
                        target: "zakuro_gpu::diag",
                        "CPU-shaded draws: {} already shaded ({} vertices), {} with a geometry shader ({}), {} into a buffer not of whole 8x8 tiles ({}, {}x{})",
                        c[0], c[1], c[2], c[3], c[4], c[5], target.buffer_width, target.buffer_height
                    );
                }
            }
        }
        // a geometry shader's draw as its slots, the vertex shader and the
        // geometry shader run for each, when the geometry program only
        // goes forward and its translation is made
        if let Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } = vertices {
            let gs_config = registers[REG_GS_CONFIG];
            let gs_mode = gs_config & 0xFF;
            if gpu_geometry() && hardware.shades() && whole && registers[REG_GEOSTAGE_CONFIG] & 0x3 == 2 && (gs_mode == 0 || gs_mode == 2) {
                let program_key = (geometry_shader.fingerprint(), geometry_shader.entry_point);
                let slots = *resources.geometry_slots.entry(program_key).or_insert_with(|| {
                    geometry_shader.prepared().and_then(|program| crate::shader::glsl::geometry_slots(&program, geometry_shader.entry_point))
                });
                let per_vertex = (registers[REG_VS_OUTPUT_TOTAL] & 0xF) + 1;
                let fixed = (gs_mode == 2).then(|| (((gs_config >> 8) & 0xF) + 1, (gs_config >> 16) & 0xFF));
                let fits = fixed.is_none_or(|(count, start)| (start + count * per_vertex) as usize <= shader::FLOAT_UNIFORMS);
                if let (Some(slots), true) = (slots, fits) {
                    let layout = crate::shader::glsl::GeometryLayout {
                        per_vertex,
                        per_invocation: (registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG] & 0xF) + 1,
                        vertex_mask: registers[REG_VS_OUTPUT_MASK] & 0xFFFF,
                        input_map: registers[REG_GS_BLOCK + SHADER_INPUT_MAP_LOW] as u64
                            | (registers[REG_GS_BLOCK + SHADER_INPUT_MAP_HIGH] as u64) << 32,
                        fixed,
                        slots,
                    };
                    let semantics = semantics_for(registers, registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK]);
                    let geometry = hardware::GeometryShading { unit: geometry_shader, layout };
                    if hardware.geometry_ready(vertex_shader, &geometry, semantics) {
                        // the vertices in draw order, each invocation taking
                        // the next few, those left over dropped
                        let mut gathered = std::mem::take(&mut resources.geometry_inputs);
                        gathered.clear();
                        match order {
                            Some(order) => gathered.extend(order.iter().map(|&i| inputs[i])),
                            None => gathered.extend_from_slice(inputs),
                        }
                        let invocations = gathered.len() as u32 / layout.vertices();
                        let mut indices = std::mem::take(&mut resources.geometry_indices);
                        indices.clear();
                        indices.extend(0..invocations * slots);
                        let shading = hardware::Shading {
                            unit: vertex_shader,
                            inputs: hardware::Inputs::Decoded(&gathered),
                            indices: &indices,
                            semantics,
                            viewport,
                            cull: cull_mode,
                            geometry: Some(geometry),
                        };
                        let drawn = {
                            let _hardware_timer = crate::stats::time(&crate::stats::HARDWARE);
                            hardware.draw(memory, &state.hardware(registers, hardware::Geometry::Shaded(&shading)))
                        };
                        let triangles = indices.len() as u32 / 3;
                        resources.geometry_inputs = gathered;
                        resources.geometry_indices = indices;
                        match drawn {
                            Ok(()) => {
                                hardware.geometry_draws += 1;
                                return Some(triangles);
                            }
                            Err(error) if error == hardware::GEOMETRY_NOT_READY => {}
                            Err(error) => log::error!("the GPU could not draw a geometry shader's draw, {error}, drawing it on the CPU"),
                        }
                    }
                }
            }
        }
        if let Some((unit, inputs, order, raw)) = unshaded.filter(|_| hardware.shades() && whole && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2) {
            let mut assembled = std::mem::take(&mut resources.triangles);
            let indices = triangle_indices(topology, vertex_count, raw, order, &mut assembled);
            let shading = hardware::Shading {
                unit,
                inputs,
                indices,
                semantics: resources.semantics.get(registers),
                viewport,
                cull: cull_mode,
                geometry: None,
            };
            let drawn = {
                let _hardware_timer = crate::stats::time(&crate::stats::HARDWARE);
                hardware.draw(memory, &state.hardware(registers, hardware::Geometry::Shaded(&shading)))
            };
            let triangles = (indices.len() / 3) as u32;
            resources.triangles = assembled;
            match drawn {
                Ok(()) => return Some(triangles),
                Err(error) => log::error!("the GPU could not shade, {error}, shading on the CPU"),
            }
        }
    }
    let held: std::sync::Arc<Vec<Vertex>>;
    let (shaded, order) = match vertices {
        Vertices::Shaded(shaded) => (shaded, None),
        Vertices::ShadedOrdered { shaded, order } => (shaded, order),
        Vertices::Unshaded { vertex_shader, geometry_shader, inputs, order } => {
            let signature = (vcache_enabled() && registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2 && inputs.len() >= 128 && !looks_at_vertices(registers))
                .then(|| draw_signature(vertex_shader, inputs));
            // 1. what shading reads is exactly as some recent frame had it
            let cache_key = signature.filter(|&signature| vcache_worth(&mut resources.vcache_score, signature)).map(|_| shading_key(registers, vertex_shader, inputs));
            let found = cache_key.and_then(|key| resources.vcache.get(&key).or_else(|| resources.vcache_old.get(&key))).cloned();
            if let (Some(signature), Some(_)) = (signature, cache_key) {
                vcache_note(&mut resources.vcache_score, signature, found.is_some());
            }
            let reused = found.is_some();
            held = match found {
                Some(vertices) => vertices,
                None => {
                    let (vertices, _) = process_vertices(registers, vertex_shader, geometry_shader, inputs, order);
                    std::sync::Arc::new(vertices)
                }
            };
            // a geometry shader's triangles are in the order it made them,
            // the draw's own order numbers its inputs
            let order = order.filter(|_| registers[REG_GEOSTAGE_CONFIG] & 0x3 != 2);
            if let (Some(key), false) = (cache_key, reused) {
                if resources.vcache.len() >= 40 {
                    resources.vcache_old = std::mem::take(&mut resources.vcache);
                }
                resources.vcache.insert(key, held.clone());
            }
            if let (Some(signature), false) = (signature, reused) {
                let last = held.len() - 1;
                let samples = [held[0], held[held.len() / 2], held[last]];
                if resources.vreuse.len() > 256 {
                    resources.vreuse.clear();
                }
                resources.vreuse.insert(signature, Reuse { vertices: held.clone(), samples, age: 0 });
            }
            (&held[..], order)
        }
        // arrays the GPU did not draw, which the CPU decodes first
        #[cfg(feature = "vulkan")]
        Vertices::Raw { .. } => return None,
    };
    // which of the vertices made each vertex of the draw is
    let at = |i: usize| order.map_or(i, |order| order[i]);
    let count = order.map_or(shaded.len(), <[usize]>::len);
    let triangle_indices = assemble(topology, count);
    let triangle_count = match topology {
        1 | 2 => count.saturating_sub(2),
        _ => count / 3,
    };
    let mut clipped_away = 0u32;
    let mut culled = 0u32;
    let mut stats = FillStats::default();
    // where each vertex inside the volume lands, worked out once however
    // many triangles share it, and the triangles as indices into those, a
    // triangle of such vertices needs no clipping. what clipping makes goes
    // after them
    const OUTSIDE: u32 = u32::MAX;
    let mut placed = std::mem::take(&mut resources.placed);
    let mut slots = std::mem::take(&mut resources.slots);
    let mut indices = std::mem::take(&mut resources.shaded_indices);
    placed.clear();
    slots.clear();
    indices.clear();
    placed.reserve(shaded.len());
    slots.reserve(shaded.len());
    indices.reserve(triangle_count * 3);

    #[cfg(feature = "vulkan")]
    let use_hardware = resources.hardware.is_some();
    #[cfg(not(feature = "vulkan"))]
    let use_hardware = false;

    let mut all_inside = true;
    for vertex in shaded {
        let valid = if use_hardware {
            vertex.clip[3] >= 1e-5
        } else {
            inside(vertex)
        };
        if valid {
            if let Some(screen) = to_screen(vertex, viewport) {
                placed.push(screen);
                slots.push(placed.len() as u32 - 1);
                continue;
            }
        }
        all_inside = false;
        slots.push(OUTSIDE);
    }
    let culls = |p: &Screen, q: &Screen, r: &Screen| {
        if cull_mode == 0 {
            return false;
        }
        let counter_clockwise = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x) > 0.0;
        counter_clockwise == (cull_mode == 1)
    };
    if all_inside {
        for (ia, ib, ic) in triangle_indices {
            let (a, b, c) = (at(ia), at(ib), at(ic));
            if !culls(&placed[a], &placed[b], &placed[c]) {
                indices.extend([a as u32, b as u32, c as u32]);
            } else {
                culled += 1;
            }
        }
    } else {
        for (ia, ib, ic) in triangle_indices {
            let (a, b, c) = (at(ia), at(ib), at(ic));
            let [sa, sb, sc] = [slots[a], slots[b], slots[c]];
            if sa != OUTSIDE && sb != OUTSIDE && sc != OUTSIDE {
                if culls(&placed[sa as usize], &placed[sb as usize], &placed[sc as usize]) {
                    culled += 1;
                } else {
                    indices.extend([sa, sb, sc]);
                }
                continue;
            }
            if use_hardware && shaded[a].clip[3] < 1e-5 && shaded[b].clip[3] < 1e-5 && shaded[c].clip[3] < 1e-5 {
                clipped_away += 1;
                continue;
            }
            let polygon = clip_triangle([shaded[a], shaded[b], shaded[c]]);
            let poly_slice = polygon.as_slice();
            let mut screen = [Screen::default(); 16];
            let mut screen_len = 0;
            for v in poly_slice {
                if let Some(s) = to_screen(v, viewport) {
                    screen[screen_len] = s;
                    screen_len += 1;
                } else {
                    break;
                }
            }
            if screen_len < 3 || screen_len != poly_slice.len() {
                clipped_away += 1;
                continue;
            }
            if culls(&screen[0], &screen[1], &screen[2]) {
                culled += 1;
                continue;
            }
            let first = placed.len() as u32;
            placed.extend_from_slice(&screen[..screen_len]);
            for i in 1..screen_len as u32 - 1 {
                indices.extend([first, first + i, first + i + 1]);
            }
        }
    }

    #[cfg(feature = "vulkan")]
    if let Some(hardware) = resources.hardware.as_mut() {
        let target = &state.target;
        // tiles are whole in any buffer a title really draws into
        if target.buffer_width.is_multiple_of(8) && target.buffer_height.is_multiple_of(8) {
            if indices.is_empty() {
                resources.placed = placed;
                resources.slots = slots;
                resources.shaded_indices = indices;
                return Some(0);
            }
            let hardware_timer = crate::stats::time(&crate::stats::HARDWARE);
            // ZAKURO_SELF_CHUNKS=N: a draw reading the buffer it draws into
            // goes in N runs of triangles, each reading what the one before
            // drew, as the console's tile by tile drawing does (a test)
            static CHUNKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let chunks = *CHUNKS.get_or_init(|| std::env::var("ZAKURO_SELF_CHUNKS").ok().and_then(|v| v.parse().ok()).unwrap_or(1));
            let own = chunks > 1
                && state.textures[0].as_ref().and_then(|t| t.drawn).is_some_and(|d| d.addr & 0x00FF_FFFF == (registers[REG_COLOR_BUFFER_ADDRESS] << 3) & 0x00FF_FFFF);
            let drawn_on_gpu = if own {
                let triangles = indices.len() / 3;
                let per = triangles.div_ceil(chunks).max(1);
                let mut result = Ok(());
                for run in indices.chunks(per * 3) {
                    let geometry = hardware::Geometry::Placed { vertices: &placed, indices: run };
                    result = hardware.draw(memory, &state.hardware(registers, geometry));
                    if result.is_err() {
                        break;
                    }
                }
                result
            } else {
                let geometry = hardware::Geometry::Placed { vertices: &placed, indices: &indices };
                hardware.draw(memory, &state.hardware(registers, geometry))
            };
            drop(hardware_timer);
            match drawn_on_gpu {
                Ok(()) => {
                    resources.placed = placed;
                    resources.slots = slots;
                    resources.shaded_indices = indices;
                    return Some(triangle_count as u32);
                }
                Err(error) => log::error!("the GPU could not draw, {error}, drawing in software"),
            }
        }
        // the draw in between goes into twins alone, which the software
        // path has none of
        if hardware.pass == Pass::Twin {
            resources.placed = placed;
            resources.slots = slots;
            resources.shaded_indices = indices;
            return Some(0);
        }
        // the software path works on guest memory, which has to hold what
        // the GPU drew
        if let Err(error) = hardware.flush(memory) {
            log::error!("the GPU could not write back its buffers, {error}");
        }
        // and on textures read from it, which the ones the GPU was to copy
        // were not
        if drawn.iter().any(Option::is_some) {
            let hardware = resources.hardware.take();
            let in_order: Vec<Vertex> = (0..order.map_or(shaded.len(), <[usize]>::len)).map(|i| shaded[at(i)]).collect();
            resources.placed = placed;
            resources.slots = slots;
            resources.shaded_indices = indices;
            let drawn = rasterize(registers, memory, resources, Vertices::Shaded(&in_order));
            resources.hardware = hardware;
            return drawn;
        }
    }

    let triangles: Vec<[Screen; 3]> =
        indices.as_chunks::<3>().0.iter().map(|&[a, b, c]| [placed[a as usize], placed[b as usize], placed[c as usize]]).collect();
    // the rows the triangles can reach, as window rows and then as rows of
    // the buffer, which counts from the other end.
    let target = &state.target;
    let low = triangles.iter().map(|t| t.iter().map(|v| v.y).fold(f32::MAX, f32::min).floor() as i32).min();
    let high = triangles.iter().map(|t| t.iter().map(|v| v.y).fold(f32::MIN, f32::max).ceil() as i32).max();
    if let (Some(low), Some(high)) = (low, high) {
        let (low, high) = (low.max(target.bottom), high.min(target.top));
        if low < high {
            let rows = (target.buffer_height - high as u32)..(target.buffer_height - low as u32);
            let (width, height) = (target.buffer_width, target.buffer_height);
            let bpp = target.format.bytes_per_pixel() as u32;
            let mut color = target
                .write
                .iter()
                .any(|&w| w)
                .then(|| Surface::load(memory, target.addr, bpp, width, height, rows.clone()));
            let mut depth = state
                .depth_stencil
                .as_ref()
                .map(|buffer| Surface::load(memory, buffer.addr, buffer.bytes, width, height, rows.clone()));
            fill(&mut color, &mut depth, &triangles, low..high, &state, &mut stats);
            if let Some(color) = color.as_ref().filter(|_| stats.written > 0) {
                color.store(memory);
            }
            if let (Some(depth), Some(buffer)) = (&depth, &state.depth_stencil) {
                if buffer.write_depth || buffer.write_stencil {
                    depth.store(memory);
                }
            }
        }
    }

    if log::log_enabled!(log::Level::Trace) {
        let first_screen = shaded.first().and_then(|v| to_screen(v, viewport));
        log::trace!(
            "draw result: {vertex_count} verts, {triangle_count} tris ({clipped_away} clipped \
             away, {culled} culled), {} pixels written ({} failed depth, {} failed stencil, {} failed alpha), \
             target 0x{target_addr:08X} {target_width}x{target_height} {target_format:?}, \
             texture={} first vertex clip={:?} screen(x,y,inv_w)={first_screen:?}, depth/color \
             mask 0x{:08X}, color op 0x{:08X}, blend 0x{:08X}, depth map {:?}, tev0 {:08X?}, \
             tev1 {:08X?}, tev buffer 0x{:08X}",
            stats.written,
            stats.depth_failed,
            stats.stencil_failed,
            stats.alpha_failed,
            textures[0].is_some(),
            shaded.first().map(|v| v.clip),
            registers[REG_DEPTH_COLOR_MASK],
            registers[REG_COLOR_OPERATION],
            registers[REG_BLEND_FUNC],
            state.depth_map,
            &registers[0x0C0..0x0C5],
            &registers[0x0C8..0x0CD],
            registers[0x0E0],
        );
    }

    resources.placed = placed;
    resources.slots = slots;
    resources.shaded_indices = indices;
    Some(stats.written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// the varyings picked out of packed attributes one by one, the way
    /// they were before VertexPlan.
    fn picked_vertex(map: &OutputMap, attributes: &[Vec4; 16]) -> Vertex {
        let get = |slot: (usize, usize)| attributes[slot.0][slot.1];
        let color_or = |slot: Option<(usize, usize)>| slot.map_or(1.0, get);
        Vertex {
            clip: map.position.map(get),
            color: map.color.map(color_or),
            texcoords: map.texcoords.map(|set| [set[0].map_or(0.0, get), set[1].map_or(0.0, get)]),
            texcoord0_w: map.texcoord0_w.map_or(1.0, get),
            quaternion: map.quaternion.map(|slot| slot.map_or(0.0, get)),
            view: map.view.map(|slot| slot.map_or(0.0, get)),
        }
    }

    fn vertex_bits(vertex: &Vertex) -> Vec<u32> {
        let texcoords = vertex.texcoords.as_flattened();
        [&vertex.clip[..], &vertex.color, texcoords, &[vertex.texcoord0_w], &vertex.quaternion, &vertex.view].concat().iter().map(|v| v.to_bits()).collect()
    }

    /// placing a vertex's fields gives the bits reading each component
    /// does, for every type and count.
    #[test]
    fn placing_fields_reads_what_each_component_does() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for case in 0..20_000 {
            let bytes: Vec<u8> = (0..64).map(|_| next() as u8).collect();
            let mut fields = Vec::new();
            let mut offset = 0u32;
            for register in 0..(1 + next() % 4) as usize {
                let (ty, count) = (next() % 4, 1 + next() % 4);
                let size = component_size(ty);
                offset = offset.next_multiple_of(size);
                fields.push(PlacedField { offset, ty, count, register: register * 3 % shader::INPUT_REGISTERS });
                offset += size * count;
            }
            let loader = LoaderPlan { index: 0, offset: 0, stride: offset, size: offset, fields };
            let mut placed = [[f32::from_bits(next()); 4]; shader::INPUT_REGISTERS];
            let mut expected = placed;
            loader.place(&bytes, &mut placed);
            for field in &loader.fields {
                let size = component_size(field.ty) as usize;
                let mut value = [0.0f32, 0.0, 0.0, 1.0];
                for (i, slot) in value.iter_mut().enumerate().take(field.count as usize) {
                    let at = field.offset as usize + i * size;
                    *slot = component(&bytes[at..at + size], field.ty);
                }
                expected[field.register] = value;
            }
            let bits = |input: &[Vec4; shader::INPUT_REGISTERS]| input.as_flattened().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&placed), bits(&expected), "case {case}");
        }
    }

    /// a plan gives the same bits as packing the outputs and picking the
    /// varyings, for any output map and mask, missing values and NaNs too.
    #[test]
    fn a_vertex_plan_picks_what_packing_did() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut registers = vec![0u32; 0x300];
        for case in 0..20_000 {
            registers[REG_SHADER_OUTPUT_TOTAL] = next() % 9;
            for word in &mut registers[REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END] {
                // semantics, the unused 0x1F among them
                *word = u32::from_le_bytes(std::array::from_fn(|_| [0x1F, (next() % 24) as u8][(next() % 4 != 0) as usize]));
            }
            let mask = match case % 4 {
                0 => 0,
                1 => 0xFFFF,
                _ => next() & 0xFFFF,
            };
            let map = read_output_map(&registers);
            let plan = VertexPlan::new(&map, mask);
            let outputs: [Vec4; shader::OUTPUT_REGISTERS] = std::array::from_fn(|_| {
                std::array::from_fn(|_| match next() % 8 {
                    0 => f32::from_bits(0x7FC0_0000 | next() & 0x3F_FFFF),
                    1 => -0.0,
                    _ => next() as f32 / 1000.0 - 1000.0,
                })
            });
            let expected = picked_vertex(&map, &pack_outputs(&outputs, mask));
            assert_eq!(vertex_bits(&plan.vertex(&outputs)), vertex_bits(&expected), "case {case}, mask 0x{mask:04X}, map {map:?}");
        }
    }

    /// guest memory seen the way the GPU sees it, physical VRAM and FCRAM
    /// translate to different virtual windows. its first megabytes lie in
    /// one piece, as the linear heap and VRAM do, unless it is scattered.
    #[derive(Clone)]
    struct ConsoleMemory {
        flat: Vec<u8>,
        rest: HashMap<u32, u8>,
        /// whether slice hands out the flat bytes.
        whole: bool,
    }

    impl Default for ConsoleMemory {
        fn default() -> ConsoleMemory {
            ConsoleMemory { flat: vec![0; 4 << 20], rest: HashMap::new(), whole: true }
        }
    }

    impl ConsoleMemory {
        /// memory nothing is in one piece of, for the paths that copy.
        fn scattered() -> ConsoleMemory {
            ConsoleMemory { whole: false, ..ConsoleMemory::default() }
        }
    }

    impl GpuMemory for ConsoleMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            for (i, byte) in out.iter_mut().enumerate() {
                let at = addr + i as u32;
                *byte = self.flat.get(at as usize).copied().unwrap_or_else(|| self.rest.get(&at).copied().unwrap_or(0));
            }
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            for (i, &byte) in data.iter().enumerate() {
                let at = addr + i as u32;
                match self.flat.get_mut(at as usize) {
                    Some(flat) => *flat = byte,
                    None => {
                        self.rest.insert(at, byte);
                    }
                }
            }
        }

        fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
            self.slice_mut(addr, len).map(|slice| &*slice)
        }

        fn slice_mut(&mut self, addr: u32, len: usize) -> Option<&mut [u8]> {
            let at = addr as usize;
            if !self.whole || at + len > self.flat.len() {
                return None;
            }
            Some(&mut self.flat[at..at + len])
        }

        fn translate(&self, paddr: u32) -> u32 {
            match paddr {
                0x2000_0000.. => 0x1400_0000 + (paddr - 0x2000_0000),
                0x1800_0000..=0x185F_FFFF => 0x1F00_0000 + (paddr - 0x1800_0000),
                _ => paddr,
            }
        }
    }

    /// titles point the attribute base at the start of VRAM and reach their
    /// vertex data in FCRAM through each loader's offset.
    #[test]
    fn a_loader_offset_can_reach_another_memory_region() {
        let mut memory = ConsoleMemory::default();
        // vertex data at physical 0x20000100, virtual 0x14000100.
        for (i, value) in [1.5f32, -2.0, 0.25, 1.0].iter().enumerate() {
            memory.write(0x1400_0100 + i as u32 * 4, &value.to_le_bytes());
        }

        let mut registers = vec![0u32; 0x300];
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0xF; // attribute 0, four floats
        registers[REG_ATTRIBUTE_LOADER] = 0x0800_0100;
        registers[REG_ATTRIBUTE_LOADER + 2] = (1 << 28) | (16 << 16);
        let layout = VertexLayout::read(&registers);

        let input = fetch_vertex(&registers, &mut memory, 0x1800_0000, &layout, &[shader::ZERO; 16], 0);
        assert_eq!(input[0], [1.5, -2.0, 0.25, 1.0]);
    }

    /// a draw that keeps the last one's layout but moves its array or
    /// changes a fixed attribute reads its own vertices and values.
    #[test]
    fn a_kept_input_plan_follows_the_draw() {
        let mut memory = ConsoleMemory::default();
        for (i, value) in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter().enumerate() {
            memory.write(0x1400_0100 + i as u32 * 4, &value.to_le_bytes());
        }
        let mut registers = vec![0u32; 0x300];
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0xF; // attribute 0, four floats
        registers[REG_ATTRIBUTE_FORMAT_HIGH] = 1 << 17; // attribute 1 fixed
        registers[REG_ATTRIBUTE_LOADER] = 0x100;
        registers[REG_ATTRIBUTE_LOADER + 2] = (1 << 28) | (16 << 16);
        registers[REG_VS_NUM_INPUT_ATTRIBUTES] = 1;
        registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x10; // attribute 1 in v1
        let mut fixed = [shader::ZERO; 16];
        fixed[1] = [0.5; 4];

        let mut plans = Plans::default();
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[..2], [[1.0, 2.0, 3.0, 4.0], [0.5; 4]]);

        registers[REG_ATTRIBUTE_LOADER] = 0x110;
        fixed[1] = [0.25; 4];
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[..2], [[5.0, 6.0, 7.0, 8.0], [0.25; 4]]);
        assert_eq!(plans.keys.len(), 1);

        // two floats a vertex is another layout
        registers[REG_ATTRIBUTE_FORMAT_LOW] = 0x7;
        let plan = plans.update(&registers, &fixed);
        assert_eq!(plan.fetch(&mut memory, 0x2000_0000, 0)[0], [5.0, 6.0, 0.0, 1.0]);
        assert_eq!(plans.keys.len(), 2);
    }

    const COLOR: u32 = 0x1000;
    const DEPTH: u32 = 0x3000;

    /// the inverse of decode_float24, for normal values.
    fn float24(value: f32) -> u32 {
        if value == 0.0 {
            return 0;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xFF) - 127 + 63;
        ((bits >> 31) << 23) | (exponent << 16) | ((bits >> 7) & 0xFFFF)
    }

    /// an 8x8 RGBA8 target with color writes on and a D24S8 buffer beside it.
    fn rasterize_shaded<M: GpuMemory>(registers: &[u32], memory: &mut M, resources: &mut Resources, shaded: &[Vertex]) -> u32 {
        rasterize(registers, memory, resources, Vertices::Shaded(shaded)).unwrap_or(0)
    }

    fn target_registers() -> Vec<u32> {
        let mut registers = vec![0u32; 0x300];
        registers[REG_VIEWPORT_WIDTH] = float24(4.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
        registers[REG_COLOR_BUFFER_ADDRESS] = COLOR >> 3;
        registers[REG_DEPTH_BUFFER_ADDRESS] = DEPTH >> 3;
        registers[REG_DEPTH_BUFFER_FORMAT] = 3;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (7 << 12);
        registers[REG_COLOR_BUFFER_WRITE] = 0xF;
        registers[REG_DEPTH_STENCIL_WRITE] = 0x3;
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8;
        registers[REG_LOGIC_OP] = 3;
        registers
    }

    /// a triangle covering the whole target at clip-space depth z.
    fn cover(z: f32, color: Vec4) -> Vec<Vertex> {
        [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .into_iter()
            .map(|[x, y]| Vertex { clip: [x, y, z, 1.0], color, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() })
            .collect()
    }

    fn pixels(memory: &mut ConsoleMemory) -> Vec<[u8; 4]> {
        (0..64)
            .map(|i| {
                let mut raw = [0u8; 4];
                memory.read(COLOR + i * 4, &mut raw);
                ColorFormat::Rgba8.decode(&raw)
            })
            .collect()
    }

    const RED: Vec4 = [1.0, 0.0, 0.0, 1.0];
    const GREEN: Vec4 = [0.0, 1.0, 0.0, 1.0];
    const BLUE: Vec4 = [0.0, 0.0, 1.0, 1.0];

    /// titles map depth with a scale of -1, so near is 1, and keep the
    /// nearer surface with a GREATER test against a buffer cleared to 0.
    #[test]
    fn reversed_depth_keeps_the_nearer_surface() {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        // test enabled, GREATER, with writes.
        registers[REG_DEPTH_COLOR_MASK] |= 1 | (6 << 4) | (1 << 12);

        let mut memory = ConsoleMemory::default();
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.2, RED));
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.8, GREEN));
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.3, BLUE));
        assert!(pixels(&mut memory).iter().all(|&p| p == [0, 255, 0, 255]));
    }

    #[test]
    fn the_stencil_test_masks_pixels() {
        let mut registers = target_registers();
        // stencil, enabled, EQUAL, write mask 0xFF, reference 1, compare 0xFF.
        registers[REG_STENCIL_TEST] = 1 | (2 << 4) | (0xFF << 8) | (1 << 16) | (0xFF << 24);
        let mut memory = ConsoleMemory::default();
        // half the samples hold a stencil of 1.
        for sample in 0..32 {
            memory.write(DEPTH + sample * 4 + 3, &[1]);
        }
        let written = rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, RED));
        assert_eq!(written, 32);
    }

    /// a floor-like triangle with its far corner behind the camera, the part in
    /// front of the camera covers everything above its near edge.
    #[test]
    fn a_triangle_crossing_the_camera_is_clipped_not_dropped() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |clip: Vec4| Vertex { clip, color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() };
        let triangle = [
            vertex([-1.0, -1.0, -0.5, 1.0]),
            vertex([1.0, -1.0, -0.5, 1.0]),
            vertex([0.0, 2.0, 0.5, -1.0]),
        ];
        let written = rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &triangle);
        assert_eq!(written, 64);
    }

    #[test]
    fn the_color_mask_keeps_channels_it_does_not_write() {
        let mut registers = target_registers();
        registers[REG_DEPTH_COLOR_MASK] = 1 << 8; // red only
        let mut memory = ConsoleMemory::default();
        for i in 0..64 {
            let mut raw = [0u8; 4];
            ColorFormat::Rgba8.encode([10, 20, 30, 40], &mut raw);
            memory.write(COLOR + i * 4, &raw);
        }
        rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, [1.0, 1.0, 1.0, 1.0]));
        assert!(pixels(&mut memory).iter().all(|&p| p == [255, 20, 30, 40]));
    }

    /// the two triangles of a quad cover each pixel exactly once, including
    /// along the diagonal they share.
    #[test]
    fn a_quad_covers_every_pixel_once() {
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() };
        let quad = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad), 64);
    }

    /// a draw big enough to be split over threads covers the same pixels,
    /// each once, that it would on one.
    #[test]
    fn a_big_draw_split_over_threads_covers_every_pixel_once() {
        const BIG_COLOR: u32 = 0x10_0000;
        const BIG_DEPTH: u32 = 0x20_0000;
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(128.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(128.0);
        registers[REG_COLOR_BUFFER_ADDRESS] = BIG_COLOR >> 3;
        registers[REG_DEPTH_BUFFER_ADDRESS] = BIG_DEPTH >> 3;
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 256 | (255 << 12);
        let mut memory = ConsoleMemory::default();
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() };
        let quad = [
            vertex(-1.0, -1.0),
            vertex(1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, -1.0),
            vertex(1.0, 1.0),
            vertex(-1.0, 1.0),
        ];
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad), 256 * 256);
        for i in 0..256 * 256 {
            let mut raw = [0u8; 4];
            memory.read(BIG_COLOR + i * 4, &mut raw);
            assert_eq!(ColorFormat::Rgba8.decode(&raw), [255, 0, 0, 255], "pixel {i}");
        }
    }

    /// culling keeps the winding the register asks for and drops the other.
    #[test]
    fn face_culling_keeps_one_winding() {
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color: RED, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() };
        // counter-clockwise with y up.
        let triangle = [vertex(-1.0, -1.0), vertex(3.0, -1.0), vertex(-1.0, 3.0)];
        for (mode, expected) in [(0, 64), (1, 0), (2, 64)] {
            let mut registers = target_registers();
            registers[REG_FACE_CULLING] = mode;
            let mut memory = ConsoleMemory::default();
            assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &triangle), expected, "mode {mode}");
        }
    }

    /// window y counts up from the bottom of the buffer, whose rows are
    /// stored top first, a viewport over the upper half of an 8x8 buffer
    /// fills its first four rows in memory and leaves the rest alone.
    #[test]
    fn a_viewport_offset_places_the_image_from_the_bottom() {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_HEIGHT] = float24(2.0);
        registers[REG_VIEWPORT_XY] = 4 << 16;
        let mut memory = ConsoleMemory::default();
        assert_eq!(rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &cover(-0.5, RED)), 32);
        for row in 0..8 {
            for x in 0..8 {
                let index = crate::format::morton_offset(x, row, 8, 1);
                let mut raw = [0u8; 4];
                memory.read(COLOR + index * 4, &mut raw);
                let red = ColorFormat::Rgba8.decode(&raw) == [255, 0, 0, 255];
                assert_eq!(red, row < 4, "row {row}, column {x}");
            }
        }
    }

    #[test]
    fn clamp_to_border_reads_the_border_color() {
        let texture = |wrap: Wrap| BoundTexture {
            drawn: None,
            texels: vec![[0xFF; 4]; 8 * 8].into(),
            linear: false,
            min_linear: false,
            mip_linear: false,
            max_level: 0,
            lod_bias: 0,
            wrap_s: wrap,
            wrap_t: wrap,
            width: 8,
            height: 8,
            border: RED,
            replacement: None,
        };
        let white = [1.0; 4];
        assert_eq!(texture(Wrap::ClampToBorder).texel(-1, 3), RED);
        assert_eq!(texture(Wrap::ClampToBorder).texel(3, 8), RED);
        assert_eq!(texture(Wrap::ClampToBorder).texel(3, 3), white);
        assert_eq!(texture(Wrap::ClampToEdge).texel(-1, 3), white);
    }

    /// within a command list memory changes only where draws write, so a
    /// texture is checked once, and again after a draw over it or in the
    /// next list.
    #[test]
    fn textures_are_checked_once_a_list_unless_drawn_over() {
        let mut cache = TextureCache::default();
        let key = (0x1000, TextureFormat::Rgba8, 8, 8);
        cache.begin_list();
        assert!(cache.checked(key, 256).is_none(), "never seen");
        cache.decoded(key, &[0x11; 256]);
        assert!(cache.checked(key, 256).is_some());
        cache.wrote(0x5000, 0x100);
        assert!(cache.checked(key, 256).is_some(), "a draw somewhere else");
        cache.wrote(0x1080, 0x10);
        assert!(cache.checked(key, 256).is_none(), "a draw over it");
        cache.decoded(key, &[0x11; 256]);
        assert!(cache.checked(key, 256).is_some(), "checked again after the draw");
        cache.begin_list();
        assert!(cache.checked(key, 256).is_none(), "a new list");
    }

    /// a few tiles of a texture rewritten, as a title animating sprites in
    /// an atlas, come out the same as the whole texture decoded again.
    #[test]
    fn a_partly_changed_texture_decodes_as_a_whole_one_would() {
        for format in [TextureFormat::Etc1, TextureFormat::Etc1A4, TextureFormat::Rgba8] {
            let (width, height) = (32, 16);
            let size = (width * height * format.bits_per_pixel() / 8) as usize;
            let before: Vec<u8> = (0..size).map(|i| (i * 7 % 251) as u8).collect();
            let mut after = before.clone();
            // the third tile of the second row
            let tile = size / 8;
            after[6 * tile..7 * tile].iter_mut().for_each(|byte| *byte = byte.wrapping_mul(3).wrapping_add(1));
            let key = (0x1000, format, width, height);
            let mut cache = TextureCache::default();
            let old = cache.decoded(key, &before).0;
            let updated = cache.decoded(key, &after).0;
            let whole = TextureCache::default().decoded(key, &after).0;
            assert_eq!(updated, whole, "{format:?}");
            let changed = old.iter().zip(updated.iter()).enumerate().filter(|(_, (a, b))| a != b);
            assert!(changed.clone().count() > 0, "{format:?} changed nothing");
            for (index, _) in changed {
                let (x, y) = (index as u32 % width, index as u32 / width);
                assert_eq!((x / 8, y / 8), (2, 1), "{format:?} changed outside the tile, at {x}, {y}");
            }
        }
    }

    #[test]
    fn the_texture_cache_notices_changed_bytes() {
        let mut cache = TextureCache::default();
        let key = (0x1000, TextureFormat::Rgba8, 8, 8);
        let first = cache.decoded(key, &[0x11; 256]).0;
        let again = cache.decoded(key, &[0x11; 256]).0;
        assert!(Arc::ptr_eq(&first, &again));
        let changed = cache.decoded(key, &[0x22; 256]).0;
        assert_ne!(first[0], changed[0]);

        let mut registers = vec![0u32; 0x300];
        let mut memory = ConsoleMemory::default();
        let base = TEXTURE_UNIT_BASES[0];
        registers[REG_TEXTURE_CONFIG] = 1;
        registers[base + 1] = (16 << 16) | 16;
        registers[base + 4] = 0x1800_1000 >> 3;
        registers[base + 13] = 4; // Etc1 (16x16 = 128B base + 32B mip1 = 160B)
        registers[base + 3] = 1 << 16;
        let mipped = bind_texture(&registers, &mut memory, &mut cache, 0, None, false).unwrap().texels;
        registers[base + 3] = 0;
        let plain = bind_texture(&registers, &mut memory, &mut cache, 0, None, false).unwrap().texels;
        assert_eq!(mipped.len(), 256 + 64);
        assert_eq!(plain.len(), 256);
        registers[base + 3] = 1 << 16;
        let mipped_again = bind_texture(&registers, &mut memory, &mut cache, 0, None, false).unwrap().texels;
        registers[base + 3] = 0;
        let plain_again = bind_texture(&registers, &mut memory, &mut cache, 0, None, false).unwrap().texels;
        assert!(Arc::ptr_eq(&mipped, &mipped_again));
        assert!(Arc::ptr_eq(&plain, &plain_again));
    }

    /// a pixel centered on an edge goes to the triangle on its right, or
    /// above a flat edge, and a corner a hair off a center counts as on it.
    #[test]
    fn edges_through_pixel_centers_follow_the_pica() {
        let registers = target_registers();
        let corner = |x: f32, y: f32| Vertex {
            clip: [x / 4.0 - 1.0, y / 4.0 - 1.0, -0.5, 1.0],
            color: RED,
            texcoords: [[0.0; 2]; 3],
            quaternion: [0.0, 0.0, 0.0, 1.0],
            view: [0.0; 3],
            ..Default::default()
        };
        for (left, right) in [(1.5, 3.5), (1.4999967, 3.499992)] {
            let quad = [(left, 1.5), (right, 1.5), (right, 3.5), (left, 1.5), (right, 3.5), (left, 3.5)];
            let mut memory = ConsoleMemory::default();
            rasterize_shaded(&registers, &mut memory, &mut Resources::default(), &quad.map(|(x, y)| corner(x, y)));
            // the rows as the window counts them, from the bottom
            let covered: Vec<(u32, u32)> = (0..8)
                .flat_map(|y| (0..8).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let mut raw = [0u8; 4];
                    memory.read(COLOR + crate::format::morton_offset(x, 7 - y, 8, 4), &mut raw);
                    raw != [0; 4]
                })
                .collect();
            assert_eq!(covered, [(1, 1), (2, 1), (1, 2), (2, 2)]);
        }
    }

    /// a pixel whose center sits on an edge, or just beside a steep one,
    /// goes to the same triangle on the GPU as in software.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_covers_the_same_pixels() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);

        let mut seed = 1u32;
        let mut random = move |range: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % range
        };
        // corners on a sixteenth of a pixel, which every GPU places exactly,
        // half of them on pixel centers
        let mut coordinate = || {
            let sixteenths = random(SIZE * 16 + 1);
            let sixteenths = if random(2) == 0 { sixteenths / 8 * 8 + 8 } else { sixteenths };
            (sixteenths as f32 / 16.0) / (SIZE as f32 / 2.0) - 1.0
        };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut differing = 0;
        // one at a time over a cleared buffer, so that every edge shows
        for _ in 0..200 {
            let triangle: Vec<Vertex> = (0..3)
                .map(|_| Vertex {
                    clip: [coordinate(), coordinate(), -0.5, 1.0],
                    color: RED,
                    texcoords: [[0.0; 2]; 3],
                    quaternion: [0.0, 0.0, 0.0, 1.0],
                    view: [0.0; 3],
                    ..Default::default()
                })
                .collect();
            let cleared = vec![0u8; (SIZE * SIZE * 4) as usize];
            software.write(COLOR, &cleared);
            gpu.write(COLOR, &cleared);
            rasterize_shaded(&registers, &mut software, &mut Resources::default(), &triangle);
            rasterize_shaded(&registers, &mut gpu, &mut resources, &triangle);
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            differing += (0..SIZE * SIZE)
                .filter(|i| {
                    let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
                    software.read(COLOR + i * 4, &mut a);
                    gpu.read(COLOR + i * 4, &mut b);
                    a != b
                })
                .count();
        }
        assert_eq!(differing, 0);
    }

    /// blending, color masks and logic ops come out on the GPU as in
    /// software, over draws one after another that each do something else,
    /// as draws sharing a pipeline do when the GPU sets that as it goes.
    /// each draw covers a row of its own, so none blends over another's.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_blends_and_masks_like_the_cpu() {
        const REG_BLEND_COLOR: usize = 0x103;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        // a device without logic ops draws without them
        let logic_ops = hardware.logic_ops();
        let mut seed = 7u32;
        let mut random = move |range: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) % range
        };
        let row = |k: u32, color: Vec4| -> Vec<Vertex> {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            let corner = |x: f32, y: f32| Vertex {
                clip: [x, y, -0.5, 1.0],
                color,
                texcoords: [[0.0; 2]; 3],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
                ..Default::default()
            };
            vec![corner(-1.0, bottom), corner(1.0, bottom), corner(1.0, top), corner(-1.0, bottom), corner(1.0, top), corner(-1.0, top)]
        };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for round in 0..40 {
            let start: Vec<u8> = (0..64 * 4).map(|_| random(256) as u8).collect();
            software.write(COLOR, &start);
            gpu.write(COLOR, &start);
            // software blends in floats and truncates, GPUs round, so a
            // blended row can be a step off, a logic op's not at all
            let mut steps = [0u8; 8];
            for (k, steps) in steps.iter_mut().enumerate() {
                let mut registers = target_registers();
                if !logic_ops || random(2) == 0 {
                    let (color, alpha) = (random(6), random(6));
                    let factors = [random(16), random(16), random(16), random(16)];
                    registers[REG_COLOR_OPERATION] = 0x100;
                    registers[REG_BLEND_FUNC] =
                        color | alpha << 8 | factors[0] << 16 | factors[1] << 20 | factors[2] << 24 | factors[3] << 28;
                    registers[REG_BLEND_COLOR] = random(1 << 16) << 16 | random(1 << 16);
                    *steps = 1;
                } else {
                    registers[REG_COLOR_OPERATION] = 0;
                    registers[REG_LOGIC_OP] = random(16);
                }
                registers[REG_DEPTH_COLOR_MASK] = random(16) << 8;
                // a quarter above a byte, which software truncates and the
                // GPU rounds to the same byte
                let color = std::array::from_fn(|_| (random(255) as f32 + 0.25) / 255.0);
                rasterize_shaded(&registers, &mut software, &mut Resources::default(), &row(k as u32, color));
                rasterize_shaded(&registers, &mut gpu, &mut resources, &row(k as u32, color));
            }
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            for (y, &steps) in steps.iter().enumerate() {
                for x in 0..8 {
                    let at = COLOR + crate::format::morton_offset(x, 7 - y as u32, 8, 4);
                    let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
                    software.read(at, &mut a);
                    gpu.read(at, &mut b);
                    let off = a.iter().zip(&b).map(|(p, q)| p.abs_diff(*q)).max().unwrap_or(0);
                    assert!(off <= steps, "round {round}, row {y} is off by {off}, {a:?} against {b:?}");
                }
            }
        }
    }

    /// draws in a row in one batch read their own uniforms, whether they
    /// are the last draw's or not, the ones the combiners take when the CPU
    /// placed the vertices and the vertex shader's when the GPU shades them,
    /// and so does a draw like the last one in the batch before.
    #[cfg(feature = "vulkan")]
    #[test]
    fn draws_in_a_row_read_their_own_uniforms() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        // the position from v0 and the color from c0
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0x00), mov(1, 0x20), 0x22 << 26]);
        unit.prepare();
        let mut registers = target_registers();
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        // the first combiner stage takes the vertex color or its constant,
        // the others hand it on
        for base in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = 0x000F_000F;
        }
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
        // a row each, with the last row's color now and then, the last
        // being the first
        let picks = [1, 0, 1, 0, 2, 2, 1, 1];
        let corners = |k: usize| {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            [(-1.0, bottom), (1.0, bottom), (1.0, top), (-1.0, bottom), (1.0, top), (-1.0, top)]
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for shaded in [false, false, true, true] {
            let mut software = ConsoleMemory::default();
            let mut gpu = ConsoleMemory::default();
            for memory in [&mut software, &mut gpu] {
                memory.write(COLOR, &[0; 64 * 4]);
            }
            for (k, &pick) in picks.iter().enumerate() {
                let color = colors[pick];
                if shaded {
                    registers[0x0C0] = 0;
                    unit.float_uniforms[0] = color;
                    let inputs: Vec<_> = corners(k)
                        .iter()
                        .map(|&(x, y)| {
                            let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
                            input[0] = [x, y, -0.5, 1.0];
                            input
                        })
                        .collect();
                    let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
                    rasterize(&registers, &mut software, &mut Resources::default(), vertices);
                    rasterize(&registers, &mut gpu, &mut resources, vertices);
                } else {
                    registers[0x0C0] = 0x000E_000E;
                    registers[0x0C3] = u32::from_le_bytes(color.map(|c| (c * 255.0) as u8));
                    let vertices: Vec<Vertex> = corners(k)
                        .iter()
                        .map(|&(x, y)| Vertex {
                            clip: [x, y, -0.5, 1.0],
                            color: [1.0; 4],
                            texcoords: [[0.0; 2]; 3],
                            quaternion: [0.0, 0.0, 0.0, 1.0],
                            view: [0.0; 3],
                            ..Default::default()
                        })
                        .collect();
                    rasterize_shaded(&registers, &mut software, &mut Resources::default(), &vertices);
                    rasterize_shaded(&registers, &mut gpu, &mut resources, &vertices);
                }
            }
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            let (mut a, mut b) = ([0u8; 64 * 4], [0u8; 64 * 4]);
            software.read(COLOR, &mut a);
            gpu.read(COLOR, &mut b);
            assert!(a.chunks(4).all(|pixel| pixel != [0; 4]), "every pixel drawn, shaded {shaded}");
            assert!(a == b, "shaded {shaded}, {a:?} against {b:?}");
        }
    }

    /// draws big enough that the ring passes RING_FLUSH, so the batch is
    /// handed over at the start of a draw whose uniform and shading words
    /// equal the draw before's, still read their own words.
    #[cfg(feature = "vulkan")]
    #[test]
    fn uniforms_survive_a_ring_flush_between_draws() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0x00), mov(1, 0x20), 0x22 << 26]);
        // left unprepared, so all 16 input registers are staged, 256 bytes
        // a vertex, and 65536 vertices take 16 MB of the ring each draw
        let mut registers = target_registers();
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        for base in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = 0x000F_000F;
        }
        // the vertex color times a constant, which only the uniform block
        // carries, so a block read from the wrong place shows
        registers[0x0C0] = 0x00E0_00E0;
        registers[0x0C2] = 0x0001_0001;
        registers[0x0C3] = 0xFF80_40C0;
        let colors = [[1.0, 0.5, 0.25, 1.0], [0.5, 1.0, 0.75, 1.0], [0.25, 0.5, 1.0, 1.0]];
        // runs of the same color, so a draw after a flush has the words of
        // the one before it
        let picks = [1, 1, 1, 0, 0, 0, 2, 2];
        let corners = |k: usize| {
            let (bottom, top) = (k as f32 / 4.0 - 1.0, (k + 1) as f32 / 4.0 - 1.0);
            [(-1.0, bottom), (1.0, bottom), (1.0, top), (-1.0, bottom), (1.0, top), (-1.0, top)]
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        for memory in [&mut software, &mut gpu] {
            memory.write(COLOR, &[0; 64 * 4]);
        }
        let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 65536];
        let order: Vec<usize> = (0..6).collect();
        for (k, &pick) in picks.iter().enumerate() {
            unit.float_uniforms[0] = colors[pick];
            unit.sync_wide_uniforms();
            for (input, &(x, y)) in inputs.iter_mut().zip(&corners(k)) {
                input[0] = [x, y, -0.5, 1.0];
            }
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: Some(&order) };
            rasterize(&registers, &mut software, &mut Resources::default(), vertices);
            assert!(rasterize(&registers, &mut gpu, &mut resources, vertices).is_some());
        }
        resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
        let (mut a, mut b) = ([0u8; 64 * 4], [0u8; 64 * 4]);
        software.read(COLOR, &mut a);
        gpu.read(COLOR, &mut b);
        assert!(a.chunks(4).all(|pixel| pixel != [0; 4]), "every pixel drawn");
        assert!(a == b, "{a:?} against {b:?}");
    }

    /// the GPU has the CPU's reads of the depth it draws ask for it first,
    /// once until guest memory gets it, and again after. the color buffer
    /// the same draws go to is read as memory holds it. writes over either
    /// ask the same way.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_guards_what_it_draws() {
        struct Guarded<'a>(&'a mut ConsoleMemory, Vec<(u32, u32)>, Vec<(u32, u32)>);
        impl GpuMemory for Guarded<'_> {
            fn read(&mut self, addr: u32, out: &mut [u8]) {
                self.0.read(addr, out)
            }
            fn write(&mut self, addr: u32, data: &[u8]) {
                self.0.write(addr, data)
            }
            fn guard(&mut self, addr: u32, len: u32) {
                self.1.push((addr, len))
            }
            fn guard_writes(&mut self, addr: u32, len: u32) {
                self.2.push((addr, len))
            }
        }
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // depth written, with no test
        registers[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        let mut memory = ConsoleMemory::default();
        let mut guarded = Guarded(&mut memory, Vec::new(), Vec::new());
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        // the whole 8 by 8 depth buffer, of four bytes a pixel
        assert_eq!(guarded.1, [(DEPTH, 8 * 8 * 4)]);
        assert_eq!(guarded.2, [(COLOR, 8 * 8 * 4), (DEPTH, 8 * 8 * 4)]);
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        assert_eq!((guarded.1.len(), guarded.2.len()), (1, 2));
        resources.hardware.as_mut().unwrap().flush(&mut guarded).unwrap();
        // depth memory the next draw writes over
        guarded.0.write(DEPTH, &[0xFF; 8 * 8 * 4]);
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, GREEN));
        assert_eq!((guarded.1.len(), guarded.2.len()), (2, 4));

        // what the CPU's read brings back is the depth, the color stays as
        // memory had it, red from the flush
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.sync_depth(&mut memory, COLOR, DEPTH + 8 * 8 * 4 - COLOR).unwrap();
        assert!(pixels(&mut memory).iter().all(|&p| p == [255, 0, 0, 255]));
        let mut depth = [0u8; 8 * 8 * 4];
        memory.read(DEPTH, &mut depth);
        // the stencil byte stays, nothing wrote it
        assert!(depth.as_chunks::<4>().0.iter().all(|sample| sample[..3] != [0xFF; 3]), "the depth drawn came back");
    }

    /// fog mixes into what the combiners made by the fragment's depth, on
    /// the GPU as the software path does it.
    #[test]
    fn fog_mixes_in_by_depth() {
        let mut registers = target_registers();
        // a z-buffer, depth 0.5 at clip z -0.5
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_VIEWPORT_DEPTH_NEAR] = float24(0.0);
        registers[crate::fog::REG_MODE] |= 5;
        registers[crate::fog::REG_COLOR] = 0x00_80_40_20;
        let mut resources = Resources::default();
        // a factor falling from 1 to 0 across the table, a half at 0.5
        for entry in 0..crate::fog::ENTRIES as u32 {
            let factor = (2048 - entry * 16).min(2047);
            resources.fog_table.write(&mut registers, (factor << 13) | (0x2000 - 16));
        }
        let mut memory = ConsoleMemory::default();
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        assert!(pixels(&mut memory).iter().all(|&p| p == [143, 32, 64, 255]), "{:?}", pixels(&mut memory)[0]);

        #[cfg(feature = "vulkan")]
        if let Ok(hardware) = hardware::Hardware::new() {
            let mut gpu = ConsoleMemory::default();
            resources.hardware = Some(hardware);
            rasterize_shaded(&registers, &mut gpu, &mut resources, &cover(-0.5, RED));
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            assert!(pixels(&mut gpu).iter().all(|&p| p == [143, 32, 64, 255]), "{:?}", pixels(&mut gpu)[0]);
        }
    }

    /// the target at another size, the viewport over all of it.
    fn sized(registers: &mut [u32], width: u32, height: u32) {
        registers[REG_VIEWPORT_WIDTH] = float24(width as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(height as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = width | ((height - 1) << 12);
    }

    /// a fill over every row the GPU drew leaves the buffer as memory holds
    /// it, and the next draw there has the CPU's writes ask first again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_over_all_that_was_drawn_guards_the_next_draw_again() {
        struct Guarded<'a>(&'a mut ConsoleMemory, Vec<(u32, u32)>);
        impl GpuMemory for Guarded<'_> {
            fn read(&mut self, addr: u32, out: &mut [u8]) {
                self.0.read(addr, out)
            }
            fn write(&mut self, addr: u32, data: &[u8]) {
                self.0.write(addr, data)
            }
            fn guard_writes(&mut self, addr: u32, len: u32) {
                self.1.push((addr, len))
            }
        }
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        sized(&mut registers, 8, 16);
        // the window's top half, the first rows of memory
        registers[REG_VIEWPORT_XY] = 8 << 16;
        registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
        let mut memory = ConsoleMemory::default();
        let mut guarded = Guarded(&mut memory, Vec::new());
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, RED));
        assert_eq!(guarded.1, [(COLOR, 256)]);
        let fill = [0u8; 256];
        guarded.0.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&registers, &mut guarded, &mut resources, &cover(-0.5, GREEN));
        assert_eq!(guarded.1, [(COLOR, 256), (COLOR, 256)]);
    }

    /// a buffer drawn over rows a fill left of a taller one, which still has
    /// other rows drawn, is the one a transfer of those rows reads.
    #[cfg(feature = "vulkan")]
    #[test]
    fn rows_drawn_after_a_fill_come_from_the_buffer_that_drew_them() {
        const OUTPUT: u32 = 0x10_0000;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut tall = target_registers();
        sized(&mut tall, 8, 16);
        tall[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        let mut short = target_registers();
        // a depth buffer of its own keeps it apart from the taller one
        short[REG_DEPTH_COLOR_MASK] |= 1 << 12;
        short[REG_DEPTH_BUFFER_ADDRESS] = 0x5000 >> 3;
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&tall, &mut memory, &mut resources, &cover(-0.5, RED));
        let fill = [0u8; 256];
        memory.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&short, &mut memory, &mut resources, &cover(-0.5, GREEN));
        let hardware = resources.hardware.as_mut().unwrap();
        // a texture of those rows is copied from the shorter buffer on the
        // GPU, one over both comes from memory
        let texture = |height| DrawnTexture { addr: COLOR, width: 8, height, format: ColorFormat::Rgba8 };
        assert!(hardware.holds(&texture(8)));
        assert!(!hardware.holds(&texture(16)));
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 8,
            output_width: 8,
            output_height: 8,
            copy: (8, 8),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
        hardware.flush(&mut memory).unwrap();
        let mut out = [0u8; 256];
        memory.read(OUTPUT, &mut out);
        assert!(out.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [0, 255, 0, 255]));
    }

    /// memory written over rows a fill cleared of a buffer still drawn
    /// elsewhere, a DMA after the GPU was synced, is what a texture of those
    /// rows shows, and what the buffer holds once used again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn rows_written_after_a_fill_reach_textures_and_the_buffer() {
        const OUTPUT: u32 = 0x10_0000;
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut registers = target_registers();
            sized(&mut registers, 8, 16);
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            let fill = [0u8; 256];
            memory.write(COLOR, &fill);
            let hardware = resources.hardware.as_mut().unwrap();
            hardware.filled(COLOR, &fill).unwrap();
            hardware.sync(&mut memory, COLOR, 256).unwrap();
            let mut green = [0u8; 4];
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            memory.write(COLOR, &green.repeat(64));
            let texture = DrawnTexture { addr: COLOR, width: 8, height: 8, format: ColorFormat::Rgba8 };
            assert!(!hardware.holds(&texture), "the texture comes from memory");
            // the rest of the drawing comes down, and the buffer is used again
            hardware.sync(&mut memory, COLOR + 256, 256).unwrap();
            let transfer = hardware::Transfer {
                input: COLOR,
                output: OUTPUT,
                input_width: 8,
                output_width: 8,
                output_height: 16,
                copy: (8, 16),
                scale: (1, 1),
                flip: false,
                input_linear: false,
                output_tiled: false,
                input_format: ColorFormat::Rgba8,
                output_format: ColorFormat::Rgba8,
            };
            assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
            hardware.flush(&mut memory).unwrap();
            let mut out = [0u8; 512];
            memory.read(OUTPUT, &mut out);
            let pixels: Vec<[u8; 4]> = out.as_chunks::<4>().0.iter().map(|p| ColorFormat::Rgba8.decode(p)).collect();
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 64, "the rows written are green");
            assert_eq!(pixels.iter().filter(|&&p| p == [255, 0, 0, 255]).count(), 64, "the rows drawn stay red");
        }
    }

    /// a narrower buffer drawn over rows a fill left of a wider one does not
    /// cost the wider one the rows it drew elsewhere, which memory lacks,
    /// when it is used again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_buffer_used_again_keeps_the_rows_it_drew() {
        const OUTPUT: u32 = 0x10_0000;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut wide = target_registers();
        sized(&mut wide, 16, 16);
        let narrow = target_registers();
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&wide, &mut memory, &mut resources, &cover(-0.5, RED));
        // its first row of tiles, 8 rows of 16 pixels
        let fill = [0u8; 512];
        memory.write(COLOR, &fill);
        resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
        rasterize_shaded(&narrow, &mut memory, &mut resources, &cover(-0.5, GREEN));
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 16,
            output_width: 16,
            output_height: 16,
            copy: (16, 16),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
        hardware.sync(&mut memory, COLOR + 512, 512).unwrap();
        let mut rest = [0u8; 512];
        memory.read(COLOR + 512, &mut rest);
        assert!(rest.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rows it drew stay red");
    }

    /// a fill over the first rows of a buffer the GPU drew leaves only the
    /// rest of the drawing to come down, so a buffer of another shape over
    /// the filled rows does not have it written back first.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_leaves_only_the_rest_of_the_drawing() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // eight by sixteen, two rows of tiles
        registers[REG_VIEWPORT_HEIGHT] = float24(8.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (15 << 12);
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let fill = [0x11u8, 0x22, 0x33, 0x44].repeat(64);
        memory.write(COLOR, &fill);
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.filled(COLOR, &fill).unwrap();
        let mut rest = [0u8; 256];
        hardware.sync(&mut memory, COLOR, 256).unwrap();
        memory.read(COLOR + 256, &mut rest);
        assert_eq!(rest, [0; 256], "nothing over the filled rows had to come down");
        hardware.sync(&mut memory, COLOR + 256, 256).unwrap();
        memory.read(COLOR + 256, &mut rest);
        assert!(rest.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rest of the drawing does");
        let mut first = [0u8; 256];
        memory.read(COLOR, &mut first);
        assert_eq!(first[..], fill[..], "and the fill stays");
    }

    /// a fill over the start of a buffer the GPU drew, in a pattern its
    /// pixels can't take, does not have the drawing come down then, which
    /// waits for the GPU. once it does come down, the bytes the fill wrote
    /// stay, even ones the fill left as memory held them.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_fill_over_part_of_a_drawing_leaves_the_rest_on_the_gpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&target_registers(), &mut memory, &mut resources, &cover(-0.5, RED));
        // a three byte pattern of zeros, as memory held them, over 100 bytes
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.before_fill(&mut memory, COLOR, 100, 3).unwrap();
        memory.write(COLOR, &[0; 100]);
        hardware.filled(COLOR, &[0; 100]).unwrap();
        let mut all = [0u8; 256];
        memory.read(COLOR, &mut all);
        assert_eq!(all, [0; 256], "the drawing came down for the fill");
        hardware.sync(&mut memory, COLOR, 256).unwrap();
        memory.read(COLOR, &mut all);
        assert_eq!(all[..100], [0; 100], "the fill stays");
        assert!(all[100..].as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]), "the rest of the drawing comes down");
    }

    /// a texture reaching past the rows of the buffer drawn into it, as its
    /// sides are powers of two, samples the same from a copy the GPU makes,
    /// without the buffer coming down first, as from memory once it has the
    /// buffer, the rows past the buffer as memory holds them either way.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_texture_reaching_past_a_drawing_samples_like_memory() {
        let (Ok(first), Ok(second)) = (hardware::Hardware::new(), hardware::Hardware::new()) else { return };
        const SIZE: u32 = 32;
        const TARGET: u32 = 0x10_0000;
        // a buffer half as tall as the texture drawn into
        let mut drawing = target_registers();
        drawing[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        drawing[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 4.0);
        drawing[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE / 2 - 1) << 12);
        let mut sampling = target_registers();
        sampling[REG_COLOR_BUFFER_ADDRESS] = TARGET >> 3;
        sampling[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        sampling[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        sampling[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        sampling[REG_TEXTURE_CONFIG] = 1;
        sampling[REG_TEXTURE0_DIMENSIONS] = SIZE | (SIZE << 16);
        sampling[REG_TEXTURE0_ADDRESS] = COLOR >> 3;
        sampling[0x0C0] = 0x3 | (0x3 << 16);
        for stage in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            sampling[stage] = 0xF | (0xF << 16);
        }

        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let triangles: Vec<Vec<Vertex>> = (0..30)
            .map(|_| {
                let color = [random(), random(), random(), 1.0];
                (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -random(), 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        // the rows past the buffer, whatever memory holds there
        let past: Vec<u8> = (0..SIZE * SIZE / 2 * 4).map(|_| (random() * 256.0) as u8).collect();
        let quad: Vec<Vertex> = [[-1.0, -1.0, 0.0, 0.0], [3.0, -1.0, 2.0, 0.0], [-1.0, 3.0, 0.0, 2.0]]
            .into_iter()
            .map(|[x, y, u, v]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [1.0; 4],
                texcoords: [[u, v], [0.0; 2], [0.0; 2]],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
                ..Default::default()
            })
            .collect();

        let mut results = Vec::new();
        for (hardware, back_first) in [(first, false), (second, true)] {
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            memory.write(COLOR + SIZE * SIZE / 2 * 4, &past);
            for triangle in &triangles {
                rasterize_shaded(&drawing, &mut memory, &mut resources, triangle);
            }
            if back_first {
                resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            }
            rasterize_shaded(&sampling, &mut memory, &mut resources, &quad);
            if !back_first {
                let mut drawn = vec![0u8; (SIZE * SIZE / 2 * 4) as usize];
                memory.read(COLOR, &mut drawn);
                assert!(drawn.iter().all(|&b| b == 0), "the drawing came down for the texture");
            }
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut sampled = vec![0u8; (SIZE * SIZE * 4) as usize];
            memory.read(TARGET, &mut sampled);
            results.push(sampled);
        }
        assert!(results[0] == results[1], "the copy samples as memory does");
        let colors: std::collections::HashSet<&[u8]> = results[0].chunks(4).collect();
        assert!(colors.len() > 8, "only {} colors", colors.len());
    }

    /// a transfer out of the last rows of a buffer the GPU drew stays on
    /// the GPU when its output lies over the buffer's first rows, which it
    /// does not read, as Monster Hunter 3 Ultimate's bottom screen does.
    /// over rows it reads, the CPU does it in order.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_transfer_over_rows_it_does_not_read_stays_on_the_gpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let mut registers = target_registers();
        // eight by sixteen, two rows of tiles
        registers[REG_VIEWPORT_HEIGHT] = float24(8.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = 8 | (15 << 12);
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = |input: u32, output: u32| hardware::Transfer {
            input,
            output,
            input_width: 8,
            output_width: 8,
            output_height: 8,
            copy: (8, 8),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        // the second row of tiles to a plain buffer over the first
        assert!(hardware.display_transfer(&mut memory, &transfer(COLOR + 256, COLOR)).unwrap(), "the CPU did it");
        hardware.flush(&mut memory).unwrap();
        let mut out = [0u8; 256];
        memory.read(COLOR, &mut out);
        assert!(out.as_chunks::<4>().0.iter().all(|p| ColorFormat::Rgba8.decode(p) == [255, 0, 0, 255]));
        // over the rows it reads
        assert!(!hardware.display_transfer(&mut memory, &transfer(COLOR + 256, COLOR + 384)).unwrap());
    }

    /// a transfer on the GPU of an 8 pixel wide buffer at COLOR, of rows
    /// pixels, to a plain one at OUTPUT, and the pixels it leaves there.
    #[cfg(feature = "vulkan")]
    fn transferred(memory: &mut ConsoleMemory, resources: &mut Resources, rows: u32) -> Vec<[u8; 4]> {
        const OUTPUT: u32 = 0x10_0000;
        let hardware = resources.hardware.as_mut().unwrap();
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: 8,
            output_width: 8,
            output_height: rows,
            copy: (8, rows),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: ColorFormat::Rgba8,
            output_format: ColorFormat::Rgba8,
        };
        assert!(hardware.display_transfer(memory, &transfer).unwrap());
        hardware.flush(memory).unwrap();
        let mut out = vec![0u8; 8 * rows as usize * 4];
        memory.read(OUTPUT, &mut out);
        out.as_chunks::<4>().0.iter().map(|p| ColorFormat::Rgba8.decode(p)).collect()
    }

    /// a pixel written over a buffer a fill cleared, as the CPU writes one,
    /// is in the buffer the next time it is used, the fill's pixel standing
    /// in for its shadow notices it as the bytes would.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_pixel_written_after_a_fill_reaches_the_buffer() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&target_registers(), &mut memory, &mut resources, &cover(-0.5, RED));
            let (mut blue, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([0, 0, 255, 255], &mut blue);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            let fill = blue.repeat(64);
            memory.write(COLOR, &fill);
            resources.hardware.as_mut().unwrap().filled(COLOR, &fill).unwrap();
            memory.write(COLOR + 4 * 37, &green);
            let pixels = transferred(&mut memory, &mut resources, 8);
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 1, "the pixel written");
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 0, 255, 255]).count(), 63, "and the fill");
        }
    }

    /// memory of one pixel but for one past the first few kilobytes is not
    /// kept as that pixel, the drawing over it reaches memory when written
    /// back, rather than looking like a write made after it.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_drawing_over_a_pixel_past_the_first_group_reaches_memory() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut registers = target_registers();
            sized(&mut registers, 64, 64);
            registers[REG_DEPTH_BUFFER_ADDRESS] = 0x8_0000 >> 3;
            let (mut gray, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([128, 128, 128, 255], &mut gray);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            let size = 64 * 64 * 4;
            memory.write(COLOR, &gray.repeat(64 * 64));
            let at = size as u32 - 4 * 3;
            memory.write(COLOR + at, &green);
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut all = vec![0u8; size];
            memory.read(COLOR, &mut all);
            let red = all.as_chunks::<4>().0.iter().filter(|p| ColorFormat::Rgba8.decode(&p[..]) == [255, 0, 0, 255]).count();
            assert_eq!(red, 64 * 64, "the drawing reached all of memory, the pixel past the first group too");
        }
    }

    /// a buffer over memory all of one pixel, which its shadow keeps as that
    /// pixel, has a pixel changed there the next time it is used, with the
    /// rows it drew.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_pixel_changed_in_memory_of_one_pixel_reaches_the_buffer() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let mut registers = target_registers();
            sized(&mut registers, 8, 16);
            // the window's top half, the first rows of memory
            registers[REG_VIEWPORT_XY] = 8 << 16;
            registers[REG_VIEWPORT_HEIGHT] = float24(4.0);
            let (mut gray, mut green) = ([0u8; 4], [0u8; 4]);
            ColorFormat::Rgba8.encode([128, 128, 128, 255], &mut gray);
            ColorFormat::Rgba8.encode([0, 255, 0, 255], &mut green);
            memory.write(COLOR, &gray.repeat(128));
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            // a sync of rows it did not draw, which has it look at memory again
            resources.hardware.as_mut().unwrap().sync(&mut memory, COLOR + 256, 256).unwrap();
            memory.write(COLOR + 256 + 4 * 5, &green);
            let pixels = transferred(&mut memory, &mut resources, 16);
            assert_eq!(pixels.iter().filter(|&&p| p == [255, 0, 0, 255]).count(), 64, "the rows drawn");
            assert_eq!(pixels.iter().filter(|&&p| p == [0, 255, 0, 255]).count(), 1, "the pixel written");
            assert_eq!(pixels.iter().filter(|&&p| p == [128, 128, 128, 255]).count(), 63, "and what memory held");
        }
    }

    /// memory written after the GPU drew over it by something that does not
    /// wait for the drawing stays when the drawing is written back.
    #[cfg(feature = "vulkan")]
    #[test]
    fn writes_after_drawing_survive_the_write_back() {
        for mut memory in [ConsoleMemory::default(), ConsoleMemory::scattered()] {
            let Ok(hardware) = hardware::Hardware::new() else { return };
            let registers = target_registers();
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
            memory.write(COLOR, &[0x5A; 16]);
            resources.hardware.as_mut().unwrap().sync(&mut memory, COLOR, 8 * 8 * 4).unwrap();
            let mut start = [0u8; 16];
            memory.read(COLOR, &mut start);
            assert_eq!(start, [0x5A; 16], "what was written after the drawing stays");
            assert!(pixels(&mut memory)[4..].iter().all(|&p| p == [255, 0, 0, 255]), "and the drawing is the rest");
        }
    }

    /// a service that has the drawing written back before it writes, as a
    /// file read does, keeps every byte it wrote, those equal to what memory
    /// held under the drawing too, when a write of the CPU later asks for
    /// the drawing again.
    #[cfg(feature = "vulkan")]
    #[test]
    fn writes_after_the_drawing_came_down_stay() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        let registers = target_registers();
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        rasterize_shaded(&registers, &mut memory, &mut resources, &cover(-0.5, RED));
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.sync(&mut memory, COLOR, 16).unwrap();
        // zeros, as memory held before the drawing
        memory.write(COLOR, &[0; 16]);
        hardware.sync(&mut memory, COLOR, 8 * 8 * 4).unwrap();
        let mut start = [0xFFu8; 16];
        memory.read(COLOR, &mut start);
        assert_eq!(start, [0; 16], "what the service wrote stays");
        assert!(pixels(&mut memory)[4..].iter().all(|&p| p == [255, 0, 0, 255]), "and the drawing is the rest");
    }

    /// vertices the GPU shades come out as the CPU shades them, clipped and
    /// culled the same, through a program with a loop, a call, both kinds
    /// of if, indexed uniforms and an output past a gap in the mask.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_shades_like_the_cpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        let descriptors = [
            0xF | IDENTITY,
            0x8 | IDENTITY,
            0x4 | IDENTITY,
            0x2 | IDENTITY,
            0x1 | IDENTITY,
            // w, a.w times b.x plus c.y
            0x1 | 0x1B << 5 | 0x55 << 23,
            // x, the first source's z
            0x8 | 0xAA << 5,
            0xE | IDENTITY,
            // the second source's x everywhere
            0xF | 0x1B << 5,
        ];
        unit.descriptors[..descriptors.len()].copy_from_slice(&descriptors);
        let op = |opcode: u32, destination: u32, src1: u32, src2: u32, index: u32, descriptor: u32| {
            opcode << 26 | destination << 21 | index << 19 | src1 << 12 | src2 << 7 | descriptor
        };
        let flow = |opcode: u32, condition: u32, destination: u32, count: u32| {
            opcode << 26 | condition << 22 | destination << 10 | count
        };
        let program = [
            op(0x12, 0, 0x02, 0, 0, 1),
            // the matrix at c1 through a0.x, which v2 sets to one
            op(0x02, 0, 0x20, 0x00, 1, 1),
            op(0x02, 0, 0x21, 0x00, 1, 2),
            op(0x02, 0, 0x22, 0x00, 1, 3),
            op(0x02, 0, 0x23, 0x00, 1, 4),
            op(0x13, 0x10, 0x2A, 0, 0, 0),
            // r0 is c5 + c6 + c7
            flow(0x29, 0, 7, 0),
            op(0x00, 0x10, 0x25, 0x10, 3, 0),
            flow(0x27, 0, 10, 1),
            op(0x08, 2, 0x01, 0x10, 0, 0),
            op(0x13, 2, 0x01, 0, 0, 0),
            // c8.x < v1.x and c8.y > v1.y
            0x2E << 26 | 2 << 24 | 4 << 21 | 0x28 << 12 | 0x01 << 7,
            flow(0x28, 0b1010, 14, 0),
            0x38 << 26 | 2 << 24 | 0x01 << 17 | 0x29 << 10 | 0x10 << 5 | 5,
            flow(0x24, 0, 20, 3),
            0x22 << 26,
        ];
        unit.program[..program.len()].copy_from_slice(&program);
        unit.program[20] = op(0x0E, 0x11, 0x29, 0, 0, 6);
        unit.program[21] = op(0x08, 0x12, 0x01, 0x11, 0, 8);
        unit.program[22] = op(0x0D, 2, 0x12, 0x10, 0, 7);
        let uniforms = [
            [0.0; 4],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.3, 1.0],
            [0.5, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.1, 0.3],
            [0.1, 0.2, 0.4, 0.4],
            [0.5, 0.5, 0.0, 0.0],
            [0.3, 0.0, 2.0, 0.0],
            [0.0; 4],
        ];
        unit.float_uniforms[..uniforms.len()].copy_from_slice(&uniforms);
        unit.int_uniforms[0] = [2, 0, 1, 0];
        unit.bool_uniforms = 1;
        unit.prepare();

        let mut registers = target_registers();
        registers[REG_VIEWPORT_XY] = 4 | 6 << 16;
        registers[REG_VIEWPORT_WIDTH] = float24(12.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(12.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | 4 << 4 | 1 << 12;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b101;

        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let (mut differing, mut drawn, mut deepest) = (0, 0, 0);
        for case in 0..36 {
            let (topology, cull) = (case % 3, case / 3 % 3);
            // half the cases reach past the near and far planes, where the
            // CPU places the vertices clipping makes on a sixteenth of a
            // pixel and the GPU the ones it clipped, which moves depth a bit
            let clipped = case >= 18;
            let (near, depth_range) = if clipped { (-1.3, 1.6) } else { (-0.7, 0.6) };
            registers[REG_PRIMITIVE_CONFIG] = topology << 8;
            registers[REG_FACE_CULLING] = cull;
            let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 24];
            for input in &mut inputs {
                input[0] = [random() * 2.6 - 1.3, random() * 2.6 - 1.3, random() * depth_range + near, 1.0];
                input[1] = [random(), random(), random(), random()];
                input[2] = [1.0, 0.0, 0.0, 0.0];
            }
            let order: Vec<usize> = (0..12).map(|_| (random() * 24.0) as usize).collect();
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: Some(&order) };

            let mut software = ConsoleMemory::default();
            let mut gpu = ConsoleMemory::default();
            for memory in [&mut software, &mut gpu] {
                memory.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
                memory.write(DEPTH, &vec![0xFFu8; (SIZE * SIZE * 4) as usize]);
            }
            rasterize(&registers, &mut software, &mut Resources::default(), vertices);
            rasterize(&registers, &mut gpu, &mut resources, vertices);
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            for i in 0..SIZE * SIZE {
                let (mut a, mut b, mut da, mut db) = ([0u8; 4], [0u8; 4], [0u8; 4], [0u8; 4]);
                software.read(COLOR + i * 4, &mut a);
                gpu.read(COLOR + i * 4, &mut b);
                software.read(DEPTH + i * 4, &mut da);
                gpu.read(DEPTH + i * 4, &mut db);
                drawn += (a != [0; 4]) as u32;
                if a.iter().zip(&b).any(|(a, b)| a.abs_diff(*b) > 3) {
                    differing += 1;
                } else if !clipped {
                    let depth = |d: [u8; 4]| u32::from_le_bytes(d) & 0xFF_FFFF;
                    deepest = deepest.max(depth(da).abs_diff(depth(db)));
                }
            }
        }
        eprintln!("{differing} of {drawn} drawn pixels differ, depth by up to {deepest}");
        assert!(drawn > 2000);
        assert!(differing * 100 < drawn, "{differing} of {drawn} pixels differ");
        assert!(deepest < 1 << 8, "depth differs by {deepest}");
    }

    /// a geometry shader making a quad around each point, its second
    /// triangle only when a bool says, draws the same on the GPU, its
    /// slots run through the vertex and geometry programs, as on the CPU.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_runs_geometry_shaders_like_the_cpu() {
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let op = |opcode: u32, destination: u32, src1: u32, src2: u32| opcode << 26 | destination << 21 | src1 << 12 | src2 << 7;
        let set_emit = |slot: u32, completes: u32| 0x2B << 26 | slot << 24 | completes << 23;
        let emit = 0x2A << 26;
        let end = 0x22 << 26;

        let mut vertex = ShaderUnit::new();
        vertex.descriptors[0] = 0xF | IDENTITY;
        let program = [op(0x13, 0, 0x00, 0), op(0x13, 1, 0x01, 0), end];
        vertex.program[..program.len()].copy_from_slice(&program);
        vertex.prepare();

        let mut geometry = ShaderUnit::new();
        geometry.descriptors[0] = 0xF | IDENTITY;
        let program = [
            // o1 the color, o0 the point moved by c0, c1 and c2 in turn
            op(0x13, 1, 0x01, 0),
            op(0x00, 0, 0x20, 0x00),
            set_emit(0, 0),
            emit,
            op(0x00, 0, 0x21, 0x00),
            set_emit(1, 0),
            emit,
            op(0x00, 0, 0x22, 0x00),
            set_emit(2, 1),
            emit,
            // the second triangle, with c3 in the first slot, when b0 is set
            0x27 << 26 | 0 << 22 | 14 << 10,
            op(0x00, 0, 0x23, 0x00),
            set_emit(0, 1),
            emit,
            end,
        ];
        geometry.program[..program.len()].copy_from_slice(&program);
        geometry.float_uniforms[..4].copy_from_slice(&[
            [-0.2, -0.2, 0.0, 0.0],
            [0.2, -0.2, 0.0, 0.0],
            [-0.2, 0.2, 0.0, 0.0],
            [0.2, 0.2, 0.0, 0.0],
        ]);
        geometry.prepare();

        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(16.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(16.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        registers[REG_VS_OUTPUT_TOTAL] = 1;
        registers[REG_GEOSTAGE_CONFIG] = 2;
        registers[REG_GS_CONFIG] = 0;
        registers[REG_GS_BLOCK + SHADER_INPUT_CONFIG] = 1;
        registers[REG_GS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x10;
        registers[REG_GS_BLOCK + SHADER_OUTPUT_MASK] = 0b11;
        registers[REG_PRIMITIVE_CONFIG] = 3 << 8;

        let mut seed = 11u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let (mut differing, mut drawn) = (0, 0);
        for case in 0..8 {
            geometry.bool_uniforms = (case % 2) as u16;
            registers[REG_FACE_CULLING] = case / 2 % 3;
            let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 6];
            for input in &mut inputs {
                input[0] = [random() * 1.6 - 0.8, random() * 1.6 - 0.8, -random() * 0.5, 1.0];
                input[1] = [random(), random(), random(), 1.0];
            }
            let order: Vec<usize> = (0..5).map(|_| (random() * 6.0) as usize).collect();
            let vertices = Vertices::Unshaded { vertex_shader: &vertex, geometry_shader: &geometry, inputs: &inputs, order: Some(&order) };
            let mut software = ConsoleMemory::default();
            let mut gpu = ConsoleMemory::default();
            for memory in [&mut software, &mut gpu] {
                memory.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
                memory.write(DEPTH, &vec![0xFFu8; (SIZE * SIZE * 4) as usize]);
            }
            rasterize(&registers, &mut software, &mut Resources::default(), vertices);
            let before = resources.hardware.as_ref().unwrap().geometry_draws;
            rasterize(&registers, &mut gpu, &mut resources, vertices);
            assert_eq!(resources.hardware.as_ref().unwrap().geometry_draws, before + 1, "case {case} went to the CPU");
            resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
            for i in 0..SIZE * SIZE {
                let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
                software.read(COLOR + i * 4, &mut a);
                gpu.read(COLOR + i * 4, &mut b);
                drawn += (a != [0; 4]) as u32;
                if a.iter().zip(&b).any(|(a, b)| a.abs_diff(*b) > 3) {
                    differing += 1;
                }
            }
        }
        eprintln!("{differing} of {drawn} drawn pixels differ");
        assert!(drawn > 500, "{drawn} pixels drawn");
        assert!(differing * 100 < drawn, "{differing} of {drawn} pixels differ");

        // fixed-primitive mode (REG_GS_CONFIG mode 2) with a backward-jump loop
        let mut fixed_gs = ShaderUnit::new();
        fixed_gs.descriptors[0] = 0xF | IDENTITY;
        let fixed_program = [
            // 0: r0 = c4 (loop counter = 0), 1: r1 = c7 (third corner offset)
            op(0x13, 0x10, 0x24, 0),
            op(0x13, 0x11, 0x27, 0),
            // 2: cmp.x = (c6.x <= r0.x) (mode 3 = LessEqual)
            (0x2E << 26) | (3 << 24) | (0x26 << 12) | (0x10 << 7),
            // 3: jmpc cmp.x -> 15 (end)
            (0x2C << 26) | (1 << 25) | (2 << 22) | (15 << 10),
            // 4..12: emit triangle from fixed vertices at c8..c11 offset by r0 / r1
            op(0x13, 1, 0x29, 0),
            op(0x00, 0, 0x28, 0x10),
            set_emit(0, 0),
            emit,
            op(0x00, 0, 0x2A, 0x10),
            set_emit(1, 0),
            emit,
            op(0x00, 0, 0x28, 0x11),
            set_emit(2, 1),
            emit,
            // 13: r0 = c5 + r0
            op(0x00, 0x10, 0x25, 0x10),
            // 14: jmpu !b0 -> 2
            (0x2D << 26) | (2 << 10) | 1,
            // 15: end
            end,
        ];
        fixed_gs.program[..fixed_program.len()].copy_from_slice(&fixed_program);
        fixed_gs.float_uniforms[4] = [0.0, 0.0, 0.0, 0.0];
        fixed_gs.float_uniforms[5] = [0.25, 0.0, 0.0, 0.0];
        fixed_gs.float_uniforms[6] = [0.4, 0.0, 0.0, 0.0];
        fixed_gs.float_uniforms[7] = [0.0, 0.6, 0.0, 0.0];
        fixed_gs.sync_wide_uniforms();
        fixed_gs.prepare();
        registers[REG_GS_CONFIG] = 2 | (1 << 8) | (8 << 16);
        registers[REG_FACE_CULLING] = 0;
        let inputs = [
            [[-0.4, -0.3, -0.2, 1.0], [0.9, 0.2, 0.3, 1.0], shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO],
            [[0.3, -0.3, -0.2, 1.0], [0.2, 0.8, 0.4, 1.0], shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO],
            [[-0.2, 0.4, -0.2, 1.0], [0.3, 0.4, 0.9, 1.0], shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO],
            [[0.4, 0.4, -0.2, 1.0], [0.8, 0.8, 0.2, 1.0], shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO, shader::ZERO],
        ];
        let vertices = Vertices::Unshaded { vertex_shader: &vertex, geometry_shader: &fixed_gs, inputs: &inputs, order: None };
        let mut software = ConsoleMemory::default();
        let mut gpu = ConsoleMemory::default();
        for memory in [&mut software, &mut gpu] {
            memory.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
            memory.write(DEPTH, &vec![0xFFu8; (SIZE * SIZE * 4) as usize]);
        }
        rasterize(&registers, &mut software, &mut Resources::default(), vertices);
        let before = resources.hardware.as_ref().unwrap().geometry_draws;
        rasterize(&registers, &mut gpu, &mut resources, vertices);
        assert_eq!(resources.hardware.as_ref().unwrap().geometry_draws, before + 1, "fixed-primitive loop went to the CPU");
        resources.hardware.as_mut().unwrap().flush(&mut gpu).unwrap();
        let (mut fixed_diff, mut fixed_drawn) = (0, 0);
        for i in 0..SIZE * SIZE {
            let (mut a, mut b) = ([0u8; 4], [0u8; 4]);
            software.read(COLOR + i * 4, &mut a);
            gpu.read(COLOR + i * 4, &mut b);
            fixed_drawn += (a != [0; 4]) as u32;
            if a.iter().zip(&b).any(|(a, b)| a.abs_diff(*b) > 3) {
                fixed_diff += 1;
            }
        }
        assert!(fixed_drawn > 20, "{fixed_drawn} fixed-primitive pixels drawn");
        assert!(fixed_diff * 50 < fixed_drawn, "{fixed_diff} of {fixed_drawn} fixed-primitive pixels differ");
    }

    /// guest memory with a stretch of it in one piece, from the linear
    /// heap's start, which slice hands out the way a title's vertex arrays
    /// lie, so draws from it go to the GPU as they are.
    #[cfg(feature = "vulkan")]
    struct ArrayMemory {
        rest: ConsoleMemory,
        arrays: Vec<u8>,
        /// how long each slice handed out was, a whole array's for the GPU
        /// among them, a vertex's when the CPU decodes them.
        sliced: Vec<usize>,
    }

    #[cfg(feature = "vulkan")]
    impl ArrayMemory {
        const START: u32 = 0x1400_0000;

        fn new() -> ArrayMemory {
            ArrayMemory { rest: ConsoleMemory::default(), arrays: vec![0; 0x1_0000], sliced: Vec::new() }
        }

        fn at(&self, addr: u32) -> Option<usize> {
            addr.checked_sub(Self::START).map(|at| at as usize).filter(|&at| at < self.arrays.len())
        }
    }

    #[cfg(feature = "vulkan")]
    impl GpuMemory for ArrayMemory {
        fn read(&mut self, addr: u32, out: &mut [u8]) {
            for (i, byte) in out.iter_mut().enumerate() {
                let addr = addr + i as u32;
                *byte = match self.at(addr) {
                    Some(at) => self.arrays[at],
                    None => {
                        let mut byte = [0];
                        self.rest.read(addr, &mut byte);
                        byte[0]
                    }
                };
            }
        }

        fn write(&mut self, addr: u32, data: &[u8]) {
            for (i, &byte) in data.iter().enumerate() {
                let addr = addr + i as u32;
                match self.at(addr) {
                    Some(at) => self.arrays[at] = byte,
                    None => self.rest.write(addr, &[byte]),
                }
            }
        }

        fn translate(&self, paddr: u32) -> u32 {
            self.rest.translate(paddr)
        }

        fn slice(&mut self, addr: u32, len: usize) -> Option<&[u8]> {
            let at = self.at(addr)?;
            let end = at.checked_add(len).filter(|&end| end <= self.arrays.len())?;
            self.sliced.push(len);
            Some(&self.arrays[at..end])
        }
    }

    /// draws from vertex arrays the GPU decodes itself come out as those
    /// whose vertices the CPU decodes, for each type of component, one to
    /// four of them, a fixed attribute, padding, one array or several,
    /// strides and offsets off a word, u8 and u16 indices and none, and
    /// lists, strips and fans.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_decodes_vertex_arrays_like_the_cpu() {
        for translates in [true, false] {
            let Ok(mut hardware) = hardware::Hardware::new() else { return };
            hardware.set_translates(translates, true);
            gpu_decodes_vertex_arrays_like_the_cpu(hardware);
        }
    }

    /// the arrays drawn through one GPU, translating programs or not.
    #[cfg(feature = "vulkan")]
    fn gpu_decodes_vertex_arrays_like_the_cpu(hardware: hardware::Hardware) {
        const SIZE: u32 = 32;
        // the arrays' base, physical FCRAM, at the linear heap's start
        const BASE: u32 = 0x2000_0000;
        const INDICES: u32 = 0x6000;
        const VERTICES: u32 = 24;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let op = |opcode: u32, destination: u32, src1: u32, src2: u32| opcode << 26 | destination << 21 | src1 << 12 | src2 << 7;
        // o0 = c0 * v0, o1 = c1 * v1 + c2 + v2, v2 the fixed attribute
        let program = [op(0x08, 0x00, 0x20, 0x00), op(0x08, 0x10, 0x21, 0x01), op(0x00, 0x10, 0x22, 0x10), op(0x00, 0x01, 0x10, 0x02), 0x22 << 26];
        unit.program[..program.len()].copy_from_slice(&program);
        unit.prepare();
        let mut fixed = [shader::ZERO; 16];
        fixed[2] = [0.1, -0.05, 0.02, 0.0];

        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b11;
        registers[REG_ATTRIBUTE_BASE] = BASE >> 3;
        // three attributes, the third fixed, each to the register of its
        // number
        registers[REG_VS_NUM_INPUT_ATTRIBUTES] = 2;
        registers[REG_VS_BLOCK + SHADER_INPUT_MAP_LOW] = 0x210;

        struct Case {
            /// each array's offset from the base, stride, and attribute ids
            /// or padding in order.
            loaders: Vec<(u32, u32, Vec<u32>)>,
            /// type and count of the position and of the color.
            position: (u32, u32),
            color: (u32, u32),
            /// u16 indices, u8 or none, and the first vertex used.
            indices: Option<bool>,
            first: u32,
        }
        let cases = [
            Case { loaders: vec![(0x100, 8, vec![0]), (0x800, 4, vec![1])], position: (3, 2), color: (1, 4), indices: Some(true), first: 0 },
            // a short at odd addresses, and a vertex longer than its stride
            Case { loaders: vec![(0x1001, 9, vec![0, 1, 12])], position: (2, 2), color: (0, 3), indices: Some(false), first: 0 },
            Case { loaders: vec![(0x2003, 3, vec![0]), (0x3002, 10, vec![1])], position: (0, 2), color: (2, 4), indices: None, first: 3 },
            // floats off a word, and one component of color
            Case { loaders: vec![(0x4001, 13, vec![0]), (0x5002, 6, vec![1])], position: (3, 2), color: (3, 1), indices: Some(true), first: 5 },
            // byte indices from well into the arrays, as a strip
            Case { loaders: vec![(0x1001, 9, vec![0, 1, 12])], position: (2, 2), color: (0, 3), indices: Some(false), first: 6 },
        ];
        let mut seed = 11u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        for (number, case) in cases.iter().enumerate() {
            let nibble = |(ty, count): (u32, u32)| (count - 1) << 2 | ty;
            registers[REG_ATTRIBUTE_FORMAT_LOW] = nibble(case.position) | nibble(case.color) << 4;
            registers[REG_ATTRIBUTE_FORMAT_HIGH] = 1 << 18 | 2 << 28;
            for loader in 0..12 {
                let at = REG_ATTRIBUTE_LOADER + loader * 3;
                registers[at..at + 3].fill(0);
            }
            for (i, (offset, stride, entries)) in case.loaders.iter().enumerate() {
                let at = REG_ATTRIBUTE_LOADER + i * 3;
                let packed = entries.iter().enumerate().fold(0u64, |packed, (k, &id)| packed | (id as u64) << (k * 4));
                registers[at] = *offset;
                registers[at + 1] = packed as u32;
                registers[at + 2] = (packed >> 32) as u32 | stride << 16 | (entries.len() as u32) << 28;
            }
            registers[REG_PRIMITIVE_CONFIG] = (number as u32 % 3) << 8;
            registers[REG_VERTEX_COUNT] = VERTICES;
            registers[REG_VERTEX_OFFSET] = case.first;
            registers[REG_INDEX_ARRAY] = INDICES | if case.indices == Some(true) { 1 << 31 } else { 0 };
            // scales that bring each type to the screen and to colors
            let scale = |ty: u32| match ty {
                0 => 1.0 / 128.0,
                1 => 1.0 / 255.0,
                2 => 1.0 / 32768.0,
                _ => 1.0,
            };
            let ps = scale(case.position.0) * 0.9;
            unit.float_uniforms[0] = [ps, ps, 1.0, 1.0];
            let cs = scale(case.color.0) * if matches!(case.color.0, 0 | 2) { 0.5 } else { 1.0 };
            let offset = if matches!(case.color.0, 0 | 2) { 0.5 } else { 0.0 };
            unit.float_uniforms[1] = [cs; 4];
            unit.float_uniforms[2] = [offset; 4];

            // the arrays' bytes, floats kept to the screen and to colors,
            // and how long each array's vertex is
            let mut bytes = vec![0u8; 0x6000];
            let mut sizes = Vec::new();
            for (offset, stride, entries) in &case.loaders {
                let mut field = 0u32;
                for &id in entries {
                    if id >= 12 {
                        field = field.next_multiple_of(4) + (id - 11) * 4;
                        continue;
                    }
                    let (ty, count) = if id == 0 { case.position } else { case.color };
                    let size: u32 = [1, 1, 2, 4][ty as usize];
                    field = field.next_multiple_of(size);
                    for vertex in 0..case.first + VERTICES + 8 {
                        for component in 0..count {
                            let at = (offset + vertex * stride + field + component * size) as usize;
                            let value = random();
                            let value = if ty == 3 {
                                let float = (value & 0xFFFF) as f32 / 65536.0;
                                (if id == 0 { float * 1.8 - 0.9 } else { float }).to_bits()
                            } else {
                                value
                            };
                            bytes[at..at + size as usize].copy_from_slice(&value.to_le_bytes()[..size as usize]);
                        }
                    }
                    field += size * count;
                }
                sizes.push(field);
            }
            let indices: Vec<u32> = (0..VERTICES).map(|_| case.first + random() % 20).collect();
            let index_bytes: Vec<u8> = match case.indices {
                Some(true) => indices.iter().flat_map(|&index| (index as u16).to_le_bytes()).collect(),
                _ => indices.iter().map(|&index| index as u8).collect(),
            };

            let mut decoded = ConsoleMemory::default();
            let mut raw = ArrayMemory::new();
            decoded.write(0x1400_0000, &bytes);
            decoded.write(0x1400_0000 + INDICES, &index_bytes);
            raw.write(0x1400_0000, &bytes);
            raw.write(0x1400_0000 + INDICES, &index_bytes);
            decoded.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
            raw.write(COLOR, &vec![0u8; (SIZE * SIZE * 4) as usize]);
            let indexed = case.indices.is_some();
            draw(&registers, &unit, &unit, &fixed, &mut decoded, &mut resources, indexed, false);
            resources.hardware.as_mut().unwrap().flush(&mut decoded).unwrap();
            draw(&registers, &unit, &unit, &fixed, &mut raw, &mut resources, indexed, false);
            resources.hardware.as_mut().unwrap().flush(&mut raw).unwrap();
            // the first array whole, from the first vertex used to the last
            let span = match indexed {
                true => indices.iter().max().unwrap() - indices.iter().min().unwrap() + 1,
                false => VERTICES,
            };
            let whole = ((span - 1) * case.loaders[0].1 + sizes[0]) as usize;
            assert!(raw.sliced.contains(&whole), "case {number}, the arrays went to the GPU as they are");
            let (mut a, mut b) = (vec![0u8; (SIZE * SIZE * 4) as usize], vec![0u8; (SIZE * SIZE * 4) as usize]);
            decoded.read(COLOR, &mut a);
            raw.read(COLOR, &mut b);
            let drawn = a.chunks(4).filter(|pixel| *pixel != [0; 4]).count();
            assert!(drawn > 50, "case {number}, only {drawn} pixels drawn");
            assert!(a == b, "case {number}, the arrays decode differently");
        }
    }

    /// a random number generator for the translation tests, the same
    /// numbers every run.
    #[cfg(feature = "vulkan")]
    struct Random(u64);

    #[cfg(feature = "vulkan")]
    impl Random {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// an arithmetic instruction of any kind, writing a temporary or
        /// the color, o2.
        fn arithmetic(&mut self) -> u32 {
            let (wide, narrow, index, descriptor) = (self.next() % 0x80, self.next() % 0x20, self.next() % 4, 1 + self.next() % 127);
            let destination = if self.next().is_multiple_of(4) { 2 } else { 0x10 + self.next() % 16 };
            let ops = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x12, 0x13];
            match self.next() % 12 {
                0..=7 => ops[(self.next() % ops.len() as u32) as usize] << 26 | destination << 21 | index << 19 | wide << 12 | narrow << 7 | descriptor,
                8 => [0x18, 0x19, 0x1A, 0x1B][(self.next() % 4) as usize] << 26 | destination << 21 | index << 19 | narrow << 14 | wide << 7 | descriptor,
                9 => 0x2E << 26 | (self.next() % 8) << 24 | (self.next() % 8) << 21 | index << 19 | wide << 12 | narrow << 7 | descriptor,
                10 => 0b111 << 29 | destination << 24 | index << 22 | (self.next() % 0x20) << 17 | wide << 10 | narrow << 5 | descriptor & 0x1F,
                _ => 0b110 << 29 | destination << 24 | index << 22 | (self.next() % 0x20) << 17 | narrow << 12 | wide << 5 | descriptor & 0x1F,
            }
        }

        /// a condition of any kind for a flow instruction, in its bits.
        fn condition(&mut self) -> u32 {
            (self.next() % 16) << 22
        }
    }

    /// the side of the square the translation tests draw into.
    #[cfg(feature = "vulkan")]
    const TRANSLATION_SIZE: u32 = 32;

    /// what the translation tests draw with, the position from o0 and the
    /// color from o2.
    #[cfg(feature = "vulkan")]
    fn translation_registers() -> Vec<u32> {
        let mut registers = target_registers();
        registers[REG_VIEWPORT_XY] = 4 | 6 << 16;
        registers[REG_VIEWPORT_WIDTH] = float24(12.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(12.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = TRANSLATION_SIZE | ((TRANSLATION_SIZE - 1) << 12);
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8 | 1 | 4 << 4 | 1 << 12;
        registers[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
        registers[REG_DEPTHMAP_ENABLE] = 1;
        registers[REG_SHADER_OUTPUT_TOTAL] = 2;
        registers[REG_SHADER_OUTPUT_MAP] = 0x0302_0100;
        registers[REG_SHADER_OUTPUT_MAP + 1] = 0x0B0A_0908;
        registers[REG_VS_OUTPUT_MASK] = 0b101;
        registers[REG_PRIMITIVE_CONFIG] = 0;
        registers
    }

    /// a unit running the program made some random words in, its flow moved
    /// along with it so the entry point matters, with random descriptors and
    /// uniforms, prepared, and all its words.
    #[cfg(feature = "vulkan")]
    fn random_unit(random: &mut Random, program: impl FnOnce(&mut Random) -> Vec<u32>) -> (ShaderUnit, Vec<u32>) {
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        for descriptor in &mut unit.descriptors[1..] {
            *descriptor = random.next() & 0x7FFF_FFFF;
        }
        for uniform in unit.float_uniforms.iter_mut() {
            *uniform = std::array::from_fn(|_| match random.next() % 30 {
                0 => f32::INFINITY,
                1 => -0.0,
                _ => (random.next() % 2000) as f32 / 1000.0 - 1.0,
            });
        }
        for integer in &mut unit.int_uniforms {
            *integer = [(random.next() % 3) as u8, (random.next() % 8) as u8, (random.next() % 3) as u8, 0];
        }
        unit.bool_uniforms = random.next() as u16;
        // past instructions never run
        let skip = random.next() % 8;
        let mut words: Vec<u32> = (0..skip).map(|_| random.arithmetic()).collect();
        words.extend(program(random).into_iter().map(|word| match word >> 26 {
            0x24..=0x29 | 0x2C | 0x2D => word + (skip << 10),
            _ => word,
        }));
        unit.program[..words.len()].copy_from_slice(&words);
        unit.entry_point = skip;
        unit.prepare();
        (unit, words)
    }

    /// random inputs for a translation test's draw.
    #[cfg(feature = "vulkan")]
    fn random_inputs(random: &mut Random) -> Vec<[shader::Vec4; shader::INPUT_REGISTERS]> {
        let mut inputs = vec![[shader::ZERO; shader::INPUT_REGISTERS]; 24];
        for input in &mut inputs {
            let mut float = || (random.next() % 2000) as f32 / 1000.0 - 1.0;
            input[0] = [float() * 1.3, float() * 1.3, float() * 0.6 - 0.1, 1.0];
            for register in &mut input[1..] {
                *register = [float(), float(), float(), float()];
            }
        }
        inputs
    }

    /// the color and then the depth a draw leaves on the GPU.
    #[cfg(feature = "vulkan")]
    fn draw_on_gpu(registers: &[u32], resources: &mut Resources, unit: &ShaderUnit, inputs: &[[shader::Vec4; shader::INPUT_REGISTERS]]) -> Vec<u8> {
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        let mut memory = ConsoleMemory::default();
        memory.write(COLOR, &vec![0u8; bytes]);
        memory.write(DEPTH, &vec![0xFFu8; bytes]);
        let vertices = Vertices::Unshaded { vertex_shader: unit, geometry_shader: unit, inputs, order: None };
        rasterize(registers, &mut memory, resources, vertices);
        resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
        let mut out = vec![0u8; bytes * 2];
        memory.read(COLOR, &mut out[..bytes]);
        memory.read(DEPTH, &mut out[bytes..]);
        out
    }

    /// draws programs made by program with the GPU interpreting them and
    /// with them translated, and checks color and depth come out the same
    /// to the bit, at least so many programs translated, and any other one
    /// only because it can run on past its end.
    #[cfg(feature = "vulkan")]
    fn draws_like_the_interpreter(cases: u32, least: u32, seed: u64, program: impl Fn(&mut Random) -> Vec<u32>) {
        let (Ok(mut translating), Ok(mut interpreting)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        translating.set_translates(true, true);
        interpreting.set_translates(false, true);
        let mut random = Random(seed);
        let registers = translation_registers();
        let mut translators = Resources { hardware: Some(translating), ..Default::default() };
        let mut interpreters = Resources { hardware: Some(interpreting), ..Default::default() };
        let mut drawn = 0;
        let mut structured = 0;
        for case in 0..cases {
            let (unit, program) = random_unit(&mut random, &program);
            // the only program left to the interpreter is one that can run
            // on into the words after it
            match unit.translate(&output_semantics(&registers)) {
                Err(error) => {
                    assert!(error.contains("reachable instructions"), "case {case} not translated, {error}, program {program:08X?}")
                }
                Ok(glsl) => structured += usize::from(!glsl.contains("switch (pc)")),
            }
            let inputs = random_inputs(&mut random);
            let translated = draw_on_gpu(&registers, &mut translators, &unit, &inputs);
            let interpreted = draw_on_gpu(&registers, &mut interpreters, &unit, &inputs);
            assert!(translated == interpreted, "case {case} draws differently translated, program {program:08X?}");
            let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
            drawn += translated[..bytes].chunks(4).filter(|pixel| *pixel != [0; 4]).count();
        }
        let translations = translators.hardware.as_ref().unwrap().translations();
        eprintln!("{cases} programs, {translations} translated, {structured} structured, {drawn} pixels drawn");
        assert!(drawn as u32 > cases * 40, "only {drawn} pixels drawn");
        assert!(translations as u32 >= least, "only {translations} of {cases} programs translated");
        assert_eq!(interpreters.hardware.as_ref().unwrap().translations(), 0);
    }

    /// 2D drawn on the near or far plane is drawn when z misses the plane
    /// only by rounding, as it would on the console, and clipped when it is
    /// really outside, the same on the CPU, on the GPU interpreting the
    /// program and with it translated.
    #[cfg(feature = "vulkan")]
    #[test]
    fn geometry_on_the_ends_of_the_z_range_is_drawn() {
        let mut registers = translation_registers();
        // color only, nothing but the clipping keeps a pixel out
        registers[REG_DEPTH_COLOR_MASK] = 0xF << 8;
        const IDENTITY: u32 = 0x1B << 5 | 0x1B << 14 | 0x1B << 23;
        let mut unit = ShaderUnit::new();
        unit.descriptors[0] = 0xF | IDENTITY;
        let mov = |destination: u32, source: u32| 0x13 << 26 | destination << 21 | source << 12;
        unit.program[..3].copy_from_slice(&[mov(0, 0), mov(2, 1), 0x22 << 26]);
        unit.prepare();
        let hardware = |translates: bool| {
            hardware::Hardware::new().ok().map(|mut hardware| {
                hardware.set_translates(translates, true);
                Resources { hardware: Some(hardware), ..Default::default() }
            })
        };
        let (mut interpreting, mut translating) = (hardware(false), hardware(true));
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        for (z, drawn) in [(-0.5, true), (-1.000005, true), (5e-9, true), (-1.001, false), (1e-6, false)] {
            let inputs: Vec<[shader::Vec4; shader::INPUT_REGISTERS]> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
                .into_iter()
                .map(|[x, y]| {
                    let mut input = [shader::ZERO; shader::INPUT_REGISTERS];
                    input[0] = [x, y, z, 1.0];
                    input[1] = [1.0, 0.0, 0.0, 1.0];
                    input
                })
                .collect();
            let mut memory = ConsoleMemory::default();
            memory.write(COLOR, &vec![0u8; bytes]);
            let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
            rasterize(&registers, &mut memory, &mut Resources::default(), vertices);
            let mut software = vec![0u8; bytes];
            memory.read(COLOR, &mut software);
            let red = software.chunks(4).filter(|pixel| ColorFormat::Rgba8.decode(pixel) == [255, 0, 0, 255]).count();
            assert_eq!(red > 0, drawn, "z {z} on the CPU, {red} pixels");
            for resources in [&mut interpreting, &mut translating].into_iter().flatten() {
                let gpu = draw_on_gpu(&registers, resources, &unit, &inputs);
                assert!(gpu[..bytes] == software[..], "z {z} on the GPU draws what the CPU does");
            }
        }
    }

    /// a program is interpreted while the compiler thread translates it and
    /// compiles its pipeline, then runs translated, drawing the same to the
    /// bit throughout, and the compiler stops with the GPU while it works.
    #[cfg(feature = "vulkan")]
    #[test]
    fn programs_translate_while_drawing_goes_on() {
        let (Ok(mut translating), Ok(mut interpreting)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        translating.set_translates(true, false);
        interpreting.set_translates(false, true);
        let mut random = Random(24);
        let registers = translation_registers();
        let mut translators = Resources { hardware: Some(translating), ..Default::default() };
        let mut interpreters = Resources { hardware: Some(interpreting), ..Default::default() };
        // a loop around a call to after the end, then an if
        let (unit, _) = random_unit(&mut random, |random| {
            let mut arithmetic = || random.arithmetic();
            vec![
                0x13 << 26,
                0x29 << 26 | 4 << 10,
                0x24 << 26 | 8 << 10 | 2,
                arithmetic(),
                arithmetic(),
                0x27 << 26 | 1 << 22 | 7 << 10,
                arithmetic(),
                0x22 << 26,
                arithmetic(),
                arithmetic(),
            ]
        });
        let inputs = random_inputs(&mut random);
        let started = std::time::Instant::now();
        let mut draws = 0;
        loop {
            // a pipeline made before the draw is the one it uses
            let translated = translators.hardware.as_ref().unwrap().translated_pipelines() > 0;
            assert!(draws > 0 || !translated, "the first draw waited for its program to be translated");
            let a = draw_on_gpu(&registers, &mut translators, &unit, &inputs);
            let b = draw_on_gpu(&registers, &mut interpreters, &unit, &inputs);
            assert!(a == b, "draw {draws} differs, translated {translated}");
            draws += 1;
            if translated {
                break;
            }
            assert!(started.elapsed().as_secs() < 60, "the program was never translated");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        eprintln!("translated after {draws} draws, {:?}", started.elapsed());
        // and one more the compiler is still busy with when everything goes
        let (unit, _) = random_unit(&mut random, |random| {
            std::iter::once(0x13 << 26).chain((0..60).map(|_| random.arithmetic())).chain([0x22 << 26]).collect()
        });
        draw_on_gpu(&registers, &mut translators, &unit, &inputs);
    }

    /// random fragment stages, every combiner source, operand, operation
    /// and scale, buffer updates, the three texture units over the texels at
    /// textures, procedural textures, lighting, the alpha test and both depth
    /// modes.
    #[cfg(feature = "vulkan")]
    fn random_fragment_stages(random: &mut Random, registers: &mut [u32], textures: u32) {
        for base in [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            registers[base] = random.next() & 0x0FFF_0FFF;
            registers[base + 1] = random.next() & 0x0077_7FFF;
            registers[base + 2] = (random.next() % 10) | ((random.next() % 10) << 16);
            registers[base + 3] = random.next();
            registers[base + 4] = random.next() & 0x0003_0003;
        }
        // the buffer updates and the buffer's first color
        registers[0x0E0] = random.next() & 0xFF00;
        registers[0x0FD] = random.next();
        // any of the units, unit 2 reading either coordinates, now and then a
        // procedural texture
        registers[REG_TEXTURE_CONFIG] = random.next() & 0x2307 | (random.next().is_multiple_of(4) as u32) << 10;
        for (unit, base) in TEXTURE_UNIT_BASES.into_iter().enumerate() {
            registers[base] = random.next();
            registers[base + 1] = 8 | 8 << 16;
            // the filters and the wrap modes
            registers[base + 2] = (random.next() % 4) << 1 | (random.next() % 4) << 8 | (random.next() % 4) << 12;
            registers[base + 4] = (textures + unit as u32 * 0x100) >> 3;
            registers[if unit == 0 { base + 13 } else { base + 5 }] = 0;
        }
        registers[crate::lighting::REG_ENABLE] = random.next().is_multiple_of(3) as u32;
        // on or off, a function and a reference
        registers[REG_ALPHA_TEST] = random.next() & 0xFF71;
        registers[REG_DEPTHMAP_ENABLE] = random.next() & 1;
    }

    /// the generic fragment shader, which draws while the one made for a
    /// combination compiles, draws what that one draws, to the bit, over
    /// random fragment stages, with the triangles placed by the CPU and with
    /// the vertices shaded on the GPU.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_generic_fragment_shader_draws_like_the_specialized_ones() {
        let (Ok(mut generic), Ok(specialized)) = (hardware::Hardware::new(), hardware::Hardware::new()) else {
            return;
        };
        generic.set_generic(true);
        const TEXTURES: u32 = 0x8_0000;
        let mut random = Random(25);
        let mut generics = Resources { hardware: Some(generic), ..Default::default() };
        let mut specializeds = Resources { hardware: Some(specialized), ..Default::default() };
        let bytes = (TRANSLATION_SIZE * TRANSLATION_SIZE * 4) as usize;
        let (mut drawn, mut differing) = (0, 0);
        for case in 0..180 {
            let mut registers = translation_registers();
            random_fragment_stages(&mut random, &mut registers, TEXTURES);
            let texels: Vec<u8> = (0..0x300).map(|_| random.next() as u8).collect();
            let mut results = Vec::new();
            if case < 120 {
                let triangles: Vec<Vec<Vertex>> = (0..4)
                    .map(|_| {
                        (0..3)
                            .map(|_| {
                                let mut float = || (random.next() % 2000) as f32 / 1000.0 - 1.0;
                                let w = 1.5 + float();
                                Vertex {
                                    clip: [float() * w, float() * w, (float() * 0.5 - 0.5) * w, w],
                                    color: [float().abs(), float().abs(), float().abs(), float().abs()],
                                    texcoords: [[float() * 2.0, float() * 2.0], [float() * 2.0, float() * 2.0], [float() * 2.0, float() * 2.0]],
                                    texcoord0_w: 1.0,
                                    quaternion: [float(), float(), float(), float()],
                                    view: [float(), float(), float()],
                                }
                            })
                            .collect()
                    })
                    .collect();
                for resources in [&mut generics, &mut specializeds] {
                    let mut memory = ConsoleMemory::default();
                    memory.write(COLOR, &vec![0u8; bytes]);
                    memory.write(DEPTH, &vec![0xFFu8; bytes]);
                    memory.write(TEXTURES, &texels);
                    for triangle in &triangles {
                        rasterize_shaded(&registers, &mut memory, resources, triangle);
                    }
                    resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
                    let mut out = vec![0u8; bytes * 2];
                    memory.read(COLOR, &mut out[..bytes]);
                    memory.read(DEPTH, &mut out[bytes..]);
                    results.push(out);
                }
            } else {
                // all that lighting and the units read, each output register
                // moved from the input register of its number, the position,
                // the quaternion, the color, the view and the coordinates
                registers[REG_SHADER_OUTPUT_TOTAL] = 6;
                let map = [0x0302_0100, 0x0706_0504, 0x0B0A_0908, 0x1F14_1312, 0x0F0E_0D0C, 0x1F1F_1716];
                registers[REG_SHADER_OUTPUT_MAP..REG_SHADER_OUTPUT_MAP + 6].copy_from_slice(&map);
                registers[REG_VS_OUTPUT_MASK] = 0b11_1111;
                let moves = (0..6).map(|register| 0x13 << 26 | register << 21 | register << 12);
                let (unit, _) = random_unit(&mut random, |_| moves.chain([0x22 << 26]).collect());
                let inputs = random_inputs(&mut random);
                for resources in [&mut generics, &mut specializeds] {
                    let vertices = Vertices::Unshaded { vertex_shader: &unit, geometry_shader: &unit, inputs: &inputs, order: None };
                    let mut memory = ConsoleMemory::default();
                    memory.write(COLOR, &vec![0u8; bytes]);
                    memory.write(DEPTH, &vec![0xFFu8; bytes]);
                    memory.write(TEXTURES, &texels);
                    rasterize(&registers, &mut memory, resources, vertices);
                    resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
                    let mut out = vec![0u8; bytes * 2];
                    memory.read(COLOR, &mut out[..bytes]);
                    memory.read(DEPTH, &mut out[bytes..]);
                    results.push(out);
                }
            }
            // the coordinates the units and the procedural texture read are
            // interpolated as the driver sees fit for each shader, which now
            // and then puts a texel's weight a hair apart and a channel a
            // step off. everything else comes out the same to the bit, the
            // combiners, lighting, the tests and depth
            let coordinates = registers[REG_TEXTURE_CONFIG] & 0x407 != 0;
            let stages: Vec<u32> = [0x0C0, 0x0C1, 0x0C2, 0x0C4, 0x0E0, REG_TEXTURE_CONFIG, REG_ALPHA_TEST].map(|r| registers[r]).to_vec();
            for (i, (a, b)) in results[0].chunks(4).zip(results[1].chunks(4)).enumerate() {
                let apart = a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
                let allowed = if coordinates && i < bytes / 4 { 1 } else { 0 };
                assert!(apart <= allowed, "case {case}, pixel {i} is {a:?} through the generic shader, {b:?} through its own, {stages:08X?}");
                differing += (apart > 0) as usize;
            }
            drawn += results[0][..bytes].chunks(4).filter(|pixel| *pixel != [0; 4]).count();
        }
        eprintln!("{drawn} pixels drawn, {differing} a step off");
        assert!(drawn > 180 * 40, "only {drawn} pixels drawn");
        assert!(differing * 100 < drawn, "{differing} of {drawn} pixels a step off");
    }

    /// programs of every instruction, with flow of every kind going
    /// forward, draw the same translated as interpreted, to the bit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn translated_shaders_draw_like_the_interpreter() {
        draws_like_the_interpreter(200, 200, 21, |random| {
            // the position as it comes, then instructions of every kind,
            // flow only going forward so every program ends, and blocks
            // ending within the program, past it are words of zero that
            // run to the end of the program memory
            let length = 4 + random.next() % 28;
            let mut program = vec![0x13 << 26];
            for at in 1..length {
                let forward = at + 1 + random.next() % (length - at);
                let (count, condition) = ((random.next() % 4).min(length - forward), random.condition());
                program.push(match random.next() % 16 {
                    0..=11 => random.arithmetic(),
                    12 => [0x24, 0x25, 0x26, 0x27, 0x28][(random.next() % 5) as usize] << 26 | condition | forward << 10 | count,
                    13 => [0x2C, 0x2D][(random.next() % 2) as usize] << 26 | condition | forward << 10 | count,
                    14 => 0x29 << 26 | (random.next() % 4) << 22 | (at + 1 + random.next() % 3).min(length - 1) << 10,
                    _ => [0x20 << 26, 0x23 << 26 | condition][(random.next() % 2) as usize],
                });
            }
            program.push(0x22 << 26);
            program
        });
    }

    /// programs of arithmetic, compares and forward jumps nesting inside
    /// each other or crossing, many of them out of what an enclosing jump
    /// skips to its end, as KIU's main vertex shader does, draw the same
    /// translated as interpreted, to the bit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn forward_jumps_draw_like_the_interpreter() {
        draws_like_the_interpreter(200, 200, 24, |random| {
            let length = 6 + random.next() % 30;
            let mut program = vec![0x13 << 26];
            // destinations of the jumps so far, which later ones often reuse
            let mut ends: Vec<u32> = Vec::new();
            for at in 1..length {
                program.push(match random.next() % 8 {
                    0..=3 => random.arithmetic(),
                    // a compare, the conditions the jumps read
                    4 => 0x2E << 26 | (random.next() % 6) << 24 | (random.next() % 6) << 21 | (random.next() % 0x80) << 12 | (random.next() % 0x20) << 7 | (1 + random.next() % 127),
                    _ => {
                        let reuse = ends.iter().copied().filter(|&end| end > at + 1).collect::<Vec<_>>();
                        let destination = match reuse.is_empty() || random.next() % 2 == 0 {
                            true => at + 1 + random.next() % (length - at),
                            false => reuse[(random.next() as usize) % reuse.len()],
                        };
                        ends.push(destination);
                        let condition = random.condition();
                        [0x2C, 0x2D][(random.next() % 2) as usize] << 26 | condition | destination << 10 | (random.next() % 4)
                    }
                });
            }
            program.push(0x22 << 26);
            program
        });
    }

    /// programs that run on until the interpreter gives up on them, a body
    /// writing the color repeated by a jump back, stop at the same
    /// instruction translated, however long the body, so the color comes
    /// out the same to the bit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn programs_that_never_end_draw_like_the_interpreter() {
        draws_like_the_interpreter(60, 60, 23, |random| {
            let body = 1 + random.next() % 9;
            let mut program = vec![0x13 << 26];
            program.extend((0..body).map(|_| random.arithmetic()));
            // back to the body's start when its boolean says so, which half
            // the time is always
            program.push((0x2D << 26) | ((random.next() % 16) << 22) | (1 << 10) | (random.next() % 2));
            program.push(0x22 << 26);
            program
        });
    }

    /// programs laid out the way the SDK's compiler lays them out, a main
    /// part calling subroutines after its end, each running into the next
    /// one where its call ends it, with ifs, loops, breaks and calls of
    /// their own, draw the same translated as interpreted, to the bit. a
    /// jump out of a loop leaves the loop's block over the call's, which
    /// then never ends, and execution runs on to the end of the program
    /// memory, so some of them stay interpreted, as they should.
    #[cfg(feature = "vulkan")]
    #[test]
    fn subroutines_draw_like_the_interpreter() {
        draws_like_the_interpreter(150, 100, 22, |random| {
            let routines = 1 + random.next() % 5;
            let main = 3 + random.next() % 8;
            let lengths: Vec<u32> = (0..routines).map(|_| 2 + random.next() % 10).collect();
            let starts: Vec<u32> = lengths
                .iter()
                .scan(main + 1, |at, &length| {
                    let start = *at;
                    *at += length;
                    Some(start)
                })
                .collect();
            let call = |random: &mut Random, after: u32| -> Option<u32> {
                // only routines further on, so nothing calls itself
                let first = starts.iter().position(|&start| start > after)?;
                let routine = first + (random.next() as usize % (starts.len() - first));
                let opcode = [0x24, 0x25, 0x26][(random.next() % 3) as usize];
                Some(opcode << 26 | random.condition() | starts[routine] << 10 | lengths[routine])
            };
            // the position as it comes, then the main part
            let mut program = vec![0x13 << 26];
            for at in 1..main {
                program.push(match random.next() % 4 {
                    0 => call(random, at).unwrap_or_else(|| random.arithmetic()),
                    _ => random.arithmetic(),
                });
            }
            program.push(0x22 << 26);
            // the subroutines, ifs and jumps forward within themselves, and
            // breaks within their loops, a break with no loop open would
            // close the call too and run on into the next subroutine
            for (&start, &length) in starts.iter().zip(&lengths) {
                let end = start + length;
                let mut looped = 0;
                for at in start..end {
                    let forward = at + 1 + random.next() % (end - at);
                    let count = (random.next() % 3).min(end - forward);
                    let word = match random.next() % 10 {
                        0 if at + 1 < end => {
                            let opcode = [0x27, 0x28][(random.next() % 2) as usize];
                            opcode << 26 | random.condition() | forward << 10 | count
                        }
                        1 if at + 1 < end => [0x2C, 0x2D][(random.next() % 2) as usize] << 26 | random.condition() | forward << 10,
                        2 => call(random, end - 1).unwrap_or_else(|| random.arithmetic()),
                        3 if at + 2 < end && at >= looped => {
                            let last = (at + 1 + random.next() % (end - at - 1)).min(end - 1);
                            looped = last + 1;
                            0x29 << 26 | (random.next() % 4) << 22 | last << 10
                        }
                        4 if at < looped => 0x23 << 26 | random.condition(),
                        _ => random.arithmetic(),
                    };
                    program.push(word);
                }
            }
            program
        });
    }

    /// a texture that is rows of a buffer the GPU just drew samples the
    /// same from a copy the GPU makes as from memory once it has the buffer,
    /// whether the buffer holds colors or depth and stencil.
    #[cfg(feature = "vulkan")]
    #[test]
    fn textures_drawn_on_the_gpu_sample_like_memory() {
        for depth in [false, true] {
            textures_drawn_on_the_gpu_sample_like_memory_from(depth);
        }
    }

    #[cfg(feature = "vulkan")]
    fn textures_drawn_on_the_gpu_sample_like_memory_from(depth: bool) {
        let (Ok(first), Ok(second)) = (hardware::Hardware::new(), hardware::Hardware::new()) else { return };
        const SIZE: u32 = 32;
        const TARGET: u32 = 0x10_0000;
        let mut drawing = target_registers();
        drawing[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        drawing[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        drawing[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        if depth {
            // depth written always, the stencil as it comes, the depth
            // being -z as titles map it
            drawing[REG_DEPTH_COLOR_MASK] |= 1 << 12;
            drawing[REG_VIEWPORT_DEPTH_RANGE] = float24(-1.0);
            drawing[REG_DEPTHMAP_ENABLE] = 1;
        }
        // the second draw fills a buffer half as tall with rows 8 to 23 of
        // the first, or of its depth buffer read as colors, as its texture,
        // straight from the combiners
        let mut sampling = target_registers();
        sampling[REG_COLOR_BUFFER_ADDRESS] = TARGET >> 3;
        sampling[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        sampling[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 4.0);
        sampling[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE / 2 - 1) << 12);
        sampling[REG_TEXTURE_CONFIG] = 1;
        sampling[REG_TEXTURE0_DIMENSIONS] = (SIZE / 2) | (SIZE << 16);
        let source = if depth { DEPTH } else { COLOR };
        sampling[REG_TEXTURE0_ADDRESS] = (source + 8 * SIZE * 4) >> 3;
        sampling[0x0C0] = 0x3 | (0x3 << 16);
        for stage in [0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8] {
            sampling[stage] = 0xF | (0xF << 16);
        }

        let mut seed = 5u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let triangles: Vec<Vec<Vertex>> = (0..30)
            .map(|_| {
                let color = [random(), random(), random(), random()];
                (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -random(), 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        // a triangle over the whole target, the texture spread across it
        let quad: Vec<Vertex> = [[-1.0, -1.0, 0.0, 0.0], [3.0, -1.0, 2.0, 0.0], [-1.0, 3.0, 0.0, 2.0]]
            .into_iter()
            .map(|[x, y, u, v]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [1.0; 4],
                texcoords: [[u, v], [0.0; 2], [0.0; 2]],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
                ..Default::default()
            })
            .collect();

        // once with the texture copied on the GPU, once with memory given
        // the drawing back first
        let mut results = Vec::new();
        for (hardware, back_first) in [(first, false), (second, true)] {
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            for triangle in &triangles {
                rasterize_shaded(&drawing, &mut memory, &mut resources, triangle);
            }
            if back_first {
                resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            }
            rasterize_shaded(&sampling, &mut memory, &mut resources, &quad);
            resources.hardware.as_mut().unwrap().flush(&mut memory).unwrap();
            let mut sampled = vec![0u8; (SIZE * SIZE / 2 * 4) as usize];
            memory.read(TARGET, &mut sampled);
            results.push(sampled);
        }
        assert!(results[0] == results[1], "reading {}", if depth { "depth" } else { "colors" });
        let colors: std::collections::HashSet<&[u8]> = results[0].chunks(4).collect();
        assert!(colors.len() > 8, "only {} colors", colors.len());
    }

    /// a display transfer out of a buffer drawn on the GPU leaves the same
    /// bytes as the CPU's, whatever the formats, flip, downscale and layout,
    /// and from rows into the buffer as well as its start.
    #[cfg(feature = "vulkan")]
    #[test]
    fn transfers_on_the_gpu_match_the_cpu() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        for input_format in [Rgba8, Rgb565] {
            let mut registers = target_registers();
            registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
            registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
            registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
            registers[REG_COLOR_BUFFER_FORMAT] = input_format.color_buffer_raw() << 16;
            // triangles in all sorts of colors over each other
            let mut drawn = ConsoleMemory::default();
            for _ in 0..40 {
                let color = [random(), random(), random(), random()];
                let triangle: Vec<Vertex> = (0..3)
                    .map(|_| Vertex {
                        clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                        color,
                        texcoords: [[0.0; 2]; 3],
                        quaternion: [0.0, 0.0, 0.0, 1.0],
                        view: [0.0; 3],
                        ..Default::default()
                    })
                    .collect();
                rasterize_shaded(&registers, &mut drawn, &mut resources, &triangle);
            }
            // memory gets what the GPU drew, which stays on the GPU too
            resources.hardware.as_mut().unwrap().flush(&mut drawn).unwrap();
            for output_format in [Rgba8, Rgb8, Rgb565, Rgb5A1, Rgba4] {
                let cases = [(false, 0, false, 0), (true, 0, true, 0), (false, 1, false, 0), (true, 2, false, 0), (false, 0, false, 1), (true, 2, false, 2)];
                for (flip, downscale, tiled, tile_rows) in cases {
                    let hardware = resources.hardware.as_mut().unwrap();
                    let (mut gpu, mut cpu) = (drawn.clone(), drawn.clone());
                    let (scale_x, scale_y) = [(1, 1), (2, 1), (2, 2)][downscale];
                    let input = COLOR + tile_rows * 8 * SIZE * input_format.bytes_per_pixel() as u32;
                    let input_height = SIZE - tile_rows * 8;
                    let (width, height) = (SIZE / scale_x, input_height / scale_y);
                    let transfer = hardware::Transfer {
                        input,
                        output: OUTPUT,
                        input_width: SIZE,
                        output_width: width,
                        output_height: height,
                        copy: (width, height),
                        scale: (scale_x, scale_y),
                        flip,
                        input_linear: false,
                        output_tiled: tiled,
                        input_format,
                        output_format,
                    };
                    assert!(hardware.display_transfer(&mut gpu, &transfer).unwrap());
                    hardware.flush(&mut gpu).unwrap();
                    let flags = flip as u32
                        | (tiled as u32) << 5
                        | (hardware::format_index(input_format) as u32) << 8
                        | (hardware::format_index(output_format) as u32) << 12
                        | (downscale as u32) << 24;
                    // the output's size in the register counts input pixels
                    let output_size = (width * scale_x) | ((height * scale_y) << 16);
                    let input_size = SIZE | input_height << 16;
                    crate::Gpu::new().display_transfer(&mut cpu, input, OUTPUT, input_size, output_size, flags);
                    let len = (width * height) as usize * output_format.bytes_per_pixel();
                    let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                    cpu.read(OUTPUT, &mut a);
                    gpu.read(OUTPUT, &mut b);
                    assert!(a == b, "{input_format:?} to {output_format:?}, flip {flip}, downscale {downscale}, rows {tile_rows}");
                    // something worth comparing
                    let colors: std::collections::HashSet<&[u8]> = a.chunks(output_format.bytes_per_pixel()).collect();
                    assert!(colors.len() > 8);
                }
            }
        }
    }

    /// the triangles assemble_into puts down are assemble's, for a list,
    /// a strip and a fan of any length, each through the vertex it names.
    #[test]
    fn both_assemblers_make_the_same_triangles() {
        for topology in 0..4 {
            for count in 0..12 {
                let mut indices = vec![99];
                assemble_into(topology, count, |i| 100 + i as u32, &mut indices);
                let triangles: Vec<u32> =
                    assemble(topology, count).into_iter().flat_map(|(a, b, c)| [a, b, c].map(|i| 100 + i as u32)).collect();
                assert_eq!(indices, triangles, "topology {topology}, {count} vertices");
            }
        }
    }

    /// the triangles the GPU takes are the ones assemble_into puts down, for
    /// raw arrays as they number the vertices, a list's going as they are,
    /// for vertices in an order and for vertices in a row.
    #[cfg(feature = "vulkan")]
    #[test]
    fn the_gpu_takes_the_triangles_the_assembler_makes() {
        for topology in 0..4 {
            for count in 0..14 {
                let raw: Vec<u32> = (0..count as u32).map(|i| 1000 + i * 7 % 5).collect();
                let order: Vec<usize> = (0..count).map(|i| (i * 3) % 4).collect();
                let mut assembled = vec![99];
                let mut expected = Vec::new();
                assemble_into(topology, count, |i| raw[i], &mut expected);
                assert_eq!(triangle_indices(topology, count, Some(&raw), None, &mut assembled), expected, "raw, topology {topology}, {count} vertices");
                assemble_into(topology, count, |i| order[i] as u32, &mut expected);
                assert_eq!(triangle_indices(topology, count, None, Some(&order), &mut assembled), expected, "ordered, topology {topology}, {count} vertices");
                assemble_into(topology, count, |i| i as u32, &mut expected);
                assert_eq!(triangle_indices(topology, count, None, None, &mut assembled), expected, "in a row, topology {topology}, {count} vertices");
            }
        }
    }

    /// the semantics kept from the last draw are the ones the output map
    /// and mask give, through changes to those registers and to others.
    #[cfg(feature = "vulkan")]
    #[test]
    fn kept_semantics_follow_the_output_registers() {
        let mut seed = 11u32;
        // the high bits, the low ones of the sequence repeat soon
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed >> 8
        };
        let mut registers = vec![0u32; 0x300];
        let read: Vec<usize> =
            [REG_SHADER_OUTPUT_TOTAL, REG_VS_OUTPUT_MASK].into_iter().chain(REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END).collect();
        // a map naming every semantic, which the changes below take apart
        registers[REG_SHADER_OUTPUT_TOTAL] = 7;
        for (i, register) in (REG_SHADER_OUTPUT_MAP..=REG_SHADER_OUTPUT_MAP_END).enumerate() {
            registers[register] = u32::from_le_bytes(std::array::from_fn(|c| (i * 4 + c) as u8));
        }
        registers[REG_VS_OUTPUT_MASK] = 0x7F;
        let mut kept = LastSemantics::default();
        for step in 0..2000 {
            let register = match random() % 8 {
                0 => (random() % 0x300) as usize,
                _ => read[(random() as usize) % read.len()],
            };
            // the low bits most of the time, which are what the map reads
            registers[register] ^= 1 << (random() % if random() % 4 == 0 { 32 } else { 5 });
            assert_eq!(kept.get(&registers), output_semantics(&registers), "step {step}, register {register:#X}");
        }
    }

    /// a drawing a transfer turned into a linear buffer, and another
    /// transfer turned back into one to sample, as titles do with a frame
    /// for their effects, stays on the GPU the whole way and leaves the bytes
    /// the CPU's two transfers do. a linear buffer only memory has the CPU
    /// transfers.
    #[cfg(feature = "vulkan")]
    #[test]
    fn transfers_out_of_linear_buffers_the_gpu_holds_match_the_cpu() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const LINEAR: u32 = 0x10_0000;
        const OUTPUT: u32 = 0x18_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut seed = 13u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let mut drawn = ConsoleMemory::default();
        for _ in 0..40 {
            let color = [random(), random(), random(), random()];
            let triangle: Vec<Vertex> = (0..3)
                .map(|_| Vertex {
                    clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                    color,
                    texcoords: [[0.0; 2]; 3],
                    quaternion: [0.0, 0.0, 0.0, 1.0],
                    view: [0.0; 3],
                    ..Default::default()
                })
                .collect();
            rasterize_shaded(&registers, &mut drawn, &mut resources, &triangle);
        }
        resources.hardware.as_mut().unwrap().flush(&mut drawn).unwrap();
        let flags = |flip: bool, input_linear: bool, output_tiled: bool, input: ColorFormat, output: ColorFormat, downscale: u32| {
            flip as u32
                | (input_linear as u32) << 1
                | ((input_linear != output_tiled) as u32) << 5
                | (hardware::format_index(input) as u32) << 8
                | (hardware::format_index(output) as u32) << 12
                | downscale << 24
        };
        for linear_format in [Rgba8, Rgb565] {
            for output_format in [Rgba8, Rgb565, Rgba4] {
                for (flip, downscale, tiled) in [(false, 0, true), (true, 0, true), (false, 2, true), (true, 1, false)] {
                    let hardware = resources.hardware.as_mut().unwrap();
                    let (mut gpu, mut cpu) = (drawn.clone(), drawn.clone());
                    let (scale_x, scale_y) = [(1, 1), (2, 1), (2, 2)][downscale as usize];
                    let (width, height) = (SIZE / scale_x, SIZE / scale_y);
                    let to_linear = hardware::Transfer {
                        input: COLOR,
                        output: LINEAR,
                        input_width: SIZE,
                        output_width: SIZE,
                        output_height: SIZE,
                        copy: (SIZE, SIZE),
                        scale: (1, 1),
                        flip: false,
                        input_linear: false,
                        output_tiled: false,
                        input_format: Rgba8,
                        output_format: linear_format,
                    };
                    let back = hardware::Transfer {
                        input: LINEAR,
                        output: OUTPUT,
                        output_width: width,
                        output_height: height,
                        copy: (width, height),
                        scale: (scale_x, scale_y),
                        flip,
                        input_linear: true,
                        output_tiled: tiled,
                        input_format: linear_format,
                        output_format,
                        ..to_linear
                    };
                    assert!(hardware.display_transfer(&mut gpu, &to_linear).unwrap());
                    assert!(hardware.display_transfer(&mut gpu, &back).unwrap(), "out of the linear buffer on the GPU");
                    hardware.flush(&mut gpu).unwrap();
                    let mut software = crate::Gpu::new();
                    let size = SIZE | SIZE << 16;
                    software.display_transfer(&mut cpu, COLOR, LINEAR, size, size, flags(false, false, false, Rgba8, linear_format, 0));
                    let output_size = (width * scale_x) | ((height * scale_y) << 16);
                    let back_flags = flags(flip, true, tiled, linear_format, output_format, downscale);
                    software.display_transfer(&mut cpu, LINEAR, OUTPUT, size, output_size, back_flags);
                    let len = (width * height) as usize * output_format.bytes_per_pixel();
                    let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
                    cpu.read(OUTPUT, &mut a);
                    gpu.read(OUTPUT, &mut b);
                    assert!(a == b, "{linear_format:?} to {output_format:?}, flip {flip}, downscale {downscale}, tiled {tiled}");
                    let colors: std::collections::HashSet<&[u8]> = a.chunks(output_format.bytes_per_pixel()).collect();
                    assert!(colors.len() > 8);
                }
            }
        }
        // nothing the GPU holds there
        let hardware = resources.hardware.as_mut().unwrap();
        let elsewhere = hardware::Transfer {
            input: 0x0C_0000,
            output: OUTPUT,
            input_width: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: true,
            output_tiled: true,
            input_format: Rgb565,
            output_format: Rgb565,
        };
        assert!(!hardware.display_transfer(&mut drawn.clone(), &elsewhere).unwrap());
        // an output of part of a tile is the CPU's too
        let partial = hardware::Transfer { input: LINEAR, output_height: SIZE - 4, copy: (SIZE, SIZE - 4), ..elsewhere };
        assert!(!hardware.display_transfer(&mut drawn.clone(), &partial).unwrap());
    }

    /// a fill over a whole linear buffer the GPU wrote, with a pattern that
    /// does not line up with its pixels, is what a transfer out of it reads,
    /// also when memory had that pattern already.
    #[cfg(feature = "vulkan")]
    #[test]
    fn a_transfer_reads_the_fill_over_a_linear_buffer_the_gpu_wrote() {
        use crate::format::ColorFormat::*;
        let Ok(hardware) = hardware::Hardware::new() else { return };
        const SIZE: u32 = 32;
        const LINEAR: u32 = 0x10_0000;
        const OUTPUT: u32 = 0x18_0000;
        let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let mut gpu = ConsoleMemory::default();
        let triangle: Vec<Vertex> = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]
            .into_iter()
            .map(|[x, y]| Vertex {
                clip: [x, y, -0.5, 1.0],
                color: [0.2, 0.6, 0.9, 1.0],
                texcoords: [[0.0; 2]; 3],
                quaternion: [0.0, 0.0, 0.0, 1.0],
                view: [0.0; 3],
                ..Default::default()
            })
            .collect();
        rasterize_shaded(&registers, &mut gpu, &mut resources, &triangle);
        let hardware = resources.hardware.as_mut().unwrap();
        hardware.flush(&mut gpu).unwrap();
        let mut cpu = gpu.clone();
        // halves of a pixel that differ
        let pattern: Vec<u8> = [0xFF, 0xFF, 0x00, 0x00].repeat((SIZE * SIZE / 2) as usize);
        let fill = |hardware: &mut hardware::Hardware, memory: &mut ConsoleMemory| {
            hardware.before_fill(memory, LINEAR, pattern.len() as u32, 4).unwrap();
            memory.write(LINEAR, &pattern);
            hardware.filled(LINEAR, &pattern).unwrap();
        };
        let to_linear = hardware::Transfer {
            input: COLOR,
            output: LINEAR,
            input_width: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: Rgba8,
            output_format: Rgb565,
        };
        let back = hardware::Transfer { input: LINEAR, output: OUTPUT, input_linear: true, output_tiled: true, input_format: Rgb565, ..to_linear };
        fill(hardware, &mut gpu);
        assert!(hardware.display_transfer(&mut gpu, &to_linear).unwrap());
        fill(hardware, &mut gpu);
        assert!(hardware.display_transfer(&mut gpu, &back).unwrap());
        hardware.flush(&mut gpu).unwrap();
        // the CPU's, the drawing gone under the fill
        let mut software = crate::Gpu::new();
        cpu.write(LINEAR, &pattern);
        let size = SIZE | SIZE << 16;
        software.display_transfer(&mut cpu, LINEAR, OUTPUT, size, size, 1 << 1 | (hardware::format_index(Rgb565) as u32) << 8 | (hardware::format_index(Rgb565) as u32) << 12);
        let len = (SIZE * SIZE * 2) as usize;
        let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
        cpu.read(OUTPUT, &mut a);
        gpu.read(OUTPUT, &mut b);
        assert!(a == b, "the transfer read what the GPU drew under the fill");
    }

    /// a screen shown straight from the GPU's image is the picture the host
    /// gets a copy of otherwise, where the image says the screen lies in it,
    /// for a screen some rows into a buffer with longer rows than it shows.
    #[cfg(feature = "vulkan")]
    #[test]
    fn screens_shown_from_the_gpu_match_the_copies() {
        use crate::format::ColorFormat::Rgba8;
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        for scale in [1, 3] {
            let Ok(copied) = hardware::Hardware::new() else { return };
            let device = hardware::own_device().unwrap();
            let direct = hardware::Hardware::with_device(std::sync::Arc::new(device), true, None).unwrap();
            let mut pictures = Vec::new();
            for mut hardware in [copied, direct] {
                let scale = hardware.set_scale(scale);
                let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
                let mut registers = target_registers();
                registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
                registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
                registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
                let mut seed = 11u32;
                let mut random = move || {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 8) as f32 / (1 << 24) as f32
                };
                let mut memory = ConsoleMemory::default();
                for _ in 0..40 {
                    let color = [random(), random(), random(), 1.0];
                    let triangle: Vec<Vertex> = (0..3)
                        .map(|_| Vertex {
                            clip: [random() * 2.0 - 1.0, random() * 2.0 - 1.0, -0.5, 1.0],
                            color,
                            texcoords: [[0.0; 2]; 3],
                            quaternion: [0.0, 0.0, 0.0, 1.0],
                            view: [0.0; 3],
                            ..Default::default()
                        })
                        .collect();
                    rasterize_shaded(&registers, &mut memory, &mut resources, &triangle);
                }
                let hardware = resources.hardware.as_mut().unwrap();
                let transfer = hardware::Transfer {
                    input: COLOR,
                    output: OUTPUT,
                    input_width: SIZE,
                    output_width: SIZE,
                    output_height: SIZE,
                    copy: (SIZE, SIZE),
                    scale: (1, 1),
                    flip: false,
                    input_linear: false,
                    output_tiled: false,
                    input_format: Rgba8,
                    output_format: Rgba8,
                };
                assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
                // 20 rows from the fourth, of 24 pixels each
                let screen = hardware.screen(OUTPUT + 4 * SIZE * 4, (24, 20), SIZE, Rgba8, &[]).expect("drawn on the GPU");
                let picture = match hardware.screen_image(screen).unwrap() {
                    Some(image) => {
                        let (pixels, width) = hardware.upright_pixels(screen, false).unwrap();
                        let height = pixels.len() as u32 / 4 / width;
                        let [x, y, w, h] = image.area.map(f64::from);
                        let [left, top] = [(x * width as f64).round() as usize, (y * height as f64).round() as usize];
                        let [right, bottom] = [((x + w) * width as f64).round() as usize, ((y + h) * height as f64).round() as usize];
                        // the bounds half a texel in from the area's edges
                        let [l, t, r, b] = image.bounds.map(f64::from);
                        let half = [0.5 / width as f64, 0.5 / height as f64];
                        assert!((l - x - half[0]).abs() < 1e-6 && (t - y - half[1]).abs() < 1e-6);
                        assert!((x + w - r - half[0]).abs() < 1e-6 && (y + h - b - half[1]).abs() < 1e-6);
                        let width = width as usize;
                        (top..bottom).flat_map(|row| pixels[(row * width + left) * 4..(row * width + right) * 4].to_vec()).collect()
                    }
                    None => hardware.picture(screen).unwrap().expect("copied").0.to_vec(),
                };
                assert_eq!(picture.len(), (20 * 24 * 4 * scale * scale) as usize);
                pictures.push(picture);
            }
            let colors: std::collections::HashSet<&[u8]> = pictures[0].chunks(4).collect();
            assert!(colors.len() > 8, "something worth comparing");
            assert!(pictures[0] == pictures[1], "at {scale}x");
        }
    }

    /// a draw in between goes into twins of the buffers it draws into, which
    /// start as the buffers were, after the title's own draw, and a transfer
    /// turns the twin's picture upright beside the screen's. a draw that did
    /// not move goes into both.
    #[cfg(feature = "vulkan")]
    #[test]
    fn pictures_in_between_go_into_twins() {
        use crate::format::ColorFormat::Rgba8;
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        // a band over x from left, half the target wide
        let band = |left: f32, color: Vec4| -> Vec<Vertex> {
            [[left, -1.0], [left + 1.0, -1.0], [left, 1.0], [left + 1.0, -1.0], [left + 1.0, 1.0], [left, 1.0]]
                .into_iter()
                .map(|[x, y]| Vertex { clip: [x, y, 0.0, 1.0], color, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() })
                .collect()
        };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: Rgba8,
            output_format: Rgba8,
        };
        // each scene drawn plainly, and with what moved in between
        let scene = |red: f32, between: Option<f32>| -> Option<(Vec<u8>, Option<Vec<u8>>)> {
            let device = hardware::own_device().ok()?;
            let mut hardware = hardware::Hardware::with_device(std::sync::Arc::new(device), true, None).unwrap();
            hardware.set_twins(between.is_some());
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            let pass = |resources: &mut Resources, pass| resources.hardware.as_mut().unwrap().pass = pass;
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(0.0, BLUE));
            if let Some(between) = between {
                pass(&mut resources, Pass::Twinned);
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(red, RED));
                pass(&mut resources, Pass::Twin);
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(between, RED));
                pass(&mut resources, Pass::Real);
            } else {
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(red, RED));
            }
            rasterize_shaded(&registers, &mut memory, &mut resources, &band(0.5, GREEN));
            let hardware = resources.hardware.as_mut().unwrap();
            assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
            let screen = hardware.screen(OUTPUT, (SIZE, SIZE), SIZE, Rgba8, &[]).expect("drawn on the GPU");
            let (own, _) = hardware.upright_pixels(screen, false).unwrap();
            let twin = between.map(|_| hardware.upright_pixels(screen, true).unwrap().0);
            Some((own, twin))
        };
        let Some((own, Some(between))) = scene(-1.0, Some(-0.5)) else { return };
        let (plain, _) = scene(-1.0, None).unwrap();
        let (halfway, _) = scene(-0.5, None).unwrap();
        let colors: std::collections::HashSet<&[u8]> = own.chunks(4).collect();
        assert_eq!(colors.len(), 3, "the background and both bands");
        assert!(own == plain, "the title's own picture");
        assert!(between == halfway, "the picture in between");
        assert!(own != between);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn pictures_in_between_render_at_mixed_twin_scale() {
        use crate::format::ColorFormat::Rgba8;
        const SIZE: u32 = 32;
        const OUTPUT: u32 = 0x10_0000;
        let band = |left: f32, color: Vec4| -> Vec<Vertex> {
            [[left, -1.0], [left + 1.0, -1.0], [left, 1.0], [left + 1.0, -1.0], [left + 1.0, 1.0], [left, 1.0]]
                .into_iter()
                .map(|[x, y]| Vertex { clip: [x, y, 0.0, 1.0], color, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() })
                .collect()
        };
        let mut registers = target_registers();
        registers[REG_VIEWPORT_WIDTH] = float24(SIZE as f32 / 2.0);
        registers[REG_VIEWPORT_HEIGHT] = float24(SIZE as f32 / 2.0);
        registers[REG_FRAMEBUFFER_DIMENSIONS] = SIZE | ((SIZE - 1) << 12);
        let transfer = hardware::Transfer {
            input: COLOR,
            output: OUTPUT,
            input_width: SIZE,
            output_width: SIZE,
            output_height: SIZE,
            copy: (SIZE, SIZE),
            scale: (1, 1),
            flip: false,
            input_linear: false,
            output_tiled: false,
            input_format: Rgba8,
            output_format: Rgba8,
        };
        let scene = |scale: u32, mixed: bool, red: f32, between: Option<f32>| -> Option<((Vec<u8>, u32), Option<(Vec<u8>, u32)>)> {
            let device = hardware::own_device().ok()?;
            let mut hardware = hardware::Hardware::with_device(std::sync::Arc::new(device), true, None).unwrap();
            hardware.mixed_twin_scale = mixed;
            assert_eq!(hardware.set_scale(scale), scale);
            hardware.set_twins(between.is_some());
            let mut resources = Resources { hardware: Some(hardware), ..Default::default() };
            let mut memory = ConsoleMemory::default();
            let pass = |resources: &mut Resources, pass| resources.hardware.as_mut().unwrap().pass = pass;
            rasterize_shaded(&registers, &mut memory, &mut resources, &cover(0.0, BLUE));
            if let Some(between) = between {
                pass(&mut resources, Pass::Twinned);
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(red, RED));
                pass(&mut resources, Pass::Twin);
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(between, RED));
                pass(&mut resources, Pass::Real);
            } else {
                rasterize_shaded(&registers, &mut memory, &mut resources, &band(red, RED));
            }
            rasterize_shaded(&registers, &mut memory, &mut resources, &band(0.5, GREEN));
            let hardware = resources.hardware.as_mut().unwrap();
            assert!(hardware.display_transfer(&mut memory, &transfer).unwrap());
            let screen = hardware.screen(OUTPUT, (SIZE, SIZE), SIZE, Rgba8, &[]).expect("drawn on the GPU");
            let real_gpu = hardware.screen_image(screen).unwrap().expect("real screen image");
            let own = hardware.upright_pixels(screen, false).unwrap();
            let twin = between.map(|_| {
                let twin_gpu = hardware.between_image(screen).unwrap().expect("twin screen image");
                assert_eq!(real_gpu.area, twin_gpu.area, "normalized screen UVs match across scales");
                hardware.upright_pixels(screen, true).unwrap()
            });
            Some((own, twin))
        };
        // 3x Real + 2x Twin: Real matches 3x reference, Twin matches 2x reference!
        let Some(((own_3x, own_w), Some((twin_2x, twin_w)))) = scene(3, true, -1.0, Some(-0.5)) else { return };
        let ((ref_3x, _), _) = scene(3, false, -1.0, None).unwrap();
        let ((ref_2x, _), _) = scene(2, false, -0.5, None).unwrap();
        assert_eq!(own_w, SIZE * 3);
        assert_eq!(twin_w, SIZE * 2);
        assert_eq!(own_3x.len(), (SIZE * 3 * SIZE * 3 * 4) as usize);
        assert_eq!(twin_2x.len(), (SIZE * 2 * SIZE * 2 * 4) as usize);
        assert_eq!(own_3x, ref_3x, "3x real frame is full 3x sharpness");
        assert_eq!(twin_2x, ref_2x, "2x twin frame matches 2x reference");
    }

    /// a quad covering x from `x.0` to `x.1` and y from `y.0` to `y.1` in
    /// normalized device coordinates, as a list of two triangles.
    fn quad(x: (f32, f32), y: (f32, f32), color: Vec4) -> Vec<Vertex> {
        let vertex = |x: f32, y: f32| Vertex { clip: [x, y, -0.5, 1.0], color, texcoords: [[0.0; 2]; 3], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() };
        [(x.0, y.0), (x.1, y.0), (x.1, y.1), (x.0, y.0), (x.1, y.1), (x.0, y.1)].into_iter().map(|(x, y)| vertex(x, y)).collect()
    }

    /// a portrait on one side of the bottom screen goes to the top screen's
    /// corner on that side, scaled by 0.4, with the edge the bottom screen
    /// cuts on the top screen's bottom edge. y runs leftward from the right
    /// edge on both screens, so a low y is the right side.
    #[test]
    fn a_portrait_goes_to_the_bottom_corner_of_its_side() {
        let mut registers = target_registers();
        sized(&mut registers, 240, 320);
        let mut resources = Resources::default();
        let top = [0x1F08_0000 >> 3, 0, 256 | (511 << 12)];
        resources.portraits.top = Some(top);
        // x past the bottom screen's edge, which cuts it at 0, y from 40 to 120
        mirror_portrait(&registers, &mut resources, Vertices::Shaded(&quad((-1.5, 0.0), (-0.75, -0.25), RED)));
        // y from 200 to 280, all of it on the screen, x from 60 to 180
        mirror_portrait(&registers, &mut resources, Vertices::Shaded(&quad((-0.5, 0.5), (0.25, 0.75), RED)));
        let placed = |side: usize| {
            let [draw] = &resources.portraits.sides[side].incoming[..] else { panic!("one draw on side {side}") };
            let xy = draw.registers[REG_VIEWPORT_XY];
            let signed10 = |value: u32| ((value & 0x3FF) << 22) as i32 >> 22;
            assert_eq!(draw.registers[REG_COLOR_BUFFER_ADDRESS], top[0]);
            assert_eq!(draw.registers[REG_FRAMEBUFFER_DIMENSIONS], top[2]);
            assert_eq!(read_float24(&draw.registers, REG_VIEWPORT_WIDTH), 48.0);
            assert_eq!(read_float24(&draw.registers, REG_VIEWPORT_HEIGHT), 64.0);
            assert_eq!(draw.registers[REG_DEPTH_COLOR_MASK] & (1 | 1 << 12), 0, "no depth");
            (signed10(xy), signed10(xy >> 16))
        };
        // the cut edge at 0, the side against its edge: -40 * 0.4
        assert_eq!(placed(0), (0, -16));
        // 60 * 0.4 below, and 400 - 280 * 0.4 along
        assert_eq!(placed(1), (-24, 288));
    }

    /// a portrait as Kid Icarus Uprising draws it, its picture from x -16 to
    /// the bottom screen's top edge at 240, fades out over its last rows on
    /// the top screen, so where the picture is cut does not show as a line.
    #[test]
    fn a_portrait_fades_out_at_the_bottom_screens_top_edge() {
        let mut registers = target_registers();
        sized(&mut registers, 240, 320);
        let mut resources = Resources::default();
        resources.portraits.top = Some([0x1F08_0000 >> 3, 0, 256 | (511 << 12)]);
        let quad = quad((-16.0 / 120.0 - 1.0, 1.0), (-0.75, -0.25), RED);
        mirror_portrait(&registers, &mut resources, Vertices::Shaded(&quad));
        let [draw] = &resources.portraits.sides[0].incoming[..] else { panic!("one draw on the right") };
        assert_eq!((draw.registers[REG_PRIMITIVE_CONFIG] >> 8) & 3, 0, "a triangle list");
        assert_eq!(draw.vertices.len() % 3, 0);
        let x = |vertex: &Vertex| (vertex.clip[0] / vertex.clip[3] * 0.5 + 0.5) * 240.0;
        // every corner's alpha follows its rows: whole up to the band, none
        // at the edge, and in between as far as it is into the band
        for vertex in &draw.vertices {
            let expected = ((240.0 - x(vertex)) / PORTRAIT_FADE).clamp(0.0, 1.0);
            assert!((vertex.color[3] - expected).abs() < 1e-4, "alpha {} at x {}", vertex.color[3], x(vertex));
        }
        // the band has corners of its own on its lower edge, so the ramp
        // runs across it alone
        assert!(draw.vertices.iter().any(|vertex| (x(vertex) - (240.0 - PORTRAIT_FADE)).abs() < 1e-3));
        // and all of the picture is still there: the triangles cover what
        // the quad covered
        let area = |vertices: &[Vertex]| -> f32 {
            vertices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| ((t[1].clip[0] - t[0].clip[0]) * (t[2].clip[1] - t[0].clip[1]) - (t[2].clip[0] - t[0].clip[0]) * (t[1].clip[1] - t[0].clip[1])).abs() / 2.0)
                .sum()
        };
        assert!((area(&draw.vertices) - area(&quad)).abs() < 1e-4);
    }

    /// a draw inside the counters' island goes beside the score on the top
    /// screen; one over the whole bottom screen, a background, stays there.
    #[test]
    fn the_battle_display_goes_to_its_island_on_the_top_screen() {
        let mut registers = target_registers();
        sized(&mut registers, 240, 320);
        let mut resources = Resources::default();
        resources.portraits.top = Some([0x1F08_0000 >> 3, 0, 256 | (511 << 12)]);
        // across 200 to 300 and down 5 to 40 of the bottom screen's 320x240,
        // whose buffer runs up from its bottom edge and left from its right
        mirror_hud(&registers, &mut resources, Vertices::Shaded(&quad((0.6667, 0.9583), (-0.875, -0.25), RED)));
        mirror_hud(&registers, &mut resources, Vertices::Shaded(&quad((-1.0, 1.0), (-1.0, 1.0), RED)));
        let [draw] = &resources.portraits.hud[..] else { panic!("one draw taken, not the background") };
        // drawn over the whole top screen, its vertices moved there
        assert_eq!(draw.registers[REG_VIEWPORT_XY], 0);
        assert_eq!((read_float24(&draw.registers, REG_VIEWPORT_WIDTH), read_float24(&draw.registers, REG_VIEWPORT_HEIGHT)), (120.0, 200.0));
        // the island, 196 to 320 across and 0 to 52 down, lands 232 to 301.44
        // across (beside SCORE) and 3 to 32.12 down the top screen, scaled by
        // 0.56: the corner 300 across and 40 down goes to 290.24 across and
        // 25.4 down, which in the top screen's buffer is 214.6 up from its
        // bottom edge and 109.76 left of its right one
        let screen = to_screen(&draw.vertices[0], (0.0, 0.0, 240.0, 400.0)).unwrap();
        assert!((screen.x - 214.6).abs() < 0.1 && (screen.y - 109.76).abs() < 0.1, "at {}, {}", screen.x, screen.y);
    }

    /// bottom-screen text from the 512x128 RGBA8 text buffer routes How to
    /// Play's top dialogue band to `dialogue`, while power name callouts and
    /// item pickup cards stay in their respective HUD islands.
    #[test]
    fn bottom_text_and_item_pickup_card_route_to_hud_islands() {
        let mut registers = target_registers();
        sized(&mut registers, 240, 320);
        let base = TEXTURE_UNIT_BASES[0];
        registers[REG_TEXTURE_CONFIG] = 1;
        registers[base + 1] = 512 << 16 | 128;
        registers[base + 13] = 0; // Rgba8
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources::default();
        resources.portraits.in_play = true;
        resources.portraits.in_battle = true;
        resources.portraits.top_subtitle_quiet = 60;
        resources.portraits.top = Some([0x1F08_0000 >> 3, 0, 256 | (511 << 12)]);

        let bottom_rect = |left: f32, upper: f32, right: f32, lower: f32| {
            rect(240.0 - lower, 320.0 - right, 240.0 - upper, 320.0 - left, (240.0, 320.0))
        };

        // 1. How to Play top-of-bottom-screen tutorial text (`upper = 14..52`) -> dialogue
        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(12.0, 14.0, 290.0, 52.0)));
        assert_eq!(resources.portraits.dialogue.len(), 1);
        assert_eq!(resources.portraits.hud.len(), 0);

        // 2. Power Name callout (`[18..120, 166..179]` inside speech bubble `56..78`) routes
        //    according to PowerLabelMode, while Practice Range stat text above the bubble
        //    (`[14..152, 148..161]`) is ignored.
        resources.portraits.power_bubble_x = Some([56.0, 78.0]);
        set_power_label_mode(PowerLabelMode::Bottom);
        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(14.0, 148.0, 152.0, 161.0)));
        assert_eq!(resources.portraits.power_label.len(), 0);
        assert_eq!(resources.portraits.hud.len(), 0);

        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(18.0, 166.0, 120.0, 179.0)));
        assert_eq!(resources.portraits.dialogue.len(), 1);
        assert_eq!(resources.portraits.power_label.len(), 1);
        assert_eq!(resources.portraits.hud.len(), 0);
        assert_eq!(&resources.portraits.power_label[0].vertices[0].color[..3], &[1.0, 1.0, 1.0]);

        set_power_label_mode(PowerLabelMode::Off);
        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(18.0, 166.0, 120.0, 179.0)));
        assert_eq!(resources.portraits.power_label.len(), 1);
        assert_eq!(resources.portraits.hud.len(), 0);

        set_power_label_mode(PowerLabelMode::AboveBar);
        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(18.0, 166.0, 120.0, 179.0)));
        assert_eq!(resources.portraits.dialogue.len(), 1);
        assert_eq!(resources.portraits.hud.len(), 1);
        set_power_label_mode(PowerLabelMode::Bottom);

        // 3. Item Collection text (`[105..284, 118..163]`) + 64x64 Etc1 icon (`[36..100, 118..182]`) -> ItemCard island
        rasterize(&registers, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(105.0, 118.0, 284.0, 163.0)));
        let mut icon_regs = registers.clone();
        icon_regs[base + 1] = 64 << 16 | 64;
        icon_regs[base + 13] = 12; // Etc1
        rasterize(&icon_regs, &mut memory, &mut resources, Vertices::Shaded(&bottom_rect(36.0, 118.0, 100.0, 182.0)));
        assert_eq!(resources.portraits.dialogue.len(), 1);
        assert_eq!(resources.portraits.hud.len(), 3);
        let icon_screen = to_screen(&resources.portraits.hud[2].vertices[0], (0.0, 0.0, 240.0, 400.0)).unwrap();
        // bottom-right of `[36..100, 118..182]` -> `x_top < 395`, `y_top` in `74..120` (`screen.x` in `120..166`)
        assert!(icon_screen.x > 120.0 && icon_screen.x < 166.0 && icon_screen.y < 130.0, "icon at {:?}", icon_screen);
    }

    /// a vertex at x, y of a viewport w by h, with texture coordinates u, v.
    fn at(x: f32, y: f32, (w, h): (f32, f32), [u, v]: [f32; 2]) -> Vertex {
        let clip = [x / w * 2.0 - 1.0, y / h * 2.0 - 1.0, -0.5, 1.0];
        Vertex { clip, color: RED, texcoords: [[u, v], [0.0; 2], [0.0; 2]], quaternion: [0.0, 0.0, 0.0, 1.0], view: [0.0; 3], ..Default::default() }
    }

    /// a quad from x0, y0 to x1, y1 of a viewport, as two triangles.
    fn rect(x0: f32, y0: f32, x1: f32, y1: f32, size: (f32, f32)) -> Vec<Vertex> {
        [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)].into_iter().map(|(x, y)| at(x, y, size, [0.0; 2])).collect()
    }

    /// letters written into the text buffer make its lines, row bands with
    /// how far across they go, and the clearing before new text forgets them.
    #[test]
    fn the_text_buffer_lines_follow_the_letters_written() {
        let mut registers = target_registers();
        sized(&mut registers, 512, 128);
        let base = TEXTURE_UNIT_BASES[0];
        registers[REG_TEXTURE_CONFIG] = 1;
        registers[base + 13] = 11; // A4, the font
        let mut resources = Resources::default();
        let size = (512.0, 128.0);
        // two batches of letters, the second on both lines
        let mut letters = rect(17.0, 35.0, 108.0, 50.0, size);
        letters.extend(rect(17.0, 19.0, 60.0, 34.0, size));
        note_text(&registers, &mut resources, Vertices::Shaded(&letters));
        note_text(&registers, &mut resources, Vertices::Shaded(&rect(113.0, 35.0, 215.0, 50.0, size)));
        let mut lines = resources.portraits.text_lines.clone();
        lines.sort_by(|a, b| a.bottom.total_cmp(&b.bottom));
        assert_eq!(
            lines,
            [
                TextLine { bottom: 19.0, top: 34.0, left: 17.0, right: 60.0 },
                TextLine { bottom: 35.0, top: 50.0, left: 17.0, right: 215.0 }
            ]
        );
        // the clearing, no font, over the text
        registers[base + 13] = 0;
        note_text(&registers, &mut resources, Vertices::Shaded(&rect(3.0, 15.0, 287.0, 58.0, size)));
        assert!(resources.portraits.text_lines.is_empty());
    }

    /// each line of the subtitles slides so its middle is the top screen's
    /// middle, the quads cut into one slice per line. the quad is the one
    /// Kid Icarus Uprising draws: across 97.6 to 361 of the top screen's
    /// buffer, its text buffer's u 0.522 to 0.008 that way.
    #[test]
    fn each_subtitle_line_goes_to_the_middle_of_the_screen() {
        let size = (240.0, 400.0);
        let corner = |x: f32, y: f32, u: f32, v: f32| at(x, y, size, [u, v]);
        let (a, b, c, d) = (corner(56.0, 361.0, 0.008, 0.445), corner(56.0, 97.6, 0.522, 0.445), corner(15.1, 361.0, 0.008, 0.126), corner(15.1, 97.6, 0.522, 0.126));
        let quad = [a, b, c, b, d, c];
        let lines = [TextLine { bottom: 35.0, top: 50.0, left: 17.0, right: 215.0 }, TextLine { bottom: 19.0, top: 34.0, left: 17.0, right: 183.0 }];
        let centered = centered_subtitle(&quad, &lines).expect("centered");
        assert_eq!(centered.len(), 12, "two slices of the quad");
        for (slice, line) in centered.chunks_exact(6).zip([lines[1], lines[0]]) {
            // where the line's middle is now, between the slice's ends in u
            let low_u = slice.iter().find(|v| (v.texcoords[0][0] - 0.008).abs() < 1e-4).unwrap();
            let high_u = slice.iter().find(|v| (v.texcoords[0][0] - 0.522).abs() < 1e-4).unwrap();
            let middle = ((line.left + line.right) / 2.0 / 512.0 - 0.008) / (0.522 - 0.008);
            let y = low_u.clip[1] + (high_u.clip[1] - low_u.clip[1]) * middle;
            assert!(y.abs() < 1e-4, "line {line:?} at {y}");
            // and it covers its own rows of the text buffer
            let vs: Vec<f32> = slice.iter().map(|v| v.texcoords[0][1] * 128.0).collect();
            let (low_v, high_v) = (vs.iter().copied().fold(f32::MAX, f32::min), vs.iter().copied().fold(f32::MIN, f32::max));
            assert!(low_v <= line.bottom && high_v >= line.top, "{line:?} in {low_v}..{high_v}");
        }
        // no lines known: the subtitles are left as the title drew them
        assert!(centered_subtitle(&quad, &[]).is_none());
        assert!(is_top_subtitle_quad(&quad), "real KIU 2-line battle subtitle quad must pass");

        // Short 1-line battle subtitles (narrow width and/or tight single-row height)
        // must also pass `is_top_subtitle_quad` and center cleanly.
        let one_line_short = [
            corner(44.0, 361.0, 0.008, 0.445),
            corner(44.0, 313.0, 0.102, 0.445),
            corner(26.0, 361.0, 0.008, 0.281),
            corner(44.0, 313.0, 0.102, 0.445),
            corner(26.0, 313.0, 0.102, 0.281),
            corner(26.0, 361.0, 0.008, 0.281),
        ];
        assert!(is_top_subtitle_quad(&one_line_short), "short 1-line battle subtitle quad must pass");
        let one_line = [TextLine { bottom: 38.0, top: 53.0, left: 17.0, right: 49.0 }];
        let centered_one = centered_subtitle(&one_line_short, &one_line).expect("1-line centered");
        assert_eq!(centered_one.len(), 6, "single slice for 1-line subtitle");
        let low_u = centered_one.iter().find(|v| (v.texcoords[0][0] - 0.008).abs() < 1e-4).unwrap();
        let high_u = centered_one.iter().find(|v| (v.texcoords[0][0] - 0.102).abs() < 1e-4).unwrap();
        let middle = ((one_line[0].left + one_line[0].right) / 2.0 / 512.0 - 0.008) / (0.102 - 0.008);
        let y = low_u.clip[1] + (high_u.clip[1] - low_u.clip[1]) * middle;
        assert!(y.abs() < 1e-4, "1-line subtitle centered at {y}");

        // Chapter End bonus labels (mid-screen, right-corner badges, or batched multi-row UVs)
        // must NOT be intercepted or shifted to screen center.
        let bonus_row = [
            corner(140.0, 360.0, 0.01, 0.25),
            corner(140.0, 210.0, 0.30, 0.25),
            corner(122.0, 360.0, 0.01, 0.12),
            corner(140.0, 210.0, 0.30, 0.25),
            corner(122.0, 210.0, 0.30, 0.12),
            corner(122.0, 360.0, 0.01, 0.12),
        ];
        assert!(!is_top_subtitle_quad(&bonus_row), "mid-screen chapter-end bonus row must not be treated as subtitle");

        let right_corner_badge = [
            corner(44.0, 150.0, 0.01, 0.25),
            corner(44.0, 20.0, 0.26, 0.25),
            corner(26.0, 150.0, 0.01, 0.12),
            corner(44.0, 20.0, 0.26, 0.25),
            corner(26.0, 20.0, 0.26, 0.12),
            corner(26.0, 150.0, 0.01, 0.12),
        ];
        assert!(!is_top_subtitle_quad(&right_corner_badge), "bottom-right corner badge must not be treated as subtitle");

        let mut batched_rows = quad.to_vec();
        batched_rows.extend_from_slice(&[
            corner(56.0, 361.0, 0.008, 0.700),
            corner(56.0, 97.6, 0.522, 0.700),
            corner(15.1, 361.0, 0.008, 0.500),
            corner(56.0, 97.6, 0.522, 0.700),
            corner(15.1, 97.6, 0.522, 0.500),
            corner(15.1, 361.0, 0.008, 0.500),
        ]);
        assert!(!is_top_subtitle_quad(&batched_rows), "batched multi-UV rows must not be treated as a single subtitle");
        assert!(centered_subtitle(&batched_rows, &lines).is_none());
    }

    /// a portrait fades in over 8 frames, the next picture fades in over it
    /// while it fades out, and one the game stops drawing fades out.
    #[test]
    fn a_portrait_fades_in_over_and_out() {
        let mut registers = target_registers();
        // as Kid Icarus Uprising blends its portraits, by their alpha
        registers[REG_COLOR_OPERATION] = 0x00E4_0100;
        registers[REG_BLEND_FUNC] = 0x0176_0000;
        let mut memory = ConsoleMemory::default();
        let mut resources = Resources::default();
        let draw = |picture: u32| {
            let mut registers = registers.clone();
            registers[TEXTURE_UNIT_BASES[0] + 4] = picture;
            PortraitDraw { registers, vertices: cover(-0.5, RED) }
        };
        let alphas = |resources: &Resources| {
            let side = &resources.portraits.sides[0];
            (side.shown.as_ref().map(|faded| faded.alpha), side.leaving.as_ref().map(|faded| faded.alpha))
        };

        resources.portraits.sides[0].incoming.push(draw(1));
        assert!(flush_portraits(&mut memory, &mut resources));
        assert_eq!(alphas(&resources), (Some(0.125), None));
        let red = pixels(&mut memory)[0][0];
        assert!((28..=36).contains(&red), "an eighth of red over black, not {red}");

        for _ in 0..7 {
            resources.portraits.sides[0].incoming.push(draw(1));
            flush_portraits(&mut memory, &mut resources);
        }
        assert_eq!(alphas(&resources), (Some(PORTRAIT_ALPHA), None), "it stops short of all, to see through");

        resources.portraits.sides[0].incoming.push(draw(2));
        flush_portraits(&mut memory, &mut resources);
        assert_eq!(alphas(&resources), (Some(0.125), Some(PORTRAIT_ALPHA - 0.125)), "the new one over the old");

        // the game stops drawing the portrait on its bottom-screen frames:
        // the new one had an eighth to lose, the old one the rest
        resources.portraits.bottom_seen = true;
        assert!(flush_portraits(&mut memory, &mut resources));
        assert_eq!(alphas(&resources), (None, Some(PORTRAIT_ALPHA - 0.25)));
        for _ in 0..4 {
            resources.portraits.bottom_seen = true;
            assert!(flush_portraits(&mut memory, &mut resources));
        }
        resources.portraits.bottom_seen = true;
        assert!(!flush_portraits(&mut memory, &mut resources), "the last step leaves nothing to draw");
        assert_eq!(alphas(&resources), (None, None));
    }

    #[test]
    fn title_sprite_cache_miss_does_not_deadlock() {
        let mut cache = TextureCache::default();
        let data = vec![0xAAu8; 8 * 8 * 4];
        let key = (0x1002, TextureFormat::Rgba8, 8, 8);
        let (first, _) = cache.decoded(key, &data);
        assert!(first.len() >= 16 * 16);
        let mut cache2 = TextureCache::default();
        let (second, _) = cache2.decoded(key, &data);
        assert!(Arc::ptr_eq(&first, &second), "second miss in TextureCache reuses TITLE_CACHE entry");
    }

    fn top_tex_regs(width: u32, height: u32, fmt: TextureFormat) -> Vec<u32> {
        let mut regs = vec![0u32; 0x300];
        regs[REG_FRAMEBUFFER_DIMENSIONS] = 240 | (399 << 12);
        regs[REG_VIEWPORT_WIDTH] = float24(120.0);
        regs[REG_VIEWPORT_HEIGHT] = float24(200.0);
        regs[REG_TEXTURE_CONFIG] = 1;
        let base = TEXTURE_UNIT_BASES[0];
        regs[base + 1] = (width << 16) | height;
        regs[base + 13] = match fmt {
            TextureFormat::Rgb565 => 3,
            TextureFormat::Rgba4 => 4,
            TextureFormat::La4 => 9,
            TextureFormat::L4 => 10,
            TextureFormat::Etc1 => 12,
            TextureFormat::Etc1A4 => 13,
            _ => unreachable!("unused test format {fmt:?}"),
        };
        regs
    }

    #[test]
    fn mode_select_detects_all_hovered_modes_including_after_vault() {
        // Every Mode Select item shares `256x32 Etc1 -> [0..=2 mode silhouettes] -> 256x256 L4 -> 32x8 Rgb565`.
        let cases: &[(&str, &[(u32, u32, TextureFormat)])] = &[
            ("Solo", &[(256, 128, TextureFormat::Etc1)]),
            ("Vault", &[(128, 128, TextureFormat::Etc1)]),
            ("Options", &[(128, 64, TextureFormat::Etc1)]),
            ("StreetPass", &[(32, 32, TextureFormat::Rgb565)]),
            ("NoSilhouette", &[]),
        ];
        for &(name, silhouettes) in cases {
            let mut portraits = Portraits::default();
            note_title_draw(&top_tex_regs(256, 32, TextureFormat::Etc1), &mut portraits);
            for &(w, h, fmt) in silhouettes {
                note_title_draw(&top_tex_regs(w, h, fmt), &mut portraits);
            }
            note_title_draw(&top_tex_regs(256, 256, TextureFormat::L4), &mut portraits);
            note_title_draw(&top_tex_regs(256, 256, TextureFormat::L4), &mut portraits);
            note_title_draw(&top_tex_regs(32, 8, TextureFormat::Rgb565), &mut portraits);
            assert!(
                portraits.mode_active && portraits.mode_top_seen && portraits.mode_top_latched,
                "Mode Select should activate immediately when hovering {name}"
            );
        }
    }

    #[test]
    fn solo_hub_detects_both_viridi_and_palutena_domains() {
        // Viridi's Sanctuary: `16x32 Rgb565 -> 64x64 Rgba4 -> 128x128 L4`
        let mut viridi = Portraits::default();
        note_title_draw(&top_tex_regs(16, 32, TextureFormat::Rgb565), &mut viridi);
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut viridi);
        note_title_draw(&top_tex_regs(128, 128, TextureFormat::L4), &mut viridi);
        assert!(viridi.solo_active && viridi.solo_top_seen, "Viridi Solo Hub detected");

        // Palutena's Temple: `16x16 Etc1 -> 64x64 Rgba4 (vignette) -> 64x64 Rgba4 (hearts pouch)`
        let mut palutena = Portraits::default();
        note_title_draw(&top_tex_regs(16, 16, TextureFormat::Etc1), &mut palutena);
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut palutena);
        assert!(!palutena.solo_active, "single 64x64 Rgba4 must not trigger Solo Hub");
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut palutena);
        assert!(palutena.solo_active && palutena.solo_top_seen, "Palutena Solo Hub detected");

        // Viridi fallback if `16x32 Rgb565` is frustum-culled: `64x64 Rgba4 -> 128x128 L4 -> 64x64 Rgba4`
        let mut viridi_culled = Portraits::default();
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut viridi_culled);
        note_title_draw(&top_tex_regs(128, 128, TextureFormat::L4), &mut viridi_culled);
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut viridi_culled);
        assert!(viridi_culled.solo_active && viridi_culled.solo_top_seen, "Viridi fallback detected");

        // File Select must not false-trigger Solo Hub
        let mut file_select = Portraits::default();
        file_select.file_top_seen = true;
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut file_select);
        note_title_draw(&top_tex_regs(64, 64, TextureFormat::Rgba4), &mut file_select);
        assert!(!file_select.solo_active, "File Select must not trigger Solo Hub");
    }

    #[test]
    fn chapter_select_does_not_false_trigger_on_arms_altar_redeem_hearts() {
        // Arms Altar hovering Redeem Hearts draws `512x256 Etc1` (temple backdrop)
        // followed by a `128x128 Etc1A4` 3D weapon texture, without Chapter Select's
        // `256x64 Etc1` stage banner.
        let mut altar = Portraits::default();
        note_title_draw(&top_tex_regs(512, 256, TextureFormat::Etc1), &mut altar);
        note_title_draw(&top_tex_regs(128, 128, TextureFormat::Etc1A4), &mut altar);
        assert!(!altar.chapter_active && !altar.chapter_top_seen, "Arms Altar Redeem Hearts must not trigger Chapter Select");

        // Real Chapter Select (cleared chapter with optional `32x32 La4` laurel wreath):
        let mut chap_cleared = Portraits::default();
        note_title_draw(&top_tex_regs(512, 256, TextureFormat::Etc1), &mut chap_cleared);
        note_title_draw(&top_tex_regs(128, 128, TextureFormat::Etc1A4), &mut chap_cleared);
        note_title_draw(&top_tex_regs(32, 32, TextureFormat::La4), &mut chap_cleared);
        assert!(!chap_cleared.chapter_active, "must wait for 256x64 Etc1 stage banner");
        note_title_draw(&top_tex_regs(256, 64, TextureFormat::Etc1), &mut chap_cleared);
        assert!(chap_cleared.chapter_active && chap_cleared.chapter_top_seen, "cleared Chapter Select detected");

        // Real Chapter Select (uncleared chapter without laurel wreath):
        let mut chap_uncleared = Portraits::default();
        note_title_draw(&top_tex_regs(512, 256, TextureFormat::Etc1), &mut chap_uncleared);
        note_title_draw(&top_tex_regs(128, 128, TextureFormat::Etc1A4), &mut chap_uncleared);
        note_title_draw(&top_tex_regs(256, 64, TextureFormat::Etc1), &mut chap_uncleared);
        assert!(chap_uncleared.chapter_active && chap_uncleared.chapter_top_seen, "uncleared Chapter Select detected");
    }
}
