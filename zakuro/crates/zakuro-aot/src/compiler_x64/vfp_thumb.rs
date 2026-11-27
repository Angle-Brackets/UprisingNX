//! x86_64 AOT lowering for VFPv2 floating-point and Thumb-1 instructions.

#![allow(dead_code, unused_variables)]

use super::FnEmitter;

impl FnEmitter<'_> {
    pub(crate) fn vfp_data_processing(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn vfp_register_transfer(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn vfp_load_store(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn lower_thumb(&mut self, _a: u32, _op: u32, _second: u32) -> bool {
        unimplemented!()
    }
}
