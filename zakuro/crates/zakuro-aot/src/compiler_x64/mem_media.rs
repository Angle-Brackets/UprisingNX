//! x86_64 AOT lowering for ARM memory transfers, swap/exclusive, media/SIMD, and dual-multiply ops.

#![allow(dead_code, unused_variables)]

use super::FnEmitter;

impl FnEmitter<'_> {
    pub(crate) fn arm_single_transfer(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn arm_extra_transfer(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn arm_block_transfer(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn arm_swap(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn arm_exclusive(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }

    pub(crate) fn arm_media(&mut self, _a: u32, _op: u32) -> bool {
        unimplemented!()
    }
}
