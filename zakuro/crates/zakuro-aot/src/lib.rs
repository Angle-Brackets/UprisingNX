//! Direct ARMv6K + Thumb-1 + VFPv2 -> AArch64 Ahead-of-Time (AOT) compiler and
//! position-independent executable runtime for Zakuro & UprisingNX.

pub mod a64;
pub mod compiler;
pub mod discover;
pub mod hints_kiu;
pub mod kiu_overrides;
pub mod runtime;

pub use compiler::{compile_program, compile_title, CompileOptions};
pub use hints_kiu::{KIU_USA_PROGRAM_ID, KIU_USA_TEXT_HASH};
pub use runtime::{
    AotExecContext, AotImage, CacheHeader, ExecBuffer, LoadedAot, CACHE_MAGIC, FORMAT_VERSION,
};
