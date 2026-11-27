//! Direct ARMv6K + Thumb-1 + VFPv2 -> AArch64 / x86_64 Ahead-of-Time (AOT)
//! compiler and position-independent executable runtime for Zakuro.

pub mod a64;
pub mod compiler;
pub mod discover;
pub mod runtime;

pub use compiler::{compile_program, compile_title, CompileOptions};
pub use runtime::{
    AotExecContext, AotImage, CacheHeader, ExecBuffer, LoadedAot, OverrideFn, PluginState,
    CACHE_MAGIC, FORMAT_VERSION,
};
