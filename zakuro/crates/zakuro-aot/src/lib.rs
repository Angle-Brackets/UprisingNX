//! Direct ARMv6K + Thumb-1 + VFPv2 -> AArch64 / x86_64 Ahead-of-Time (AOT)
//! compiler and position-independent executable runtime for Zakuro.

pub mod a64;
pub mod compiler_a64;
pub mod compiler_x64;
pub mod discover;
pub mod runtime;
pub mod x64;

pub use runtime::{
    AotExecContext, AotImage, CacheHeader, CompileOptions, ExecBuffer, LoadedAot, OverrideFn,
    PluginState, CACHE_MAGIC, FORMAT_VERSION,
};

pub fn compile_title(
    program_id: u64,
    exheader: &zakuro_fs::ExHeader,
    code: &[u8],
    override_addrs: &[u32],
    hints: &[u32],
    options: CompileOptions,
    progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    #[cfg(target_arch = "x86_64")]
    {
        compiler_x64::compile_title(program_id, exheader, code, override_addrs, hints, options, progress)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        compiler_a64::compile_title(program_id, exheader, code, override_addrs, hints, options, progress)
    }
}

pub fn compile_program(
    program_id: u64,
    text_hash: u64,
    program: &discover::Program,
    override_addrs: &[u32],
    options: CompileOptions,
    progress: impl Fn(usize, usize) + Sync,
) -> AotImage {
    #[cfg(target_arch = "x86_64")]
    {
        compiler_x64::compile_program(program_id, text_hash, program, override_addrs, options, progress)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        compiler_a64::compile_program(program_id, text_hash, program, override_addrs, options, progress)
    }
}
