//! the time the GPU's emulation spends in a frame, by what it is on, so a
//! slow frame can say where its time went.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// nanoseconds, since the last take.
pub static DRAWING: AtomicU64 = AtomicU64::new(0);
pub static SHADING: AtomicU64 = AtomicU64::new(0);
pub static TEXTURES: AtomicU64 = AtomicU64::new(0);
pub static WAITING: AtomicU64 = AtomicU64::new(0);
pub static DRAWS: AtomicU64 = AtomicU64::new(0);
/// the rest of the console's frame, which zakuro-core times: its recompiled
/// code, its services and syscalls, the end of the frame and the sound.
pub static RECOMPILED: AtomicU64 = AtomicU64::new(0);
pub static SERVICES: AtomicU64 = AtomicU64::new(0);
pub static FRAME_END: AtomicU64 = AtomicU64::new(0);
pub static AUDIO: AtomicU64 = AtomicU64::new(0);
/// what the GPU drew brought back for the CPU's reads and writes, and how often.
pub static SYNC: AtomicU64 = AtomicU64::new(0);
pub static SYNCS: AtomicU64 = AtomicU64::new(0);
/// time in the host GPU's side of a draw (Hardware::draw: state, pipelines,
/// uploads, the Vulkan recording), inside DRAWING.
pub static HARDWARE: AtomicU64 = AtomicU64::new(0);
/// time the GPU's thread waited for vertices a helper was shading, see
/// raster::finish_pending.
pub static SHADE_WAIT: AtomicU64 = AtomicU64::new(0);
/// time the shader compile thread spent compiling or reading back code, and
/// how many programs, since the last take: not on the GPU's thread, but on
/// the cores it shares.
pub static JIT_NS: AtomicU64 = AtomicU64::new(0);
pub static JIT_PROGRAMS: AtomicU64 = AtomicU64::new(0);
/// the renderer waiting for the Vulkan queue the presenter holds, in
/// nanoseconds, and how long the presenter held it.
pub static QUEUE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
pub static QUEUE_HELD_NS: AtomicU64 = AtomicU64::new(0);
/// the CPU's side waiting for the GPU's thread to finish, by who waits: the
/// scheduler with everything blocked, a guarded read or write, anything that
/// reaches for the GPU, and how long, in nanoseconds, and how often.
pub static BARRIER_NS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
pub static BARRIER_COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
pub const BARRIER_NAMES: [&str; 3] = ["idle", "guard", "deref"];
/// write-backs of what the GPU drew to guest memory, each a wait for the
/// GPU, by what asked for them (see WRITE_BACK_NAMES), and their time.
pub static WRITE_BACKS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub static WRITE_BACK: AtomicU64 = AtomicU64::new(0);
pub const WRITE_BACK_NAMES: [&str; 6] = ["overlap", "beside", "fill", "flush", "sync", "other"];
pub const BACK_OVERLAP: usize = 0;
pub const BACK_BESIDE: usize = 1;
pub const BACK_FILL: usize = 2;
pub const BACK_FLUSH: usize = 3;
pub const BACK_SYNC: usize = 4;

/// draws that took a slower way while a faster one was not ready, by kind,
/// see FALLBACK_NAMES: shaded on the CPU, a program still being translated,
/// the generic pipeline, the pipeline without the lighting's constants.
pub static FALLBACKS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
pub const FALLBACK_NAMES: [&str; 4] = ["cpu shaded", "translating", "generic", "unspecialized"];
pub const FALL_CPU: usize = 0;
pub const FALL_TRANSLATING: usize = 1;
pub const FALL_GENERIC: usize = 2;
pub const FALL_UNSPECIALIZED: usize = 3;

/// counts a draw that fell back, see FALLBACKS.
pub fn fell_back(kind: usize) {
    FALLBACKS[kind].fetch_add(1, Ordering::Relaxed);
}

/// counts a write-back of surfaces for a reason, when there are any.
pub fn write_back(why: usize, surfaces: usize) {
    if surfaces > 0 {
        WRITE_BACKS[why].fetch_add(1, Ordering::Relaxed);
    }
}

/// what a frame spent, in milliseconds, and the draws it made.
#[derive(Debug, Clone, Copy, Default)]
pub struct Spent {
    pub drawing: f64,
    pub shading: f64,
    pub textures: f64,
    pub waiting: f64,
    pub draws: u64,
    pub recompiled: f64,
    pub services: f64,
    pub frame_end: f64,
    pub audio: f64,
    pub sync: f64,
    pub syncs: u64,
    pub write_backs: [u64; 6],
    pub write_back: f64,
    pub hardware: f64,
    pub shade_wait: f64,
    pub jit: f64,
    pub jit_programs: u64,
    pub fallbacks: [u64; 4],
}

/// what was spent since the last take, starting afresh.
pub fn take() -> Spent {
    let ms = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed) as f64 / 1e6;
    Spent {
        drawing: ms(&DRAWING),
        shading: ms(&SHADING),
        textures: ms(&TEXTURES),
        waiting: ms(&WAITING),
        draws: DRAWS.swap(0, Ordering::Relaxed),
        recompiled: ms(&RECOMPILED),
        services: ms(&SERVICES),
        frame_end: ms(&FRAME_END),
        audio: ms(&AUDIO),
        sync: ms(&SYNC),
        syncs: SYNCS.swap(0, Ordering::Relaxed),
        write_backs: std::array::from_fn(|i| WRITE_BACKS[i].swap(0, Ordering::Relaxed)),
        write_back: ms(&WRITE_BACK),
        hardware: ms(&HARDWARE),
        shade_wait: ms(&SHADE_WAIT),
        jit: ms(&JIT_NS),
        jit_programs: JIT_PROGRAMS.swap(0, Ordering::Relaxed),
        fallbacks: std::array::from_fn(|i| FALLBACKS[i].swap(0, Ordering::Relaxed)),
    }
}

/// adds the time until it is dropped to a counter.
pub struct Timer(&'static AtomicU64, Instant);

pub fn time(counter: &'static AtomicU64) -> Timer {
    Timer(counter, Instant::now())
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.0.fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timer_adds_to_its_counter_until_taken() {
        take();
        {
            let _timer = time(&SHADING);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(take().shading >= 2.0);
        assert_eq!(take().shading, 0.0);
    }
}
