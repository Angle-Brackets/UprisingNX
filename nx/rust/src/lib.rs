//! UprisingNX's Switch frontend over Zakuro. The libnx C shell (`../c`) owns
//! the process, the window, input and nxlink; this side loads the game, runs
//! frames and presents them through NVK.

use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use zakuro_common::Screen;
use zakuro_core::services::hid::{InputState, PadState};
use zakuro_core::{loader, Config, FrameOutcome, System};
use zakuro_gpu::backend::vulkan::VulkanPresenter;
use zakuro_gpu::backend::{Overlay, PresentError, Presenter, ScreenImage, ScreenLayout};

struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}: {}", record.level(), record.target(), record.args());
        }
    }
    fn flush(&self) {}
}

static LOGGER: Logger = Logger;

/// what the C shell read from the controllers this frame, already mapped to
/// the 3DS's buttons (PadState bits) and axes.
#[repr(C)]
pub struct UnxInput {
    pub buttons: u32,
    pub circle_x: f32,
    pub circle_y: f32,
    pub c_stick_x: f32,
    pub c_stick_y: f32,
    pub touching: u32,
    pub touch_x: u16,
    pub touch_y: u16,
}

pub struct Unx {
    system: System,
    presenter: VulkanPresenter,
    frame: u64,
    window_start: Instant,
    window_frames: u32,
    window_emulating: Duration,
    worst: Duration,
}

fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the C shell passes NUL-terminated strings
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// starts Zakuro on the ROM at `rom`, presenting to `nwindow` (libnx's
/// nwindowGetDefault()); `data_dir` holds saves and the like. Null on
/// failure, with the reason logged.
///
/// # Safety
/// `nwindow` is a live NWindow for the life of the returned handle; the
/// strings are NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn unx_create(nwindow: *mut c_void, rom: *const c_char, data_dir: *const c_char, interpret: u32) -> *mut Unx {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);
    let (Some(rom), Some(data_dir)) = (cstr(rom), cstr(data_dir)) else {
        log::error!("unx: no ROM or data directory");
        return std::ptr::null_mut();
    };

    let started = Instant::now();
    let mut presenter = match VulkanPresenter::new_vi(nwindow, (1280, 720)) {
        Ok(presenter) => presenter,
        Err(error) => {
            log::error!("unx: Vulkan presenter failed, {error}");
            return std::ptr::null_mut();
        }
    };
    log::info!("unx: presenter {} up in {:?}", presenter.name(), started.elapsed());
    presenter.set_layout(ScreenLayout::TopOnly);

    let config = Config {
        data_dir: Some(PathBuf::from(&data_dir)),
        hardware_renderer: true,
        device: presenter.shared_device(),
        find_recompiled: interpret == 0,
        username: "Pit".into(),
        ..Config::default()
    };
    let started = Instant::now();
    let system = match loader::load(&rom, config) {
        Ok(system) => system,
        Err(error) => {
            log::error!("unx: could not load {rom}, {error}");
            return std::ptr::null_mut();
        }
    };
    log::info!("unx: loaded {rom} in {:?}", started.elapsed());
    Box::into_raw(Box::new(Unx {
        system,
        presenter,
        frame: 0,
        window_start: Instant::now(),
        window_frames: 0,
        window_emulating: Duration::ZERO,
        worst: Duration::ZERO,
    }))
}

/// a screen as the presenter takes it, straight from the GPU when the
/// renderer drew it on the shared device, else its pixels.
fn shown(system: &mut System, screen: Screen) -> (zakuro_core::Screen, Option<zakuro_gpu::GpuScreen>) {
    if let Some(gpu) = system.gpu_screen(screen) {
        let scale = system.gpu.scale();
        return ((std::sync::Arc::new(Vec::new()), screen.width() * scale, screen.height() * scale), Some(gpu));
    }
    (system.read_screen_scaled(screen), None)
}

/// runs one emulated frame and presents it. 0 to go on, 1 when the game
/// exited or faulted.
///
/// # Safety
/// `unx` came from unx_create; `input` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn unx_frame(unx: *mut Unx, input: *const UnxInput) -> u32 {
    let unx = &mut *unx;
    if let Some(input) = input.as_ref() {
        unx.system.set_input(InputState {
            buttons: PadState::from_bits_truncate(input.buttons),
            circle_x: input.circle_x,
            circle_y: input.circle_y,
            c_stick_x: input.c_stick_x,
            c_stick_y: input.c_stick_y,
            touch: (input.touching != 0).then_some((input.touch_x, input.touch_y)),
            tilt: [0.0, 0.0],
        });
    }

    let started = Instant::now();
    let outcome = unx.system.run_frame();
    let took = started.elapsed();
    unx.frame += 1;

    let (top, bottom) = (shown(&mut unx.system, Screen::Top), shown(&mut unx.system, Screen::Bottom));
    let blank = std::sync::Arc::new(Vec::new());
    let (((top_px, tw, th), top_gpu), ((bottom_px, bw, bh), bottom_gpu)) = (top, bottom);
    let result = unx.presenter.present(
        ScreenImage { width: tw, height: th, pixels: &top_px, gpu: top_gpu },
        ScreenImage { width: bw, height: bh, pixels: &bottom_px, gpu: bottom_gpu },
        ScreenImage { width: tw, height: th, pixels: &blank, gpu: None },
        &Overlay::default(),
    );
    if let Err(error) = result {
        if !matches!(error, PresentError::OutOfDate) {
            log::error!("unx: present failed, {error}");
        }
    }

    // a line every 2 s: emulated frames per second, how long emulating took
    // on average and at worst, and the interpreter's share
    unx.window_frames += 1;
    unx.window_emulating += took;
    unx.worst = unx.worst.max(took);
    let window = unx.window_start.elapsed();
    if window >= Duration::from_secs(2) {
        let frames = unx.window_frames as f64;
        println!(
            "unx: frame {} {:.1} fps, emulating {:.1} ms avg {:.1} ms worst, {} interpreted",
            unx.frame,
            frames / window.as_secs_f64(),
            unx.window_emulating.as_secs_f64() * 1000.0 / frames,
            unx.worst.as_secs_f64() * 1000.0,
            unx.system.interpreted_instructions,
        );
        unx.window_start = Instant::now();
        unx.window_frames = 0;
        unx.window_emulating = Duration::ZERO;
        unx.worst = Duration::ZERO;
    }

    match outcome {
        FrameOutcome::Completed => 0,
        FrameOutcome::Exited | FrameOutcome::Faulted => {
            log::warn!("unx: frame {} ended {outcome:?}", unx.frame);
            1
        }
    }
}

/// stops the game, writing what it keeps, and frees the handle.
///
/// # Safety
/// `unx` came from unx_create and isn't used again.
#[no_mangle]
pub unsafe extern "C" fn unx_destroy(unx: *mut Unx) {
    if !unx.is_null() {
        drop(Box::from_raw(unx));
    }
}
