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
use zakuro_gpu::backend::{Overlay, PresentError, Presenter, ScreenFilter, ScreenImage, ScreenLayout};

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

/// the tables 3dsrecomp writes beside the code (`recomp_abi`,
/// `recomp_entries`, ...), from librecomp.a linked into the NRO.
#[repr(C)]
pub struct UnxRecomp {
    pub abi: u32,
    pub entries: *const c_void,
    pub entry_count: u32,
    pub modules: *const c_void,
    pub module_count: u32,
}

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
    window_presenting: Duration,
    worst: Duration,
    /// zakuro_gpu::stats summed over the window, in ms
    spent: [f64; 9],
    draws: u64,
    syncs: u64,
    write_backs: [u64; 6],
    write_back: f64,
    /// frames since the game's overrides last said it was in play (a battle),
    /// see IN_PLAY_QUIET; the layout follows it
    out_of_play: u32,
    /// frames since the bottom screen drew a large dialogue portrait
    since_portrait: u32,
    in_play: bool,
}

/// frames the overrides may stay quiet before the game counts as out of play
/// (a menu, a cutscene), as in Zakuro's desktop frontend
const IN_PLAY_QUIET: u32 = 10;
/// frames a conversation's big portrait may go undrawn before the game counts
/// as out of play: a battle's dialogue, before the player has control, while
/// the overrides are quiet; it is seen again each frame the portrait is up
const PORTRAIT_QUIET: u32 = 120;

extern "C" {
    fn nx_audio_init();
    fn nx_audio_play(samples: *const i16, count: u32);
    fn nx_audio_exit();
}

fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the C shell passes NUL-terminated strings
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// starts Zakuro on the ROM at `rom`, presenting to `nwindow` (libnx's
/// nwindowGetDefault()); `data_dir` holds saves and the like. `recomp` is the
/// statically linked recompiled code, or null to interpret. Null on failure,
/// with the reason logged.
///
/// # Safety
/// `nwindow` is a live NWindow for the life of the returned handle; the
/// strings are NUL-terminated; `recomp` is null or points to 3dsrecomp's
/// tables, which live as long as the program.
#[no_mangle]
pub unsafe extern "C" fn unx_create(nwindow: *mut c_void, rom: *const c_char, data_dir: *const c_char, recomp: *const UnxRecomp) -> *mut Unx {
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
    // out of play until the game's overrides say otherwise, see unx_frame
    presenter.set_layout(ScreenLayout::LargeTop);
    presenter.set_scaling(ScreenFilter::Smooth, false);
    nx_audio_init();

    let linked = recomp.as_ref().map(|r| {
        log::info!("unx: recompiled code linked in, ABI {}, {} entries, {} modules", r.abi, r.entry_count, r.module_count);
        zakuro_core::recompiled::Linked::new(0x0004000000030100, r.abi, r.entries, r.entry_count, r.modules, r.module_count)
    });
    let resolution = std::env::var("ZAKURO_RESOLUTION")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let config = Config {
        data_dir: Some(PathBuf::from(&data_dir)),
        hardware_renderer: true,
        resolution,
        device: presenter.shared_device(),
        linked,
        find_recompiled: false,
        username: "Pit".into(),
        ..Config::default()
    };
    let started = Instant::now();
    let system = match loader::load(&rom, config) {
        Ok(mut system) => {
            // what the recompiled code leaves to the interpreter, printed
            // every 10 s (diagnostics)
            system.interpreted_pcs = Some(Default::default());
            // where the linked code still leaves the interpreter, for
            // 3dsrecomp to make functions of (Zakuro only sets this up where
            // it finds a library dir, which Horizon has none of)
            if system.recompiled.is_some() && system.hints.is_none() {
                if let Some(title) = &system.title {
                    let text = &title.exheader.text;
                    let range = text.address..text.address + text.num_pages * 0x1000;
                    system.hints = Some(zakuro_core::hints::Hints::new(std::path::Path::new("sdmc:/switch/uprisingnx/kiu.so"), range));
                    log::info!("unx: noting interpreted code in sdmc:/switch/uprisingnx/kiu.hints");
                }
            }
            system
        }
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
        window_presenting: Duration::ZERO,
        worst: Duration::ZERO,
        spent: [0.0; 9],
        draws: 0,
        syncs: 0,
        write_backs: [0; 6],
        write_back: 0.0,
        out_of_play: u32::MAX,
        since_portrait: u32::MAX,
        in_play: false,
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

/// where a touch on the 1280x720 screen lands on the 3DS bottom screen's 320x240,
/// if it does. in play the bottom screen is not shown, and the touch aims the
/// reticle over a 4:3 area in the middle of the screen; in a menu it is the
/// small bottom screen of the layout, and only a touch there counts.
fn touch_on_bottom(in_play: bool, x: u16, y: u16) -> Option<(u16, u16)> {
    let (x, y) = (x as f32, y as f32);
    let (left, top, width, height) = if in_play {
        (160.0, 0.0, 960.0, 720.0)
    } else {
        let bottom = zakuro_gpu::layout(1280, 720, ScreenLayout::LargeTop, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        (bottom.x, bottom.y, bottom.width, bottom.height)
    };
    let (u, v) = ((x - left) / width, (y - top) / height);
    ((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)).then(|| ((u * 320.0) as u16, (v * 240.0) as u16))
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
            touch: (input.touching != 0).then(|| touch_on_bottom(unx.in_play, input.touch_x, input.touch_y)).flatten(),
            tilt: [0.0, 0.0],
        });
    }

    // In a battle the bottom screen is never drawn: its portraits and the
    // battle display (hearts, powers) are drawn onto the top screen instead,
    // see zakuro_gpu::raster::mirror_portrait. In menus the game is played on
    // the bottom screen, so it is shown over the top one. the game's
    // overrides say which it is, in the host mailbox, as the desktop does.
    let said = std::mem::take(&mut unx.system.memory.host_mailbox[3]);
    unx.out_of_play = if said != 0 { 0 } else { unx.out_of_play.saturating_add(1) };
    unx.since_portrait = if unx.system.gpu.take_big_portrait() { 0 } else { unx.since_portrait.saturating_add(1) };
    let in_play = unx.out_of_play <= IN_PLAY_QUIET || unx.since_portrait <= PORTRAIT_QUIET;
    unx.system.gpu.set_in_play(in_play);
    if said != 0 && unx.frame % 120 == 0 {
        log::info!("unx: overrides say {said} (frame {})", unx.frame);
    }
    if in_play != unx.in_play {
        log::info!("unx: {} at frame {} (said {said}, quiet for {})", if in_play { "in play" } else { "out of play" }, unx.frame, unx.out_of_play);
        unx.in_play = in_play;
        unx.presenter.set_layout(if in_play { ScreenLayout::TopOnly } else { ScreenLayout::LargeTop });
    }

    let started = Instant::now();
    let outcome = unx.system.run_frame();
    let took = started.elapsed();
    unx.frame += 1;
    let audio = unx.system.take_audio();
    if !audio.is_empty() {
        nx_audio_play(audio.as_ptr() as *const i16, audio.len() as u32);
    }
    let gpu = zakuro_gpu::stats::take();
    for (sum, ms) in unx.spent.iter_mut().zip([
        gpu.recompiled, gpu.services, gpu.drawing, gpu.shading, gpu.textures, gpu.waiting, gpu.sync, gpu.frame_end, gpu.audio,
    ]) {
        *sum += ms;
    }
    unx.draws += gpu.draws;
    unx.syncs += gpu.syncs;
    for (sum, n) in unx.write_backs.iter_mut().zip(gpu.write_backs) {
        *sum += n;
    }
    unx.write_back += gpu.write_back;
    let presenting = Instant::now();
    // debug: with sdmc:/switch/uprisingnx/dump_top present, the top screen's own
    // picture (before the presenter crops or scales it) goes to top.ppm and
    // bottom.ppm every 2 s, to tell what the game draws from what is shown
    if unx.frame % 120 == 0 && std::path::Path::new("sdmc:/switch/uprisingnx/dump_top").exists() {
        for (screen, name) in [(Screen::Top, "top"), (Screen::Bottom, "bottom")] {
            if let Some((px, w, h)) = unx.system.upright_screen(screen, false) {
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                for p in px.chunks_exact(4) {
                    out.extend_from_slice(&p[..3]);
                }
                let _ = std::fs::write(format!("sdmc:/switch/uprisingnx/{name}.ppm"), out);
            }
        }
    }

    let top = shown(&mut unx.system, Screen::Top);
    let blank = std::sync::Arc::new(Vec::new());
    let ((top_px, tw, th), top_gpu) = top;
    // the bottom screen is only drawn out of play
    let ((bottom_px, bw, bh), bottom_gpu) = if unx.in_play {
        (((blank.clone()), 0, 0), None)
    } else {
        shown(&mut unx.system, Screen::Bottom)
    };
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

    unx.window_presenting += presenting.elapsed();

    // a line every 2 s: emulated frames per second, how long emulating took
    // on average and at worst, and the interpreter's share
    unx.window_frames += 1;
    unx.window_emulating += took;
    unx.worst = unx.worst.max(took);
    let window = unx.window_start.elapsed();
    if window >= Duration::from_secs(2) {
        let frames = unx.window_frames as f64;
        let fallbacks = unx.system.recompiled.as_ref().map_or(0, |library| library.fallbacks());
        println!(
            "unx: frame {} {:.1} fps, emulating {:.1} ms avg {:.1} ms worst, {} interpreted, {} recompiled, {} fallbacks",
            unx.frame,
            frames / window.as_secs_f64(),
            unx.window_emulating.as_secs_f64() * 1000.0 / frames,
            unx.worst.as_secs_f64() * 1000.0,
            unx.system.interpreted_instructions,
            unx.system.recompiled_instructions,
            fallbacks,
        );
        let [recompiled, services, drawing, shading, textures, waiting, sync, frame_end, audio] = unx.spent.map(|ms| ms / frames);
        println!(
            "  per frame ms: recompiled {recompiled:.1} services {services:.1} | gpu: draw {drawing:.1} shade {shading:.1} tex {textures:.1} wait {waiting:.1} sync {sync:.1} ({:.1} syncs) | frame end {frame_end:.1} audio {audio:.1} | present {:.1} | {:.0} draws",
            unx.syncs as f64 / frames,
            unx.window_presenting.as_secs_f64() * 1000.0 / frames,
            unx.draws as f64 / frames,
        );
        let backs: Vec<String> = zakuro_gpu::stats::WRITE_BACK_NAMES
            .iter()
            .zip(unx.write_backs)
            .filter(|(_, n)| *n > 0)
            .map(|(name, n)| format!("{name} {:.1}", n as f64 / frames))
            .collect();
        if !backs.is_empty() {
            println!("  write-backs per frame: {} ({:.1} ms)", backs.join(", "), unx.write_back / frames);
        }
        if unx.frame % 600 < frames as u64 {
            if let Some(pcs) = &mut unx.system.interpreted_pcs {
                let mut top: Vec<(u32, u64)> = pcs.drain().collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                let total: u64 = top.iter().map(|(_, n)| n).sum();
                let list: Vec<String> = top.iter().take(12).map(|(pc, n)| format!("{pc:08X}:{n}")).collect();
                println!("  interpreted in the last 10 s: {total} at {} PCs; top: {}", top.len(), list.join(" "));
            }
        }
        unx.spent = [0.0; 9];
        unx.draws = 0;
        unx.syncs = 0;
        unx.write_backs = [0; 6];
        unx.write_back = 0.0;
        unx.window_presenting = Duration::ZERO;
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
    nx_audio_exit();
    if !unx.is_null() {
        drop(Box::from_raw(unx));
    }
}
