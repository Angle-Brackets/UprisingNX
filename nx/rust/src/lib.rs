//! UprisingNX's Switch frontend over Zakuro. The libnx C shell (`../c`) owns
//! the process, the window, input and nxlink; this side loads the game, runs
//! frames and presents them through NVK.

mod aim;

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
            report_line(format!("[{}] {}: {}", record.level(), record.target(), record.args()));
        }
    }
    fn flush(&self) {}
}

static LOGGER: Logger = Logger;

/// lines to print, and the frame rate ones to keep in fps.log, handed to a
/// thread of their own: printing goes over the network and the log to the
/// SD card, 30 ms a report, which stalled the frame it came in.
fn report_line(line: String) {
    static LINES: std::sync::OnceLock<Option<std::sync::mpsc::Sender<String>>> = std::sync::OnceLock::new();
    let sender = LINES.get_or_init(|| {
        let (send, receive) = std::sync::mpsc::channel::<String>();
        let spawned = std::thread::Builder::new().name("report".to_owned()).spawn(move || {
            use std::io::Write;
            let quiet = std::env::var_os("UNX_QUIET").is_some();
            let mut fps_log: Option<std::fs::File> = None;
            let mut fresh = true;
            for line in receive {
                // the reports and their indented lines, slow frames among
                // them, so a session played away from nxlink says where its
                // time went
                if line.starts_with("frame ") || line.starts_with("  ") {
                    // opened at the first line, the SD card is mounted by then
                    if fps_log.is_none() {
                        if fresh {
                            let _ = std::fs::remove_file("sdmc:/switch/uprisingnx/fps.log");
                        }
                        fresh = false;
                        fps_log = match std::fs::OpenOptions::new().create(true).append(true).open("sdmc:/switch/uprisingnx/fps.log") {
                            Ok(file) => Some(file),
                            Err(error) => {
                                println!("[ERROR] fps.log: {error}");
                                None
                            }
                        };
                    }
                    if let Some(log) = fps_log.as_mut() {
                        let _ = writeln!(log, "{line}");
                        let _ = log.flush();
                    }
                    if quiet {
                        continue;
                    }
                }
                // quiet: only the log's own lines go out
                if quiet && !line.starts_with('[') {
                    continue;
                }
                println!("{line}");
            }
        });
        spawned.ok().map(|_| send)
    });
    match sender {
        Some(sender) => {
            let _ = sender.send(line);
        }
        None => println!("{line}"),
    }
}

/// println, through the report thread.
macro_rules! report {
    ($($arg:tt)*) => { report_line(format!($($arg)*)) };
}

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
    pub gyro_x: f32,
    pub gyro_y: f32,
    pub gyro_z: f32,
    pub touching: u32,
    pub touch_x: u16,
    pub touch_y: u16,
    /// the frontend's own buttons, HOST_BOTTOM and so on
    pub host: u32,
}

/// UnxInput::host: the right stick's click, which shows or hides the bottom
/// screen
const HOST_BOTTOM: u32 = 1 << 0;

pub struct Unx {
    system: System,
    presenter: VulkanPresenter,
    frame: u64,
    window_start: Instant,
    window_frames: u32,
    /// the pictures the player sees: TOP_SWAPS when last looked, when the
    /// last new one came, the wall-clock gaps between new ones this window,
    /// a running average of those gaps, and vblanks that drew anything
    seen_swaps: u64,
    /// the last refresh showed frame interpolation's picture in between
    last_between: bool,
    /// when unx_frame last returned, to see time spent outside it
    last_exit: Option<Instant>,
    last_picture: Option<Instant>,
    picture_gaps: Vec<f32>,
    gap_average: f64,
    window_drawn: u32,
    window_emulating: Duration,
    window_presenting: Duration,
    worst: Duration,
    /// zakuro_gpu::stats summed over the window, in ms
    spent: [f64; 9],
    draws: u64,
    syncs: u64,
    write_backs: [u64; 6],
    write_back: f64,
    /// ms in the host GPU's side of the draws, see zakuro_gpu::stats::HARDWARE
    hardware: f64,
    /// ms the GPU's thread waited for vertices a helper shaded
    shade_wait: f64,
    /// frames since the game's overrides last said it was in play (a battle)
    out_of_play: u32,
    /// the bottom screen is shown: see Bottom
    bottom: Bottom,
    /// a benchmark under way, see Bench
    bench: Option<Bench>,
    in_play: bool,
    aim: aim::Aim,
    gyro_test: ([f32; 3], u32),
}

/// when the bottom screen is shown. Nothing guesses it from what the game
/// draws or how long it has been quiet: a guess put the bottom screen up in
/// the middle of battles (where the game's overrides go quiet, on foot, or in
/// Chapter 14's flight), costing the frame its drawing and the player the view.
/// It is shown
/// - from boot until the game first says it is in play (the title, the save
///   files and the menus before a first battle, which are played by touch),
/// - when the player clicks the right stick, which shows or hides it anywhere
///   (the pause menu, the menus after a battle, the arsenal, fusing weapons).
#[derive(Default)]
struct Bottom {
    shown: bool,
    /// the game has said it is in play once
    played: bool,
    /// last frame's host buttons, for presses
    host: u32,
}

impl Bottom {
    fn new() -> Bottom {
        Bottom { shown: true, ..Bottom::default() }
    }

    /// this frame's host input and what the overrides said; whether the bottom
    /// screen is shown now.
    fn update(&mut self, host: u32, said: u32) -> bool {
        let host_pressed = host & !self.host;
        self.host = host;
        if said != 0 && !self.played {
            self.played = true;
            self.shown = false;
        }
        if host_pressed & HOST_BOTTOM != 0 {
            self.shown = !self.shown;
        }
        self.shown
    }
}

/// a benchmark, from cmd.txt `bench <seconds> <slot>[:fire] ...`: each save
/// state in turn is loaded, left BENCH_SETTLE to settle, then measured for the
/// seconds given, holding fire (the 3DS's L) for a slot named with `:fire`.
/// One line a slot goes to sdmc:/switch/uprisingnx/bench.txt: new pictures a
/// second and their gaps, and where a picture's time went. The same states the
/// same way each time, so that a change is measured against the last.
struct Bench {
    seconds: f64,
    slots: std::collections::VecDeque<String>,
    slot: String,
    fire: bool,
    loaded: Instant,
    gaps: Vec<f32>,
    /// recompiled, GPU thread draw, of which shading, host GPU side, waiting
    /// for the GPU thread, summed over the measured frames
    spent: [f64; 5],
    /// draws that fell back, see zakuro_gpu::stats::FALLBACKS
    fallbacks: [u64; 4],
    lines: Vec<String>,
}

const BENCH_SETTLE: f64 = 5.0;
const BENCH_PATH: &str = "sdmc:/switch/uprisingnx/bench.txt";

impl Bench {
    fn measuring(&self) -> bool {
        self.loaded.elapsed().as_secs_f64() >= BENCH_SETTLE
    }

    fn done(&self) -> bool {
        self.loaded.elapsed().as_secs_f64() >= BENCH_SETTLE + self.seconds
    }

    fn line(&self) -> String {
        let mut gaps = self.gaps.clone();
        gaps.sort_by(f32::total_cmp);
        let at = |q: f64| gaps.get(((gaps.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
        let total: f32 = gaps.iter().sum();
        let pictures = gaps.len().max(1) as f64;
        let [recompiled, draw, shade, host, wait] = self.spent.map(|ms| ms / pictures);
        format!(
            "{:<14} {:5.1} pictures/s | gaps median {:5.1} 95th {:5.1} 99th {:5.1} worst {:6.1} ms, {:3} over 50 ms | per picture: recompiled {:4.1} draw {:4.1} shade {:4.1} host GPU {:4.1} wait {:4.1} ms",
            format!("{}{}", self.slot, if self.fire { ":fire" } else { "" }),
            if total > 0.0 { gaps.len() as f64 * 1000.0 / total as f64 } else { 0.0 },
            at(0.5), at(0.95), at(0.99), at(1.0),
            gaps.iter().filter(|g| **g > 50.0).count(),
            recompiled, draw, shade, host, wait,
        ) + &format!(
            " | fallback draws per picture: {}",
            zakuro_gpu::stats::FALLBACK_NAMES.iter().zip(self.fallbacks).map(|(name, n)| format!("{name} {:.1}", n as f64 / pictures)).collect::<Vec<_>>().join(", ")
        )
    }
}

/// loads a save state from the SD card, see the cmd.txt commands.
fn load_slot(system: &mut System, slot: &str) -> Result<String, String> {
    let file = format!("sdmc:/switch/uprisingnx/{slot}.zst");
    std::fs::read(&file).map_err(|e| e.to_string()).and_then(|bytes| system.load_state(&bytes).map(|_| "loaded".to_string()))
}

/// starts the next slot of a benchmark, or ends it, writing what it has.
fn bench_next(unx: &mut Unx) {
    let Some(bench) = unx.bench.as_mut() else { return };
    let Some(next) = bench.slots.pop_front() else {
        let mut text = bench.lines.join("\n");
        text.push_str("\ndone\n");
        let _ = std::fs::write(BENCH_PATH, text);
        log::info!("unx: bench done");
        unx.bench = None;
        return;
    };
    let (slot, fire) = match next.strip_suffix(":fire") {
        Some(slot) => (slot.to_owned(), true),
        None => (next, false),
    };
    if let Err(error) = load_slot(&mut unx.system, &slot) {
        bench.lines.push(format!("{slot}: failed to load, {error}"));
    }
    bench.slot = slot;
    bench.fire = fire;
    bench.loaded = Instant::now();
    bench.gaps.clear();
    bench.spent = [0.0; 5];
    bench.fallbacks = [0; 4];
}

extern "C" {
    fn nx_audio_init();
    fn nx_audio_play(samples: *const i16, count: u32);
    fn nx_audio_exit();
    fn unx_thread_role(role: *const c_char, index: i32);
}

fn place_thread(role: &str, index: usize) {
    if let Ok(role) = std::ffi::CString::new(role) {
        unsafe { unx_thread_role(role.as_ptr(), index as i32) };
    }
}

fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the C shell passes NUL-terminated strings
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// hands Zakuro a TrueType font (the Switch's own system font, from libnx's
/// pl:u) for the stand-in of the 3DS shared font, see zakuro-core
/// services/shared_font.rs. call before unx_create; the data is copied.
///
/// # Safety
/// `data` points to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn unx_set_font(data: *const u8, len: usize) {
    if !data.is_null() && len > 0 {
        zakuro_core::services::shared_font::set_frontend_font(std::slice::from_raw_parts(data, len));
    }
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
    let _ = zakuro_gpu::THREAD_ROLE.set(place_thread);
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
        // a picture in between each two of the game's, which draws every
        // other refresh: ZAKURO_INTERPOLATE=1
        interpolate: std::env::var("ZAKURO_INTERPOLATE").is_ok_and(|v| v == "1"),
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
            // recording where the interpreter ran is for finding code to recompile, and
    // writes a file to the SD card every minute: only with ZAKURO_HINTS=1
    if std::env::var_os("ZAKURO_HINTS").is_some() && system.recompiled.is_some() && system.hints.is_none() {
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
        seen_swaps: 0,
        last_between: false,
        last_exit: None,
        last_picture: None,
        picture_gaps: Vec::with_capacity(256),
        gap_average: 0.0,
        window_drawn: 0,
        window_emulating: Duration::ZERO,
        window_presenting: Duration::ZERO,
        worst: Duration::ZERO,
        spent: [0.0; 9],
        draws: 0,
        syncs: 0,
        write_backs: [0; 6],
        write_back: 0.0,
        hardware: 0.0,
        shade_wait: 0.0,
        out_of_play: u32::MAX,
        bottom: Bottom::new(),
        bench: None,
        in_play: false,
        aim: aim::Aim::from_env(),
        gyro_test: ([0.0; 3], 0),
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

/// how long the last 2 s report took, it runs between two frames.
static REPORT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// runs one emulated frame and presents it. 0 to go on, 1 when the game
/// exited or faulted.
///
/// # Safety
/// `unx` came from unx_create; `input` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn unx_frame(unx: *mut Unx, input: *const UnxInput) -> u32 {
    let unx = &mut *unx;
    let entered = Instant::now();
    // what the game's overrides say it is doing this frame (see below)
    let said = std::mem::take(&mut unx.system.memory.host_mailbox[3]);
    unx.aim.said(said);
    let controls = aim::GuestControls::read(&mut unx.system.memory);
    unx.aim.set_controls(controls);
    if !unx.in_play {
        unx.aim.patch_controls_ui(&mut unx.system.memory.phys.fcram, unx.frame);
    }
    let mut look = (0.0, 0.0);
    if let Some(input) = input.as_ref() {
        let mut state = InputState {
            buttons: PadState::from_bits_truncate(input.buttons),
            circle_x: input.circle_x,
            circle_y: input.circle_y,
            c_stick_x: input.c_stick_x,
            c_stick_y: input.c_stick_y,
            touch: (input.touching != 0).then(|| touch_on_bottom(unx.in_play, input.touch_x, input.touch_y)).flatten(),
            tilt: [0.0, 0.0],
        };
        if unx.bench.as_ref().is_some_and(|bench| bench.fire) {
            state.buttons |= PadState::L;
        }
        let mut gyro = [input.gyro_x, input.gyro_y, input.gyro_z];
        if unx.gyro_test.1 > 0 {
            unx.gyro_test.1 -= 1;
            for (g, t) in gyro.iter_mut().zip(unx.gyro_test.0) {
                *g += t;
            }
        }
        look = unx.aim.apply(&mut state, gyro, input.host);
        unx.system.set_input(state);
    }
    // on foot the game's mouse-look override reads the camera's turn from here
    unx.aim.feed(&mut unx.system.memory.host_mailbox, look);

    // In a battle the bottom screen is not drawn: its portraits and the
    // battle display (hearts, powers) are drawn onto the top screen instead,
    // see zakuro_gpu::raster::mirror_portrait. When it is shown is the
    // player's to say, see Bottom.
    let big_portrait = unx.system.gpu.flags().big_portrait.swap(false, std::sync::atomic::Ordering::Relaxed);
    unx.out_of_play = if said != 0 || big_portrait { 0 } else { unx.out_of_play.saturating_add(1) };
    let in_battle = unx.out_of_play <= 90;
    let host = input.as_ref().map_or(0, |input| input.host);
    let in_play = !unx.bottom.update(host, said);
    let flags = unx.system.gpu.flags();
    flags.in_play.store(in_play, std::sync::atomic::Ordering::Relaxed);
    flags.in_battle.store(in_battle, std::sync::atomic::Ordering::Relaxed);
    if said != 0 && unx.frame % 120 == 0 {
        log::info!("unx: overrides say {said} (frame {})", unx.frame);
    }
    if in_play != unx.in_play {
        log::info!("unx: bottom screen {} at frame {} (said {said}, quiet for {})", if in_play { "hidden" } else { "shown" }, unx.frame, unx.out_of_play);
        unx.in_play = in_play;
        unx.presenter.set_layout(if in_play { ScreenLayout::TopOnly } else { ScreenLayout::LargeTop });
    }

    // development: sdmc:/switch/uprisingnx/cmd.txt holding "save" or "load" (optionally
    // followed by a slot name) saves or restores the whole console, so that a test
    // can start from the same moment every time. the file is removed when read, and
    // the result is written to cmd.result
    if unx.frame % 15 == 0 {
        let cmd_path = "sdmc:/switch/uprisingnx/cmd.txt";
        if let Ok(text) = std::fs::read_to_string(cmd_path) {
            let _ = std::fs::remove_file(cmd_path);
            let mut words = text.split_whitespace();
            let (verb, slot) = (words.next().unwrap_or(""), words.next().unwrap_or("slot0"));
            let file = format!("sdmc:/switch/uprisingnx/{slot}.zst");
            let started = Instant::now();
            let result = match verb {
                "save" => unx.system.save_state().and_then(|bytes| {
                    std::fs::write(&file, &bytes).map(|_| format!("saved {} bytes", bytes.len())).map_err(|e| e.to_string())
                }),
                "load" => load_slot(&mut unx.system, slot),
                "gyro" => {
                    let gx: f32 = slot.parse().unwrap_or(0.0);
                    let gy: f32 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0.0);
                    let frames: u32 = words.next().and_then(|w| w.parse().ok()).unwrap_or(30);
                    unx.gyro_test = ([gx, gy, 0.0], frames);
                    Ok(format!("gyro ({gx}, {gy}) for {frames} frames"))
                }
                "bench" => {
                    let mut words = text.split_whitespace().skip(1);
                    let seconds = words.next().and_then(|w| w.parse::<f64>().ok()).unwrap_or(40.0);
                    let slots: std::collections::VecDeque<String> = words.map(str::to_owned).collect();
                    let count = slots.len();
                    unx.bench = Some(Bench {
                        seconds,
                        slots,
                        slot: String::new(),
                        fire: false,
                        loaded: Instant::now(),
                        gaps: Vec::new(),
                        spent: [0.0; 5],
                        fallbacks: [0; 4],
                        lines: vec![format!("bench: {seconds} s a slot after {BENCH_SETTLE} s to settle, env {:?}", std::fs::read_to_string("sdmc:/switch/uprisingnx/env.txt").unwrap_or_default().lines().filter(|l| !l.starts_with('#')).collect::<Vec<_>>().join(" "))],
                    });
                    let _ = std::fs::remove_file(BENCH_PATH);
                    bench_next(unx);
                    Ok(format!("started, {count} slots"))
                }
                other => Err(format!("unknown command {other:?}")),
            };
            let message = match result {
                Ok(done) => format!("{verb} {slot}: {done} in {:.0} ms", started.elapsed().as_secs_f64() * 1000.0),
                Err(error) => format!("{verb} {slot}: failed, {error}"),
            };
            log::info!("unx: {message}");
            let _ = std::fs::write("sdmc:/switch/uprisingnx/cmd.result", message);
        }
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
    // a slow frame says where it went, to find what the spikes are
    if took.as_secs_f64() * 1000.0 > 45.0 && unx.frame > 600 {
        report!(
            "  slow frame {}: {:.0} ms | recompiled {:.1} draw {:.1} shade {:.1} tex {:.1} hw {:.1} wait {:.1} sync {:.1} writeback {:.1} | shader jit {:.0} ms for {} programs | in play {} | fallback draws: cpu shaded {}, translating {}, generic {}, unspecialized {}",
            unx.frame,
            took.as_secs_f64() * 1000.0,
            gpu.recompiled, gpu.drawing, gpu.shading, gpu.textures, gpu.hardware, gpu.waiting, gpu.sync, gpu.write_back,
            gpu.jit, gpu.jit_programs, unx.in_play,
            gpu.fallbacks[0], gpu.fallbacks[1], gpu.fallbacks[2], gpu.fallbacks[3],
        );
    }
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
    unx.hardware += gpu.hardware;
    unx.shade_wait += gpu.shade_wait;
    let presenting = Instant::now();
    // debug: with UNX_DUMP_TOP set, the top screen's own
    // picture (before the presenter crops or scales it) goes to top.ppm and
    // bottom.ppm every 2 s, to tell what the game draws from what is shown
    // UNX_DUMP_TOP=1 only: reading both screens back and writing them out
    // stalls the frame ~90 ms every 2 s (it was a file flag, left on the SD
    // card, which made every session hitch)
    static DUMP_TOP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if unx.frame % 120 == 0 && *DUMP_TOP.get_or_init(|| std::env::var_os("UNX_DUMP_TOP").is_some()) {
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

    let presented = presenting.elapsed();
    unx.window_presenting += presented;

    // a new picture is a swap of the top screen; the gap since the last one
    // is what the player feels, whatever the vblank loop runs at
    let swaps = zakuro_core::TOP_SWAPS.load(std::sync::atomic::Ordering::Relaxed);
    if gpu.draws > 0 {
        unx.window_drawn += 1;
    }
    // with interpolation, the refresh after one showing a picture in
    // between shows the game's own, a new one with no swap of its own
    let after_between = std::mem::replace(&mut unx.last_between, unx.system.showing_between());
    if swaps != unx.seen_swaps || after_between {
        unx.seen_swaps = swaps;
        let now = Instant::now();
        if let Some(last) = unx.last_picture {
            let gap = (now - last).as_secs_f64() * 1000.0;
            unx.picture_gaps.push(gap as f32);
            if let Some(bench) = unx.bench.as_mut().filter(|bench| bench.measuring()) {
                bench.gaps.push(gap as f32);
            }
            // about the last half second of pictures
            unx.gap_average = if unx.gap_average == 0.0 { gap } else { unx.gap_average * 0.94 + gap * 0.06 };
        }
        unx.last_picture = Some(now);
    }

    if let Some(bench) = unx.bench.as_mut() {
        if bench.measuring() {
            for (sum, ms) in bench.spent.iter_mut().zip([gpu.recompiled, gpu.drawing, gpu.shading, gpu.hardware, gpu.waiting]) {
                *sum += ms;
            }
            for (sum, n) in bench.fallbacks.iter_mut().zip(gpu.fallbacks) {
                *sum += n;
            }
        }
        if bench.done() {
            let line = bench.line();
            log::info!("unx: bench {line}");
            bench.lines.push(line);
            let _ = std::fs::write(BENCH_PATH, bench.lines.join("\n") + "\n");
            bench_next(unx);
        }
    }

    // a line every 2 s: emulated frames per second, how long emulating took
    // on average and at worst, and the interpreter's share
    unx.window_frames += 1;
    unx.window_emulating += took;
    unx.worst = unx.worst.max(took);
    let window = unx.window_start.elapsed();
    if window >= Duration::from_secs(2) {
        let reporting = Instant::now();
        let frames = unx.window_frames as f64;
        let fallbacks = unx.system.recompiled.as_ref().map_or(0, |library| library.fallbacks());
        report!(
            "unx: frame {} {:.1} fps, emulating {:.1} ms avg {:.1} ms worst, {} interpreted, {} recompiled, {} fallbacks",
            unx.frame,
            frames / window.as_secs_f64(),
            unx.window_emulating.as_secs_f64() * 1000.0 / frames,
            unx.worst.as_secs_f64() * 1000.0,
            unx.system.interpreted_instructions,
            unx.system.recompiled_instructions,
            fallbacks,
        );
        {
            let mut gaps = std::mem::take(&mut unx.picture_gaps);
            gaps.sort_by(|a, b| a.total_cmp(b));
            let total: f64 = gaps.iter().map(|&g| g as f64).sum();
            let at = |q: f64| gaps.get(((gaps.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
            let line = format!(
                "shown: {:.1} new pictures/s (avg gap {:.1} ms = {:.1} fps, running avg {:.1} ms = {:.1} fps, median {:.1}, 95th {:.1}, worst {:.1} ms), {} of {} vblanks drew, in play {}, the last report took {:.1} ms",
                gaps.len() as f64 / window.as_secs_f64(),
                if gaps.is_empty() { 0.0 } else { total / gaps.len() as f64 },
                if total > 0.0 { gaps.len() as f64 * 1000.0 / total } else { 0.0 },
                unx.gap_average,
                if unx.gap_average > 0.0 { 1000.0 / unx.gap_average } else { 0.0 },
                at(0.5),
                at(0.95),
                gaps.last().copied().unwrap_or(0.0),
                unx.window_drawn,
                unx.window_frames,
                unx.in_play,
                REPORT_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
            );
            report!("{line}");
            report_line(format!("frame {} {line}", unx.frame));
            // whether the title waits a refresh of its own accord: emulated
            // cycles it was busy for per picture, against the 4.48M of one
            // refresh, and how late it heard its command lists were drawn
            let take = |i: usize| zakuro_core::PACING[i].swap(0, std::sync::atomic::Ordering::Relaxed);
            let (idle, lists, latency, worst, late) = (take(0), take(1), take(2), take(3), take(4));
            let all = frames * zakuro_core::CYCLES_PER_FRAME as f64;
            let pictures = gaps.len().max(1) as f64;
            report!(
                "  pacing: guest busy {:.0}% of emulated time, {:.2}M cycles per picture (a refresh is 4.48M), {} P3D lists, {:.0}k cycles to done avg, {:.0}k worst, {} heard after a refresh",
                100.0 * (1.0 - idle as f64 / all),
                (all - idle as f64) / pictures / 1e6,
                lists,
                if lists > 0 { latency as f64 / lists as f64 / 1000.0 } else { 0.0 },
                worst as f64 / 1000.0,
                late,
            );
            gaps.clear();
            unx.picture_gaps = gaps;
            unx.window_drawn = 0;
        }
        let [recompiled, services, drawing, shading, textures, waiting, sync, frame_end, audio] = unx.spent.map(|ms| ms / frames);
        report!(
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
        let waits: Vec<String> = zakuro_gpu::stats::BARRIER_NAMES
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                let (ns, n) = (zakuro_gpu::stats::BARRIER_NS[i].swap(0, std::sync::atomic::Ordering::Relaxed), zakuro_gpu::stats::BARRIER_COUNT[i].swap(0, std::sync::atomic::Ordering::Relaxed));
                (n > 0).then(|| format!("{name} {:.1} ms in {:.1}", ns as f64 / 1e6 / frames, n as f64 / frames))
            })
            .collect();
        report!("  gpu thread per frame: draw {:.1} ms of which shading {:.1}, the host GPU's side {:.1}, textures {:.1}, waiting for shaded vertices {:.1}", unx.spent[2] / frames, unx.spent[3] / frames, unx.hardware / frames, unx.spent[4] / frames, unx.shade_wait / frames);
        let (queue_wait, queue_held) = (
            zakuro_gpu::stats::QUEUE_WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
            zakuro_gpu::stats::QUEUE_HELD_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
        );
        report!("  vulkan queue per frame: renderer waited {:.2} ms, presenter held {:.2} ms", queue_wait as f64 / 1e6 / frames, queue_held as f64 / 1e6 / frames);
        if unx.window_emulating.as_secs_f64() * 1000.0 / frames < 2.0 && unx.frame % 600 < frames as u64 {
            report!("  the game is quiet, its threads:");
            for line in unx.system.debug_threads() {
                report!("    {line}");
            }
        }
        let (pending, unraised) = unx.system.gpu.state();
        if pending != 0 || unraised != 0 {
            report!("  GPU thread: {pending} jobs pending, {unraised} interrupts not raised");
        }
        if !waits.is_empty() {
            report!("  waits for the GPU thread per frame: {}", waits.join(", "));
        }
        if !backs.is_empty() {
            report!("  write-backs per frame: {} ({:.1} ms)", backs.join(", "), unx.write_back / frames);
        }
        if unx.frame % 600 < frames as u64 {
            if let Some(pcs) = &mut unx.system.interpreted_pcs {
                let mut top: Vec<(u32, u64)> = pcs.drain().collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                let total: u64 = top.iter().map(|(_, n)| n).sum();
                let list: Vec<String> = top.iter().take(12).map(|(pc, n)| format!("{pc:08X}:{n}")).collect();
                report!("  interpreted in the last 10 s: {total} at {} PCs; top: {}", top.len(), list.join(" "));
            }
        }
        unx.spent = [0.0; 9];
        unx.draws = 0;
        unx.syncs = 0;
        unx.write_backs = [0; 6];
        unx.write_back = 0.0;
        unx.hardware = 0.0;
        unx.shade_wait = 0.0;
        unx.window_presenting = Duration::ZERO;
        unx.window_start = Instant::now();
        unx.window_frames = 0;
        unx.window_emulating = Duration::ZERO;
        unx.worst = Duration::ZERO;
        REPORT_US.store(reporting.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    // a frame that kept the screen waiting says where the time went, in or
    // out of emulating, presenting, or outside this call
    let exiting = Instant::now();
    if let Some(last) = unx.last_exit {
        let total = exiting - last;
        if total >= Duration::from_millis(50) && unx.frame > 300 {
            let ms = |d: Duration| d.as_secs_f64() * 1000.0;
            report!(
                "  hitch frame {}: {:.1} ms since the last | outside unx_frame {:.1}, before emulating {:.1}, emulating {:.1}, after it {:.1}, present {:.1}, after present {:.1} | {} draws | in play {}",
                unx.frame,
                ms(total),
                ms(entered - last),
                ms(started - entered),
                ms(took),
                ms(presenting - started - took),
                ms(presented),
                ms(exiting - presenting - presented),
                gpu.draws,
                unx.in_play,
            );
        }
    }
    unx.last_exit = Some(Instant::now());

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
