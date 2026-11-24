//! UprisingNX's Switch frontend over Zakuro. The libnx C shell (`../c`) owns
//! the process, the window, input and nxlink; this side loads the game, runs
//! frames and presents them through NVK.

mod aim;
mod bootstrap;
mod menu;
mod saves;
mod settings;
mod updater;

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

static CURRENT_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FRAME_HEARTBEAT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FRAME_PHASE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static WATCHDOG_STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const PHASE_IDLE: u8 = 0;
const PHASE_EMULATING: u8 = 1;
const PHASE_PRESENTING: u8 = 2;
const PHASE_EXITING: u8 = 3;

extern "C" {
    fn unx_write_crash(buf: *const u8, len: usize);
    fn unx_kernel_memory_mib(used: *mut u64, total: *mut u64);
}

/// Starts a lightweight lock-free stall watchdog thread: if the main loop stays
/// inside `emulating` or `presenting` for 5+ awake seconds without advancing a
/// frame, appends a diagnostic snapshot directly to `crash.txt` on the stack.
fn start_watchdog() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    WATCHDOG_STOP.store(false, std::sync::atomic::Ordering::Release);
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("watchdog".into())
            .stack_size(32 << 10)
            .spawn(|| {
                use std::fmt::Write as _;
                use std::sync::atomic::Ordering::Acquire;
                let mut last_beat = FRAME_HEARTBEAT.load(Acquire);
                let mut stalled_secs = 0u32;
                while !WATCHDOG_STOP.load(Acquire) {
                    std::thread::sleep(Duration::from_secs(1));
                    if WATCHDOG_STOP.load(Acquire) {
                        break;
                    }
                    let beat = FRAME_HEARTBEAT.load(Acquire);
                    let phase = FRAME_PHASE.load(Acquire);
                    if beat == last_beat && (phase == PHASE_EMULATING || phase == PHASE_PRESENTING) {
                        stalled_secs = stalled_secs.saturating_add(1);
                        if stalled_secs >= 5 && (stalled_secs - 5).is_multiple_of(15) {
                            let frame = CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
                            let phase_name = if phase == PHASE_EMULATING { "emulating" } else { "presenting" };
                            let (mut used_mib, mut total_mib) = (0u64, 0u64);
                            unsafe { unx_kernel_memory_mib(&mut used_mib, &mut total_mib) };
                            let mut buf = CrashBuf { bytes: [0; 2048], len: 0 };
                            let _ = writeln!(
                                buf,
                                "WATCHDOG STALL ({stalled_secs}s) {} frame {frame} phase '{phase_name}' (mem {used_mib}/{total_mib} MiB)",
                                menu::VERSION_LABEL,
                            );
                            unsafe { unx_write_crash(buf.bytes.as_ptr(), buf.len) };
                            if let Ok(text) = std::str::from_utf8(&buf.bytes[..buf.len]) {
                                safe_println(text.trim_end());
                            }
                        }
                    } else {
                        last_beat = beat;
                        stalled_secs = 0;
                    }
                }
            });
    });
}

struct CrashBuf {
    bytes: [u8; 2048],
    len: usize,
}

impl std::fmt::Write for CrashBuf {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let room = self.bytes.len().saturating_sub(self.len);
        let take = s.len().min(room);
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

/// a panic aborts the process (panic = "abort"), and its message went only to
/// stderr, which nobody reads away from nxlink: the abort then closes HID and
/// the game's thread faults there, all Atmosphère's report shows. so the
/// message, the thread and the return addresses (frame pointer chain, as
/// offsets into the module for addr2line) go to crash.txt first on the stack,
/// without touching the allocator.
fn install_panic_hook() {
    extern "C" {
        fn _start();
        fn unx_note_panic_logged();
    }
    let shown = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        use std::fmt::Write as _;
        let base = _start as *const () as usize;
        let frame = CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed);
        let (mut used_mib, mut total_mib) = (0u64, 0u64);
        unsafe { unx_kernel_memory_mib(&mut used_mib, &mut total_mib) };
        let thread = std::thread::current();
        let mut buf = CrashBuf { bytes: [0; 2048], len: 0 };
        let _ = writeln!(
            buf,
            "PANIC frame {frame} thread '{}' (mem {used_mib}/{total_mib} MiB): {info}",
            thread.name().unwrap_or("<unnamed>"),
        );
        let _ = writeln!(buf, "  module base {base:#x}, return addresses (uprising_nx+offset):");
        let mut fp: usize;
        unsafe { std::arch::asm!("mov {}, x29", out(reg) fp) };
        for _ in 0..48 {
            if fp == 0 || fp % 8 != 0 {
                break;
            }
            let (next, lr) = unsafe { (*(fp as *const usize), *((fp + 8) as *const usize)) };
            if lr == 0 {
                break;
            }
            if lr >= base {
                let _ = writeln!(buf, "    uprising_nx+{:#x}", lr - base);
            } else {
                let _ = writeln!(buf, "    {lr:#x}");
            }
            if next <= fp {
                break;
            }
            fp = next;
        }
        unsafe {
            unx_note_panic_logged();
            unx_write_crash(buf.bytes.as_ptr(), buf.len);
        }
        if let Ok(text) = std::str::from_utf8(&buf.bytes[..buf.len]) {
            safe_println(text);
        }
        shown(info);
    }));
}

/// Print to stdout without panicking if stdout fails (e.g. nxlink disconnects or
/// socket closes with EPIPE / os error 32).
fn safe_println(text: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{text}");
    let _ = out.flush();
}

struct ReportQueue {
    state: std::sync::Mutex<(std::collections::VecDeque<String>, bool)>,
    cv: std::sync::Condvar,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

static REPORT_QUEUE: std::sync::OnceLock<ReportQueue> = std::sync::OnceLock::new();

fn report_queue() -> &'static ReportQueue {
    REPORT_QUEUE.get_or_init(|| ReportQueue {
        state: std::sync::Mutex::new((std::collections::VecDeque::new(), false)),
        cv: std::sync::Condvar::new(),
        handle: std::sync::Mutex::new(None),
    })
}

/// lines to print, and the frame rate ones to keep in fps.log, handed to a
/// thread of their own: printing goes over the network and the log to the
/// SD card, 30 ms a report, which stalled the frame it came in.
fn report_line(line: String) {
    static REPORT_STARTED: std::sync::Once = std::sync::Once::new();
    let q = report_queue();
    REPORT_STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("report".to_owned()).spawn(move || {
            use std::io::Write;
            let quiet = std::env::var_os("UNX_QUIET").is_some();
            let mut fps_log: Option<std::fs::File> = None;
            let mut tried_open = false;
            loop {
                let line = {
                    let mut guard = match q.state.lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(),
                    };
                    while guard.0.is_empty() && !guard.1 {
                        guard = match q.cv.wait(guard) {
                            Ok(g) => g,
                            Err(e) => e.into_inner(),
                        };
                    }
                    match guard.0.pop_front() {
                        Some(line) => line,
                        None => break,
                    }
                };
                // the reports and their indented lines, slow frames among
                // them, so a session played away from nxlink says where its
                // time went
                if line.starts_with("frame ") || line.starts_with("  ") || line.starts_with("[ERROR]") || line.starts_with("[WARN]") {
                    // opened at the first line, the SD card is mounted by then
                    if fps_log.is_none() && !tried_open {
                        tried_open = true;
                        let _ = std::fs::remove_file("sdmc:/switch/uprisingnx/fps.log");
                        fps_log = match std::fs::OpenOptions::new().create(true).append(true).open("sdmc:/switch/uprisingnx/fps.log") {
                            Ok(file) => Some(file),
                            Err(error) => {
                                safe_println(&format!("[ERROR] fps.log: {error}"));
                                None
                            }
                        };
                    }
                    if let Some(log) = fps_log.as_mut() {
                        let _ = writeln!(log, "{line}");
                        let _ = log.flush();
                    }
                    if quiet && !line.starts_with('[') {
                        continue;
                    }
                }
                // quiet: only the log's own lines go out
                if quiet && !line.starts_with('[') {
                    continue;
                }
                safe_println(&line);
            }
        });
        if let Ok(handle) = spawned {
            if let Ok(mut slot) = q.handle.lock() {
                *slot = Some(handle);
            }
        }
    });
    let mut guard = match q.state.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if guard.1 {
        drop(guard);
        safe_println(&line);
        return;
    }
    if guard.0.is_empty() && q.handle.lock().map_or(true, |h| h.is_none()) {
        drop(guard);
        safe_println(&line);
        return;
    }
    guard.0.push_back(line);
    q.cv.notify_one();
}

/// flushes queued log lines and joins the background reporter thread before shutdown.
fn stop_reporter() {
    let Some(q) = REPORT_QUEUE.get() else { return };
    {
        let mut guard = match q.state.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        guard.1 = true;
        q.cv.notify_all();
    }
    if let Ok(mut slot) = q.handle.lock() {
        if let Some(handle) = slot.take() {
            let _ = handle.join();
        }
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
/// UnxInput::host: the console is in the dock (AppletOperationMode_Console)
const HOST_DOCKED: u32 = 1 << 2;
/// UnxInput::host: the right stick's click held with the D-pad: Up raises the
/// internal resolution, Down lowers it, Left gives it back to the automatic choice
const HOST_SCALE_UP: u32 = 1 << 3;
const HOST_SCALE_DOWN: u32 = 1 << 4;
const HOST_SCALE_AUTO: u32 = 1 << 5;
const HOST_SMOOTH: u32 = 1 << 6;
/// UnxInput::host: the right stick's click held with Minus opens or closes
/// the settings menu (menu.rs)
const HOST_MENU: u32 = 1 << 7;

/// smooth mode's guard looks at this many refreshes at a time
const SMOOTH_CHECK: u32 = 30;
/// what the emulation waited for the GPU per refresh, on average over a
/// check, above which the GPU can't draw each frame twice and smooth mode
/// turns off. it waits 0.3-1.5 ms when it keeps up (Chapter 14 at 1x
/// handheld, 44-58 pictures/s) and 10+ ms when it doesn't (2x handheld,
/// 21 pictures/s against 27-30 without)
const SMOOTH_WAIT_MS: f64 = 4.0;
/// ... or most refreshes of a check take longer than this in real time: the
/// loop no longer keeps to the screen's 60 Hz, the presenter waiting on the
/// host GPU (at 2x handheld in a battle the game fell to ~21 frames/s with
/// no wait the first test sees). most, so a load's one long stall is not it
const SMOOTH_REFRESH_MS: f64 = 20.0;
/// how many consecutive slow checks (~0.5 s each) before smooth_guard steps
/// in, so a single heavy explosion or shader compile hitch does not trip it
const SMOOTH_STRIKES: u32 = 2;
/// how long smooth mode stays off before it tries again, doubling each
/// time it fails up to the most, back to the first once it kept up for
/// SMOOTH_SETTLED checks or the scene changes (into or out of a battle, a
/// load). kept short so heavy moments recover automatically without waiting
/// for a load screen
const SMOOTH_BACKOFF_FIRST: u64 = 3 * 60;
const SMOOTH_BACKOFF_MOST: u64 = 6 * 60;
const SMOOTH_SETTLED: u32 = 20;
/// refreshes ignored after toggling smooth mode or stepping battle resolution,
/// so surface rebuild / swapchain recovery frames don't immediately trip the guard
const SMOOTH_GRACE_FRAMES: u32 = 20;

/// frame interpolation, smooth mode: a picture in between each two of the
/// game's, which draws every other refresh. it doubles what the GPU draws,
/// so a guard turns it off while the GPU falls behind and tries again later
/// or when the resolution goes down.
struct Smooth {
    /// the player's choice: settings.ini's preset, right stick + Right, or
    /// ZAKURO_INTERPOLATE in env.txt
    wanted: bool,
    /// whether it runs now
    on: bool,
    /// the frame it tries again at, after the guard turned it off
    retry_at: u64,
    backoff: u64,
    /// the wait for the GPU summed over the current check
    wait: f64,
    elapsed: f64,
    frames: u32,
    /// when the last refresh was, and the refreshes of the check that took
    /// longer than SMOOTH_REFRESH_MS
    last: Option<Instant>,
    slow: u32,
    /// consecutive slow checks seen so far
    strikes: u32,
    /// checks in a row that kept up
    kept_up: u32,
    /// whether dropping battle resolution 1 step already failed to keep smooth mode on in this scene
    scale_drop_failed: bool,
    /// remaining refreshes to ignore before measuring again (after a resize or mode toggle)
    grace: u32,
}
/// the highest scale the controller asks for, 1600x960 a screen
const MAX_SCALE: u32 = 4;

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
    /// the presenter's current screen arrangement
    layout: ScreenLayout,
    /// whether the presenter and resolution are currently in docked mode
    docked: bool,
    /// the presenter swapchain dimensions matching the current docked/display state
    display_size: (u32, u32),
    /// the hardware's last reported docked state from appletGetOperationMode
    hw_docked: bool,
    /// a docked state forced by cmd.txt `dock 0|1`, cleared when hardware state changes
    dock_override: Option<bool>,
    /// a fixed resolution scale from ZAKURO_RESOLUTION in env.txt or cmd.txt
    /// `scale <n>`, over settings.ini's; the hotkeys clear it
    fixed_resolution: Option<u32>,
    /// sdmc:/switch/uprisingnx/settings.ini, see settings.rs
    settings: settings::Settings,
    /// ZAKURO_INTERPOLATE in env.txt decides smooth mode over settings.ini
    smooth_forced: Option<bool>,
    /// frames since the GPU last reported drawing the title screen
    title_quiet: u32,
    /// whether the game is currently showing the title screen (for version watermark)
    on_title: bool,
    /// whether internal resolution is currently boosted to 2x for the title screen
    title_boosted: bool,
    /// last seen sub-screen (`1..=3`) for `ScreenLayout::WeaponSelect`
    weapon_sub: u8,
    /// battle resolution steps dropped by smooth_guard during the current
    /// battle so smooth mode stays on (0 or 1, cleared when the scene changes)
    battle_scale_drop: u32,
    /// last frame's host buttons, for the resolution buttons' presses
    host_last: u32,
    smooth: Smooth,
    /// the in-game settings menu
    menu: menu::Menu,
    /// root data directory (`sdmc:/switch/uprisingnx/data`) for live saves & mods
    data_dir: PathBuf,
    /// background GitHub Releases auto-updater
    updater: updater::Updater,
    aim: aim::Aim,
    gyro_test: ([f32; 3], u32),
    /// Fiend's Cauldron live slider cursor X (0..320), floating badge opacity, and hold frames
    cauldron_cursor_x: f32,
    cauldron_opacity: f32,
    cauldron_hold: u8,
    /// background watcher delivering commands read from `sdmc:/switch/uprisingnx/cmd.txt`
    cmd_rx: std::sync::mpsc::Receiver<String>,
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
    /// while on Fuse Weapons, R-Stick click toggles between FuseSplitDeck and LargeTop
    fuse_override: bool,
    /// while watching a standalone How to Play video (`in_battle && !res_battle`),
    /// R-Stick click toggles LargeTop without altering the menu's `shown` state
    tutorial_override: bool,
    /// last frame's host buttons, for presses
    host: u32,
}

impl Bottom {
    fn new() -> Bottom {
        Bottom { shown: true, ..Bottom::default() }
    }

    /// this frame's host input and what the overrides said; whether the bottom
    /// screen is shown now.
    fn update(&mut self, host: u32, said: u32, on_fuse: bool, on_tutorial: bool, in_frontend_menu: bool) -> bool {
        let host_pressed = host & !self.host;
        self.host = host;
        if !on_fuse {
            self.fuse_override = false;
        }
        if !on_tutorial {
            self.tutorial_override = false;
        }
        if said != 0 && !self.played {
            self.played = true;
            self.shown = false;
        } else if (in_frontend_menu || on_fuse) && self.played {
            // Returning from a battle into KIU's frontend menus restores menu visibility
            // and non-battle resolution automatically.
            self.played = false;
            self.shown = true;
        }
        if host_pressed & HOST_BOTTOM != 0 {
            if on_fuse {
                self.fuse_override = !self.fuse_override;
            } else if on_tutorial {
                self.tutorial_override = !self.tutorial_override;
            } else {
                self.shown = !self.shown;
            }
        }
        if on_tutorial {
            self.tutorial_override
        } else {
            self.shown
        }
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
    unx.out_of_play = u32::MAX;
    unx.scene_changed("a benchmark slot");
}

extern "C" {
    fn nx_audio_init();
    fn nx_audio_play(samples: *const i16, count: u32);
    fn nx_audio_pause(paused: bool);
    fn nx_audio_exit();
    fn unx_thread_role(role: *const c_char, index: i32);
    fn unx_memory(out: *mut u64);
    fn unx_display_resolution(out_w: *mut u32, out_h: *mut u32, force: i32);
    fn appletGetOperationMode() -> u32;
    fn unx_is_applet_mode() -> i32;
}

fn query_window_size(docked: bool, hw_docked: bool, force: bool) -> (u32, u32) {
    if !docked {
        return (1280, 720);
    }
    if !hw_docked {
        return (1920, 1080);
    }
    let (mut w, mut h) = (0u32, 0u32);
    // SAFETY: valid out-pointers to stack u32s.
    unsafe { unx_display_resolution(&mut w, &mut h, force as i32) };
    if w == 0 || h == 0 {
        (1920, 1080)
    } else {
        (w.clamp(640, 1920), h.clamp(360, 1080))
    }
}

impl Unx {
    /// the scale settings.ini's preset for this mode wants now, in a battle
    /// or out of one (see settings::Mode::video)
    fn auto_scale(&self) -> u32 {
        let video = self.settings.mode(self.docked).video(self.docked);
        if self.title_boosted {
            video.menu_scale
        } else {
            video.battle_scale.saturating_sub(self.battle_scale_drop).max(1)
        }
    }

    /// whether smooth frame interpolation is allowed right now: enabled in
    /// settings, inside an active battle (`!title_boosted`), unpaused
    /// (`out_of_play <= 2`), and in `TopOnly` gameplay layout so frontend menus
    /// and menu transitions never display morphed in-between frames.
    fn smooth_allowed(&self) -> bool {
        self.smooth.wanted
            && !self.title_boosted
            && self.out_of_play <= 2
            && self.layout == ScreenLayout::TopOnly
    }

    /// puts the settings for this mode in place: resolution (unless fixed)
    /// and smooth mode (unless env.txt decides it)
    fn apply_settings(&mut self) {
        if self.fixed_resolution.is_none() {
            let target = self.auto_scale();
            self.set_scale(target);
        }
        let wanted = self.smooth_forced.unwrap_or(self.settings.mode(self.docked).video(self.docked).smooth);
        if wanted != self.smooth.wanted {
            self.smooth.wanted = wanted;
            self.smooth.backoff = SMOOTH_BACKOFF_FIRST;
            self.smooth.kept_up = 0;
            self.smooth.retry_at = 0;
            self.smooth.scale_drop_failed = false;
            if !wanted && self.battle_scale_drop != 0 {
                self.battle_scale_drop = 0;
                if self.fixed_resolution.is_none() {
                    let target = self.auto_scale();
                    self.set_scale(target);
                }
            }
            let on = self.smooth_allowed();
            self.set_smooth(on);
        }
    }

    /// the menu's overlay for this refresh (or the title-screen version watermark)
    fn menu_overlay(&mut self) -> Overlay {
        if !self.menu.has_overlay(self.on_title) {
            return Overlay::default();
        }
        let (width, height) = self.presenter.extent();
        let now = (self.system.gpu.scale(), self.smooth.on);
        let badge = self.updater.title_badge();
        self.menu.overlay(&self.settings, self.docked, now, width, height, self.on_title, badge)
    }

    /// writes the settings back and says what this mode runs at now
    fn settings_changed(&mut self, what: &str) {
        self.settings.save();
        let mode = self.settings.mode(self.docked);
        let video = mode.video(self.docked);
        log::info!(
            "unx: {what}: {} preset {:?}, battles {}x, menus {}x, smooth {} (now {}x, smooth {}, frame {})",
            if self.docked { "docked" } else { "handheld" },
            mode.preset,
            video.battle_scale,
            video.menu_scale,
            if video.smooth { "on" } else { "off" },
            self.system.gpu.scale(),
            if self.smooth.on { "on" } else { "off" },
            self.frame,
        );
    }

    /// sets the internal resolution; a lower one may leave the GPU room for
    /// smooth mode, which tries again
    fn set_scale(&mut self, target: u32) -> u32 {
        let before = self.system.gpu.scale();
        let scale = self.system.set_resolution(target);
        if scale < before {
            self.smooth.retry_at = self.frame;
        }
        scale
    }

    /// a new scene says nothing of what the last one cost: smooth mode, if
    /// wanted and off, tries again now, and the backoff starts over
    fn scene_changed(&mut self, why: &str) {
        self.smooth.scale_drop_failed = false;
        if self.battle_scale_drop != 0 {
            self.battle_scale_drop = 0;
            if self.fixed_resolution.is_none() {
                let target = self.auto_scale();
                let scale = self.set_scale(target);
                log::info!("unx: battle resolution restored -> {scale}x, {why} (frame {})", self.frame);
            }
        }
        self.system.reset_interpolation();
        self.smooth.strikes = 0;
        self.smooth.backoff = SMOOTH_BACKOFF_FIRST;
        if self.smooth.wanted && !self.smooth.on && self.smooth.retry_at > self.frame {
            self.smooth.retry_at = self.frame;
            log::info!("unx: smooth mode tries again, {why} (frame {})", self.frame);
        }
    }

    fn set_smooth(&mut self, on: bool) {
        self.smooth.on = self.system.set_interpolation(on);
        self.smooth.wait = 0.0;
        self.smooth.elapsed = 0.0;
        self.smooth.frames = 0;
        self.smooth.last = None;
        self.smooth.slow = 0;
        self.smooth.strikes = 0;
        self.smooth.grace = if self.smooth.on { SMOOTH_GRACE_FRAMES } else { 0 };
    }

    /// right stick + Right turns smooth mode on or off: this mode's preset
    /// becomes custom with smooth flipped
    fn controller_smooth(&mut self, pressed: u32) {
        if pressed & HOST_SMOOTH == 0 {
            return;
        }
        let docked = self.docked;
        let video = self.settings.mode_mut(docked).customize(docked);
        video.smooth = !self.smooth.wanted;
        self.smooth_forced = None;
        self.battle_scale_drop = 0;
        self.smooth.scale_drop_failed = false;
        self.apply_settings();
        self.settings_changed("smooth mode");
        self.notify_video();
    }

    /// a hotkey changed the video settings: says so over the game
    fn notify_video(&mut self) {
        let mode = *self.settings.mode(self.docked);
        let video = mode.video(self.docked);
        let preset = match mode.preset {
            settings::Preset::Performance => "Performance",
            settings::Preset::Quality => "Quality",
            settings::Preset::Custom => "Custom",
        };
        self.menu.notify(format!(
            "{preset}: battles {}x, menus {}x, smooth {}   ({}: settings)",
            video.battle_scale,
            video.menu_scale,
            if video.smooth { "on" } else { "off" },
            if self.settings.minus_opens_menu() { "Minus" } else { "RS + Minus" },
        ));
    }

    /// smooth mode's guard, each refresh with what it waited for the GPU
    fn smooth_guard(&mut self, waiting: f64) {
        if !self.smooth_allowed() {
            if self.smooth.on {
                self.set_smooth(false);
            }
            return;
        }
        // UNX_SMOOTH_GUARD=0 keeps smooth mode on whatever it costs, to measure it
        static UNGUARDED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *UNGUARDED.get_or_init(|| std::env::var("UNX_SMOOTH_GUARD").is_ok_and(|v| v.trim() == "0")) {
            if !self.smooth.on {
                self.set_smooth(true);
            }
            return;
        }
        if !self.smooth.on {
            if self.frame >= self.smooth.retry_at {
                self.set_smooth(true);
                log::info!("unx: smooth mode tries again (frame {})", self.frame);
            }
            return;
        }
        if self.smooth.grace > 0 {
            self.smooth.grace -= 1;
            self.smooth.last = None;
            return;
        }
        let now = Instant::now();
        let Some(last) = self.smooth.last.replace(now) else {
            return;
        };
        let dt = (now - last).as_secs_f64() * 1000.0;
        self.smooth.elapsed += dt;
        self.smooth.slow += (dt > SMOOTH_REFRESH_MS) as u32;
        self.smooth.wait += waiting;
        self.smooth.frames += 1;
        if self.smooth.frames < SMOOTH_CHECK {
            return;
        }
        let average = self.smooth.wait / self.smooth.frames as f64;
        let refresh_avg = std::mem::take(&mut self.smooth.elapsed) / self.smooth.frames as f64;
        let slow = std::mem::take(&mut self.smooth.slow);
        let frames = self.smooth.frames;
        self.smooth.wait = 0.0;
        self.smooth.frames = 0;
        // Odd/even interpolation splits a 33.33 ms pair into ~20.2 ms + ~13.1 ms at a
        // locked 60 Hz, so require the 30-refresh mean interval to exceed 18.5 ms (<54 Hz)
        // before treating individual >20 ms deltas as falling behind.
        if average <= SMOOTH_WAIT_MS && (slow * 2 <= frames || refresh_avg <= 18.5) {
            self.smooth.strikes = 0;
            self.smooth.kept_up += 1;
            if self.smooth.kept_up >= SMOOTH_SETTLED {
                self.smooth.backoff = SMOOTH_BACKOFF_FIRST;
                self.smooth.scale_drop_failed = false;
            }
            return;
        }
        self.smooth.kept_up = 0;
        self.smooth.strikes += 1;
        if self.smooth.strikes < SMOOTH_STRIKES {
            return;
        }
        self.smooth.strikes = 0;
        let allow_drop = self.settings.smooth_scale_drop();
        if allow_drop
            && !self.smooth.scale_drop_failed
            && !self.title_boosted
            && self.fixed_resolution.is_none()
            && self.battle_scale_drop == 0
        {
            let before = self.system.gpu.scale();
            if before > 1 {
                self.battle_scale_drop = 1;
                let target = self.auto_scale();
                let scale = self.set_scale(target);
                if scale < before {
                    self.smooth.wait = 0.0;
                    self.smooth.elapsed = 0.0;
                    self.smooth.frames = 0;
                    self.smooth.last = None;
                    self.smooth.slow = 0;
                    self.smooth.strikes = 0;
                    self.smooth.kept_up = 0;
                    self.smooth.grace = SMOOTH_GRACE_FRAMES;
                    log::info!(
                        "unx: smooth guard stepped battle resolution {before}x -> {scale}x to keep smooth mode on ({average:.1} ms waited a refresh, {refresh_avg:.1} ms/refresh, {slow} of {frames} refreshes slow, frame {})",
                        self.frame
                    );
                    return;
                }
            }
        }
        let dropped_failed = self.battle_scale_drop != 0;
        let backoff = if dropped_failed { SMOOTH_BACKOFF_MOST } else { self.smooth.backoff };
        self.smooth.retry_at = self.frame + backoff;
        self.smooth.backoff = (backoff * 2).min(SMOOTH_BACKOFF_MOST);
        self.smooth.kept_up = 0;
        self.set_smooth(false);
        if dropped_failed {
            self.battle_scale_drop = 0;
            self.smooth.scale_drop_failed = true;
            if self.fixed_resolution.is_none() {
                let target = self.auto_scale();
                let restored = self.set_scale(target);
                log::info!(
                    "unx: smooth mode off ({average:.1} ms waited a refresh, {refresh_avg:.1} ms/refresh, {slow} of {frames} refreshes slow), restored battle resolution -> {restored}x, again in {} s (frame {})",
                    backoff / 60,
                    self.frame
                );
                return;
            }
        }
        log::info!(
            "unx: smooth mode off, the GPU fell behind ({average:.1} ms waited a refresh, {refresh_avg:.1} ms/refresh, {slow} of {frames} refreshes slow), again in {} s (frame {})",
            backoff / 60,
            self.frame
        );
    }

    /// the controller's resolution buttons. up or down changes this mode's
    /// battle or menu scale, whichever is in use now, and makes its preset
    /// custom; the auto button steps to the next preset (performance,
    /// quality, custom). either gives the resolution back from a fixed one
    fn controller_scale(&mut self, pressed: u32) {
        let docked = self.docked;
        let what = if pressed & HOST_SCALE_AUTO != 0 {
            let mode = self.settings.mode_mut(docked);
            mode.preset = mode.preset.next();
            "preset"
        } else if pressed & (HOST_SCALE_UP | HOST_SCALE_DOWN) != 0 {
            let now = self.system.gpu.scale();
            let target = if pressed & HOST_SCALE_UP != 0 { (now + 1).min(MAX_SCALE) } else { now.saturating_sub(1).max(1) };
            let battle = !self.title_boosted;
            let video = self.settings.mode_mut(docked).customize(docked);
            *(if battle { &mut video.battle_scale } else { &mut video.menu_scale }) = target;
            if battle { "battle resolution" } else { "menu resolution" }
        } else {
            return;
        };
        self.fixed_resolution = None;
        self.battle_scale_drop = 0;
        self.smooth.scale_drop_failed = false;
        self.apply_settings();
        self.settings_changed(what);
        self.notify_video();
    }

    fn apply_docked(&mut self, docked: bool) {
        self.docked = docked;
        self.battle_scale_drop = 0;
        self.smooth.scale_drop_failed = false;
        self.smooth.grace = SMOOTH_GRACE_FRAMES;
        let (width, height) = query_window_size(docked, self.hw_docked, true);
        self.display_size = (width, height);
        self.presenter.resize(width, height);
        if let Some(fixed) = self.fixed_resolution {
            self.set_scale(fixed);
        }
        self.apply_settings();
        self.scene_changed(if docked { "docked" } else { "handheld" });
        let scale = self.system.gpu.scale();
        log::info!(
            "unx: {} -> presenter {}x{}, internal resolution {}x (frame {})",
            if docked { "docked" } else { "handheld" },
            width,
            height,
            scale,
            self.frame,
        );
    }
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
        let font = std::slice::from_raw_parts(data, len);
        zakuro_core::services::shared_font::set_frontend_font(font);
        let _ = menu::FONT.set(font.to_vec());
    }
}

pub enum UnxHandle {
    Running(Box<Unx>),
    Notice {
        presenter: VulkanPresenter,
        menu: menu::Menu,
        notice: bootstrap::SetupNotice,
        armed: bool,
    },
}

/// starts Zakuro on the ROM at `rom` (or auto-discovers it in
/// `sdmc:/switch/uprisingnx/rom/` when `rom` is null), presenting to `nwindow`
/// (libnx's nwindowGetDefault()); `data_dir` holds saves and the like. `recomp`
/// is optional statically linked recompiled code; when null, UprisingNX loads or
/// constructs its native Switch AOT cache automatically.
///
/// # Safety
/// `nwindow` is a live NWindow for the life of the returned handle; the
/// strings are null or NUL-terminated; `recomp` is null or points to
/// 3dsrecomp's tables, which live as long as the program.
#[no_mangle]
pub unsafe extern "C" fn unx_create(nwindow: *mut c_void, rom: *const c_char, data_dir: *const c_char, recomp: *const UnxRecomp) -> *mut UnxHandle {
    install_panic_hook();
    let _ = log::set_logger(&LOGGER);
    let _ = zakuro_gpu::THREAD_ROLE.set(place_thread);
    log::set_max_level(log::LevelFilter::Info);
    log::info!("unx: UprisingNX {}", menu::VERSION_LABEL);
    let Some(data_dir) = cstr(data_dir) else {
        log::error!("unx: no data directory");
        return std::ptr::null_mut();
    };
    let explicit_rom = cstr(rom);

    let docked = appletGetOperationMode() == 1;
    let window = query_window_size(docked, docked, true);
    let fixed_resolution = std::env::var("ZAKURO_RESOLUTION")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    let settings = settings::Settings::load();
    settings.export_controls();
    zakuro_gpu::set_power_label_mode(settings.power_label_mode());
    MINUS_MENU.store(settings.minus_opens_menu(), std::sync::atomic::Ordering::Relaxed);
    let video = settings.mode(docked).video(docked);
    // the game starts out of battle
    let resolution = fixed_resolution.unwrap_or(video.menu_scale);

    let started = Instant::now();
    let mut presenter = match VulkanPresenter::new_vi(nwindow, window) {
        Ok(presenter) => presenter,
        Err(error) => {
            log::error!("unx: Vulkan presenter failed, {error}");
            return std::ptr::null_mut();
        }
    };
    log::info!(
        "unx: presenter {} up in {:?} ({} {}x{}, internal {}x)",
        presenter.name(),
        started.elapsed(),
        if docked { "docked" } else { "handheld" },
        window.0,
        window.1,
        resolution,
    );
    // start in fullscreen (top screen only) by default
    presenter.set_layout(ScreenLayout::TopOnly);
    presenter.set_scaling(ScreenFilter::Smooth, false);

    let mut menu = menu::Menu::new();
    let make_notice = |presenter: VulkanPresenter, menu: menu::Menu, notice: bootstrap::SetupNotice| -> *mut UnxHandle {
        log::warn!("unx: setup notice '{}': {}", notice.heading, notice.paragraphs.join(" "));
        Box::into_raw(Box::new(UnxHandle::Notice { presenter, menu, notice, armed: false }))
    };

    if unx_is_applet_mode() != 0 {
        return make_notice(presenter, menu, bootstrap::SetupNotice::applet_mode());
    }

    let (rom_path, title) = match bootstrap::discover_rom(explicit_rom.as_deref()) {
        Ok(found) => found,
        Err(notice) => return make_notice(presenter, menu, notice),
    };
    let rom = rom_path.to_string_lossy().into_owned();

    let force_interpret = std::env::var("UNX_INTERPRET").is_ok_and(|v| v == "1");
    let linked = recomp.as_ref().map(|r| {
        log::info!("unx: recompiled code linked in, ABI {}, {} entries, {} modules", r.abi, r.entry_count, r.module_count);
        zakuro_core::recompiled::Linked::new(0x0004000000030100, r.abi, r.entries, r.entry_count, r.modules, r.module_count)
    });

    let (loaded_aot, preloaded_code) = if linked.is_none() && !force_interpret {
        match bootstrap::load_or_build_aot(&title, std::path::Path::new(&data_dir), &mut presenter, &mut menu) {
            Ok((aot, code)) => (Some(aot), code),
            Err(notice) => return make_notice(presenter, menu, notice),
        }
    } else {
        (None, None)
    };

    nx_audio_init();

    let data_dir_path = PathBuf::from(&data_dir);
    if let Some(import_notice) = saves::auto_import_on_boot(&data_dir_path) {
        menu.notify(import_notice);
    }

    let smooth_forced = std::env::var("ZAKURO_INTERPOLATE").ok().map(|v| v.trim() != "0");
    let smooth_wanted = smooth_forced.unwrap_or(video.smooth);
    let config = Config {
        data_dir: Some(data_dir_path.clone()),
        hardware_renderer: true,
        resolution,
        device: presenter.shared_device(),
        linked,
        // smooth mode runs during active battles (`smooth_allowed`) so menus
        // and menu transitions never show interpolated frames
        interpolate: false,
        find_recompiled: false,
        username: "Pit".into(),
        ..Config::default()
    };
    let started = Instant::now();
    let system = match loader::load_preloaded(title, preloaded_code, config, None) {
        Ok(mut system) => {
            if let Some(aot) = loaded_aot {
                system.set_aot(aot);
            }
            // what the recompiled code leaves to the interpreter, printed
            // every 10 s (only when diagnostics or hints are requested)
            if std::env::var_os("UNX_DIAG").is_some() || std::env::var_os("ZAKURO_HINTS").is_some() {
                system.interpreted_pcs = Some(Default::default());
            }
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
            return make_notice(
                presenter,
                menu,
                bootstrap::SetupNotice {
                    heading: "Failed to Load ROM".to_owned(),
                    paragraphs: vec![format!("Could not start {rom}: {error}")],
                },
            );
        }
    };
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<String>();
    let _ = std::thread::Builder::new()
        .name("cmd-watch".into())
        .stack_size(64 << 10)
        .spawn(move || {
            let cmd_path = "sdmc:/switch/uprisingnx/cmd.txt";
            loop {
                std::thread::sleep(Duration::from_millis(500));
                if let Ok(text) = std::fs::read_to_string(cmd_path) {
                    let _ = std::fs::remove_file(cmd_path);
                    if cmd_tx.send(text).is_err() {
                        break;
                    }
                }
            }
        });
    start_watchdog();
    log::info!("unx: loaded {rom} in {:?}", started.elapsed());
    Box::into_raw(Box::new(UnxHandle::Running(Box::new(Unx {
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
        layout: ScreenLayout::TopOnly,
        docked,
        display_size: window,
        hw_docked: docked,
        dock_override: None,
        fixed_resolution,
        settings,
        smooth_forced,
        title_quiet: u32::MAX,
        on_title: false,
        title_boosted: true,
        weapon_sub: 1,
        battle_scale_drop: 0,
        host_last: 0,
        smooth: Smooth {
            wanted: smooth_wanted,
            on: false,
            retry_at: 0,
            backoff: SMOOTH_BACKOFF_FIRST,
            wait: 0.0,
            elapsed: 0.0,
            frames: 0,
            last: None,
            slow: 0,
            strikes: 0,
            kept_up: 0,
            scale_drop_failed: false,
            grace: 0,
        },
        menu,
        data_dir: data_dir_path,
        updater: updater::Updater::new(menu::VERSION_LABEL),
        aim: aim::Aim::from_env(),
        gyro_test: ([0.0; 3], 0),
        cauldron_cursor_x: 0.0,
        cauldron_opacity: 0.48,
        cauldron_hold: 0,
        cmd_rx,
    }))))
}

fn empty_pixels() -> std::sync::Arc<Vec<u8>> {
    static EMPTY_PIXELS: std::sync::OnceLock<std::sync::Arc<Vec<u8>>> = std::sync::OnceLock::new();
    EMPTY_PIXELS.get_or_init(|| std::sync::Arc::new(Vec::new())).clone()
}

/// a screen as the presenter takes it, straight from the GPU when the
/// renderer drew it on the shared device, else its pixels.
fn shown(system: &mut System, screen: Screen) -> (zakuro_core::Screen, Option<zakuro_gpu::GpuScreen>) {
    if let Some(gpu) = system.gpu_screen(screen) {
        let scale = system.gpu.scale();
        return ((empty_pixels(), screen.width() * scale, screen.height() * scale), Some(gpu));
    }
    (system.read_screen_scaled(screen), None)
}

/// where a touch on the 1280x720 screen lands on the 3DS bottom screen's 320x240,
/// if it does. in TopOnly the bottom screen is not shown, and the touch aims the
/// reticle over a 4:3 area in the middle of the screen; otherwise it maps 1:1 to
/// the layout's bottom viewport (R3 matrix in FuseSplitDeck, corner in LargeTop).
fn touch_on_bottom(layout: ScreenLayout, weapon_sub: u8, x: u16, y: u16) -> Option<(u16, u16)> {
    if layout == ScreenLayout::FiendsCauldron && x <= 140 && y >= 595 {
        // Relocated bottom-left 16:9 Back button maps to native (16, 221) on 320x240.
        return Some((16, 221));
    }
    if layout == ScreenLayout::ChapterSelect {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        let u = (x as f32 - bottom.x) / bottom.width;
        let v = (y as f32 - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        const KX: f32 = zakuro_gpu::raster::CHAPTER_BOT_KX;
        let sx_buf = u * 320.0;
        let by = (4.0 + v * 234.5).clamp(0.0, 239.0);
        let bx = if by >= 194.0 {
            if sx_buf <= 34.0 {
                (3.0 + (sx_buf - 3.0) / KX).clamp(0.0, 319.0)
            } else if sx_buf >= 254.0 {
                (286.0 - (286.0 - sx_buf) / KX).clamp(0.0, 319.0)
            } else {
                let t = ((sx_buf - 44.0) / (242.0 - 44.0)).clamp(0.0, 1.0);
                80.1 + t * (243.9 - 80.1)
            }
        } else {
            (160.0 + (sx_buf - 160.0) / KX).clamp(0.0, 319.0)
        };
        return Some((bx as u16, by as u16));
    }
    if layout == ScreenLayout::ModeSelect {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        let u = (x as f32 - bottom.x) / bottom.width;
        let v = (y as f32 - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let wx = u * 426.66666;
        let wy = v * 240.0;
        if wy >= 202.0 {
            return None;
        }
        const S: f32 = zakuro_gpu::raster::MODE_BOT_SCALE;
        const X0: f32 = zakuro_gpu::raster::MODE_BOT_X_ORIGIN;
        let bx = (wx - X0) / S;
        let by = (wy - 2.0) / S;
        if (0.0..320.0).contains(&bx) && (0.0..240.0).contains(&by) {
            return Some((bx as u16, by as u16));
        }
        return None;
    }
    if layout == ScreenLayout::SoloHub {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        let u = (x as f32 - bottom.x) / bottom.width;
        let v = (y as f32 - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let wx = u * 426.66666;
        let wy = v * 240.0;
        const BOUNDS: [[f32; 4]; 7] = [
            [34.0, -22.0, 228.0, 145.0],  // 0: Move Out!
            [220.0, 0.0, 320.0, 78.0],    // 1: Treasure Hunt
            [0.0, 88.0, 92.0, 194.0],     // 2: Practice Range
            [195.0, 68.0, 320.0, 176.0],  // 3: Arms Altar
            [82.0, 176.0, 178.0, 232.0],  // 4: Idol Toss
            [192.0, 172.0, 320.0, 240.0], // 5: Gear
            [0.0, 194.0, 44.0, 239.0],    // 6: Back
        ];
        for ([ox, oy, dx, dy, scale], [x0, y0, x1, y1]) in
            zakuro_gpu::raster::SOLO_PERIMETER_SLOTS.into_iter().zip(BOUNDS)
        {
            let sx = ox + (wx - dx) / scale;
            let sy = oy + (wy - dy) / scale;
            if sx >= x0 && sx <= x1 && sy >= y0 && sy <= y1 {
                return Some((sx.clamp(2.0, 317.0) as u16, sy.clamp(2.0, 237.0) as u16));
            }
        }
        return None;
    }
    if layout == ScreenLayout::FileSelect {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        let u = (x as f32 - bottom.x) / bottom.width;
        let v = (y as f32 - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let wx = u * 426.66666;
        let wy = v * 240.0;
        if wx <= 56.0 && wy >= 195.0 {
            return Some((27, 219));
        }
        if wx >= 312.0 && wy >= 192.0 {
            return Some((258, 220));
        }
        if wy < 114.0 {
            return None;
        }
        let sx = 160.0 + (wx - 213.33333) / zakuro_gpu::raster::FILE_BOT_SCALE;
        let sy = 95.0 + (wy - zakuro_gpu::raster::FILE_BOT_TOMES_Y) / zakuro_gpu::raster::FILE_BOT_SCALE;
        if (0.0..=319.0).contains(&sx) && (0.0..=239.0).contains(&sy) {
            return Some((sx as u16, sy as u16));
        }
        return None;
    }
    if layout == ScreenLayout::WeaponSelect {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        let u = (x as f32 - bottom.x) / bottom.width;
        let v = (y as f32 - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let wx = u * 426.66666;
        let wy = v * 240.0;
        let (bot_x0, bot_s) = match weapon_sub {
            2 => (0.0, 0.78),
            3 => (4.0, 0.78),
            4 => (4.0, 0.80),
            _ => (4.0, 0.77),
        };
        let bot_y0 = 240.0 - 240.0 * bot_s;
        let bx = (wx - bot_x0) / bot_s;
        let by = (wy - bot_y0) / bot_s;
        if !(0.0..320.0).contains(&bx) || !(0.0..240.0).contains(&by) {
            return None;
        }
        let sx = if weapon_sub == 2 {
            if bx <= 56.0 && by >= 194.0 {
                bx - 4.0
            } else {
                bx + 48.0
            }
        } else {
            bx
        };
        if (0.0..320.0).contains(&sx) {
            return Some((sx as u16, by as u16));
        }
        return None;
    }
    let (x, y) = (x as f32, y as f32);
    let (left, top, width, height) = if layout == ScreenLayout::TopOnly {
        (160.0, 0.0, 960.0, 720.0)
    } else {
        let bottom = zakuro_gpu::layout(1280, 720, layout, zakuro_gpu::backend::TOP_WIDTH, false).1?;
        (bottom.x, bottom.y, bottom.width, bottom.height)
    };
    let (u, v) = ((x - left) / width, (y - top) / height);
    ((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)).then(|| ((u * 320.0) as u16, (v * 240.0) as u16))
}

/// settings.ini's minus_button: Minus alone opens the menu (the C shell
/// asks each frame through unx_minus_opens_menu)
static MINUS_MENU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[no_mangle]
pub extern "C" fn unx_minus_opens_menu() -> u32 {
    MINUS_MENU.load(std::sync::atomic::Ordering::Relaxed) as u32
}

/// shows the screens as they are, with the overlay over them
fn present_screens(unx: &mut Unx, overlay: &Overlay) {
    let top = shown(&mut unx.system, Screen::Top);
    let blank = empty_pixels();
    let ((top_px, tw, th), top_gpu) = top;
    // the bottom screen is only fetched when the active layout places it
    let ((bottom_px, bw, bh), bottom_gpu) = if unx.layout == ScreenLayout::TopOnly {
        (((blank.clone()), 0, 0), None)
    } else {
        shown(&mut unx.system, Screen::Bottom)
    };
    let result = unx.presenter.present(
        ScreenImage { width: tw, height: th, pixels: &top_px, gpu: top_gpu },
        ScreenImage { width: bw, height: bh, pixels: &bottom_px, gpu: bottom_gpu },
        ScreenImage { width: tw, height: th, pixels: &blank, gpu: None },
        overlay,
    );
    if let Err(error) = result {
        if !matches!(error, PresentError::OutOfDate) {
            log::error!("unx: present failed, {error}");
        }
    }
}

fn audio_pause(paused: bool) {
    // SAFETY: nx/c audio.c, a flag for its thread
    unsafe { nx_audio_pause(paused) }
}

/// the settings menu: opens and closes on right stick + Minus, and while it
/// is open the game is paused, only the last picture and the menu shown.
/// what unx_frame returns when it took the frame
fn unx_menu(unx: &mut Unx, input: Option<&UnxInput>) -> Option<u32> {
    let (host, buttons) = input.map_or((0, 0), |input| (input.host, input.buttons));
    let was_open = unx.menu.open;
    if host & HOST_MENU != 0 && unx.host_last & HOST_MENU == 0 {
        unx.menu.toggle(buttons, unx.docked);
    }
    if !unx.menu.open {
        if was_open {
            audio_pause(false);
        }
        return None;
    }
    if !was_open {
        audio_pause(true);
        unx.menu.set_save_counts(saves::pending_import_count(), saves::live_save_count(&unx.data_dir));
    }
    unx.menu.set_update_view(unx.updater.view());
    unx.host_last = host;
    let docked = unx.docked;
    match unx.menu.input(buttons, &mut unx.settings, docked) {
        Some(menu::Changed::Video) => {
            unx.fixed_resolution = None;
            unx.battle_scale_drop = 0;
            unx.smooth.scale_drop_failed = false;
            unx.smooth_forced = None;
            unx.apply_settings();
            unx.settings_changed("menu");
        }
        Some(menu::Changed::Controls) => {
            unx.settings.export_controls();
            zakuro_gpu::set_power_label_mode(unx.settings.power_label_mode());
            MINUS_MENU.store(unx.settings.minus_opens_menu(), std::sync::atomic::Ordering::Relaxed);
            unx.aim.reload();
            unx.settings_changed("menu");
        }
        Some(menu::Changed::Update) => {
            if unx.updater.trigger_action() {
                return Some(1);
            }
            unx.menu.set_update_view(unx.updater.view());
        }
        Some(menu::Changed::ImportSave) => {
            match saves::import_from_menu(&unx.data_dir) {
                Ok(msg) => unx.menu.set_status(msg, false),
                Err(err) => unx.menu.set_status(err, true),
            }
            unx.menu.set_save_counts(saves::pending_import_count(), saves::live_save_count(&unx.data_dir));
        }
        Some(menu::Changed::ExportSave) => {
            match saves::export_from_menu(&unx.data_dir) {
                Ok(msg) => unx.menu.set_status(msg, false),
                Err(err) => unx.menu.set_status(err, true),
            }
            unx.menu.set_save_counts(saves::pending_import_count(), saves::live_save_count(&unx.data_dir));
        }
        Some(menu::Changed::Quit) => {
            log::info!("unx: quit from the menu (frame {})", unx.frame);
            return Some(1);
        }
        None => {}
    }
    let overlay = unx.menu_overlay();
    FRAME_PHASE.store(PHASE_PRESENTING, std::sync::atomic::Ordering::Release);
    present_screens(unx, &overlay);
    FRAME_PHASE.store(PHASE_IDLE, std::sync::atomic::Ordering::Release);
    if !unx.menu.open {
        audio_pause(false);
    }
    // the time paused is not a gap in the game's pictures, nor a hitch
    unx.last_picture = None;
    unx.last_exit = None;
    unx.smooth.last = None;
    Some(0)
}

/// how long the last 2 s report took, it runs between two frames.
static REPORT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// runs one emulated frame and presents it. 0 to go on, 1 when the game
/// exited or faulted.
///
/// # Safety
/// `unx` came from unx_create; `input` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn unx_frame(handle: *mut UnxHandle, input: *const UnxInput) -> u32 {
    let unx: &mut Unx = match &mut *handle {
        UnxHandle::Running(unx) => unx,
        UnxHandle::Notice { presenter, menu, notice, armed } => {
            let buttons = input.as_ref().map_or(0, |i| i.buttons);
            if buttons == 0 {
                *armed = true;
            } else if *armed && (buttons & ((1 << 1) | (1 << 3))) != 0 {
                // B (1<<1) or Plus/Start (1<<3) exits setup notice cleanly
                return 1;
            }
            let refs: Vec<&str> = notice.paragraphs.iter().map(String::as_str).collect();
            bootstrap::render_card(
                presenter,
                menu,
                &notice.heading,
                &refs,
                None,
                Some("Press + or B to exit to Homebrew Menu."),
                true,
            );
            return 0;
        }
    };
    CURRENT_FRAME.store(unx.frame, std::sync::atomic::Ordering::Relaxed);
    FRAME_HEARTBEAT.fetch_add(1, std::sync::atomic::Ordering::Release);
    FRAME_PHASE.store(PHASE_IDLE, std::sync::atomic::Ordering::Release);
    unx.updater.maybe_auto_check(unx.on_title);
    if !unx.in_play && !unx.menu.open {
        if let Some(notice) = unx.updater.take_announcement() {
            unx.menu.notify(notice);
        }
    }
    if let Some(done) = unx_menu(unx, input.as_ref()) {
        return done;
    }
    let entered = Instant::now();
    // what the game's overrides say it is doing this frame (see below)
    let said = std::mem::take(&mut unx.system.memory.host_mailbox[3]);
    unx.aim.said(said);
    let controls = aim::GuestControls::read(&mut unx.system.memory);
    unx.aim.set_controls(controls);
    if !unx.in_play {
        unx.aim.patch_controls_ui(&mut unx.system.memory.phys.fcram, unx.frame);
    }
    let on_title = unx.system.gpu.flags().on_title.swap(false, std::sync::atomic::Ordering::Relaxed);
    let mut look = (0.0, 0.0);
    if let Some(input) = input.as_ref() {
        // after the menu closes, nothing reaches the game until all is let go
        let (game_buttons, game_touch) = unx.menu.filter(input.buttons, input.touching != 0);
        let mut state = InputState {
            buttons: PadState::from_bits_truncate(game_buttons),
            circle_x: input.circle_x,
            circle_y: input.circle_y,
            c_stick_x: input.c_stick_x,
            c_stick_y: input.c_stick_y,
            touch: game_touch.then(|| touch_on_bottom(unx.layout, unx.weapon_sub, input.touch_x, input.touch_y)).flatten(),
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
        let motion_active = unx.system.services.hid.accelerometer_users > 0
            || unx.system.services.hid.gyroscope_users > 0;
        unx.aim.apply_motion_tilt(&mut state, gyro, input.host, motion_active);
        let fallback_ok = unx.in_play && !unx.title_boosted && (input.host & HOST_BOTTOM) == 0;
        unx.aim.set_in_battle_play(fallback_ok);
        look = unx.aim.apply(&mut state, gyro, input.host);
        if on_title && (!state.buttons.is_empty() || game_touch) {
            state.buttons |= PadState::A;
        }
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
    let gpu_flags = unx.system.gpu.flags();
    let in_frontend_menu = gpu_flags.on_frontend_menu.swap(false, std::sync::atomic::Ordering::Relaxed) && !in_battle;
    let on_file_pre = gpu_flags.on_file_select.load(std::sync::atomic::Ordering::Relaxed) && !in_battle;
    unx.title_quiet = if on_file_pre {
        u32::MAX
    } else if on_title && !in_battle {
        0
    } else {
        unx.title_quiet.saturating_add(1)
    };
    let on_title_pre = unx.title_quiet <= 15 && !in_battle && !on_file_pre;
    let on_special_pre = (on_title_pre
        || gpu_flags.on_fuse.load(std::sync::atomic::Ordering::Relaxed)
        || gpu_flags.on_cauldron.load(std::sync::atomic::Ordering::Relaxed)
        || gpu_flags.on_chapter.load(std::sync::atomic::Ordering::Relaxed)
        || gpu_flags.on_mode.load(std::sync::atomic::Ordering::Relaxed)
        || gpu_flags.on_solo.load(std::sync::atomic::Ordering::Relaxed)
        || gpu_flags.on_weapon.load(std::sync::atomic::Ordering::Relaxed)
        || on_file_pre)
        && !in_battle;
    // Resolution stays locked to battle_scale across quiet flight rails and
    // cutscenes once play starts, until returning to a frontend menu.
    let res_battle = said != 0 || (unx.bottom.played && !in_frontend_menu && !on_special_pre);
    let on_tutorial_movie = in_battle && !res_battle;
    let bottom_shown = unx.bottom.update(host, said, on_special_pre, on_tutorial_movie, in_frontend_menu);
    unx.system
        .gpu
        .flags()
        .menu_override
        .store(unx.bottom.fuse_override, std::sync::atomic::Ordering::Relaxed);
    let in_play = !bottom_shown && !on_special_pre;
    let host_pressed = host & !unx.host_last;
    unx.host_last = host;
    unx.controller_scale(host_pressed);
    unx.controller_smooth(host_pressed);
    let flags = unx.system.gpu.flags();
    flags.in_play.store(in_play, std::sync::atomic::Ordering::Relaxed);
    flags.in_battle.store(in_battle, std::sync::atomic::Ordering::Relaxed);
    flags.active_battle.store(res_battle, std::sync::atomic::Ordering::Relaxed);
    if said != 0 && unx.frame % 120 == 0 {
        log::info!("unx: overrides say {said} (frame {})", unx.frame);
    }
    if in_play != unx.in_play {
        log::info!("unx: bottom screen {} at frame {} (said {said}, quiet for {})", if in_play { "hidden" } else { "shown" }, unx.frame, unx.out_of_play);
        unx.in_play = in_play;
    }

    let hw_docked = if input.is_null() {
        appletGetOperationMode() == 1
    } else {
        host & HOST_DOCKED != 0
    };
    if hw_docked != unx.hw_docked {
        unx.hw_docked = hw_docked;
        unx.dock_override = None;
    }
    let docked = unx.dock_override.unwrap_or(hw_docked);
    if docked != unx.docked {
        unx.apply_docked(docked);
    } else {
        let next_size = query_window_size(docked, unx.hw_docked, false);
        if next_size != unx.display_size {
            unx.display_size = next_size;
            unx.presenter.resize(next_size.0, next_size.1);
            unx.smooth.grace = SMOOTH_GRACE_FRAMES;
            log::info!(
                "unx: display resolution changed -> {}x{} (frame {})",
                next_size.0,
                next_size.1,
                unx.frame,
            );
        }
    }

    let title_boost = !res_battle;
    if title_boost != unx.title_boosted {
        unx.title_boosted = title_boost;
        unx.battle_scale_drop = 0;
        unx.scene_changed(if title_boost { "out of a battle" } else { "into a battle" });
        if unx.fixed_resolution.is_none() {
            let target = unx.auto_scale();
            let scale = unx.set_scale(target);
            log::info!(
                "unx: non-battle resolution {} -> {}x (frame {})",
                if title_boost { "boost" } else { "restore" },
                scale,
                unx.frame,
            );
        }
    }

    // development: sdmc:/switch/uprisingnx/cmd.txt holding "save" or "load" (optionally
    // followed by a slot name) saves or restores the whole console, so that a test
    // can start from the same moment every time. the file is removed when read, and
    // the result is written to cmd.result
    if unx.frame % 15 == 0 {
        while let Ok(text) = unx.cmd_rx.try_recv() {
            let mut words = text.split_whitespace();
            let (verb, slot) = (words.next().unwrap_or(""), words.next().unwrap_or("slot0"));
            let file = format!("sdmc:/switch/uprisingnx/{slot}.zst");
            let started = Instant::now();
            let result = match verb {
                "save" => unx.system.save_state().and_then(|bytes| {
                    std::fs::write(&file, &bytes).map(|_| format!("saved {} bytes", bytes.len())).map_err(|e| e.to_string())
                }),
                "load" => {
                    let loaded = load_slot(&mut unx.system, slot);
                    unx.out_of_play = u32::MAX;
                    unx.scene_changed("a state loaded");
                    loaded
                }
                "scale" => match slot {
                    "auto" => {
                        unx.fixed_resolution = None;
                        unx.battle_scale_drop = 0;
                        unx.smooth.scale_drop_failed = false;
                        let target = unx.auto_scale();
                        let scale = unx.set_scale(target);
                        Ok(format!("auto resolution -> {scale}x"))
                    }
                    _ => match slot.parse::<u32>() {
                        Ok(target) => {
                            unx.fixed_resolution = Some(target);
                            let scale = unx.set_scale(target);
                            Ok(format!("resolution -> {scale}x"))
                        }
                        Err(error) => Err(format!("bad scale {slot:?}, {error}")),
                    },
                },
                "dock" => match slot {
                    "1" | "docked" | "console" => {
                        unx.dock_override = Some(true);
                        unx.apply_docked(true);
                        Ok(format!("docked {}x{} at {}x", unx.display_size.0, unx.display_size.1, unx.system.gpu.scale()))
                    }
                    "0" | "handheld" => {
                        unx.dock_override = Some(false);
                        unx.apply_docked(false);
                        Ok(format!("handheld {}x{} at {}x", unx.display_size.0, unx.display_size.1, unx.system.gpu.scale()))
                    }
                    "auto" => {
                        unx.dock_override = None;
                        unx.apply_docked(unx.hw_docked);
                        Ok(format!(
                            "auto ({} {}x{}) at {}x",
                            if unx.docked { "docked" } else { "handheld" },
                            unx.display_size.0,
                            unx.display_size.1,
                            unx.system.gpu.scale(),
                        ))
                    }
                    _ => Err(format!("bad dock mode {slot:?}, expected 0, 1, handheld, docked, or auto")),
                },
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
    FRAME_PHASE.store(PHASE_EMULATING, std::sync::atomic::Ordering::Release);
    let outcome = unx.system.run_frame();
    FRAME_PHASE.store(PHASE_IDLE, std::sync::atomic::Ordering::Release);
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
    unx.smooth_guard(gpu.waiting);
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

    if unx.system.gpu.flags().on_title.load(std::sync::atomic::Ordering::Relaxed) && !in_battle {
        unx.title_quiet = 0;
    }
    let on_file_post = unx.system.gpu.flags().on_file_select.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle;
    if on_file_post {
        unx.title_quiet = u32::MAX;
    }
    let on_title_active = unx.title_quiet <= 15 && !in_battle && !on_file_post;
    unx.on_title = on_title_active;
    let on_fuse_post = unx.system.gpu.flags().on_fuse.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active;
    let on_cauldron_post = unx.system.gpu.flags().on_cauldron.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active
        && !on_fuse_post;
    let on_chapter_post = unx.system.gpu.flags().on_chapter.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active
        && !on_fuse_post
        && !on_cauldron_post;
    let on_solo_post = unx.system.gpu.flags().on_solo.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active
        && !on_fuse_post
        && !on_cauldron_post
        && !on_chapter_post;
    let on_mode_post = unx.system.gpu.flags().on_mode.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active
        && !on_fuse_post
        && !on_cauldron_post
        && !on_chapter_post
        && !on_solo_post;
    let on_weapon_post = unx.system.gpu.flags().on_weapon.load(std::sync::atomic::Ordering::Relaxed)
        && !in_battle
        && !on_title_active
        && !on_fuse_post
        && !on_cauldron_post
        && !on_chapter_post
        && !on_solo_post
        && !on_mode_post;
    let on_file_post = on_file_post
        && !on_fuse_post
        && !on_cauldron_post
        && !on_chapter_post
        && !on_solo_post
        && !on_mode_post
        && !on_weapon_post;
    let bottom_blank = unx.system.gpu.flags().bottom_blank.load(std::sync::atomic::Ordering::Relaxed);
    let desired_layout = if on_fuse_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::FuseSplitDeck
        }
    } else if on_cauldron_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::FiendsCauldron
        }
    } else if on_chapter_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::ChapterSelect
        }
    } else if on_solo_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::SoloHub
        }
    } else if on_mode_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::ModeSelect
        }
    } else if on_weapon_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::WeaponSelect
        }
    } else if on_file_post {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::FileSelect
        }
    } else if on_title_active {
        if unx.bottom.fuse_override {
            ScreenLayout::LargeTop
        } else {
            ScreenLayout::TopOnly
        }
    } else if !bottom_shown || bottom_blank {
        ScreenLayout::TopOnly
    } else {
        ScreenLayout::LargeTop
    };
    if on_chapter_post {
        let raw_bits = unx.system.gpu.flags().chapter_title_right.load(std::sync::atomic::Ordering::Relaxed);
        let title_r = if raw_bits != 0 { f32::from_bits(raw_bits) } else { 302.0 };
        unx.presenter.set_chapter_state(title_r);
    }
    if on_weapon_post {
        let sub = unx.system.gpu.flags().weapon_sub.load(std::sync::atomic::Ordering::Relaxed).max(1);
        if sub != unx.weapon_sub {
            unx.weapon_sub = sub;
            unx.system.reset_interpolation();
        }
        unx.presenter.set_weapon_state(sub);
    }
    if on_cauldron_post {
        let raw_bits = unx.system.gpu.flags().cauldron_cursor_x.load(std::sync::atomic::Ordering::Relaxed);
        let cx = if raw_bits != 0 { f32::from_bits(raw_bits) } else { 237.4 };
        let touching_slider = input
            .as_ref()
            .is_some_and(|inp| inp.touching != 0 && !unx.menu.open && inp.touch_y < 320 && inp.touch_x >= 165 && inp.touch_x <= 1125);
        if unx.cauldron_cursor_x > 0.0 && (cx - unx.cauldron_cursor_x).abs() > 0.04 {
            unx.cauldron_hold = 30;
        } else if touching_slider {
            unx.cauldron_hold = 30;
        }
        unx.cauldron_cursor_x = cx;
        if touching_slider || unx.cauldron_hold > 0 {
            if !touching_slider {
                unx.cauldron_hold = unx.cauldron_hold.saturating_sub(1);
            }
            unx.cauldron_opacity = (unx.cauldron_opacity + 0.16).min(1.0);
        } else {
            unx.cauldron_opacity = (unx.cauldron_opacity - 0.035).max(0.48);
        }
        let hud_visible = unx.system.gpu.flags().cauldron_hud.load(std::sync::atomic::Ordering::Relaxed);
        let room_rgb = unx.system.gpu.flags().cauldron_room_rgb.load(std::sync::atomic::Ordering::Relaxed);
        let hud_flags = (hud_visible as u32) | (room_rgb << 8);
        unx.presenter
            .set_cauldron_state(cx.clamp(16.0, 304.0), unx.cauldron_opacity, hud_flags);
    } else {
        unx.cauldron_cursor_x = 0.0;
        unx.cauldron_opacity = 0.48;
        unx.cauldron_hold = 0;
    }
    if desired_layout != unx.layout {
        log::info!(
            "unx: layout {:?} -> {:?} at frame {} (said {said}, quiet {}, on_fuse {on_fuse_post}, on_cauldron {on_cauldron_post}, blank {bottom_blank})",
            unx.layout,
            desired_layout,
            unx.frame,
            unx.out_of_play,
        );
        unx.layout = desired_layout;
        unx.presenter.set_layout(desired_layout);
        unx.system.reset_interpolation();
        if !unx.smooth_allowed() && unx.smooth.on {
            unx.set_smooth(false);
        }
    }

    let overlay = unx.menu_overlay();
    FRAME_PHASE.store(PHASE_PRESENTING, std::sync::atomic::Ordering::Release);
    present_screens(unx, &overlay);
    FRAME_PHASE.store(PHASE_IDLE, std::sync::atomic::Ordering::Release);

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
            if unx.smooth.on {
                report!("  smooth mode: last frame's draws {:?}", unx.system.interpolation_matches());
            }
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
        // quiet, or busy drawing nothing for a while (a stall), every 10 s
        let stalled = unx.draws == 0 && unx.frame > 1200;
        if (unx.window_emulating.as_secs_f64() * 1000.0 / frames < 2.0 || stalled) && unx.frame % 600 < frames as u64 {
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
        if unx.frame % 600 < frames as u64 {
            // the process's memory, to see a leak build up over a long session
            let mut memory = [0u64; 4];
            unsafe { unx_memory(memory.as_mut_ptr()) };
            let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
            report!(
                "  memory: process {:.1} of {:.1} MB, malloc in use {:.1} MB of {:.1} MB taken",
                mb(memory[0]), mb(memory[1]), mb(memory[2]), mb(memory[3])
            );
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
pub unsafe extern "C" fn unx_destroy(handle: *mut UnxHandle) {
    FRAME_PHASE.store(PHASE_EXITING, std::sync::atomic::Ordering::Release);
    WATCHDOG_STOP.store(true, std::sync::atomic::Ordering::Release);
    if !handle.is_null() {
        let boxed = Box::from_raw(handle);
        if matches!(*boxed, UnxHandle::Running(_)) {
            nx_audio_exit();
        }
        drop(boxed);
    }
    stop_reporter();
}
