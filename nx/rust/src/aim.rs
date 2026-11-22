//! Aiming and camera control with the right stick (and optional left-stick mode).
//!
//! How the game aims, from its overrides (`kiu_air_aim.c` and `kiu_mouse_look.c`):
//! - Flying (`GAME_FLYING`): the circle pad (left stick) steers Pit. When the
//!   host mailbox holds `MOUSE_AIM` (`"MAM1"`), `0x006D269C` holds the reticle
//!   steady against Pit's movement drift and `0x006D2ABC` moves the reticle by
//!   `(across, down) * 1.25` top-screen pixels before clamping it to the screen.
//! - On foot (`GAME_ON_FOOT`): the circle pad (left stick) walks. When the host
//!   mailbox holds `MOUSE_LOOK` (`"MLK1"`), `0x007462A4` turns the camera yaw
//!   (`s0`) and pitch (`s1`) directly each tick without the 3DS C-stick's heavy
//!   damping or coasting inertia.
//!
//! Environment switches:
//! - `UNX_RIGHT_STICK_AIM` (default `1`): right stick aims the reticle via
//!   `MOUSE_AIM` while flying and turns the camera via `MOUSE_LOOK` on foot.
//! - `UNX_GYRO_AIM` (default `1`): Switch SixAxis gyroscope angular velocity
//!   contributes fine aiming deltas in both air and land battles.
//! - `UNX_GYRO_SPEED` (default `1.0`): additional multiplier on gyro sensitivity.
//! - `UNX_LEFT_STICK_AIM` (default `0`):
//!   `0` off, the left stick only steers/walks Pit;
//!   `1` flying: the left stick also places a synthetic stylus touch, and on
//!       foot holding the modifier turns the camera with the left stick;
//!   `2` as `1`, but flying only aims with the left stick while the modifier is held.
//! - `UNX_AIM_MODIFIER`: `ZL` (default), `ZR`, `L`, `R`, `X`, `Y` or `A`.
//! - `UNX_AIM_INVERT_Y=1`: flips vertical aim/camera direction.
//! - `UNX_AIM_SPEED`: scales aiming and camera turn speed (default `1.0`).
//!
//! Native in-game Options -> Controls menu integration (`0x00EF9638` in guest BSS):
//! - Air Battle Reticle Speed slider (`+0x160`, `[-1.0, +1.0]`) scales air right-stick
//!   and gyro speed from `0.5x` to `2.0x` via the game's `f_006014EC` curve.
//! - Air Battle Stop Sensitivity slider (`+0x154`, `[-1.0, +1.0]`) scales air gyro
//!   sensitivity from `0.0x` (off at `<= -0.95`) through `1.0x` (default `0.0`) to `2.0x`.
//! - Land Battle Left/Right & Up/Down Speed sliders (`+0x158`, `+0x15C`, `[-1.0, +1.0]`)
//!   scale on-foot horizontal and vertical right-stick and gyro speed from `0.5x` to `2.0x`.
//! - Land Battle Stop Sensitivity slider (`+0x150`, `[-1.0, +1.0]`) scales on-foot gyro
//!   sensitivity from `0.0x` (off at `<= -0.95`) through `1.0x` (default `0.0`) to `2.0x`.
//! - Air Battle & Land Battle Up/Down Invert toggles (`+0x173` bits 1 and 0) invert
//!   vertical aiming in air and on foot respectively.

use zakuro_core::memory::Memory;
use zakuro_core::services::hid::{InputState, PadState};

/// Guest BSS address of the active profile's Controls configuration block
/// (`0x00EE33C0 + 0x16278`, see `f_007DAA9C` in `code564.c` and `code063.c`).
pub const GUEST_CONTROLS_VADDR: u32 = 0x00EF9638;

/// Host mailbox mark for on-foot camera look (`"MLK1"`), followed by `across`
/// and `down` as `f32` bottom-screen pixels (`zakuro::input::MOUSE_LOOK`).
const MOUSE_LOOK: u32 = u32::from_le_bytes(*b"MLK1");
/// Host mailbox mark to level on-foot camera pitch (`"MLKR"`).
const MOUSE_LOOK_RECENTER: u32 = u32::from_le_bytes(*b"MLKR");
/// Host mailbox mark for flying reticle aim (`"MAM1"`), followed by `across`
/// and `down` as `f32` bottom-screen pixels (`zakuro::input::MOUSE_AIM`).
const MOUSE_AIM: u32 = u32::from_le_bytes(*b"MAM1");
/// Host mailbox mark to snap flying reticle back to rail center (`"MAMR"`).
const MOUSE_AIM_RECENTER: u32 = u32::from_le_bytes(*b"MAMR");
/// Host mailbox fourth word written by the game's overrides each tick.
pub const GAME_ON_FOOT: u32 = 1;
pub const GAME_FLYING: u32 = 2;
pub const GAME_VEHICLE: u32 = 3;

/// Bottom-screen touch coordinates (`320x240`) of the Vehicle Special Action HUD
/// button (Exo Tank Dash/Boost, Cherubot Jump/Hover, Aether Ring Shield).
const VEHICLE_SPECIAL_TOUCH: (u16, u16) = (28, 212);
/// Bottom-screen touch coordinates (`320x240`) of the Vehicle Dismount HUD button.
const VEHICLE_DISMOUNT_TOUCH: (u16, u16) = (86, 218);
/// Number of 60 Hz frames to hold a synthetic vehicle HUD tap before releasing
/// so the 3DS UI widget's `is_clicked()` (`vtable + 0x8`) callback fires immediately
/// without waiting for the physical `A` or `B` button to be released.
const VEHICLE_TAP_FRAMES: u8 = 3;

/// Right-stick radial dead zone and linear+quadratic precision curve weight.
const DEAD_ZONE: f32 = 0.10;
const LINEAR_GAIN: f32 = 0.35;
/// Left-stick radial dead zone to prevent minor stick drift from walking Pit.
const LEFT_DEAD_ZONE: f32 = 0.12;
/// Gyro radial deadband in turns/sec (`0.0025` turns/s = `0.9 deg/s`) to suppress
/// resting MEMS sensor bias without introducing a step on micro-adjustments.
const GYRO_DEAD_ZONE: f32 = 0.0025;
/// Flying reticle speed in bottom-screen pixels per 60 Hz frame at full stick
/// deflection (`6.0 * 1.25 = 7.5` top-screen px/frame = `450` px/s across 400x240).
const AIR_PIXELS_PER_FRAME: f32 = 6.0;
/// On-foot camera turn speed in bottom-screen pixels per 60 Hz frame at full
/// stick deflection (`10.0 * 0.00408 rad = 140 deg/s` yaw).
const LAND_X_PIXELS_PER_FRAME: f32 = 10.0;
const LAND_Y_PIXELS_PER_FRAME: f32 = 7.0;
/// Gyro sensitivity in bottom-screen pixels per frame per `1.0` turn/sec (`360 deg/s`).
const AIR_GYRO_PIXELS_PER_TURN_RATE: f32 = 50.0;
const LAND_GYRO_X_PIXELS_PER_TURN_RATE: f32 = 64.0;
const LAND_GYRO_Y_PIXELS_PER_TURN_RATE: f32 = 52.0;
/// Frames the game may leave the mailbox movement unread before it is dropped.
const STALE_FRAMES: u8 = 4;
/// Frames without a word from the game's overrides before it is out of play.
const QUIET_FRAMES: u8 = 10;
/// 60 Hz frames after releasing R / L3 during which gyro stays muted so the
/// physical button-release jolt does not pull the reticle/camera off center.
const RECENTER_GRACE_FRAMES: u8 = 6;
/// Stylus reach from the center of the 320x240 bottom screen for legacy
/// left-stick touch mode.
const REACH_X: f32 = 150.0;
const REACH_Y: f32 = 112.0;
/// Virtual bottom-screen stylus dimensions (`320x240`) for in-battle fallback
/// aiming when neither `kiu_air_aim` nor `kiu_mouse_look` runs (`said == 0`,
/// e.g. Chapter 17/21/25 falling/aiming reticle sequences).
const FALLBACK_TOUCH_CENTER: (f32, f32) = (160.0, 120.0);
const FALLBACK_TOUCH_MAX_X: f32 = 319.0;
const FALLBACK_TOUCH_MAX_Y: f32 = 239.0;
/// Deliberate gyro angular rate (`0.025` turns/s = `9 deg/s`) required to
/// initially engage fallback touch aiming from idle, so resting hand tremor
/// during cutscenes or pause menus never injects a stray bottom-screen touch.
const FALLBACK_GYRO_ENGAGE_THRESHOLD: f32 = 0.025;
/// 60 Hz frames to keep the fallback virtual stylus lifted when `R`/`L3`
/// recenters it, so relative-drag modes re-anchor cleanly at `(160, 120)`.
const FALLBACK_RECENTER_LIFT_FRAMES: u8 = 2;
/// 3DS `hid:USER` motion tilt limit (`±90 deg`, matching desktop `zakuro`).
const MOTION_TILT_LIMIT: f32 = std::f32::consts::FRAC_PI_2;
/// Radians per 60 Hz frame of virtual console tilt at full right-stick deflection.
const MOTION_TILT_STICK_RAD_PER_FRAME: f32 = 0.05;
/// Radians per 60 Hz frame per `1.0` turn/sec (`2π / 60`).
const MOTION_TILT_GYRO_RAD_PER_TURN_RATE: f32 = std::f32::consts::TAU / 60.0;

/// Live snapshot of the in-game Options -> Controls menu sliders and invert toggles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuestControls {
    pub initialized: bool,
    pub land_stop: f32,
    pub air_stop: f32,
    pub land_x_speed: f32,
    pub land_y_speed: f32,
    pub air_speed: f32,
    pub flags_173: u8,
}

impl Default for GuestControls {
    fn default() -> Self {
        Self {
            initialized: false,
            land_stop: 0.0,
            air_stop: 0.0,
            land_x_speed: 0.0,
            land_y_speed: 0.0,
            air_speed: 0.0,
            flags_173: 0x02,
        }
    }
}

impl GuestControls {
    /// Reads the live Controls settings block from guest memory at `0x00EF9638`.
    pub fn read(memory: &mut Memory) -> Self {
        let word_170 = memory.peek32(GUEST_CONTROLS_VADDR + 0x170);
        if (word_170 & 0xFFFF) == 0 {
            return Self::default();
        }
        let clean = |bits: u32| {
            let f = f32::from_bits(bits);
            if f.is_finite() && (-1.0..=1.0).contains(&f) { f } else { 0.0 }
        };
        Self {
            initialized: true,
            land_stop: clean(memory.peek32(GUEST_CONTROLS_VADDR + 0x150)),
            air_stop: clean(memory.peek32(GUEST_CONTROLS_VADDR + 0x154)),
            land_x_speed: clean(memory.peek32(GUEST_CONTROLS_VADDR + 0x158)),
            land_y_speed: clean(memory.peek32(GUEST_CONTROLS_VADDR + 0x15C)),
            air_speed: clean(memory.peek32(GUEST_CONTROLS_VADDR + 0x160)),
            flags_173: (word_170 >> 24) as u8,
        }
    }

    /// Exact slider curve from `f_006014EC` (`code449.c:24419`): maps `[-1.0, +1.0]`
    /// to `[0.5, 2.0]` with `0.0 -> 1.0`.
    pub fn slider_scale(s: f32) -> f32 {
        if !s.is_finite() {
            return 1.0;
        }
        let s = s.clamp(-1.0, 1.0);
        if s >= 0.0 { 1.0 + s } else { 1.0 + 0.5 * s }
    }

    /// Maps the Stop Sensitivity slider (`[-1.0, +1.0]`) to Gyro Sensitivity
    /// (`0.0x` at minimum `<= -0.95` to disable gyro, `1.0x` at `0.0`, `2.0x` at `+1.0`).
    pub fn gyro_scale(s: f32) -> f32 {
        if !s.is_finite() {
            return 1.0;
        }
        let s = s.clamp(-1.0, 1.0);
        if s <= -0.95 {
            0.0
        } else if s >= 0.0 {
            1.0 + s
        } else {
            (1.0 + s).max(0.0)
        }
    }

    pub fn air_invert_y(self) -> bool {
        self.initialized && (self.flags_173 & 0x02) == 0
    }

    pub fn land_invert_y(self) -> bool {
        self.initialized && (self.flags_173 & 0x01) != 0
    }
}

pub struct Aim {
    right_stick: bool,
    gyro: bool,
    gyro_speed: f32,
    mode: u32,
    modifier: PadState,
    invert_y: bool,
    speed: f32,
    /// What the overrides last said (`GAME_ON_FOOT` or `GAME_FLYING`), `0` for nothing.
    game: u32,
    quiet: u8,
    untaken: u8,
    /// Whether real touchscreen input was active this frame.
    touching: bool,
    /// Whether the modifier was held last frame, for logging.
    aiming: bool,
    /// Last seen in-game Controls settings, for change logging and scaling.
    controls: GuestControls,
    /// Cached FCRAM byte offset of the Options/Menu `MsgStdBn` (`_r/b_03`, size `0x26DD0`).
    msbt_offset: Option<usize>,
    /// Rising-edge tracker for the gyro/reticle recenter button (`R` or `StickL`).
    recenter_held: bool,
    /// Set when a gyro/reticle recenter has been requested for the next guest tick.
    recenter_pending: bool,
    /// Remaining 60 Hz frames to mute gyro after releasing recenter (gyro ratchet settle).
    recenter_grace: u8,
    /// Rising-edge tracker for `A` in `GAME_VEHICLE` (tap-and-release Special Action).
    vehicle_a_held: bool,
    /// Rising-edge tracker for `B` in `GAME_VEHICLE` (tap-and-release Dismount).
    vehicle_b_held: bool,
    /// Active synthetic touchscreen tap pulse `((x, y), frames_remaining)` for vehicle HUD buttons.
    vehicle_tap: Option<((u16, u16), u8)>,
    /// Whether the game is in active battle play with the bottom screen hidden
    /// (`in_play && !title_boosted`), enabling fallback virtual-stylus aiming
    /// when `self.game == 0` (Chapter 17/21/25 falling reticle sequences, etc.).
    in_battle_play: bool,
    /// Virtual bottom-screen stylus `(x, y)` while fallback touch aiming is active.
    fallback_touch: Option<(f32, f32)>,
    /// Remaining 60 Hz frames to keep the fallback stylus lifted after `R`/`L3` recenter.
    fallback_lift: u8,
    /// Integrated 3DS `hid:USER` motion tilt `[x, y]` in radians when a title
    /// enables `EnableAccelerometer` or `EnableGyroscopeLow`.
    motion_tilt: [f32; 2],
}

/// Relative byte offsets and maximum UTF-16 character counts inside the Options/Menu
/// `MsgStdBn` (`_r/b_03`, total size `0x0002_6DD0`) for Controls UI strings.
const MSBT_MENU_SIZE: u32 = 0x0002_6DD0;
const MSBT_CONTROLS_PATCHES: &[(usize, usize, &str)] = &[
    (0x10E7C, 14, "Stick / Gyro"),
    (0x10F0E, 35, "Adjust Right-Stick & Gyro Aiming."),
    (0x116E0, 22, "Stick &\nGyro\n(Default)"),
    (0x1170E, 29, "Reticle: Right Stick & Gyro"),
    (0x11904, 13, "\u{2190}Gyro Off"),
    (0x11920, 12, "Gyro Fast\u{2192}"),
    (0x119B6, 26, "R-Stick & Gyro (R:Center)"),
    (0x11A3A, 36, "Adjusts vertical Stick & Gyro speed."),
    (0x11A84, 38, "Adjusts horizontal Stick & Gyro speed."),
    (0x11AD2, 51, "Adjusts Right-Stick and Gyro Aiming reticle speed."),
    (
        0x11B3A,
        77,
        "This adjusts Gyro Aiming sensitivity.\nSlide left for Off. Press R to center.",
    ),
    (0x11BD6, 51, "Adjust Air Battle Stick, Gyro & Invert Y controls."),
    (0x11C3E, 28, "Adjust Land Stick & Gyro."),
];

fn is_menu_msbt(fcram: &[u8], base: usize) -> bool {
    base + (MSBT_MENU_SIZE as usize) <= fcram.len()
        && &fcram[base..base + 8] == b"MsgStdBn"
        && u32::from_le_bytes([
            fcram[base + 0x12],
            fcram[base + 0x13],
            fcram[base + 0x14],
            fcram[base + 0x15],
        ]) == MSBT_MENU_SIZE
}

fn modifier(name: &str) -> PadState {
    match name.trim().to_ascii_uppercase().as_str() {
        "ZR" => PadState::ZR,
        "L" => PadState::L,
        "R" => PadState::R,
        "X" => PadState::X,
        "Y" => PadState::Y,
        "A" => PadState::A,
        _ => PadState::ZL,
    }
}

impl Aim {
    /// takes the player's options from the env vars again (the menu changed
    /// them), keeping what the aim is doing
    pub fn reload(&mut self) {
        let fresh = Aim::from_env();
        self.right_stick = fresh.right_stick;
        self.gyro = fresh.gyro;
        self.gyro_speed = fresh.gyro_speed;
        self.mode = fresh.mode;
        self.modifier = fresh.modifier;
        self.invert_y = fresh.invert_y;
        self.speed = fresh.speed;
    }

    pub fn from_env() -> Aim {
        let var = |name: &str| std::env::var(name).ok();
        let aim = Aim {
            right_stick: var("UNX_RIGHT_STICK_AIM").map_or(true, |v| v.trim() != "0"),
            gyro: var("UNX_GYRO_AIM").map_or(true, |v| v.trim() != "0"),
            gyro_speed: var("UNX_GYRO_SPEED").and_then(|v| v.trim().parse().ok()).unwrap_or(1.0),
            mode: var("UNX_LEFT_STICK_AIM").and_then(|v| v.trim().parse().ok()).unwrap_or(0),
            modifier: modifier(var("UNX_AIM_MODIFIER").as_deref().unwrap_or("ZL")),
            invert_y: var("UNX_AIM_INVERT_Y").is_some_and(|v| v.trim() == "1"),
            speed: var("UNX_AIM_SPEED").and_then(|v| v.trim().parse().ok()).unwrap_or(1.0),
            game: 0,
            quiet: 0,
            untaken: 0,
            touching: false,
            aiming: false,
            controls: GuestControls::default(),
            msbt_offset: None,
            recenter_held: false,
            recenter_pending: false,
            recenter_grace: 0,
            vehicle_a_held: false,
            vehicle_b_held: false,
            vehicle_tap: None,
            in_battle_play: false,
            fallback_touch: None,
            fallback_lift: 0,
            motion_tilt: [0.0, 0.0],
        };
        log::info!(
            "unx: aim right_stick {} gyro {} (gyro_speed {}) left_stick_mode {} (modifier {:?}, invert y {}, speed {})",
            aim.right_stick, aim.gyro, aim.gyro_speed, aim.mode, aim.modifier, aim.invert_y, aim.speed
        );
        aim
    }

    /// Patches the Options -> Controls UTF-16LE string table (`_r/b_03` `MsgStdBn`)
    /// in guest FCRAM so the in-game UI natively displays Right-Stick & SixAxis Gyro
    /// Aiming labels and descriptions.
    pub fn patch_controls_ui(&mut self, fcram: &mut [u8], frame: u64) {
        if !self.right_stick && !self.gyro {
            return;
        }
        let valid_cached = self.msbt_offset.filter(|&off| is_menu_msbt(fcram, off));
        let base = match valid_cached {
            Some(off) => off,
            None => {
                self.msbt_offset = None;
                if is_menu_msbt(fcram, 0x0172_d300) {
                    self.msbt_offset = Some(0x0172_d300);
                    0x0172_d300
                } else {
                    if !frame.is_multiple_of(15) {
                        return;
                    }
                    let start_min = 0x0100_0000usize;
                    let search_end = fcram.len().min(0x0280_0000).saturating_sub(MSBT_MENU_SIZE as usize);
                    if search_end <= start_min {
                        return;
                    }
                    let slice_idx = ((frame / 15) % 4) as usize;
                    let span = (search_end - start_min).div_ceil(4);
                    let mut off = (start_min + slice_idx * span) & !0xFF;
                    let slice_end = (off + span).min(search_end);
                    let mut found = None;
                    while off <= slice_end {
                        if fcram[off] == b'M' && is_menu_msbt(fcram, off) {
                            found = Some(off);
                            break;
                        }
                        off += 0x100;
                    }
                    let Some(off) = found else { return };
                    self.msbt_offset = Some(off);
                    off
                }
            }
        };

        // Check sentinel at `base + 0x11904`: unpatched is `\u{2190}S` (`[0x90, 0x21, b'S', 0x00]`).
        let sentinel = base + 0x11904;
        if &fcram[sentinel..sentinel + 4] != &[0x90, 0x21, b'S', 0x00] {
            return;
        }

        for &(rel_off, max_chars, text) in MSBT_CONTROLS_PATCHES {
            let dst_start = base + rel_off;
            let slot_bytes = (max_chars + 1) * 2;
            let dst = &mut fcram[dst_start..dst_start + slot_bytes];
            dst.fill(0);
            for (i, unit) in text.encode_utf16().take(max_chars).enumerate() {
                let bytes = unit.to_le_bytes();
                dst[i * 2] = bytes[0];
                dst[i * 2 + 1] = bytes[1];
            }
        }
        log::info!("unx: patched Controls menu UI strings for Stick & Gyro Aiming at fcram+{base:#x}");
    }

    /// Updates the cached in-game Controls menu settings (`0x00EF9638`), logging when changed.
    pub fn set_controls(&mut self, controls: GuestControls) {
        if controls.initialized && controls != self.controls {
            log::info!(
                "unx: guest controls updated: air_speed {:+.2} (x{:.2}), air_gyro_slider {:+.2} (x{:.2}), land_x {:+.2} (x{:.2}), land_y {:+.2} (x{:.2}), land_gyro_slider {:+.2} (x{:.2}), air_inv_y {}, land_inv_y {}",
                controls.air_speed,
                GuestControls::slider_scale(controls.air_speed),
                controls.air_stop,
                GuestControls::gyro_scale(controls.air_stop),
                controls.land_x_speed,
                GuestControls::slider_scale(controls.land_x_speed),
                controls.land_y_speed,
                GuestControls::slider_scale(controls.land_y_speed),
                controls.land_stop,
                GuestControls::gyro_scale(controls.land_stop),
                controls.air_invert_y(),
                controls.land_invert_y(),
            );
            self.controls = controls;
        }
    }

    /// Updates what the game's overrides reported this frame (`0` for nothing).
    pub fn said(&mut self, said: u32) {
        if said != 0 {
            self.game = said;
            self.quiet = 0;
        } else if self.game != 0 {
            self.quiet += 1;
            if self.quiet > QUIET_FRAMES {
                self.game = 0;
            }
        }
    }

    /// Right-stick response curve after radial dead zone: blends linear gain
    /// (`LINEAR_GAIN`) for immediate micro-adjustments with quadratic ramp for
    /// fast full-stick sweeps while preserving exact stick direction.
    fn stick(x: f32, y: f32) -> (f32, f32) {
        let length = (x * x + y * y).sqrt();
        if length < DEAD_ZONE {
            return (0.0, 0.0);
        }
        let t = ((length - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0);
        let curved = LINEAR_GAIN * t + (1.0 - LINEAR_GAIN) * t * t;
        let gain = curved / length;
        (x * gain, y * gain)
    }

    /// Radial dead zone for the left stick so small hardware drift does not
    /// walk Pit or pull him off course.
    fn left_stick(x: f32, y: f32) -> (f32, f32) {
        let length = (x * x + y * y).sqrt();
        if length < LEFT_DEAD_ZONE {
            return (0.0, 0.0);
        }
        let t = ((length - LEFT_DEAD_ZONE) / (1.0 - LEFT_DEAD_ZONE)).min(1.0);
        let gain = t / length;
        (x * gain, y * gain)
    }

    /// Radial deadband on SixAxis angular velocity (`[gyro_x, gyro_y, gyro_z]` in
    /// turns/sec) so resting sensor bias does not drift the reticle or camera.
    /// Horizontal yaw combines `-gyro_y` (controller yaw) with `-0.5 * gyro_z`
    /// (controller roll) so split Joy-Con pointer aiming as well as handheld /
    /// Pro Controller turns feel natural at any holding angle; vertical pitch uses `gyro_x`.
    fn gyro_rate(gyro: [f32; 3]) -> (f32, f32) {
        let [gx, gy, gz] = gyro;
        if !gx.is_finite() || !gy.is_finite() || !gz.is_finite() {
            return (0.0, 0.0);
        }
        let yaw_right = -gy - 0.5 * gz;
        let pitch_up = gx;
        let mag = (yaw_right * yaw_right + pitch_up * pitch_up).sqrt();
        if mag < GYRO_DEAD_ZONE {
            return (0.0, 0.0);
        }
        let scale = (mag - GYRO_DEAD_ZONE) / mag;
        (yaw_right * scale, pitch_up * scale)
    }

    /// Updates whether the game is in an active battle with the bottom screen hidden.
    pub fn set_in_battle_play(&mut self, in_battle_play: bool) {
        self.in_battle_play = in_battle_play;
        if !in_battle_play {
            self.fallback_touch = None;
            self.fallback_lift = 0;
        }
    }

    /// Feeds Switch SixAxis gyro and right stick into 3DS `hid:USER` `InputState::tilt`
    /// whenever a title has enabled the 3DS accelerometer or gyroscope.
    pub fn apply_motion_tilt(&mut self, input: &mut InputState, gyro: [f32; 3], host: u32, motion_active: bool) {
        if !motion_active {
            self.motion_tilt = [0.0, 0.0];
            input.tilt = [0.0, 0.0];
            return;
        }
        let recenter_now = input.buttons.contains(PadState::R) || (host & (1 << 1)) != 0;
        if recenter_now {
            self.motion_tilt = [0.0, 0.0];
            input.tilt = [0.0, 0.0];
            return;
        }
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        if self.right_stick {
            let (sx, sy) = Aim::stick(input.c_stick_x, input.c_stick_y);
            dx += sx * MOTION_TILT_STICK_RAD_PER_FRAME * self.speed;
            dy += -sy * MOTION_TILT_STICK_RAD_PER_FRAME * self.speed;
        }
        if self.gyro {
            let (gx, gy) = Aim::gyro_rate(gyro);
            dx += gx * MOTION_TILT_GYRO_RAD_PER_TURN_RATE * self.gyro_speed * self.speed;
            dy += -gy * MOTION_TILT_GYRO_RAD_PER_TURN_RATE * self.gyro_speed * self.speed;
        }
        self.motion_tilt[0] = (self.motion_tilt[0] + dx).clamp(-MOTION_TILT_LIMIT, MOTION_TILT_LIMIT);
        self.motion_tilt[1] = (self.motion_tilt[1] + dy).clamp(-MOTION_TILT_LIMIT, MOTION_TILT_LIMIT);
        input.tilt = self.motion_tilt;
    }

    /// Fallback virtual-stylus aiming when `self.game == 0` during an active battle
    /// (`in_battle_play == true`), such as Chapter 17/21/25 falling reticle QTEs
    /// where neither `kiu_air_aim` nor `kiu_mouse_look` runs.
    fn apply_fallback_touch(&mut self, input: &mut InputState, gyro: [f32; 3], host: u32) {
        self.aiming = false;
        self.vehicle_a_held = false;
        self.vehicle_b_held = false;
        self.vehicle_tap = None;

        if !self.in_battle_play
            || self.touching
            || input.buttons.intersects(PadState::START | PadState::SELECT)
        {
            self.fallback_touch = None;
            self.fallback_lift = 0;
            self.recenter_held = false;
            self.recenter_grace = 0;
            return;
        }

        let recenter_now = input.buttons.contains(PadState::R) || (host & (1 << 1)) != 0;
        input.buttons.remove(PadState::R);
        if recenter_now && !self.recenter_held && self.fallback_touch.is_some() {
            self.fallback_touch = None;
            self.fallback_lift = FALLBACK_RECENTER_LIFT_FRAMES;
        }
        if recenter_now {
            self.recenter_grace = RECENTER_GRACE_FRAMES;
        } else {
            self.recenter_grace = self.recenter_grace.saturating_sub(1);
        }
        self.recenter_held = recenter_now;

        if self.mode == 0 && input.buttons.intersects(PadState::ZL | PadState::ZR) {
            input.buttons.remove(PadState::ZL | PadState::ZR);
            input.buttons.insert(PadState::L);
        }

        let menu_invert_y = self.controls.air_invert_y();
        let invert_y = self.invert_y != menu_invert_y;
        let menu_scale = GuestControls::slider_scale(self.controls.air_speed);
        let menu_gyro_scale = GuestControls::gyro_scale(self.controls.air_stop);

        let mut across = 0.0f32;
        let mut down = 0.0f32;
        let mut stick_t = 0.0f32;
        let mut stick_active = false;

        if self.right_stick {
            let (rx, ry) = (input.c_stick_x, input.c_stick_y);
            let stick_len = (rx * rx + ry * ry).sqrt();
            if stick_len >= DEAD_ZONE {
                stick_t = ((stick_len - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0);
            }
            let (sx, sy) = Aim::stick(rx, ry);
            stick_active = sx != 0.0 || sy != 0.0;
            if stick_active || self.fallback_touch.is_some() {
                input.c_stick_x = 0.0;
                input.c_stick_y = 0.0;
            }
            let dy = if !invert_y { -sy } else { sy };
            across += sx * AIR_PIXELS_PER_FRAME * menu_scale * self.speed;
            down += dy * AIR_PIXELS_PER_FRAME * menu_scale * self.speed;
        }

        let mut gyro_active = false;
        let gyro_paused = recenter_now || self.recenter_grace > 0;
        let stick_damp = 1.0 - stick_t;
        if self.gyro && !gyro_paused && menu_gyro_scale > 0.0 && stick_damp > 0.0 {
            let (gx, gy) = Aim::gyro_rate(gyro);
            let gyro_mag = (gx * gx + gy * gy).sqrt();
            if self.fallback_touch.is_some() || gyro_mag >= FALLBACK_GYRO_ENGAGE_THRESHOLD {
                gyro_active = gyro_mag > 0.0;
                let dy = if !invert_y { -gy } else { gy };
                let g_mult = self.gyro_speed * menu_gyro_scale * self.speed * stick_damp;
                across += gx * AIR_GYRO_PIXELS_PER_TURN_RATE * menu_scale * g_mult;
                down += dy * AIR_GYRO_PIXELS_PER_TURN_RATE * menu_scale * g_mult;
            }
        }

        if self.fallback_lift > 0 {
            self.fallback_lift -= 1;
            return;
        }

        if let Some((vx, vy)) = &mut self.fallback_touch {
            *vx = (*vx + across).clamp(0.0, FALLBACK_TOUCH_MAX_X);
            *vy = (*vy + down).clamp(0.0, FALLBACK_TOUCH_MAX_Y);
            input.touch = Some((vx.round() as u16, vy.round() as u16));
        } else if stick_active || gyro_active {
            self.fallback_touch = Some(FALLBACK_TOUCH_CENTER);
            input.touch = Some((FALLBACK_TOUCH_CENTER.0 as u16, FALLBACK_TOUCH_CENTER.1 as u16));
            log::info!("unx: fallback touch aim engaged (said 0 in battle)");
        }
    }

    /// Processes stick and gyro inputs for the current frame and returns the
    /// `(across, down)` delta in bottom-screen pixels to feed into the host
    /// mailbox (`MOUSE_AIM` while flying, `MOUSE_LOOK` on foot).
    pub fn apply(&mut self, input: &mut InputState, gyro: [f32; 3], host: u32) -> (f32, f32) {
        self.touching = input.touch.is_some();
        (input.circle_x, input.circle_y) = Aim::left_stick(input.circle_x, input.circle_y);

        if self.game == 0 {
            self.apply_fallback_touch(input, gyro, host);
            return (0.0, 0.0);
        }

        if self.fallback_touch.take().is_some() {
            log::info!("unx: fallback touch aim released (game mode {})", self.game);
        }
        self.fallback_lift = 0;

        // In battle:
        // - R (Right Bumper) or Left Stick Click (`UNX_HOST_RECENTER`, bit 1) snaps
        //   the Air Battle reticle to the rail center or levels Land Battle pitch on
        //   press, and pauses gyro while held (+ brief settle grace on release) so
        //   players can ratchet/reposition their hands comfortably back to neutral.
        // - ZL (Nunchuk Z trigger) and ZR (Wii Remote B trigger) act as Fire (`PadState::L`).
        let recenter_now = input.buttons.contains(PadState::R) || (host & (1 << 1)) != 0;
        input.buttons.remove(PadState::R);
        if recenter_now && !self.recenter_held {
            self.recenter_pending = true;
        }
        if recenter_now {
            self.recenter_grace = RECENTER_GRACE_FRAMES;
        } else {
            self.recenter_grace = self.recenter_grace.saturating_sub(1);
        }
        self.recenter_held = recenter_now;

        if self.mode == 0 && input.buttons.intersects(PadState::ZL | PadState::ZR) {
            input.buttons.remove(PadState::ZL | PadState::ZR);
            input.buttons.insert(PadState::L);
        }

        // When mounted in a vehicle (Exo Tank, Cherubot, Aether Ring), the 3DS
        // original uses bottom-screen touchscreen HUD buttons (`f_00A2CBB4` ->
        // `vtable + 0x8` `is_clicked()` in `f_006D76E0`), which fire on touch
        // release. Map pressing `A` to a tap-and-release pulse on the Special
        // Action widget `(28, 212)` (Dash, Jump/Hover, Shield) and pressing `B`
        // to a tap-and-release pulse on the Dismount widget `(86, 218)`.
        if self.game == GAME_VEHICLE {
            let a_now = input.buttons.contains(PadState::A);
            let b_now = input.buttons.contains(PadState::B);
            input.buttons.remove(PadState::A | PadState::B);
            if b_now && !self.vehicle_b_held {
                self.vehicle_tap = Some((VEHICLE_DISMOUNT_TOUCH, VEHICLE_TAP_FRAMES));
            } else if a_now && !self.vehicle_a_held {
                self.vehicle_tap = Some((VEHICLE_SPECIAL_TOUCH, VEHICLE_TAP_FRAMES));
            }
            self.vehicle_a_held = a_now;
            self.vehicle_b_held = b_now;
            if let Some((pos, rem)) = self.vehicle_tap {
                if input.touch.is_none() {
                    input.touch = Some(pos);
                }
                self.vehicle_tap = if rem > 1 { Some((pos, rem - 1)) } else { None };
            }
        } else {
            self.vehicle_a_held = false;
            self.vehicle_b_held = false;
            self.vehicle_tap = None;
        }

        let mut across = 0.0;
        let mut down = 0.0;

        let menu_invert_y = if self.game == GAME_FLYING {
            self.controls.air_invert_y()
        } else {
            self.controls.land_invert_y()
        };
        let invert_y = self.invert_y != menu_invert_y;

        let (menu_scale_x, menu_scale_y, menu_gyro_scale) = if self.game == GAME_FLYING {
            let s = GuestControls::slider_scale(self.controls.air_speed);
            (s, s, GuestControls::gyro_scale(self.controls.air_stop))
        } else {
            (
                GuestControls::slider_scale(self.controls.land_x_speed),
                GuestControls::slider_scale(self.controls.land_y_speed),
                GuestControls::gyro_scale(self.controls.land_stop),
            )
        };

        let mut stick_t = 0.0f32;
        if self.right_stick {
            let (rx, ry) = (input.c_stick_x, input.c_stick_y);
            // Consume c_stick so 3DS ir:USER C-stick emulation does not fight
            // the direct mailbox overrides or add sluggish coasting inertia.
            input.c_stick_x = 0.0;
            input.c_stick_y = 0.0;
            let stick_len = (rx * rx + ry * ry).sqrt();
            if stick_len >= DEAD_ZONE {
                stick_t = ((stick_len - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0);
            }
            let (sx, sy) = Aim::stick(rx, ry);
            // In GAME_FLYING, top-screen y=0 is at the top (so stick up -> down < 0).
            // In GAME_ON_FOOT, kiu_mouse_look adds `down * PITCH_PER_PIXEL` where
            // positive pitch looks up (so stick up -> down > 0).
            let dy = if (self.game == GAME_FLYING) != invert_y { -sy } else { sy };
            let (px_x, px_y) = if self.game == GAME_FLYING {
                (AIR_PIXELS_PER_FRAME, AIR_PIXELS_PER_FRAME)
            } else {
                (LAND_X_PIXELS_PER_FRAME, LAND_Y_PIXELS_PER_FRAME)
            };
            across += sx * px_x * menu_scale_x * self.speed;
            down += dy * px_y * menu_scale_y * self.speed;
        }

        // Pause gyro while R / L3 is held (and briefly after release) for hand ratcheting,
        // and smoothly damp gyro during large right-stick sweeps so wrist wobble does not
        // fight stick-heavy aiming.
        let gyro_paused = recenter_now || self.recenter_grace > 0;
        let stick_damp = 1.0 - stick_t;
        if self.gyro && !gyro_paused && menu_gyro_scale > 0.0 && stick_damp > 0.0 {
            let (gx, gy) = Aim::gyro_rate(gyro);
            let dy = if (self.game == GAME_FLYING) != invert_y { -gy } else { gy };
            let (g_px_x, g_px_y) = if self.game == GAME_FLYING {
                (AIR_GYRO_PIXELS_PER_TURN_RATE, AIR_GYRO_PIXELS_PER_TURN_RATE)
            } else {
                (LAND_GYRO_X_PIXELS_PER_TURN_RATE, LAND_GYRO_Y_PIXELS_PER_TURN_RATE)
            };
            let g_mult = self.gyro_speed * menu_gyro_scale * self.speed * stick_damp;
            across += gx * g_px_x * menu_scale_x * g_mult;
            down += dy * g_px_y * menu_scale_y * g_mult;
        }

        if self.mode == 0 {
            self.aiming = false;
            return (across, down);
        }

        let held = input.buttons.contains(self.modifier);
        if held != self.aiming {
            log::info!(
                "unx: left stick aim {} ({})",
                if held { "on" } else { "off" },
                if self.game == GAME_FLYING { "flying" } else { "on foot" }
            );
            self.aiming = held;
        }
        let (x, y) = (input.circle_x, input.circle_y);
        if self.game == GAME_FLYING {
            if (held || self.mode == 1) && input.touch.is_none() {
                let (sx, sy) = Aim::stick(x, y);
                if sx != 0.0 || sy != 0.0 {
                    input.touch = Some(((160.0 + sx * REACH_X) as u16, (120.0 - sy * REACH_Y) as u16));
                    self.touching = true;
                }
            }
            if held {
                input.buttons.remove(self.modifier);
                input.circle_x = 0.0;
                input.circle_y = 0.0;
            }
            return (across, down);
        }
        if !held {
            return (across, down);
        }
        input.buttons.remove(self.modifier);
        input.circle_x = 0.0;
        input.circle_y = 0.0;
        let (sx, sy) = Aim::stick(x, y);
        let dy = if invert_y { -sy } else { sy };
        (
            across + sx * LAND_X_PIXELS_PER_FRAME * menu_scale_x * self.speed,
            down + dy * LAND_Y_PIXELS_PER_FRAME * menu_scale_y * self.speed,
        )
    }

    /// Fills the host mailbox's first three words (`[mark, across, down]`):
    /// - `MOUSE_AIM` (`"MAM1"`) while flying without touchscreen touch (keeping
    ///   `MOUSE_AIM` set even at zero stick velocity so `0x006D269C` locks the
    ///   reticle against left-stick movement drift), or `MOUSE_AIM_RECENTER` (`"MAMR"`)
    ///   when a recenter is requested.
    /// - `MOUSE_LOOK` (`"MLK1"`) while on foot, or `MOUSE_LOOK_RECENTER` (`"MLKR"`)
    ///   when a recenter is requested.
    pub fn feed(&mut self, mailbox: &mut [u32], (across, down): (f32, f32)) {
        let (mark, recenter_mark) = match self.game {
            GAME_ON_FOOT | GAME_VEHICLE if self.right_stick || self.gyro || self.mode != 0 => {
                (MOUSE_LOOK, MOUSE_LOOK_RECENTER)
            }
            GAME_FLYING if (self.right_stick || self.gyro) && !self.touching => {
                (MOUSE_AIM, MOUSE_AIM_RECENTER)
            }
            _ => {
                self.recenter_pending = false;
                mailbox[..3].fill(0);
                return;
            }
        };
        let waiting_recenter = mailbox[0] == recenter_mark;
        let waiting = waiting_recenter || (mailbox[0] == mark && (mailbox[1] != 0 || mailbox[2] != 0));
        self.untaken = if waiting { self.untaken.saturating_add(1) } else { 0 };
        let keep = waiting && self.untaken <= STALE_FRAMES;
        if self.recenter_pending {
            self.recenter_pending = false;
            self.untaken = 0;
            mailbox[..3].copy_from_slice(&[recenter_mark, 0, 0]);
            return;
        }
        let active_mark = if waiting_recenter && keep { recenter_mark } else { mark };
        let carried = |word: u32| if keep { f32::from_bits(word) } else { 0.0 };
        let (across, down) = (carried(mailbox[1]) + across, carried(mailbox[2]) + down);
        mailbox[..3].copy_from_slice(&[active_mark, across.to_bits(), down.to_bits()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_aim() -> Aim {
        Aim {
            right_stick: true,
            gyro: true,
            gyro_speed: 1.0,
            mode: 0,
            modifier: PadState::ZL,
            invert_y: false,
            speed: 1.0,
            game: 0,
            quiet: 0,
            untaken: 0,
            touching: false,
            aiming: false,
            controls: GuestControls::default(),
            msbt_offset: None,
            recenter_held: false,
            recenter_pending: false,
            recenter_grace: 0,
            vehicle_a_held: false,
            vehicle_b_held: false,
            vehicle_tap: None,
            in_battle_play: false,
            fallback_touch: None,
            fallback_lift: 0,
            motion_tilt: [0.0, 0.0],
        }
    }

    #[test]
    fn flying_right_stick_feeds_mouse_aim_without_left_stick_interference() {
        let mut aim = test_aim();
        aim.said(GAME_FLYING);
        let mut input = InputState {
            circle_x: 1.0,
            circle_y: -0.5,
            c_stick_x: 1.0,
            c_stick_y: 1.0,
            ..InputState::default()
        };
        let (across, down) = aim.apply(&mut input, [0.0; 3], 0);
        assert_eq!(input.touch, None, "left stick must not inject a fake stylus touch");
        assert!(input.circle_x > 0.8 && input.circle_y < -0.4, "left stick still steers Pit");
        assert_eq!((input.c_stick_x, input.c_stick_y), (0.0, 0.0), "c_stick is consumed");
        assert!(across > 0.0 && down < 0.0, "right + up moves reticle right and up");

        let mut mailbox = [0u32; 4];
        aim.feed(&mut mailbox, (across, down));
        assert_eq!(mailbox[0], MOUSE_AIM);
        assert_eq!(f32::from_bits(mailbox[1]), across);
        assert_eq!(f32::from_bits(mailbox[2]), down);
    }

    #[test]
    fn flying_at_rest_keeps_mouse_aim_mark_to_prevent_reticle_drift() {
        let mut aim = test_aim();
        aim.said(GAME_FLYING);
        let mut input = InputState {
            circle_x: -1.0,
            ..InputState::default()
        };
        let delta = aim.apply(&mut input, [0.0; 3], 0);
        assert_eq!(delta, (0.0, 0.0));
        let mut mailbox = [0u32; 4];
        aim.feed(&mut mailbox, delta);
        assert_eq!(mailbox[..3], [MOUSE_AIM, 0, 0]);
    }

    #[test]
    fn on_foot_right_stick_feeds_mouse_look() {
        let mut aim = test_aim();
        aim.said(GAME_ON_FOOT);
        let mut input = InputState {
            c_stick_x: -1.0,
            c_stick_y: 0.0,
            ..InputState::default()
        };
        let (across, down) = aim.apply(&mut input, [0.0; 3], 0);
        assert_eq!((across, down), (-LAND_X_PIXELS_PER_FRAME, 0.0));
        assert_eq!(input.c_stick_x, 0.0);
        let mut mailbox = [0u32; 4];
        aim.feed(&mut mailbox, (across, down));
        assert_eq!(mailbox[..3], [MOUSE_LOOK, (-LAND_X_PIXELS_PER_FRAME).to_bits(), 0]);

        let mut input_up = InputState {
            c_stick_x: 0.0,
            c_stick_y: 1.0,
            ..InputState::default()
        };
        let (across_up, down_up) = aim.apply(&mut input_up, [0.0; 3], 0);
        assert_eq!(across_up, 0.0);
        assert_eq!(down_up, LAND_Y_PIXELS_PER_FRAME, "pushing right stick up on foot must pitch camera up (down > 0)");
    }

    #[test]
    fn gyro_and_guest_controls_sliders_scale_and_invert_aim() {
        let mut aim = test_aim();
        aim.said(GAME_FLYING);
        aim.set_controls(GuestControls {
            initialized: true,
            air_speed: 1.0,   // 2.0x reticle speed
            air_stop: 0.5,    // 1.5x gyro sensitivity
            flags_173: 0x00,  // Air Up/Down inverted (bit 1 == 0)
            ..GuestControls::default()
        });
        let mut input = InputState::default();
        // Pitch nose up (+0.1 turns/s) and yaw right (gy = -0.1 turns/s)
        let (across, down) = aim.apply(&mut input, [0.1, -0.1, 0.0], 0);
        assert!(across > 10.0, "gyro yaw right moves reticle right, scaled by air_speed * air_stop");
        assert!(down > 10.0, "inverted air_invert_y makes gyro pitch up move reticle down (down > 0)");
    }

    #[test]
    fn split_joycon_triggers_fire_and_r_recenters_reticle() {
        let mut aim = test_aim();
        aim.said(GAME_FLYING);
        let mut input = InputState {
            buttons: PadState::ZR | PadState::R,
            ..InputState::default()
        };
        let delta = aim.apply(&mut input, [0.0; 3], 0);
        assert!(input.buttons.contains(PadState::L), "ZR trigger maps to 3DS L (Fire)");
        assert!(!input.buttons.contains(PadState::ZR));
        assert!(!input.buttons.contains(PadState::R), "R is consumed as Recenter in battle");
        let mut mailbox = [0u32; 4];
        aim.feed(&mut mailbox, delta);
        assert_eq!(mailbox[0], MOUSE_AIM_RECENTER);
    }

    #[test]
    fn vehicle_a_and_b_buttons_tap_and_release_special_and_dismount_widgets() {
        let mut aim = test_aim();
        aim.said(GAME_VEHICLE);

        // Holding A across 4 frames taps (28, 212) for VEHICLE_TAP_FRAMES frames and then releases
        for frame in 0..=VEHICLE_TAP_FRAMES {
            let mut input = InputState {
                buttons: PadState::A,
                c_stick_x: 1.0,
                ..InputState::default()
            };
            let (across, down) = aim.apply(&mut input, [0.0; 3], 0);
            assert!(!input.buttons.contains(PadState::A), "A button is consumed in vehicle");
            if frame < VEHICLE_TAP_FRAMES {
                assert_eq!(input.touch, Some(VEHICLE_SPECIAL_TOUCH));
            } else {
                assert_eq!(input.touch, None, "touch automatically releases even while A remains held");
            }
            assert_eq!((across, down), (LAND_X_PIXELS_PER_FRAME, 0.0));
            let mut mailbox = [0u32; 4];
            aim.feed(&mut mailbox, (across, down));
            assert_eq!(mailbox[..3], [MOUSE_LOOK, LAND_X_PIXELS_PER_FRAME.to_bits(), 0]);
        }

        // Pressing B taps and releases the Dismount widget (86, 218)
        for frame in 0..=VEHICLE_TAP_FRAMES {
            let mut input = InputState {
                buttons: PadState::B,
                ..InputState::default()
            };
            aim.apply(&mut input, [0.0; 3], 0);
            assert!(!input.buttons.contains(PadState::B), "B button is consumed in vehicle");
            if frame < VEHICLE_TAP_FRAMES {
                assert_eq!(input.touch, Some(VEHICLE_DISMOUNT_TOUCH));
            } else {
                assert_eq!(input.touch, None, "dismount touch automatically releases while B remains held");
            }
        }
    }

    #[test]
    fn holding_recenter_pauses_gyro_and_settles_after_release() {
        let mut aim = test_aim();
        aim.said(GAME_FLYING);
        let motion = [0.1, -0.1, 0.0];

        // While R is held, recenter fires once and gyro stays muted
        let mut input = InputState { buttons: PadState::R, ..InputState::default() };
        assert_eq!(aim.apply(&mut input, motion, 0), (0.0, 0.0));
        let mut mailbox = [0u32; 4];
        aim.feed(&mut mailbox, (0.0, 0.0));
        assert_eq!(mailbox[0], MOUSE_AIM_RECENTER);

        // Continuing to hold R still mutes gyro without re-triggering recenter
        let mut input_held = InputState { buttons: PadState::R, ..InputState::default() };
        assert_eq!(aim.apply(&mut input_held, motion, 0), (0.0, 0.0));

        // Release R: muted for RECENTER_GRACE_FRAMES - 1 settle frames, then resumes
        for _ in 0..(RECENTER_GRACE_FRAMES - 1) {
            let mut released = InputState::default();
            assert_eq!(aim.apply(&mut released, motion, 0), (0.0, 0.0));
        }
        let mut resumed = InputState::default();
        let (across, down) = aim.apply(&mut resumed, motion, 0);
        assert!(across > 0.0 && down < 0.0, "gyro resumes smoothly after settle grace");
    }

    #[test]
    fn right_stick_sweep_damps_gyro_wobble() {
        let mut aim = test_aim();
        aim.said(GAME_ON_FOOT);
        // Full right-stick horizontal sweep with vertical gyro wobble should not add pitch wobble
        let mut input = InputState { c_stick_x: 1.0, c_stick_y: 0.0, ..InputState::default() };
        let (across, down) = aim.apply(&mut input, [0.2, 0.0, 0.0], 0);
        assert_eq!((across, down), (LAND_X_PIXELS_PER_FRAME, 0.0));
    }

    #[test]
    fn fallback_touch_engages_in_battle_when_overrides_quiet_and_respects_pause_and_recenter() {
        let mut aim = test_aim();
        aim.set_in_battle_play(true);

        // Idle and subtle resting gyro noise must not inject a phantom touch on cutscenes/pause
        let mut idle = InputState::default();
        assert_eq!(aim.apply(&mut idle, [0.005, -0.005, 0.0], 0), (0.0, 0.0));
        assert_eq!(idle.touch, None);

        // Right stick engages fallback stylus at screen center, consumes c-stick, and maps ZR -> L
        let mut engage = InputState {
            buttons: PadState::ZR,
            c_stick_x: 1.0,
            c_stick_y: 0.0,
            ..InputState::default()
        };
        assert_eq!(aim.apply(&mut engage, [0.0; 3], 0), (0.0, 0.0));
        assert_eq!(engage.touch, Some((160, 120)));
        assert_eq!((engage.c_stick_x, engage.c_stick_y), (0.0, 0.0));
        assert!(engage.buttons.contains(PadState::L));
        assert!(!engage.buttons.contains(PadState::ZR));

        // Subsequent frames move the virtual stylus right and up via right stick + fine gyro
        let mut move_stick = InputState { c_stick_x: 1.0, c_stick_y: 1.0, ..InputState::default() };
        aim.apply(&mut move_stick, [0.0; 3], 0);
        let (x1, y1) = move_stick.touch.expect("virtual stylus stays down");
        assert!(x1 > 160 && y1 < 120, "right + up moves virtual stylus toward top-right: ({x1}, {y1})");

        let mut move_gyro = InputState::default();
        aim.apply(&mut move_gyro, [0.02, -0.02, 0.0], 0);
        let (x2, y2) = move_gyro.touch.expect("fine gyro tracks once engaged");
        assert!(x2 > x1 && y2 < y1, "fine gyro moves stylus right/up: ({x2}, {y2}) vs ({x1}, {y1})");

        // Pressing R lifts the virtual stylus for FALLBACK_RECENTER_LIFT_FRAMES
        let mut recenter = InputState { buttons: PadState::R, c_stick_x: 1.0, ..InputState::default() };
        aim.apply(&mut recenter, [0.0; 3], 0);
        assert_eq!(recenter.touch, None);
        assert!(!recenter.buttons.contains(PadState::R));

        // Remaining lift frame keeps touch up, then next stick move re-anchors at center
        for _ in 1..FALLBACK_RECENTER_LIFT_FRAMES {
            let mut lifting = InputState { c_stick_x: 1.0, ..InputState::default() };
            aim.apply(&mut lifting, [0.0; 3], 0);
            assert_eq!(lifting.touch, None);
        }
        let mut reanchor = InputState { c_stick_x: 1.0, ..InputState::default() };
        aim.apply(&mut reanchor, [0.0; 3], 0);
        assert_eq!(reanchor.touch, Some((160, 120)));

        // Pressing START immediately lifts the virtual stylus so pause menus stay clean
        let mut pause = InputState { buttons: PadState::START, ..InputState::default() };
        aim.apply(&mut pause, [0.0; 3], 0);
        assert_eq!(pause.touch, None);

        // Resuming normal GAME_FLYING releases fallback touch cleanly
        let mut engage2 = InputState { c_stick_x: 1.0, ..InputState::default() };
        aim.apply(&mut engage2, [0.0; 3], 0);
        assert_eq!(engage2.touch, Some((160, 120)));
        aim.said(GAME_FLYING);
        let mut flying = InputState::default();
        aim.apply(&mut flying, [0.0; 3], 0);
        assert_eq!(flying.touch, None);
    }

    #[test]
    fn motion_tilt_integrates_stick_and_gyro_and_recenters() {
        let mut aim = test_aim();
        let mut input = InputState { c_stick_x: 1.0, c_stick_y: 0.5, ..InputState::default() };
        aim.apply_motion_tilt(&mut input, [0.1, -0.1, 0.0], 0, false);
        assert_eq!(input.tilt, [0.0, 0.0]);

        aim.apply_motion_tilt(&mut input, [0.1, -0.1, 0.0], 0, true);
        assert!(input.tilt[0] > 0.0 && input.tilt[1] < 0.0, "right + up tilts right and up: {:?}", input.tilt);

        let mut recenter = InputState { buttons: PadState::R, ..InputState::default() };
        aim.apply_motion_tilt(&mut recenter, [0.1, -0.1, 0.0], 0, true);
        assert_eq!(recenter.tilt, [0.0, 0.0]);
    }
}
