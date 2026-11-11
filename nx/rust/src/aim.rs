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
/// Host mailbox mark for flying reticle aim (`"MAM1"`), followed by `across`
/// and `down` as `f32` bottom-screen pixels (`zakuro::input::MOUSE_AIM`).
const MOUSE_AIM: u32 = u32::from_le_bytes(*b"MAM1");
/// Host mailbox fourth word written by the game's overrides each tick.
pub const GAME_ON_FOOT: u32 = 1;
pub const GAME_FLYING: u32 = 2;

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
/// Stylus reach from the center of the 320x240 bottom screen for legacy
/// left-stick touch mode.
const REACH_X: f32 = 150.0;
const REACH_Y: f32 = 112.0;

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
    (0x119B6, 26, "R-Stick & Gyro Aim active."),
    (0x11A3A, 36, "Adjusts vertical Stick & Gyro speed."),
    (0x11A84, 38, "Adjusts horizontal Stick & Gyro speed."),
    (0x11AD2, 51, "Adjusts Right-Stick and Gyro Aiming reticle speed."),
    (
        0x11B3A,
        77,
        "This adjusts SixAxis Gyro Aiming sensitivity.\nSlide all the way left for Off.",
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
                if !frame.is_multiple_of(4) {
                    return;
                }
                let search_end = fcram.len().min(0x0400_0000).saturating_sub(MSBT_MENU_SIZE as usize);
                let mut found = None;
                let mut off = 0;
                while off <= search_end {
                    if fcram[off] == b'M' && is_menu_msbt(fcram, off) {
                        found = Some(off);
                        break;
                    }
                    off += 0x80;
                }
                let Some(off) = found else { return };
                self.msbt_offset = Some(off);
                off
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
    /// Horizontal yaw combines `-gyro_y` (controller yaw) with `-0.35 * gyro_z`
    /// (controller roll) so handheld/tabletop/pro-controller turns feel natural at
    /// any holding angle; vertical pitch uses `gyro_x`.
    fn gyro_rate(gyro: [f32; 3]) -> (f32, f32) {
        let [gx, gy, gz] = gyro;
        if !gx.is_finite() || !gy.is_finite() || !gz.is_finite() {
            return (0.0, 0.0);
        }
        let yaw_right = -gy - 0.35 * gz;
        let pitch_up = gx;
        let mag = (yaw_right * yaw_right + pitch_up * pitch_up).sqrt();
        if mag < GYRO_DEAD_ZONE {
            return (0.0, 0.0);
        }
        let scale = (mag - GYRO_DEAD_ZONE) / mag;
        (yaw_right * scale, pitch_up * scale)
    }

    /// Processes stick and gyro inputs for the current frame and returns the
    /// `(across, down)` delta in bottom-screen pixels to feed into the host
    /// mailbox (`MOUSE_AIM` while flying, `MOUSE_LOOK` on foot).
    pub fn apply(&mut self, input: &mut InputState, gyro: [f32; 3]) -> (f32, f32) {
        self.touching = input.touch.is_some();
        (input.circle_x, input.circle_y) = Aim::left_stick(input.circle_x, input.circle_y);

        if self.game == 0 {
            self.aiming = false;
            return (0.0, 0.0);
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

        if self.right_stick {
            let (rx, ry) = (input.c_stick_x, input.c_stick_y);
            // Consume c_stick so 3DS ir:USER C-stick emulation does not fight
            // the direct mailbox overrides or add sluggish coasting inertia.
            input.c_stick_x = 0.0;
            input.c_stick_y = 0.0;
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

        if self.gyro && menu_gyro_scale > 0.0 {
            let (gx, gy) = Aim::gyro_rate(gyro);
            let dy = if (self.game == GAME_FLYING) != invert_y { -gy } else { gy };
            let (g_px_x, g_px_y) = if self.game == GAME_FLYING {
                (AIR_GYRO_PIXELS_PER_TURN_RATE, AIR_GYRO_PIXELS_PER_TURN_RATE)
            } else {
                (LAND_GYRO_X_PIXELS_PER_TURN_RATE, LAND_GYRO_Y_PIXELS_PER_TURN_RATE)
            };
            let g_mult = self.gyro_speed * menu_gyro_scale * self.speed;
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
    ///   reticle against left-stick movement drift).
    /// - `MOUSE_LOOK` (`"MLK1"`) while on foot.
    pub fn feed(&mut self, mailbox: &mut [u32], (across, down): (f32, f32)) {
        let mark = match self.game {
            GAME_ON_FOOT if self.right_stick || self.gyro || self.mode != 0 => MOUSE_LOOK,
            GAME_FLYING if (self.right_stick || self.gyro) && !self.touching => MOUSE_AIM,
            _ => {
                mailbox[..3].fill(0);
                return;
            }
        };
        let waiting = mailbox[0] == mark && (mailbox[1] != 0 || mailbox[2] != 0);
        self.untaken = if waiting { self.untaken.saturating_add(1) } else { 0 };
        let keep = waiting && self.untaken <= STALE_FRAMES;
        let carried = |word: u32| if keep { f32::from_bits(word) } else { 0.0 };
        let (across, down) = (carried(mailbox[1]) + across, carried(mailbox[2]) + down);
        mailbox[..3].copy_from_slice(&[mark, across.to_bits(), down.to_bits()]);
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
        let (across, down) = aim.apply(&mut input, [0.0; 3]);
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
        let delta = aim.apply(&mut input, [0.0; 3]);
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
        let (across, down) = aim.apply(&mut input, [0.0; 3]);
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
        let (across_up, down_up) = aim.apply(&mut input_up, [0.0; 3]);
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
        let (across, down) = aim.apply(&mut input, [0.1, -0.1, 0.0]);
        assert!(across > 10.0, "gyro yaw right moves reticle right, scaled by air_speed * air_stop");
        assert!(down > 10.0, "inverted air_invert_y makes gyro pitch up move reticle down (down > 0)");
    }
}
