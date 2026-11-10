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
//! - `UNX_LEFT_STICK_AIM` (default `0`):
//!   `0` off, the left stick only steers/walks Pit;
//!   `1` flying: the left stick also places a synthetic stylus touch, and on
//!       foot holding the modifier turns the camera with the left stick;
//!   `2` as `1`, but flying only aims with the left stick while the modifier is held.
//! - `UNX_AIM_MODIFIER`: `ZL` (default), `ZR`, `L`, `R`, `X`, `Y` or `A`.
//! - `UNX_AIM_INVERT_Y=1`: flips vertical aim/camera direction.
//! - `UNX_AIM_SPEED`: scales aiming and camera turn speed (default `1.0`).

use zakuro_core::services::hid::{InputState, PadState};

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
/// Flying reticle speed in bottom-screen pixels per 60 Hz frame at full stick
/// deflection (`6.0 * 1.25 = 7.5` top-screen px/frame = `450` px/s across 400x240).
const AIR_PIXELS_PER_FRAME: f32 = 6.0;
/// On-foot camera turn speed in bottom-screen pixels per 60 Hz frame at full
/// stick deflection (`10.0 * 0.00408 rad = 140 deg/s` yaw).
const LAND_X_PIXELS_PER_FRAME: f32 = 10.0;
const LAND_Y_PIXELS_PER_FRAME: f32 = 7.0;
/// Frames the game may leave the mailbox movement unread before it is dropped.
const STALE_FRAMES: u8 = 4;
/// Frames without a word from the game's overrides before it is out of play.
const QUIET_FRAMES: u8 = 10;
/// Stylus reach from the center of the 320x240 bottom screen for legacy
/// left-stick touch mode.
const REACH_X: f32 = 150.0;
const REACH_Y: f32 = 112.0;

pub struct Aim {
    right_stick: bool,
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
            mode: var("UNX_LEFT_STICK_AIM").and_then(|v| v.trim().parse().ok()).unwrap_or(0),
            modifier: modifier(var("UNX_AIM_MODIFIER").as_deref().unwrap_or("ZL")),
            invert_y: var("UNX_AIM_INVERT_Y").is_some_and(|v| v.trim() == "1"),
            speed: var("UNX_AIM_SPEED").and_then(|v| v.trim().parse().ok()).unwrap_or(1.0),
            game: 0,
            quiet: 0,
            untaken: 0,
            touching: false,
            aiming: false,
        };
        log::info!(
            "unx: aim right_stick {} left_stick_mode {} (modifier {:?}, invert y {}, speed {})",
            aim.right_stick, aim.mode, aim.modifier, aim.invert_y, aim.speed
        );
        aim
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

    /// Processes stick inputs for the current frame and returns the `(across, down)`
    /// delta in bottom-screen pixels to feed into the host mailbox (`MOUSE_AIM`
    /// while flying, `MOUSE_LOOK` on foot).
    pub fn apply(&mut self, input: &mut InputState) -> (f32, f32) {
        self.touching = input.touch.is_some();
        (input.circle_x, input.circle_y) = Aim::left_stick(input.circle_x, input.circle_y);

        if self.game == 0 {
            self.aiming = false;
            return (0.0, 0.0);
        }

        let mut across = 0.0;
        let mut down = 0.0;

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
            let dy = if (self.game == GAME_FLYING) != self.invert_y { -sy } else { sy };
            let (px_x, px_y) = if self.game == GAME_FLYING {
                (AIR_PIXELS_PER_FRAME, AIR_PIXELS_PER_FRAME)
            } else {
                (LAND_X_PIXELS_PER_FRAME, LAND_Y_PIXELS_PER_FRAME)
            };
            across += sx * px_x * self.speed;
            down += dy * px_y * self.speed;
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
        let dy = if self.invert_y { -sy } else { sy };
        (
            across + sx * LAND_X_PIXELS_PER_FRAME * self.speed,
            down + dy * LAND_Y_PIXELS_PER_FRAME * self.speed,
        )
    }

    /// Fills the host mailbox's first three words (`[mark, across, down]`):
    /// - `MOUSE_AIM` (`"MAM1"`) while flying without touchscreen touch (keeping
    ///   `MOUSE_AIM` set even at zero stick velocity so `0x006D269C` locks the
    ///   reticle against left-stick movement drift).
    /// - `MOUSE_LOOK` (`"MLK1"`) while on foot.
    pub fn feed(&mut self, mailbox: &mut [u32], (across, down): (f32, f32)) {
        let mark = match self.game {
            GAME_ON_FOOT if self.right_stick || self.mode != 0 => MOUSE_LOOK,
            GAME_FLYING if self.right_stick && !self.touching => MOUSE_AIM,
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
            mode: 0,
            modifier: PadState::ZL,
            invert_y: false,
            speed: 1.0,
            game: 0,
            quiet: 0,
            untaken: 0,
            touching: false,
            aiming: false,
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
        let (across, down) = aim.apply(&mut input);
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
        let delta = aim.apply(&mut input);
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
        let (across, down) = aim.apply(&mut input);
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
        let (across_up, down_up) = aim.apply(&mut input_up);
        assert_eq!(across_up, 0.0);
        assert_eq!(down_up, LAND_Y_PIXELS_PER_FRAME, "pushing right stick up on foot must pitch camera up (down > 0)");
    }
}
