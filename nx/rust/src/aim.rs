//! aiming with the left stick.
//!
//! How the game aims, from its overrides (kiu-pc's, run with the USA addresses):
//! flying, the circle pad moves Pit and the reticle follows the stylus on the
//! bottom screen (bottom 320x240 pixels onto the top 400x240), drifting with
//! Pit when there is no stylus; on foot, the circle pad walks, a stylus drag
//! turns the camera and the stylus moves the reticle. The C-stick (the
//! Switch's right stick, through ir:rst) turns the camera on foot.
//!
//! UNX_LEFT_STICK_AIM (read once, default set in nx/c/source/main.c to 1):
//!   0  off, the left stick only moves as before;
//!   1  flying: the left stick still steers Pit and also puts a stylus on the
//!      screen where the stick points, so the reticle goes there (the touch
//!      is only made while the stick is pushed and the real screen is not
//!      touched). On foot: while the modifier button is held the left stick
//!      stops walking and turns the camera instead, through the host mailbox
//!      the game's mouse-look override reads, as the desktop frontend's mouse
//!      does. Flying, holding the modifier freezes Pit, so the stick only aims.
//!   2  as 1, but flying the stick aims only while the modifier is held (Pit
//!      steers as the circle pad always did).
//! UNX_AIM_MODIFIER: ZL (default), ZR, L, R, X, Y or A; the game does not see
//! that button while aiming. UNX_AIM_INVERT_Y=1 flips the on-foot camera's
//! vertical direction. UNX_AIM_SPEED scales the on-foot turn (default 1.0).

use zakuro_core::services::hid::{InputState, PadState};

/// the host mailbox's first word for mouse look on foot, "MLK1", then across
/// and down as f32 bottom screen pixels (zakuro::input::MOUSE_LOOK).
const MOUSE_LOOK: u32 = u32::from_le_bytes(*b"MLK1");
/// the mailbox's fourth word, as the game's overrides say it each tick
pub const GAME_ON_FOOT: u32 = 1;
pub const GAME_FLYING: u32 = 2;

/// the stick's dead zone, and pixels a frame at full deflection on foot
/// (the desktop's quick-sweep knee is 8).
const DEAD_ZONE: f32 = 0.2;
const PIXELS_PER_FRAME: f32 = 9.0;
/// frames the game may leave the look unread before it is dropped
const STALE_FRAMES: u8 = 4;
/// frames without a word from the game's overrides before it is out of play
const QUIET_FRAMES: u8 = 10;
/// the stylus's reach from the middle of the 320x240 bottom screen when it
/// stands for the stick, a little short of the edges so it never lifts there
const REACH_X: f32 = 150.0;
const REACH_Y: f32 = 112.0;

pub struct Aim {
    mode: u32,
    modifier: PadState,
    invert_y: bool,
    speed: f32,
    /// what the overrides last said (GAME_ON_FOOT or GAME_FLYING), 0 for nothing
    game: u32,
    quiet: u8,
    untaken: u8,
    /// whether the modifier was held last frame, for the log
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
            mode: var("UNX_LEFT_STICK_AIM").and_then(|v| v.trim().parse().ok()).unwrap_or(1),
            modifier: modifier(var("UNX_AIM_MODIFIER").as_deref().unwrap_or("ZL")),
            invert_y: var("UNX_AIM_INVERT_Y").is_some_and(|v| v.trim() == "1"),
            speed: var("UNX_AIM_SPEED").and_then(|v| v.trim().parse().ok()).unwrap_or(1.0),
            game: 0,
            quiet: 0,
            untaken: 0,
            aiming: false,
        };
        log::info!(
            "unx: left stick aim mode {} (modifier {:?}, invert y {}, speed {})",
            aim.mode, aim.modifier, aim.invert_y, aim.speed
        );
        aim
    }

    /// what the game's overrides said this frame (0 for nothing).
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

    /// the stick after the dead zone, easing in so small pushes are fine.
    fn stick(x: f32, y: f32) -> (f32, f32) {
        let length = (x * x + y * y).sqrt();
        if length < DEAD_ZONE {
            return (0.0, 0.0);
        }
        let t = ((length - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0);
        let gain = t * t / length;
        (x * gain, y * gain)
    }

    /// changes the input the game gets, and returns the camera's turn this
    /// frame in bottom screen pixels (across, down) to leave in the mailbox.
    pub fn apply(&mut self, input: &mut InputState) -> (f32, f32) {
        if self.mode == 0 || self.game == 0 {
            self.aiming = false;
            return (0.0, 0.0);
        }
        let held = input.buttons.contains(self.modifier);
        if held != self.aiming {
            log::info!("unx: left stick aim {} ({})", if held { "on" } else { "off" }, if self.game == GAME_FLYING { "flying" } else { "on foot" });
            self.aiming = held;
        }
        let (x, y) = (input.circle_x, input.circle_y);
        if self.game == GAME_FLYING {
            if (held || self.mode == 1) && input.touch.is_none() {
                let (sx, sy) = Aim::stick(x, y);
                if sx != 0.0 || sy != 0.0 {
                    input.touch = Some(((160.0 + sx * REACH_X) as u16, (120.0 - sy * REACH_Y) as u16));
                }
            }
            if held {
                input.buttons.remove(self.modifier);
                input.circle_x = 0.0;
                input.circle_y = 0.0;
            }
            return (0.0, 0.0);
        }
        if !held {
            return (0.0, 0.0);
        }
        input.buttons.remove(self.modifier);
        input.circle_x = 0.0;
        input.circle_y = 0.0;
        let (sx, sy) = Aim::stick(x, y);
        let down = if self.invert_y { sy } else { -sy };
        (sx * PIXELS_PER_FRAME * self.speed, down * PIXELS_PER_FRAME * self.speed)
    }

    /// fills the mailbox's first three words as zakuro's desktop frontend does:
    /// the mark and the look the game has not yet taken, plus this frame's.
    pub fn feed(&mut self, mailbox: &mut [u32], (across, down): (f32, f32)) {
        if self.mode == 0 || self.game != GAME_ON_FOOT {
            mailbox[..3].fill(0);
            return;
        }
        let waiting = mailbox[0] == MOUSE_LOOK && (mailbox[1] != 0 || mailbox[2] != 0);
        self.untaken = if waiting { self.untaken.saturating_add(1) } else { 0 };
        let keep = waiting && self.untaken <= STALE_FRAMES;
        let carried = |word: u32| if keep { f32::from_bits(word) } else { 0.0 };
        let (across, down) = (carried(mailbox[1]) + across, carried(mailbox[2]) + down);
        mailbox[..3].copy_from_slice(&[MOUSE_LOOK, across.to_bits(), down.to_bits()]);
    }
}
