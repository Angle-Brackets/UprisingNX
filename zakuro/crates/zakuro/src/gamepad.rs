//! game controllers, any the system knows, read through gilrs.

use gilrs::{Axis, Button, EventType, Gilrs};
use zakuro_core::services::hid::{InputState, PadState};

use crate::settings::PadButtons;

/// how far a stick has to lean before it counts, some rest a little off
/// center.
const DEADZONE: f32 = 0.15;

/// KIU right-stick aiming curve & speeds (matching Switch `uprising_nx::aim`).
const AIM_DEADZONE: f32 = 0.10;
const AIM_LINEAR_GAIN: f32 = 0.35;
const AIR_PX_PER_FRAME: f32 = 6.0;
const LAND_X_PX_PER_FRAME: f32 = 10.0;
const LAND_Y_PX_PER_FRAME: f32 = 7.0;
const STALE_FRAMES: u8 = 4;
const QUIET_FRAMES: u8 = 10;

const MOUSE_LOOK: u32 = u32::from_le_bytes(*b"MLK1");
const MOUSE_LOOK_RECENTER: u32 = u32::from_le_bytes(*b"MLKR");
const MOUSE_LOOK_HOLD: u32 = u32::from_le_bytes(*b"MLKH");
const MOUSE_LOOK_CENTER: u32 = u32::from_le_bytes(*b"MLKC");
const MOUSE_RETICLE_CENTER: u32 = u32::from_le_bytes(*b"MLRC");
const MOUSE_AIM: u32 = u32::from_le_bytes(*b"MAM1");
const MOUSE_AIM_RECENTER: u32 = u32::from_le_bytes(*b"MAMR");

const VEHICLE_SPECIAL_TOUCH: (u16, u16) = (28, 212);
const VEHICLE_DISMOUNT_TOUCH: (u16, u16) = (86, 218);
const VEHICLE_TAP_FRAMES: u8 = 3;

fn stick_curve(x: f32, y: f32) -> (f32, f32) {
    let mag = x.hypot(y);
    if mag <= AIM_DEADZONE {
        return (0.0, 0.0);
    }
    let norm = ((mag - AIM_DEADZONE) / (1.0 - AIM_DEADZONE)).clamp(0.0, 1.0);
    let curved = AIM_LINEAR_GAIN * norm + (1.0 - AIM_LINEAR_GAIN) * norm * norm;
    let scale = curved / mag;
    (x * scale, y * scale)
}

pub struct Gamepads {
    /// none when the system could not be asked for controllers.
    gilrs: Option<Gilrs>,
    buttons: PadButtons,
    /// whether the player last aimed/played with a controller rather than KBM.
    active: bool,
    game: u32,
    quiet: u8,
    untaken: u8,
    recenter_held: bool,
    center_held: bool,
    reticle_center_held: bool,
    recenter_pending: bool,
    center_pending: bool,
    reticle_center_pending: bool,
    vehicle_a_held: bool,
    vehicle_b_held: bool,
    vehicle_tap: Option<((u16, u16), u8)>,
}

impl Gamepads {
    pub fn new(buttons: PadButtons) -> Gamepads {
        let gilrs = Gilrs::new()
            .inspect_err(|error| log::warn!("controllers are not available, {error}"))
            .ok();
        Gamepads {
            gilrs,
            buttons,
            active: false,
            game: 0,
            quiet: 0,
            untaken: 0,
            recenter_held: false,
            center_held: false,
            reticle_center_held: false,
            recenter_pending: false,
            center_pending: false,
            reticle_center_pending: false,
            vehicle_a_held: false,
            vehicle_b_held: false,
            vehicle_tap: None,
        }
    }

    pub fn set_buttons(&mut self, buttons: PadButtons) {
        self.buttons = buttons;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn set_inactive(&mut self) {
        self.active = false;
    }

    pub fn clear_kiu_game(&mut self) {
        self.game = 0;
        self.quiet = 0;
        self.recenter_pending = false;
        self.center_pending = false;
        self.reticle_center_pending = false;
        self.vehicle_tap = None;
    }

    /// takes in what the controllers did since the last call, and returns the
    /// buttons pressed meanwhile.
    pub fn poll(&mut self) -> Vec<Button> {
        let mut pressed = Vec::new();
        let Some(gilrs) = &mut self.gilrs else { return pressed };
        while let Some(event) = gilrs.next_event() {
            match event.event {
                EventType::ButtonPressed(button, _) => {
                    self.active = true;
                    pressed.push(button);
                }
                EventType::Connected => {
                    log::info!("controller connected, {}", gilrs.gamepad(event.id).name());
                }
                EventType::Disconnected => log::info!("controller disconnected"),
                _ => {}
            }
        }
        pressed
    }

    /// whether a controller holds button down.
    pub fn holding(&self, button: Button) -> bool {
        self.gilrs.as_ref().is_some_and(|gilrs| gilrs.gamepads().any(|(_, pad)| pad.is_pressed(button)))
    }

    /// raw right-stick axes `(x, y)` (up positive) from the first deflected controller.
    pub fn right_stick(&self) -> (f32, f32) {
        let Some(gilrs) = &self.gilrs else { return (0.0, 0.0) };
        for (_, pad) in gilrs.gamepads() {
            let (x, y) = (pad.value(Axis::RightStickX), pad.value(Axis::RightStickY));
            if x.hypot(y) > AIM_DEADZONE {
                return (x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0));
            }
        }
        (0.0, 0.0)
    }

    /// Drives KIU's `MLK1`/`MAM1` mailbox and Z-targeting from a connected controller.
    /// Returns `true` when the controller owns the mailbox this frame.
    pub fn feed_kiu_mailbox(
        &mut self,
        said: u32,
        invert_y: bool,
        input: &mut InputState,
        mailbox: &mut [u32],
    ) -> bool {
        if said != 0 {
            self.game = said;
            self.quiet = 0;
        } else if self.game != 0 {
            self.quiet += 1;
            if self.quiet > QUIET_FRAMES {
                self.game = 0;
                self.quiet = 0;
            }
        }

        let (rx, ry) = self.right_stick();
        if rx != 0.0 || ry != 0.0 {
            self.active = true;
        }
        if !self.active || self.game == 0 {
            self.recenter_held = false;
            self.center_held = false;
            self.reticle_center_held = false;
            self.vehicle_a_held = false;
            self.vehicle_b_held = false;
            self.vehicle_tap = None;
            return false;
        }

        let r_now = self.holding(self.buttons.r);
        let l3_now = self.holding(Button::LeftThumb);
        let y_now = self.holding(self.buttons.y);
        if r_now && !self.recenter_held {
            self.recenter_pending = true;
        }
        if l3_now && !self.center_held {
            self.center_pending = true;
        }
        if y_now && !self.reticle_center_held {
            self.reticle_center_pending = true;
        }
        self.recenter_held = r_now;
        self.center_held = l3_now;
        self.reticle_center_held = y_now;

        // Vehicle Special Action (A) and Dismount (B) HUD touch pulses.
        if self.game == zakuro_kiu::GAME_VEHICLE {
            let a_now = self.holding(self.buttons.a);
            let b_now = self.holding(self.buttons.b);
            if a_now && !self.vehicle_a_held {
                self.vehicle_tap = Some((VEHICLE_SPECIAL_TOUCH, VEHICLE_TAP_FRAMES));
            } else if b_now && !self.vehicle_b_held {
                self.vehicle_tap = Some((VEHICLE_DISMOUNT_TOUCH, VEHICLE_TAP_FRAMES));
            }
            self.vehicle_a_held = a_now;
            self.vehicle_b_held = b_now;
            if let Some((pos, rem)) = self.vehicle_tap {
                input.touch = Some(pos);
                self.vehicle_tap = (rem > 1).then_some((pos, rem - 1));
            }
        } else {
            self.vehicle_a_held = false;
            self.vehicle_b_held = false;
            self.vehicle_tap = None;
        }

        let (cx, cy) = stick_curve(rx, ry);
        let sign_y = if invert_y { 1.0 } else { -1.0 };
        let (dx, dy, mark) = if self.game == zakuro_kiu::GAME_FLYING {
            let dx = cx * AIR_PX_PER_FRAME;
            let dy = cy * sign_y * AIR_PX_PER_FRAME;
            let mark = if std::mem::take(&mut self.recenter_pending)
                || std::mem::take(&mut self.center_pending)
                || std::mem::take(&mut self.reticle_center_pending)
            {
                MOUSE_AIM_RECENTER
            } else {
                MOUSE_AIM
            };
            (dx, dy, mark)
        } else {
            let dx = cx * LAND_X_PX_PER_FRAME;
            let dy = cy * sign_y * LAND_Y_PX_PER_FRAME;
            let mark = if std::mem::take(&mut self.recenter_pending) {
                MOUSE_LOOK_RECENTER
            } else if std::mem::take(&mut self.center_pending) {
                MOUSE_LOOK_CENTER
            } else if std::mem::take(&mut self.reticle_center_pending) {
                MOUSE_RETICLE_CENTER
            } else if self.recenter_held {
                MOUSE_LOOK_HOLD
            } else {
                MOUSE_LOOK
            };
            (dx, dy, mark)
        };

        let same_family = |a: u32, b: u32| {
            let is_air = |m: u32| matches!(m, MOUSE_AIM | MOUSE_AIM_RECENTER);
            let is_land = |m: u32| {
                matches!(
                    m,
                    MOUSE_LOOK | MOUSE_LOOK_RECENTER | MOUSE_LOOK_HOLD | MOUSE_LOOK_CENTER | MOUSE_RETICLE_CENTER
                )
            };
            (is_air(a) && is_air(b)) || (is_land(a) && is_land(b))
        };
        let waiting = same_family(mailbox[0], mark) && (mailbox[1] != 0 || mailbox[2] != 0);
        self.untaken = if waiting { self.untaken.saturating_add(1) } else { 0 };
        let keep = waiting && self.untaken <= STALE_FRAMES;
        let carried = |word: u32| if keep { f32::from_bits(word) } else { 0.0 };
        let across = carried(mailbox[1]) + dx;
        let down = carried(mailbox[2]) + dy;
        let write_mark = if matches!(mailbox[0], MOUSE_LOOK_RECENTER | MOUSE_LOOK_CENTER | MOUSE_RETICLE_CENTER | MOUSE_AIM_RECENTER)
            && matches!(mark, MOUSE_LOOK | MOUSE_LOOK_HOLD | MOUSE_AIM)
        {
            mailbox[0]
        } else {
            mark
        };
        mailbox[..3].copy_from_slice(&[write_mark, across.to_bits(), down.to_bits()]);
        true
    }

    /// what the controllers hold down, over what the keyboard does.
    pub fn apply(&self, mut state: InputState) -> InputState {
        let Some(gilrs) = &self.gilrs else { return state };
        for (_, pad) in gilrs.gamepads() {
            for (button, bound) in self.buttons.map() {
                if pad.is_pressed(bound) {
                    state.buttons.set(button, true);
                }
            }
            // the d-pad of some controllers is a pair of axes
            let dpad = [
                (PadState::RIGHT, pad.value(Axis::DPadX) > 0.5),
                (PadState::LEFT, pad.value(Axis::DPadX) < -0.5),
                (PadState::UP, pad.value(Axis::DPadY) > 0.5),
                (PadState::DOWN, pad.value(Axis::DPadY) < -0.5),
            ];
            for (button, held) in dpad {
                if held {
                    state.buttons.set(button, true);
                }
            }
            // the left stick is the circle pad and the right one the
            // C-stick, up positive as the console has it
            let (x, y) = (pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY));
            if x.hypot(y) > DEADZONE {
                state.circle_x = x.clamp(-1.0, 1.0);
                state.circle_y = y.clamp(-1.0, 1.0);
            }
            let (x, y) = (pad.value(Axis::RightStickX), pad.value(Axis::RightStickY));
            if x.hypot(y) > DEADZONE {
                state.c_stick_x = x.clamp(-1.0, 1.0);
                state.c_stick_y = y.clamp(-1.0, 1.0);
            }
        }
        state
    }
}

/// a controller button as a person would call it.
pub fn button_name(button: Button) -> &'static str {
    match button {
        Button::South => "Bottom face",
        Button::East => "Right face",
        Button::North => "Top face",
        Button::West => "Left face",
        Button::LeftTrigger => "LB / L1",
        Button::RightTrigger => "RB / R1",
        Button::LeftTrigger2 => "LT / L2",
        Button::RightTrigger2 => "RT / R2",
        Button::Select => "Back / Select",
        Button::Start => "Start",
        Button::Mode => "Home",
        Button::LeftThumb => "Left stick",
        Button::RightThumb => "Right stick",
        Button::DPadUp => "D-pad up",
        Button::DPadDown => "D-pad down",
        Button::DPadLeft => "D-pad left",
        Button::DPadRight => "D-pad right",
        Button::C => "C",
        Button::Z => "Z",
        Button::Unknown => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// with no controller, or no way to ask for one, the keyboard's input
    /// goes through as it was.
    #[test]
    fn no_controller_leaves_the_keyboard_alone() {
        let gamepads = Gamepads::new(PadButtons::default());
        if gamepads.gilrs.as_ref().is_some_and(|gilrs| gilrs.gamepads().next().is_some()) {
            return;
        }
        let state = InputState { buttons: PadState::A, circle_x: 1.0, circle_y: 0.0, touch: Some((1, 2)), ..InputState::default() };
        let applied = gamepads.apply(state);
        assert_eq!(applied.buttons, PadState::A);
        assert_eq!((applied.circle_x, applied.circle_y, applied.touch), (1.0, 0.0, Some((1, 2))));
    }
}
