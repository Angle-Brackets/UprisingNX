//! game controllers, any the system knows, read through gilrs.

use gilrs::{Axis, Button, EventType, Gamepad, Gilrs};
use zakuro_core::services::hid::{InputState, PadState};

use crate::settings::{ButtonLabels, PadButtons};

/// Analog trigger press threshold so digital and analog triggers behave identically.
const TRIGGER_THRESHOLD: f32 = 0.40;

/// KIU right-stick base aiming curve & speeds (matching Switch `uprising_nx::aim`).
const DEFAULT_DEADZONE: f32 = 0.10;
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

/// Live controller aim tuning passed from user settings.
#[derive(Debug, Clone, Copy)]
pub struct PadAimSettings {
    pub air_speed: f32,
    pub land_speed: f32,
    pub deadzone: f32,
    pub invert_y: bool,
}

impl Default for PadAimSettings {
    fn default() -> Self {
        Self {
            air_speed: 1.0,
            land_speed: 1.0,
            deadzone: DEFAULT_DEADZONE,
            invert_y: false,
        }
    }
}

/// Snapshot of connected controllers and live stick deflection for the settings UI.
#[derive(Debug, Clone, Default)]
pub struct GamepadSnapshot {
    pub connected: Vec<String>,
    pub detected_style: ButtonLabels,
    pub left_stick: (f32, f32),
    pub right_stick: (f32, f32),
}

fn stick_curve(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
    let dz = deadzone.clamp(0.02, 0.45);
    let mag = x.hypot(y);
    if mag <= dz {
        return (0.0, 0.0);
    }
    let norm = ((mag - dz) / (1.0 - dz)).clamp(0.0, 1.0);
    let curved = AIM_LINEAR_GAIN * norm + (1.0 - AIM_LINEAR_GAIN) * norm * norm;
    let scale = curved / mag;
    (x * scale, y * scale)
}

/// Checks whether a button (digital OR analog trigger past threshold) is held on `pad`.
fn holding_on(pad: Gamepad<'_>, button: Button) -> bool {
    if button == Button::Unknown {
        return false;
    }
    if pad.is_pressed(button) {
        return true;
    }
    if matches!(button, Button::LeftTrigger2 | Button::RightTrigger2) {
        if let Some(data) = pad.button_data(button) {
            return data.value() >= TRIGGER_THRESHOLD;
        }
    }
    false
}

/// Guesses the best button-label style from a controller's reported device name.
pub fn detect_button_style(name: &str) -> ButtonLabels {
    let lower = name.to_ascii_lowercase();
    if lower.contains("dualsense")
        || lower.contains("dualshock")
        || lower.contains("playstation")
        || lower.contains("ps4")
        || lower.contains("ps5")
        || lower.contains("sony")
    {
        ButtonLabels::PlayStation
    } else if lower.contains("nintendo")
        || lower.contains("switch")
        || lower.contains("pro controller")
        || lower.contains("joy-con")
        || lower.contains("joycon")
    {
        ButtonLabels::Nintendo
    } else if lower.contains("xbox")
        || lower.contains("xinput")
        || lower.contains("x-box")
        || lower.contains("microsoft")
        || lower.contains("steam deck")
        || lower.contains("valve")
    {
        ButtonLabels::Xbox
    } else {
        ButtonLabels::Generic
    }
}

pub struct Gamepads {
    /// none when the system could not be asked for controllers.
    gilrs: Option<Gilrs>,
    buttons: PadButtons,
    deadzone: f32,
    /// whether the player last aimed/played with a controller rather than KBM.
    active: bool,
    game: u32,
    quiet: u8,
    untaken: u8,
    /// Edge-tracking for analog triggers so `poll()` reports trigger pulls cleanly for rebinding.
    lt_analog_held: bool,
    rt_analog_held: bool,
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
            deadzone: DEFAULT_DEADZONE,
            active: false,
            game: 0,
            quiet: 0,
            untaken: 0,
            lt_analog_held: false,
            rt_analog_held: false,
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

    pub fn set_deadzone(&mut self, deadzone: f32) {
        self.deadzone = deadzone.clamp(0.03, 0.30);
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

    /// Live connected controller info and stick deflections for the Controller settings UI.
    pub fn snapshot(&self) -> GamepadSnapshot {
        let Some(gilrs) = &self.gilrs else {
            return GamepadSnapshot { detected_style: ButtonLabels::Generic, ..Default::default() };
        };
        let mut connected = Vec::new();
        let mut left_stick = (0.0f32, 0.0f32);
        let mut right_stick = (0.0f32, 0.0f32);
        for (_, pad) in gilrs.gamepads() {
            if !pad.is_connected() {
                continue;
            }
            let lower = pad.name().to_ascii_lowercase();
            if lower.contains("system control") || lower.contains("consumer control") {
                continue;
            }
            connected.push(pad.name().to_owned());
            let (lx, ly) = (pad.value(Axis::LeftStickX).clamp(-1.0, 1.0), pad.value(Axis::LeftStickY).clamp(-1.0, 1.0));
            if lx.hypot(ly) > left_stick.0.hypot(left_stick.1) {
                left_stick = (lx, ly);
            }
            let (rx, ry) = (pad.value(Axis::RightStickX).clamp(-1.0, 1.0), pad.value(Axis::RightStickY).clamp(-1.0, 1.0));
            if rx.hypot(ry) > right_stick.0.hypot(right_stick.1) {
                right_stick = (rx, ry);
            }
        }
        let detected_style = connected
            .first()
            .map(|name| detect_button_style(name))
            .unwrap_or(ButtonLabels::Generic);
        GamepadSnapshot { connected, detected_style, left_stick, right_stick }
    }

    /// takes in what the controllers did since the last call, and returns the
    /// buttons pressed meanwhile (including analog trigger pulls past threshold).
    pub fn poll(&mut self) -> Vec<Button> {
        let mut pressed = Vec::new();
        let Some(gilrs) = &mut self.gilrs else { return pressed };
        while let Some(event) = gilrs.next_event() {
            match event.event {
                EventType::ButtonPressed(button, _) => {
                    if button != Button::Unknown {
                        self.active = true;
                        if button == Button::LeftTrigger2 {
                            self.lt_analog_held = true;
                        } else if button == Button::RightTrigger2 {
                            self.rt_analog_held = true;
                        }
                        pressed.push(button);
                    }
                }
                EventType::ButtonReleased(button, _) => {
                    if button == Button::LeftTrigger2 {
                        self.lt_analog_held = false;
                    } else if button == Button::RightTrigger2 {
                        self.rt_analog_held = false;
                    }
                }
                EventType::ButtonChanged(button @ (Button::LeftTrigger2 | Button::RightTrigger2), val, _) => {
                    let held = if button == Button::LeftTrigger2 {
                        &mut self.lt_analog_held
                    } else {
                        &mut self.rt_analog_held
                    };
                    if val >= TRIGGER_THRESHOLD && !*held {
                        *held = true;
                        self.active = true;
                        pressed.push(button);
                    } else if val < TRIGGER_THRESHOLD * 0.6 {
                        *held = false;
                    }
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

    /// whether a controller holds button down (handles both digital and analog triggers).
    pub fn holding(&self, button: Button) -> bool {
        if button == Button::Unknown {
            return false;
        }
        self.gilrs.as_ref().is_some_and(|gilrs| gilrs.gamepads().any(|(_, pad)| holding_on(pad, button)))
    }

    /// raw right-stick axes `(x, y)` (up positive) from the first deflected controller.
    pub fn right_stick(&self, deadzone: f32) -> (f32, f32) {
        let Some(gilrs) = &self.gilrs else { return (0.0, 0.0) };
        for (_, pad) in gilrs.gamepads() {
            let (x, y) = (pad.value(Axis::RightStickX), pad.value(Axis::RightStickY));
            if x.hypot(y) > deadzone {
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
        aim: PadAimSettings,
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

        let dz = aim.deadzone.clamp(0.03, 0.30);
        let (rx, ry) = self.right_stick(dz);
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
        if self.game == kiu_common::GAME_VEHICLE {
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

        let (cx, cy) = stick_curve(rx, ry, dz);
        let sign_y = if aim.invert_y { 1.0 } else { -1.0 };
        let air_mult = aim.air_speed.clamp(0.2, 3.0);
        let land_mult = aim.land_speed.clamp(0.2, 3.0);
        let (dx, dy, mark) = if self.game == kiu_common::GAME_FLYING {
            let dx = cx * AIR_PX_PER_FRAME * air_mult;
            let dy = cy * sign_y * AIR_PX_PER_FRAME * air_mult;
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
            let dx = cx * LAND_X_PX_PER_FRAME * land_mult;
            let dy = cy * sign_y * LAND_Y_PX_PER_FRAME * land_mult;
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
        let dz = self.deadzone.max(0.05);
        for (_, pad) in gilrs.gamepads() {
            for (button, bound) in self.buttons.map() {
                if holding_on(pad, bound) {
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
            if x.hypot(y) > dz {
                state.circle_x = x.clamp(-1.0, 1.0);
                state.circle_y = y.clamp(-1.0, 1.0);
            }
            let (x, y) = (pad.value(Axis::RightStickX), pad.value(Axis::RightStickY));
            if x.hypot(y) > dz {
                state.c_stick_x = x.clamp(-1.0, 1.0);
                state.c_stick_y = y.clamp(-1.0, 1.0);
            }
        }
        state
    }
}

/// Controller button name formatted in the chosen glyph style (`Xbox`, `PlayStation`, `Nintendo`, `Generic`).
pub fn button_label(button: Button, style: ButtonLabels) -> &'static str {
    if button == Button::Unknown {
        return "Unbound";
    }
    match style {
        ButtonLabels::Xbox => match button {
            Button::South => "A",
            Button::East => "B",
            Button::North => "Y",
            Button::West => "X",
            Button::LeftTrigger => "LB",
            Button::RightTrigger => "RB",
            Button::LeftTrigger2 => "LT",
            Button::RightTrigger2 => "RT",
            Button::Select => "View",
            Button::Start => "Menu",
            Button::Mode => "Xbox",
            Button::LeftThumb => "LS",
            Button::RightThumb => "RS",
            _ => button_name(button),
        },
        ButtonLabels::PlayStation => match button {
            Button::South => "Cross (×)",
            Button::East => "Circle (○)",
            Button::North => "Triangle (△)",
            Button::West => "Square (□)",
            Button::LeftTrigger => "L1",
            Button::RightTrigger => "R1",
            Button::LeftTrigger2 => "L2",
            Button::RightTrigger2 => "R2",
            Button::Select => "Create",
            Button::Start => "Options",
            Button::Mode => "PS",
            Button::LeftThumb => "L3",
            Button::RightThumb => "R3",
            _ => button_name(button),
        },
        ButtonLabels::Nintendo => match button {
            Button::South => "B",
            Button::East => "A",
            Button::North => "X",
            Button::West => "Y",
            Button::LeftTrigger => "L",
            Button::RightTrigger => "R",
            Button::LeftTrigger2 => "ZL",
            Button::RightTrigger2 => "ZR",
            Button::Select => "Minus (−)",
            Button::Start => "Plus (+)",
            Button::Mode => "Home",
            Button::LeftThumb => "L Stick",
            Button::RightThumb => "R Stick",
            _ => button_name(button),
        },
        ButtonLabels::Auto | ButtonLabels::Generic => button_name(button),
    }
}

/// a controller button as a person would call it (generic positional labels).
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
        Button::Unknown => "Unbound",
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

    #[test]
    fn detects_controller_glyph_styles_from_name() {
        assert_eq!(detect_button_style("DualSense Wireless Controller"), ButtonLabels::PlayStation);
        assert_eq!(detect_button_style("Nintendo Switch Pro Controller"), ButtonLabels::Nintendo);
        assert_eq!(detect_button_style("Xbox Wireless Controller"), ButtonLabels::Xbox);
        assert_eq!(detect_button_style("Valve Steam Deck"), ButtonLabels::Xbox);
        assert_eq!(detect_button_style("8BitDo SN30"), ButtonLabels::Generic);
    }

    #[test]
    fn button_labels_format_unbound_and_styles_cleanly() {
        assert_eq!(button_label(Button::Unknown, ButtonLabels::Xbox), "Unbound");
        assert_eq!(button_label(Button::East, ButtonLabels::Nintendo), "A");
        assert_eq!(button_label(Button::East, ButtonLabels::Xbox), "B");
        assert_eq!(button_label(Button::East, ButtonLabels::PlayStation), "Circle (○)");
        assert_eq!(button_label(Button::LeftTrigger2, ButtonLabels::PlayStation), "L2");
        assert_eq!(button_label(Button::LeftTrigger2, ButtonLabels::Nintendo), "ZL");
    }

    #[test]
    fn stick_curve_respects_custom_deadzone() {
        assert_eq!(stick_curve(0.12, 0.0, 0.15), (0.0, 0.0));
        let (x, y) = stick_curve(0.60, 0.0, 0.15);
        assert!(x > 0.2 && x < 0.7 && y == 0.0);
        assert_eq!(stick_curve(1.0, 0.0, 0.15), (1.0, 0.0));
    }
}
