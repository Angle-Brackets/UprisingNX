//! the in-game settings menu: Minus (or right stick click + Minus) opens it, the game
//! pauses while it is open. D-pad Up/Down picks a row, Left/Right (or A)
//! changes it, B or right stick click + Minus again closes it. every change
//! is applied at once and written to settings.ini; the chosen row's
//! description shows under the rows. it also draws the hotkeys' notices.
//!
//! drawn with the presenter's overlay: one texture holds the printable ASCII
//! glyphs of the Switch's system font (handed over by unx_set_font) and a
//! white block for the panels.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use zakuro_gpu::backend::{Overlay, OverlayMesh, OverlayTexture, OverlayVertex};

use crate::settings::{Preset, Settings};

/// the system font, kept by unx_set_font
pub static FONT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

const FIRST: u8 = b' ';
/// '~' is the last glyph; the cell after it is the white block
const COUNT: usize = (b'~' - FIRST + 1) as usize;
const COLUMNS: usize = 16;
/// empty texels between cells, so filtering at a quad's edge never picks up
/// the neighbour (the white block under 'o' drew a line under it)
const GUTTER: f32 = 4.0;

/// the 3DS PadState bits the menu reads (zakuro-core services/hid.rs)
const PAD_A: u32 = 1 << 0;
const PAD_B: u32 = 1 << 1;
const PAD_RIGHT: u32 = 1 << 4;
const PAD_LEFT: u32 = 1 << 5;
const PAD_UP: u32 = 1 << 6;
const PAD_DOWN: u32 = 1 << 7;
/// frames a held direction waits before it repeats, then between repeats
const REPEAT_FIRST: u32 = 20;
const REPEAT_EVERY: u32 = 6;

struct Atlas {
    id: u64,
    /// the text size it was made for, in window pixels
    px: f32,
    size: [u32; 2],
    cell: [f32; 2],
    advance: [f32; COUNT],
    /// the texture still has to be handed to the presenter
    fresh: Option<Vec<u8>>,
}

impl Atlas {
    fn new(id: u64, px: f32) -> Option<Atlas> {
        let font = FontRef::try_from_slice(FONT.get()?).ok()?;
        let scaled = font.as_scaled(PxScale::from(px));
        let pad = 2.0;
        let cell = [(px * 0.9 + pad * 2.0).ceil(), (scaled.ascent() - scaled.descent() + pad * 2.0).ceil()];
        let rows = (COUNT + 1).div_ceil(COLUMNS);
        let stride = [cell[0] + GUTTER, cell[1] + GUTTER];
        let size = [(stride[0] as usize * COLUMNS) as u32, (stride[1] as usize * rows) as u32];
        let mut pixels = vec![0u8; (size[0] * size[1] * 4) as usize];
        let mut advance = [0.0; COUNT];
        for (index, advance) in advance.iter_mut().enumerate() {
            let c = (FIRST + index as u8) as char;
            let id = font.glyph_id(c);
            *advance = scaled.h_advance(id);
            let (cx, cy) = ((index % COLUMNS) as f32 * stride[0], (index / COLUMNS) as f32 * stride[1]);
            let glyph = id.with_scale_and_position(PxScale::from(px), point(cx + pad, cy + pad + scaled.ascent()));
            let Some(outline) = font.outline_glyph(glyph) else { continue };
            let bounds = outline.px_bounds();
            outline.draw(|x, y, coverage| {
                let (x, y) = (bounds.min.x as i32 + x as i32, bounds.min.y as i32 + y as i32);
                // keep to the glyph's own cell
                if x < cx as i32 || y < cy as i32 || x >= (cx + cell[0]) as i32 || y >= (cy + cell[1]) as i32 {
                    return;
                }
                let at = ((y as u32 * size[0] + x as u32) * 4) as usize;
                let a = (coverage.clamp(0.0, 1.0) * 255.0) as u8;
                pixels[at..at + 4].copy_from_slice(&[a, a, a, a]);
            });
        }
        // the white block
        let (bx, by) = ((COUNT % COLUMNS) * stride[0] as usize, (COUNT / COLUMNS) * stride[1] as usize);
        for y in by..by + cell[1] as usize {
            for x in bx..bx + cell[0] as usize {
                let at = (y * size[0] as usize + x) * 4;
                pixels[at..at + 4].copy_from_slice(&[255; 4]);
            }
        }
        Some(Atlas { id, px, size, cell, advance, fresh: Some(pixels) })
    }

    fn uv(&self, index: usize, inset: f32) -> [f32; 4] {
        let (x, y) = ((index % COLUMNS) as f32 * (self.cell[0] + GUTTER), (index / COLUMNS) as f32 * (self.cell[1] + GUTTER));
        let (w, h) = (self.size[0] as f32, self.size[1] as f32);
        [(x + inset) / w, (y + inset) / h, (x + self.cell[0] - inset) / w, (y + self.cell[1] - inset) / h]
    }

    fn width(&self, text: &str) -> f32 {
        text.bytes().map(|b| self.advance.get(b.wrapping_sub(FIRST) as usize).copied().unwrap_or(self.px * 0.5)).sum()
    }
}

/// triangles for one frame
struct Painter<'a> {
    atlas: &'a Atlas,
    vertices: Vec<OverlayVertex>,
    indices: Vec<u32>,
}

impl Painter<'_> {
    fn quad(&mut self, rect: [f32; 4], uv: [f32; 4], color: [u8; 4]) {
        let base = self.vertices.len() as u32;
        let [x0, y0, x1, y1] = rect;
        let [u0, v0, u1, v1] = uv;
        for (position, uv) in [([x0, y0], [u0, v0]), ([x1, y0], [u1, v0]), ([x1, y1], [u1, v1]), ([x0, y1], [u0, v1])] {
            self.vertices.push(OverlayVertex { position, uv, color });
        }
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    fn rect(&mut self, rect: [f32; 4], color: [u8; 4]) {
        let uv = self.atlas.uv(COUNT, self.atlas.cell[0] / 2.0 - 0.5);
        self.quad(rect, uv, color);
    }

    /// text with its top left at x, y, `size` times the atlas's size
    fn text(&mut self, mut x: f32, y: f32, text: &str, size: f32, color: [u8; 4]) {
        let pad = 2.0 * size;
        let atlas = self.atlas;
        for byte in text.bytes() {
            let index = byte.wrapping_sub(FIRST) as usize;
            if index >= COUNT {
                continue;
            }
            if byte != b' ' {
                let rect = [x - pad, y - pad, x - pad + atlas.cell[0] * size, y - pad + atlas.cell[1] * size];
                self.quad(rect, atlas.uv(index, 0.0), color);
            }
            x += atlas.advance[index] * size;
        }
    }
}

/// premultiplied
fn color(r: u8, g: u8, b: u8, a: u8) -> [u8; 4] {
    let m = |c: u8| (c as u16 * a as u16 / 255) as u8;
    [m(r), m(g), m(b), a]
}

/// `text` broken into lines no wider than `width` at `size`
fn wrap(atlas: &Atlas, text: &str, size: f32, width: f32) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let tried = if line.is_empty() { word.to_string() } else { format!("{line} {word}") };
        if atlas.width(&tried) * size > width && !line.is_empty() {
            lines.push(std::mem::replace(&mut line, word.to_string()));
        } else {
            line = tried;
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Row {
    Editing,
    Preset,
    BattleScale,
    MenuScale,
    Smooth,
    PowerLabel,
    RightStick,
    Gyro,
    GyroSpeed,
    AimSpeed,
    InvertY,
    LeftStick,
    Modifier,
    Minus,
    Resume,
    Quit,
}

const ROWS: [Row; 16] = [
    Row::Editing,
    Row::Preset,
    Row::BattleScale,
    Row::MenuScale,
    Row::Smooth,
    Row::PowerLabel,
    Row::RightStick,
    Row::Gyro,
    Row::GyroSpeed,
    Row::AimSpeed,
    Row::InvertY,
    Row::LeftStick,
    Row::Modifier,
    Row::Minus,
    Row::Resume,
    Row::Quit,
];

const MODIFIERS: [&str; 7] = ["ZL", "ZR", "L", "R", "X", "Y", "A"];
const POWER_LABELS: [&str; 3] = ["bottom", "bar", "off"];

impl Row {
    fn label(self) -> &'static str {
        match self {
            Row::Editing => "Settings for",
            Row::Preset => "Preset",
            Row::BattleScale => "Battle resolution",
            Row::MenuScale => "Menu resolution",
            Row::Smooth => "Smooth mode (60 fps)",
            Row::PowerLabel => "Power label",
            Row::RightStick => "Right stick aim",
            Row::Gyro => "Gyro aim",
            Row::GyroSpeed => "Gyro speed",
            Row::AimSpeed => "Stick aim speed",
            Row::InvertY => "Invert aim Y",
            Row::LeftStick => "Left stick aim",
            Row::Modifier => "Aim button",
            Row::Minus => "Minus button",
            Row::Resume => "Resume",
            Row::Quit => "Quit game",
        }
    }

    fn description(self, docked: bool) -> &'static str {
        match self {
            Row::Editing => "Which video settings the rows below change: handheld or docked. Each has its own preset and resolutions, and the other one takes over when you dock or undock. The aim settings are shared.",
            Row::Preset => {
                if docked {
                    "Performance: 3x in battles, 4x elsewhere, smooth mode on. Quality: 4x everywhere at the game's own 30 fps. Custom: the values below."
                } else {
                    "Performance: 1x in battles, 2x elsewhere, smooth mode on. Quality: 2x everywhere at the game's own 30 fps. Custom: the values below."
                }
            }
            Row::BattleScale => "How sharp battles are drawn, in multiples of the 3DS screen. Higher looks better but can drop below 30 fps or turn smooth mode off. Changing it makes the preset Custom.",
            Row::MenuScale => "How sharp everything outside battles is drawn: menus, the hub and cutscenes. Changing it makes the preset Custom.",
            Row::Smooth => "Draws a picture between each two of the game's for 60 fps. It turns itself off for a while when the GPU can't keep up. Changing it makes the preset Custom.",
            Row::PowerLabel => "Where the selected power's name appears on foot. Bottom: centered at the bottom in crisp white text. Above bar: in a bubble over the power icons. Off: icons only.",
            Row::RightStick => "Aims the reticle in the air and turns the camera on foot with the right stick, in place of the touch screen.",
            Row::Gyro => "Fine aiming by tilting the console or controller.",
            Row::GyroSpeed => "How far the reticle moves for a tilt. The game's own Controls options scale it further.",
            Row::AimSpeed => "How fast the right stick aims and turns the camera.",
            Row::InvertY => "Flips up and down for aiming and the camera.",
            Row::LeftStick => "Off: the left stick only moves Pit. 1: in the air it also aims, and on foot holding the aim button turns the camera with it. 2: as 1, but in the air it aims only while the aim button is held.",
            Row::Modifier => "The button held with the left stick to aim or turn the camera (Left stick aim 1 or 2). Fire stays on L/ZL.",
            Row::Minus => "Settings menu: Minus alone opens and closes this menu (Plus still pauses; the game's Select only pauses in battles, like Plus). 3DS Select: Minus goes to the game, and right stick click + Minus opens this menu.",
            Row::Resume => "Back to the game. B or Minus (or right stick click + Minus) also closes this menu.",
            Row::Quit => "Closes UprisingNX. Progress since the game last saved is lost.",
        }
    }
}

/// what changed, for Unx to put in place
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Changed {
    Video,
    Controls,
    Quit,
}

/// refreshes the game sees no input for after the menu closes
const QUIET_AFTER_CLOSE: u32 = 6;

/// how long a hotkey's notice stays, and fades over its end
const TOAST_SECONDS: f32 = 2.5;
const TOAST_FADE: f32 = 0.5;

pub struct Menu {
    pub open: bool,
    cursor: usize,
    /// Quit was pressed once; a second press quits
    quit_armed: bool,
    /// whose video settings the menu shows and changes, docked or handheld;
    /// the mode the console is in when it opens
    editing_docked: bool,
    /// after it closes, the game gets no buttons or touch until all are let
    /// go, and none for at least this many refreshes
    blocking: bool,
    quiet: u32,
    last_buttons: u32,
    /// the direction held and for how many frames, for repeats
    held: (u32, u32),
    atlas: Option<Atlas>,
    next_id: u64,
    /// a hotkey's notice, while the menu is closed
    toast: Option<(String, std::time::Instant)>,
}

fn control<'a>(settings: &'a Settings, key: &str) -> &'a str {
    settings.controls.iter().find(|(k, _)| k == key).map_or("", |(_, v)| v.as_str())
}

fn set_control(settings: &mut Settings, key: &str, value: String) {
    match settings.controls.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => settings.controls.push((key.to_string(), value)),
    }
}

fn speed(settings: &Settings, key: &str) -> f32 {
    control(settings, key).trim().parse().unwrap_or(1.0)
}

fn on(settings: &Settings, key: &str) -> bool {
    control(settings, key).trim() != "0"
}

fn modifier_index(settings: &Settings) -> usize {
    let name = control(settings, "aim_modifier").trim().to_ascii_uppercase();
    MODIFIERS.iter().position(|m| *m == name).unwrap_or(0)
}

fn power_label_index(settings: &Settings) -> usize {
    match settings.power_label_mode() {
        zakuro_gpu::PowerLabelMode::Bottom => 0,
        zakuro_gpu::PowerLabelMode::AboveBar => 1,
        zakuro_gpu::PowerLabelMode::Off => 2,
    }
}

fn left_stick(settings: &Settings) -> u32 {
    control(settings, "left_stick_aim").trim().parse::<u32>().unwrap_or(0).min(2)
}

impl Menu {
    pub fn new() -> Menu {
        Menu {
            open: false,
            cursor: 0,
            quit_armed: false,
            editing_docked: false,
            blocking: false,
            quiet: 0,
            last_buttons: 0,
            held: (0, 0),
            atlas: None,
            next_id: 0x4D45_4E55_0000,
            toast: None,
        }
    }

    pub fn toggle(&mut self, buttons: u32, docked: bool) {
        self.open = !self.open;
        // what is held as it opens or closes is not a press
        self.last_buttons = buttons;
        self.held = (0, 0);
        self.quit_armed = false;
        self.editing_docked = docked;
        // the button that closed it, or any other, must not reach the game
        self.blocking = !self.open;
        self.quiet = if self.open { 0 } else { QUIET_AFTER_CLOSE };
        self.toast = None;
        log::info!("unx: menu {}", if self.open { "opened, the game paused" } else { "closed" });
    }

    /// a notice over the game for a few seconds, for the hotkeys
    pub fn notify(&mut self, text: String) {
        self.toast = Some((text, std::time::Instant::now()));
    }

    /// the game's buttons and whether it may see a touch: after the menu
    /// closed, nothing until every button and the screen are let go (and
    /// QUIET_AFTER_CLOSE refreshes have passed)
    pub fn filter(&mut self, buttons: u32, touching: bool) -> (u32, bool) {
        if self.quiet > 0 {
            self.quiet -= 1;
            return (0, false);
        }
        if self.blocking {
            if buttons != 0 || touching {
                return (0, false);
            }
            self.blocking = false;
        }
        (buttons, touching)
    }

    /// reads the buttons (3DS PadState bits) while open: what changed, and
    /// closes on B or Resume
    pub fn input(&mut self, buttons: u32, settings: &mut Settings, docked_now: bool) -> Option<Changed> {
        let pressed = buttons & !self.last_buttons;
        self.last_buttons = buttons;
        // a held direction repeats
        let direction = buttons & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT);
        let mut step = pressed & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT);
        if direction != 0 && direction == self.held.0 {
            self.held.1 += 1;
            if self.held.1 >= REPEAT_FIRST && (self.held.1 - REPEAT_FIRST) % REPEAT_EVERY == 0 {
                step |= direction;
            }
        } else {
            self.held = (direction, 0);
        }
        if pressed & PAD_B != 0 {
            self.toggle(buttons, docked_now);
            return None;
        }
        if step & (PAD_UP | PAD_DOWN) != 0 {
            self.quit_armed = false;
        }
        if step & PAD_UP != 0 {
            self.cursor = (self.cursor + ROWS.len() - 1) % ROWS.len();
        }
        if step & PAD_DOWN != 0 {
            self.cursor = (self.cursor + 1) % ROWS.len();
        }
        let a = pressed & PAD_A != 0;
        let delta: i32 = if step & PAD_RIGHT != 0 || a {
            1
        } else if step & PAD_LEFT != 0 {
            -1
        } else {
            return None;
        };
        let row = ROWS[self.cursor];
        let docked = self.editing_docked;
        let mode = settings.mode_mut(docked);
        let changed = match row {
            Row::Editing => {
                self.editing_docked = !self.editing_docked;
                return None;
            }
            Row::Preset => {
                mode.preset = match (mode.preset, delta > 0) {
                    (p, true) => p.next(),
                    (Preset::Performance, false) => Preset::Custom,
                    (Preset::Quality, false) => Preset::Performance,
                    (Preset::Custom, false) => Preset::Quality,
                };
                Changed::Video
            }
            Row::BattleScale | Row::MenuScale => {
                let video = mode.customize(docked);
                let scale = if row == Row::BattleScale { &mut video.battle_scale } else { &mut video.menu_scale };
                *scale = (*scale as i32 + delta).clamp(1, 4) as u32;
                Changed::Video
            }
            Row::Smooth => {
                let video = mode.customize(docked);
                video.smooth = !video.smooth;
                Changed::Video
            }
            Row::PowerLabel => {
                let index = (power_label_index(settings) as i32 + delta).rem_euclid(POWER_LABELS.len() as i32) as usize;
                set_control(settings, "power_label", POWER_LABELS[index].into());
                Changed::Controls
            }
            Row::RightStick | Row::Gyro | Row::InvertY => {
                let key = match row {
                    Row::RightStick => "right_stick_aim",
                    Row::Gyro => "gyro_aim",
                    _ => "aim_invert_y",
                };
                let value = if on(settings, key) { "0" } else { "1" };
                set_control(settings, key, value.into());
                Changed::Controls
            }
            Row::GyroSpeed | Row::AimSpeed => {
                let key = if row == Row::GyroSpeed { "gyro_speed" } else { "aim_speed" };
                let value = (speed(settings, key) + delta as f32 * 0.25).clamp(0.25, 4.0);
                set_control(settings, key, format!("{value:.2}"));
                Changed::Controls
            }
            Row::LeftStick => {
                let value = (left_stick(settings) as i32 + delta).rem_euclid(3);
                set_control(settings, "left_stick_aim", value.to_string());
                Changed::Controls
            }
            Row::Minus => {
                let value = if settings.minus_opens_menu() { "select" } else { "menu" };
                set_control(settings, "minus_button", value.into());
                Changed::Controls
            }
            Row::Modifier => {
                let index = (modifier_index(settings) as i32 + delta).rem_euclid(MODIFIERS.len() as i32) as usize;
                set_control(settings, "aim_modifier", MODIFIERS[index].into());
                Changed::Controls
            }
            Row::Resume => {
                if a {
                    self.toggle(buttons, docked_now);
                }
                return None;
            }
            Row::Quit => {
                if !a {
                    return None;
                }
                if !self.quit_armed {
                    self.quit_armed = true;
                    return None;
                }
                Changed::Quit
            }
        };
        Some(changed)
    }

    /// the atlas for this window height, made (and the old one freed) when
    /// the size changes; false without a font
    fn ready(&mut self, height: u32, overlay: &mut Overlay) -> bool {
        let px = (height as f32 / 720.0 * 22.0).round();
        if self.atlas.as_ref().is_none_or(|atlas| atlas.px != px) {
            if let Some(old) = self.atlas.take() {
                overlay.free.push(old.id);
            }
            self.next_id += 1;
            self.atlas = Atlas::new(self.next_id, px);
            if self.atlas.is_none() {
                log::warn!("unx: menu has no font to draw with");
            }
        }
        let Some(atlas) = self.atlas.as_mut() else { return false };
        if let Some(pixels) = atlas.fresh.take() {
            overlay.textures.push(OverlayTexture { id: atlas.id, offset: None, size: atlas.size, pixels, linear: true });
        }
        true
    }

    /// the menu, or a hotkey's notice, over the game, in a window `width` x
    /// `height` pixels
    pub fn overlay(&mut self, settings: &Settings, docked: bool, now: (u32, bool), width: u32, height: u32) -> Overlay {
        let mut overlay = Overlay::default();
        let toast = self.toast.as_ref().map(|(text, at)| (text.clone(), at.elapsed().as_secs_f32()));
        if toast.as_ref().is_some_and(|(_, age)| *age > TOAST_SECONDS) {
            self.toast = None;
        }
        if !self.open && self.toast.is_none() {
            return overlay;
        }
        if !self.ready(height, &mut overlay) {
            return overlay;
        }
        let atlas = self.atlas.as_ref().unwrap();
        let mut painter = Painter { atlas, vertices: Vec::new(), indices: Vec::new() };
        let k = height as f32 / 720.0;
        if !self.open {
            if let Some((text, age)) = toast {
                let alpha = ((TOAST_SECONDS - age) / TOAST_FADE).clamp(0.0, 1.0);
                let a = |v: u8| (v as f32 * alpha) as u8;
                let size = 0.85;
                let (w, h) = (atlas.width(&text) * size + 36.0 * k, atlas.px * size * 1.7);
                let (x, y) = ((width as f32 - w) / 2.0, 24.0 * k);
                painter.rect([x, y, x + w, y + h], color(16, 20, 40, a(225)));
                painter.rect([x, y + h - 3.0 * k, x + w, y + h], color(250, 200, 80, a(255)));
                painter.text(x + 18.0 * k, y + (h - atlas.px * size * 1.2) / 2.0, &text, size, color(235, 238, 245, a(255)));
            }
        } else {
            self.paint_menu(&mut painter, settings, docked, now, width, height);
        }
        let (vertices, indices) = (painter.vertices, painter.indices);
        overlay.meshes.push(OverlayMesh { texture: atlas.id, clip: [0, 0, width, height], vertices, indices });
        overlay
    }

    fn paint_menu(&self, painter: &mut Painter, settings: &Settings, docked_now: bool, now: (u32, bool), width: u32, height: u32) {
        let atlas = painter.atlas;
        let docked = self.editing_docked;
        let mode = settings.mode(docked);
        let video = mode.video(docked);
        let scale = |s: u32| format!("{s}x ({}p)", 240 * s);
        let flag = |b: bool| if b { "On" } else { "Off" }.to_string();
        let value = |row: Row| -> String {
            match row {
                Row::Editing => {
                    let name = if docked { "Docked" } else { "Handheld" };
                    if docked == docked_now { format!("{name} (now)") } else { name.into() }
                }
                Row::Preset => match mode.preset {
                    Preset::Performance => "Performance".into(),
                    Preset::Quality => "Quality".into(),
                    Preset::Custom => "Custom".into(),
                },
                Row::BattleScale => scale(video.battle_scale),
                Row::MenuScale => scale(video.menu_scale),
                Row::Smooth => flag(video.smooth),
                Row::PowerLabel => match settings.power_label_mode() {
                    zakuro_gpu::PowerLabelMode::Bottom => "Bottom".into(),
                    zakuro_gpu::PowerLabelMode::AboveBar => "Above bar".into(),
                    zakuro_gpu::PowerLabelMode::Off => "Off".into(),
                },
                Row::RightStick => flag(on(settings, "right_stick_aim")),
                Row::Gyro => flag(on(settings, "gyro_aim")),
                Row::GyroSpeed => format!("{:.2}", speed(settings, "gyro_speed")),
                Row::AimSpeed => format!("{:.2}", speed(settings, "aim_speed")),
                Row::InvertY => flag(on(settings, "aim_invert_y")),
                Row::LeftStick => match left_stick(settings) {
                    0 => "Off".into(),
                    n => n.to_string(),
                },
                Row::Modifier => MODIFIERS[modifier_index(settings)].into(),
                Row::Minus => if settings.minus_opens_menu() { "Settings menu".into() } else { "3DS Select".into() },
                Row::Resume => String::new(),
                Row::Quit => if self.quit_armed { "Press A again".into() } else { String::new() },
            }
        };

        let k = height as f32 / 720.0;
        let line = (atlas.px * 1.4).round();
        let small = 0.78;
        let small_line = line * 0.8;
        let panel_w = 760.0 * k;
        let margin = 26.0 * k;
        let row = ROWS[self.cursor];
        let description = wrap(atlas, row.description(docked), small, panel_w - margin * 2.0);
        // room for the longest description, so the panel doesn't jump
        let described = ROWS.iter().map(|r| wrap(atlas, r.description(docked), small, panel_w - margin * 2.0).len()).max().unwrap_or(1);
        let panel_h = line * (1.5 + ROWS.len() as f32 + 0.5) + small_line * described as f32 + line * 1.4;
        let (left, top) = ((width as f32 - panel_w) / 2.0, ((height as f32 - panel_h) / 2.0).max(0.0));

        // dim the game, then the panel
        painter.rect([0.0, 0.0, width as f32, height as f32], color(0, 0, 0, 120));
        painter.rect([left, top, left + panel_w, top + panel_h], color(16, 20, 40, 238));
        painter.rect([left, top, left + panel_w, top + 3.0 * k], color(250, 200, 80, 255));
        let title = format!("UprisingNX settings - playing {}", if docked_now { "docked" } else { "handheld" });
        painter.text(left + margin, top + line * 0.4, &title, 1.0, color(250, 200, 80, 255));
        for (index, row) in ROWS.iter().enumerate() {
            let y = top + line * (1.5 + index as f32);
            let chosen = index == self.cursor;
            if chosen {
                painter.rect([left + 10.0 * k, y - line * 0.14, left + panel_w - 10.0 * k, y + line * 0.86], color(70, 90, 160, 255));
            }
            let text = if *row == Row::Quit && self.quit_armed {
                color(255, 150, 130, 255)
            } else if chosen {
                color(255, 255, 255, 255)
            } else {
                color(200, 205, 220, 255)
            };
            painter.text(left + margin, y, row.label(), 1.0, text);
            let value = value(*row);
            if !value.is_empty() {
                let cycles = !matches!(row, Row::Quit);
                let shown = if chosen && cycles { format!("< {value} >") } else { value };
                let x = left + panel_w - margin - atlas.width(&shown);
                painter.text(x, y, &shown, 1.0, text);
            }
        }
        // the chosen row's description under a rule
        let rule = top + line * (1.5 + ROWS.len() as f32) + line * 0.15;
        painter.rect([left + margin, rule, left + panel_w - margin, rule + 1.5 * k], color(90, 100, 130, 255));
        for (index, text) in description.iter().enumerate() {
            let y = rule + line * 0.3 + small_line * index as f32;
            painter.text(left + margin, y, text, small, color(175, 185, 210, 255));
        }
        let (now_scale, now_smooth) = now;
        let help = format!("running {now_scale}x, smooth {}    A/Left/Right: change    B: back", if now_smooth { "on" } else { "off" });
        let y = top + panel_h - line * 1.0;
        painter.text(left + margin, y, &help, small, color(130, 140, 165, 255));
    }
}
