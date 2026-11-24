//! the in-game settings menu: Minus (or right stick click + Minus) opens it, the game
//! pauses while it is open. L/R switches tabs (Video, Controls, Save & System),
//! D-pad Up/Down picks a row, Left/Right (or A) changes it, and B or Minus closes it.
//! Every setting change is applied at once and written to settings.ini; the chosen
//! row's description shows under the rows. It also draws hotkey notices and the
//! pre-boot animated wing setup/progress screen.
//!
//! drawn with the presenter's overlay: one texture holds the printable ASCII
//! glyphs of the Switch's system font (handed over by unx_set_font) and a
//! white block for the panels.

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use zakuro_gpu::backend::{Overlay, OverlayMesh, OverlayTexture, OverlayVertex};

use crate::settings::{Preset, Settings};
use crate::updater::{UpdateTone, UpdateView};

/// the system font, kept by unx_set_font
pub static FONT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

/// build version + short git hash stamped at compile time by build.rs
pub const VERSION_LABEL: &str = env!("UNX_VERSION_LABEL");

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
const PAD_R: u32 = 1 << 8;
const PAD_L: u32 = 1 << 9;
const PAD_ZL: u32 = 1 << 14;
const PAD_ZR: u32 = 1 << 15;
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
enum Tab {
    Video,
    Controls,
    System,
}

const TABS: [Tab; 3] = [Tab::Video, Tab::Controls, Tab::System];
const MAX_TAB_ROWS: usize = 8;

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Tab::Video => "Video",
            Tab::Controls => "Controls",
            Tab::System => "Save & System",
        }
    }

    fn rows(self) -> &'static [Row] {
        match self {
            Tab::Video => VIDEO_ROWS,
            Tab::Controls => CONTROLS_ROWS,
            Tab::System => SYSTEM_ROWS,
        }
    }

    fn step(self, delta: i32) -> Tab {
        let idx = TABS.iter().position(|&t| t == self).unwrap_or(0) as i32;
        TABS[(idx + delta).rem_euclid(TABS.len() as i32) as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Row {
    Editing,
    Preset,
    BattleScale,
    MenuScale,
    Smooth,
    DynamicRes,
    PowerLabel,
    RightStick,
    Gyro,
    GyroSpeed,
    AimSpeed,
    InvertY,
    LeftStick,
    Modifier,
    Minus,
    Update,
    UpdateChannel,
    ImportSave,
    ExportSave,
    Resume,
    Quit,
}

const VIDEO_ROWS: &[Row] = &[
    Row::Editing,
    Row::Preset,
    Row::BattleScale,
    Row::MenuScale,
    Row::Smooth,
    Row::DynamicRes,
    Row::PowerLabel,
];

const CONTROLS_ROWS: &[Row] = &[
    Row::RightStick,
    Row::Gyro,
    Row::GyroSpeed,
    Row::AimSpeed,
    Row::InvertY,
    Row::LeftStick,
    Row::Modifier,
    Row::Minus,
];

const SYSTEM_ROWS: &[Row] = &[
    Row::Update,
    Row::UpdateChannel,
    Row::ImportSave,
    Row::ExportSave,
    Row::Resume,
    Row::Quit,
];

const ALL_ROWS: [Row; 21] = [
    Row::Editing,
    Row::Preset,
    Row::BattleScale,
    Row::MenuScale,
    Row::Smooth,
    Row::DynamicRes,
    Row::PowerLabel,
    Row::RightStick,
    Row::Gyro,
    Row::GyroSpeed,
    Row::AimSpeed,
    Row::InvertY,
    Row::LeftStick,
    Row::Modifier,
    Row::Minus,
    Row::Update,
    Row::UpdateChannel,
    Row::ImportSave,
    Row::ExportSave,
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
            Row::DynamicRes => "Dynamic resolution",
            Row::PowerLabel => "Power label",
            Row::RightStick => "Right stick aim",
            Row::Gyro => "Gyro aim",
            Row::GyroSpeed => "Gyro speed",
            Row::AimSpeed => "Stick aim speed",
            Row::InvertY => "Invert aim Y",
            Row::LeftStick => "Left stick aim",
            Row::Modifier => "Aim button",
            Row::Minus => "Minus button",
            Row::Update => "Software update",
            Row::UpdateChannel => "Update channel",
            Row::ImportSave => "Import 3DS save",
            Row::ExportSave => "Export save backup",
            Row::Resume => "Resume game",
            Row::Quit => "Quit UprisingNX",
        }
    }

    fn description(self, docked: bool) -> &'static str {
        match self {
            Row::Editing => "Which video settings the rows below change: handheld or docked. Each has its own preset and resolutions, and the other one takes over when you dock or undock. Aim settings are shared.",
            Row::Preset => {
                if docked {
                    "Performance: 3x in battles, 4x elsewhere, smooth mode on. Quality: 4x everywhere at the game's own 30 fps. Custom: the values below."
                } else {
                    "Performance: 1x in battles, 2x elsewhere, smooth mode on. Quality: 2x everywhere at the game's own 30 fps. Custom: the values below."
                }
            }
            Row::BattleScale => "How sharp battles are drawn, in multiples of the 3DS screen. Higher looks better but can drop below 30 fps or turn smooth mode off. Changing it makes the preset Custom.",
            Row::MenuScale => "How sharp everything outside battles is drawn: menus, the hub and cutscenes. Changing it makes the preset Custom.",
            Row::Smooth => "Draws an in-between frame at 1 step lower resolution for 60 fps. Briefly pauses and resumes automatically if the GPU falls behind. Changing it makes the preset Custom.",
            Row::DynamicRes => "Off (default): keeps your chosen battle resolution fixed; only in-between 60 fps frames render lower. On: tries dropping real battle resolution 1 step before pausing smooth mode.",
            Row::PowerLabel => "Where the selected power's name appears on foot. Bottom: centered at the bottom in crisp white text. Above bar: in a bubble over the power icons. Off: icons only.",
            Row::RightStick => "Aims the reticle in the air and turns the camera on foot with the right stick, in place of the touch screen.",
            Row::Gyro => "Fine aiming by tilting the console or controller. Press or hold R / L3 on foot to Z-target the nearest enemy (or center behind Pit), while pausing gyro to reposition your hands.",
            Row::GyroSpeed => "How far the reticle moves for a tilt. The game's own Controls options scale it further.",
            Row::AimSpeed => "How fast the right stick aims and turns the camera.",
            Row::InvertY => "Flips up and down for aiming and the camera.",
            Row::LeftStick => "Off: the left stick only moves Pit. 1: in the air it also aims, and on foot holding the aim button turns the camera with it. 2: as 1, but in the air it aims only while the aim button is held.",
            Row::Modifier => "The button held with the left stick to aim or turn the camera (Left stick aim 1 or 2). Fire stays on L/ZL.",
            Row::Minus => "Settings menu: Minus alone opens and closes this menu (Plus still pauses; the game's Select only pauses in battles, like Plus). 3DS Select: Minus goes to the game, and right stick click + Minus opens this menu.",
            Row::Update => "Checks GitHub Releases for a newer UprisingNX Switch build, downloads and verifies uprising_nx.nro in the background, then swaps and relaunches on restart.",
            Row::UpdateChannel => "Stable (default): only checks full releases. Pre-release: also picks up release candidates (e.g. v0.1.1-rc.1) for early testing.",
            Row::ImportSave => "Imports decrypted 3DS saves (00.SAV, 01.SAV...) from sdmc:/switch/uprisingnx/save/import/. Automatically backs up your current Switch save first. Restart or return to Title Screen to load.",
            Row::ExportSave => "Copies your current Switch save files to sdmc:/switch/uprisingnx/save/export/ for use with Checkpoint, JKSM, Citra, or Azahar.",
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
    Update,
    UpdateChannel,
    ImportSave,
    ExportSave,
    Quit,
}

/// refreshes the game sees no input for after the menu closes
const QUIET_AFTER_CLOSE: u32 = 6;

/// how long a hotkey's notice stays, and fades over its end
const TOAST_SECONDS: f32 = 2.5;
const TOAST_FADE: f32 = 0.5;

struct WingSprite {
    id: u64,
    fresh: Option<Vec<u8>>,
}

const WING_W: u32 = 640;
const WING_H: u32 = 520;

type CubicSeg = ([f32; 2], [f32; 2], [f32; 2], [f32; 2]);

// Unified concave-arch soaring angelic wing SVG (640x520 viewBox):
// Concave sweeping upper crest rising steeply to a razor-sharp skyward primary tip,
// with 6 stepped upward-hooking crescent feathers and a smooth rounded base.
const WING_BLADES: &[&[CubicSeg]] = &[&[
    // Concave sweeping top leading edge -> soaring Primary Tip #1
    ([74.0, 218.0], [196.0, 232.0], [382.0, 184.0], [558.0, 18.0]),
    // Feather #1 trailing edge -> deep tapered Notch #1
    ([558.0, 18.0], [576.0, 98.0], [512.0, 174.0], [402.0, 224.0]),
    // Notch #1 -> upward-hooking Primary Tip #2
    ([402.0, 224.0], [464.0, 200.0], [522.0, 174.0], [566.0, 134.0]),
    // Feather #2 trailing edge -> deep tapered Notch #2
    ([566.0, 134.0], [560.0, 212.0], [488.0, 266.0], [380.0, 294.0]),
    // Notch #2 -> upward-hooking Primary Tip #3
    ([380.0, 294.0], [438.0, 282.0], [488.0, 268.0], [524.0, 242.0]),
    // Feather #3 trailing edge -> deep tapered Notch #3
    ([524.0, 242.0], [498.0, 304.0], [424.0, 338.0], [336.0, 344.0]),
    // Notch #3 -> Primary Tip #4
    ([336.0, 344.0], [388.0, 342.0], [430.0, 336.0], [460.0, 320.0]),
    // Feather #4 trailing edge -> Notch #4
    ([460.0, 320.0], [428.0, 368.0], [362.0, 388.0], [288.0, 384.0]),
    // Notch #4 -> Primary Tip #5
    ([288.0, 384.0], [334.0, 388.0], [370.0, 386.0], [396.0, 376.0]),
    // Feather #5 trailing edge -> Notch #5
    ([396.0, 376.0], [358.0, 412.0], [298.0, 422.0], [232.0, 412.0]),
    // Notch #5 -> lower trailing plume Tip #6
    ([232.0, 412.0], [272.0, 424.0], [304.0, 428.0], [328.0, 424.0]),
    // Feather #6 lower curve sweeping smoothly into rounded shoulder base
    ([328.0, 424.0], [274.0, 456.0], [176.0, 454.0], [104.0, 408.0]),
    ([104.0, 408.0], [40.0, 366.0], [26.0, 272.0], [74.0, 218.0]),
]];

/// Rasterizes the cubic-Bezier angelic wing SVG paths into a crisp 4x-supersampled
/// premultiplied white-alpha RGBA texture (`WING_W` x `WING_H`).
fn rasterize_wing_svg() -> Vec<u8> {
    let w = WING_W as usize;
    let h = WING_H as usize;
    let mut coverage = vec![0u16; w * h];
    const STEPS: usize = 32;
    const SUB_Y: usize = 4;

    let mut edges: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(256);
    let mut xs: Vec<f32> = Vec::with_capacity(32);

    for &blade in WING_BLADES {
        edges.clear();
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for &(p0, p1, p2, p3) in blade {
            let mut prev = p0;
            for step in 1..=STEPS {
                let t = step as f32 / STEPS as f32;
                let u = 1.0 - t;
                let cur = [
                    u * u * u * p0[0] + 3.0 * u * u * t * p1[0] + 3.0 * u * t * t * p2[0] + t * t * t * p3[0],
                    u * u * u * p0[1] + 3.0 * u * u * t * p1[1] + 3.0 * u * t * t * p2[1] + t * t * t * p3[1],
                ];
                if (cur[1] - prev[1]).abs() > 1e-5 {
                    let (y0, x0, y1, x1) = if prev[1] < cur[1] {
                        (prev[1], prev[0], cur[1], cur[0])
                    } else {
                        (cur[1], cur[0], prev[1], prev[0])
                    };
                    min_y = min_y.min(y0);
                    max_y = max_y.max(y1);
                    edges.push((y0, x0, y1, x1));
                }
                prev = cur;
            }
        }
        let py_start = (min_y.floor() as isize).clamp(0, h as isize) as usize;
        let py_end = (max_y.ceil() as isize).clamp(0, h as isize) as usize;
        for py in py_start..py_end {
            let row = py * w;
            for sy in 0..SUB_Y {
                let y = py as f32 + (sy as f32 + 0.5) / SUB_Y as f32;
                xs.clear();
                for &(y0, x0, y1, x1) in &edges {
                    if y >= y0 && y < y1 {
                        let frac = (y - y0) / (y1 - y0);
                        xs.push(x0 + (x1 - x0) * frac);
                    }
                }
                if xs.len() < 2 {
                    continue;
                }
                xs.sort_unstable_by(|a, b| a.total_cmp(b));
                for pair in xs.chunks_exact(2) {
                    let x0 = pair[0].clamp(0.0, w as f32);
                    let x1 = pair[1].clamp(0.0, w as f32);
                    if x1 <= x0 {
                        continue;
                    }
                    let ix0 = x0.floor() as usize;
                    let ix1 = (x1.ceil() as usize).min(w);
                    for px in ix0..ix1 {
                        let cov = ((px as f32 + 1.0).min(x1) - (px as f32).max(x0)).clamp(0.0, 1.0);
                        let add = (cov * 255.0 + 0.5) as u16;
                        coverage[row + px] = (coverage[row + px] + add).min(255 * SUB_Y as u16);
                    }
                }
            }
        }
    }

    let mut pixels = vec![0u8; w * h * 4];
    for (i, &c) in coverage.iter().enumerate() {
        let a = ((c as u32 + (SUB_Y as u32 / 2)) / SUB_Y as u32).min(255) as u8;
        let at = i * 4;
        pixels[at..at + 4].copy_from_slice(&[a, a, a, a]);
    }
    pixels
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn wing_grad_color(u: f32, v: f32, phase: f32, is_error: bool) -> [u8; 4] {
    let d = 0.62 * u + 0.38 * (1.0 - v);
    let tau = std::f32::consts::TAU;
    let wave = 0.5 + 0.5 * ((d * 0.85 - phase) * tau).sin();
    let s = ((d * 1.10 - phase) * tau).sin().max(0.0);
    let sheen = s * s * s * s;
    let (r, g, b, alpha) = if !is_error {
        let r = lerp_f32(lerp_f32(64.0, 250.0, wave), 255.0, sheen * 0.65);
        let g = lerp_f32(lerp_f32(148.0, 206.0, wave), 246.0, sheen * 0.65);
        let b = lerp_f32(lerp_f32(244.0, 92.0, wave), 214.0, sheen * 0.65);
        let a = (lerp_f32(0.16, 0.34, wave) + sheen * 0.16).min(1.0);
        (r, g, b, a)
    } else {
        let r = lerp_f32(lerp_f32(175.0, 245.0, wave), 255.0, sheen * 0.50);
        let g = lerp_f32(lerp_f32(48.0, 128.0, wave), 204.0, sheen * 0.50);
        let b = lerp_f32(lerp_f32(82.0, 92.0, wave), 178.0, sheen * 0.50);
        let a = (lerp_f32(0.15, 0.30, wave) + sheen * 0.14).min(1.0);
        (r, g, b, a)
    };
    color(r as u8, g as u8, b as u8, (alpha * 255.0) as u8)
}

pub struct Menu {
    pub open: bool,
    tab: Tab,
    cursor: usize,
    /// Quit was pressed once; a second press quits
    quit_armed: bool,
    /// Import save was pressed once; a second press overwrites (after backup)
    import_armed: bool,
    /// Cached save counts refreshed on menu open and after import/export
    pending_saves: usize,
    live_saves: usize,
    /// Snapshot of the GitHub Releases auto-updater state
    update: UpdateView,
    /// Inline status message shown on the Save & System tab (`(text, is_error)`)
    status: Option<(String, bool)>,
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
    wing: Option<WingSprite>,
    anim_start: std::time::Instant,
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
            tab: Tab::Video,
            cursor: 0,
            quit_armed: false,
            import_armed: false,
            pending_saves: 0,
            live_saves: 0,
            update: UpdateView {
                value: "Check now".to_owned(),
                tone: UpdateTone::Neutral,
                heading: format!("UprisingNX {VERSION_LABEL}"),
                detail: "Checks GitHub Releases for a newer Switch build. Press A to check now.".to_owned(),
                progress: None,
            },
            status: None,
            editing_docked: false,
            blocking: false,
            quiet: 0,
            last_buttons: 0,
            held: (0, 0),
            atlas: None,
            wing: None,
            anim_start: std::time::Instant::now(),
            next_id: 0x4D45_4E55_0000,
            toast: None,
        }
    }

    pub fn set_save_counts(&mut self, pending: usize, live: usize) {
        self.pending_saves = pending;
        self.live_saves = live;
    }

    pub fn set_update_view(&mut self, view: UpdateView) {
        self.update = view;
    }

    pub fn set_status(&mut self, text: String, is_error: bool) {
        self.status = Some((text, is_error));
    }

    pub fn toggle(&mut self, buttons: u32, docked: bool) {
        self.open = !self.open;
        // what is held as it opens or closes is not a press
        self.last_buttons = buttons;
        self.held = (0, 0);
        self.quit_armed = false;
        self.import_armed = false;
        self.status = None;
        self.cursor = self.cursor.min(self.tab.rows().len().saturating_sub(1));
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
        if pressed & (PAD_L | PAD_ZL) != 0 {
            self.tab = self.tab.step(-1);
            self.cursor = 0;
            self.quit_armed = false;
            self.import_armed = false;
            if self.tab != Tab::System {
                self.status = None;
            }
            return None;
        }
        if pressed & (PAD_R | PAD_ZR) != 0 {
            self.tab = self.tab.step(1);
            self.cursor = 0;
            self.quit_armed = false;
            self.import_armed = false;
            if self.tab != Tab::System {
                self.status = None;
            }
            return None;
        }
        let rows = self.tab.rows();
        if self.cursor >= rows.len() {
            self.cursor = 0;
        }
        if step & (PAD_UP | PAD_DOWN) != 0 {
            self.quit_armed = false;
            self.import_armed = false;
        }
        if step & PAD_UP != 0 {
            self.cursor = (self.cursor + rows.len() - 1) % rows.len();
        }
        if step & PAD_DOWN != 0 {
            self.cursor = (self.cursor + 1) % rows.len();
        }
        let a = pressed & PAD_A != 0;
        let delta: i32 = if step & PAD_RIGHT != 0 || a {
            1
        } else if step & PAD_LEFT != 0 {
            -1
        } else {
            return None;
        };
        let row = rows[self.cursor];
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
            Row::DynamicRes => {
                let value = if settings.smooth_scale_drop() { "0" } else { "1" };
                set_control(settings, "smooth_scale_drop", value.into());
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
            Row::Update => {
                if !a {
                    return None;
                }
                Changed::Update
            }
            Row::UpdateChannel => {
                let value = if settings.update_prerelease() { "0" } else { "1" };
                set_control(settings, "update_prerelease", value.into());
                Changed::UpdateChannel
            }
            Row::ImportSave => {
                if !a {
                    return None;
                }
                if self.pending_saves > 0 && !self.import_armed {
                    self.import_armed = true;
                    return None;
                }
                self.import_armed = false;
                Changed::ImportSave
            }
            Row::ExportSave => {
                if !a {
                    return None;
                }
                Changed::ExportSave
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
        let px = (height as f32 / 720.0 * 22.0).round().max(14.0);
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

    /// whether the menu, a non-expired toast, or the title-screen version
    /// watermark needs drawing this refresh.
    pub fn has_overlay(&mut self, show_version: bool) -> bool {
        if self.toast.as_ref().is_some_and(|(_, at)| at.elapsed().as_secs_f32() > TOAST_SECONDS) {
            self.toast = None;
        }
        self.open || self.toast.is_some() || show_version
    }

    /// the menu, a hotkey's notice, or the title-screen version watermark over
    /// the game, in a window `width` x `height` pixels
    pub fn overlay(
        &mut self,
        settings: &Settings,
        docked: bool,
        now: (u32, bool),
        width: u32,
        height: u32,
        show_version: bool,
        update_badge: Option<&str>,
    ) -> Overlay {
        let mut overlay = Overlay::default();
        // Release the bootstrap wing texture once normal gameplay takes over.
        if let Some(wing) = self.wing.take() {
            overlay.free.push(wing.id);
        }
        let toast = self.toast.as_ref().map(|(text, at)| (text.clone(), at.elapsed().as_secs_f32()));
        if toast.as_ref().is_some_and(|(_, age)| *age > TOAST_SECONDS) {
            self.toast = None;
        }
        if !self.open && self.toast.is_none() && !show_version {
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
            if show_version {
                let size = 0.78;
                let tw = atlas.width(VERSION_LABEL) * size;
                let th = atlas.px * size;
                let pad_x = 18.0 * k;
                let pad_y = 14.0 * k;
                let x = (width as f32 - tw - pad_x).max(0.0);
                let y = (height as f32 - th - pad_y).max(0.0);
                let sh = (1.5 * k).max(1.0);
                painter.text(x + sh, y + sh, VERSION_LABEL, size, color(0, 0, 0, 175));
                painter.text(x, y, VERSION_LABEL, size, color(245, 248, 255, 220));
                if let Some(badge) = update_badge {
                    let bw = atlas.width(badge) * size;
                    let bx = (width as f32 - bw - pad_x).max(0.0);
                    let by = (y - th - 5.0 * k).max(0.0);
                    painter.text(bx + sh, by + sh, badge, size, color(0, 0, 0, 205));
                    painter.text(bx, by, badge, size, color(255, 226, 72, 248));
                }
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
                Row::DynamicRes => flag(settings.smooth_scale_drop()),
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
                Row::Update => self.update.value.clone(),
                Row::UpdateChannel => {
                    if settings.update_prerelease() {
                        "Pre-release".into()
                    } else {
                        "Stable".into()
                    }
                }
                Row::ImportSave => {
                    if self.import_armed {
                        "Press A again".into()
                    } else if self.pending_saves > 0 {
                        let s = if self.pending_saves == 1 { "" } else { "s" };
                        format!("{} file{s} ready", self.pending_saves)
                    } else {
                        "No files in import/".into()
                    }
                }
                Row::ExportSave => {
                    if self.live_saves > 0 {
                        let s = if self.live_saves == 1 { "" } else { "s" };
                        format!("{} file{s}", self.live_saves)
                    } else {
                        "No save yet".into()
                    }
                }
                Row::Resume => String::new(),
                Row::Quit => if self.quit_armed { "Press A again".into() } else { String::new() },
            }
        };

        let k = height as f32 / 720.0;
        let line = (atlas.px * 1.40).round();
        let small = 0.78;
        let small_line = (line * 0.80).round();
        let panel_w = 780.0 * k;
        let margin = 28.0 * k;
        let inner_w = panel_w - margin * 2.0;

        let rows = self.tab.rows();
        let cursor = self.cursor.min(rows.len().saturating_sub(1));
        let row = rows[cursor];
        let description = wrap(atlas, row.description(docked), small, inner_w);
        // Reserve height for the longest description across all tabs so the panel never jumps.
        let described = ALL_ROWS
            .iter()
            .map(|r| wrap(atlas, r.description(docked), small, inner_w).len())
            .max()
            .unwrap_or(1);

        let header_h = line * 1.38;
        let tab_h = line * 1.20;
        let tab_gap = line * 0.45;
        let rows_h = line * MAX_TAB_ROWS as f32;
        let desc_h = line * 0.60 + small_line * described as f32 + line * 0.25;
        let footer_h = line * 1.25;
        let panel_h = header_h + tab_h + tab_gap + rows_h + desc_h + footer_h;
        let (left, top) = ((width as f32 - panel_w) / 2.0, ((height as f32 - panel_h) / 2.0).max(0.0));

        // Dim the game, then draw the panel frame.
        painter.rect([0.0, 0.0, width as f32, height as f32], color(0, 0, 0, 128));
        painter.rect([left, top, left + panel_w, top + panel_h], color(16, 20, 40, 242));
        let bw = (1.5 * k).max(1.0);
        let border_c = color(68, 86, 138, 175);
        painter.rect([left, top, left + bw, top + panel_h], border_c);
        painter.rect([left + panel_w - bw, top, left + panel_w, top + panel_h], border_c);
        painter.rect([left, top + panel_h - bw, left + panel_w, top + panel_h], border_c);
        painter.rect([left, top, left + panel_w, top + 3.0 * k], color(250, 200, 80, 255));

        // Header: title on left, docked/handheld indicator on right.
        let title = format!("UprisingNX {}", VERSION_LABEL);
        painter.text(left + margin, top + line * 0.36, &title, 1.0, color(250, 200, 80, 255));
        let mode_badge = format!("Playing: {}", if docked_now { "Docked" } else { "Handheld" });
        let badge_w = atlas.width(&mode_badge) * small;
        painter.text(
            left + panel_w - margin - badge_w,
            top + line * 0.46,
            &mode_badge,
            small,
            color(165, 180, 215, 255),
        );

        // Tab bar: [L]  Video | Controls | Save & System  [R]
        let tabs_top = top + header_h;
        let tabs_bot = tabs_top + tab_h;
        painter.rect([left + margin, tabs_top, left + panel_w - margin, tabs_bot], color(10, 14, 30, 230));
        let shoulder_w = 48.0 * k;
        let tab_area_w = inner_w - shoulder_w * 2.0;
        let tab_w = tab_area_w / TABS.len() as f32;
        let tab_text_scale = 0.88;
        let tab_text_y = tabs_top + (tab_h - atlas.px * tab_text_scale * 1.15) / 2.0;

        let l_hint = "[L]";
        let r_hint = "[R]";
        let l_w = atlas.width(l_hint) * small;
        let r_w = atlas.width(r_hint) * small;
        let hint_y = tabs_top + (tab_h - atlas.px * small * 1.15) / 2.0;
        painter.text(left + margin + (shoulder_w - l_w) / 2.0, hint_y, l_hint, small, color(135, 148, 180, 255));
        painter.text(
            left + panel_w - margin - shoulder_w + (shoulder_w - r_w) / 2.0,
            hint_y,
            r_hint,
            small,
            color(135, 148, 180, 255),
        );

        for (i, &tab) in TABS.iter().enumerate() {
            let tx0 = left + margin + shoulder_w + tab_w * i as f32;
            let tx1 = tx0 + tab_w;
            let active = tab == self.tab;
            if active {
                painter.rect([tx0 + 2.0 * k, tabs_top + 2.0 * k, tx1 - 2.0 * k, tabs_bot], color(44, 60, 114, 255));
                painter.rect([tx0 + 2.0 * k, tabs_bot - 2.5 * k, tx1 - 2.0 * k, tabs_bot], color(250, 200, 80, 255));
            }
            let label = tab.label();
            let lw = atlas.width(label) * tab_text_scale;
            let tc = if active { color(255, 255, 255, 255) } else { color(152, 162, 194, 255) };
            painter.text(tx0 + (tab_w - lw) / 2.0, tab_text_y, label, tab_text_scale, tc);
        }
        painter.rect([left + margin, tabs_bot, left + panel_w - margin, tabs_bot + 1.0 * k], color(64, 78, 120, 200));

        // Rows for active tab (fixed MAX_TAB_ROWS body height).
        let rows_top = tabs_bot + tab_gap;
        for (index, row) in rows.iter().enumerate() {
            let y = rows_top + line * index as f32;
            let chosen = index == cursor;
            if chosen {
                painter.rect(
                    [left + 14.0 * k, y - line * 0.14, left + panel_w - 14.0 * k, y + line * 0.84],
                    color(64, 84, 152, 255),
                );
                painter.rect(
                    [left + 14.0 * k, y - line * 0.14, left + 17.5 * k, y + line * 0.84],
                    color(250, 200, 80, 255),
                );
            }
            let armed = (*row == Row::Quit && self.quit_armed) || (*row == Row::ImportSave && self.import_armed);
            let text = if armed {
                color(255, 168, 120, 255)
            } else if chosen {
                color(255, 255, 255, 255)
            } else {
                color(202, 208, 224, 255)
            };
            painter.text(left + margin, y, row.label(), 1.0, text);
            let val = value(*row);
            if !val.is_empty() {
                let cycles = !matches!(row, Row::Update | Row::ImportSave | Row::ExportSave | Row::Resume | Row::Quit);
                let shown = if chosen && cycles { format!("< {val} >") } else { val };
                let val_color = if armed {
                    color(255, 190, 110, 255)
                } else if *row == Row::Update && !chosen {
                    match self.update.tone {
                        UpdateTone::Accent => color(250, 208, 92, 255),
                        UpdateTone::Success => color(140, 232, 168, 255),
                        UpdateTone::Error => color(245, 130, 130, 255),
                        UpdateTone::Neutral => text,
                    }
                } else if *row == Row::ImportSave && self.pending_saves > 0 && !chosen {
                    color(140, 232, 168, 255)
                } else {
                    text
                };
                let x = left + panel_w - margin - atlas.width(&shown);
                painter.text(x, y, &shown, 1.0, val_color);
            }
        }

        // Inline status / quick-tip box in the lower body area of Save & System.
        if self.tab == Tab::System {
            let box_top = rows_top + line * (rows.len() as f32 + 0.28);
            let box_bot = rows_top + rows_h - line * 0.10;
            let box_pad = 14.0 * k;
            let show_updater = matches!(row, Row::Update | Row::UpdateChannel) || self.update.progress.is_some();
            let (bg, accent, heading, body, progress) = if show_updater {
                let (bg, accent) = match self.update.tone {
                    UpdateTone::Success => (color(20, 48, 42, 230), color(110, 228, 158, 255)),
                    UpdateTone::Accent => (color(36, 36, 56, 230), color(250, 204, 88, 255)),
                    UpdateTone::Error => (color(52, 22, 34, 230), color(240, 105, 105, 255)),
                    UpdateTone::Neutral => (color(22, 30, 56, 210), color(110, 152, 225, 255)),
                };
                (
                    bg,
                    accent,
                    self.update.heading.as_str(),
                    self.update.detail.as_str(),
                    self.update.progress,
                )
            } else {
                match &self.status {
                    Some((msg, true)) => (
                        color(52, 22, 34, 230),
                        color(240, 105, 105, 255),
                        "Save Action Notice",
                        msg.as_str(),
                        None,
                    ),
                    Some((msg, false)) => (
                        color(20, 48, 42, 230),
                        color(110, 228, 158, 255),
                        "Save Action Complete",
                        msg.as_str(),
                        None,
                    ),
                    None => (
                        color(22, 30, 56, 210),
                        color(110, 152, 225, 255),
                        "Drop-In 3DS Save Folder",
                        "Copy decrypted 00.SAV / 01.SAV files from Checkpoint, JKSM, Citra, or Azahar into sdmc:/switch/uprisingnx/save/import/ to auto-import on launch, or press A above.",
                        None,
                    ),
                }
            };
            painter.rect([left + margin, box_top, left + panel_w - margin, box_bot], bg);
            painter.rect([left + margin, box_top, left + margin + 3.0 * k, box_bot], accent);
            painter.text(left + margin + box_pad, box_top + 8.0 * k, heading, small, accent);
            let max_lines = if progress.is_some() { 1 } else { 2 };
            let lines = wrap(atlas, body, small, inner_w - box_pad * 2.0);
            for (i, ln) in lines.iter().take(max_lines).enumerate() {
                let ly = box_top + 8.0 * k + small_line * (1.02 + i as f32);
                if ly + small_line <= box_bot + 4.0 * k {
                    painter.text(left + margin + box_pad, ly, ln, small, color(215, 222, 238, 255));
                }
            }
            if let Some(frac) = progress {
                let bar_h = 6.0 * k;
                let bar_y0 = box_bot - bar_h - 8.0 * k;
                let bar_x0 = left + margin + box_pad;
                let bar_x1 = left + panel_w - margin - box_pad;
                painter.rect([bar_x0, bar_y0, bar_x1, bar_y0 + bar_h], color(12, 16, 32, 240));
                let fill_x1 = bar_x0 + (bar_x1 - bar_x0) * frac.clamp(0.0, 1.0);
                if fill_x1 > bar_x0 + 1.0 {
                    painter.rect([bar_x0, bar_y0, fill_x1, bar_y0 + bar_h], accent);
                }
            }
        }

        // Description box under a horizontal rule.
        let rule = rows_top + rows_h + line * 0.25;
        painter.rect([left + margin, rule, left + panel_w - margin, rule + 1.5 * k], color(86, 98, 134, 255));
        for (index, text) in description.iter().enumerate() {
            let y = rule + line * 0.30 + small_line * index as f32;
            painter.text(left + margin, y, text, small, color(182, 192, 218, 255));
        }

        // Footer bar: live video state on left, navigation hints on right.
        let (now_scale, now_smooth) = now;
        let left_help = format!("Running {now_scale}x, smooth {}", if now_smooth { "on" } else { "off" });
        let right_help = "L/R: Tab   A/Left/Right: Change   B: Close";
        let fy = top + panel_h - line * 0.92;
        painter.text(left + margin, fy, &left_help, small, color(135, 146, 174, 255));
        let rw = atlas.width(right_help) * small;
        painter.text(left + panel_w - margin - rw, fy, right_help, small, color(152, 164, 194, 255));
    }

    /// Full-screen bootstrap / setup / progress card drawn before gameplay starts,
    /// layered over a soaring angelic wing SVG backdrop with an animated gradient fill.
    pub fn status_overlay(
        &mut self,
        heading: &str,
        paragraphs: &[&str],
        progress: Option<(f32, &str)>,
        footer: Option<&str>,
        is_error: bool,
        width: u32,
        height: u32,
    ) -> Overlay {
        let mut overlay = Overlay::default();
        if !self.ready(height, &mut overlay) {
            return overlay;
        }
        if self.wing.is_none() {
            self.next_id += 1;
            self.wing = Some(WingSprite { id: self.next_id, fresh: Some(rasterize_wing_svg()) });
        }
        let wing_id = {
            let wing = self.wing.as_mut().unwrap();
            if let Some(pixels) = wing.fresh.take() {
                overlay.textures.push(OverlayTexture {
                    id: wing.id,
                    offset: None,
                    size: [WING_W, WING_H],
                    pixels,
                    linear: true,
                });
            }
            wing.id
        };

        let phase = (self.anim_start.elapsed().as_secs_f32() * 0.22).fract();
        let atlas = self.atlas.as_ref().unwrap();
        let w_f = width as f32;
        let h_f = height as f32;
        let k = h_f / 720.0;

        // Mesh 1: Smooth vertical twilight sky gradient backdrop
        {
            let uv = atlas.uv(COUNT, atlas.cell[0] / 2.0 - 0.5);
            let top_c = color(7, 11, 26, 255);
            let bot_c = color(15, 22, 48, 255);
            let vertices = vec![
                OverlayVertex { position: [0.0, 0.0], uv: [uv[0], uv[1]], color: top_c },
                OverlayVertex { position: [w_f, 0.0], uv: [uv[2], uv[1]], color: top_c },
                OverlayVertex { position: [w_f, h_f], uv: [uv[2], uv[3]], color: bot_c },
                OverlayVertex { position: [0.0, h_f], uv: [uv[0], uv[3]], color: bot_c },
            ];
            overlay.meshes.push(OverlayMesh {
                texture: atlas.id,
                clip: [0, 0, width, height],
                vertices,
                indices: vec![0, 1, 2, 0, 2, 3],
            });
        }

        // Mesh 2: Logo-inspired concave-arch soaring wing SVG with animated celestial gradient
        {
            const COLS: usize = 20;
            const ROWS: usize = 16;
            let ww = 960.0 * k;
            let wh = 780.0 * k;
            let wx0 = w_f * 0.56 - ww * 0.42;
            let wy0 = h_f * 0.50 - wh * 0.52;
            let mut vertices = Vec::with_capacity((COLS + 1) * (ROWS + 1));
            for ry in 0..=ROWS {
                let v = ry as f32 / ROWS as f32;
                let y = wy0 + wh * v;
                for rx in 0..=COLS {
                    let u = rx as f32 / COLS as f32;
                    let x = wx0 + ww * u;
                    vertices.push(OverlayVertex {
                        position: [x, y],
                        uv: [u, v],
                        color: wing_grad_color(u, v, phase, is_error),
                    });
                }
            }
            let stride = (COLS + 1) as u32;
            let mut indices = Vec::with_capacity(COLS * ROWS * 6);
            for ry in 0..ROWS as u32 {
                for rx in 0..COLS as u32 {
                    let i0 = ry * stride + rx;
                    let i1 = i0 + 1;
                    let i2 = i0 + stride + 1;
                    let i3 = i0 + stride;
                    indices.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
                }
            }
            overlay.meshes.push(OverlayMesh {
                texture: wing_id,
                clip: [0, 0, width, height],
                vertices,
                indices,
            });
        }

        // Mesh 3: Frosted glass card, hairline border, animated progress bar, and crisp typography
        let mut painter = Painter { atlas, vertices: Vec::new(), indices: Vec::new() };
        let line = (atlas.px * 1.38).round();
        let body_scale = 0.88;
        let body_line = line * 0.90;
        let panel_w = 820.0 * k;
        let margin = 34.0 * k;
        let max_text_w = panel_w - margin * 2.0;

        let mut wrapped_blocks: Vec<Vec<String>> = Vec::with_capacity(paragraphs.len());
        let mut total_body_lines = 0usize;
        for p in paragraphs {
            if p.is_empty() {
                wrapped_blocks.push(Vec::new());
                total_body_lines += 1;
            } else {
                let w = wrap(atlas, p, body_scale, max_text_w);
                total_body_lines += w.len().max(1);
                wrapped_blocks.push(w);
            }
        }

        let progress_h = if progress.is_some() { line * 2.4 } else { 0.0 };
        let footer_h = if footer.is_some() { line * 1.5 } else { 0.0 };
        let panel_h = line * 2.2 + body_line * (total_body_lines as f32) + progress_h + footer_h + margin * 0.95;
        let left = (w_f - panel_w) * 0.5;
        let top = ((h_f - panel_h) * 0.5).max(20.0 * k);

        // Multi-layer soft drop shadow + translucent frosted celestial card
        for &(pad_s, alpha_s) in &[(18.0 * k, 22u8), (10.0 * k, 36), (4.0 * k, 52)] {
            painter.rect(
                [left - pad_s, top - pad_s * 0.5, left + panel_w + pad_s, top + panel_h + pad_s],
                color(3, 5, 14, alpha_s),
            );
        }
        painter.rect([left, top, left + panel_w, top + panel_h], color(12, 17, 36, 192));
        let accent = if is_error { color(242, 102, 102, 255) } else { color(250, 206, 88, 255) };
        let border = if is_error { color(242, 102, 102, 92) } else { color(250, 206, 88, 92) };
        let bw = (1.0 * k).max(1.0);
        painter.rect([left, top, left + panel_w, top + (3.0 * k).max(2.0)], accent);
        painter.rect([left, top + panel_h - bw, left + panel_w, top + panel_h], border);
        painter.rect([left, top, left + bw, top + panel_h], border);
        painter.rect([left + panel_w - bw, top, left + panel_w, top + panel_h], border);

        let title = format!("UprisingNX {} - {heading}", VERSION_LABEL);
        painter.text(left + margin, top + line * 0.55, &title, 1.05, accent);

        let mut y = top + line * 1.9;
        for block in &wrapped_blocks {
            if block.is_empty() {
                y += body_line * 0.6;
                continue;
            }
            for text in block {
                let c = if text.starts_with("sdmc:/") || text.starts_with("switch/") {
                    color(145, 226, 255, 255)
                } else {
                    color(234, 238, 248, 255)
                };
                painter.text(left + margin, y, text, body_scale, c);
                y += body_line;
            }
            y += body_line * 0.2;
        }

        if let Some((pct, status)) = progress {
            y += line * 0.35;
            let bar_x0 = left + margin;
            let bar_x1 = left + panel_w - margin;
            let bar_h = 22.0 * k;
            painter.rect([bar_x0, y, bar_x1, y + bar_h], color(26, 34, 66, 225));
            let fill_w = (bar_x1 - bar_x0) * pct.clamp(0.0, 1.0);
            if fill_w > 1.0 {
                // Smooth GPU-interpolated animated gold shimmer bar
                const SEGS: usize = 16;
                let white_uv = atlas.uv(COUNT, atlas.cell[0] / 2.0 - 0.5);
                let u_mid = (white_uv[0] + white_uv[2]) * 0.5;
                let v_mid = (white_uv[1] + white_uv[3]) * 0.5;
                let bar_color = |t: f32| -> [u8; 4] {
                    let s = ((t * 1.2 - phase * 1.5) * std::f32::consts::TAU).sin().max(0.0);
                    let sh = s * s * s;
                    let cr = lerp_f32(222.0, 255.0, (t * 0.55 + sh * 0.45).min(1.0)) as u8;
                    let cg = lerp_f32(168.0, 238.0, (t * 0.45 + sh * 0.55).min(1.0)) as u8;
                    let cb = lerp_f32(56.0, 166.0, (sh * 0.75).min(1.0)) as u8;
                    color(cr, cg, cb, 255)
                };
                for i in 0..SEGS {
                    let t0 = i as f32 / SEGS as f32;
                    let t1 = (i + 1) as f32 / SEGS as f32;
                    let x0 = bar_x0 + fill_w * t0;
                    let x1 = bar_x0 + fill_w * t1;
                    let c0 = bar_color(t0);
                    let c1 = bar_color(t1);
                    let base = painter.vertices.len() as u32;
                    painter.vertices.push(OverlayVertex { position: [x0, y], uv: [u_mid, v_mid], color: c0 });
                    painter.vertices.push(OverlayVertex { position: [x1, y], uv: [u_mid, v_mid], color: c1 });
                    painter.vertices.push(OverlayVertex { position: [x1, y + bar_h], uv: [u_mid, v_mid], color: c1 });
                    painter.vertices.push(OverlayVertex { position: [x0, y + bar_h], uv: [u_mid, v_mid], color: c0 });
                    painter.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            let pct_label = format!("{:.0}%", (pct.clamp(0.0, 1.0) * 100.0));
            let px_x = bar_x1 - atlas.width(&pct_label) * 0.85;
            painter.text(bar_x0, y + bar_h + 8.0 * k, status, 0.82, color(198, 210, 236, 255));
            painter.text(px_x, y + bar_h + 8.0 * k, &pct_label, 0.85, accent);
        }

        if let Some(foot) = footer {
            let fy = top + panel_h - line * 1.15;
            painter.text(left + margin, fy, foot, 0.80, color(154, 166, 198, 255));
        }

        let (vertices, indices) = (painter.vertices, painter.indices);
        overlay.meshes.push(OverlayMesh {
            texture: atlas.id,
            clip: [0, 0, width, height],
            vertices,
            indices,
        });
        overlay
    }
}
