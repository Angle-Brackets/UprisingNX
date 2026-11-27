//! Shared Kid Icarus: Uprising (USA 1.0) single-screen 16:9 UI state machine,
//! guest pause-menu inspection, and screen-to-3DS touch coordinate mapping.
//!
//! Used by both the Nintendo Switch frontend (`uprising_nx`) and the desktop PC
//! frontend (`zakuro`) so all 10 merged 16:9 hubs, touch hitboxes, and battle
//! presentation transitions stay in lockstep across platforms.

use std::sync::atomic::Ordering;

use zakuro_core::memory::Memory;
use zakuro_gpu::backend::TOP_WIDTH;
use zakuro_gpu::{Flags, ScreenLayout};

/// Kid Icarus: Uprising (USA, Rev 0 / Rev 1) 3DS Title / Program ID.
pub const KIU_USA_PROGRAM_ID: u64 = 0x0004_0000_0003_0100;

/// Host mailbox fourth word (`HOST_MAILBOX_VADDR + 12`) values written by KIU overrides.
pub const GAME_ON_FOOT: u32 = 1;
pub const GAME_FLYING: u32 = 2;
pub const GAME_VEHICLE: u32 = 3;

/// Reads KIU (USA 1.0) in-battle PauseMenu state from guest RAM.
///
/// Returns `Some(sub)` strictly while the in-battle Pause menu is open
/// (`InfoManager+0xC0` bit 16):
/// - `1, 2, 4` = Main Pause
/// - `3` = Quit Confirmation modal
/// - `5, 6, 7` = Options / Controls / Volume
/// - `9` = Remove Display (photo mode, `info_btl + 0x243C` bit 30)
pub fn read_guest_pause_state(memory: &mut Memory) -> Option<u8> {
    if (memory.peek32(0x00D6_36C0) & 1) == 0 {
        return None;
    }
    if (memory.peek32(0x00EE_3158) & 0x0001_0000) == 0 {
        return None;
    }
    let info_btl = memory.peek32(0x00EE_31D8);
    if (0x0800_0000..0x1800_0000).contains(&info_btl) && (info_btl & 3) == 0 {
        let flags = memory.peek32(info_btl + 0x243C);
        let sub = memory.peek32(info_btl + 0x2440);
        if (flags & 0x4000_0000) != 0 && sub <= 2 {
            return Some(9);
        }
        if (1..=7).contains(&sub) {
            return Some(sub as u8);
        }
    }
    Some(2)
}

/// Collapses raw guest pause sub-state IDs into the presenter's distinct card modes.
pub fn pause_sub_mode(sub: u8) -> u8 {
    match sub {
        3 => 3,
        9 => 9,
        5..=7 => 5,
        _ => 2,
    }
}

/// Maps a pointer/touch position `(x, y)` in host window pixels `(win_w x win_h)`
/// onto the 3DS bottom screen (`320 x 240`), inverting each KIU 16:9 merged
/// hub shader's exact layout transform.
///
/// When `layout == ScreenLayout::TopOnly`, maps a centered 4:3 region inside
/// the top viewport onto `320 x 240` (used for touch-aim fallback and title taps).
#[allow(clippy::too_many_arguments)]
pub fn touch_on_bottom(
    win_w: u32,
    win_h: u32,
    top_width: f32,
    integer: bool,
    layout: ScreenLayout,
    weapon_sub: u8,
    pause_sub: u8,
    x: f32,
    y: f32,
) -> Option<(u16, u16)> {
    let (top_vp, bottom_vp) = zakuro_gpu::layout(win_w, win_h, layout, top_width, integer);

    if layout == ScreenLayout::Pause {
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return None;
        }
        let wx = u * 426.66666;
        let wy = v * 240.0;
        if pause_sub >= 9 {
            // Remove Display photo mode: only the bottom-right `DISPLAY` tab (`bx=222..320, by=207..240`).
            let tab_x = 426.66666 - 88.0;
            let tab_y = 240.0 - 30.0;
            if wx >= tab_x && wy >= tab_y {
                let lx = ((wx - tab_x) / 88.0).clamp(0.0, 1.0);
                let ly = ((wy - tab_y) / 30.0).clamp(0.0, 1.0);
                return Some((
                    (222.0 + lx * 98.0).min(319.0) as u16,
                    (207.0 + ly * 33.0).min(239.0) as u16,
                ));
            }
            return None;
        }
        let (src_x0, src_y0, src_w, src_h, card_w, card_h, card_y0) = if pause_sub == 3 {
            (16.5, 18.5, 287.0, 203.0, 246.0, 174.0, (240.0 - 174.0) * 0.5)
        } else if pause_sub >= 5 {
            (0.0, 0.0, 320.0, 240.0, 224.0, 168.0, 10.0)
        } else {
            (0.0, 42.0, 320.0, 198.0, 256.0, 158.4, 50.0)
        };
        let card_x0 = (426.66666 - card_w) * 0.5;
        let lx = (wx - card_x0) / card_w;
        let ly = (wy - card_y0) / card_h;
        if (0.0..1.0).contains(&lx) && (0.0..1.0).contains(&ly) {
            let bx = (src_x0 + lx * src_w).clamp(0.0, 319.0) as u16;
            let by = (src_y0 + ly * src_h).clamp(0.0, 239.0) as u16;
            return Some((bx, by));
        }
        return None;
    }

    if layout == ScreenLayout::FiendsCauldron {
        if let Some(top) = top_vp {
            let u = (x - top.x) / top.width;
            let v = (y - top.y) / top.height;
            // Relocated bottom-left 16:9 Back button (140/1280, 595/720 on 720p) maps to native (16, 221).
            if (0.0..=(140.0 / 1280.0)).contains(&u) && ((595.0 / 720.0)..=1.0).contains(&v) {
                return Some((16, 221));
            }
        }
    }

    if layout == ScreenLayout::ChapterSelect {
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
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
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
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
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
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
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
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
        let bottom = bottom_vp?;
        let u = (x - bottom.x) / bottom.width;
        let v = (y - bottom.y) / bottom.height;
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

    let (left, top, width, height) = if layout == ScreenLayout::TopOnly {
        let top = top_vp?;
        // Centered 4:3 touch area inside the top viewport (matches (160, 0, 960, 720) at 1280x720).
        let w = (top.height * (4.0 / 3.0)).min(top.width);
        let l = top.x + (top.width - w) * 0.5;
        (l, top.y, w, top.height)
    } else {
        let bottom = bottom_vp?;
        (bottom.x, bottom.y, bottom.width, bottom.height)
    };
    let (u, v) = ((x - left) / width, (y - top) / height);
    ((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)).then(|| ((u * 320.0) as u16, (v * 240.0) as u16))
}

/// Whether a pointer/touch at `(x, y)` in `(win_w x win_h)` is over the
/// Fiend's Cauldron top-bar intensity slider region.
pub fn is_touching_cauldron_slider(
    win_w: u32,
    win_h: u32,
    top_width: f32,
    integer: bool,
    x: f32,
    y: f32,
) -> bool {
    let (Some(top), _) = zakuro_gpu::layout(win_w, win_h, ScreenLayout::FiendsCauldron, top_width, integer) else {
        return false;
    };
    let u = (x - top.x) / top.width;
    let v = (y - top.y) / top.height;
    (0.0..(320.0 / 720.0)).contains(&v) && ((165.0 / 1280.0)..=(1125.0 / 1280.0)).contains(&u)
}

/// Tracks whether the classic bottom screen is shown and handles the user's
/// manual toggle (`R-Stick` click on Switch, `F2` / `R-Stick` on PC).
#[derive(Debug, Clone)]
pub struct BottomToggle {
    pub shown: bool,
    /// True once the game has entered battle at least once (cleared upon returning to frontend menus).
    pub played: bool,
    /// While on a merged 16:9 hub or Pause, toggles between the merged hub and `LargeTop`.
    pub menu_override: bool,
    /// While watching a standalone How to Play video, toggles `LargeTop` without altering `shown`.
    pub tutorial_override: bool,
}

impl Default for BottomToggle {
    fn default() -> Self {
        Self::new()
    }
}

impl BottomToggle {
    pub fn new() -> Self {
        Self {
            shown: true,
            played: false,
            menu_override: false,
            tutorial_override: false,
        }
    }

    pub fn update(
        &mut self,
        toggle_pressed: bool,
        said: u32,
        on_special: bool,
        on_pause: bool,
        on_tutorial: bool,
        in_frontend_menu: bool,
    ) -> bool {
        if !on_special && !on_pause {
            self.menu_override = false;
        }
        if !on_tutorial {
            self.tutorial_override = false;
        }
        if said != 0 && !self.played {
            self.played = true;
            self.shown = false;
        } else if (in_frontend_menu || on_special) && self.played {
            self.played = false;
            self.shown = true;
        }
        if toggle_pressed {
            if on_special || on_pause {
                self.menu_override = !self.menu_override;
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

/// Snapshot computed before `run_frame()` to feed into `post_frame()`.
#[derive(Debug, Clone, Copy)]
pub struct PreFrame {
    pub on_title: bool,
    pub on_pause: bool,
    pub in_battle: bool,
    pub res_battle: bool,
    pub bottom_shown: bool,
    pub in_play: bool,
    pub reset_interpolation: bool,
}

/// Presenter and layout updates emitted after `run_frame()`.
#[derive(Debug, Clone, Copy)]
pub struct PostFrame {
    pub layout: ScreenLayout,
    pub layout_changed: bool,
    pub reset_interpolation: bool,
    pub on_pause: bool,
    pub pause_state: Option<u8>,
    pub chapter_title_right: Option<f32>,
    pub weapon_state: Option<u8>,
    pub cauldron_state: Option<(f32, f32, u32)>,
}

/// Full KIU 16:9 merged-screen UI state machine shared between Switch and PC.
#[derive(Debug, Clone)]
pub struct KiuUiState {
    pub bottom: BottomToggle,
    pub layout: ScreenLayout,
    pub out_of_play: u32,
    pub title_quiet: u32,
    pub on_title: bool,
    pub weapon_sub: u8,
    pub pause_sub: u8,
    pub cauldron_cursor_x: f32,
    pub cauldron_opacity: f32,
    pub cauldron_hold: u8,
}

impl Default for KiuUiState {
    fn default() -> Self {
        Self::new()
    }
}

impl KiuUiState {
    pub fn new() -> Self {
        Self {
            bottom: BottomToggle::new(),
            layout: ScreenLayout::LargeTop,
            out_of_play: u32::MAX,
            title_quiet: u32::MAX,
            on_title: false,
            weapon_sub: 0,
            pause_sub: 0,
            cauldron_cursor_x: 0.0,
            cauldron_opacity: 0.48,
            cauldron_hold: 0,
        }
    }

    /// Resets battle-quiet tracking after loading a savestate.
    pub fn on_state_loaded(&mut self) {
        self.out_of_play = u32::MAX;
    }

    /// Updates pre-frame GPU flags, pause detection, and battle/menu state.
    pub fn pre_frame(
        &mut self,
        gpu_flags: &Flags,
        memory: &mut Memory,
        said: u32,
        toggle_bottom_pressed: bool,
    ) -> PreFrame {
        let on_title = gpu_flags.on_title.swap(false, Ordering::Relaxed);
        let guest_pause = read_guest_pause_state(memory);
        let on_pause = guest_pause.is_some();
        let mut reset_interpolation = false;
        if let Some(sub) = guest_pause {
            if pause_sub_mode(sub) != pause_sub_mode(self.pause_sub) {
                reset_interpolation = true;
            }
            self.pause_sub = sub;
        }

        let big_portrait = gpu_flags.big_portrait.swap(false, Ordering::Relaxed);
        self.out_of_play = if said != 0 || big_portrait {
            0
        } else {
            self.out_of_play.saturating_add(1)
        };
        let in_battle = self.out_of_play <= 90;
        let in_frontend_menu = gpu_flags.on_frontend_menu.swap(false, Ordering::Relaxed) && !in_battle;
        let on_file_pre = gpu_flags.on_file_select.load(Ordering::Relaxed) && !in_battle;
        self.title_quiet = if on_file_pre {
            u32::MAX
        } else if on_title && !in_battle {
            0
        } else {
            self.title_quiet.saturating_add(1)
        };
        let on_title_pre = self.title_quiet <= 15 && !in_battle && !on_file_pre;
        let on_special_pre = (on_title_pre
            || gpu_flags.on_fuse.load(Ordering::Relaxed)
            || gpu_flags.on_cauldron.load(Ordering::Relaxed)
            || gpu_flags.on_chapter.load(Ordering::Relaxed)
            || gpu_flags.on_mode.load(Ordering::Relaxed)
            || gpu_flags.on_solo.load(Ordering::Relaxed)
            || gpu_flags.on_weapon.load(Ordering::Relaxed)
            || on_file_pre)
            && !in_battle;

        let res_battle = said != 0 || (self.bottom.played && !in_frontend_menu && !on_special_pre);
        let on_tutorial_movie = in_battle && !res_battle;
        let bottom_shown = self.bottom.update(
            toggle_bottom_pressed,
            said,
            on_special_pre,
            on_pause,
            on_tutorial_movie,
            in_frontend_menu,
        );
        gpu_flags.menu_override.store(self.bottom.menu_override, Ordering::Relaxed);
        let in_play = !bottom_shown && !on_special_pre && !on_pause;
        gpu_flags.in_play.store(in_play, Ordering::Relaxed);
        gpu_flags.in_battle.store(in_battle, Ordering::Relaxed);
        gpu_flags.active_battle.store(res_battle, Ordering::Relaxed);
        gpu_flags.on_pause.store(on_pause, Ordering::Relaxed);

        PreFrame {
            on_title,
            on_pause,
            in_battle,
            res_battle,
            bottom_shown,
            in_play,
            reset_interpolation,
        }
    }

    /// Resolves post-frame layout selection and Vulkan hub shader uniforms.
    ///
    /// When `supports_merged_hubs` is `false` (e.g. OpenGL fallback), merged
    /// 16:9 hub screens fall back cleanly to `ScreenLayout::LargeTop`.
    pub fn post_frame(
        &mut self,
        gpu_flags: &Flags,
        memory: &mut Memory,
        pre: PreFrame,
        touching_cauldron_slider: bool,
        supports_merged_hubs: bool,
    ) -> PostFrame {
        let in_battle = pre.in_battle;
        if gpu_flags.on_title.load(Ordering::Relaxed) && !in_battle {
            self.title_quiet = 0;
        }
        let on_file_post = gpu_flags.on_file_select.load(Ordering::Relaxed) && !in_battle;
        if on_file_post {
            self.title_quiet = u32::MAX;
        }
        let on_title_active = self.title_quiet <= 15 && !in_battle && !on_file_post;
        self.on_title = on_title_active;

        let on_fuse_post = gpu_flags.on_fuse.load(Ordering::Relaxed) && !in_battle && !on_title_active;
        let on_cauldron_post = gpu_flags.on_cauldron.load(Ordering::Relaxed)
            && !in_battle
            && !on_title_active
            && !on_fuse_post;
        let on_chapter_post = gpu_flags.on_chapter.load(Ordering::Relaxed)
            && !in_battle
            && !on_title_active
            && !on_fuse_post
            && !on_cauldron_post;
        let on_solo_post = gpu_flags.on_solo.load(Ordering::Relaxed)
            && !in_battle
            && !on_title_active
            && !on_fuse_post
            && !on_cauldron_post
            && !on_chapter_post;
        let on_mode_post = gpu_flags.on_mode.load(Ordering::Relaxed)
            && !in_battle
            && !on_title_active
            && !on_fuse_post
            && !on_cauldron_post
            && !on_chapter_post
            && !on_solo_post;
        let on_weapon_post = gpu_flags.on_weapon.load(Ordering::Relaxed)
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

        let guest_pause_post = read_guest_pause_state(memory);
        let on_pause_post = pre.on_pause && guest_pause_post.is_some();
        let bottom_blank = gpu_flags.bottom_blank.load(Ordering::Relaxed);

        let merged = |layout: ScreenLayout| {
            if self.bottom.menu_override || !supports_merged_hubs {
                ScreenLayout::LargeTop
            } else {
                layout
            }
        };

        let desired_layout = if on_pause_post {
            merged(ScreenLayout::Pause)
        } else if on_fuse_post {
            merged(ScreenLayout::FuseSplitDeck)
        } else if on_cauldron_post {
            merged(ScreenLayout::FiendsCauldron)
        } else if on_chapter_post {
            merged(ScreenLayout::ChapterSelect)
        } else if on_solo_post {
            merged(ScreenLayout::SoloHub)
        } else if on_mode_post {
            merged(ScreenLayout::ModeSelect)
        } else if on_weapon_post {
            merged(ScreenLayout::WeaponSelect)
        } else if on_file_post {
            merged(ScreenLayout::FileSelect)
        } else if on_title_active {
            if self.bottom.menu_override {
                ScreenLayout::LargeTop
            } else {
                ScreenLayout::TopOnly
            }
        } else if !pre.bottom_shown || bottom_blank {
            ScreenLayout::TopOnly
        } else {
            ScreenLayout::LargeTop
        };

        let mut reset_interpolation = pre.reset_interpolation;
        let mut pause_state = None;
        if let Some(sub) = guest_pause_post.filter(|_| on_pause_post) {
            if pause_sub_mode(sub) != pause_sub_mode(self.pause_sub) {
                reset_interpolation = true;
            }
            self.pause_sub = sub;
            pause_state = Some(sub);
        }

        let chapter_title_right = if on_chapter_post {
            let raw_bits = gpu_flags.chapter_title_right.load(Ordering::Relaxed);
            Some(if raw_bits != 0 { f32::from_bits(raw_bits) } else { 302.0 })
        } else {
            None
        };

        let mut weapon_state = None;
        if on_weapon_post {
            let sub = gpu_flags.weapon_sub.load(Ordering::Relaxed).max(1);
            if sub != self.weapon_sub {
                self.weapon_sub = sub;
                reset_interpolation = true;
            }
            weapon_state = Some(sub);
        }

        let cauldron_state = if on_cauldron_post {
            let raw_bits = gpu_flags.cauldron_cursor_x.load(Ordering::Relaxed);
            let cx = if raw_bits != 0 { f32::from_bits(raw_bits) } else { 237.4 };
            if self.cauldron_cursor_x > 0.0 && (cx - self.cauldron_cursor_x).abs() > 0.04 {
                self.cauldron_hold = 30;
            } else if touching_cauldron_slider {
                self.cauldron_hold = 30;
            }
            self.cauldron_cursor_x = cx;
            if touching_cauldron_slider || self.cauldron_hold > 0 {
                if !touching_cauldron_slider {
                    self.cauldron_hold = self.cauldron_hold.saturating_sub(1);
                }
                self.cauldron_opacity = (self.cauldron_opacity + 0.16).min(1.0);
            } else {
                self.cauldron_opacity = (self.cauldron_opacity - 0.035).max(0.48);
            }
            let hud_visible = gpu_flags.cauldron_hud.load(Ordering::Relaxed);
            let room_rgb = gpu_flags.cauldron_room_rgb.load(Ordering::Relaxed);
            let hud_flags = (hud_visible as u32) | (room_rgb << 8);
            Some((cx.clamp(16.0, 304.0), self.cauldron_opacity, hud_flags))
        } else {
            self.cauldron_cursor_x = 0.0;
            self.cauldron_opacity = 0.48;
            self.cauldron_hold = 0;
            None
        };

        let layout_changed = desired_layout != self.layout;
        if layout_changed {
            self.layout = desired_layout;
            reset_interpolation = true;
        }

        PostFrame {
            layout: desired_layout,
            layout_changed,
            reset_interpolation,
            on_pause: on_pause_post,
            pause_state,
            chapter_title_right,
            weapon_state,
            cauldron_state,
        }
    }

    /// Convenience wrapper mapping host window coordinates using the active layout and sub-modes.
    pub fn touch(
        &self,
        win_w: u32,
        win_h: u32,
        top_width: f32,
        integer: bool,
        x: f32,
        y: f32,
    ) -> Option<(u16, u16)> {
        touch_on_bottom(
            win_w,
            win_h,
            top_width,
            integer,
            self.layout,
            self.weapon_sub,
            self.pause_sub,
            x,
            y,
        )
    }

    /// Standard 16:9 top width used by KIU's merged layouts (`426.666...`).
    pub const KIU_TOP_WIDTH: f32 = TOP_WIDTH;
}

/// Embedded C function overrides for Kid Icarus: Uprising (USA, `0004000000030100`).
pub const KIU_USA_C_OVERRIDES: [(&str, &str); 3] = [
    ("kiu_aim_mode.c", include_str!("../overrides-usa/kiu_aim_mode.c")),
    ("kiu_air_aim.c", include_str!("../overrides-usa/kiu_air_aim.c")),
    ("kiu_mouse_look.c", include_str!("../overrides-usa/kiu_mouse_look.c")),
];

/// Materializes the built-in KIU (USA) C overrides into `dir`, ready to pass
/// to `recomp3ds::build::Options::overrides`.
pub fn write_usa_overrides(dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, source) in KIU_USA_C_OVERRIDES {
        std::fs::write(dir.join(name), source)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_on_bottom_matches_720p_and_scales_to_1080p_and_1440p() {
        // Test scale invariance across 720p, 1080p, 1440p, and pillarboxed 2560x1080 ultrawide.
        let layouts = [
            (ScreenLayout::SoloHub, 0, 0, 0.50, 0.6875),      // Move Out!
            (ScreenLayout::SoloHub, 0, 0, 0.858, 0.504),      // Arms Altar
            (ScreenLayout::ModeSelect, 0, 0, 0.50, 0.50),
            (ScreenLayout::FileSelect, 0, 0, 0.50, 0.72),
            (ScreenLayout::ChapterSelect, 0, 0, 0.50, 0.50),
            (ScreenLayout::WeaponSelect, 1, 0, 0.25, 0.65),
            (ScreenLayout::WeaponSelect, 2, 0, 0.30, 0.65),
            (ScreenLayout::Pause, 0, 2, 0.50, 0.50),
            (ScreenLayout::Pause, 0, 3, 0.50, 0.50),
            (ScreenLayout::Pause, 0, 9, 0.92, 0.95),
            (ScreenLayout::FiendsCauldron, 0, 0, 0.05, 0.90),
            (ScreenLayout::TopOnly, 0, 0, 0.50, 0.50),
        ];

        for (layout, wsub, psub, u, v) in layouts {
            let p720 = touch_on_bottom(1280, 720, TOP_WIDTH, false, layout, wsub, psub, u * 1280.0, v * 720.0);
            let p1080 = touch_on_bottom(1920, 1080, TOP_WIDTH, false, layout, wsub, psub, u * 1920.0, v * 1080.0);
            let p1440 = touch_on_bottom(2560, 1440, TOP_WIDTH, false, layout, wsub, psub, u * 2560.0, v * 1440.0);
            assert!(p720.is_some(), "expected hit at 720p for {layout:?} ({wsub}, {psub})");
            let (x7, y7) = p720.unwrap();
            let (x10, y10) = p1080.unwrap();
            let (x14, y14) = p1440.unwrap();
            assert!((x7 as i32 - x10 as i32).abs() <= 1 && (y7 as i32 - y10 as i32).abs() <= 1,
                "1080p mismatch for {layout:?}: 720p={p720:?} vs 1080p={p1080:?}");
            assert!((x7 as i32 - x14 as i32).abs() <= 1 && (y7 as i32 - y14 as i32).abs() <= 1,
                "1440p mismatch for {layout:?}: 720p={p720:?} vs 1440p={p1440:?}");

            // Pillarboxed 2560x1080: 16:9 content is centered at x = 320..2240 (width 1920).
            let ux = 320.0 + u * 1920.0;
            let uy = v * 1080.0;
            let puw = touch_on_bottom(2560, 1080, TOP_WIDTH, false, layout, wsub, psub, ux, uy);
            assert!(puw.is_some(), "expected hit on pillarboxed ultrawide for {layout:?}");
            let (xu, yu) = puw.unwrap();
            assert!((x7 as i32 - xu as i32).abs() <= 1 && (y7 as i32 - yu as i32).abs() <= 1,
                "ultrawide mismatch for {layout:?}: 720p={p720:?} vs uw={puw:?}");
        }
    }

    #[test]
    fn cauldron_back_button_and_slider_scale_across_resolutions() {
        // Bottom-left back button hits (16, 221) at both 720p and 1440p.
        assert_eq!(
            touch_on_bottom(1280, 720, TOP_WIDTH, false, ScreenLayout::FiendsCauldron, 0, 0, 80.0, 650.0),
            Some((16, 221))
        );
        assert_eq!(
            touch_on_bottom(2560, 1440, TOP_WIDTH, false, ScreenLayout::FiendsCauldron, 0, 0, 160.0, 1300.0),
            Some((16, 221))
        );
        // Slider hit test matches 720p and 1440p.
        assert!(is_touching_cauldron_slider(1280, 720, TOP_WIDTH, false, 640.0, 150.0));
        assert!(is_touching_cauldron_slider(2560, 1440, TOP_WIDTH, false, 1280.0, 300.0));
        assert!(!is_touching_cauldron_slider(2560, 1440, TOP_WIDTH, false, 100.0, 1200.0));
    }

    #[test]
    fn kiu_ui_state_switches_between_merged_hubs_and_battle() {
        let mut ui = KiuUiState::new();
        let flags = Flags::default();
        let mut mem = Memory::new(false, 64 * 1024 * 1024);

        // Solo Hub on Vulkan -> ScreenLayout::SoloHub
        flags.on_solo.store(true, Ordering::Relaxed);
        let pre = ui.pre_frame(&flags, &mut mem, 0, false);
        assert!(!pre.in_play);
        let post = ui.post_frame(&flags, &mut mem, pre, false, true);
        assert_eq!(post.layout, ScreenLayout::SoloHub);
        assert!(post.layout_changed);

        // OpenGL fallback -> ScreenLayout::LargeTop
        let pre_gl = ui.pre_frame(&flags, &mut mem, 0, false);
        let post_gl = ui.post_frame(&flags, &mut mem, pre_gl, false, false);
        assert_eq!(post_gl.layout, ScreenLayout::LargeTop);

        // Enter battle -> ScreenLayout::TopOnly and in_play = true
        flags.on_solo.store(false, Ordering::Relaxed);
        let pre_b = ui.pre_frame(&flags, &mut mem, GAME_FLYING, false);
        assert!(pre_b.in_play);
        assert!(pre_b.in_battle);
        let post_b = ui.post_frame(&flags, &mut mem, pre_b, false, true);
        assert_eq!(post_b.layout, ScreenLayout::TopOnly);
    }
}
