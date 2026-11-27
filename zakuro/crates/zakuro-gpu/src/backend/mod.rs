//! presentation backends.

#[cfg(feature = "opengl")]
pub mod gl;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "vulkan")]
mod vulkan_overlay;

/// one screen's pixels, already converted to straight RGBA8, or where they
/// are on the GPU, for a presenter sharing the renderer's device.
pub struct ScreenImage<'a> {
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
    pub gpu: Option<GpuScreen>,
}

impl ScreenImage<'_> {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0 || (self.pixels.is_empty() && self.gpu.is_none())
    }
}

/// a screen's picture upright in an image of the renderer's, in the general
/// layout, which the Vulkan presenter draws straight from when the two share
/// a device.
#[derive(Debug, Clone, Copy)]
pub struct GpuScreen {
    #[cfg(feature = "vulkan")]
    pub(crate) view: ash::vk::ImageView,
    /// where the screen's pixels lie in the image, as texture coordinates,
    /// the corner and the size, then the corners inset by half a texel,
    /// which filtering keeps within.
    pub(crate) area: [f32; 4],
    pub(crate) bounds: [f32; 4],
}

/// what is drawn over the screens, a user interface, as textured triangles
/// in window pixels.
#[derive(Default)]
pub struct Overlay {
    /// textures to make or change before drawing.
    pub textures: Vec<OverlayTexture>,
    pub meshes: Vec<OverlayMesh>,
    /// textures that go away once this frame is drawn.
    pub free: Vec<u64>,
}

/// pixels for a texture of the overlay.
pub struct OverlayTexture {
    pub id: u64,
    /// where the pixels go in a texture that exists, none to make it anew at
    /// this size.
    pub offset: Option<[u32; 2]>,
    pub size: [u32; 2],
    /// RGBA, premultiplied by alpha, in sRGB.
    pub pixels: Vec<u8>,
    /// filtered when scaled, rather than nearest.
    pub linear: bool,
}

/// one corner of an overlay triangle.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OverlayVertex {
    /// window pixels from the top left.
    pub position: [f32; 2],
    pub uv: [f32; 2],
    /// premultiplied sRGB.
    pub color: [u8; 4],
}

/// triangles of the overlay that share a texture and a clip.
pub struct OverlayMesh {
    pub texture: u64,
    /// the part of the window they may draw in, left, top, right, bottom.
    pub clip: [u32; 4],
    pub vertices: Vec<OverlayVertex>,
    pub indices: Vec<u32>,
}

/// where a screen goes in the window, in pixels from the top left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// how the screens are arranged in the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenLayout {
    /// the top screen over the bottom one, as on the console.
    #[default]
    Stacked,
    /// next to each other, the top screen on the left.
    SideBySide,
    /// the top screen alone.
    TopOnly,
    /// the bottom screen alone.
    BottomOnly,
    /// the top screen big, the bottom one small at its lower right.
    LargeTop,
    /// the top screen, with the bottom one over its middle, for menus played
    /// on the bottom screen while the window keeps the top screen's shape.
    BottomOverTop,
    /// Kid Icarus: Uprising's Fuse Weapons workbench on a single 16:9 screen:
    /// weapon cards A & B and the AxB banner on the left, result stats, live
    /// preview, and the full 320x240 fusion matrix on the right, every slice
    /// at 1:1 console pixel aspect.
    FuseSplitDeck,
    /// Kid Icarus: Uprising's Fiend's Cauldron on a single 16:9 screen:
    /// full-width Reaper/Treasure Chest mural hearth backdrop, screen-blended
    /// 3D Cauldron & flames, top-left Total Hearts pouch, bottom-left Back
    /// button, and a dynamic-opacity floating Intensity cursor thumb.
    FiendsCauldron,
    /// Kid Icarus: Uprising's Chapter Select on a single 16:9 screen:
    /// centered chapter title ribbon at top, panoramic stage art + records
    /// side-by-side in the upper showcase, and full-width unstretched 25-chapter
    /// carousel across the lower half.
    ChapterSelect,
    /// Kid Icarus: Uprising's Mode Select on a single 16:9 screen:
    /// native bottom-screen interlocking buttons lifted to 1:1 aspect over the
    /// full-width pastel backdrop, with the top screen's undulating black-wave
    /// description banner composited along the bottom.
    ModeSelect,
    /// Kid Icarus: Uprising's Solo Hub on a single 16:9 screen (Perimeter Frame):
    /// full-bleed 3D Sanctuary camera with top-left Total Hearts pouch and the
    /// 7 native bottom-screen station sprites repurposed around the perimeter.
    SoloHub,
    /// Kid Icarus: Uprising's File Select on a single 16:9 screen (Vertical Stack):
    /// full-bleed lined rainbow parchment with centered 1:1 Speech-Bubble Summary
    /// connected via its native pointer tail to the 3 leather-bound Tomes below.
    FileSelect,
    /// Kid Icarus: Uprising's Gear Hub, Weapon Category Select, and Weapon List on
    /// a single 16:9 screen (Option B): left-hand 1:1 interactive Armory / Category /
    /// Gear panel with canonical bottom-left Back button, and right-hand Weapon
    /// Inspector with mirrored right-aligned Stars Plaque, left-side Weapon-Type
    /// icon, right-aligned Modifier pills, and 3D Weapon showcase.
    WeaponSelect,
    /// Kid Icarus: Uprising's in-battle Pause screen (Main Pause, Options/Controls/Sound,
    /// and Quit Confirmation) merged onto a single 16:9 screen: full-bleed frozen 3D top
    /// screen keeping `|| PAUSED` (upper-left) and Options help text (bottom strip) visible,
    /// with the interactive 320x240 bottom pause menu framed cleanly at 1:1 pixel aspect.
    Pause,
}

/// how the screens' pixels are drawn when scaled to the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenFilter {
    /// blended between pixels.
    #[default]
    Smooth,
    /// whole pixels kept sharp and blended only where two meet, so that a
    /// screen scaled by an odd amount stays crisp without stairs.
    Sharp,
    /// every pixel a square of the nearest one.
    Pixels,
}

impl ScreenFilter {
    /// the number the present shaders know it by.
    pub(crate) fn mode(self) -> i32 {
        match self {
            ScreenFilter::Smooth => 0,
            ScreenFilter::Pixels => 1,
            ScreenFilter::Sharp => 2,
        }
    }
}

impl ScreenLayout {
    /// the size the screens take up together at the console's resolution,
    /// with the top screen top_width wide.
    pub fn size(self, top_width: f32) -> (u32, u32) {
        let (width, height) = self.size_f(top_width);
        (width.ceil() as u32, height as u32)
    }

    fn size_f(self, top_width: f32) -> (f32, f32) {
        match self {
            ScreenLayout::Stacked => (top_width.max(320.0), 480.0),
            ScreenLayout::SideBySide => (top_width + 320.0, 240.0),
            ScreenLayout::TopOnly | ScreenLayout::BottomOverTop => (top_width, 240.0),
            ScreenLayout::BottomOnly => (320.0, 240.0),
            ScreenLayout::LargeTop => (top_width + LARGE_TOP_BOTTOM.0 as f32, 240.0),
            ScreenLayout::FuseSplitDeck
            | ScreenLayout::FiendsCauldron
            | ScreenLayout::ChapterSelect
            | ScreenLayout::ModeSelect
            | ScreenLayout::SoloHub
            | ScreenLayout::FileSelect
            | ScreenLayout::WeaponSelect
            | ScreenLayout::Pause => (FUSE_DECK_WIDTH, 240.0),
        }
    }
}

/// the top screen's width at the console's resolution, its own shape. a
/// wider one shows a game that a widescreen code makes draw a wider scene
/// into the same 400x240.
pub const TOP_WIDTH: f32 = 400.0;

/// the 16:9 Fuse Split-Deck logical width at 240p console height.
pub const FUSE_DECK_WIDTH: f32 = 240.0 * 16.0 / 9.0;

/// the bottom screen's size next to the big top one, half the console's.
const LARGE_TOP_BOTTOM: (u32, u32) = (160, 120);

/// how big the bottom screen is over the top one, for the top's scale.
const BOTTOM_OVER_TOP: f32 = 0.9;

fn window_scale_and_offset(window_width: u32, window_height: u32, total_width: f32, total_height: f32, integer: bool) -> (f32, f32, f32) {
    let mut scale = (window_width as f32 / total_width).min(window_height as f32 / total_height);
    if integer && scale >= 1.0 {
        scale = scale.floor();
    }
    let mut offset_x = (window_width as f32 - total_width * scale) / 2.0;
    let mut offset_y = (window_height as f32 - total_height * scale) / 2.0;
    if integer {
        // whole pixels, or every texel would not get the same size
        offset_x = offset_x.floor();
        offset_y = offset_y.floor();
    }
    (scale, offset_x, offset_y)
}

/// works out where the screens go inside a window, keeping the 3DS's
/// aspect ratio, or the top screen top_w wide, and centring them, scaled by
/// a whole number when integer asks and the window has room. a screen that
/// does not show has no place.
pub fn layout(window_width: u32, window_height: u32, arrangement: ScreenLayout, top_w: f32, integer: bool) -> (Option<Viewport>, Option<Viewport>) {
    let (total_width, total_height) = arrangement.size_f(top_w);
    let (scale, offset_x, offset_y) = window_scale_and_offset(window_width, window_height, total_width, total_height, integer);

    if arrangement == ScreenLayout::FuseSplitDeck {
        // Left column holds Top's A/B stat cards; right column holds the
        // continuous Result + Fusion Matrix stack so touch_on_bottom maps 1:1.
        let slices = fuse_deck_slices(window_width, window_height, integer, Shown::Top);
        return (Some(slices[0].viewport), Some(slices[3].viewport));
    }
    if arrangement == ScreenLayout::FiendsCauldron {
        let stage = Viewport {
            x: offset_x,
            y: offset_y,
            width: FUSE_DECK_WIDTH * scale,
            height: 240.0 * scale,
        };
        let cauldron_bot = Viewport {
            x: offset_x + 62.22222 * scale,
            y: offset_y + 13.0 * scale,
            width: (320.0 * (227.0 / 240.0)) * scale,
            height: 227.0 * scale,
        };
        return (Some(stage), Some(cauldron_bot));
    }
    if arrangement == ScreenLayout::ChapterSelect {
        let stage = Viewport {
            x: offset_x,
            y: offset_y,
            width: FUSE_DECK_WIDTH * scale,
            height: 240.0 * scale,
        };
        let chapter_bot = Viewport {
            x: offset_x + 18.67 * scale,
            y: offset_y + 111.0 * scale,
            width: (408.0 - 18.67) * scale,
            height: (238.5 - 111.0) * scale,
        };
        return (Some(stage), Some(chapter_bot));
    }

    if matches!(
        arrangement,
        ScreenLayout::ModeSelect
            | ScreenLayout::SoloHub
            | ScreenLayout::FileSelect
            | ScreenLayout::WeaponSelect
            | ScreenLayout::Pause
    ) {
        let stage = Viewport {
            x: offset_x,
            y: offset_y,
            width: FUSE_DECK_WIDTH * scale,
            height: 240.0 * scale,
        };
        return (Some(stage), Some(stage));
    }

    let top = (arrangement != ScreenLayout::BottomOnly).then_some(Viewport {
        x: offset_x,
        y: offset_y,
        width: top_w * scale,
        height: 240.0 * scale,
    });
    let bottom = match arrangement {
        // the bottom screen is narrower, so it is centered under the top one.
        ScreenLayout::Stacked => Some(Viewport {
            x: offset_x + (total_width - 320.0) / 2.0 * scale,
            y: offset_y + 240.0 * scale,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::SideBySide => Some(Viewport {
            x: offset_x + top_w * scale,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::LargeTop => {
            // half the top screen's scale, a whole number of it when the
            // screens are scaled by whole numbers
            let half = (LARGE_TOP_BOTTOM.0 as f32 / 320.0) * scale;
            let small = if integer && half >= 1.0 { half.floor() } else { half };
            Some(Viewport {
                x: offset_x + top_w * scale,
                y: offset_y + 240.0 * scale - 240.0 * small,
                width: 320.0 * small,
                height: 240.0 * small,
            })
        }
        ScreenLayout::BottomOverTop => {
            // nine tenths of the top screen's height, in its middle, so its
            // edges still show around it
            let small = BOTTOM_OVER_TOP * scale;
            Some(Viewport {
                x: offset_x + (top_w * scale - 320.0 * small) / 2.0,
                y: offset_y + (240.0 * scale - 240.0 * small) / 2.0,
                width: 320.0 * small,
                height: 240.0 * small,
            })
        }
        ScreenLayout::TopOnly => None,
        ScreenLayout::BottomOnly => Some(Viewport {
            x: offset_x,
            y: offset_y,
            width: 320.0 * scale,
            height: 240.0 * scale,
        }),
        ScreenLayout::FuseSplitDeck
        | ScreenLayout::FiendsCauldron
        | ScreenLayout::ChapterSelect
        | ScreenLayout::ModeSelect
        | ScreenLayout::SoloHub
        | ScreenLayout::FileSelect
        | ScreenLayout::WeaponSelect
        | ScreenLayout::Pause => unreachable!(),
    };
    (top, bottom)
}

/// how the top screen's two pictures, one for each eye, are shown when the
/// title draws in 3D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StereoMode {
    /// the left eye's picture alone, as with the 3D slider down.
    #[default]
    Off,
    /// the eyes mixed into red for the left one and cyan for the right, for
    /// glasses of those colors.
    Anaglyph,
    /// each eye in its half of the window, for 3D TVs and headsets.
    SideBySide,
    /// the same squeezed to half width, which a 3D TV stretches back.
    HalfSideBySide,
    /// the eyes on alternate rows of the window, for passive 3D monitors.
    Interlaced,
}

/// the 3D a presenter shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stereo {
    pub mode: StereoMode,
    /// the right eye's picture where the left one's goes, for viewing cross
    /// eyed, glasses worn the other way round or rows counted the other way.
    pub swap: bool,
}

/// which picture a place in the window shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shown {
    /// the top screen, as the left eye sees it.
    Top,
    /// the top screen as the right eye sees it.
    Right,
    Bottom,
    /// the top screen's two pictures mixed the way the stereo mode does.
    Both,
}

/// the screens a frame draws, where and from which picture. the 3D shows
/// when the title drew a picture for the right eye, side by side shows the
/// left one's twice otherwise, so a 3D TV keeps its halves.
pub fn place(window_width: u32, window_height: u32, arrangement: ScreenLayout, top_width: f32, integer: bool, stereo: Stereo, right: bool) -> Vec<(Viewport, Shown)> {
    let screens = |(top, bottom): (Option<Viewport>, Option<Viewport>), eye: Shown| {
        top.map(|viewport| (viewport, eye)).into_iter().chain(bottom.map(|viewport| (viewport, Shown::Bottom)))
    };
    let (first, second) = match (right, stereo.swap) {
        (false, _) => (Shown::Top, Shown::Top),
        (true, false) => (Shown::Top, Shown::Right),
        (true, true) => (Shown::Right, Shown::Top),
    };
    match stereo.mode {
        StereoMode::Off => screens(layout(window_width, window_height, arrangement, top_width, integer), Shown::Top).collect(),
        StereoMode::Anaglyph | StereoMode::Interlaced => {
            let top = if right { Shown::Both } else { Shown::Top };
            screens(layout(window_width, window_height, arrangement, top_width, integer), top).collect()
        }
        StereoMode::SideBySide => {
            // each half a window of its own
            let half = window_width / 2;
            let places = layout(half, window_height, arrangement, top_width, integer);
            let moved = |(viewport, shown): (Viewport, Shown)| (Viewport { x: viewport.x + half as f32, ..viewport }, shown);
            screens(places, first).chain(screens(places, second).map(moved)).collect()
        }
        StereoMode::HalfSideBySide => {
            // the whole window's places, squeezed into each half
            let places = layout(window_width, window_height, arrangement, top_width, integer);
            let half = window_width as f32 / 2.0;
            let squeezed = |by: f32| move |(viewport, shown): (Viewport, Shown)| {
                (Viewport { x: viewport.x / 2.0 + by, width: viewport.width / 2.0, ..viewport }, shown)
            };
            screens(places, first).map(squeezed(0.0)).chain(screens(places, second).map(squeezed(half))).collect()
        }
    }
}

/// one screen slice drawn into the window, with its normalized sub-rectangle
/// `[x, y, width, height]` inside that screen's picture (`[0, 0, 1, 1]` for a
/// whole screen).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedSlice {
    pub viewport: Viewport,
    pub shown: Shown,
    pub uv: [f32; 4],
}

fn fuse_deck_slices(window_width: u32, window_height: u32, integer: bool, top_eye: Shown) -> [PlacedSlice; 4] {
    let (scale, offset_x, offset_y) = window_scale_and_offset(window_width, window_height, FUSE_DECK_WIDTH, 240.0, integer);
    let l1_w = 228.0;
    let l1_h = l1_w * (211.0 / 252.0);
    let l1_y = (240.0 - l1_h) * 0.5;

    let rx = (offset_x + 235.5 * scale).round();
    let rw = (187.5 * scale).round();

    let ban_h = ((187.5 * 28.0 / 400.0) * scale).round();
    let sep = (0.45 * scale).round().max(1.0);
    let body_h = ((187.5 * 52.0 / 146.0) * scale).round();
    let r3_h = ((187.5 * 240.0 / 320.0) * scale).round();
    let total_r_h = ban_h + sep + body_h + sep + r3_h;
    let ban_y = (offset_y + (240.0 * scale - total_r_h) * 0.5).round();
    let body_y = ban_y + ban_h + sep;
    let r3_y = body_y + body_h + sep;

    [
        // L1: Weapon A & B stat/mod cards (252x211 px of Top 400x240)
        PlacedSlice {
            viewport: Viewport {
                x: (offset_x + 3.5 * scale).round(),
                y: (offset_y + l1_y * scale).round(),
                width: (l1_w * scale).round(),
                height: (l1_h * scale).round(),
            },
            shown: top_eye,
            uv: [0.0, 0.0, 252.0 / 400.0, 211.0 / 240.0],
        },
        // R0: Result card header banner — weapon class icon + AxB-> + result name (400x28 px of Top 400x240)
        PlacedSlice {
            viewport: Viewport {
                x: rx,
                y: ban_y,
                width: rw,
                height: ban_h,
            },
            shown: top_eye,
            uv: [0.0, 212.0 / 240.0, 1.0, 28.0 / 240.0],
        },
        // R1: Single continuous Result card body bridging directly onto R3 —
        // Ranged / Melee / Value + sparkling result weapon over one seamless
        // filigree strip (146x52 px of Top 400x240)
        PlacedSlice {
            viewport: Viewport {
                x: rx,
                y: body_y,
                width: rw,
                height: body_h,
            },
            shown: top_eye,
            uv: [254.0 / 400.0, 0.0, 146.0 / 400.0, 52.0 / 240.0],
        },
        // R3: Full uncut 320x240 Fusion Matrix, sort bar, Back button & modals
        PlacedSlice {
            viewport: Viewport {
                x: rx,
                y: r3_y,
                width: rw,
                height: r3_h,
            },
            shown: Shown::Bottom,
            uv: [0.0, 0.0, 1.0, 1.0],
        },
    ]
}

/// the screen slices a frame draws. For `FuseSplitDeck` this returns the four
/// 1:1 sub-crops across the 16:9 workbench; for all other layouts each screen
/// is drawn whole (`uv = [0, 0, 1, 1]`).
pub fn place_slices(
    window_width: u32,
    window_height: u32,
    arrangement: ScreenLayout,
    top_width: f32,
    integer: bool,
    stereo: Stereo,
    right: bool,
) -> Vec<PlacedSlice> {
    if arrangement != ScreenLayout::FuseSplitDeck {
        return place(window_width, window_height, arrangement, top_width, integer, stereo, right)
            .into_iter()
            .map(|(viewport, shown)| PlacedSlice {
                viewport,
                shown,
                uv: [0.0, 0.0, 1.0, 1.0],
            })
            .collect();
    }
    let (first, second) = match (right, stereo.swap) {
        (false, _) => (Shown::Top, Shown::Top),
        (true, false) => (Shown::Top, Shown::Right),
        (true, true) => (Shown::Right, Shown::Top),
    };
    match stereo.mode {
        StereoMode::Off => fuse_deck_slices(window_width, window_height, integer, Shown::Top).into(),
        StereoMode::Anaglyph | StereoMode::Interlaced => {
            let top = if right { Shown::Both } else { Shown::Top };
            fuse_deck_slices(window_width, window_height, integer, top).into()
        }
        StereoMode::SideBySide => {
            let half = window_width / 2;
            let left_eye = fuse_deck_slices(half, window_height, integer, first);
            let right_eye = fuse_deck_slices(half, window_height, integer, second).map(|s| PlacedSlice {
                viewport: Viewport { x: s.viewport.x + half as f32, ..s.viewport },
                ..s
            });
            left_eye.into_iter().chain(right_eye).collect()
        }
        StereoMode::HalfSideBySide => {
            let half = window_width as f32 / 2.0;
            let squeeze = |s: PlacedSlice, by: f32| PlacedSlice {
                viewport: Viewport { x: s.viewport.x / 2.0 + by, width: s.viewport.width / 2.0, ..s.viewport },
                ..s
            };
            let left_eye = fuse_deck_slices(window_width, window_height, integer, first).map(|s| squeeze(s, 0.0));
            let right_eye = fuse_deck_slices(window_width, window_height, integer, second).map(|s| squeeze(s, half));
            left_eye.into_iter().chain(right_eye).collect()
        }
    }
}

/// subtle obsidian workbench fill, antique-gold card borders, and center
/// divider cleared behind `ScreenLayout::FuseSplitDeck` slices (colors in sRGB
/// `[0..1]`).
pub fn fuse_backdrops(window_width: u32, window_height: u32, integer: bool) -> Vec<(Viewport, [f32; 4])> {
    let (scale, offset_x, offset_y) = window_scale_and_offset(window_width, window_height, FUSE_DECK_WIDTH, 240.0, integer);
    let border = (0.45 * scale).round().max(1.0);
    let divider_w = (0.35 * scale).round().max(1.0);
    let slices = fuse_deck_slices(window_width, window_height, integer, Shown::Top);
    let mut rects = Vec::with_capacity(7);
    // Obsidian workbench canvas
    rects.push((
        Viewport {
            x: offset_x.round(),
            y: offset_y.round(),
            width: (FUSE_DECK_WIDTH * scale).round(),
            height: (240.0 * scale).round(),
        },
        [0.065, 0.072, 0.095, 1.0],
    ));
    // Center vertical divider between Left Column and Right Column
    rects.push((
        Viewport {
            x: (offset_x + 233.5 * scale - divider_w * 0.5).round(),
            y: (offset_y + 10.0 * scale).round(),
            width: divider_w,
            height: (220.0 * scale).round(),
        },
        [0.24, 0.23, 0.27, 1.0],
    ));
    // Two cohesive antique-gold deck frames:
    // 1. Left A/B stat deck (slices[0])
    // 2. Unified right Result + Fusion Matrix stack (slices[1..=3], bridged flush)
    let cards = [
        slices[0].viewport,
        Viewport {
            x: slices[1].viewport.x,
            y: slices[1].viewport.y,
            width: slices[1].viewport.width,
            height: (slices[3].viewport.y + slices[3].viewport.height) - slices[1].viewport.y,
        },
    ];
    for vp in cards {
        rects.push((
            Viewport {
                x: vp.x - border,
                y: vp.y - border,
                width: vp.width + 2.0 * border,
                height: vp.height + 2.0 * border,
            },
            [0.45, 0.39, 0.25, 1.0],
        ));
    }
    rects
}

#[derive(Debug, thiserror::Error)]
pub enum PresentError {
    #[error("{0}")]
    Backend(String),
    #[error("the swapchain is out of date and was recreated")]
    OutOfDate,
}

pub trait Presenter {
    fn name(&self) -> &'static str;

    /// draws both screens into the window, and the overlay over them. right
    /// is the top screen's picture for the right eye, empty when the title
    /// draws no 3D or none is shown.
    fn present(
        &mut self,
        top: ScreenImage<'_>,
        bottom: ScreenImage<'_>,
        right: ScreenImage<'_>,
        overlay: &Overlay,
    ) -> Result<(), PresentError>;

    /// the window changed size.
    fn resize(&mut self, width: u32, height: u32);

    /// how the screens are arranged from the next frame on.
    fn set_layout(&mut self, arrangement: ScreenLayout);

    /// how wide the top screen shows, TOP_WIDTH for the console's shape.
    fn set_top_width(&mut self, width: f32);

    /// how the screens are filtered, and whether they are scaled by whole
    /// numbers, from the next frame on.
    fn set_scaling(&mut self, filter: ScreenFilter, integer: bool);

    /// how the 3D shows from the next frame on.
    fn set_stereo(&mut self, stereo: Stereo);

    /// live state for `ScreenLayout::FiendsCauldron`: bottom-screen slider
    /// cursor X (`0..320`), floating badge opacity (`0..1`), and packed HUD
    /// flags (`bit 0` = interactive prep HUD visible, `bits 8..31` = room RGB).
    fn set_cauldron_state(&mut self, _cursor_x: f32, _opacity: f32, _hud_flags: u32) {}

    /// live state for `ScreenLayout::ChapterSelect`: rightmost X (`0..400`) of
    /// the top-screen chapter title text so `[Laurel + Chapter XX + Title]` is
    /// dynamically centered over the 16:9 ribbon.
    fn set_chapter_state(&mut self, _title_right: f32) {}

    /// live sub-state for `ScreenLayout::WeaponSelect` (`1` = Weapon List,
    /// `2` = Weapon Category Select, `3` = Gear Hub).
    fn set_weapon_state(&mut self, _sub: u8) {}
    /// live sub-state for `ScreenLayout::Pause` (`PauseMenu + 0x04`: `1..=2, 4` Main Pause,
    /// `3` Quit Confirmation modal, `5..=7` Options / Controls / Sound).
    fn set_pause_state(&mut self, _sub: u8) {}

    /// fraction (`0.80..=1.0`) of the window/swapchain used for the active
    /// display area, centering it inside black overscan borders on TVs.
    fn set_safe_area(&mut self, _factor: f32) {}

    /// whether presentation waits for vertical sync (FIFO / Mailbox) or unlocks immediately.
    fn set_vsync(&mut self, _vsync: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_keeps_the_aspect_ratio_and_centers() {
        // a window exactly 400x480 needs no scaling or offset.
        let (top, bottom) = layout(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!(top.x, 0.0);
        assert_eq!(top.y, 0.0);
        assert_eq!(top.width, 400.0);
        assert_eq!(bottom.y, 240.0);
        assert_eq!(bottom.x, 40.0);
        assert_eq!(bottom.width, 320.0);

        // doubling both dimensions doubles the scale.
        let top = layout(800, 960, ScreenLayout::Stacked, TOP_WIDTH, false).0.unwrap();
        assert_eq!(top.width, 800.0);
        assert_eq!(top.height, 480.0);

        // a window that is too wide letterboxes horizontally.
        let top = layout(1000, 480, ScreenLayout::Stacked, TOP_WIDTH, false).0.unwrap();
        assert_eq!(top.width, 400.0);
        assert_eq!(top.x, 300.0);
    }

    /// the bottom screen over the top one: the window keeps the top screen's
    /// shape, and the bottom screen sits in its middle at nine tenths of its
    /// height, drawn after it so it shows on top.
    #[test]
    fn the_bottom_screen_goes_over_the_middle_of_the_top_one() {
        assert_eq!(ScreenLayout::BottomOverTop.size(TOP_WIDTH), (400, 240));
        let (top, bottom) = layout(800, 480, ScreenLayout::BottomOverTop, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 0.0, 800.0, 480.0));
        assert_eq!((bottom.width, bottom.height), (576.0, 432.0));
        assert_eq!((bottom.x, bottom.y), (112.0, 24.0));
        let shown: Vec<Shown> = place(800, 480, ScreenLayout::BottomOverTop, TOP_WIDTH, false, Stereo::default(), false).into_iter().map(|(_, shown)| shown).collect();
        assert_eq!(shown, [Shown::Top, Shown::Bottom]);
    }

    #[test]
    fn the_other_layouts_place_the_screens_their_way() {
        // side by side, the bottom screen starts where the top one ends
        let (top, bottom) = layout(1440, 480, ScreenLayout::SideBySide, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width), (0.0, 0.0, 800.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 0.0, 640.0, 480.0));

        // the top screen alone fills a window of its shape, and a taller
        // one centers it
        let (top, bottom) = layout(800, 960, ScreenLayout::TopOnly, TOP_WIDTH, false);
        assert!(bottom.is_none());
        let top = top.unwrap();
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 240.0, 800.0, 480.0));

        // and so does the bottom screen alone, a wider window centering it
        let (top, bottom) = layout(800, 480, ScreenLayout::BottomOnly, TOP_WIDTH, false);
        assert!(top.is_none());
        let bottom = bottom.unwrap();
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (80.0, 0.0, 640.0, 480.0));
    }

    #[test]
    fn the_large_top_screen_takes_most_of_the_window() {
        // the bottom screen at half size sits at the top one's lower right
        let (top, bottom) = layout(1120, 480, ScreenLayout::LargeTop, TOP_WIDTH, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.y, top.width, top.height), (0.0, 0.0, 800.0, 480.0));
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (800.0, 240.0, 320.0, 240.0));
    }

    /// a wider top screen fills a window of its shape, and the screens
    /// around it move over to make room.
    #[test]
    fn a_wider_top_screen_takes_its_shape() {
        let wide = 240.0 * 16.0 / 9.0;
        let (top, _) = layout(1920, 1080, ScreenLayout::TopOnly, wide, false);
        let top = top.unwrap();
        assert_eq!((top.x, top.y, top.width.round(), top.height), (0.0, 0.0, 1920.0, 1080.0));

        // stacked, the bottom screen stays centered under it
        let (top, bottom) = layout(560, 480, ScreenLayout::Stacked, 560.0, false);
        let (top, bottom) = (top.unwrap(), bottom.unwrap());
        assert_eq!((top.x, top.width), (0.0, 560.0));
        assert_eq!((bottom.x, bottom.width), (120.0, 320.0));

        // side by side, it starts where the top screen ends
        let (_, bottom) = layout(880, 240, ScreenLayout::SideBySide, 560.0, false);
        assert_eq!(bottom.unwrap().x, 560.0);
        assert_eq!(ScreenLayout::TopOnly.size(wide), (427, 240));
    }

    /// the screens each stereo mode draws, from which picture and where.
    #[test]
    fn the_stereo_modes_place_the_eyes() {
        let on = |mode| Stereo { mode, swap: false };
        let shown = |places: &[(Viewport, Shown)]| places.iter().map(|(_, shown)| *shown).collect::<Vec<_>>();
        let at = |places: &[(Viewport, Shown)], index: usize| {
            let viewport = places[index].0;
            (viewport.x, viewport.y, viewport.width, viewport.height)
        };

        // off, or a title drawing no 3D, the screens as ever
        let off = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Off), true);
        assert_eq!(shown(&off), [Shown::Top, Shown::Bottom]);
        let flat = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Anaglyph), false);
        assert_eq!(shown(&flat), [Shown::Top, Shown::Bottom]);

        // red and cyan or rows, both pictures where the top screen goes
        let mixed = place(400, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::Interlaced), true);
        assert_eq!(shown(&mixed), [Shown::Both, Shown::Bottom]);
        assert_eq!(at(&mixed, 0), (0.0, 0.0, 400.0, 240.0));

        // side by side, each half a window of its own
        let halves = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::SideBySide), true);
        assert_eq!(shown(&halves), [Shown::Top, Shown::Bottom, Shown::Right, Shown::Bottom]);
        assert_eq!(at(&halves, 0), (0.0, 0.0, 400.0, 240.0));
        assert_eq!(at(&halves, 2), (400.0, 0.0, 400.0, 240.0));
        assert_eq!(at(&halves, 3), (440.0, 240.0, 320.0, 240.0));

        // squeezed, the whole window's places at half width in each half
        let squeezed = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, on(StereoMode::HalfSideBySide), true);
        assert_eq!(shown(&squeezed), [Shown::Top, Shown::Bottom, Shown::Right, Shown::Bottom]);
        assert_eq!(at(&squeezed, 0), (100.0, 0.0, 200.0, 240.0));
        assert_eq!(at(&squeezed, 1), (120.0, 240.0, 160.0, 240.0));
        assert_eq!(at(&squeezed, 2), (500.0, 0.0, 200.0, 240.0));

        // swapped, the right eye on the left, and with no 3D drawn both
        // halves the left eye's
        let swapped = place(800, 480, ScreenLayout::Stacked, TOP_WIDTH, false, Stereo { mode: StereoMode::SideBySide, swap: true }, true);
        assert_eq!(shown(&swapped), [Shown::Right, Shown::Bottom, Shown::Top, Shown::Bottom]);
        let flat = place(800, 480, ScreenLayout::TopOnly, TOP_WIDTH, false, on(StereoMode::SideBySide), false);
        assert_eq!(shown(&flat), [Shown::Top, Shown::Top]);
    }

    #[test]
    fn integer_scaling_keeps_whole_multiples() {
        // 2.5 times fits, 2 is used, and the rest is border
        let top = layout(1000, 1200, ScreenLayout::Stacked, TOP_WIDTH, true).0.unwrap();
        assert_eq!((top.width, top.height), (800.0, 480.0));
        assert_eq!((top.x, top.y), (100.0, 120.0));
        // a window smaller than the console's screens still shows them whole
        let top = layout(200, 240, ScreenLayout::Stacked, TOP_WIDTH, true).0.unwrap();
        assert_eq!(top.width, 200.0);
        // an odd window puts the screens on whole pixels
        let (top, bottom) = layout(1001, 961, ScreenLayout::Stacked, TOP_WIDTH, true);
        assert_eq!((top.unwrap().x, top.unwrap().y), (100.0, 0.0));
        assert_eq!(bottom.unwrap().x, 180.0);
        // the small bottom screen of the big top layout keeps a whole scale
        let (top, bottom) = layout(1680, 720, ScreenLayout::LargeTop, TOP_WIDTH, true);
        assert_eq!(top.unwrap().width, 1200.0);
        assert_eq!((bottom.unwrap().width, bottom.unwrap().height), (320.0, 240.0));
    }

    #[test]
    fn fuse_split_deck_keeps_square_pixels_and_non_overlapping_cards() {
        for (w, h) in [(1920, 1080), (1280, 720)] {
            let slices = place_slices(w, h, ScreenLayout::FuseSplitDeck, TOP_WIDTH, false, Stereo::default(), false);
            assert_eq!(slices.len(), 4);
            for (i, s) in slices.iter().enumerate() {
                let (sw, sh) = if s.shown == Shown::Bottom { (320.0, 240.0) } else { (400.0, 240.0) };
                let src_aspect = (s.uv[2] * sw) / (s.uv[3] * sh);
                let dst_aspect = s.viewport.width / s.viewport.height;
                assert!(
                    ((dst_aspect / src_aspect) - 1.0).abs() < 0.015,
                    "slice {i} aspect {dst_aspect} vs {src_aspect}"
                );
                assert!(s.viewport.x >= 0.0 && s.viewport.y >= 0.0);
                assert!(s.viewport.x + s.viewport.width <= w as f32);
                assert!(s.viewport.y + s.viewport.height <= h as f32);
                for (j, other) in slices.iter().enumerate().skip(i + 1) {
                    let separated = s.viewport.x + s.viewport.width <= other.viewport.x
                        || other.viewport.x + other.viewport.width <= s.viewport.x
                        || s.viewport.y + s.viewport.height <= other.viewport.y
                        || other.viewport.y + other.viewport.height <= s.viewport.y;
                    assert!(separated, "slice {i} overlaps slice {j} at {w}x{h}");
                }
            }
            let (_, bottom_touch) = layout(w, h, ScreenLayout::FuseSplitDeck, TOP_WIDTH, false);
            assert_eq!(bottom_touch.unwrap(), slices[3].viewport);
        }
    }
}
