//! Unified Kid Icarus: Uprising parchment settings page with smooth page-navigation
//! transitions and category tabs (Graphics, Audio & Window, Controls, Controller, System).

use egui::{pos2, vec2, Color32, CornerRadius, FontId, Margin, RichText, Sense, Shape, Stroke, Ui};

use crate::gamepad::{button_label, GamepadSnapshot};
use crate::menus::{
    heavy_title, key_name, parchment_button, parchment_frame, smootherstep, vine, Action, Menus, SettingsTab,
    CONTROLS_HINT_ENABLED, INK, PAPER, PAPER_LINE, PAPER_SHADOW, RED,
};
use crate::settings::{
    ButtonLabels, CameraSource, Filter, Keys, Language, PadButtons, Renderer, Screens, Settings, Stereo3d, TopShape,
};

/// Parchment inner margin around the unified settings page.
const MARGIN: i8 = 18;
const MUTED_INK: Color32 = Color32::from_rgb(84, 74, 66);

impl Menus {
    /// Unified Kid Icarus parchment settings page with animated horizontal slide navigation.
    /// Shown whenever `options_open` is true, or in KIU mode whenever `settings_open` is true
    /// (and during the brief slide-out transition after closing).
    pub fn options(&mut self, ctx: &egui::Context, settings: &mut Settings, pad_snapshot: Option<&GamepadSnapshot>) -> Vec<Action> {
        let mut actions = Vec::new();
        let eased = self.step_settings_anim(ctx);
        let vis = ((eased - 0.35) / 0.65).clamp(0.0, 1.0);
        if vis <= 0.01 {
            if !self.parchment_settings_target() {
                self.rebinding = None;
                self.rebinding_pad = None;
            }
            return actions;
        }

        let before = settings.clone();
        let screen = ctx.content_rect();
        let outer_pad = 32.0_f32;
        let width = (screen.width() - outer_pad).clamp(320.0, 660.0);
        let max_height = (screen.height() - outer_pad).clamp(240.0, 610.0);
        let slide = (screen.width() * 0.24).clamp(120.0, 300.0);
        let offset_x = slide * (1.0 - eased);
        let frame = parchment_frame(MARGIN, vis);

        let mut close_requested = false;

        egui::Window::new("Settings")
            .id(egui::Id::new("kiu options"))
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .interactable(vis >= 0.95)
            .anchor(egui::Align2::CENTER_CENTER, vec2(offset_x, 0.0))
            .frame(frame)
            .min_width(width)
            .max_width(width)
            .max_height(max_height)
            .show(ctx, |ui| {
                ui.multiply_opacity(vis);
                paper(ui.visuals_mut());
                ui.set_width(width - 2.0 * MARGIN as f32);

                let (title, _) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::hover());
                heavy_title(ui.painter(), "SETTINGS", title.center().x, title.min.y, 1.0);
                ui.add_space(6.0);

                // Category tab bar (centered row on desktop, wraps cleanly on narrow screens).
                self.draw_tab_bar(ui);
                ui.add_space(10.0);

                // Scrollable tab content sized to leave room for header + pinned footer.
                let scroll_max_h = (max_height - 215.0).max(110.0);
                let scroll_min_h = scroll_max_h.min(340.0);
                let tab_eased = smootherstep(self.tab_anim);
                let tab_alpha = (0.35 + 0.65 * tab_eased).clamp(0.0, 1.0);
                let tab_shift = ((1.0 - tab_eased) * 14.0 * self.tab_dir).clamp(-14.0, 14.0);

                egui::ScrollArea::vertical()
                    .id_salt(("kiu_settings_scroll", self.settings_tab.index()))
                    .min_scrolled_height(scroll_min_h)
                    .max_height(scroll_max_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.multiply_opacity(tab_alpha);
                        let content_w = (ui.available_width() - 8.0).max(240.0);
                        ui.horizontal(|ui| {
                            if tab_shift > 0.25 {
                                ui.add_space(tab_shift);
                            }
                            ui.vertical(|ui| {
                                ui.set_width((content_w - tab_shift.abs()).max(220.0));
                                match self.settings_tab {
                                    SettingsTab::Graphics => tab_graphics(ui, settings),
                                    SettingsTab::AudioWindow => tab_audio_window(ui, settings, &mut actions),
                                    SettingsTab::Controls => tab_controls(ui, self, settings),
                                    SettingsTab::Controller => tab_controller(ui, self, settings, pad_snapshot, &mut actions),
                                    SettingsTab::System => tab_system(ui, self.kiu_mode, settings, &mut actions),
                                }
                                ui.add_space(6.0);
                            });
                        });
                    });

                ui.add_space(10.0);
                // Subtle parchment rule above pinned Back bar.
                let (rule_rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 2.0), Sense::hover());
                ui.painter().line_segment(
                    [pos2(rule_rect.min.x + 8.0, rule_rect.center().y), pos2(rule_rect.max.x - 8.0, rule_rect.center().y)],
                    Stroke::new(1.0, PAPER_SHADOW),
                );
                ui.add_space(10.0);

                // Pinned bottom Back button, centered.
                let back_w = 148.0_f32;
                let back_lead = ((ui.available_width() - back_w) * 0.5).max(0.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.add_space(back_lead);
                    if parchment_button(ui, "← Back", vec2(back_w, 32.0), false).clicked() {
                        close_requested = true;
                    }
                });
                ui.add_space(10.0);

                footer(ui, "Esc goes back. Changes save automatically.", vis);

                // Decorative black vines in the parchment's top corners.
                let edge = ui.min_rect().expand(MARGIN as f32);
                let painter = ui.painter().with_clip_rect(ctx.content_rect());
                vine(&painter, edge.min + vec2(7.0, 7.0), vec2(1.0, 0.0), vec2(0.0, 1.0), 36.0);
                vine(&painter, pos2(edge.max.x - 7.0, edge.min.y + 7.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), 36.0);
            });

        if close_requested {
            self.options_open = false;
            self.settings_open = false;
            self.rebinding = None;
            self.rebinding_pad = None;
        }
        if *settings != before {
            actions.push(Action::Settings);
        }
        actions
    }

    fn draw_tab_bar(&mut self, ui: &mut Ui) {
        let gap = 8.0_f32;
        let widths = [96.0_f32, 134.0, 96.0, 106.0, 90.0];
        let total_row_w: f32 = widths.iter().sum::<f32>() + gap * (widths.len() as f32 - 1.0);
        let avail = ui.available_width();

        if avail >= total_row_w {
            let lead = (avail - total_row_w) * 0.5;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.add_space(lead);
                for (i, (&tab, &w)) in SettingsTab::ALL.iter().zip(&widths).enumerate() {
                    if i > 0 {
                        ui.add_space(gap);
                    }
                    let active = self.settings_tab == tab;
                    if parchment_button(ui, tab.label(), vec2(w, 30.0), active).clicked() && !active {
                        self.tab_dir = if tab.index() >= self.settings_tab.index() { 1.0 } else { -1.0 };
                        self.settings_tab = tab;
                        self.tab_anim = 0.0;
                        self.rebinding = None;
                        self.rebinding_pad = None;
                    }
                }
            });
        } else {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = vec2(6.0, 6.0);
                for (&tab, &w) in SettingsTab::ALL.iter().zip(&widths) {
                    let active = self.settings_tab == tab;
                    let compact_w = w.min(avail * 0.46);
                    if parchment_button(ui, tab.label(), vec2(compact_w, 28.0), active).clicked() && !active {
                        self.tab_dir = if tab.index() >= self.settings_tab.index() { 1.0 } else { -1.0 };
                        self.settings_tab = tab;
                        self.tab_anim = 0.0;
                        self.rebinding = None;
                        self.rebinding_pad = None;
                    }
                }
            });
        }
    }
}

fn label_col_width(ui: &Ui) -> f32 {
    (ui.available_width() * 0.36).clamp(116.0, 195.0)
}

fn combo_width(ui: &Ui) -> f32 {
    (ui.available_width() - label_col_width(ui) - 24.0).clamp(150.0, 320.0)
}

fn tab_graphics(ui: &mut Ui, settings: &mut Settings) {
    let col_w = label_col_width(ui);
    let cw = combo_width(ui);

    plate(ui, "RENDERING");
    egui::Grid::new("kiu_gfx_render")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Backend", "Graphics API used to present frames. Takes effect on next launch.", |ui| {
                egui::ComboBox::from_id_salt("kiu_renderer")
                    .width(cw)
                    .selected_text(match settings.renderer {
                        Renderer::Vulkan => "Vulkan",
                        Renderer::OpenGl => "OpenGL",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut settings.renderer, Renderer::Vulkan, "Vulkan");
                        ui.selectable_value(&mut settings.renderer, Renderer::OpenGl, "OpenGL");
                    })
                    .response
            });
            row(ui, "Hardware 3D rasterizer", "Draw 3D scenes on your desktop GPU instead of the software rasterizer.", |ui| {
                ui.checkbox(&mut settings.hardware_rasterizer, "")
            });
            let gpu = settings.hardware_rasterizer;
            row(ui, "Internal resolution", "Internal 3D rendering resolution (changes live; up to 16x = 6400x3840).", |ui| {
                ui.add_enabled(
                    gpu,
                    egui::Slider::new(&mut settings.resolution, 1..=zakuro_gpu::MAX_SCALE).custom_formatter(|v, _| {
                        let s = v.round() as u32;
                        format!("{s}x ({}p)", s * 240)
                    }),
                )
            });
            let af_label = |v: u32| if v <= 1 { "Off".to_owned() } else { format!("{v}x") };
            row(ui, "Anisotropic filtering", "Sharpens 3D textures viewed at shallow angles with negligible cost on desktop GPUs.", |ui| {
                ui.add_enabled_ui(gpu && settings.renderer == Renderer::Vulkan, |ui| {
                    egui::ComboBox::from_id_salt("kiu_af")
                        .width(cw)
                        .selected_text(af_label(settings.anisotropy))
                        .show_ui(ui, |ui| {
                            for level in [1u32, 2, 4, 8, 16] {
                                ui.selectable_value(&mut settings.anisotropy, level, af_label(level));
                            }
                        });
                })
                .response
            });
            let interpolates = gpu && settings.renderer == Renderer::Vulkan;
            row(ui, "Smoother motion (60 fps)", "Draws an interpolated frame between each pair of 30 fps frames (requires GPU + Vulkan).", |ui| {
                ui.add_enabled(interpolates, egui::Checkbox::without_text(&mut settings.frame_interpolation))
            });
            row(ui, "VSync", "Syncs frames to display refresh. Disable for lowest mouse-aim latency on high-Hz monitors.", |ui| {
                ui.checkbox(&mut settings.vsync, "")
            });
            row(ui, "Texture packs", "Load custom texture packs placed in the game's mods/textures folder.", |ui| {
                ui.add_enabled(gpu, egui::Checkbox::without_text(&mut settings.texture_packs))
            });
        });

    ui.add_space(12.0);
    plate(ui, "DISPLAY");
    egui::Grid::new("kiu_gfx_display")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Screen layout", "How the dual screens are arranged. Press F9 while playing to cycle, F10 to swap.", |ui| {
                egui::ComboBox::from_id_salt("kiu_screens")
                    .width(cw)
                    .selected_text(settings.layout.name())
                    .show_ui(ui, |ui| {
                        for layout in Screens::ALL {
                            ui.selectable_value(&mut settings.layout, layout, layout.name());
                        }
                    })
                    .response
            });
            row(ui, "Upscaling filter", "How screen pixels are filtered when scaled to your window.", |ui| {
                egui::ComboBox::from_id_salt("kiu_filter")
                    .width(cw)
                    .selected_text(settings.filter.name())
                    .show_ui(ui, |ui| {
                        for filter in Filter::ALL {
                            ui.selectable_value(&mut settings.filter, filter, filter.name());
                        }
                    })
                    .response
            });
            row(ui, "Window scale", "Default window size multiplier.", |ui| {
                ui.add(egui::Slider::new(&mut settings.scale, 1..=9))
            });
            row(ui, "Whole pixels only", "Scale by integer multiples only, adding clean borders when needed.", |ui| {
                ui.checkbox(&mut settings.integer_scale, "")
            });
            row(ui, "Top screen shape", "Aspect ratio for the top screen (use widescreen with a matching widescreen patch).", |ui| {
                egui::ComboBox::from_id_salt("kiu_top_shape")
                    .width(cw)
                    .selected_text(settings.top_shape.name())
                    .show_ui(ui, |ui| {
                        for shape in TopShape::ALL {
                            ui.selectable_value(&mut settings.top_shape, shape, shape.name());
                        }
                    })
                    .response
            });
        });

    ui.add_space(12.0);
    plate(ui, "STEREOSCOPIC 3D");
    egui::Grid::new("kiu_gfx_3d")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "3D mode", "Stereoscopic 3D output mode for glasses, 3D displays, or headsets.", |ui| {
                egui::ComboBox::from_id_salt("kiu_3d")
                    .width(cw)
                    .selected_text(settings.stereo.name())
                    .show_ui(ui, |ui| {
                        for stereo in Stereo3d::ALL {
                            ui.selectable_value(&mut settings.stereo, stereo, stereo.name());
                        }
                    })
                    .response
            });
            let stereo = settings.stereo != Stereo3d::Off;
            row(ui, "3D depth", "Stereoscopic depth slider intensity.", |ui| {
                ui.add_enabled(stereo, egui::Slider::new(&mut settings.depth_3d, 0..=100).suffix("%"))
            });
            row(ui, "Swap eyes", "Swap left and right eye views.", |ui| {
                ui.add_enabled(stereo, egui::Checkbox::without_text(&mut settings.swap_eyes))
            });
        });

    ui.add_space(6.0);
    ui.label(
        RichText::new("F9 cycles layouts in-game, F10 swaps single-screen view. Backend changes apply on next launch.")
            .small()
            .color(MUTED_INK),
    );
}

fn tab_audio_window(ui: &mut Ui, settings: &mut Settings, actions: &mut Vec<Action>) {
    let col_w = label_col_width(ui);

    plate(ui, "SOUND");
    egui::Grid::new("kiu_sound")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Master volume", "Overall game audio volume.", |ui| {
                ui.add(
                    egui::Slider::new(&mut settings.volume, 0.0..=1.0)
                        .custom_formatter(|value, _| format!("{:.0}%", value * 100.0)),
                )
            });
            row(ui, "Mute", "Silence all game audio.", |ui| ui.checkbox(&mut settings.mute, ""));
        });

    ui.add_space(12.0);
    plate(ui, "WINDOW & OVERLAY");
    egui::Grid::new("kiu_window")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Fullscreen", "Toggle fullscreen window mode (F11 also toggles it anytime).", |ui| {
                let response = parchment_button(ui, "Toggle Fullscreen", vec2(150.0, 28.0), false);
                if response.clicked() {
                    actions.push(Action::Fullscreen);
                }
                response
            });
            row(ui, "Pause in background", "Pause automatically when another window takes focus (disabled in multiplayer).", |ui| {
                ui.checkbox(&mut settings.pause_unfocused, "")
            });
            row(ui, "Recapture mouse", "Restore active mouse-aim mode when returning focus to the game window.", |ui| {
                ui.checkbox(&mut settings.recapture_mouse, "")
            });
            row(ui, "Discord status", "Show whether you're in menus, flying, or on foot on your Discord profile.", |ui| {
                ui.checkbox(&mut settings.discord_presence, "")
            });
            row(ui, "Show frame rate", "Display the live FPS counter in the top-left corner.", |ui| {
                ui.checkbox(&mut settings.show_fps, "")
            });
            if CONTROLS_HINT_ENABLED {
                row(ui, "Show controls panel", "Show the parchment controls guide beside the screens when space allows.", |ui| {
                    ui.checkbox(&mut settings.show_controls, "")
                });
            }
        });
}

fn tab_controls(ui: &mut Ui, menus: &mut Menus, settings: &mut Settings) {
    let col_w = label_col_width(ui);

    plate(ui, "MOUSE AIM");
    egui::Grid::new("kiu_mouse_aim")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Aim speed, flying (2)", "Reticle sensitivity in air battles.", |ui| {
                ui.add(egui::Slider::new(&mut settings.air_aim_speed, 0.1..=3.0))
            });
            row(ui, "Aim speed, on foot (3)", "Camera turn sensitivity in land battles.", |ui| {
                ui.add(egui::Slider::new(&mut settings.land_aim_speed, 0.1..=5.0))
            });
            row(ui, "Invert vertical aim", "Move aim up when pulling the mouse down.", |ui| {
                ui.checkbox(&mut settings.invert_aim_y, "")
            });
        });

    ui.add_space(12.0);
    plate(ui, "KEYBOARD BINDINGS");
    let wide_enough = ui.available_width() >= 470.0;
    let btn_w = if wide_enough { 112.0 } else { 126.0 };
    egui::Grid::new("kiu_kb_grid")
        .num_columns(if wide_enough { 4 } else { 2 })
        .spacing([12.0, 6.0])
        .show(ui, |ui| {
            for (i, (name, key)) in settings.keys.all_mut().into_iter().enumerate() {
                ui.label(name);
                let active = menus.rebinding == Some(i);
                let text = if active { "Press a key...".to_owned() } else { key_name(*key) };
                if parchment_button(ui, &text, vec2(btn_w, 26.0), active).clicked() {
                    menus.rebinding = Some(i);
                    menus.rebinding_pad = None;
                }
                if !wide_enough || i % 2 == 1 {
                    ui.end_row();
                }
            }
        });

    ui.add_space(8.0);
    if parchment_button(ui, "Reset Default Keys", vec2(160.0, 28.0), false).clicked() {
        settings.keys = Keys::default();
        menus.rebinding = None;
    }
    ui.add_space(4.0);
    ui.label(
        RichText::new("Press Esc while rebinding to cancel. Right-drag tilts the motion sensor.")
            .small()
            .color(MUTED_INK),
    );
}

fn stick_scope(ui: &mut Ui, tag: &str, stick: (f32, f32), deadzone: f32) {
    let size = 46.0_f32;
    let (rect, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter();
    let center = rect.center();
    let radius = size * 0.5 - 2.0;

    // Outer parchment dial + crosshair.
    painter.circle_filled(center, radius, PAPER);
    painter.circle_stroke(center, radius, Stroke::new(1.3, INK));
    painter.line_segment([pos2(center.x - radius + 3.0, center.y), pos2(center.x + radius - 3.0, center.y)], Stroke::new(0.8, PAPER_SHADOW));
    painter.line_segment([pos2(center.x, center.y - radius + 3.0), pos2(center.x, center.y + radius - 3.0)], Stroke::new(0.8, PAPER_SHADOW));

    // Live deadzone ring.
    let dz_r = (deadzone.clamp(0.03, 0.45) * radius).max(2.0);
    painter.circle_filled(center, dz_r, Color32::from_rgba_unmultiplied(222, 32, 36, 26));
    painter.circle_stroke(center, dz_r, Stroke::new(1.0, Color32::from_rgba_unmultiplied(222, 32, 36, 140)));

    // Tiny corner tag (`L` / `R`).
    let label = painter.layout_no_wrap(tag.to_owned(), FontId::proportional(9.5), MUTED_INK);
    painter.galley(pos2(rect.min.x + 2.0, rect.min.y), label, MUTED_INK);

    // Live stick dot (up is +Y from gilrs -> -Y on screen).
    let mag = stick.0.hypot(stick.1);
    let (sx, sy) = if mag > 1.0 { (stick.0 / mag, stick.1 / mag) } else { stick };
    let dot_pos = center + vec2(sx * (radius - 3.5), -sy * (radius - 3.5));
    let active = mag > deadzone;
    painter.circle_filled(dot_pos, 3.6, if active { RED } else { INK });
    painter.circle_stroke(dot_pos, 3.6, Stroke::new(1.0, Color32::WHITE));
}

fn tab_controller(
    ui: &mut Ui,
    menus: &mut Menus,
    settings: &mut Settings,
    pad_snapshot: Option<&GamepadSnapshot>,
    actions: &mut Vec<Action>,
) {
    let default_snap = GamepadSnapshot::default();
    let snap = pad_snapshot.unwrap_or(&default_snap);
    let effective_style = match settings.pad_labels {
        ButtonLabels::Auto => snap.detected_style,
        other => other,
    };

    if !snap.connected.is_empty() || snap.gyro_device.is_some() {
        // Keep the live stick scopes responsive while a controller is attached.
        ui.ctx().request_repaint();
    }

    // Connected controller banner + live L/R stick & deadzone preview.
    egui::Frame::new()
        .fill(PAPER_LINE)
        .stroke(Stroke::new(1.2, PAPER_SHADOW))
        .corner_radius(6)
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let (dot_rect, _) = ui.allocate_exact_size(vec2(10.0, 14.0), Sense::hover());
                        let any_connected = !snap.connected.is_empty() || snap.gyro_device.is_some();
                        let dot_color = if !any_connected {
                            Color32::from_rgb(154, 142, 130)
                        } else {
                            Color32::from_rgb(44, 142, 74)
                        };
                        ui.painter().circle_filled(dot_rect.center(), 4.2, dot_color);
                        ui.painter().circle_stroke(dot_rect.center(), 4.2, Stroke::new(1.0, INK));
                        let heading = match snap.connected.as_slice() {
                            [] => snap
                                .gyro_device
                                .as_ref()
                                .map(|g| format!("{g} (HID)"))
                                .unwrap_or_else(|| "No controller detected".to_owned()),
                            [one] => one.clone(),
                            [first, rest @ ..] => format!("{first} (+{} more)", rest.len()),
                        };
                        ui.label(RichText::new(heading).strong().color(INK));
                    });
                    ui.add_space(2.0);
                    let status_sub = match &snap.gyro_device {
                        Some(g) => format!("Gyro active ({g}) · Hold R / Y / L3 to pause gyro while ratcheting."),
                        None => "Both triggers (LT/RT) and LB fire in battle. Supports DualSense, DS4, Switch Pro & Steam Deck gyro.".to_owned(),
                    };
                    ui.label(RichText::new(status_sub).small().color(MUTED_INK));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    stick_scope(ui, "R", snap.right_stick, settings.pad_deadzone);
                    ui.add_space(6.0);
                    stick_scope(ui, "L", snap.left_stick, settings.pad_deadzone);
                });
            });
        });

    ui.add_space(10.0);
    plate(ui, "LAYOUT PRESET");
    let presets = [
        ("Default (All Fire)", PadButtons::default()),
        ("Swap RB / RT", PadButtons::swap_r_zr()),
        ("No ZL / ZR", PadButtons::no_secondary_triggers()),
    ];
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(8.0, 6.0);
        for (name, preset) in presets {
            let active = settings.pad == preset;
            if parchment_button(ui, name, vec2(138.0, 28.0), active).clicked() {
                settings.pad = preset;
                menus.rebinding_pad = None;
            }
        }
    });
    ui.add_space(8.0);
    let col_w = label_col_width(ui);
    let cw = combo_width(ui);
    egui::Grid::new("kiu_pad_labels")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            let auto_suffix = if settings.pad_labels == ButtonLabels::Auto {
                match snap.detected_style {
                    ButtonLabels::Xbox => "Auto (Xbox / Deck)",
                    ButtonLabels::PlayStation => "Auto (PlayStation)",
                    ButtonLabels::Nintendo => "Auto (Nintendo)",
                    _ => "Auto (Generic)",
                }
            } else {
                settings.pad_labels.name()
            };
            row(ui, "Button labels", "Choose button names matching your controller (does not change bindings).", |ui| {
                egui::ComboBox::from_id_salt("kiu_pad_label_style")
                    .width(cw)
                    .selected_text(auto_suffix)
                    .show_ui(ui, |ui| {
                        for style in ButtonLabels::ALL {
                            ui.selectable_value(&mut settings.pad_labels, style, style.name());
                        }
                    })
                    .response
            });
        });

    ui.add_space(12.0);
    plate(ui, "STICK & GYRO AIM");
    egui::Grid::new("kiu_pad_aim")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "Aim speed, flying", "Right-stick reticle speed multiplier in air battles.", |ui| {
                ui.add(
                    egui::Slider::new(&mut settings.pad_air_speed, 0.3..=2.5)
                        .custom_formatter(|v, _| format!("{v:.2}x")),
                )
            });
            row(ui, "Aim speed, on foot", "Right-stick camera turn speed multiplier in land battles.", |ui| {
                ui.add(
                    egui::Slider::new(&mut settings.pad_land_speed, 0.3..=2.5)
                        .custom_formatter(|v, _| format!("{v:.2}x")),
                )
            });
            row(ui, "Stick deadzone", "Inner stick radius ignored to prevent drift (shown by red ring above).", |ui| {
                ui.add(
                    egui::Slider::new(&mut settings.pad_deadzone, 0.03..=0.30)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                )
            });
            row(ui, "Invert vertical aim", "Aim up when pulling the right stick or tilting down.", |ui| {
                ui.checkbox(&mut settings.pad_invert_y, "")
            });
            row(ui, "Gyro aiming", "Enable motion aiming on DualSense, DualShock 4, Switch Pro, and Steam Deck.", |ui| {
                ui.checkbox(&mut settings.pad_gyro, "")
            });
            row(ui, "Gyro sensitivity", "Motion-aim sensitivity multiplier.", |ui| {
                ui.add_enabled(
                    settings.pad_gyro,
                    egui::Slider::new(&mut settings.pad_gyro_speed, 0.2..=3.0)
                        .custom_formatter(|v, _| format!("{v:.2}x")),
                )
            });
            row(ui, "Zero-drift calibration", "Recalibrate resting gyro offset if you notice any slow drift.", |ui| {
                let resp = parchment_button(ui, "Calibrate Gyro", vec2(138.0, 26.0), false);
                if resp.clicked() {
                    actions.push(Action::CalibrateGyro);
                }
                resp
            });
        });

    ui.add_space(12.0);
    plate(ui, "ACTION BINDINGS");
    let mut all = settings.pad.all_mut();
    egui::Grid::new("kiu_pad_actions")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            for (idx, action_name, optional) in PadButtons::KIU_ACTIONS {
                ui.label(action_name);
                let active = menus.rebinding_pad == Some(idx);
                let bound = *all[idx].1;
                let text = if active {
                    "Press button..."
                } else {
                    button_label(bound, effective_style)
                };
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    if parchment_button(ui, text, vec2(138.0, 26.0), active).clicked() {
                        menus.rebinding_pad = Some(idx);
                        menus.rebinding = None;
                    }
                    if optional && bound != gilrs::Button::Unknown {
                        if parchment_button(ui, "×", vec2(28.0, 26.0), false)
                            .on_hover_text("Clear optional binding (leave unbound)")
                            .clicked()
                        {
                            *all[idx].1 = gilrs::Button::Unknown;
                            if menus.rebinding_pad == Some(idx) {
                                menus.rebinding_pad = None;
                            }
                        }
                    }
                });
                ui.end_row();
            }
        });

    ui.add_space(8.0);
    ui.label(
        RichText::new("Press Esc to cancel rebinding, or Backspace to unbind. Fixed sticks: Left = Move, Right = Aim, L3 = Center camera, R3 = Bottom screen, Home = Pause.")
            .small()
            .color(MUTED_INK),
    );
}

fn tab_system(ui: &mut Ui, kiu_mode: bool, settings: &mut Settings, actions: &mut Vec<Action>) {
    let col_w = label_col_width(ui);
    let cw = combo_width(ui);

    plate(ui, if kiu_mode { "GAME DATA" } else { "GAMES" });
    egui::Grid::new("kiu_sys_rom")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            if kiu_mode {
                row(ui, "Imported ROM", "Your decrypted USA Kid Icarus: Uprising ROM copied into app data.", |ui| {
                    ui.vertical(|ui| {
                        match &settings.rom_path {
                            Some(path) => {
                                let display = path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| path.display().to_string());
                                ui.label(RichText::new(display).strong().color(INK))
                                    .on_hover_text(path.display().to_string());
                            }
                            None => {
                                ui.label(RichText::new("Not imported yet").color(MUTED_INK));
                            }
                        }
                        ui.add_space(4.0);
                        if parchment_button(ui, "Change ROM...", vec2(138.0, 28.0), false).clicked() {
                            actions.push(Action::ChangeRom);
                        }
                    })
                    .response
                });
            } else {
                row(ui, "Games folder", "Folder scanned for titles.", |ui| {
                    ui.horizontal(|ui| {
                        match &settings.games {
                            Some(folder) => ui.label(folder.display().to_string()),
                            None => ui.label(RichText::new("none").color(MUTED_INK)),
                        };
                        if parchment_button(ui, "Choose...", vec2(96.0, 26.0), false).clicked() {
                            actions.push(Action::ChooseFolder);
                        }
                    })
                    .response
                });
            }
        });

    ui.add_space(12.0);
    plate(ui, "EMULATION");
    egui::Grid::new("kiu_sys_emu")
        .num_columns(2)
        .min_col_width(col_w)
        .spacing([16.0, 8.0])
        .show(ui, |ui| {
            row(ui, "System language", "Console language used for in-game text and voices. Applies on next launch or Reset.", |ui| {
                egui::ComboBox::from_id_salt("kiu_lang")
                    .width(cw)
                    .selected_text(settings.language.name())
                    .show_ui(ui, |ui| {
                        for language in Language::ALL {
                            ui.selectable_value(&mut settings.language, language, language.name());
                        }
                    })
                    .response
            });
            row(ui, "Camera source", "What the virtual camera sees (useful for AR Idol Cards).", |ui| {
                egui::ComboBox::from_id_salt("kiu_cam")
                    .width(cw)
                    .selected_text(settings.camera.name())
                    .show_ui(ui, |ui| {
                        for source in CameraSource::ALL {
                            ui.selectable_value(&mut settings.camera, source, source.name());
                        }
                    })
                    .response
            });
            if settings.camera == CameraSource::Picture {
                row(ui, "Camera picture", "Static image fed to the virtual camera.", |ui| {
                    ui.horizontal(|ui| {
                        match &settings.camera_picture {
                            Some(path) => ui.label(
                                path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                            ),
                            None => ui.label(RichText::new("none").color(MUTED_INK)),
                        };
                        if parchment_button(ui, "Choose...", vec2(96.0, 26.0), false).clicked() {
                            actions.push(Action::ChooseCameraPicture);
                        }
                    })
                    .response
                });
            }
            row(ui, "CPU execution", "Native AOT recompiled code runs at full speed; interpreter is for debugging only.", |ui| {
                ui.vertical(|ui| {
                    ui.radio_value(&mut settings.recompiled, true, "Native AOT recompiled code");
                    ui.radio_value(&mut settings.recompiled, false, "Interpreter only (debug)");
                })
                .response
            });
        });

    ui.add_space(6.0);
    ui.label(
        RichText::new("Language and CPU mode changes apply on next launch or Reset.")
            .small()
            .color(MUTED_INK),
    );
}

/// A single settings row: label on the left, control on the right, both with hover explanation.
fn row(ui: &mut Ui, name: &str, hint: &str, add: impl FnOnce(&mut Ui) -> egui::Response) {
    ui.label(name).on_hover_text(hint);
    add(ui).on_hover_text(hint);
    ui.end_row();
}

/// Section heading: white uppercase text on a slanted black Kid Icarus plate.
fn plate(ui: &mut Ui, name: &str) {
    let (slant, pad) = (4.0, 7.0);
    let galley = ui.painter().layout_no_wrap(name.to_owned(), FontId::proportional(11.0), Color32::WHITE);
    let (rect, _) = ui.allocate_exact_size(galley.size() + vec2(2.0 * pad + slant, 4.0), Sense::hover());
    let points = vec![
        pos2(rect.min.x + slant, rect.min.y),
        rect.right_top(),
        pos2(rect.max.x - slant, rect.max.y),
        rect.left_bottom(),
    ];
    ui.painter().add(Shape::convex_polygon(points, INK, Stroke::NONE));
    ui.painter()
        .galley(pos2(rect.min.x + pad + slant / 2.0, rect.center().y - galley.size().y / 2.0), galley, Color32::WHITE);
    ui.add_space(4.0);
}

/// Dark ink advice strip at the foot of the parchment card.
fn footer(ui: &mut Ui, text: &str, alpha: f32) {
    let corner = CornerRadius { nw: 0, ne: 0, sw: 8, se: 8 };
    egui::Frame::new()
        .fill(INK.gamma_multiply(alpha))
        .corner_radius(corner)
        .inner_margin(Margin::symmetric(MARGIN, 6))
        .outer_margin(Margin { left: -MARGIN, right: -MARGIN, top: 0, bottom: -MARGIN })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.vertical_centered(|ui| {
                ui.label(RichText::new(text).color(Color32::WHITE).small());
            });
        });
}

/// Parchment widget styling: crisp ink on white, crimson when hovered/selected.
fn paper(visuals: &mut egui::Visuals) {
    visuals.override_text_color = Some(INK);
    visuals.extreme_bg_color = Color32::WHITE;
    visuals.slider_trailing_fill = true;
    visuals.selection.bg_fill = RED;
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    let widgets = &mut visuals.widgets;
    for (widget, edge) in [(&mut widgets.inactive, INK), (&mut widgets.hovered, RED), (&mut widgets.active, RED), (&mut widgets.open, RED)] {
        widget.bg_fill = Color32::WHITE;
        widget.weak_bg_fill = Color32::WHITE;
        widget.bg_stroke = Stroke::new(1.2, edge);
        widget.fg_stroke = Stroke::new(1.5, INK);
    }
    widgets.hovered.weak_bg_fill = PAPER_LINE;
    widgets.noninteractive.bg_stroke = Stroke::new(1.0, INK);
    widgets.noninteractive.fg_stroke = Stroke::new(1.0, INK);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unified settings page on a screen this size, after it settles.
    fn shown(width: f32, height: f32) -> (egui::Rect, egui::Rect) {
        let ctx = egui::Context::default();
        let mut menus = Menus::default();
        menus.kiu_mode = true;
        menus.settings_open = true;
        let mut settings = Settings::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(width, height));
        for _ in 0..3 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            ctx.run_ui(input, |ui| {
                assert!(menus.options(ui.ctx(), &mut settings, None).is_empty(), "nothing changed, nothing to do");
            })
            .drop_without_applying_deltas();
        }
        (ctx.memory(|memory| memory.area_rect(egui::Id::new("kiu options"))).unwrap(), screen)
    }

    #[test]
    fn the_options_fit_the_smallest_window_and_a_wide_one() {
        for (width, height) in [(400.0, 480.0), (1280.0, 720.0)] {
            let (window, screen) = shown(width, height);
            assert!(screen.contains_rect(window), "{window:?} in {screen:?}");
        }
    }

    #[test]
    fn the_options_open_tall_enough_to_read_on_a_big_window() {
        let (window, _) = shown(1920.0, 1080.0);
        assert!(window.height() > 300.0, "{window:?}");
    }

    #[test]
    fn closed_options_show_nothing() {
        let ctx = egui::Context::default();
        let mut menus = Menus::default();
        let mut settings = Settings::default();
        ctx.run_ui(egui::RawInput::default(), |ui| {
            menus.options(ui.ctx(), &mut settings, None);
        })
        .drop_without_applying_deltas();
        assert!(ctx.memory(|memory| memory.area_rect(egui::Id::new("kiu options"))).is_none());
    }

    #[test]
    fn all_settings_tabs_render_cleanly_within_viewport() {
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1280.0, 720.0));
        for tab in SettingsTab::ALL {
            let ctx = egui::Context::default();
            let mut menus = Menus::default();
            menus.kiu_mode = true;
            menus.settings_open = true;
            menus.settings_tab = tab;
            let mut settings = Settings::default();
            for _ in 0..3 {
                let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
                ctx.run_ui(input, |ui| {
                    assert!(menus.options(ui.ctx(), &mut settings, None).is_empty());
                })
                .drop_without_applying_deltas();
            }
            let rect = ctx.memory(|m| m.area_rect(egui::Id::new("kiu options"))).unwrap();
            assert!(screen.contains_rect(rect), "tab {tab:?}: {rect:?} not in {screen:?}");
        }
    }

    #[test]
    fn page_navigation_animates_smoothly_between_setup_and_settings() {
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1280.0, 720.0));
        let mut menus = Menus::default();
        menus.kiu_mode = true;
        let mut settings = Settings::default();

        // Frame 0: settled on setup page (anim == 0.0).
        ctx.run_ui(egui::RawInput { screen_rect: Some(screen), time: Some(0.0), ..Default::default() }, |ui| {
            menus.kiu_setup(ui, &crate::menus::KiuSetupStage::NeedsRom { notice: None, hovered_file: false });
            menus.options(ui.ctx(), &mut settings, None);
        })
        .drop_without_applying_deltas();
        assert_eq!(menus.settings_anim, 0.0);

        // User opens Settings -> advance through midpoint and settle.
        menus.settings_open = true;
        let mut saw_mid = false;
        for step in 1..=20 {
            let t = step as f64 * 0.025;
            ctx.run_ui(egui::RawInput { screen_rect: Some(screen), time: Some(t), ..Default::default() }, |ui| {
                menus.kiu_setup(ui, &crate::menus::KiuSetupStage::NeedsRom { notice: None, hovered_file: false });
                menus.options(ui.ctx(), &mut settings, None);
            })
            .drop_without_applying_deltas();
            if menus.settings_anim > 0.1 && menus.settings_anim < 0.9 {
                saw_mid = true;
            }
        }
        assert!(saw_mid, "expected smooth intermediate animation frames");
        assert!((menus.settings_anim - 1.0).abs() < 1e-4, "expected settled at 1.0, got {}", menus.settings_anim);

        // User navigates back -> animates smoothly back to 0.0.
        menus.settings_open = false;
        let mut saw_back_mid = false;
        for step in 21..=45 {
            let t = step as f64 * 0.025;
            ctx.run_ui(egui::RawInput { screen_rect: Some(screen), time: Some(t), ..Default::default() }, |ui| {
                menus.kiu_setup(ui, &crate::menus::KiuSetupStage::NeedsRom { notice: None, hovered_file: false });
                menus.options(ui.ctx(), &mut settings, None);
            })
            .drop_without_applying_deltas();
            if menus.settings_anim > 0.1 && menus.settings_anim < 0.9 {
                saw_back_mid = true;
            }
        }
        assert!(saw_back_mid, "expected smooth reverse animation frames");
        assert!(menus.settings_anim.abs() < 1e-4, "expected settled at 0.0, got {}", menus.settings_anim);
    }
}
