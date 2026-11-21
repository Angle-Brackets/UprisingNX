//! the player's settings, sdmc:/switch/uprisingnx/settings.ini: a preset of
//! resolution and smooth mode for handheld and one for docked, and the aim
//! controls. read at start, written back whenever a hotkey changes them.
//! env.txt stays for development switches; ZAKURO_RESOLUTION and
//! ZAKURO_INTERPOLATE there still win over the file.

use std::fmt::Write as _;

pub const PATH: &str = "sdmc:/switch/uprisingnx/settings.ini";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    /// 60 pictures/s: the lower resolution in battles, smooth mode on
    Performance,
    /// a sharper picture at the game's own 30: smooth mode off
    Quality,
    /// battle_scale, menu_scale and smooth as written
    Custom,
}

impl Preset {
    fn name(self) -> &'static str {
        match self {
            Preset::Performance => "performance",
            Preset::Quality => "quality",
            Preset::Custom => "custom",
        }
    }

    fn parse(text: &str) -> Option<Preset> {
        match text {
            "performance" => Some(Preset::Performance),
            "quality" => Some(Preset::Quality),
            "custom" => Some(Preset::Custom),
            _ => None,
        }
    }

    pub fn next(self) -> Preset {
        match self {
            Preset::Performance => Preset::Quality,
            Preset::Quality => Preset::Custom,
            Preset::Custom => Preset::Performance,
        }
    }
}

/// what one mode (handheld or docked) runs at
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Video {
    pub battle_scale: u32,
    pub menu_scale: u32,
    pub smooth: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mode {
    pub preset: Preset,
    /// the custom preset's values, kept while another preset is chosen
    pub custom: Video,
}

impl Mode {
    /// what runs now. measured on Chapter 14's flight: handheld 1x keeps
    /// smooth mode at 52-60 pictures/s, 2x holds a locked 30 without it;
    /// docked 3x keeps 28-30 frames in battles, 4x falls to ~20
    pub fn video(&self, docked: bool) -> Video {
        match (self.preset, docked) {
            (Preset::Performance, false) => Video { battle_scale: 1, menu_scale: 2, smooth: true },
            (Preset::Performance, true) => Video { battle_scale: 3, menu_scale: 4, smooth: true },
            (Preset::Quality, false) => Video { battle_scale: 2, menu_scale: 2, smooth: false },
            (Preset::Quality, true) => Video { battle_scale: 4, menu_scale: 4, smooth: false },
            (Preset::Custom, _) => self.custom,
        }
    }

    /// the preset as custom, for a hotkey that changes one value of it
    pub fn customize(&mut self, docked: bool) -> &mut Video {
        self.custom = self.video(docked);
        self.preset = Preset::Custom;
        &mut self.custom
    }
}

/// the aim and HUD keys, each the env var the runtime reads
const CONTROLS: [(&str, &str); 9] = [
    ("right_stick_aim", "UNX_RIGHT_STICK_AIM"),
    ("gyro_aim", "UNX_GYRO_AIM"),
    ("gyro_speed", "UNX_GYRO_SPEED"),
    ("left_stick_aim", "UNX_LEFT_STICK_AIM"),
    ("aim_modifier", "UNX_AIM_MODIFIER"),
    ("aim_invert_y", "UNX_AIM_INVERT_Y"),
    ("aim_speed", "UNX_AIM_SPEED"),
    ("power_label", "ZAKURO_POWER_LABEL"),
    ("minus_button", "UNX_MINUS_BUTTON"),
];

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    pub handheld: Mode,
    pub docked: Mode,
    /// active screen area percentage (80..=100) when docked on a TV with overscan
    pub screen_area: u32,
    /// [controls], as written (key, value)
    pub controls: Vec<(String, String)>,
}

impl Default for Settings {
    fn default() -> Settings {
        let custom = |docked| Mode { preset: Preset::Performance, custom: Video { battle_scale: 1, menu_scale: 2, smooth: true } }.video(docked);
        Settings {
            handheld: Mode { preset: Preset::Performance, custom: custom(false) },
            docked: Mode { preset: Preset::Performance, custom: custom(true) },
            screen_area: 100,
            controls: vec![
                ("right_stick_aim".into(), "1".into()),
                ("gyro_aim".into(), "1".into()),
                ("gyro_speed".into(), "1.0".into()),
                ("left_stick_aim".into(), "0".into()),
                ("aim_modifier".into(), "ZL".into()),
                ("aim_invert_y".into(), "0".into()),
                ("aim_speed".into(), "1.0".into()),
                ("power_label".into(), "bottom".into()),
                ("minus_button".into(), "menu".into()),
            ],
        }
    }
}

fn scale(text: &str) -> Option<u32> {
    text.parse::<u32>().ok().filter(|s| (1..=4).contains(s))
}

fn on_off(text: &str) -> Option<bool> {
    match text {
        "on" | "1" | "true" | "yes" => Some(true),
        "off" | "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

impl Settings {
    /// the file's settings over the defaults; a missing file or a bad line
    /// keeps the default and is logged
    pub fn parse(text: &str) -> Settings {
        let mut settings = Settings::default();
        let mut section = String::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.split(['#', ';']).next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = name.trim().to_ascii_lowercase();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                log::warn!("unx: settings.ini line {} has no '=': {line:?}", number + 1);
                continue;
            };
            let (key, value) = (key.trim().to_ascii_lowercase(), value.trim());
            let good = match section.as_str() {
                "handheld" | "docked" => {
                    let mode = if section == "docked" { &mut settings.docked } else { &mut settings.handheld };
                    match key.as_str() {
                        "preset" => Preset::parse(&value.to_ascii_lowercase()).map(|p| mode.preset = p).is_some(),
                        "battle_scale" => scale(value).map(|s| mode.custom.battle_scale = s).is_some(),
                        "menu_scale" => scale(value).map(|s| mode.custom.menu_scale = s).is_some(),
                        "smooth" => on_off(&value.to_ascii_lowercase()).map(|s| mode.custom.smooth = s).is_some(),
                        "screen_area" if section == "docked" => {
                            value.parse::<u32>().ok().filter(|s| (80..=100).contains(s)).map(|s| settings.screen_area = s).is_some()
                        }
                        _ => false,
                    }
                }
                "controls" if CONTROLS.iter().any(|(name, _)| *name == key) => {
                    match settings.controls.iter_mut().find(|(name, _)| *name == key) {
                        Some(entry) => entry.1 = value.to_string(),
                        None => settings.controls.push((key.clone(), value.to_string())),
                    }
                    true
                }
                _ => false,
            };
            if !good {
                log::warn!("unx: settings.ini line {} not understood: {line:?}", number + 1);
            }
        }
        settings
    }

    pub fn load() -> Settings {
        match std::fs::read_to_string(PATH) {
            Ok(text) => {
                let settings = Settings::parse(&text);
                log::info!("unx: settings.ini read: {settings:?}");
                settings
            }
            Err(_) => {
                let settings = Settings::default();
                settings.save();
                log::info!("unx: no settings.ini, wrote the defaults");
                settings
            }
        }
    }

    pub fn to_text(&self) -> String {
        let mut text = String::from(
            "# UprisingNX settings. The hotkeys change these and write them back;\n\
             # you can also edit this file while the game is closed.\n\
             #\n\
             # preset: performance (smooth mode, lower resolution in battles),\n\
             #         quality (higher resolution, the game's own 30 pictures/s)\n\
             #         or custom (the three values below).\n\
             # battle_scale / menu_scale: internal resolution 1-4 in and out of battles.\n\
             # smooth: on or off. On, it still turns itself off for a while when\n\
             #         the GPU can't keep up.\n\
             # screen_area: 80-100 (%) in [docked], shrinks the picture inside black\n\
             #              borders for TVs with overscan.\n",
        );
        for (name, mode) in [("handheld", &self.handheld), ("docked", &self.docked)] {
            let _ = write!(
                text,
                "\n[{name}]\npreset = {}\nbattle_scale = {}\nmenu_scale = {}\nsmooth = {}\n",
                mode.preset.name(),
                mode.custom.battle_scale,
                mode.custom.menu_scale,
                if mode.custom.smooth { "on" } else { "off" },
            );
            if name == "docked" {
                let _ = writeln!(text, "screen_area = {}", self.screen_area.clamp(80, 100));
            }
        }
        text.push_str(
            "\n# aim: right_stick_aim / gyro_aim 1 or 0, left_stick_aim 0 (off) / 1 / 2,\n\
             # aim_modifier ZL, ZR, L, R, X, Y or A, aim_invert_y 1 or 0, speeds 1.0 = normal\n\
             # power_label: bottom (centered at bottom), bar (above power icons) or off\n\
             # minus_button: menu (Minus alone opens the settings menu) or select (the 3DS's Select)\n\
             [controls]\n",
        );
        for (key, value) in &self.controls {
            let _ = writeln!(text, "{key} = {value}");
        }
        text
    }

    pub fn save(&self) {
        let temp = format!("{PATH}.tmp");
        let written = std::fs::write(&temp, self.to_text()).and_then(|_| {
            let _ = std::fs::remove_file(PATH);
            std::fs::rename(&temp, PATH)
        });
        if let Err(error) = written {
            log::warn!("unx: settings.ini not written, {error}");
        }
    }

    /// Minus alone opens the settings menu, rather than being the 3DS's Select
    pub fn minus_opens_menu(&self) -> bool {
        self.controls.iter().find(|(k, _)| k == "minus_button").map_or(true, |(_, v)| v.trim() != "select")
    }

    /// Where the selected land-battle power's name is drawn on the top screen
    pub fn power_label_mode(&self) -> zakuro_gpu::PowerLabelMode {
        match self.controls.iter().find(|(k, _)| k == "power_label").map(|(_, v)| v.trim()) {
            Some("bar" | "above" | "above_bar") => zakuro_gpu::PowerLabelMode::AboveBar,
            Some("off" | "0" | "none") => zakuro_gpu::PowerLabelMode::Off,
            _ => zakuro_gpu::PowerLabelMode::Bottom,
        }
    }

    pub fn mode(&self, docked: bool) -> &Mode {
        if docked { &self.docked } else { &self.handheld }
    }

    pub fn mode_mut(&mut self, docked: bool) -> &mut Mode {
        if docked { &mut self.docked } else { &mut self.handheld }
    }

    /// hands [controls] to Aim::from_env: the file wins over env.txt. at
    /// start and from the menu, on the main thread; nothing else reads these
    pub fn export_controls(&self) {
        for (key, value) in &self.controls {
            if let Some((_, var)) = CONTROLS.iter().find(|(name, _)| name == key) {
                std::env::set_var(var, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut settings = Settings::default();
        settings.docked.preset = Preset::Custom;
        settings.docked.custom = Video { battle_scale: 2, menu_scale: 3, smooth: false };
        settings.screen_area = 92;
        settings.controls[1].1 = "0".into();
        assert_eq!(Settings::parse(&settings.to_text()), settings);
    }

    #[test]
    fn bad_lines_keep_defaults() {
        let settings = Settings::parse("[handheld]\npreset = fast\nbattle_scale = 9\n[docked]\npreset = quality # sharp\nscreen_area = 40\n");
        assert_eq!(settings.handheld, Settings::default().handheld);
        assert_eq!(settings.docked.preset, Preset::Quality);
        assert!(!settings.docked.video(true).smooth);
        assert_eq!(settings.screen_area, 100);
    }
}
