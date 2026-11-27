//! Native PC controller gyro (IMU) input via OS-native HID (`hidapi`).
//!
//! Supports Sony DualSense / DualSense Edge, DualShock 4, Nintendo Switch Pro
//! Controller, and Valve Steam Deck on Windows, Linux (`hidraw`), and macOS.
//! Runs on a lightweight background thread so high-rate IMU packets (60–1000 Hz)
//! are cleanly integrated with zero dropped motion and zero frame-time stalls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SONY_VID: u16 = 0x054C;
const NINTENDO_VID: u16 = 0x057E;
const VALVE_VID: u16 = 0x28DE;

/// Standard Bosch BMI055 / BMI270 ±2000 deg/s sensitivity (DS4, DualSense, Steam Deck).
const SONY_DPS_PER_LSB: f32 = 2000.0 / 32768.0;
/// Nintendo Switch Pro LSM6DS3 ±2000 deg/s sensitivity (70 mdps / LSB).
const SWITCH_PRO_DPS_PER_LSB: f32 = 0.070;

/// Player-space roll-to-yaw blend so aiming stays intuitive whether holding the pad flat or tilted.
const ROLL_BLEND: f32 = 0.35;
/// Radial deadband in turns/sec (`0.0025` turns/s = `0.9 deg/s`, matching Switch `aim.rs`).
const GYRO_DEAD_ZONE_TURNS_PER_SEC: f32 = 0.0025;
/// Maximum raw rate (deg/s) considered stationary for automatic zero-drift calibration.
const CALIB_MAX_DPS: f32 = 12.0;
const CALIB_SAMPLES: u32 = 40;
/// Gentle online drift-tracking threshold (deg/s) when resting on a desk or held still.
const DRIFT_TRACK_DPS: f32 = 1.8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GyroKind {
    DualSense,
    DualShock4,
    SwitchPro,
    SteamDeck,
}

impl GyroKind {
    pub fn display_name(self) -> &'static str {
        match self {
            GyroKind::DualSense => "DualSense",
            GyroKind::DualShock4 => "DualShock 4",
            GyroKind::SwitchPro => "Switch Pro Controller",
            GyroKind::SteamDeck => "Steam Deck",
        }
    }
}

pub fn identify_device(vid: u16, pid: u16) -> Option<GyroKind> {
    match (vid, pid) {
        (SONY_VID, 0x0CE6 | 0x0DF2) => Some(GyroKind::DualSense),
        (SONY_VID, 0x05C4 | 0x09CC | 0x0BA0) => Some(GyroKind::DualShock4),
        (NINTENDO_VID, 0x2009) => Some(GyroKind::SwitchPro),
        (VALVE_VID, 0x1205) => Some(GyroKind::SteamDeck),
        _ => None,
    }
}

/// Integrated gyro turns (`+yaw_right` turns right, `+pitch_up` aims up) accumulated since last drain.
#[derive(Debug, Clone, Copy, Default)]
pub struct GyroDelta {
    /// Integrated horizontal rotation in full turns (`1.0` = 360° right).
    pub yaw_right_turns: f32,
    /// Integrated vertical rotation in full turns (`1.0` = 360° up).
    pub pitch_up_turns: f32,
}

#[derive(Default)]
struct SharedState {
    device_name: Option<String>,
    yaw_turns: f32,
    pitch_turns: f32,
    recalibrate: bool,
}

/// Manages background HID gyro polling for supported PC controllers.
pub struct GyroReader {
    shared: Arc<Mutex<SharedState>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl GyroReader {
    pub fn start() -> Self {
        let shared = Arc::new(Mutex::new(SharedState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let shared_bg = Arc::clone(&shared);
        let stop_bg = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("zakuro-gyro".into())
            .spawn(move || poll_loop(shared_bg, stop_bg))
            .ok();
        Self { shared, stop, thread }
    }

    /// Name of the active gyro-capable HID controller, if one is connected and streaming.
    pub fn active_device(&self) -> Option<String> {
        self.shared.lock().ok().and_then(|s| s.device_name.clone())
    }

    /// Requests immediate zero-bias recalibration (call when user clicks "Calibrate Gyro").
    pub fn recalibrate(&self) {
        if let Ok(mut s) = self.shared.lock() {
            s.recalibrate = true;
            s.yaw_turns = 0.0;
            s.pitch_turns = 0.0;
        }
    }

    /// Drains accumulated gyro turns since the previous frame.
    pub fn take_delta(&self) -> GyroDelta {
        let Ok(mut s) = self.shared.lock() else {
            return GyroDelta::default();
        };
        let delta = GyroDelta {
            yaw_right_turns: s.yaw_turns,
            pitch_up_turns: s.pitch_turns,
        };
        s.yaw_turns = 0.0;
        s.pitch_turns = 0.0;
        delta
    }
}

impl Drop for GyroReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

/// Parses raw `(yaw_right_dps, pitch_up_dps)` in degrees/second from a controller HID report.
pub fn parse_report_dps(kind: GyroKind, buf: &[u8]) -> Option<(f32, f32)> {
    let i16_le = |off: usize| -> Option<f32> {
        let bytes: [u8; 2] = buf.get(off..off + 2)?.try_into().ok()?;
        Some(i16::from_le_bytes(bytes) as f32)
    };

    match kind {
        GyroKind::DualSense => {
            let base = match *buf.first()? {
                0x01 if buf.len() >= 22 => 16, // USB full report
                0x31 if buf.len() >= 23 => 17, // Bluetooth full report
                _ => return None,
            };
            let pitch = i16_le(base)? * SONY_DPS_PER_LSB;
            let yaw = i16_le(base + 2)? * SONY_DPS_PER_LSB;
            let roll = i16_le(base + 4)? * SONY_DPS_PER_LSB;
            Some((-(yaw + ROLL_BLEND * roll), pitch))
        }
        GyroKind::DualShock4 => {
            let base = match *buf.first()? {
                0x01 if buf.len() >= 19 => 13, // USB report
                0x11 if buf.len() >= 21 => 15, // Bluetooth extended report
                _ => return None,
            };
            let pitch = i16_le(base)? * SONY_DPS_PER_LSB;
            let yaw = i16_le(base + 2)? * SONY_DPS_PER_LSB;
            let roll = i16_le(base + 4)? * SONY_DPS_PER_LSB;
            Some((-(yaw + ROLL_BLEND * roll), pitch))
        }
        GyroKind::SwitchPro => {
            // Standard 60 Hz full input report `0x30`: 3 IMU frames starting at byte 13 (12 B each).
            if *buf.first()? != 0x30 || buf.len() < 49 {
                return None;
            }
            let mut sum_yaw = 0.0f32;
            let mut sum_pitch = 0.0f32;
            for sample in 0..3 {
                let off = 13 + sample * 12 + 6;
                let gx = i16_le(off)? * SWITCH_PRO_DPS_PER_LSB;
                let gy = i16_le(off + 2)? * SWITCH_PRO_DPS_PER_LSB;
                let gz = i16_le(off + 4)? * SWITCH_PRO_DPS_PER_LSB;
                sum_yaw += -gz + ROLL_BLEND * gx;
                sum_pitch += gy;
            }
            Some((sum_yaw / 3.0, sum_pitch / 3.0))
        }
        GyroKind::SteamDeck => {
            if *buf.first()? != 0x01 || buf.len() < 40 || buf.get(2).copied() != Some(0x09) {
                return None;
            }
            let pitch = i16_le(34)? * SONY_DPS_PER_LSB;
            let roll = i16_le(36)? * SONY_DPS_PER_LSB;
            let yaw = i16_le(38)? * SONY_DPS_PER_LSB;
            Some((-(yaw + ROLL_BLEND * roll), pitch))
        }
    }
}

struct BiasFilter {
    bias_yaw: f32,
    bias_pitch: f32,
    sum_yaw: f32,
    sum_pitch: f32,
    samples: u32,
}

impl BiasFilter {
    fn new() -> Self {
        Self { bias_yaw: 0.0, bias_pitch: 0.0, sum_yaw: 0.0, sum_pitch: 0.0, samples: 0 }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Removes resting MEMS bias and applies radial deadband, returning `(yaw_right, pitch_up)` in turns/sec.
    fn filter_turns_per_sec(&mut self, raw_yaw_dps: f32, raw_pitch_dps: f32) -> (f32, f32) {
        if !raw_yaw_dps.is_finite() || !raw_pitch_dps.is_finite() {
            return (0.0, 0.0);
        }
        if self.samples < CALIB_SAMPLES {
            if raw_yaw_dps.hypot(raw_pitch_dps) <= CALIB_MAX_DPS {
                self.sum_yaw += raw_yaw_dps;
                self.sum_pitch += raw_pitch_dps;
                self.samples += 1;
                self.bias_yaw = self.sum_yaw / self.samples as f32;
                self.bias_pitch = self.sum_pitch / self.samples as f32;
            }
            return (0.0, 0.0);
        }
        let dy = raw_yaw_dps - self.bias_yaw;
        let dp = raw_pitch_dps - self.bias_pitch;
        if dy.hypot(dp) < DRIFT_TRACK_DPS {
            self.bias_yaw += dy * 0.004;
            self.bias_pitch += dp * 0.004;
        }
        let yaw_tps = dy / 360.0;
        let pitch_tps = dp / 360.0;
        let mag = yaw_tps.hypot(pitch_tps);
        if mag <= GYRO_DEAD_ZONE_TURNS_PER_SEC {
            return (0.0, 0.0);
        }
        let scale = (mag - GYRO_DEAD_ZONE_TURNS_PER_SEC) / mag;
        (yaw_tps * scale, pitch_tps * scale)
    }
}

fn init_device(kind: GyroKind, dev: &hidapi::HidDevice) {
    match kind {
        GyroKind::SwitchPro => {
            // Enable IMU (subcommand 0x40, arg 0x01) and switch to full 60 Hz report mode (0x03, arg 0x30).
            let mut pkt = [0u8; 49];
            pkt[0] = 0x01;
            pkt[1] = 0x00;
            pkt[10] = 0x40;
            pkt[11] = 0x01;
            let _ = dev.write(&pkt);
            thread::sleep(Duration::from_millis(20));
            pkt[1] = 0x01;
            pkt[10] = 0x03;
            pkt[11] = 0x30;
            let _ = dev.write(&pkt);
        }
        GyroKind::DualSense => {
            // Reading feature report 0x05 switches Bluetooth DualSense into full 0x31 IMU reporting.
            let mut feat = [0u8; 64];
            feat[0] = 0x05;
            let _ = dev.get_feature_report(&mut feat);
        }
        GyroKind::DualShock4 => {
            // Reading feature report 0x02 switches Bluetooth DS4 into full 0x11 IMU reporting.
            let mut feat = [0u8; 64];
            feat[0] = 0x02;
            let _ = dev.get_feature_report(&mut feat);
        }
        GyroKind::SteamDeck => {}
    }
}

fn poll_loop(shared: Arc<Mutex<SharedState>>, stop: Arc<AtomicBool>) {
    let mut api = match hidapi::HidApi::new() {
        Ok(api) => api,
        Err(err) => {
            log::debug!("gyro: hidapi unavailable ({err}); PC gyro aiming disabled");
            return;
        }
    };

    let mut buf = [0u8; 128];
    while !stop.load(Ordering::Relaxed) {
        let _ = api.refresh_devices();
        let found = api.device_list().find_map(|info| {
            let kind = identify_device(info.vendor_id(), info.product_id())?;
            let path = info.path().to_owned();
            let name = info
                .product_string()
                .filter(|s| !s.trim().is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| kind.display_name().to_owned());
            Some((kind, path, name))
        });

        let Some((kind, path, name)) = found else {
            if let Ok(mut s) = shared.lock() {
                s.device_name = None;
            }
            // Wait up to 1.5 s before rescanning, waking quickly on shutdown.
            for _ in 0..15 {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
            continue;
        };

        let dev = match api.open_path(&path) {
            Ok(d) => d,
            Err(err) => {
                log::debug!("gyro: could not open {name}: {err}");
                thread::sleep(Duration::from_millis(800));
                continue;
            }
        };

        log::info!("gyro: connected {name} ({:?})", kind);
        init_device(kind, &dev);
        if let Ok(mut s) = shared.lock() {
            s.device_name = Some(name.clone());
            s.recalibrate = false;
        }

        let mut filter = BiasFilter::new();
        let mut last_sample = Instant::now();
        let mut empty_reads = 0u32;

        while !stop.load(Ordering::Relaxed) {
            if let Ok(mut s) = shared.lock() {
                if s.recalibrate {
                    s.recalibrate = false;
                    filter.reset();
                }
            }

            match dev.read_timeout(&mut buf, 40) {
                Ok(0) => {
                    empty_reads = empty_reads.saturating_add(1);
                    if empty_reads > 150 {
                        // ~6 seconds of silence; re-enumerate in case device disconnected.
                        break;
                    }
                }
                Ok(n) => {
                    empty_reads = 0;
                    let now = Instant::now();
                    let dt = now.saturating_duration_since(last_sample).as_secs_f32().clamp(0.0005, 0.05);
                    last_sample = now;
                    if let Some((raw_yaw_dps, raw_pitch_dps)) = parse_report_dps(kind, &buf[..n]) {
                        let (yaw_tps, pitch_tps) = filter.filter_turns_per_sec(raw_yaw_dps, raw_pitch_dps);
                        if let Ok(mut s) = shared.lock() {
                            s.yaw_turns = (s.yaw_turns + yaw_tps * dt).clamp(-1.0, 1.0);
                            s.pitch_turns = (s.pitch_turns + pitch_tps * dt).clamp(-1.0, 1.0);
                        }
                    }
                }
                Err(err) => {
                    log::info!("gyro: {name} disconnected ({err})");
                    break;
                }
            }
        }

        if let Ok(mut s) = shared.lock() {
            s.device_name = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_major_gyro_controllers() {
        assert_eq!(identify_device(0x054C, 0x0CE6), Some(GyroKind::DualSense));
        assert_eq!(identify_device(0x054C, 0x0DF2), Some(GyroKind::DualSense));
        assert_eq!(identify_device(0x054C, 0x09CC), Some(GyroKind::DualShock4));
        assert_eq!(identify_device(0x057E, 0x2009), Some(GyroKind::SwitchPro));
        assert_eq!(identify_device(0x28DE, 0x1205), Some(GyroKind::SteamDeck));
        assert_eq!(identify_device(0x045E, 0x02EA), None); // Xbox One (no gyro)
    }

    #[test]
    fn parses_dualsense_usb_and_bt_reports() {
        let mut usb = [0u8; 64];
        usb[0] = 0x01;
        usb[16..18].copy_from_slice(&1638i16.to_le_bytes()); // +100 dps pitch up
        usb[18..20].copy_from_slice(&(-1638i16).to_le_bytes()); // -100 dps yaw -> +100 dps right
        let (yaw, pitch) = parse_report_dps(GyroKind::DualSense, &usb).unwrap();
        assert!((yaw - 100.0).abs() < 0.2, "yaw={yaw}");
        assert!((pitch - 100.0).abs() < 0.2, "pitch={pitch}");

        let mut bt = [0u8; 78];
        bt[0] = 0x31;
        bt[17..19].copy_from_slice(&1638i16.to_le_bytes());
        bt[19..21].copy_from_slice(&(-1638i16).to_le_bytes());
        let (bt_yaw, bt_pitch) = parse_report_dps(GyroKind::DualSense, &bt).unwrap();
        assert!((bt_yaw - yaw).abs() < 1e-4);
        assert!((bt_pitch - pitch).abs() < 1e-4);
    }

    #[test]
    fn parses_switch_pro_3_sample_report() {
        let mut pkt = [0u8; 49];
        pkt[0] = 0x30;
        for s in 0..3 {
            let off = 13 + s * 12 + 6;
            pkt[off..off + 2].copy_from_slice(&0i16.to_le_bytes());
            pkt[off + 2..off + 4].copy_from_slice(&1000i16.to_le_bytes()); // +70 dps pitch up
            pkt[off + 4..off + 6].copy_from_slice(&(-1000i16).to_le_bytes()); // +70 dps yaw right
        }
        let (yaw, pitch) = parse_report_dps(GyroKind::SwitchPro, &pkt).unwrap();
        assert!((yaw - 70.0).abs() < 0.05);
        assert!((pitch - 70.0).abs() < 0.05);
    }

    #[test]
    fn bias_filter_calibrates_and_suppresses_resting_drift() {
        let mut f = BiasFilter::new();
        for _ in 0..CALIB_SAMPLES {
            assert_eq!(f.filter_turns_per_sec(2.5, -1.5), (0.0, 0.0));
        }
        // Stationary bias is completely nulled out.
        assert_eq!(f.filter_turns_per_sec(2.5, -1.5), (0.0, 0.0));
        // Intentional 90 deg/s right turn passes cleanly (~0.25 turns/s minus tiny deadband).
        let (yaw, pitch) = f.filter_turns_per_sec(92.5, -1.5);
        assert!(yaw > 0.24 && yaw < 0.26, "yaw={yaw}");
        assert!(pitch.abs() < 0.005, "pitch={pitch}");
    }
}
