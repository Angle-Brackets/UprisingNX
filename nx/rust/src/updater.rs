//! In-app GitHub Releases auto-updater (modeled after PaperBoat64's SwitchUpdater).
//!
//! - Checks `https://api.github.com/repos/Angle-Brackets/UprisingNX/releases/latest`
//!   once ~15s after launch (and on demand from `Settings -> Save & System`).
//! - Downloads the release's `uprising_nx.nro` asset to `<running_nro>.new` with
//!   live progress, verifies the `"NRO0"` header at offset `0x10`, and on restart
//!   swaps it atomically into place and relaunches via `envSetNextLoad`.

#[cfg(target_os = "horizon")]
use std::ffi::{c_char, c_int, c_long, CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[cfg(target_os = "horizon")]
const LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/Angle-Brackets/UprisingNX/releases/latest";
const ASSET_NAME: &str = "uprising_nx.nro";
const AUTO_CHECK_DELAY: Duration = Duration::from_secs(3);
const MIN_VALID_NRO_BYTES: u64 = 1024 * 1024; // 1 MiB sanity floor
#[cfg(target_os = "horizon")]
const HTTP_BUF_CAP: usize = 128 * 1024;

#[cfg(target_os = "horizon")]
extern "C" {
    fn unx_updater_nro_path() -> *const c_char;
    fn unx_updater_mark_pending_swap();
    fn unx_updater_has_internet() -> c_int;
    fn unx_updater_http_get(
        url: *const c_char,
        out_buf: *mut c_char,
        out_cap: usize,
        out_len: *mut usize,
        http_status: *mut c_long,
        cancel_flag: *const c_int,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
    fn unx_updater_download_file(
        url: *const c_char,
        dest_path: *const c_char,
        progress_pct: *mut c_int,
        cancel_flag: *const c_int,
        err_buf: *mut c_char,
        err_cap: usize,
    ) -> c_int;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateState {
    Idle,
    Checking,
    UpToDate(String),
    Available { tag: String, url: String },
    Downloading { tag: String, pct: i32 },
    Ready { tag: String },
    Failed(String),
}

/// Visual tone for the Save & System menu row and inline info box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateTone {
    Neutral,
    Accent,
    Success,
    Error,
}

/// Snapshot passed to `Menu` each frame while the settings overlay is open.
#[derive(Clone, Debug)]
pub struct UpdateView {
    pub value: String,
    pub tone: UpdateTone,
    pub heading: String,
    pub detail: String,
    pub progress: Option<f32>,
}

struct Shared {
    state: Mutex<UpdateState>,
    progress_pct: AtomicI32,
    cancel: AtomicI32,
    announce: AtomicBool,
}

pub struct Updater {
    started_at: Instant,
    auto_checked: bool,
    current_version: String,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Updater {
    pub fn new(current_version: &str) -> Self {
        Self {
            started_at: Instant::now(),
            auto_checked: false,
            current_version: current_version.to_owned(),
            shared: Arc::new(Shared {
                state: Mutex::new(UpdateState::Idle),
                progress_pct: AtomicI32::new(0),
                cancel: AtomicI32::new(0),
                announce: AtomicBool::new(false),
            }),
            worker: None,
        }
    }

    pub fn state(&self) -> UpdateState {
        let mut s = self
            .shared
            .state
            .lock()
            .map(|g| g.clone())
            .unwrap_or(UpdateState::Idle);
        if let UpdateState::Downloading { tag, .. } = s {
            let pct = self.shared.progress_pct.load(Ordering::Relaxed).clamp(0, 100);
            s = UpdateState::Downloading { tag, pct };
        }
        s
    }

    pub fn view(&self) -> UpdateView {
        match self.state() {
            UpdateState::Idle => UpdateView {
                value: "Check now".to_owned(),
                tone: UpdateTone::Neutral,
                heading: format!("UprisingNX {}", self.current_version),
                detail: "Checks GitHub Releases for a newer Switch build. Press A to check now.".to_owned(),
                progress: None,
            },
            UpdateState::Checking => UpdateView {
                value: "Checking...".to_owned(),
                tone: UpdateTone::Accent,
                heading: "Checking GitHub Releases".to_owned(),
                detail: "Contacting github.com/Angle-Brackets/UprisingNX for the latest release...".to_owned(),
                progress: None,
            },
            UpdateState::UpToDate(tag) => UpdateView {
                value: "Up to date".to_owned(),
                tone: UpdateTone::Success,
                heading: format!("Up to Date ({})", self.current_version),
                detail: format!("Latest GitHub release is {tag}. Press A to check again."),
                progress: None,
            },
            UpdateState::Available { tag, .. } => UpdateView {
                value: format!("Install {tag}"),
                tone: UpdateTone::Accent,
                heading: format!("Update Available: {tag}"),
                detail: format!(
                    "Currently running {}. Press A to download {ASSET_NAME} (~22 MB) in the background.",
                    self.current_version
                ),
                progress: None,
            },
            UpdateState::Downloading { tag, pct } => UpdateView {
                value: format!("Downloading {pct}%"),
                tone: UpdateTone::Accent,
                heading: format!("Downloading {tag} ({pct}%)"),
                detail: "Downloading new Switch binary to SD card. You can keep playing or wait here.".to_owned(),
                progress: Some(pct as f32 / 100.0),
            },
            UpdateState::Ready { tag } => UpdateView {
                value: "Restart to apply".to_owned(),
                tone: UpdateTone::Success,
                heading: format!("Update Ready: {tag}"),
                detail: "Verified NRO staged on SD card. Press A to swap it in and relaunch UprisingNX.".to_owned(),
                progress: Some(1.0),
            },
            UpdateState::Failed(err) => UpdateView {
                value: "Retry check".to_owned(),
                tone: UpdateTone::Error,
                heading: "Update Check Failed".to_owned(),
                detail: err,
                progress: None,
            },
        }
    }

    /// Triggers one silent background check once on the Title Screen (or after
    /// `AUTO_CHECK_DELAY`) if internet is connected. Skips quietly when offline.
    pub fn maybe_auto_check(&mut self, on_title: bool) {
        if self.auto_checked {
            return;
        }
        if !on_title && self.started_at.elapsed() < AUTO_CHECK_DELAY {
            return;
        }
        self.auto_checked = true;
        if !has_internet() {
            log::info!("unx: updater skipping auto-check (offline)");
            return;
        }
        if matches!(self.state(), UpdateState::Idle) {
            self.start_check(true);
        }
    }

    /// Short yellow badge text shown above the version label on the Title Screen
    /// whenever a newer release is available or ready to apply.
    pub fn title_badge(&self) -> Option<&'static str> {
        match self.state() {
            UpdateState::Available { .. } => Some("Update Available!"),
            UpdateState::Ready { .. } => Some("Update Ready!"),
            _ => None,
        }
    }

    /// Returns a one-shot toast message when a background launch check finds a
    /// newer release (caller only displays it while out of active battle).
    pub fn take_announcement(&self) -> Option<String> {
        if !self.shared.announce.swap(false, Ordering::AcqRel) {
            return None;
        }
        match self.state() {
            UpdateState::Available { tag, .. } => Some(format!(
                "Update available: {tag} (Minus -> Save & System)"
            )),
            _ => None,
        }
    }

    /// Handles pressing `A` on the `Software update` row. Returns `true` when
    /// the caller should exit immediately to apply a staged update & relaunch.
    pub fn trigger_action(&mut self) -> bool {
        match self.state() {
            UpdateState::Idle | UpdateState::UpToDate(_) | UpdateState::Failed(_) => {
                self.start_check(false);
                false
            }
            UpdateState::Available { tag, url } => {
                self.start_download(tag, url);
                false
            }
            UpdateState::Ready { tag } => {
                log::info!("unx: applying staged update {tag} and relaunching");
                #[cfg(target_os = "horizon")]
                unsafe {
                    unx_updater_mark_pending_swap();
                }
                true
            }
            UpdateState::Checking | UpdateState::Downloading { .. } => false,
        }
    }

    fn join_finished_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }

    fn set_state(shared: &Shared, next: UpdateState) {
        if let Ok(mut guard) = shared.state.lock() {
            *guard = next;
        }
    }

    pub fn start_check(&mut self, announce: bool) {
        if matches!(
            self.state(),
            UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }
        ) {
            return;
        }
        self.join_finished_worker();
        Self::set_state(&self.shared, UpdateState::Checking);
        let shared = Arc::clone(&self.shared);
        let current = self.current_version.clone();
        let spawned = std::thread::Builder::new()
            .name("unx-upd-chk".into())
            .stack_size(256 << 10)
            .spawn(move || {
                run_check(&shared, &current, announce);
            });
        match spawned {
            Ok(h) => self.worker = Some(h),
            Err(e) => Self::set_state(&self.shared, UpdateState::Failed(format!("Could not start check thread: {e}"))),
        }
    }

    fn start_download(&mut self, tag: String, url: String) {
        self.join_finished_worker();
        self.shared.progress_pct.store(0, Ordering::Relaxed);
        Self::set_state(
            &self.shared,
            UpdateState::Downloading {
                tag: tag.clone(),
                pct: 0,
            },
        );
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("unx-upd-dl".into())
            .stack_size(256 << 10)
            .spawn(move || {
                run_download(&shared, &tag, &url);
            });
        match spawned {
            Ok(h) => self.worker = Some(h),
            Err(e) => Self::set_state(
                &self.shared,
                UpdateState::Failed(format!("Could not start download thread: {e}")),
            ),
        }
    }
}

impl Drop for Updater {
    fn drop(&mut self) {
        self.shared.cancel.store(1, Ordering::Release);
        self.join_finished_worker();
    }
}

#[cfg(target_os = "horizon")]
fn has_internet() -> bool {
    unsafe { unx_updater_has_internet() != 0 }
}

#[cfg(not(target_os = "horizon"))]
fn has_internet() -> bool {
    false
}

#[cfg(target_os = "horizon")]
fn run_check(shared: &Shared, current_version: &str, announce: bool) {
    if !has_internet() {
        if announce {
            Updater::set_state(shared, UpdateState::Idle);
        } else {
            Updater::set_state(shared, UpdateState::Failed("No internet connection.".to_owned()));
        }
        return;
    }
    let Ok(c_url) = CString::new(LATEST_RELEASE_URL) else {
        Updater::set_state(shared, UpdateState::Failed("Invalid release URL".to_owned()));
        return;
    };
    let mut body = vec![0u8; HTTP_BUF_CAP];
    let mut err = [0u8; 256];
    let mut out_len: usize = 0;
    let mut http_status: c_long = 0;
    let cancel_ptr = &shared.cancel as *const AtomicI32 as *const c_int;

    let rc = unsafe {
        unx_updater_http_get(
            c_url.as_ptr(),
            body.as_mut_ptr() as *mut c_char,
            body.len(),
            &mut out_len,
            &mut http_status,
            cancel_ptr,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if shared.cancel.load(Ordering::Acquire) != 0 {
        return;
    }
    if rc != 0 {
        if http_status == 404 {
            Updater::set_state(shared, UpdateState::UpToDate(current_version.to_owned()));
            return;
        }
        let msg = c_buf_to_string(&err);
        let detail = if msg.is_empty() {
            format!("Could not reach GitHub (HTTP {http_status}, code {rc})")
        } else {
            format!("Could not reach GitHub: {msg}")
        };
        log::warn!("unx: updater check failed: {detail}");
        if announce {
            Updater::set_state(shared, UpdateState::Idle);
        } else {
            Updater::set_state(shared, UpdateState::Failed(detail));
        }
        return;
    }

    let text = String::from_utf8_lossy(&body[..out_len.min(body.len())]);
    let Some(release) = parse_github_release(&text, ASSET_NAME) else {
        log::warn!("unx: updater got unexpected JSON from GitHub ({out_len} bytes)");
        Updater::set_state(
            shared,
            UpdateState::Failed("GitHub returned an unexpected reply.".to_owned()),
        );
        return;
    };

    let newer = is_newer_release(&release.tag_name, current_version);
    log::info!(
        "unx: updater latest={} current={} newer={} asset={}",
        release.tag_name,
        current_version,
        newer,
        release.asset_url.is_some()
    );
    if !newer {
        Updater::set_state(shared, UpdateState::UpToDate(release.tag_name));
        return;
    }
    let Some(url) = release.asset_url else {
        Updater::set_state(
            shared,
            UpdateState::Failed(format!(
                "Release {} does not include a standalone {ASSET_NAME} asset.",
                release.tag_name
            )),
        );
        return;
    };

    Updater::set_state(
        shared,
        UpdateState::Available {
            tag: release.tag_name,
            url,
        },
    );
    if announce {
        shared.announce.store(true, Ordering::Release);
    }
}

#[cfg(not(target_os = "horizon"))]
fn run_check(shared: &Shared, _current_version: &str, _announce: bool) {
    Updater::set_state(
        shared,
        UpdateState::Failed("Auto-updater is only supported on Nintendo Switch.".to_owned()),
    );
}

#[cfg(target_os = "horizon")]
fn run_download(shared: &Shared, tag: &str, url: &str) {
    let nro_path = unsafe {
        let p = unx_updater_nro_path();
        if p.is_null() {
            "sdmc:/switch/uprisingnx/uprising_nx.nro".to_owned()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    let dest_path = format!("{nro_path}.new");
    let (Ok(c_url), Ok(c_dest)) = (CString::new(url), CString::new(dest_path.as_str())) else {
        Updater::set_state(shared, UpdateState::Failed("Invalid download path or URL".to_owned()));
        return;
    };

    let mut err = [0u8; 256];
    let pct_ptr = &shared.progress_pct as *const AtomicI32 as *mut c_int;
    let cancel_ptr = &shared.cancel as *const AtomicI32 as *const c_int;
    let rc = unsafe {
        unx_updater_download_file(
            c_url.as_ptr(),
            c_dest.as_ptr(),
            pct_ptr,
            cancel_ptr,
            err.as_mut_ptr() as *mut c_char,
            err.len(),
        )
    };
    if shared.cancel.load(Ordering::Acquire) != 0 {
        let _ = std::fs::remove_file(&dest_path);
        return;
    }
    if rc != 0 {
        let msg = c_buf_to_string(&err);
        let detail = if msg.is_empty() {
            format!("Download failed (code {rc})")
        } else {
            format!("Download failed: {msg}")
        };
        log::warn!("unx: updater download failed: {detail}");
        Updater::set_state(shared, UpdateState::Failed(detail));
        return;
    }

    if let Err(reason) = validate_nro_file(Path::new(&dest_path)) {
        let _ = std::fs::remove_file(&dest_path);
        log::warn!("unx: downloaded NRO invalid: {reason}");
        Updater::set_state(shared, UpdateState::Failed(reason));
        return;
    }

    log::info!("unx: updater staged verified NRO for {tag} at {dest_path}");
    Updater::set_state(shared, UpdateState::Ready { tag: tag.to_owned() });
}

#[cfg(not(target_os = "horizon"))]
fn run_download(shared: &Shared, _tag: &str, _url: &str) {
    Updater::set_state(
        shared,
        UpdateState::Failed("Auto-updater is only supported on Nintendo Switch.".to_owned()),
    );
}

#[cfg(target_os = "horizon")]
fn c_buf_to_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_owned()
}

/// Validates that `path` is a real Switch NRO (`"NRO0"` at offset `0x10` and
/// at least `MIN_VALID_NRO_BYTES` long) so error pages or truncated downloads
/// can never replace the working binary.
pub fn validate_nro_file(path: &Path) -> Result<(), String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("Could not inspect downloaded file: {e}"))?;
    if meta.len() < MIN_VALID_NRO_BYTES {
        return Err(format!(
            "Downloaded file is too small ({} KB) to be a valid NRO.",
            meta.len() / 1024
        ));
    }
    let mut f = File::open(path).map_err(|e| format!("Could not read downloaded file: {e}"))?;
    f.seek(SeekFrom::Start(0x10))
        .map_err(|e| format!("Could not seek downloaded file: {e}"))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("Could not read NRO header: {e}"))?;
    if &magic != b"NRO0" {
        return Err("Downloaded file is not a valid Switch NRO (missing NRO0 header).".to_owned());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Version Comparison
// ---------------------------------------------------------------------------

fn strip_v_prefix(s: &str) -> &str {
    s.trim().trim_start_matches(['v', 'V'])
}

fn parse_numeric_prefix(s: &str) -> Vec<u32> {
    let core = strip_v_prefix(s).split('-').next().unwrap_or("");
    core.split('.')
        .filter_map(|part| part.parse::<u32>().ok())
        .collect()
}

/// Returns `true` if `remote_tag` (e.g. `"v0.1.0-abc1234"` or `"v0.2.0"`) is
/// newer than `current_label` (e.g. `"0.1.0-73a9696"`).
pub fn is_newer_release(remote_tag: &str, current_label: &str) -> bool {
    let r_norm = strip_v_prefix(remote_tag);
    let c_norm = strip_v_prefix(current_label);
    if r_norm.is_empty() {
        return false;
    }
    let r_nums = parse_numeric_prefix(r_norm);
    let c_nums = parse_numeric_prefix(c_norm);
    let max_len = r_nums.len().max(c_nums.len());
    for i in 0..max_len {
        let r = r_nums.get(i).copied().unwrap_or(0);
        let c = c_nums.get(i).copied().unwrap_or(0);
        if r != c {
            return r > c;
        }
    }
    // Equal semver numbers (e.g. rolling `0.1.0-<sha>` releases): any different
    // tag on GitHub's `/releases/latest` is offered as an update until installed.
    !r_norm.eq_ignore_ascii_case(c_norm)
}

// ---------------------------------------------------------------------------
// Minimal Zero-Dependency JSON Parser for GitHub `/releases/latest`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseInfo {
    pub tag_name: String,
    pub asset_url: Option<String>,
}

struct JsonScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonScanner<'a> {
    fn new(s: &'a str) -> Self {
        Self { bytes: s.as_bytes(), pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, ch: u8) -> bool {
        if self.peek() == Some(ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        if !self.expect(b'"') {
            return None;
        }
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Some(out),
                b'\\' => {
                    let esc = *self.bytes.get(self.pos)?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self.bytes.get(self.pos..self.pos + 4)?;
                            self.pos += 4;
                            let code = std::str::from_utf8(hex)
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok())?;
                            if let Some(c) = char::from_u32(code) {
                                out.push(c);
                            }
                        }
                        _ => out.push(esc as char),
                    }
                }
                _ => out.push(b as char),
            }
        }
        None
    }

    fn skip_value(&mut self) -> bool {
        let Some(b) = self.peek() else { return false };
        match b {
            b'"' => self.parse_string().is_some(),
            b'{' => {
                self.pos += 1;
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return true;
                }
                loop {
                    if self.parse_string().is_none() || !self.expect(b':') || !self.skip_value() {
                        return false;
                    }
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return true;
                        }
                        _ => return false,
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return true;
                }
                loop {
                    if !self.skip_value() {
                        return false;
                    }
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return true;
                        }
                        _ => return false,
                    }
                }
            }
            _ => {
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && !matches!(self.bytes[self.pos], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')
                {
                    self.pos += 1;
                }
                self.pos > start
            }
        }
    }

    fn parse_asset_object(&mut self) -> Option<(Option<String>, Option<String>)> {
        if !self.expect(b'{') {
            return None;
        }
        let mut name = None;
        let mut url = None;
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Some((name, url));
        }
        loop {
            let key = self.parse_string()?;
            if !self.expect(b':') {
                return None;
            }
            match key.as_str() {
                "name" => name = self.parse_string(),
                "browser_download_url" => url = self.parse_string(),
                _ => {
                    if !self.skip_value() {
                        return None;
                    }
                }
            }
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Some((name, url));
                }
                _ => return None,
            }
        }
    }
}

/// Parses GitHub's `/releases/latest` JSON payload, extracting `tag_name` and
/// the `browser_download_url` of the asset matching `wanted_asset`.
pub fn parse_github_release(json: &str, wanted_asset: &str) -> Option<ReleaseInfo> {
    let mut sc = JsonScanner::new(json);
    if !sc.expect(b'{') {
        return None;
    }
    let mut tag_name: Option<String> = None;
    let mut asset_url: Option<String> = None;
    if sc.peek() == Some(b'}') {
        return None;
    }
    loop {
        let key = sc.parse_string()?;
        if !sc.expect(b':') {
            return None;
        }
        match key.as_str() {
            "tag_name" => tag_name = sc.parse_string(),
            "assets" => {
                if !sc.expect(b'[') {
                    return None;
                }
                if sc.peek() != Some(b']') {
                    loop {
                        let (name, url) = sc.parse_asset_object()?;
                        if name.as_deref() == Some(wanted_asset) {
                            asset_url = url;
                        }
                        match sc.peek() {
                            Some(b',') => sc.pos += 1,
                            Some(b']') => {
                                sc.pos += 1;
                                break;
                            }
                            _ => return None,
                        }
                    }
                } else {
                    sc.pos += 1;
                }
            }
            _ => {
                if !sc.skip_value() {
                    return None;
                }
            }
        }
        match sc.peek() {
            Some(b',') => sc.pos += 1,
            Some(b'}') => break,
            _ => break, // allow graceful completion even if response was truncated after assets
        }
    }
    Some(ReleaseInfo {
        tag_name: tag_name?,
        asset_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_comparison_handles_semver_and_rolling_sha() {
        assert!(is_newer_release("v0.2.0", "0.1.0-73a9696"));
        assert!(is_newer_release("v0.1.1-abc1234", "0.1.0-73a9696"));
        assert!(is_newer_release("v0.1.0-def5678", "0.1.0-73a9696"));
        assert!(is_newer_release("v0.1.0", "0.1.0-73a9696"));
        assert!(!is_newer_release("v0.1.0-73a9696", "0.1.0-73a9696"));
        assert!(!is_newer_release("0.1.0-73a9696", "0.1.0-73a9696"));
        assert!(!is_newer_release("v0.0.9-fffffff", "0.1.0-73a9696"));
    }

    #[test]
    fn parses_github_release_json_any_field_order() {
        let sample = r#"{
            "id": 12345,
            "name": "UprisingNX \"v0.1.0-abc1234\"",
            "tag_name": "v0.1.0-abc1234",
            "assets": [
                {
                    "id": 1,
                    "browser_download_url": "https://github.com/Angle-Brackets/UprisingNX/releases/download/v0.1.0-abc1234/UprisingNX-v0.1.0-abc1234.zip",
                    "name": "UprisingNX-v0.1.0-abc1234.zip"
                },
                {
                    "id": 2,
                    "browser_download_url": "https://github.com/Angle-Brackets/UprisingNX/releases/download/v0.1.0-abc1234/uprising_nx.nro",
                    "uploader": { "login": "Angle-Brackets", "nested": [1, 2, {"k": "v"}] },
                    "name": "uprising_nx.nro"
                }
            ],
            "body": "Release notes with } and ] and \"quotes\""
        }"#;
        let info = parse_github_release(sample, "uprising_nx.nro").unwrap();
        assert_eq!(info.tag_name, "v0.1.0-abc1234");
        assert_eq!(
            info.asset_url.as_deref(),
            Some("https://github.com/Angle-Brackets/UprisingNX/releases/download/v0.1.0-abc1234/uprising_nx.nro")
        );
    }

    #[test]
    fn validates_nro_magic_and_size() {
        let path = std::env::temp_dir().join("unx_test_valid.nro");
        let mut bytes = vec![0u8; (MIN_VALID_NRO_BYTES + 64) as usize];
        bytes[0x10..0x14].copy_from_slice(b"NRO0");
        std::fs::write(&path, &bytes).unwrap();
        assert!(validate_nro_file(&path).is_ok());

        bytes[0x10..0x14].copy_from_slice(b"HTML");
        std::fs::write(&path, &bytes).unwrap();
        assert!(validate_nro_file(&path).is_err());

        std::fs::write(&path, b"short").unwrap();
        assert!(validate_nro_file(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }
}
