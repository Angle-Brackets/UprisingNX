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
const VERSIONS_MANIFEST_URL: &str =
    "https://github.com/Angle-Brackets/UprisingNX/releases/download/snapshots/versions.json";
#[cfg(target_os = "horizon")]
const LATEST_RELEASE_URL: &str =
    "https://api.github.com/repos/Angle-Brackets/UprisingNX/releases/latest";
#[cfg(target_os = "horizon")]
const RELEASES_LIST_URL: &str =
    "https://api.github.com/repos/Angle-Brackets/UprisingNX/releases?per_page=5";
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

/// One historical release or snapshot entry from `versions.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionEntry {
    pub id: String,
    pub is_snapshot: bool,
    pub date: String,
    pub commit: String,
    pub nro_url: String,
    pub notes: String,
}

/// Parsed Minecraft-style `versions.json` manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionManifest {
    pub latest_release: Option<String>,
    pub latest_snapshot: Option<String>,
    pub versions: Vec<VersionEntry>,
}

/// Snapshot passed to `Menu` for the `Version history` picker row.
#[derive(Clone, Debug)]
pub struct PickerView {
    pub value: String,
    pub heading: String,
    pub detail: String,
    pub loaded: bool,
}

impl Default for PickerView {
    fn default() -> Self {
        Self {
            value: "Press A to load".to_owned(),
            heading: "Version History (versions.json)".to_owned(),
            detail: "Press A to load all past releases and snapshots, then Left/Right to pick and A to install or roll back.".to_owned(),
            loaded: false,
        }
    }
}

struct Shared {
    state: Mutex<UpdateState>,
    versions: Mutex<Vec<VersionEntry>>,
    progress_pct: AtomicI32,
    cancel: AtomicI32,
    announce: AtomicBool,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(target_os = "horizon"), allow(dead_code))]
struct BuildVersion {
    version: String,
    git_sha: String,
    release_tag: String,
    display_label: String,
}

pub struct Updater {
    started_at: Instant,
    auto_checked: bool,
    include_prereleases: bool,
    picker_index: usize,
    build: BuildVersion,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Updater {
    pub fn new(
        version: &str,
        git_sha: &str,
        release_tag: &str,
        display_label: &str,
        include_prereleases: bool,
    ) -> Self {
        Self {
            started_at: Instant::now(),
            auto_checked: false,
            include_prereleases,
            picker_index: 0,
            build: BuildVersion {
                version: version.to_owned(),
                git_sha: git_sha.to_owned(),
                release_tag: release_tag.to_owned(),
                display_label: display_label.to_owned(),
            },
            shared: Arc::new(Shared {
                state: Mutex::new(UpdateState::Idle),
                versions: Mutex::new(Vec::new()),
                progress_pct: AtomicI32::new(0),
                cancel: AtomicI32::new(0),
                announce: AtomicBool::new(false),
            }),
            worker: None,
        }
    }

    /// Switches between Stable (`false`) and Pre-release (`true`) channels.
    /// If no download/staging is active, immediately re-checks the new channel.
    pub fn set_include_prereleases(&mut self, include: bool) {
        if self.include_prereleases == include {
            return;
        }
        self.include_prereleases = include;
        self.picker_index = 0;
        if !matches!(
            self.state(),
            UpdateState::Downloading { .. } | UpdateState::Ready { .. }
        ) {
            Self::set_state(&self.shared, UpdateState::Idle);
            self.start_check(false);
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
                heading: format!("UprisingNX {}", self.build.display_label),
                detail: if self.include_prereleases {
                    "Checks GitHub Releases (including pre-releases) for a newer build. Press A to check now.".to_owned()
                } else {
                    "Checks GitHub Releases for a newer Switch build. Press A to check now.".to_owned()
                },
                progress: None,
            },
            UpdateState::Checking => UpdateView {
                value: "Checking...".to_owned(),
                tone: UpdateTone::Accent,
                heading: "Checking GitHub Releases".to_owned(),
                detail: if self.include_prereleases {
                    "Contacting github.com/Angle-Brackets/UprisingNX for the latest release or pre-release...".to_owned()
                } else {
                    "Contacting github.com/Angle-Brackets/UprisingNX for the latest release...".to_owned()
                },
                progress: None,
            },
            UpdateState::UpToDate(tag) => UpdateView {
                value: "Up to date".to_owned(),
                tone: UpdateTone::Success,
                heading: format!("Up to Date ({})", self.build.display_label),
                detail: format!("Latest GitHub release is {tag}. Press A to check again."),
                progress: None,
            },
            UpdateState::Available { tag, .. } => UpdateView {
                value: format!("Install {tag}"),
                tone: UpdateTone::Accent,
                heading: format!("Update Available: {tag}"),
                detail: format!(
                    "Currently running {}. Press A to download {ASSET_NAME} (~22 MB) in the background.",
                    self.build.display_label
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

    fn filtered_versions(&self) -> Vec<VersionEntry> {
        let Ok(guard) = self.shared.versions.lock() else {
            return Vec::new();
        };
        guard
            .iter()
            .filter(|v| self.include_prereleases || !v.is_snapshot)
            .cloned()
            .collect()
    }

    /// Snapshot of the `Version history` picker row for `Menu`.
    pub fn picker_view(&self) -> PickerView {
        let list = self.filtered_versions();
        if list.is_empty() {
            if matches!(self.state(), UpdateState::Checking) {
                return PickerView {
                    value: "Loading...".to_owned(),
                    heading: "Loading versions.json".to_owned(),
                    detail: "Fetching full release and snapshot history from GitHub...".to_owned(),
                    loaded: false,
                };
            }
            return PickerView::default();
        }
        let idx = self.picker_index % list.len();
        let entry = &list[idx];
        let kind = if entry.is_snapshot { "Snapshot" } else { "Release" };
        let date_short = entry.date.split('T').next().unwrap_or(&entry.date);
        let commit_part = if entry.commit.is_empty() {
            String::new()
        } else {
            format!("  ({})", entry.commit)
        };
        let heading = if date_short.is_empty() {
            format!("{}  -  {}{}", entry.id, kind, commit_part)
        } else {
            format!("{}  -  {}  -  {}{}", entry.id, kind, date_short, commit_part)
        };
        let detail = if entry.notes.is_empty() {
            format!("Press A to download and switch to {} ({}/{}).", entry.id, idx + 1, list.len())
        } else {
            format!("{} (Press A to install, {}/{})", entry.notes, idx + 1, list.len())
        };
        PickerView {
            value: entry.id.clone(),
            heading,
            detail,
            loaded: true,
        }
    }

    /// Cycles the selected entry in `Version history` left (`-1`) or right (`+1`).
    pub fn step_picker(&mut self, delta: i32) {
        let list = self.filtered_versions();
        if list.is_empty() {
            self.start_check(false);
            return;
        }
        let len = list.len() as i32;
        let next = ((self.picker_index as i32 % len) + delta).rem_euclid(len) as usize;
        self.picker_index = next;
    }

    /// Handles pressing `A` on the `Version history` row: loads `versions.json` if
    /// not yet fetched, restarts to apply if a download is already ready, or
    /// downloads the currently highlighted version (allowing upgrades or rollbacks).
    pub fn trigger_picker_install(&mut self) -> bool {
        if let UpdateState::Ready { tag } = self.state() {
            log::info!("unx: applying staged version {tag} and relaunching");
            #[cfg(target_os = "horizon")]
            unsafe {
                unx_updater_mark_pending_swap();
            }
            return true;
        }
        if matches!(self.state(), UpdateState::Checking | UpdateState::Downloading { .. }) {
            return false;
        }
        let list = self.filtered_versions();
        if list.is_empty() {
            self.start_check(false);
            return false;
        }
        let entry = &list[self.picker_index % list.len()];
        if entry.nro_url.is_empty() {
            Self::set_state(
                &self.shared,
                UpdateState::Failed(format!("Version {} has no downloadable NRO URL.", entry.id)),
            );
            return false;
        }
        self.start_download(entry.id.clone(), entry.nro_url.clone());
        false
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
        let build = self.build.clone();
        let include_prereleases = self.include_prereleases;
        let spawned = std::thread::Builder::new()
            .name("unx-upd-chk".into())
            .stack_size(256 << 10)
            .spawn(move || {
                run_check(&shared, &build, include_prereleases, announce);
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
fn http_get_text(
    shared: &Shared,
    url: &str,
) -> Result<String, (c_long, c_int, String)> {
    let Ok(c_url) = CString::new(url) else {
        return Err((0, -1, "Invalid URL".to_owned()));
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
    if rc != 0 {
        return Err((http_status, rc, c_buf_to_string(&err)));
    }
    Ok(String::from_utf8_lossy(&body[..out_len.min(body.len())]).into_owned())
}

#[cfg(target_os = "horizon")]
fn run_check(shared: &Shared, build: &BuildVersion, include_prereleases: bool, announce: bool) {
    if !has_internet() {
        if announce {
            Updater::set_state(shared, UpdateState::Idle);
        } else {
            Updater::set_state(shared, UpdateState::Failed("No internet connection.".to_owned()));
        }
        return;
    }

    // 1. Prefer the Minecraft-style `versions.json` manifest (populates full
    //    Version history picker and avoids GitHub REST API rate limits).
    let mut resolved: Option<ReleaseInfo> = None;
    match http_get_text(shared, VERSIONS_MANIFEST_URL) {
        Ok(text) => {
            if shared.cancel.load(Ordering::Acquire) != 0 {
                return;
            }
            if let Some(manifest) = parse_versions_manifest(&text) {
                log::info!(
                    "unx: loaded versions.json ({} versions, latest release={:?}, snapshot={:?})",
                    manifest.versions.len(),
                    manifest.latest_release,
                    manifest.latest_snapshot,
                );
                if let Ok(mut guard) = shared.versions.lock() {
                    *guard = manifest.versions.clone();
                }
                resolved = select_latest_from_manifest(&manifest, include_prereleases);
            } else {
                log::warn!("unx: versions.json parse failed ({} bytes), falling back to GitHub API", text.len());
            }
        }
        Err((status, rc, msg)) => {
            if shared.cancel.load(Ordering::Acquire) != 0 {
                return;
            }
            log::warn!("unx: versions.json fetch failed (HTTP {status}, code {rc}: {msg}), falling back to GitHub API");
        }
    }

    // 2. Fallback to GitHub Releases API if `versions.json` wasn't reachable.
    let release = match resolved {
        Some(info) => info,
        None => {
            let endpoint = if include_prereleases {
                RELEASES_LIST_URL
            } else {
                LATEST_RELEASE_URL
            };
            let text = match http_get_text(shared, endpoint) {
                Ok(t) => t,
                Err((http_status, rc, msg)) => {
                    if shared.cancel.load(Ordering::Acquire) != 0 {
                        return;
                    }
                    if http_status == 404 {
                        Updater::set_state(shared, UpdateState::UpToDate(build.display_label.clone()));
                        return;
                    }
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
            };
            if shared.cancel.load(Ordering::Acquire) != 0 {
                return;
            }
            match parse_github_release_for_channel(&text, ASSET_NAME, include_prereleases) {
                Some(info) => info,
                None => {
                    log::warn!("unx: updater got unexpected JSON from GitHub ({} bytes)", text.len());
                    Updater::set_state(
                        shared,
                        UpdateState::Failed("GitHub returned an unexpected reply.".to_owned()),
                    );
                    return;
                }
            }
        }
    };

    let newer = is_newer_release(
        &release.tag_name,
        &build.version,
        &build.git_sha,
        &build.release_tag,
    );
    log::info!(
        "unx: updater latest={} current={} (ver={} sha={} tag={}) newer={} asset={}",
        release.tag_name,
        build.display_label,
        build.version,
        build.git_sha,
        build.release_tag,
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
fn run_check(shared: &Shared, _build: &BuildVersion, _include_prereleases: bool, _announce: bool) {
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
    let core = strip_v_prefix(s).split(['-', ' ', '(']).next().unwrap_or("");
    core.split('.')
        .filter_map(|part| part.parse::<u32>().ok())
        .collect()
}

/// Parses a pre-release label like `"rc.2"`, `"rc2"`, `"beta.1"`, `"alpha1"` into
/// `(lowercase_prefix, trailing_number)` so `rc.2 > rc.1` and `rc.1 > beta.2`.
fn parse_prerelease_tag(suffix: &str) -> Option<(String, u32)> {
    let s = suffix.trim().to_ascii_lowercase();
    let num_start = s.rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    if num_start >= s.len() {
        return None;
    }
    let num = s[num_start..].parse::<u32>().ok()?;
    let prefix = s[..num_start].trim_end_matches(['.', '-', '_']).to_owned();
    if prefix.is_empty() {
        return None;
    }
    Some((prefix, num))
}

/// Returns `true` if `remote_tag` (e.g. `"v0.1.0-abc1234"`, `"v0.1.1-rc.1"`, or `"v0.2.0"`) is
/// newer than the currently running build (`cur_version`, `cur_sha`, `cur_release_tag`).
pub fn is_newer_release(
    remote_tag: &str,
    cur_version: &str,
    cur_sha: &str,
    cur_release_tag: &str,
) -> bool {
    let r_norm = strip_v_prefix(remote_tag);
    if r_norm.is_empty() {
        return false;
    }
    let r_nums = parse_numeric_prefix(r_norm);
    if r_nums.is_empty() {
        return false;
    }

    let cur_rel_norm = strip_v_prefix(cur_release_tag);
    let c_nums = if !cur_rel_norm.is_empty() {
        let rel_nums = parse_numeric_prefix(cur_rel_norm);
        if !rel_nums.is_empty() {
            rel_nums
        } else {
            parse_numeric_prefix(cur_version)
        }
    } else {
        parse_numeric_prefix(cur_version)
    };

    let max_len = r_nums.len().max(c_nums.len());
    for i in 0..max_len {
        let r = r_nums.get(i).copied().unwrap_or(0);
        let c = c_nums.get(i).copied().unwrap_or(0);
        if r != c {
            return r > c;
        }
    }

    // Equal numeric (major, minor, patch) prefix:
    // 1. Exact release tag match (case-insensitive, ignoring leading `v`).
    if !cur_rel_norm.is_empty() && cur_rel_norm.eq_ignore_ascii_case(r_norm) {
        return false;
    }

    // 2. Remote rolling/pre-release `-<suffix>` matching current git SHA.
    let r_suffix = r_norm.split_once('-').map(|(_, s)| s.trim()).filter(|s| !s.is_empty());
    let c_suffix = cur_rel_norm.split_once('-').map(|(_, s)| s.trim()).filter(|s| !s.is_empty());
    let cur_sha = cur_sha.trim();
    if let Some(suffix) = r_suffix {
        if !cur_sha.is_empty() && !cur_sha.eq_ignore_ascii_case("unknown") && suffix.eq_ignore_ascii_case(cur_sha) {
            return false;
        }
        // 3. Clean release (e.g. `v0.1.0`, no `-` suffix) never downgrades to a
        //    same-version `-<sha>` or `-rc.N` pre-release tag.
        if !cur_rel_norm.is_empty() && c_suffix.is_none() {
            return false;
        }
        // 4. Both current and remote are structured pre-releases of the same version
        //    (e.g. `v0.1.1-rc.2` vs `v0.1.1-rc.1`): order them so we never downgrade.
        if let (Some((rp, rn)), Some((cp, cn))) = (
            parse_prerelease_tag(suffix),
            c_suffix.and_then(parse_prerelease_tag),
        ) {
            return (rp, rn) > (cp, cn);
        }
    }

    // 5. Otherwise offer update (e.g. rolling `v0.1.0 (3deff39)` -> clean `v0.1.0`
    //    or newer rolling `v0.1.0-abc1234`, or pre-release `v0.1.1-rc.1` -> `v0.1.1`).
    true
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
    fn parse_bool(&mut self) -> Option<bool> {
        self.skip_ws();
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Some(true)
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Some(false)
        } else {
            None
        }
    }

    fn parse_release_object(&mut self, wanted_asset: &str) -> Option<(ReleaseInfo, bool, bool)> {
        if !self.expect(b'{') {
            return None;
        }
        let mut tag_name: Option<String> = None;
        let mut asset_url: Option<String> = None;
        let mut draft = false;
        let mut prerelease = false;
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return None;
        }
        loop {
            let key = self.parse_string()?;
            if !self.expect(b':') {
                return None;
            }
            match key.as_str() {
                "tag_name" => tag_name = self.parse_string(),
                "draft" => draft = self.parse_bool().unwrap_or(false),
                "prerelease" => prerelease = self.parse_bool().unwrap_or(false),
                "assets" => {
                    if !self.expect(b'[') {
                        return None;
                    }
                    if self.peek() != Some(b']') {
                        loop {
                            let (name, url) = self.parse_asset_object()?;
                            if name.as_deref() == Some(wanted_asset) {
                                asset_url = url;
                            }
                            match self.peek() {
                                Some(b',') => self.pos += 1,
                                Some(b']') => {
                                    self.pos += 1;
                                    break;
                                }
                                _ => return None,
                            }
                        }
                    } else {
                        self.pos += 1;
                    }
                }
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
                    break;
                }
                _ => break, // allow graceful completion even if response was truncated after assets
            }
        }
        Some((
            ReleaseInfo {
                tag_name: tag_name?,
                asset_url,
            },
            draft,
            prerelease,
        ))
    }

    fn parse_latest_object(&mut self) -> Option<(Option<String>, Option<String>)> {
        if !self.expect(b'{') {
            return None;
        }
        let mut release = None;
        let mut snapshot = None;
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Some((release, snapshot));
        }
        loop {
            let key = self.parse_string()?;
            if !self.expect(b':') {
                return None;
            }
            match key.as_str() {
                "release" => release = self.parse_string(),
                "snapshot" => snapshot = self.parse_string(),
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
                    return Some((release, snapshot));
                }
                _ => return None,
            }
        }
    }

    fn parse_version_entry_object(&mut self) -> Option<VersionEntry> {
        if !self.expect(b'{') {
            return None;
        }
        let mut id: Option<String> = None;
        let mut is_snapshot = false;
        let mut date = String::new();
        let mut commit = String::new();
        let mut nro_url = String::new();
        let mut notes = String::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return None;
        }
        loop {
            let key = self.parse_string()?;
            if !self.expect(b':') {
                return None;
            }
            match key.as_str() {
                "id" => id = self.parse_string(),
                "type" => {
                    if let Some(t) = self.parse_string() {
                        is_snapshot = !t.eq_ignore_ascii_case("release");
                    }
                }
                "date" => {
                    if let Some(s) = self.parse_string() {
                        date = s;
                    }
                }
                "commit" => {
                    if let Some(s) = self.parse_string() {
                        commit = s;
                    }
                }
                "nro_url" => {
                    if let Some(s) = self.parse_string() {
                        nro_url = s;
                    }
                }
                "notes" => {
                    if let Some(s) = self.parse_string() {
                        notes = s;
                    }
                }
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
                    break;
                }
                _ => return None,
            }
        }
        Some(VersionEntry {
            id: id?,
            is_snapshot,
            date,
            commit,
            nro_url,
            notes,
        })
    }
}

/// Parses a Minecraft-style `versions.json` manifest (`latest` + `versions` array).
pub fn parse_versions_manifest(json: &str) -> Option<VersionManifest> {
    let mut sc = JsonScanner::new(json);
    if !sc.expect(b'{') {
        return None;
    }
    let mut latest_release: Option<String> = None;
    let mut latest_snapshot: Option<String> = None;
    let mut versions: Vec<VersionEntry> = Vec::new();
    if sc.peek() == Some(b'}') {
        return None;
    }
    loop {
        let key = sc.parse_string()?;
        if !sc.expect(b':') {
            return None;
        }
        match key.as_str() {
            "latest" => {
                let (rel, snap) = sc.parse_latest_object()?;
                latest_release = rel;
                latest_snapshot = snap;
            }
            "versions" => {
                if !sc.expect(b'[') {
                    return None;
                }
                if sc.peek() != Some(b']') {
                    loop {
                        if let Some(entry) = sc.parse_version_entry_object() {
                            versions.push(entry);
                        } else {
                            return None;
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
            _ => break,
        }
    }
    if versions.is_empty() {
        return None;
    }
    Some(VersionManifest {
        latest_release,
        latest_snapshot,
        versions,
    })
}

/// Selects the newest downloadable release from a parsed `versions.json` manifest
/// for the active channel (`Stable` = latest release, `Pre-release` = newer of
/// `latest.release` and `latest.snapshot`).
pub fn select_latest_from_manifest(
    manifest: &VersionManifest,
    include_prereleases: bool,
) -> Option<ReleaseInfo> {
    let find_by_id = |id: &str| {
        manifest
            .versions
            .iter()
            .find(|v| v.id.eq_ignore_ascii_case(id) && !v.nro_url.is_empty())
    };
    let best_release = manifest
        .latest_release
        .as_deref()
        .and_then(find_by_id)
        .or_else(|| manifest.versions.iter().find(|v| !v.is_snapshot && !v.nro_url.is_empty()));

    let chosen = if !include_prereleases {
        best_release
    } else {
        let best_snapshot = manifest
            .latest_snapshot
            .as_deref()
            .and_then(find_by_id)
            .or_else(|| manifest.versions.iter().find(|v| v.is_snapshot && !v.nro_url.is_empty()));
        match (best_release, best_snapshot) {
            (Some(r), Some(s)) => {
                if is_newer_release(&s.id, "", "", &r.id) {
                    Some(s)
                } else {
                    Some(r)
                }
            }
            (Some(r), None) => Some(r),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        }
    }?;

    Some(ReleaseInfo {
        tag_name: chosen.id.clone(),
        asset_url: Some(chosen.nro_url.clone()),
    })
}

/// Parses GitHub's `/releases/latest` JSON payload, extracting `tag_name` and
/// the `browser_download_url` of the asset matching `wanted_asset`.
#[cfg(test)]
pub fn parse_github_release(json: &str, wanted_asset: &str) -> Option<ReleaseInfo> {
    parse_github_release_for_channel(json, wanted_asset, true)
}

/// Parses either a single GitHub release object (`/releases/latest`) or a release
/// list array (`/releases?per_page=N`), skipping drafts (and pre-releases when
/// `include_prereleases` is `false`) and preferring the newest entry that includes
/// `wanted_asset`.
pub fn parse_github_release_for_channel(
    json: &str,
    wanted_asset: &str,
    include_prereleases: bool,
) -> Option<ReleaseInfo> {
    let mut sc = JsonScanner::new(json);
    match sc.peek()? {
        b'{' => {
            let (info, draft, prerelease) = sc.parse_release_object(wanted_asset)?;
            if draft || (!include_prereleases && prerelease) {
                return None;
            }
            Some(info)
        }
        b'[' => {
            sc.pos += 1;
            if sc.peek() == Some(b']') {
                return None;
            }
            let mut fallback: Option<ReleaseInfo> = None;
            loop {
                match sc.parse_release_object(wanted_asset) {
                    Some((info, draft, prerelease)) => {
                        if !draft && (include_prereleases || !prerelease) {
                            if info.asset_url.is_some() {
                                return Some(info);
                            }
                            if fallback.is_none() {
                                fallback = Some(info);
                            }
                        }
                    }
                    None => break,
                }
                match sc.peek() {
                    Some(b',') => sc.pos += 1,
                    Some(b']') => break,
                    _ => break,
                }
            }
            fallback
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_comparison_handles_semver_and_rolling_sha() {
        // Rolling dev build `v0.1.0 (3deff39)` (cur_release_tag == "")
        assert!(!is_newer_release("v0.1.0-3deff39", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("0.1.0-3deff39", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("V0.1.0-3DEFF39", "0.1.0", "3deff39", ""));
        assert!(is_newer_release("v0.1.0-abc1234", "0.1.0", "3deff39", ""));
        assert!(is_newer_release("v0.1.0", "0.1.0", "3deff39", ""));
        assert!(is_newer_release("v0.1.1-abc1234", "0.1.0", "3deff39", ""));
        assert!(is_newer_release("v0.2.0", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("v0.0.9-fffffff", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("v0.0.9", "0.1.0", "3deff39", ""));

        // Clean tagged release `v0.1.0` (cur_release_tag == "v0.1.0")
        assert!(!is_newer_release("v0.1.0", "0.1.0", "3deff39", "v0.1.0"));
        assert!(!is_newer_release("0.1.0", "0.1.0", "3deff39", "v0.1.0"));
        assert!(!is_newer_release("V0.1.0", "0.1.0", "3deff39", "v0.1.0"));
        // Never downgrade a clean `v0.1.0` to a same-version rolling `-<sha>` tag
        assert!(!is_newer_release("v0.1.0-3deff39", "0.1.0", "3deff39", "v0.1.0"));
        assert!(!is_newer_release("v0.1.0-abc1234", "0.1.0", "3deff39", "v0.1.0"));
        // Higher semver always triggers update from clean release
        assert!(is_newer_release("v0.1.1", "0.1.0", "3deff39", "v0.1.0"));
        assert!(is_newer_release("v0.1.1-abc1234", "0.1.0", "3deff39", "v0.1.0"));
        assert!(is_newer_release("v0.2.0", "0.1.0", "3deff39", "v0.1.0"));
        assert!(!is_newer_release("v0.0.9", "0.1.0", "3deff39", "v0.1.0"));

        // Pre-release tag (e.g. `v0.1.0-rc.1`) upgrades to `v0.1.0-rc.2` or clean `v0.1.0`,
        // never downgrades `rc.2` -> `rc.1` or clean `v0.1.1` -> `v0.1.1-rc.1`.
        assert!(!is_newer_release("v0.1.0-rc1", "0.1.0", "3deff39", "v0.1.0-rc1"));
        assert!(is_newer_release("v0.1.0-rc2", "0.1.0", "3deff39", "v0.1.0-rc1"));
        assert!(!is_newer_release("v0.1.0-rc1", "0.1.0", "3deff39", "v0.1.0-rc2"));
        assert!(is_newer_release("v0.1.1-rc.2", "0.1.1", "3deff39", "v0.1.1-rc.1"));
        assert!(!is_newer_release("v0.1.1-rc.1", "0.1.1", "3deff39", "v0.1.1-rc.2"));
        assert!(is_newer_release("v0.1.1-rc.1", "0.1.0", "3deff39", "v0.1.0"));
        assert!(is_newer_release("v0.1.0", "0.1.0", "3deff39", "v0.1.0-rc1"));
        assert!(is_newer_release("v0.1.1", "0.1.1", "3deff39", "v0.1.1-rc.2"));
        assert!(!is_newer_release("v0.1.1-rc.2", "0.1.1", "3deff39", "v0.1.1"));
        assert!(!is_newer_release("v0.1.0", "0.1.1", "3deff39", "v0.1.1-rc.1"));

        // Malformed / non-semver remote tags never trigger bogus updates
        assert!(!is_newer_release("", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("build-8b5738b", "0.1.0", "3deff39", ""));
        assert!(!is_newer_release("build-8b5738b", "0.1.0", "3deff39", "v0.1.0"));
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
    fn parses_github_releases_array_for_prerelease_and_stable_channels() {
        let sample = r#"[
            {
                "tag_name": "v0.1.2-draft",
                "draft": true,
                "prerelease": true,
                "assets": [{"name": "uprising_nx.nro", "browser_download_url": "https://example.invalid/draft.nro"}]
            },
            {
                "tag_name": "v0.1.1-rc.1",
                "draft": false,
                "prerelease": true,
                "assets": [{"name": "uprising_nx.nro", "browser_download_url": "https://example.invalid/rc1.nro"}]
            },
            {
                "tag_name": "v0.1.0",
                "draft": false,
                "prerelease": false,
                "assets": [{"name": "uprising_nx.nro", "browser_download_url": "https://example.invalid/010.nro"}]
            }
        ]"#;
        let pre = parse_github_release_for_channel(sample, "uprising_nx.nro", true).unwrap();
        assert_eq!(pre.tag_name, "v0.1.1-rc.1");
        assert_eq!(pre.asset_url.as_deref(), Some("https://example.invalid/rc1.nro"));

        let stable = parse_github_release_for_channel(sample, "uprising_nx.nro", false).unwrap();
        assert_eq!(stable.tag_name, "v0.1.0");
        assert_eq!(stable.asset_url.as_deref(), Some("https://example.invalid/010.nro"));
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

    #[test]
    fn parses_versions_manifest_and_selects_by_channel() {
        let sample = r#"{
            "schema": 1,
            "latest": {
                "release": "v0.1.1",
                "snapshot": "v0.1.1-rc.9"
            },
            "versions": [
                {
                    "id": "v0.1.1",
                    "type": "release",
                    "date": "2026-11-25T00:33:35Z",
                    "commit": "0fa63b0",
                    "size": 22803391,
                    "sha256": "abcd",
                    "nro_url": "https://example.invalid/v0.1.1/uprising_nx.nro",
                    "notes": "Official v0.1.1 release."
                },
                {
                    "id": "v0.1.1-rc.9",
                    "type": "snapshot",
                    "date": "2026-11-25T00:25:21Z",
                    "commit": "cd41797",
                    "nro_url": "https://example.invalid/snapshots/uprising_nx-v0.1.1-rc.9.nro",
                    "notes": "Final RC."
                },
                {
                    "id": "v0.1.0",
                    "type": "release",
                    "date": "2026-11-24T04:31:00Z",
                    "commit": "c787de2",
                    "nro_url": "https://example.invalid/v0.1.0/uprising_nx.nro",
                    "notes": "Initial release."
                }
            ]
        }"#;
        let m = parse_versions_manifest(sample).unwrap();
        assert_eq!(m.latest_release.as_deref(), Some("v0.1.1"));
        assert_eq!(m.latest_snapshot.as_deref(), Some("v0.1.1-rc.9"));
        assert_eq!(m.versions.len(), 3);
        assert!(!m.versions[0].is_snapshot);
        assert!(m.versions[1].is_snapshot);

        // Both channels choose v0.1.1 over older v0.1.1-rc.9
        let stable = select_latest_from_manifest(&m, false).unwrap();
        assert_eq!(stable.tag_name, "v0.1.1");
        let pre = select_latest_from_manifest(&m, true).unwrap();
        assert_eq!(pre.tag_name, "v0.1.1");

        // When a newer v0.1.2-rc.1 snapshot exists, Pre-release picks it while Stable stays on v0.1.1
        let mut m2 = m.clone();
        m2.latest_snapshot = Some("v0.1.2-rc.1".into());
        m2.versions.insert(
            0,
            VersionEntry {
                id: "v0.1.2-rc.1".into(),
                is_snapshot: true,
                date: "2026-11-26T00:00:00Z".into(),
                commit: "1234567".into(),
                nro_url: "https://example.invalid/snapshots/uprising_nx-v0.1.2-rc.1.nro".into(),
                notes: "Next snapshot".into(),
            },
        );
        assert_eq!(select_latest_from_manifest(&m2, false).unwrap().tag_name, "v0.1.1");
        assert_eq!(select_latest_from_manifest(&m2, true).unwrap().tag_name, "v0.1.2-rc.1");

        // Picker filters snapshots out on Stable channel and cycles cleanly on Pre-release channel
        let mut upd = Updater::new("0.1.1", "0fa63b0", "v0.1.1", "v0.1.1", false);
        *upd.shared.versions.lock().unwrap() = m2.versions.clone();
        assert_eq!(upd.picker_view().value, "v0.1.1");
        upd.step_picker(1);
        assert_eq!(upd.picker_view().value, "v0.1.0");
        upd.set_include_prereleases(true);
        assert_eq!(upd.picker_view().value, "v0.1.2-rc.1");
        upd.step_picker(1);
        assert_eq!(upd.picker_view().value, "v0.1.1");
    }
}
