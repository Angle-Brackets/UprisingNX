//! Save file import/export helpers for Kid Icarus: Uprising (USA `0004000000030100`).
//!
//! Decrypted 3DS saves dumped with Checkpoint, JKSM, Citra, or Azahar (`00.SAV`,
//! `01.SAV`, `02.SAV`, `03.SAV`) are binary-compatible with Zakuro's host archive.
//! This module supports:
//! - Drop-in auto-import on launch from `sdmc:/switch/uprisingnx/save/import/`
//! - Manual in-menu import with automatic timestamped backup of any live Switch save
//! - Manual in-menu export to `sdmc:/switch/uprisingnx/save/export/`

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SAVE_ROOT: &str = "sdmc:/switch/uprisingnx/save";
pub const SAVE_IMPORT_DIR: &str = "sdmc:/switch/uprisingnx/save/import";
pub const SAVE_EXPORT_DIR: &str = "sdmc:/switch/uprisingnx/save/export";

pub const KIU_TITLE_ID_HEX: &str = "0004000000030100";

/// Exact 16-byte archive format metadata expected by Kid Icarus: Uprising
/// (`total_size = 0x0004_0000` / 256 KiB, `directories = 1`, `files = 4`,
/// `duplicate_data = 1`). Writing this alongside imported `.SAV` files ensures
/// a fresh install never treats an imported save as unformatted.
pub const KIU_SAVE_FORMAT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x04, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
];

/// Maximum size of a single `.SAV` file we will accept (KIU's entire save
/// archive is 256 KiB).
const MAX_SAV_FILE_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveOpReport {
    pub files: Vec<String>,
    pub backup_dir: Option<PathBuf>,
}

impl SaveOpReport {
    pub fn count(&self) -> usize {
        self.files.len()
    }
}

pub fn live_save_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("user").join("savedata").join(KIU_TITLE_ID_HEX)
}

pub fn live_format_file(data_dir: &Path) -> PathBuf {
    data_dir
        .join("user")
        .join("savedata")
        .join(format!("{KIU_TITLE_ID_HEX}.format"))
}

fn is_valid_sav_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    // Skip macOS AppleDouble (`._00.SAV`) and other hidden files.
    if name.starts_with('.') {
        return false;
    }
    let is_sav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("sav"));
    if !is_sav {
        return false;
    }
    match fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.len() > 0 && meta.len() <= MAX_SAV_FILE_BYTES,
        Err(_) => false,
    }
}

fn direct_sav_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_valid_sav_file(p))
        .collect();
    files.sort();
    files
}

/// Finds `.SAV` files inside `dir`, checking direct files first and then up to
/// `max_depth` nested subdirectories (handles Checkpoint/JKSM/Citra folders
/// dropped whole into `save/import/`).
fn find_sav_files_recursive(dir: &Path, max_depth: usize) -> Option<(PathBuf, Vec<PathBuf>)> {
    let direct = direct_sav_files(dir);
    if !direct.is_empty() {
        return Some((dir.to_path_buf(), direct));
    }
    if max_depth == 0 {
        return None;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return None;
    };
    let mut subdirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| !n.starts_with('.') && n != "__MACOSX")
        })
        .collect();
    subdirs.sort();
    for sub in subdirs {
        if let Some(found) = find_sav_files_recursive(&sub, max_depth - 1) {
            return Some(found);
        }
    }
    None
}

fn find_import_batch(save_root: &Path, check_root_fallback: bool) -> Option<(PathBuf, Vec<PathBuf>)> {
    let import_dir = save_root.join("import");
    if let Some(found) = find_sav_files_recursive(&import_dir, 3) {
        return Some(found);
    }
    if check_root_fallback {
        let direct = direct_sav_files(save_root);
        if !direct.is_empty() {
            return Some((save_root.to_path_buf(), direct));
        }
    }
    None
}

/// Counts `.SAV` files currently queued for import.
pub fn pending_import_count_in(save_root: &Path) -> usize {
    find_import_batch(save_root, true).map_or(0, |(_, files)| files.len())
}

/// Counts `.SAV` files currently in the live Switch save directory.
pub fn live_save_count(data_dir: &Path) -> usize {
    direct_sav_files(&live_save_dir(data_dir)).len()
}

pub fn pending_import_count() -> usize {
    pending_import_count_in(Path::new(SAVE_ROOT))
}

fn unique_stamped_dir(parent: &Path, prefix: &str) -> PathBuf {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let base = parent.join(format!("{prefix}_{secs}"));
    if !base.exists() {
        return base;
    }
    for n in 1u32..1000 {
        let candidate = parent.join(format!("{prefix}_{secs}_{n}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    base
}

fn backup_live_saves(data_dir: &Path, save_root: &Path) -> Result<Option<PathBuf>, String> {
    let live_dir = live_save_dir(data_dir);
    let existing = direct_sav_files(&live_dir);
    if existing.is_empty() {
        return Ok(None);
    }
    let backup_dir = unique_stamped_dir(save_root, "backup");
    fs::create_dir_all(&backup_dir)
        .map_err(|e| format!("Could not create backup folder: {e}"))?;
    for src in &existing {
        let Some(name) = src.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let dst = backup_dir.join(name.to_ascii_uppercase());
        fs::copy(src, &dst).map_err(|e| format!("Backup failed for {name}: {e}"))?;
    }
    let fmt = live_format_file(data_dir);
    if fmt.is_file() {
        let _ = fs::copy(&fmt, backup_dir.join(format!("{KIU_TITLE_ID_HEX}.format")));
    }
    Ok(Some(backup_dir))
}

fn move_file(src: &Path, dst: &Path) {
    if fs::rename(src, dst).is_err() && fs::copy(src, dst).is_ok() {
        let _ = fs::remove_file(src);
    }
}

fn prune_empty_parents(mut dir: PathBuf, stop_at: &Path) {
    while dir != stop_at && dir.starts_with(stop_at) {
        if fs::remove_dir(&dir).is_err() {
            break;
        }
        if !dir.pop() {
            break;
        }
    }
}

/// Core save import routine: backs up any existing live saves, copies the
/// incoming `.SAV` files with uppercase filenames, writes KIU's 16-byte
/// `.format` metadata header, and archives the consumed import files under
/// `save/imported/` so future boots never overwrite new progress.
pub fn import_saves_in(
    data_dir: &Path,
    save_root: &Path,
    check_root_fallback: bool,
) -> Result<Option<SaveOpReport>, String> {
    let Some((source_dir, source_files)) = find_import_batch(save_root, check_root_fallback) else {
        return Ok(None);
    };

    let backup_dir = backup_live_saves(data_dir, save_root)?;

    let live_dir = live_save_dir(data_dir);
    fs::create_dir_all(&live_dir).map_err(|e| format!("Could not create save folder: {e}"))?;

    // Clear old live .SAV files now that they are safely backed up so an
    // imported save never mixes files from two different playthroughs.
    for old in direct_sav_files(&live_dir) {
        let _ = fs::remove_file(old);
    }

    let mut imported_names = Vec::with_capacity(source_files.len());
    for src in &source_files {
        let name = src
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "Invalid save filename".to_owned())?
            .to_ascii_uppercase();
        let dst = live_dir.join(&name);
        fs::copy(src, &dst).map_err(|e| format!("Could not copy {name}: {e}"))?;
        imported_names.push(name);
    }

    let fmt_path = live_format_file(data_dir);
    if let Some(parent) = fmt_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&fmt_path, KIU_SAVE_FORMAT_BYTES)
        .map_err(|e| format!("Could not write save format header: {e}"))?;

    // Archive consumed files out of import/ so they are not re-imported.
    let imported_root = save_root.join("imported");
    let archive_dir = unique_stamped_dir(&imported_root, "batch");
    if fs::create_dir_all(&archive_dir).is_ok() {
        for src in &source_files {
            if let Some(name) = src.file_name().and_then(|n| n.to_str()) {
                move_file(src, &archive_dir.join(name.to_ascii_uppercase()));
            }
        }
    }
    let import_dir = save_root.join("import");
    if source_dir != import_dir && source_dir.starts_with(&import_dir) {
        prune_empty_parents(source_dir, &import_dir);
    }

    Ok(Some(SaveOpReport {
        files: imported_names,
        backup_dir,
    }))
}

/// Core save export routine: copies all live `.SAV` files into `save/export/`.
pub fn export_saves_in(data_dir: &Path, save_root: &Path) -> Result<SaveOpReport, String> {
    let live_dir = live_save_dir(data_dir);
    let live_files = direct_sav_files(&live_dir);
    if live_files.is_empty() {
        return Err("No live Switch save files found yet.".to_owned());
    }

    let export_dir = save_root.join("export");
    fs::create_dir_all(&export_dir).map_err(|e| format!("Could not create export folder: {e}"))?;

    let mut exported_names = Vec::with_capacity(live_files.len());
    for src in &live_files {
        let name = src
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "Invalid save filename".to_owned())?
            .to_ascii_uppercase();
        let dst = export_dir.join(&name);
        fs::copy(src, &dst).map_err(|e| format!("Could not export {name}: {e}"))?;
        exported_names.push(name);
    }

    Ok(SaveOpReport {
        files: exported_names,
        backup_dir: None,
    })
}

/// Called during startup before `loader::load` to auto-import any saves dropped
/// into `sdmc:/switch/uprisingnx/save/import/`.
pub fn auto_import_on_boot(data_dir: &Path) -> Option<String> {
    match import_saves_in(data_dir, Path::new(SAVE_ROOT), false) {
        Ok(Some(report)) => {
            let n = report.count();
            let s = if n == 1 { "" } else { "s" };
            log::info!(
                "unx: auto-imported {n} save file{s} ({:?}), backup={:?}",
                report.files,
                report.backup_dir
            );
            Some(format!("Imported {n} save file{s} from save/import/"))
        }
        Ok(None) => None,
        Err(err) => {
            log::warn!("unx: save auto-import failed: {err}");
            Some(format!("Save import failed: {err}"))
        }
    }
}

/// Called from the in-game Settings menu's `Save & System` tab.
pub fn import_from_menu(data_dir: &Path) -> Result<String, String> {
    match import_saves_in(data_dir, Path::new(SAVE_ROOT), true)? {
        Some(report) => {
            let n = report.count();
            let s = if n == 1 { "" } else { "s" };
            let backup_note = if report.backup_dir.is_some() {
                " (old save backed up)"
            } else {
                ""
            };
            log::info!(
                "unx: menu imported {n} save file{s} ({:?}), backup={:?}",
                report.files,
                report.backup_dir
            );
            Ok(format!("Imported {n} file{s}{backup_note}. Restart to load."))
        }
        None => Err("No .SAV files found in switch/uprisingnx/save/import/".to_owned()),
    }
}

/// Called from the in-game Settings menu's `Save & System` tab.
pub fn export_from_menu(data_dir: &Path) -> Result<String, String> {
    let report = export_saves_in(data_dir, Path::new(SAVE_ROOT))?;
    let n = report.count();
    let s = if n == 1 { "" } else { "s" };
    log::info!("unx: exported {n} save file{s} ({:?}) to {SAVE_EXPORT_DIR}", report.files);
    Ok(format!("Exported {n} file{s} to switch/uprisingnx/save/export/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("unx_saves_test_{tag}_{secs}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn import_normalizes_case_writes_format_and_backs_up_existing() {
        let root = temp_root("import");
        let data_dir = root.join("data");
        let save_root = root.join("save");
        let import_dir = save_root.join("import");
        fs::create_dir_all(&import_dir).unwrap();

        // Seed an existing live Switch save
        let live = live_save_dir(&data_dir);
        fs::create_dir_all(&live).unwrap();
        fs::write(live.join("00.SAV"), b"OLD_SAVE_00").unwrap();
        fs::write(live.join("02.SAV"), b"OLD_SAVE_02").unwrap();

        // Drop lowercase Checkpoint files + a macOS AppleDouble dotfile to ignore
        fs::write(import_dir.join("00.sav"), b"NEW_3DS_00").unwrap();
        fs::write(import_dir.join("01.SAV"), b"NEW_3DS_01").unwrap();
        fs::write(import_dir.join("._00.sav"), b"MACOS_JUNK").unwrap();

        assert_eq!(pending_import_count_in(&save_root), 2);
        assert_eq!(live_save_count(&data_dir), 2);

        let report = import_saves_in(&data_dir, &save_root, false)
            .unwrap()
            .expect("should import");
        assert_eq!(report.files, vec!["00.SAV", "01.SAV"]);
        let backup = report.backup_dir.expect("should back up live save");
        assert_eq!(fs::read(backup.join("00.SAV")).unwrap(), b"OLD_SAVE_00");
        assert_eq!(fs::read(backup.join("02.SAV")).unwrap(), b"OLD_SAVE_02");

        // Live dir should contain only the imported files in uppercase + exact .format header
        assert_eq!(fs::read(live.join("00.SAV")).unwrap(), b"NEW_3DS_00");
        assert_eq!(fs::read(live.join("01.SAV")).unwrap(), b"NEW_3DS_01");
        assert!(!live.join("02.SAV").exists());
        assert_eq!(fs::read(live_format_file(&data_dir)).unwrap(), KIU_SAVE_FORMAT_BYTES);

        // Consumed import files should be archived out of import/
        assert_eq!(pending_import_count_in(&save_root), 0);
        assert!(!import_dir.join("00.sav").exists());
        assert!(!import_dir.join("01.SAV").exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn import_handles_nested_checkpoint_subfolder_and_export_roundtrips() {
        let root = temp_root("nested");
        let data_dir = root.join("data");
        let save_root = root.join("save");
        let nested = save_root.join("import").join("2026_05_01_Checkpoint");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("00.SAV"), b"SLOT_A_DATA").unwrap();
        fs::write(nested.join("01.sav"), b"SLOT_B_DATA").unwrap();

        let report = import_saves_in(&data_dir, &save_root, false)
            .unwrap()
            .expect("should import nested folder");
        assert_eq!(report.files, vec!["00.SAV", "01.SAV"]);
        assert!(report.backup_dir.is_none());
        assert!(!nested.exists(), "empty imported subfolder should be pruned");

        let export = export_saves_in(&data_dir, &save_root).unwrap();
        assert_eq!(export.files, vec!["00.SAV", "01.SAV"]);
        assert_eq!(
            fs::read(save_root.join("export").join("00.SAV")).unwrap(),
            b"SLOT_A_DATA"
        );
        assert_eq!(
            fs::read(save_root.join("export").join("01.SAV")).unwrap(),
            b"SLOT_B_DATA"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
