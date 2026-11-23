//! Self-bootstrapping ROM discovery, validation, on-Switch ARMv6K -> AArch64 AOT
//! cache construction, and single-CodeMemory `jitCreate` streaming loader.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use zakuro_aot::{
    compile_title, AotImage, CompileOptions, ExecBuffer, LoadedAot, FORMAT_VERSION,
    KIU_USA_PROGRAM_ID,
};
use zakuro_fs::{FsError, Title};
use zakuro_gpu::backend::vulkan::VulkanPresenter;
use zakuro_gpu::backend::{Presenter, ScreenImage};

use crate::menu::Menu;

extern "C" {
    fn unx_aot_alloc(len: usize, rw_out: *mut *mut u8, rx_out: *mut *const u8) -> i32;
    fn unx_aot_finish() -> i32;
    fn unx_aot_free();
    fn unx_bootstrap_step() -> i32;
}

pub const CACHE_DIR: &str = "sdmc:/switch/uprisingnx/cache";
pub const PRIMARY_ROM_DIR: &str = "sdmc:/switch/uprisingnx/rom";

#[derive(Debug, Clone)]
pub struct SetupNotice {
    pub heading: String,
    pub paragraphs: Vec<String>,
}

impl SetupNotice {
    pub fn applet_mode() -> Self {
        Self {
            heading: "Title Override Required".to_owned(),
            paragraphs: vec![
                "UprisingNX was launched in Album (Applet) mode, which restricts memory and JIT code permissions.".to_owned(),
                String::new(),
                "To play UprisingNX:".to_owned(),
                "1. Return to the Switch HOME Menu.".to_owned(),
                "2. Hold R while launching any installed Switch game to open Homebrew Menu with full Title Override access.".to_owned(),
                "3. Launch UprisingNX again.".to_owned(),
            ],
        }
    }

    pub fn missing_rom() -> Self {
        Self {
            heading: "ROM Required".to_owned(),
            paragraphs: vec![
                "Place a decrypted US copy of Kid Icarus: Uprising (.cia, .3ds, or .cci) in:".to_owned(),
                "sdmc:/switch/uprisingnx/rom/".to_owned(),
                String::new(),
                "Any filename works (for example: baserom.cia, baserom.3ds, or baserom.cci).".to_owned(),
                "Dump and decrypt your cartridge or eShop title on your 3DS using GodMode9.".to_owned(),
            ],
        }
    }

    pub fn encrypted_rom(name: &str) -> Self {
        Self {
            heading: "Encrypted ROM Detected".to_owned(),
            paragraphs: vec![
                format!("Found '{name}', but it is still encrypted."),
                String::new(),
                "Please export a decrypted .cia, .3ds, or .cci using GodMode9 on your 3DS (NCSD/CIA image options -> Build CIA / Decrypt file) and place it in:".to_owned(),
                "sdmc:/switch/uprisingnx/rom/".to_owned(),
            ],
        }
    }

    pub fn wrong_title(name: &str, found_id: u64) -> Self {
        let region_hint = match found_id {
            0x0004_0000_0003_0200 => " (this is the European release)",
            0x0004_0000_0003_0000 => " (this is the Japanese release)",
            _ => "",
        };
        Self {
            heading: "Unsupported ROM Version".to_owned(),
            paragraphs: vec![
                format!("Found '{name}' with Title ID {found_id:016X}{region_hint}."),
                String::new(),
                "UprisingNX requires the USA release of Kid Icarus: Uprising (Title ID 0004000000030100, CTR-P-AKDE).".to_owned(),
                "Place your decrypted US .cia, .3ds, or .cci in:".to_owned(),
                "sdmc:/switch/uprisingnx/rom/".to_owned(),
            ],
        }
    }
}

pub fn ensure_sd_layout() {
    for dir in [
        "sdmc:/switch/uprisingnx",
        PRIMARY_ROM_DIR,
        CACHE_DIR,
        "sdmc:/switch/uprisingnx/data",
    ] {
        let _ = fs::create_dir_all(dir);
    }
}

fn is_rom_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "cia" | "3ds" | "cci" | "cxi"))
}

fn rom_priority(path: &Path) -> (u8, String) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
    let rank = if name.starts_with("baserom.") {
        0
    } else if name.contains("icarus") || name.contains("uprising") || name.contains("akde") {
        1
    } else {
        2
    };
    (rank, name)
}

/// Discovers and validates a decrypted US Kid Icarus: Uprising ROM on the SD card.
pub fn discover_rom(explicit_rom: Option<&str>) -> Result<(PathBuf, Title), SetupNotice> {
    ensure_sd_layout();

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(arg) = explicit_rom.filter(|s| !s.is_empty()) {
        let p = PathBuf::from(arg);
        if p.is_file() {
            candidates.push(p);
        }
    }

    for dir in [
        PRIMARY_ROM_DIR,
        "sdmc:/switch/uprisingnx/roms",
        "sdmc:/switch/uprisingnx",
        "sdmc:/uprisingnx/rom",
        "sdmc:/uprisingnx",
        "sdmc:/switch/dekopon/roms",
    ] {
        if let Ok(entries) = fs::read_dir(dir) {
            let mut dir_files: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && is_rom_extension(p))
                .collect();
            dir_files.sort_by_key(|p| rom_priority(p));
            for f in dir_files {
                if !candidates.contains(&f) {
                    candidates.push(f);
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(SetupNotice::missing_rom());
    }

    let mut encrypted_seen: Option<String> = None;
    let mut wrong_id_seen: Option<(String, u64)> = None;

    for path in candidates {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("rom")
            .to_owned();
        match Title::load(&path) {
            Ok(title) if title.program_id() == KIU_USA_PROGRAM_ID => {
                log::info!("unx: selected ROM {} ({})", path.display(), title.describe());
                return Ok((path, title));
            }
            Ok(title) => {
                log::warn!(
                    "unx: skipping {} (title_id {:016X} != {:016X})",
                    path.display(),
                    title.program_id(),
                    KIU_USA_PROGRAM_ID
                );
                if wrong_id_seen.is_none() {
                    wrong_id_seen = Some((file_name, title.program_id()));
                }
            }
            Err(FsError::Encrypted(_) | FsError::EncryptedCia) => {
                log::warn!("unx: ROM {} is encrypted", path.display());
                if encrypted_seen.is_none() {
                    encrypted_seen = Some(file_name);
                }
            }
            Err(err) => {
                log::warn!("unx: could not open {}: {err}", path.display());
            }
        }
    }

    if let Some(name) = encrypted_seen {
        Err(SetupNotice::encrypted_rom(&name))
    } else if let Some((name, id)) = wrong_id_seen {
        Err(SetupNotice::wrong_title(&name, id))
    } else {
        Err(SetupNotice::missing_rom())
    }
}

pub fn render_card(
    presenter: &mut VulkanPresenter,
    menu: &mut Menu,
    heading: &str,
    paragraphs: &[&str],
    progress: Option<(f32, &str)>,
    footer: Option<&str>,
    is_error: bool,
) {
    let (w, h) = presenter.extent();
    let overlay = menu.status_overlay(heading, paragraphs, progress, footer, is_error, w, h);
    let blank: &[u8] = &[0, 0, 0, 255];
    let mk_img = || ScreenImage { width: 1, height: 1, pixels: blank, gpu: None };
    let _ = presenter.present(mk_img(), mk_img(), mk_img(), &overlay);
}

fn alloc_nx_jit(
    len: usize,
    fill: &mut dyn FnMut(&mut [u8]) -> Result<(), String>,
) -> Result<ExecBuffer, String> {
    let mut rw: *mut u8 = std::ptr::null_mut();
    let mut rx: *const u8 = std::ptr::null();
    let rc = unsafe { unx_aot_alloc(len, &mut rw, &mut rx) };
    if rc != 0 || rw.is_null() || rx.is_null() {
        return Err(format!(
            "jitCreate failed for {:.1} MiB — ensure UprisingNX is running via Title Override (not Album)",
            len as f64 / (1024.0 * 1024.0)
        ));
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(rw, len) };
    if let Err(e) = fill(slice) {
        unsafe { unx_aot_free() };
        return Err(e);
    }
    if unsafe { unx_aot_finish() } != 0 {
        unsafe { unx_aot_free() };
        return Err("jitTransitionToExecutable failed for AOT CodeMemory".to_owned());
    }
    Ok(unsafe {
        ExecBuffer::from_external(
            rx,
            len,
            Some(Box::new(|| {
                unx_aot_free();
            })),
        )
    })
}

/// Loads a cached AOT binary from `sdmc:/switch/uprisingnx/cache/` in ~0.3s, or compiles it on
/// the Switch from the user's ROM on first launch while displaying a live progress bar.
pub fn load_or_build_aot(
    title: &Title,
    data_dir: &Path,
    presenter: &mut VulkanPresenter,
    menu: &mut Menu,
) -> Result<LoadedAot, SetupNotice> {
    let (code_bytes, modded) = zakuro_core::mods::code(title, Some(data_dir)).map_err(|e| SetupNotice {
        heading: "Could Not Read ROM Code".to_owned(),
        paragraphs: vec![format!("Failed to decompress .code from ROM: {e}")],
    })?;
    let parsed = zakuro_aot::discover::Image::from_exheader(&title.exheader, &code_bytes);
    let text_hash = parsed.text.hash();
    let program_id = title.program_id();
    drop(parsed);

    let cache_path = PathBuf::from(format!(
        "{CACHE_DIR}/aot_v{FORMAT_VERSION}_{program_id:016X}_{text_hash:016X}.bin"
    ));

    if cache_path.is_file() {
        render_card(
            presenter,
            menu,
            "Starting Game",
            &["Loading native Switch code cache from SD card..."],
            Some((1.0, "Reading cached AOT binary...")),
            None,
            false,
        );
        let t0 = Instant::now();
        match File::open(&cache_path)
            .map_err(|e| e.to_string())
            .and_then(|f| {
                let reader = BufReader::with_capacity(1 << 20, f);
                LoadedAot::read_streaming(reader, alloc_nx_jit)
            }) {
            Ok(loaded) if loaded.program_id == program_id && loaded.text_hash == text_hash => {
                log::info!(
                    "unx: streamed cached AOT {} in {:?}{}",
                    cache_path.display(),
                    t0.elapsed(),
                    if modded { " (modded code)" } else { "" }
                );
                return Ok(loaded);
            }
            Ok(_) => {
                log::warn!("unx: cached AOT header mismatch, rebuilding");
                let _ = fs::remove_file(&cache_path);
            }
            Err(err) => {
                log::warn!("unx: cached AOT load failed ({err}), rebuilding");
                let _ = fs::remove_file(&cache_path);
            }
        }
    }

    // Cache miss: build native AArch64 AOT on the Switch with live progress bar!
    log::info!(
        "unx: building AOT cache {} (program_id={program_id:016X}, text_hash={text_hash:016X})",
        cache_path.display()
    );
    let pct = Arc::new(AtomicUsize::new(0));
    let pct_worker = Arc::clone(&pct);
    let cache_worker = cache_path.clone();

    let handle = std::thread::Builder::new()
        .name("aot-build".into())
        .stack_size(16 << 20)
        .spawn({
            let exheader = title.exheader.clone();
            move || -> Result<AotImage, String> {
                let t0 = Instant::now();
                let image = compile_title(
                    program_id,
                    &exheader,
                    &code_bytes,
                    &[],
                    CompileOptions::for_title(program_id),
                    |done, total| {
                        let p = if total > 0 { (done * 88) / total } else { 0 };
                        pct_worker.store(p.min(88), Ordering::Relaxed);
                    },
                );
                drop(code_bytes);
                log::info!(
                    "unx: compiled {:.2} MiB AArch64 AOT in {:.2?}",
                    image.code_bytes().len() as f64 / (1024.0 * 1024.0),
                    t0.elapsed()
                );

                pct_worker.store(90, Ordering::Relaxed);
                let tmp_path = cache_worker.with_extension("bin.tmp");
                let write_res = File::create(&tmp_path)
                    .and_then(|f| {
                        let mut writer = BufWriter::with_capacity(1 << 20, f);
                        image.write_to(&mut writer)
                    })
                    .and_then(|()| fs::rename(&tmp_path, &cache_worker));
                if let Err(e) = write_res {
                    let _ = fs::remove_file(&tmp_path);
                    log::warn!("unx: could not save AOT cache {}: {e}", cache_worker.display());
                } else {
                    log::info!("unx: saved AOT cache to {}", cache_worker.display());
                }
                pct_worker.store(98, Ordering::Relaxed);
                Ok(image)
            }
        })
        .map_err(|e| SetupNotice {
            heading: "Bootstrap Error".to_owned(),
            paragraphs: vec![format!("Could not spawn AOT compiler thread: {e}")],
        })?;

    while !handle.is_finished() {
        if unsafe { unx_bootstrap_step() } != 0 {
            return Err(SetupNotice {
                heading: "Cancelled".to_owned(),
                paragraphs: vec!["Setup was cancelled.".to_owned()],
            });
        }
        let p = pct.load(Ordering::Relaxed);
        let stage = if p < 10 {
            "Analyzing ARMv6K control flow & jump tables..."
        } else if p < 89 {
            "Compiling ARMv6K & VFPv2 instructions to native Switch AArch64..."
        } else {
            "Writing compiled game cache to SD card..."
        };
        render_card(
            presenter,
            menu,
            "First-Time Game Setup",
            &[
                "Constructing native Switch executable cache from your Kid Icarus: Uprising ROM.",
                "This only runs once (~10-25 seconds). Subsequent launches start in under a second!",
            ],
            Some((p as f32 / 100.0, stage)),
            Some("Please do not turn off the console while the cache is saved."),
            false,
        );
        std::thread::sleep(Duration::from_millis(16));
    }

    let image = match handle.join() {
        Ok(Ok(img)) => img,
        Ok(Err(err)) => {
            return Err(SetupNotice {
                heading: "AOT Compilation Failed".to_owned(),
                paragraphs: vec![err],
            });
        }
        Err(_) => {
            return Err(SetupNotice {
                heading: "AOT Compilation Panicked".to_owned(),
                paragraphs: vec!["The AOT compiler thread panicked unexpectedly.".to_owned()],
            });
        }
    };

    render_card(
        presenter,
        menu,
        "First-Time Game Setup",
        &["Finalizing executable memory..."],
        Some((1.0, "Launching Kid Icarus: Uprising...")),
        None,
        false,
    );

    let code_len = image.code_bytes().len();
    let exec = alloc_nx_jit(code_len, &mut |dst| {
        dst.copy_from_slice(image.code_bytes());
        Ok(())
    })
    .map_err(|err| SetupNotice {
        heading: "Executable Memory Error".to_owned(),
        paragraphs: vec![err],
    })?;

    let AotImage { program_id, text_hash, enter_offset, entries, origins, code } = image;
    drop(code);
    Ok(LoadedAot::new(program_id, text_hash, enter_offset, exec, entries, origins))
}
