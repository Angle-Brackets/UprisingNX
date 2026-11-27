# Changelog

All notable changes to **UprisingNX** are documented in this file.

## [0.2.0] - 2026-11-27

### Added
- **Native PC & macOS Port (`uprising-recompiled`):** Self-bootstrapping ahead-of-time (AOT) recompilation of *Kid Icarus: Uprising* now runs natively on **Windows (x64)**, **macOS (Apple Silicon `arm64`)**, and **Linux (x64)** alongside **Nintendo Switch**.
  - **x86-64 AOT Backend:** Direct ARMv6K / Thumb-1 / VFPv2 → x86-64 compiler (`zakuro/crates/zakuro-aot/src/compiler_x64/`) with 1.0× instruction deduplication, direct-mapped icache, and fast first-launch ROM compilation.
  - **macOS Apple Silicon Support:** Native `MAP_JIT` + `pthread_jit_write_protect_np` + 16 KiB page support, bundled `MoltenVK >= 1.3.0` (`VK_KHR_portability_enumeration` / `VK_KHR_portability_subset`), `.app` + `.dmg` bundler (`scripts/pc/build-macos.sh`), Hardened Runtime entitlements, and automated Apple Developer ID signing + notarization in CI.
  - **Up to 16× HD Upscaling & Hardware-Aware Defaults:** Vulkan 1.3 renderer with automatic first-launch quality/resolution profiling based on detected GPU class (dedicated desktop GPU, Apple Silicon unified memory, integrated/handheld).
  - **Keyboard + Mouse, Twin-Stick & Native HID Gyro Aiming:** Direct mouse aiming (`WASD` + `LMB` fire + `RMB`/scroll powers), Switch-parity twin-stick + Z-targeting (`L`/`ZL`/`ZR` fire, `R` Z-target, `Y`/`L3` recenter), and native HID gyroscope support (`hidapi`) for DualSense, DualShock 4, Switch Pro, and Steam Deck controllers.
  - **Unified Kid Icarus Parchment UI:** Animated 5-tab Settings screen (`Graphics`, `Audio`, `Controls`, `Controller`, `System`) with live stick/deadzone visualizer, auto-detected button glyph styles (`Xbox` / `PlayStation` / `Nintendo`), first-launch ROM importer, and in-game `Esc` Pause menu.
- **3-Platform Desktop CI:** GitHub Actions (`tag-build.yml`) now runs native Apple Silicon `zakuro-aot` differential/fuzz tests on `macos-15` and publishes `uprising-recompiled-windows-x64.exe`, `uprising-recompiled-macos-arm64.dmg`, and `uprising-recompiled-linux-x64.tar.gz`.

### Fixed
- **5 AArch64 AOT compiler correctness fixes (benefits both Nintendo Switch and Apple Silicon):**
  - `STREX*` now unconditionally clears the exclusive monitor on both pass and fail.
  - `UMULLS` / `SMULLS` / `UMLALS` / `SMLALS` no longer corrupt guest `CPSR.C` when testing the 64-bit zero flag.
  - Added `STALE_PTR` checks on function entry so patched/stale direct-called and tail-jumped callees unwind cleanly after `LoadedAot::check()`.
  - Lowered `VCVTR` natively via `FRINTI` + `FCVTZS`/`FCVTZU` (honoring `FPSCR.RMode` and `FPSCR.FZ` with zero interpreter fallbacks).
  - Isolated host `FPCR` across `LoadedAot::enter` and Rust helper calls.

---

## [0.1.3] - 2026-11-26

### Fixed
- **1-line & chapter-intro battle subtitle alignment:** Fixed a regression from `v0.1.1` where short/medium single-line subtitles (narrower than `200 px` or shorter than `20 px`) were rejected by `is_top_subtitle_quad` and left-aligned instead of centered. Also made 512×128 glyph line tracking (`note_text`) unconditional and preserved `text_lines` across transitions/pauses so the opening subtitle of a chapter, post-cutscene lines, and mid-dialogue pauses stay centered while keeping Chapter End bonus labels untouched.
- **Arms Altar Redeem Hearts blackout:** Fixed a false-positive in Chapter Select detection (`chapter_step`) that triggered prematurely on `512x256 Etc1` → `128x128 Etc1A4` when hovering **Redeem Hearts** (*Buy weapons.*) in Arms Altar, blacking out the temple background and squishing the buttons. Chapter Select now waits for its `256x64 Etc1` stage preview banner before activating.

### Documentation
- Added a minimalist Switch controls diagram (`docs/controls.svg`) to `README.md`.

---

## [0.1.2] - 2026-11-25

### Added
- **Minecraft-style `versions.json` manifest:** In-app Switch auto-updater now reads `versions.json` directly from the rolling `snapshots` archive release (avoiding GitHub REST API rate limits, with automatic fallback to the GitHub Releases API).
- **In-app Version History picker:** Added **Settings → Save & System → Version history** to browse past official releases (and archived snapshots on the Pre-release channel) and upgrade or roll back directly on the Switch.
- **Rolling `snapshots` pre-release archive:** Consolidated pre-release / RC binaries under a single rolling `snapshots` pre-release with `scripts/snapshot.sh`.

---

## [0.1.1] - 2026-11-25

### Fixed
- **Land Battle Y-axis inversion (#1):** Pushing the right stick or tilting the gyro upward on foot now consistently aims upward.
- **Orbit & air-battle reticle stability:** Suppressed idle auto-centering and orbit screen-follow drift so the crosshair stays locked on target while strafing or orbiting (including the Chapter 25 Great Sacred Treasure battle).
- **Chapter End bonus alignment:** Prevented Chapter End bonus labels from being intercepted and shifted by battle subtitle centering.

### Added
- **Separate Air & Ground Invert Y settings:** Independent **Invert Y (air)** and **Invert Y (ground)** toggles in **Settings → Controls** with automatic migration of existing `settings.ini` values.
- **Upgraded Z-targeting & split recenter controls:** Split **R** (Quick Turn / 2-tier Z-targeting + reticle center), **Y** (reticle-only center on foot), and **L3** (behind-Pit camera + reticle center).
- **Merged 16:9 in-battle Pause screen:** Main Pause, Options/Controls/Volume, Quit Confirmation, and Remove Display (photo mode) render seamlessly over the frozen top screen.
- **Opt-in Pre-release update channel:** Added **Settings → Save & System → Update channel** (`Stable` vs `Pre-release`).

---

## [0.1.0] - 2026-11-24

### Added
- First public release of **UprisingNX**: self-bootstrapping ARMv6K → AArch64 AOT recompiler on Nintendo Switch, native 16:9 presentation, dual-stick + gyro aiming, 2-zone Z-targeting, baked-in 493-key PICA200 shader warmup, 3DS save import/export, and in-app OTA auto-updater.
