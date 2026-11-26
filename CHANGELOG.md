# Changelog

All notable changes to **UprisingNX** are documented in this file.

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
