# UprisingNX

**UprisingNX** is a self-bootstrapping ahead-of-time (AOT) recompilation and runtime port of *Kid Icarus: Uprising* (Nintendo 3DS, USA `0004000000030100` / `CTR-P-AKDE`) for the Nintendo Switch (Horizon OS / `aarch64-nintendo-switch-freestanding`).

`uprising_nx.nro` ships as a standalone, legally distributable homebrew binary containing **zero Nintendo game code or assets**. On first launch on your Switch, UprisingNX validates your decrypted US *Kid Icarus: Uprising* ROM, compiles all `12.14 MB` of ARMv6K + VFPv2 game code directly to native AArch64 machine code across the Switch's CPU cores, and saves the compiled cache to your SD card for fast launches from then on.

---

## Quick Start (Switch Setup)

1. **Extract to SD Root**: Extract the release `.zip` to the root of your Switch SD card so you have:
   - `sdmc:/switch/uprisingnx/uprising_nx.nro`
   - `sdmc:/switch/uprisingnx/rom/`
2. **Add Your Decrypted US ROM**: Copy your legally dumped, **decrypted** US *Kid Icarus: Uprising* 1.0 (`0004000000030100`, `CTR-P-AKDE`) ROM — `.cia`, `.3ds`, or `.cci` (for example `baserom.cia`) — into:
   ```text
   sdmc:/switch/uprisingnx/rom/
   ```
   *(Legacy `sdmc:/switch/uprisingnx/roms/` paths are also detected automatically. Encrypted dumps must be decrypted first via GodMode9 on your 3DS.)*
3. **Launch via Title Override**: Hold **R** while launching any installed Switch game from the HOME Menu to open Homebrew Menu in **Title Override** (full-RAM + JIT) mode, then start **UprisingNX**:
   - **First launch**: Compiles the ROM directly to ~87 MiB of native AArch64 machine code on the Switch (~30–60s with an on-screen progress bar) and writes `sdmc:/switch/uprisingnx/cache/aot_v1_0004000000030100_D6E028C6175A2B07.bin`.
   - **Every launch after**: Streams the cached AArch64 image straight into a single Horizon `CodeMemory` region without recompiling.

---

## Controls & Features

- **Dual-Stick & Gyro Aiming**: Native camera and reticle control using both analog sticks and built-in gyroscope support (`nx/rust/src/aim.rs`).
- **Z-Targeting & Behind-Pit Camera Recenter (`R` / `L3`)**: Tap or hold **R** (or **Left Stick Click**) on foot for Zelda-style lock-on: acquires any nearby enemy within `16` units (360°) or in front of you (`±85°` forward hemisphere up to `52` units, prioritizing your crosshair), keeps a tight 1:1 camera lock while held (continuously re-acquiring if an enemy enters range or dies, and surviving brief hit-stun flicker frames), or cleanly swings behind Pit when no enemy is nearby. Moving the right stick after a tap breaks lock naturally.
- **Unified Widescreen Single-Screen UI**: Custom 16:9 layouts across the Title Screen, Mode Select, Solo Hub (Palutena's & Viridi's Domains), Chapter Select, Fiend's Cauldron, Arms Altar, Gear, Weapon Fusion, and top-screen battle HUD & dialogue portraits, with automatic touch-to-stick and button mapping.
- **Baked-In Shader Warmup**: Ships with **493 portable *Kid Icarus: Uprising* Vulkan pipeline keys** baked directly into `uprising_nx.nro`. On startup, UprisingNX warms up Vulkan graphics pipelines in the background (caching compiled pipelines on your SD card) to minimize shader compilation stutters during early chapters and battles.
- **In-App Auto-Updater**: Automatically performs an offline-safe (`nifm`) check against GitHub Releases when you reach the Title Screen—silently skipping when offline—and displays a yellow **`Update Available!`** badge above the version watermark whenever a newer release is out. Open **Settings → Save & System → Software update** to check on demand, download the verified `uprising_nx.nro` in the background while you play, and atomically swap & relaunch into the new version.
- **Tabbed Settings Menu (`Minus` / `L` & `R`)**: Press `Minus` (or right stick click + `Minus`) to pause the game and open the settings menu. Use **L / R** (or **ZL / ZR**) to switch between **Video**, **Controls**, and **Save & System** tabs; D-pad Up/Down picks a row (its description shows underneath), Left/Right (or A) changes it, and B or `Minus` closes it.
  - **Video**: Per-mode Handheld/Docked preset (`Performance`, `Quality`, `Custom`), battle & menu resolution (`1x`–`4x`), smooth mode (60 fps), dynamic resolution, and power label placement.
  - **Controls**: Right stick aim, gyro aim & speed, stick aim speed, invert aim Y, left stick aim mode & modifier button, and `Minus` button behavior.
  - **Save & System**: In-app **Software update** (check, download, and relaunch), import decrypted 3DS saves from `sdmc:/switch/uprisingnx/save/import/` (with automatic backup of any existing Switch save), export your Switch save to `sdmc:/switch/uprisingnx/save/export/`, resume, or quit (press A twice).
- **3DS Save Import & Export**: Decrypted US *Kid Icarus: Uprising* save files (`00.SAV`, `01.SAV`, etc.) from Checkpoint, JKSM, Citra, or Azahar are directly compatible:
  - **Drop-in auto-import**: Copy your `.SAV` files (or Checkpoint backup folder) into `sdmc:/switch/uprisingnx/save/import/` before launching UprisingNX. On boot, UprisingNX backs up any existing Switch save to `save/backup_<timestamp>/`, imports your saves with the required `.format` header, and archives the consumed files to `save/imported/`.
  - **In-menu import/export**: Open **Settings → Save & System** to import saves or export a clean backup to `sdmc:/switch/uprisingnx/save/export/` anytime.
- **Hotkeys**:
  - **Right stick click**: Shows or hides the bottom screen (on release). `Plus` is the 3DS's Start; `Minus` opens the settings menu (or acts as Select when configured in Controls).
  - **Hold Right stick click + D-pad Left**: Cycles the active Handheld or Docked preset (`Performance` → `Quality` → `Custom`).
  - **Hold Right stick click + D-pad Up / Down**: Raises or lowers the internal resolution (`1x`–`4x`) for battles or menus, switching the preset to `Custom`.
  - **Hold Right stick click + D-pad Right**: Toggles Smooth Mode (frame interpolation, up to 60 fps from the game's 30 fps); automatically backs off briefly whenever the GPU can't keep up with double-rate rendering.
  - **`Plus + Minus`**: Quits immediately.

---

## Settings

`sdmc:/switch/uprisingnx/settings.ini` is created with defaults on first launch and updated automatically by the Settings menu and hotkeys. You can also edit it while the game is closed.

| Preset | Handheld | Docked |
| --- | --- | --- |
| `performance` (default) | battles 1x, menus 2x, smooth on (≈60 fps) | battles 3x, menus 4x, smooth on |
| `quality` | 2x everywhere, smooth off (locked 30 fps) | 4x everywhere, smooth off |
| `custom` | `battle_scale`, `menu_scale`, `smooth` from the file | same |

`[controls]` stores aiming preferences (`right_stick_aim`, `gyro_aim`, `gyro_speed`, `left_stick_aim`, `aim_modifier`, `aim_invert_y`, `aim_speed`).

---

## Known Limitations (v0.1)

- **Title Override Required**: UprisingNX requires full memory and JIT permissions—**Album / Applet mode is not supported**. Always launch Homebrew Menu by holding **R** while starting an installed Switch title.
- **Decrypted USA 1.0 ROM Only**: Currently targets the decrypted USA 1.0 release (`0004000000030100` / `CTR-P-AKDE`). Encrypted dumps, EU/JP regions, or other revisions are not supported yet.
- **First-Launch AOT Compilation**: The very first launch compiles the game's ARMv6K code to native AArch64 on the Switch itself, which takes ~30–60 seconds. Subsequent launches reuse the cached binary on your SD card.
- **Offline / Single-Player Only**: Local solo play only—online multiplayer (Together / Nearby) and StreetPass are not implemented.
- **Early Shader Stutters**: While 493 common KIU pipelines are warmed up in the background on launch, first-playthrough encounters with new effects or scenes before warmup finishes may occasionally hitch once while their Vulkan pipeline is compiled and cached.

---

## SD Card Layout

- **NRO Executable**: `sdmc:/switch/uprisingnx/uprising_nx.nro`
- **Decrypted US ROM**: `sdmc:/switch/uprisingnx/rom/baserom.cia` (any `.cia`, `.3ds`, or `.cci` matching US Title ID `0004000000030100`; `sdmc:/switch/uprisingnx/roms/` is also scanned)
- **Generated AOT Cache**: `sdmc:/switch/uprisingnx/cache/aot_v1_0004000000030100_D6E028C6175A2B07.bin` (created automatically on first launch)
- **Save Data (Live)**: `sdmc:/switch/uprisingnx/data/user/savedata/0004000000030100/`
- **3DS Save Import / Export**: `sdmc:/switch/uprisingnx/save/import/` and `sdmc:/switch/uprisingnx/save/export/`
- **Optional Switch Tutorial Videos (RomFS Mod)**: `sdmc:/switch/uprisingnx/data/mods/0004000000030100/romfs/us/movie/`

---

## Architecture

- **On-Switch Direct ARMv6K → AArch64 AOT Compiler (`zakuro/crates/zakuro-aot/`)**: Multi-core position-independent ARMv6K/Thumb/VFPv2 → AArch64 compiler with 1.0× single-owner instruction deduplication (`86.9 MiB` total native code, fitting inside a single `±128 MiB` Horizon `CodeMemory` slot while leaving 9 slots for shader JIT), `.text` literal-pool constant folding, hardware ASIMD MobiClip media intrinsics, and built-in Rust gameplay hooks (`kiu_overrides.rs`).
- **Horizon Runtime & PICA200 → Vulkan Renderer (`zakuro/`)**: Horizon OS port of the `zakuro` 3DS kernel, HLE services, and GPU renderer, translating PICA200 commands and geometry/vertex shaders to Vulkan (`VK_NN_vi_surface`) via `mesa-switch` (Nouveau / Tegra X1 Vulkan driver).
- **Switch Frontend (`nx/`)**: Rust staticlib (`nx/rust/`) + `libnx` C shell (`nx/c/`) providing unified widescreen single-screen UI layouts, dual-stick + gyro aiming (`nx/rust/src/aim.rs`), in-app GitHub Releases auto-updater (`nx/rust/src/updater.rs`), 3DS save import/export (`nx/rust/src/saves.rs`), and `audren` audio output.

---

## Building

1. **Rust Runtime, On-Switch AOT Compiler & Frontend (`nx/rust`)**:
   ```bash
   cd nx/rust && RUST_MIN_STACK=16777216 ../../scripts/cargo-nx.sh build --release
   ```
   *(For tagged releases, pass `UNX_RELEASE_TAG=v0.1.0` so the watermark and auto-updater match the release tag cleanly.)*
2. **C Shell & NRO Packaging (`nx/c`)**:
   Make sure `devkitA64` and `libnx` are configured in your environment (`DEVKITPRO`).
   ```bash
   make -C nx/c KICKSTAND=0
   ```

---

## Credits & License

- **License**: UprisingNX is licensed under the [MIT License](LICENSE). Built on top of **Zakuro** (`zakuro/LICENSE`, MIT License, Copyright © 2026 fearkov).
- **3D Nintendo Switch Tutorial Assets** (licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); modified for UprisingNX with articulated button/stick sub-meshes, dry-matte PBR materials, and animated control choreographies, see `tools/models/switch/CREDITS.md`):
  - *"Nintendo Switch"* console body & screen mesh by **gabrieljuan0322** ([Sketchfab](https://sketchfab.com/3d-models/nintendo-switch-19f3d13edffd4e85acfbbe8adefb72fb))
  - *"Joy-con Switch"* sculpted Neon-Blue & Neon-Red Joy-Con controllers, analog sticks, triggers, and buttons by **Wevensson** ([Sketchfab](https://sketchfab.com/3d-models/joy-con-switch-f8cf0fb92ac544d38079e8d294a5330a))
- **MobiClip `.moflex` Video Encoding**: [`quatric/mobipeg`](https://github.com/quatric/mobipeg) + [`quatric/x264`](https://github.com/quatric/x264) open-source stereoscopic MobiClip encoder
