# UprisingNX

**UprisingNX** is a static recompilation and runtime port of *Kid Icarus: Uprising* (Nintendo 3DS, USA Rev 1 `0004000000030100`) to the Nintendo Switch (Horizon OS / `aarch64-nintendo-switch-freestanding`).

---

## Architecture

- **Static AOT Recompilation (`recomp/`)**: Uses `3dsrecomp` to ahead-of-time translate the 3DS ARMv6K executable (`.code`) into C, compiled to native ARM64 (`aarch64`) for the Switch.
- **Horizon Runtime & PICA200 → Vulkan Renderer (`zakuro/`)**: Horizon OS port of the `zakuro` 3DS runtime and GPU renderer, translating PICA200 commands and shaders to Vulkan (`VK_NN_vi_surface`) via `mesa-switch` (NVK / Nouveau Tegra X1 Vulkan driver).
- **Switch Frontend (`nx/`)**: Rust staticlib (`nx/rust/`) + `libnx` C shell (`nx/c/`) providing single-screen widescreen/top-screen presentation, composited HD title screen, top-screen battle HUD & dialogue portraits, dual-stick + gyro aiming (`nx/rust/src/aim.rs`), and `audren` audio output.

---

## Controls & Features

- **Dual-Stick & Gyro Aiming**: Precise camera control using both analog sticks and built-in gyroscope support (`nx/rust/src/aim.rs`).
- **Button Remapping & Touch Mapping**: Automatic touch-to-stick and button mapping for menus and intensive battles.
- **Settings menu**: press `Minus` (or right stick click + `Minus`) to open it; the game pauses. D-pad Up/Down picks a row (its description shows underneath), Left/Right (or A) changes it, B or `Minus` closes it. The first row picks whose video settings you change, handheld or docked (it starts on the mode you're in; the other applies when you dock or undock). Changes apply at once and are saved to `settings.ini`: preset, battle and menu resolution, smooth mode, right stick / gyro aim, gyro and stick aim speed, invert Y, left stick aim mode and its aim button, and what Minus does. Minus opens the menu when let go, unless another button was pressed while it was held; the game's own Select only pauses in battles (as Plus does), and the menu's "Minus button" row gives Minus back to the game as Select (then right stick click + `Minus` opens the menu). Quit game (A twice) closes UprisingNX. The button that closes the menu never reaches the game.
- **Hotkeys**:
  - Right stick click: shows or hides the bottom screen (on release). `Plus` is the 3DS's Start; `Minus` opens the settings menu (or is Select, see above).
  - The resolution and smooth hotkeys below show what they set at the top of the screen for a moment.
  - Hold right stick click + D-pad Left: steps this mode's (handheld or docked) preset: performance → quality → custom.
  - Hold right stick click + D-pad Up / Down: raises / lowers the internal resolution (`1x` – `4x`) for battles or for menus, whichever you're in; the preset becomes custom.
  - Hold right stick click + D-pad Right: turns smooth mode (frame interpolation, up to 60 pictures/s from the game's 30) on or off; the preset becomes custom. When on, it turns itself off for a while whenever the GPU can't draw every frame twice.
  - `Plus + Minus`: quits.
  - Docked/handheld override: `dock 0|1|auto` in `cmd.txt`.

---

## Settings

`sdmc:/switch/uprisingnx/settings.ini` is written with the defaults on first start, and the menu and hotkeys above write their changes back to it. You can also edit it while the game is closed.

| Preset | Handheld | Docked |
| --- | --- | --- |
| `performance` (default) | battles 1x, menus 2x, smooth on (≈60 pictures/s) | battles 3x, menus 4x, smooth on |
| `quality` | 2x everywhere, smooth off (locked 30) | 4x everywhere, smooth off |
| `custom` | `battle_scale`, `menu_scale`, `smooth` from the file | same |

`[controls]` holds the aim settings (`right_stick_aim`, `gyro_aim`, `gyro_speed`, `left_stick_aim`, `aim_modifier`, `aim_invert_y`, `aim_speed`). `env.txt` remains for development switches; `ZAKURO_RESOLUTION` and `ZAKURO_INTERPOLATE` there override the file.

---

## SD Card Layout

Prepare your SD card with the following structure:

- **ROM Path**: `sdmc:/switch/uprisingnx/roms/Kid Icarus - Uprising (USA) (En,Fr,Es) (Rev 1).cci` (or `.3ds`)
- **NRO Path**: `sdmc:/switch/uprisingnx/uprising_nx.nro`
- **Switch Tutorial Videos (RomFS Mod)**: `sdmc:/switch/uprisingnx/data/mods/0004000000030100/romfs/us/movie/` (copy `mods/0004000000030100/` into `sdmc:/switch/uprisingnx/data/mods/` to replace the 3DS hardware diagrams in *How to Play* and Chapter 1 tutorials with 3D Nintendo Switch hardware)

---

## Building

1. **Rust Frontend (`nx/rust`)**:
   ```bash
   cd nx/rust && ../../scripts/cargo-nx.sh build --release
   ```
2. **C Shell & Linking (`nx/c`)**:
   Make sure you have devkitA64 and libnx configured in your environment.
   ```bash
   make -C nx/c RECOMP=1
   ```

---

## Credits & Third-Party Assets

- **3D Nintendo Switch Tutorial Assets** (licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); modified for UprisingNX with articulated button/stick sub-meshes, dry-matte PBR materials, and animated control choreographies, see `tools/models/switch/CREDITS.md`):
  - *"Nintendo Switch"* console body & screen mesh by **gabrieljuan0322** ([Sketchfab](https://sketchfab.com/3d-models/nintendo-switch-19f3d13edffd4e85acfbbe8adefb72fb))
  - *"Joy-con Switch"* sculpted Neon-Blue & Neon-Red Joy-Con controllers, analog sticks, triggers, and buttons by **Wevensson** ([Sketchfab](https://sketchfab.com/3d-models/joy-con-switch-f8cf0fb92ac544d38079e8d294a5330a))
- **MobiClip `.moflex` Video Encoding**: [`quatric/mobipeg`](https://github.com/quatric/mobipeg) + [`quatric/x264`](https://github.com/quatric/x264) open-source stereoscopic MobiClip encoder
