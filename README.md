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
- **Hotkeys**:
  - `Minus`: Toggles the bottom screen view.
  - `L + R + X`: Cycles internal resolution scale (`1x` – `4x`).
  - `L + R + Y`: Toggles docked/handheld mode override.

---

## SD Card Layout

Prepare your SD card with the following structure:

- **ROM Path**: `sdmc:/switch/uprisingnx/roms/Kid Icarus - Uprising (USA) (En,Fr,Es) (Rev 1).cci` (or `.3ds`)
- **NRO Path**: `sdmc:/switch/uprisingnx/uprising_nx.nro`

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
