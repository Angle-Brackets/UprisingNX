#!/bin/sh
# Rebuilds zakuro-gpu's raster.frag SPIR-V (both builds) with glslc -O, as
# upstream's were made: plain glslang -V output runs far slower on NVK.
# Uses glslc from PATH, else the one in the flatpak freedesktop SDK, run
# through that SDK's own loader.
set -e
cd "$(dirname "$0")/../zakuro/crates/zakuro-gpu/shaders"
if command -v glslc >/dev/null 2>&1; then
    run() { glslc "$@"; }
else
    sdk=$(ls -d /var/lib/flatpak/runtime/org.freedesktop.Sdk/x86_64/*/*/files 2>/dev/null | tail -1)
    [ -x "$sdk/bin/glslc" ] || { echo "no glslc in PATH or a flatpak freedesktop SDK" >&2; exit 1; }
    run() { "$sdk/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2" --library-path "$sdk/lib/x86_64-linux-gnu" "$sdk/bin/glslc" "$@"; }
fi
run -O -fshader-stage=frag raster.frag -o raster.frag.spv
run -O -fshader-stage=frag -DWRITES_DEPTH raster.frag -o raster_depth.frag.spv
echo "built raster.frag.spv and raster_depth.frag.spv"
