#!/usr/bin/env bash
# Cross-compiles 3dsrecomp's generated C (build/desktop/port-usa/code) for
# aarch64 Horizon into build/nx-recomp/librecomp.a, for nx/c to link with
# RECOMP=1. Same flags as scripts/a1-codesize.sh. Incremental: a shard whose
# object is newer than its source is skipped.
#   scripts/build-recomp-nx.sh [opt=-Os] [jobs=11]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPT="${1:--Os}"; JOBS="${2:-11}"
CODE="$ROOT/build/desktop/port-usa/code"
OUT="$ROOT/build/nx-recomp"
OBJ="$OUT/obj${OPT}"
CC=/opt/devkitpro/devkitA64/bin/aarch64-none-elf-gcc
AR=/opt/devkitpro/devkitA64/bin/aarch64-none-elf-ar
mkdir -p "$OBJ"
export CC CODE OBJ FLAGS_STR="-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE $OPT -ffp-contract=off -fno-math-errno -w -ffunction-sections -fdata-sections --param ggc-min-expand=20 --param ggc-min-heapsize=65536 -I$CODE"

compile() {
  local src="$1" o="$OBJ/$(echo "$1" | tr / _).o"
  [ "$o" -nt "$CODE/$src" ] && return 0
  $CC $FLAGS_STR -c "$CODE/$src" -o "$o.tmp" && mv "$o.tmp" "$o" && echo "$src"
}
export -f compile

# biggest first, so the long poles start early
(cd "$CODE" && ls -S code*.c entries.c hand/*.c) | xargs -P "$JOBS" -I{} bash -c 'compile {}' \
  | awk '{ n++; if (n % 50 == 0) printf "%d compiled\n", n; fflush() }'
rm -f "$OUT/librecomp.a"
$AR rcs "$OUT/librecomp.a" "$OBJ"/*.o
ls -la "$OUT/librecomp.a"
size -t "$OBJ"/*.o | tail -1
