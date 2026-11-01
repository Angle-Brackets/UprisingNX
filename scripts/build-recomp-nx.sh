#!/usr/bin/env bash
# Cross-compiles 3dsrecomp's generated C (build/desktop/port-usa/code) for
# aarch64 Horizon into build/nx-recomp/librecomp.a, for nx/c to link with
# RECOMP=1. Same flags as scripts/a1-codesize.sh. Incremental by content: each
# object keeps a key (sha1 of its source plus recomp.h and overrides.h) and is
# reused while that matches, so a regenerated port only recompiles the shards
# that changed. functions.h is left out of the key: it only declares
# f_XXXXXXXX functions, all with one signature, and a stale reference to one
# that went away fails at link time.
#   scripts/build-recomp-nx.sh [opt=-Os] [jobs=11] [code dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPT="${1:--Os}"; JOBS="${2:-11}"
CODE="$(realpath "${3:-$ROOT/build/desktop/port-usa/code}")"
OUT="$ROOT/build/nx-recomp"
OBJ="$OUT/obj${OPT}"
CC=/opt/devkitpro/devkitA64/bin/aarch64-none-elf-gcc
AR=/opt/devkitpro/devkitA64/bin/aarch64-none-elf-ar
mkdir -p "$OBJ"
export CC CODE OBJ FLAGS_STR="-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE $OPT -ffp-contract=off -fno-math-errno -w -ffunction-sections -fdata-sections --param ggc-min-expand=20 --param ggc-min-heapsize=65536 -I$CODE"

compile() {
  local src="$1" o="$OBJ/$(echo "$1" | tr / _).o"
  local key; key="$(cat "$CODE/$src" | sha1sum | cut -c1-40) $HEADERS $FLAGS_SUM"
  [ -f "$o" ] && [ "$(cat "$o.key" 2>/dev/null)" = "$key" ] && return 0
  $CC $FLAGS_STR -c "$CODE/$src" -o "$o.tmp" && mv "$o.tmp" "$o" && echo "$key" > "$o.key" && echo "$src"
}
export -f compile
export HEADERS="$(cat "$CODE/recomp.h" "$CODE/overrides.h" | sha1sum | cut -c1-40)"
export FLAGS_SUM="$(echo "$FLAGS_STR" | sed "s|$CODE||" | sha1sum | cut -c1-8)"
echo "code: $CODE" >&2

# biggest first, so the long poles start early
(cd "$CODE" && ls -S code*.c entries.c hand/*.c) | xargs -P "$JOBS" -I{} bash -c 'compile {}' \
  | awk '{ n++; if (n % 50 == 0) printf "%d compiled\n", n; fflush() }'
rm -f "$OUT/librecomp.a"
# only this port's objects, not those of shards an earlier port had
mapfile -t OBJS < <(cd "$CODE" && ls code*.c entries.c hand/*.c | tr / _ | sed "s|^|$OBJ/|; s|$|.o|")
$AR rcs "$OUT/librecomp.a" "${OBJS[@]}"
ls -la "$OUT/librecomp.a"
size -t "${OBJS[@]}" | tail -1
