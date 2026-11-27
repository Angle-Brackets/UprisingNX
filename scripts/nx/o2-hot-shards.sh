#!/usr/bin/env bash
# Recompiles the hottest recompiled-game-code shards at -O2 into build/nx-recomp/obj-Os, over
# the -Os objects (their .key files stay, so build-recomp-nx.sh keeps them). The list is the 25
# shards holding ~60% of the game thread's game-code samples (alley, flight, attract demo), found
# with scripts/nx/prof.py symbols mapped to code NNN.c. Run after build-recomp-nx.sh.
#   scripts/nx/o2-hot-shards.sh [code dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="$(realpath "${1:-$ROOT/build/desktop/port-usa-ovr/code}")"
OBJ="$ROOT/build/nx-recomp/obj-Os"
FLAGS="-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE -O2 -ffp-contract=off -fno-math-errno -w -ffunction-sections -fdata-sections --param ggc-min-expand=20 --param ggc-min-heapsize=65536 -I$CODE"
export CODE OBJ FLAGS
printf 'code%s.c\n' 624 626 625 631 630 750 751 441 639 775 774 641 395 432 635 542 260 050 396 265 126 064 643 632 471 |
  xargs -P 11 -I{} bash -c 'aarch64-none-elf-gcc $FLAGS -c "$CODE/{}" -o "$OBJ/{}.o.tmp" && mv "$OBJ/{}.o.tmp" "$OBJ/{}.o"'
echo "recompiled 25 shards at -O2; now: scripts/build-recomp-nx.sh -Os 11 (reuses them) and make RECOMP=1"
