#!/usr/bin/env bash
# Plan A1: how big and how slow is the generated C for aarch64 Horizon?
# Compiles every Nth shard of build/desktop/port-usa/code with devkitA64 gcc,
# with 3dsrecomp's flags plus devkitA64's switch_rules arch/PIE flags, and
# prints per-shard time, aarch64 .text and the x86 .text of the same shard.
#   scripts/nx/a1-codesize.sh [opt=-O2] [step=40] [jobs=10]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OPT="${1:--O2}"; STEP="${2:-40}"; JOBS="${3:-10}"
CODE="$ROOT/build/desktop/port-usa/code"
OUT="$ROOT/build/a1/${OPT#-}"
CC=/opt/devkitpro/devkitA64/bin/aarch64-none-elf-gcc
mkdir -p "$OUT"
FLAGS=(-march=armv8-a+crc+crypto -mtune=cortex-a57 -mtp=soft -fPIE
       "$OPT" -ffp-contract=off -fno-math-errno -w
       -ffunction-sections -fdata-sections
       --param ggc-min-expand=20 --param ggc-min-heapsize=65536 -I"$CODE")

mapfile -t SHARDS < <(ls "$CODE" | grep -E '^code[0-9]+\.c$' | sort | awk -v s="$STEP" 'NR % s == 1')
echo "opt $OPT, ${#SHARDS[@]} shards (every ${STEP}th), $JOBS jobs" >&2

compile() {
  local c="$1" o="$OUT/${1%.c}.o"
  local t0=$(date +%s.%N)
  "$CC" "${FLAGS[@]}" -c "$CODE/$c" -o "$o"
  local t1=$(date +%s.%N)
  local a64=$(size "$o" | awk 'NR==2{print $1}')
  local x86=$(size "$CODE/${c%.c}.o" | awk 'NR==2{print $1}')
  printf '%s %s %.1f %s %s\n' "$c" "$(stat -c %s "$CODE/$c")" "$(echo "$t1 - $t0" | bc)" "$a64" "$x86"
}
export -f compile; export CC CODE OUT; export FLAGS_STR="${FLAGS[*]}"
# bash can't export arrays; rebuild inside the subshell
printf '%s\n' "${SHARDS[@]}" | xargs -P "$JOBS" -I{} bash -c 'FLAGS=($FLAGS_STR); compile {}' \
  | tee "$OUT/results.txt"

awk -v total="$(ls "$CODE" | grep -cE '^code[0-9]+\.c$')" '
  { c+=$2; t+=$3; a+=$4; x+=$5; n++ }
  END {
    printf "\n%d shards: C %.1f MB, a64 text %.1f MB, x86 text %.1f MB, a64/x86 %.3f\n", n, c/1e6, a/1e6, x/1e6, a/x
    printf "cpu time %.0f s (%.1f s/shard)\n", t, t/n
    printf "extrapolated to %d shards: a64 text %.0f MB, cpu time %.1f h (%.1f h wall on 10 jobs)\n", total, a/n*total/1e6, t/n*total/3600, t/n*total/3600/10
  }' "$OUT/results.txt"
