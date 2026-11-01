#!/usr/bin/env bash
# Desktop (Linux x86-64) build of kiu-pc: the reference to measure against.
# Mirrors kiu-pc's scripts/setup.ps1: 3dsrecomp at the pinned commit, kiu-pc's
# overrides moved to USA addresses, `3dsrecomp port`, then cargo against kiu-pc's
# vendored Zakuro. Output in build/desktop/ (gitignored; it holds game code).
#   scripts/desktop-port.sh <rom.cci>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROM="$(realpath "$1")"
KIU="$ROOT/third_party/Clefox9820_kiu-pc"
PIN=c3f21f94ba50d688025febc8974ff348d471b587
OUT="$ROOT/build/desktop"
mkdir -p "$OUT"

# 3dsrecomp at kiu-pc's pin, in a worktree of the clone
RECOMP="$OUT/3dsrecomp-pin"
if [ ! -d "$RECOMP" ]; then
  git -C "$ROOT/third_party/fearkov_3dsrecomp" worktree add --detach "$RECOMP" "$PIN"
fi
(cd "$RECOMP" && cargo build --release)

# overrides with USA addresses (table from kiu-pc setup.ps1)
declare -A USA=(
  [0x006EA18C]=0x006EA040 [0x00A3F050]=0x00A3EDCC [0x0074643C]=0x007462A4
  [0x006D27E8]=0x006D269C [0x006D2C08]=0x006D2ABC [0x00A426AC]=0x00A42428
  [0x0070419C]=0x00704004 [0x00606534]=0x006063E8
)
OVR="$OUT/overrides-usa"
rm -rf "$OVR" && mkdir -p "$OVR"
for f in "$KIU"/overrides/*.c; do
  s="$(cat "$f")"
  for eur in "${!USA[@]}"; do s="${s//$eur/${USA[$eur]}}"; done
  printf '%s' "$s" > "$OVR/$(basename "$f")"
done

PORT="$OUT/port-usa"
if [ ! -d "$PORT/code" ]; then
  time "$RECOMP/target/release/3dsrecomp" port "$ROM" "$PORT" --name kid-icarus \
    --overrides "$OVR" --zakuro "$KIU/zakuro"
fi
mkdir -p "$PORT/.cargo"
printf '[build]\nrustflags = ["-C", "target-cpu=native"]\n' > "$PORT/.cargo/config.toml"
# the generated project's own Cargo.lock would take 3dsrecomp's HEAD; Zakuro in
# kiu-pc speaks the ABI of the pin
(cd "$PORT" && cargo update -p recomp-abi --precise "$PIN")
(cd "$PORT" && time cargo build --release)
ls -la "$PORT/target/release/kid-icarus"
