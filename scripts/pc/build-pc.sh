#!/usr/bin/env bash
# Builds the standalone Kid Icarus: Uprising Recompiled PC binary, writes a
# stripped copy under build/pc/, and reports raw vs stripped binary sizes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

cargo build --release -p uprising-recompiled "$@"

EXE=""
if [[ -f "$ROOT/target/release/uprising-recompiled.exe" ]]; then
  EXE=".exe"
fi

RAW_BIN="$ROOT/target/release/uprising-recompiled${EXE}"
OUT_DIR="${OUT_DIR:-$ROOT/build/pc}"
STRIPPED_BIN="$OUT_DIR/uprising-recompiled${EXE}"

mkdir -p "$OUT_DIR"
cp -f "$RAW_BIN" "$STRIPPED_BIN"

STRIP_BIN="${STRIP:-}"
if [[ -z "$STRIP_BIN" ]]; then
  if command -v strip >/dev/null 2>&1; then
    STRIP_BIN="strip"
  elif command -v llvm-strip >/dev/null 2>&1; then
    STRIP_BIN="llvm-strip"
  fi
fi

if [[ -n "$STRIP_BIN" ]]; then
  "$STRIP_BIN" "$STRIPPED_BIN"
else
  echo "warning: neither strip nor llvm-strip found; leaving copy unstripped" >&2
fi

size_bytes() {
  if stat -c%s "$1" >/dev/null 2>&1; then
    stat -c%s "$1"
  else
    stat -f%z "$1"
  fi
}

RAW_BYTES="$(size_bytes "$RAW_BIN")"
STRIPPED_BYTES="$(size_bytes "$STRIPPED_BIN")"
awk -v raw="$RAW_BYTES" -v stripped="$STRIPPED_BYTES" -v out="$STRIPPED_BIN" '
BEGIN {
  printf "Built %s\n  raw:      %d bytes (%.2f MiB)\n  stripped: %d bytes (%.2f MiB)\n",
    out, raw, raw / 1048576.0, stripped, stripped / 1048576.0;
}'
