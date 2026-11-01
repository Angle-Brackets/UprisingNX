#!/usr/bin/env bash
# Sets up Rust std for the Switch (aarch64 Horizon, libnx) in build/:
#   build/vendor/libc        libc 0.2.191 + patches/libc-0.2.191-horizon-aarch64.patch
#   build/rust-src/library   the nightly's std sources, with libc patched to the above
# Build with: __CARGO_TESTS_ONLY_SRC_ROOT=<repo>/build/rust-src/library cargo build ...
# (scripts/cargo-nx.sh does that). build-std ignores a crate's own [patch], so
# std's copy of libc has to be patched in its own workspace.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN="${TOOLCHAIN:-nightly-2026-10-20}"
LIBC_VER=0.2.191

SYSROOT="$(rustc +"$TOOLCHAIN" --print sysroot)"
LIB="$SYSROOT/lib/rustlib/src/rust/library"
[ -d "$LIB" ] || { echo "rust-src missing: rustup component add rust-src --toolchain $TOOLCHAIN" >&2; exit 1; }
grep -q "name = \"libc\"" "$LIB/Cargo.lock"
grep -A1 'name = "libc"' "$LIB/Cargo.lock" | grep -q "version = \"$LIBC_VER\"" \
  || { echo "std on $TOOLCHAIN doesn't use libc $LIBC_VER; regenerate the patch" >&2; exit 1; }

# libc source: from the cargo registry, fetched by any build that needs it
REG="$(ls -d "$HOME"/.cargo/registry/src/*/libc-$LIBC_VER 2>/dev/null | head -1)"
if [ -z "$REG" ]; then
  (cd "$ROOT/probes/std-probe/rust" && cargo +"$TOOLCHAIN" fetch >/dev/null)
  REG="$(ls -d "$HOME"/.cargo/registry/src/*/libc-$LIBC_VER | head -1)"
fi

mkdir -p "$ROOT/build/vendor" "$ROOT/build/rust-src"
rm -rf "$ROOT/build/vendor/libc"
cp -r "$REG" "$ROOT/build/vendor/libc"
(cd "$ROOT/build/vendor" && patch -s -p1 -d libc < "$ROOT/patches/libc-$LIBC_VER-horizon-aarch64.patch")

rm -rf "$ROOT/build/rust-src/library"
cp -r "$LIB" "$ROOT/build/rust-src/library"
sed -i "/^\[patch.crates-io\]/a libc = { path = '$ROOT/build/vendor/libc' }" "$ROOT/build/rust-src/library/Cargo.toml"
# std/build.rs marks std `restricted_std` unless env=newlib, which the
# freestanding target spec doesn't set (and cargo's own CARGO_CFG_TARGET_ENV
# wins over .cargo/config [env]). Accept horizon outright.
sed -i 's/|| (target_vendor == "nintendo" \&\& target_env == "newlib")/|| (target_vendor == "nintendo" \&\& (target_env == "newlib" || target_os == "horizon"))/' \
  "$ROOT/build/rust-src/library/std/build.rs"
grep -q 'target_os == "horizon"))' "$ROOT/build/rust-src/library/std/build.rs"
echo "ok: std sources in build/rust-src/library use build/vendor/libc"
