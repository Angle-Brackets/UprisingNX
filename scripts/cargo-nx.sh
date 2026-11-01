#!/usr/bin/env bash
# cargo for the Switch: points build-std at the patched std sources from
# scripts/horizon-std-setup.sh. Run it in a crate whose .cargo/config.toml sets
# the target and build-std flags (see probes/std-probe/rust/.cargo/config.toml).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -d "$ROOT/build/rust-src/library" ] || "$ROOT/scripts/horizon-std-setup.sh"
export __CARGO_TESTS_ONLY_SRC_ROOT="$ROOT/build/rust-src/library"
exec cargo "$@"
