#!/usr/bin/env bash
# cargo for the Switch: points build-std at the patched std sources from
# scripts/nx/horizon-std-setup.sh. Run inside platforms/nx/rust (whose
# .cargo/config.toml sets the target and build-std flags).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ -d "$ROOT/build/rust-src/library" ] || "$ROOT/scripts/nx/horizon-std-setup.sh"
export __CARGO_TESTS_ONLY_SRC_ROOT="$ROOT/build/rust-src/library"
exec cargo "$@"
