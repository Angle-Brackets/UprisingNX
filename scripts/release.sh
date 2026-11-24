#!/usr/bin/env bash
# Cut a clean UprisingNX Switch release with strict VERSION bump enforcement.
#
# Usage:
#   scripts/release.sh [--check-only | --build-only] [--title "..."] [--notes-file path]
#
# Enforces before building or publishing:
#   1. Clean tracked working tree on `main`, matching `origin/main`
#   2. Strict MAJOR.MINOR.PATCH in `VERSION`, matching `nx/rust/Cargo.toml`
#   3. Tag `v<VERSION>` does not already exist locally or on `origin`
#   4. `VERSION` is strictly greater than the latest existing `vX.Y.Z` release tag
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

MODE="publish"
TITLE=""
NOTES_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) MODE="check"; shift ;;
    --build-only) MODE="build"; shift ;;
    --title) TITLE="${2:?missing --title value}"; shift 2 ;;
    --notes-file) NOTES_FILE="${2:?missing --notes-file value}"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# 1. Verify git state & VERSION bump
git fetch --tags origin >/dev/null 2>&1 || true

VER="$(python3 - <<'PY'
import re, subprocess, sys
from pathlib import Path

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
CLEAN_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

dirty = git("status", "--porcelain", "--untracked-files=no")
if dirty:
    sys.exit(f"ERROR: Working tree has uncommitted tracked changes:\n{dirty}\nCommit and push before releasing.")

branch = git("rev-parse", "--abbrev-ref", "HEAD")
if branch != "main":
    sys.exit(f"ERROR: Must release from 'main' (currently on '{branch}').")

head = git("rev-parse", "HEAD")
try:
    origin_main = git("rev-parse", "origin/main")
    if head != origin_main:
        sys.exit(f"ERROR: HEAD ({head[:7]}) != origin/main ({origin_main[:7]}). Push main first.")
except subprocess.CalledProcessError:
    sys.exit("ERROR: origin/main ref not found.")

ver_raw = Path("VERSION").read_text(encoding="utf-8").strip()
m = SEMVER_RE.match(ver_raw)
if not m:
    sys.exit(f"ERROR: VERSION must be strict MAJOR.MINOR.PATCH semver (got {ver_raw!r}).")
ver = tuple(int(g) for g in m.groups())

cargo_toml = Path("nx/rust/Cargo.toml").read_text(encoding="utf-8")
m_cargo = re.search(r'(?m)^version\s*=\s*"([^"]+)"', cargo_toml)
if not m_cargo or m_cargo.group(1) != ver_raw:
    found = m_cargo.group(1) if m_cargo else "<missing>"
    sys.exit(f"ERROR: nx/rust/Cargo.toml version ({found}) != VERSION ({ver_raw}). Update both files.")

tag = f"v{ver_raw}"
local_tags = [t for t in git("tag", "-l", "v*").splitlines() if t.strip()]
remote_tags = []
for line in git("ls-remote", "--tags", "origin").splitlines():
    parts = line.split()
    if len(parts) == 2 and parts[1].startswith("refs/tags/") and not parts[1].endswith("^{}"):
        remote_tags.append(parts[1][len("refs/tags/"):])

all_tags = sorted(set(local_tags) | set(remote_tags))
if tag in all_tags:
    sys.exit(f"ERROR: Release tag {tag} already exists. Bump VERSION (and nx/rust/Cargo.toml) before releasing.")

clean_tags = []
for t in all_tags:
    mc = CLEAN_TAG_RE.match(t)
    if mc:
        clean_tags.append((tuple(int(g) for g in mc.groups()), t))
clean_tags.sort()

if clean_tags:
    prev_ver, prev_tag = clean_tags[-1]
    if ver <= prev_ver:
        sys.exit(
            f"ERROR: VERSION ({ver_raw}) must be strictly greater than latest release {prev_tag}. "
            "Update VERSION before releasing."
        )

print(ver_raw)
PY
)"

TAG="v${VER}"
echo "[release] Verified VERSION=${VER} (tag ${TAG}) on clean main ($(git rev-parse --short HEAD))."

if [[ "$MODE" == "check" ]]; then
  exit 0
fi

# 2. Build clean Switch NRO (KICKSTAND=0, UNX_RELEASE_TAG=vX.Y.Z)
export DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
export DEVKITA64="${DEVKITA64:-${DEVKITPRO}/devkitA64}"
export PATH="${DEVKITA64}/bin:${DEVKITPRO}/tools/bin:${PATH}"
export RUST_MIN_STACK=16777216
export UNX_RELEASE_TAG="${TAG}"

echo "[release] Building Rust staticlib (UNX_RELEASE_TAG=${TAG})..."
touch nx/rust/build.rs
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
REMAP_FLAGS=(
  "-C" "target-cpu=cortex-a57"
  "-C" "force-frame-pointers=yes"
  "--cfg" 'target_family="unix"'
  "--cfg" 'target_env="newlib"'
  "--cfg" "unix"
  "-A" "explicit_builtin_cfgs_in_flags"
  "-Z" "has-thread-local=true"
  "--remap-path-prefix=${REPO_ROOT}=."
  "--remap-path-prefix=${CARGO_HOME_DIR}=/cargo"
  "--remap-path-prefix=${HOME}=/builder"
)
export CARGO_ENCODED_RUSTFLAGS="$(IFS=$'\x1f'; echo "${REMAP_FLAGS[*]}")"
(cd nx/rust && ../../scripts/cargo-nx.sh build --release)

echo "[release] Linking Switch NRO (KICKSTAND=0)..."
make -C nx/c clean
make -C nx/c -j"$(nproc)" KICKSTAND=0 UNX_RELEASE_TAG="${TAG}" \
  EXTRA_CFLAGS="-ffile-prefix-map=${REPO_ROOT}=."

NRO="nx/c/uprising_nx.nro"
python3 - "$NRO" "$TAG" "$HOME" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

nro = Path(sys.argv[1]).read_bytes()
tag = sys.argv[2].encode("ascii")
home = sys.argv[3].encode("utf-8")
repo = sys.argv[4].encode("utf-8")
assert len(nro) >= 1_048_576, f"NRO unexpectedly small: {len(nro)} bytes"
assert nro[0x10:0x14] == b"NRO0", f"Invalid NRO0 magic at 0x10: {nro[0x10:0x14]!r}"
assert tag in nro, f"Release tag {tag!r} not found in built NRO"
assert home not in nro, f"Host home path {home!r} leaked into built NRO!"
assert repo not in nro, f"Host repo path {repo!r} leaked into built NRO!"
print(f"[release] Verified {sys.argv[1]}: {len(nro):,} bytes, NRO0 magic OK, embedded {tag.decode()} OK, zero host paths leaked")
PY

# 3. Package release ZIP + standalone NRO
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/switch/uprisingnx/rom" \
         "$STAGE/switch/uprisingnx/save/import" \
         "$STAGE/switch/uprisingnx/save/export"

cp "$NRO" "$STAGE/switch/uprisingnx/uprising_nx.nro"

cat > "$STAGE/switch/uprisingnx/rom/PLACE_DECRYPTED_US_ROM_HERE.txt" <<'EOF'
Place your decrypted USA copy of Kid Icarus: Uprising (.cia, .3ds, or .cci,
Title ID 0004000000030100) in this folder. Any filename works (for example:
baserom.cia or baserom.3ds).

Launch UprisingNX through Title Override (hold R while opening any installed
Switch game -> Homebrew Menu -> UprisingNX).
EOF

cat > "$STAGE/switch/uprisingnx/save/import/PLACE_3DS_SAVES_HERE.txt" <<'EOF'
Drop your 3DS Kid Icarus: Uprising save files (01.SAV / 02.SAV / 03.SAV, or a
Checkpoint / JKSM / Citra / Azahar save folder) here. UprisingNX will
automatically back up your existing save and import them on the next launch
(or via Settings -> Save & System -> Import 3DS saves).
EOF

DIST_DIR="$REPO_ROOT/build/release/${TAG}"
mkdir -p "$DIST_DIR"
ZIP_PATH="${DIST_DIR}/UprisingNX-${TAG}.zip"
NRO_OUT="${DIST_DIR}/uprising_nx.nro"
rm -f "$ZIP_PATH" "$NRO_OUT"
cp "$NRO" "$NRO_OUT"
(cd "$STAGE" && zip -qr "$ZIP_PATH" switch)

echo "[release] Packaged:"
echo "  - $ZIP_PATH ($(stat -c%s "$ZIP_PATH") bytes)"
echo "  - $NRO_OUT ($(stat -c%s "$NRO_OUT") bytes)"

if [[ "$MODE" == "build" ]]; then
  exit 0
fi

# 4. Tag and publish GitHub Release
TITLE="${TITLE:-UprisingNX ${TAG}}"
git tag -a "$TAG" -m "UprisingNX ${TAG}"
git push origin "$TAG"

GH_ARGS=(release create "$TAG" "$ZIP_PATH" "$NRO_OUT" --title "$TITLE")
if [[ -n "$NOTES_FILE" ]]; then
  GH_ARGS+=(--notes-file "$NOTES_FILE")
else
  GH_ARGS+=(--generate-notes)
fi

gh "${GH_ARGS[@]}"
echo "[release] Published GitHub Release ${TAG}!"
