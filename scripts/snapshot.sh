#!/usr/bin/env bash
# Build and publish an UprisingNX snapshot (pre-release / RC) to the single
# rolling `snapshots` GitHub Pre-release and update `versions.json`.
#
# Usage:
#   scripts/snapshot.sh <snapshot-id> [--notes "Summary"] [--build-only] [--force]
#
# Example:
#   scripts/snapshot.sh v0.1.2-rc.1 --notes "Add in-app version picker"
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,10p' "$0"
  exit 0
fi

SNAP_ID="$1"
shift

NOTES=""
BUILD_ONLY=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --notes) NOTES="${2:?missing --notes value}"; shift 2 ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --force) FORCE=1; shift ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$SNAP_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$ ]]; then
  echo "ERROR: Invalid snapshot id '$SNAP_ID' (use letters, digits, '.', '-', '_')." >&2
  exit 2
fi

if [[ -z "$NOTES" ]]; then
  NOTES="$(git log -1 --pretty=%s)"
fi

if [[ "$FORCE" -eq 0 && -f versions.json ]]; then
  python3 - "$SNAP_ID" <<'PY'
import json, sys
from pathlib import Path
snap_id = sys.argv[1]
data = json.loads(Path("versions.json").read_text(encoding="utf-8"))
if any(v.get("id") == snap_id for v in data.get("versions", [])):
    sys.exit(f"ERROR: Snapshot '{snap_id}' already exists in versions.json (pass --force to overwrite).")
PY
fi

# 1. Build clean Switch NRO (KICKSTAND=0, UNX_RELEASE_TAG=<SNAP_ID>)
export DEVKITPRO="${DEVKITPRO:-/opt/devkitpro}"
export DEVKITA64="${DEVKITA64:-${DEVKITPRO}/devkitA64}"
export PATH="${DEVKITA64}/bin:${DEVKITPRO}/tools/bin:${PATH}"
export RUST_MIN_STACK=16777216
export UNX_RELEASE_TAG="${SNAP_ID}"

echo "[snapshot] Building Rust staticlib (UNX_RELEASE_TAG=${SNAP_ID})..."
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
  "--remap-path-prefix=${HOME}=/builder"
  "--remap-path-prefix=${CARGO_HOME_DIR}=/cargo"
  "--remap-path-prefix=${REPO_ROOT}=/uprisingnx"
)
export CARGO_ENCODED_RUSTFLAGS="$(IFS=$'\x1f'; echo "${REMAP_FLAGS[*]}")"
(cd nx/rust && ../../scripts/cargo-nx.sh build --release)

echo "[snapshot] Linking Switch NRO (KICKSTAND=0)..."
make -C nx/c clean
make -C nx/c -j"$(nproc)" KICKSTAND=0 UNX_RELEASE_TAG="${SNAP_ID}" \
  EXTRA_CFLAGS="-ffile-prefix-map=${REPO_ROOT}=/uprisingnx"

NRO="nx/c/uprising_nx.nro"
DIST_DIR="$REPO_ROOT/build/snapshots"
mkdir -p "$DIST_DIR"
ASSET_NAME="uprising_nx-${SNAP_ID}.nro"
ASSET_PATH="${DIST_DIR}/${ASSET_NAME}"
cp "$NRO" "$ASSET_PATH"

SHA256="$(python3 - "$ASSET_PATH" "$SNAP_ID" "$HOME" "$REPO_ROOT" <<'PY'
import hashlib, sys
from pathlib import Path

nro = Path(sys.argv[1]).read_bytes()
tag = sys.argv[2].encode("ascii")
home = Path(sys.argv[3])
repo = Path(sys.argv[4])
assert len(nro) >= 1_048_576, f"NRO unexpectedly small: {len(nro)} bytes"
assert nro[0x10:0x14] == b"NRO0", f"Invalid NRO0 magic at 0x10: {nro[0x10:0x14]!r}"
assert tag in nro, f"Snapshot tag {tag!r} not found in built NRO"
assert str(home).encode() not in nro, f"Host home path {home} leaked into built NRO!"
assert str(repo).encode() not in nro, f"Host repo path {repo} leaked into built NRO!"
assert b"/builder/" not in nro, "Unmapped /builder/ path leaked into built NRO!"
try:
    rel_parent = repo.parent.relative_to(home)
    if str(rel_parent) not in ("", "."):
        needle = f"/{rel_parent}/".encode()
        assert needle not in nro, f"Host intermediate path {needle!r} leaked into built NRO!"
except ValueError:
    pass
print(hashlib.sha256(nro).hexdigest())
PY
)"
SIZE="$(stat -c%s "$ASSET_PATH")"
COMMIT="$(git rev-parse --short HEAD)"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[snapshot] Verified ${ASSET_NAME}: ${SIZE} bytes, sha256=${SHA256:0:12}..., zero host paths leaked"

if [[ "$BUILD_ONLY" -eq 1 ]]; then
  echo "[snapshot] --build-only complete: ${ASSET_PATH}"
  exit 0
fi

# 2. Update versions.json and render snapshots release notes
NOTES_MD="$(mktemp)"
trap 'rm -f "$NOTES_MD"' EXIT

python3 - "$SNAP_ID" "$DATE_UTC" "$COMMIT" "$SIZE" "$SHA256" "$NOTES" "$NOTES_MD" <<'PY'
import json, sys
from pathlib import Path

snap_id, date_utc, commit, size_s, sha256, notes, notes_md = sys.argv[1:8]
manifest_path = Path("versions.json")
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
else:
    manifest = {"schema": 1, "latest": {"release": "", "snapshot": ""}, "versions": []}

versions = [v for v in manifest.get("versions", []) if v.get("id") != snap_id]
entry = {
    "id": snap_id,
    "type": "snapshot",
    "date": date_utc,
    "commit": commit,
    "size": int(size_s),
    "sha256": sha256,
    "nro_url": f"https://github.com/Angle-Brackets/UprisingNX/releases/download/snapshots/uprising_nx-{snap_id}.nro",
    "notes": notes,
}
versions.insert(0, entry)
manifest["versions"] = versions
manifest.setdefault("latest", {})["snapshot"] = snap_id
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

lines = [
    "Rolling archive of all UprisingNX pre-release / release-candidate snapshots (like Minecraft's snapshot history), indexed by [`versions.json`](https://github.com/Angle-Brackets/UprisingNX/blob/main/versions.json).",
    "",
    "For stable builds, see the main Releases page.",
    "",
    "### Snapshot Archive",
    "",
    "| Snapshot | Date (UTC) | Commit | SHA-256 (prefix) | Summary |",
    "| --- | --- | --- | --- | --- |",
]
for e in versions:
    if e.get("type") != "snapshot":
        continue
    d = e.get("date", "").replace("T", " ").replace("Z", "")
    lines.append(
        f"| `{e['id']}` | {d} | `{e.get('commit', '')}` | `{e.get('sha256', '')[:12]}` | {e.get('notes', '')} |"
    )
Path(notes_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

# 3. Upload snapshot NRO + updated versions.json to the single `snapshots` pre-release
if gh release view snapshots >/dev/null 2>&1; then
  gh release upload snapshots "$ASSET_PATH" versions.json --clobber
  gh release edit snapshots --notes-file "$NOTES_MD"
else
  gh release create snapshots "$ASSET_PATH" versions.json \
    --prerelease \
    --target main \
    --title "UprisingNX Snapshots & Version Archive" \
    --notes-file "$NOTES_MD"
fi

echo "[snapshot] Published ${SNAP_ID} into rolling 'snapshots' release and updated versions.json!"
