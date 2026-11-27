#!/usr/bin/env bash
# Builds the macOS .app bundle and distributable .dmg for Kid Icarus: Uprising Recompiled.
# Supports Apple Silicon (arm64), Intel (x64), and universal Mach-O binaries, bundles
# libMoltenVK.dylib (>= 1.3.0), signs with Developer ID (or ad-hoc when unconfigured),
# and optionally notarizes + staples the resulting DMG.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build-macos.sh [--arch arm64|x64|universal]

Options:
  --arch <arch>   Target architecture: arm64 (default), x64, or universal
  -h, --help      Show this help message

Optional environment variables:
  OUT_DIR                 Output directory (default: build/pc)
  MOLTENVK_DYLIB          Explicit path to libMoltenVK.dylib (>= 1.3.0)
  APPLE_SIGN_IDENTITY     Developer ID Application signing identity (ad-hoc if unset)
  APPLE_NOTARY_KEY_PATH   App Store Connect API key (.p8) path for notarization
  APPLE_NOTARY_KEY_ID     App Store Connect API key ID
  APPLE_NOTARY_ISSUER     App Store Connect API issuer UUID
  APPLE_ID                Apple ID email (alternative notarization auth)
  APPLE_TEAM_ID           Apple Developer Team ID (alternative notarization auth)
  APPLE_APP_PASSWORD      App-specific password (alternative notarization auth)
EOF
}

ARCH="arm64"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --arch requires a value (arm64|x64|universal)" >&2
        exit 1
      fi
      ARCH="$2"
      shift 2
      ;;
    --arch=*)
      ARCH="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$ARCH" in
  arm64|x64|universal) ;;
  *)
    echo "ERROR: Invalid --arch '$ARCH' (expected arm64, x64, or universal)" >&2
    exit 1
    ;;
esac

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: scripts/pc/build-macos.sh must be run on macOS." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

VERSION="$(tr -d ' \r\n' < "$ROOT/VERSION")"
if [[ -z "$VERSION" ]]; then
  echo "ERROR: VERSION file at $ROOT/VERSION is empty." >&2
  exit 1
fi

INFO_PLIST_SRC="$ROOT/platforms/pc/macos/Info.plist"
ENTITLEMENTS_PLIST="$ROOT/platforms/pc/macos/entitlements.plist"
ICON_SRC="$ROOT/platforms/nx/c/icon.jpg"

for required_file in "$INFO_PLIST_SRC" "$ENTITLEMENTS_PLIST" "$ICON_SRC"; do
  if [[ ! -f "$required_file" ]]; then
    echo "ERROR: Required file missing: $required_file" >&2
    exit 1
  fi
done

# Locate libMoltenVK.dylib (>= 1.3.0) before compiling so missing dependencies fail fast.
locate_moltenvk() {
  if [[ -n "${MOLTENVK_DYLIB:-}" ]]; then
    if [[ -f "$MOLTENVK_DYLIB" ]]; then
      echo "$MOLTENVK_DYLIB"
      return 0
    fi
    echo "ERROR: MOLTENVK_DYLIB is set to '$MOLTENVK_DYLIB', but the file does not exist." >&2
    return 1
  fi

  local brew_prefix=""
  if command -v brew >/dev/null 2>&1; then
    brew_prefix="$(brew --prefix molten-vk 2>/dev/null || true)"
    if [[ -n "$brew_prefix" && -f "$brew_prefix/lib/libMoltenVK.dylib" ]]; then
      echo "$brew_prefix/lib/libMoltenVK.dylib"
      return 0
    fi
  fi

  if [[ -f "/opt/homebrew/lib/libMoltenVK.dylib" ]]; then
    echo "/opt/homebrew/lib/libMoltenVK.dylib"
    return 0
  fi

  if [[ -f "/usr/local/lib/libMoltenVK.dylib" ]]; then
    echo "/usr/local/lib/libMoltenVK.dylib"
    return 0
  fi

  cat >&2 <<'EOF'
ERROR: libMoltenVK.dylib (>= 1.3.0) was not found.
Checked:
  1. $MOLTENVK_DYLIB
  2. $(brew --prefix molten-vk)/lib/libMoltenVK.dylib
  3. /opt/homebrew/lib/libMoltenVK.dylib
  4. /usr/local/lib/libMoltenVK.dylib

Install MoltenVK via Homebrew (`brew install molten-vk`) or set MOLTENVK_DYLIB=/path/to/libMoltenVK.dylib.
EOF
  return 1
}

MVK_DYLIB="$(locate_moltenvk)"
echo "Using MoltenVK: $MVK_DYLIB"

if command -v brew >/dev/null 2>&1; then
  MVK_BREW_VER="$(brew list --versions molten-vk 2>/dev/null || true)"
  if [[ -n "$MVK_BREW_VER" ]]; then
    echo "Homebrew MoltenVK: $MVK_BREW_VER"
    if [[ -z "${MOLTENVK_DYLIB:-}" ]]; then
      python3 - "$MVK_BREW_VER" <<'PY'
import re, sys
line = sys.argv[1]
m = re.search(r"(\d+)\.(\d+)\.(\d+)", line)
if m:
    ver = tuple(int(x) for x in m.groups())
    if ver < (1, 3, 0):
        sys.exit(
            f"ERROR: MoltenVK >= 1.3.0 is required, but Homebrew has {'.'.join(map(str, ver))}. "
            "Run `brew upgrade molten-vk`."
        )
PY
    fi
  fi
fi

OUT_DIR="${OUT_DIR:-$ROOT/build/pc}"
APP_NAME="Kid Icarus Uprising Recompiled.app"
APP_DIR="$OUT_DIR/$APP_NAME"
APP_BIN="$APP_DIR/Contents/MacOS/uprising-recompiled"
DMG_PATH="$OUT_DIR/uprising-recompiled-macos-${ARCH}.dmg"

echo "Building uprising-recompiled v${VERSION} for macOS (${ARCH})..."
case "$ARCH" in
  arm64)
    cargo build --release -p uprising-recompiled --target aarch64-apple-darwin
    ;;
  x64)
    cargo build --release -p uprising-recompiled --target x86_64-apple-darwin
    ;;
  universal)
    cargo build --release -p uprising-recompiled --target aarch64-apple-darwin
    cargo build --release -p uprising-recompiled --target x86_64-apple-darwin
    ;;
esac

WORK_TMP="$(mktemp -d)"
trap 'rm -rf "$WORK_TMP"' EXIT

mkdir -p "$OUT_DIR"
rm -rf "$APP_DIR"
mkdir -p \
  "$APP_DIR/Contents/MacOS" \
  "$APP_DIR/Contents/Frameworks" \
  "$APP_DIR/Contents/Resources"

case "$ARCH" in
  arm64)
    cp -f "$ROOT/target/aarch64-apple-darwin/release/uprising-recompiled" "$APP_BIN"
    ;;
  x64)
    cp -f "$ROOT/target/x86_64-apple-darwin/release/uprising-recompiled" "$APP_BIN"
    ;;
  universal)
    lipo -create \
      "$ROOT/target/aarch64-apple-darwin/release/uprising-recompiled" \
      "$ROOT/target/x86_64-apple-darwin/release/uprising-recompiled" \
      -output "$APP_BIN"
    ;;
esac

# Strip local symbols BEFORE signing (stripping after signing invalidates the signature).
strip -x "$APP_BIN"
chmod 755 "$APP_BIN"

# Bundle libMoltenVK.dylib (dereferencing symlinks and making it writable for codesign).
cp -fL "$MVK_DYLIB" "$APP_DIR/Contents/Frameworks/libMoltenVK.dylib"
chmod 755 "$APP_DIR/Contents/Frameworks/libMoltenVK.dylib"

# Render Info.plist with the version from VERSION.
sed "s/__VERSION__/${VERSION}/g" "$INFO_PLIST_SRC" > "$APP_DIR/Contents/Info.plist"

# Generate AppIcon.icns from platforms/nx/c/icon.jpg via sips + iconutil.
# Note: macOS iconutil uses icon_32x32@2x.png for 64x64 (icon_64x64.png is not valid).
ICONSET_DIR="$WORK_TMP/AppIcon.iconset"
mkdir -p "$ICONSET_DIR"
sips -s format png -z 16 16     "$ICON_SRC" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -s format png -z 32 32     "$ICON_SRC" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -s format png -z 32 32     "$ICON_SRC" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -s format png -z 64 64     "$ICON_SRC" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -s format png -z 128 128   "$ICON_SRC" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -s format png -z 256 256   "$ICON_SRC" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -s format png -z 256 256   "$ICON_SRC" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -s format png -z 512 512   "$ICON_SRC" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -s format png -z 512 512   "$ICON_SRC" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -s format png -z 1024 1024 "$ICON_SRC" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/AppIcon.icns"

# Clear Finder/quarantine xattrs that can cause codesign to reject the bundle.
xattr -cr "$APP_DIR" 2>/dev/null || true

# Code-sign nested framework first, then the outer .app bundle.
if [[ -n "${APPLE_SIGN_IDENTITY:-}" ]]; then
  echo "Signing .app bundle with Developer ID identity: $APPLE_SIGN_IDENTITY"
  codesign --force --options runtime --timestamp \
    --sign "$APPLE_SIGN_IDENTITY" \
    "$APP_DIR/Contents/Frameworks/libMoltenVK.dylib"
  codesign --force --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS_PLIST" \
    --sign "$APPLE_SIGN_IDENTITY" \
    "$APP_DIR"
  codesign --verify --deep --strict --verbose=2 "$APP_DIR"
else
  echo "APPLE_SIGN_IDENTITY not set; applying ad-hoc signature."
  codesign --force --deep --sign - \
    --entitlements "$ENTITLEMENTS_PLIST" \
    "$APP_DIR"
  codesign --verify --deep --strict "$APP_DIR"
fi

# Package into a compressed UDZO .dmg with an /Applications drag-install symlink.
DMG_STAGE="$WORK_TMP/dmg-staging"
mkdir -p "$DMG_STAGE"
cp -a "$APP_DIR" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "Kid Icarus Uprising Recompiled" \
  -srcfolder "$DMG_STAGE" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

# Sign and optionally notarize + staple the .dmg when Developer ID credentials are present.
if [[ -n "${APPLE_SIGN_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$APPLE_SIGN_IDENTITY" "$DMG_PATH"

  if [[ -n "${APPLE_NOTARY_KEY_PATH:-}" && -n "${APPLE_NOTARY_KEY_ID:-}" && -n "${APPLE_NOTARY_ISSUER:-}" ]]; then
    if [[ ! -f "$APPLE_NOTARY_KEY_PATH" ]]; then
      echo "ERROR: APPLE_NOTARY_KEY_PATH does not exist: $APPLE_NOTARY_KEY_PATH" >&2
      exit 1
    fi
    echo "Submitting $DMG_PATH for notarization via App Store Connect API key..."
    xcrun notarytool submit "$DMG_PATH" \
      --key "$APPLE_NOTARY_KEY_PATH" \
      --key-id "$APPLE_NOTARY_KEY_ID" \
      --issuer "$APPLE_NOTARY_ISSUER" \
      --wait
    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
  elif [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
    echo "Submitting $DMG_PATH for notarization via Apple ID..."
    xcrun notarytool submit "$DMG_PATH" \
      --apple-id "$APPLE_ID" \
      --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" \
      --wait
    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
  else
    echo "Notarization credentials not set; skipping notarytool submit and stapler."
  fi
fi

echo "Built macOS bundle: $APP_DIR"
ls -lh "$DMG_PATH"
