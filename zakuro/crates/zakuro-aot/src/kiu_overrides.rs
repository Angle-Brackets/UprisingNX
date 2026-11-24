//! Built-in native guest function overrides for Kid Icarus: Uprising (USA 1.0,
//! `0004000000030100`, `.text` hash `0xD6E028C6175A2B07`).
//!
//! Tracks all 3 KIU control & camera patches directly in git:
//! 1. `0x006EA040` & `0x00A3EDCC` — On-foot vs air-battle mode reporting (`MAILBOX + 12`).
//! 2. `0x006D269C` & `0x006D2ABC` — Air-battle right-stick / gyro reticle delta & rail-center recenter.
//! 3. `0x007462A4` — On-foot / vehicle camera right-stick + gyro look, Z-target closest enemy
//!    on `R` / `L3` (tap to snap, hold to track + gyro ratchet), and behind-Pit recenter fallback.

use recomp_abi::{Context, EXIT_UNWIND};

use crate::hints_kiu::{KIU_USA_PROGRAM_ID, KIU_USA_TEXT_HASH};
use crate::runtime::{mem_read32, mem_read8, mem_write32, mem_write8, AotExecContext};

pub type OverrideFn = unsafe extern "C" fn(*mut AotExecContext, *const u8);

pub const KIU_USA_OVERRIDE_ADDRS: [u32; 5] = [
    0x006E_A040,
    0x00A3_EDCC,
    0x006D_269C,
    0x006D_2ABC,
    0x0074_62A4,
];

pub fn overrides_for(program_id: u64, text_hash: u64) -> Vec<(u32, OverrideFn)> {
    if program_id != KIU_USA_PROGRAM_ID || text_hash != KIU_USA_TEXT_HASH {
        return Vec::new();
    }
    vec![
        (0x006E_A040, override_006ea040),
        (0x00A3_EDCC, override_00a3edcc),
        (0x006D_269C, override_006d269c),
        (0x006D_2ABC, override_006d2abc),
        (0x0074_62A4, override_007462a4),
    ]
}

const RECOMP_DEPTH_LIMIT: u32 = 2048;

const MAILBOX: u32 = 0x0700_0000;
const GAME_MODE: u32 = MAILBOX + 12;
const GAME_ON_FOOT: u32 = 1;
const GAME_FLYING: u32 = 2;

const MOUSE_AIM: u32 = 0x314D_414D; // "MAM1"
const MOUSE_AIM_RECENTER: u32 = 0x524D_414D; // "MAMR"

const MOUSE_LOOK: u32 = 0x314B_4C4D; // "MLK1"
const MOUSE_LOOK_RECENTER: u32 = 0x524B_4C4D; // "MLKR" (R: Z-target with 2-tier range)
const MOUSE_LOOK_HOLD: u32 = 0x484B_4C4D; // "MLKH" (R held: keep tracking locked target)
const MOUSE_LOOK_CENTER: u32 = 0x434B_4C4D; // "MLKC" (L3: dedicated recenter behind Pit)

const YAW_PER_PIXEL: f32 = -0.00408;
const PITCH_PER_PIXEL: f32 = 0.00442;

const ENEMY_LIST_GUARD: u32 = 0x00D6_3710;
const ENEMY_LIST_HEAD: u32 = 0x00F2_5A78;
const FIGHTER_LIST_GUARD: u32 = 0x00D6_3708;
const FIGHTER_LIST_HEAD: u32 = 0x00F2_5A68;
const TARGET_COMP_KEY: u32 = 0x00D7_6244;
/// Air-battle & Chapter 25 Great Sacred Treasure orbit reticle vtable (`code539.c`).
const AIR_RETICLE_VTABLE: u32 = 0x00CF_1B40;

const PI_F: f32 = std::f32::consts::PI;
/// Tier-1 forward on-screen lock-on range (covers full KIU arenas & Staff/Bow range).
const ZTARGET_MAX_RANGE: f32 = 160.0;
/// Range within which an already-locked target stays tracked while holding R.
const ZTARGET_HOLD_MAX_RANGE: f32 = 220.0;
/// Tier-2 360-degree fallback range when nothing is in front of the camera (e.g. a
/// boss or flanking enemy just dashed past Pit's shoulder).
const ZTARGET_REAR_FALLBACK_RANGE: f32 = 80.0;
const ZTARGET_MIN_DIST_FOOT: f32 = 0.6;
const ZTARGET_MIN_DIST_VEHICLE: f32 = 1.8;
/// Tier-1 forward on-screen cone half-angle: 65 deg (`cos(65 deg)`).
const ZTARGET_FOV_RAD: f32 = 65.0 * PI_F / 180.0;
const ZTARGET_MIN_FWD_DOT: f32 = 0.422_618_27;
const ZTARGET_TAP_TICKS: i16 = 14;
const ZTARGET_SLEW_GAIN: f32 = 0.65;
const ZTARGET_MIN_STEP: f32 = 0.09;
const ZTARGET_MAX_STEP: f32 = 1.00;
/// Angular threshold (~18 deg) under which held Z-target switches from initial
/// snap slew to 1:1 tight Zelda lock tracking so circle-strafing never lags.
const ZTARGET_TIGHT_LOCK_RAD: f32 = 0.32;
/// 30 Hz ticks to keep tracking a locked enemy through brief untargetable
/// hit-stun / hop / flash frames while holding R/L3 before dropping lock.
const ZTARGET_FLICKER_GRACE: u8 = 15;
const ZTARGET_DONE_EPS: f32 = 0.005;
const ZTARGET_PITCH_LIMIT: f32 = 0.75;
const ZTARGET_CANCEL_STICK_PX: f32 = 1.5;
/// Horizontal right-stick flick threshold (bottom-screen px/tick) to cycle
/// locked targets left/right while holding R/L3.
const ZTARGET_SWITCH_FLICK_PX: f32 = 5.0;
/// Debounce ticks (at 30 Hz = 0.4s) between right-stick target switches.
const ZTARGET_SWITCH_COOLDOWN: u8 = 12;
/// Sentinel indicating coarse AABB targeting rather than a specific fine hitbox slot.
const ZTARGET_NO_PART: u8 = 0xFF;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZTargetState {
    pub locked_entity: u32,
    pub active: u8,
    /// Non-zero once initial snap has converged onto `locked_entity` (enables 1:1 tight tracking).
    pub locked_on: u8,
    /// Consecutive ticks `locked_entity` had its targetable flag cleared while still alive in list.
    pub lost_ticks: u8,
    /// Locked fine hitbox index (`0..63`) in `comp + 0x4C`, or `ZTARGET_NO_PART` for AABB.
    pub locked_part: u8,
    pub snap_ticks: i16,
    /// Remaining cooldown ticks after a right-stick target switch while holding R/L3.
    pub switch_cd: u8,
    pub _pad: u8,
    pub fallback_yaw: f32,
}

impl Default for ZTargetState {
    fn default() -> Self {
        Self {
            locked_entity: 0,
            active: 0,
            locked_on: 0,
            lost_ticks: 0,
            locked_part: ZTARGET_NO_PART,
            snap_ticks: 0,
            switch_cd: 0,
            _pad: 0,
            fallback_yaw: 0.0,
        }
    }
}

#[inline]
unsafe fn call_original(aot_ctx: *mut AotExecContext, orig_body: *const u8, guest_pc: u32) -> bool {
    let ec = unsafe { &mut *aot_ctx };
    ec.ctx.r[15] = guest_pc;
    ec.ctx.thumb = 0;
    ec.ctx.depth += 1;
    if ec.ctx.depth > RECOMP_DEPTH_LIMIT {
        ec.ctx.depth -= 1;
        ec.ctx.exit = EXIT_UNWIND;
        return false;
    }
    let enter: unsafe extern "C" fn(*mut AotExecContext, *const u8) =
        unsafe { std::mem::transmute(ec.aot_enter_ptr) };
    unsafe { enter(aot_ctx, orig_body) };
    let ec = unsafe { &mut *aot_ctx };
    ec.ctx.depth -= 1;
    ec.ctx.exit == 0
}

#[inline]
unsafe fn vfp_s(ctx: &Context, r: usize) -> f32 {
    f32::from_bits(unsafe { *ctx.vfp.add(r & 31) })
}

#[inline]
unsafe fn vfp_set_s(ctx: &mut Context, r: usize, mut val: f32) {
    let fpscr = unsafe { *ctx.fpscr };
    if (fpscr & (1 << 24)) != 0 && val != 0.0 && val.is_subnormal() {
        val = if val.is_sign_negative() { -0.0 } else { 0.0 };
    }
    unsafe { *ctx.vfp.add(r & 31) = val.to_bits() };
}

// ---------------------------------------------------------------------------
// 1. Aim Mode Overrides (`kiu_aim_mode.c`)
// ---------------------------------------------------------------------------

unsafe extern "C" fn override_006ea040(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        mem_write32(&mut (*aot_ctx).ctx, GAME_MODE, GAME_ON_FOOT);
        call_original(aot_ctx, orig_body, 0x006E_A040);
    }
}

unsafe extern "C" fn override_00a3edcc(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        mem_write32(&mut (*aot_ctx).ctx, GAME_MODE, GAME_FLYING);
        call_original(aot_ctx, orig_body, 0x00A3_EDCC);
    }
}

// ---------------------------------------------------------------------------
// 2. Air-Battle & Orbit Boss Reticle Overrides (`kiu_air_aim.c`)
// ---------------------------------------------------------------------------

unsafe extern "C" fn override_006d269c(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        let ctx = &mut (*aot_ctx).ctx;
        let reticle = ctx.r[0];
        let player = ctx.r[2];

        let mut preserve_pos: Option<(u32, u32)> = None;
        if valid_heap_ptr(reticle) && mem_read8(ctx, reticle + 0xD5) == 0 {
            let mark = mem_read32(ctx, MAILBOX);
            if mark == MOUSE_AIM || mark == MOUSE_AIM_RECENTER {
                // Unconditionally suppress KIU's 0.5s idle auto-centering (`f_006D1FF0`
                // Mode 1) every tick—even when the player holds the right stick still
                // (`dx == 0 && dy == 0`) while orbiting with the left stick.
                mem_write32(ctx, reticle + 0x208, 0);
                if valid_heap_ptr(player) {
                    let p_aim = player + 0x98;
                    mem_write32(ctx, p_aim + 0xD8, 0);
                    mem_write32(ctx, p_aim + 0xDC, 0x3F00_0000); // 0.5f
                    mem_write8(ctx, p_aim + 0xE0, 0);
                }

                // Zero Pit/orbit screen-follow velocity (`[reticle + 0x2F4, 0x2F8]`)
                // computed by `f_0076E854` so orbiting in air battles and Chapter 25
                // (Great Sacred Treasure vs Hades) never drags the crosshair off target.
                if mem_read32(ctx, reticle) == AIR_RETICLE_VTABLE {
                    mem_write32(ctx, reticle + 0x2F4, 0);
                    mem_write32(ctx, reticle + 0x2F8, 0);
                }

                // When the touchscreen stylus is up (`[reticle + 0x1D0] == 0`), hold
                // `[reticle + 0xA4, 0xA8]` invariant across `f_006D269C` (blocking both
                // follow-drift and `f_006D131C` pull) while still allowing real touch drags.
                if mem_read8(ctx, reticle + 0x1D0) == 0 {
                    preserve_pos = Some((
                        mem_read32(ctx, reticle + 0xA4),
                        mem_read32(ctx, reticle + 0xA8),
                    ));
                }
            }
        }

        if call_original(aot_ctx, orig_body, 0x006D_269C) {
            if let Some((orig_x, orig_y)) = preserve_pos {
                let ctx = &mut (*aot_ctx).ctx;
                mem_write32(ctx, reticle + 0xA4, orig_x);
                mem_write32(ctx, reticle + 0xA8, orig_y);
            }
        }
    }
}

unsafe extern "C" fn override_006d2abc(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        let reticle = (*aot_ctx).ctx.r[0];

        // Run KIU's native `f_006D2ABC` FIRST (matching `v0.1.0`):
        // On foot (`reticle + 0x81 != 0`), any reticle position written beyond KIU's inner
        // box (`140..260 x 48..168`) on the previous tick is clamped back to the box edge
        // and its exact overflow is written to `[cam_mgr + 0xD38, 0xD3C]` to smoothly turn
        // the camera, while movements inside the box give jitter-free pointer-style aiming.
        if !call_original(aot_ctx, orig_body, 0x006D_2ABC) {
            return;
        }

        if !valid_heap_ptr(reticle) {
            return;
        }

        let ctx = &mut (*aot_ctx).ctx;
        // Report `GAME_FLYING` for full-screen Air/Orbit modes (`0x81 == 0`) without
        // clobbering `GAME_ON_FOOT` / `GAME_VEHICLE` when `0x81 != 0`.
        if mem_read8(ctx, reticle + 0x81) == 0 {
            mem_write32(ctx, MAILBOX + 12, GAME_FLYING);
        }

        if mem_read8(ctx, reticle + 0xD5) != 0 {
            return;
        }

        // Keep KIU's aiming reticle active/visible even while the touchscreen stylus is up.
        let flags84 = mem_read32(ctx, reticle + 0x84);
        mem_write32(ctx, reticle + 0x84, flags84 | 2);

        let mark = mem_read32(ctx, MAILBOX);
        // During active Z-target (`R` press/hold) or behind-Pit recenter (`L3`), leave
        // `MAILBOX + 4/8` untouched so `override_007462a4` receives right-stick flicks to
        // switch locked targets and keeps the reticle centered on the locked enemy.
        if mark == MOUSE_LOOK_RECENTER || mark == MOUSE_LOOK_HOLD || mark == MOUSE_LOOK_CENTER {
            return;
        }

        let dx = f32::from_bits(mem_read32(ctx, MAILBOX + 4));
        let dy = f32::from_bits(mem_read32(ctx, MAILBOX + 8));
        if dx != 0.0 || dy != 0.0 {
            mem_write32(ctx, MAILBOX + 4, 0);
            mem_write32(ctx, MAILBOX + 8, 0);
        }

        let mut x = f32::from_bits(mem_read32(ctx, reticle + 0xA4));
        let mut y = f32::from_bits(mem_read32(ctx, reticle + 0xA8));
        if !x.is_finite() {
            x = 200.0;
        }
        if !y.is_finite() {
            y = 120.5;
        }

        if mark == MOUSE_AIM_RECENTER {
            mem_write32(ctx, MAILBOX, MOUSE_AIM);
            let cx = f32::from_bits(mem_read32(ctx, reticle + 0x2EC));
            let cy = f32::from_bits(mem_read32(ctx, reticle + 0x2F0));
            x = if cx.is_finite() && cx > 0.0 && cx < 400.0 { cx } else { 200.0 };
            y = if cy.is_finite() && cy > 0.0 && cy < 240.0 { cy } else { 120.5 };
            mem_write32(ctx, reticle + 0x208, 0);
            mem_write32(ctx, reticle + 0x20C, 0);
        } else if dx.is_finite() && dy.is_finite() {
            x += dx;
            y += dy;
        }

        // Always clamp to the FULL top screen (`0..400 x 0..240`), NEVER to KIU's inner
        // camera-pan box (`140..260 x 48..168`)—otherwise `f_006D2ABC` on the next tick
        // sees zero overflow and never turns the Land camera, while trapping gyro range.
        x = x.clamp(0.0, 400.0);
        y = y.clamp(0.0, 240.0);

        mem_write32(ctx, reticle + 0xA4, x.to_bits());
        mem_write32(ctx, reticle + 0xA8, y.to_bits());
    }
}

// ---------------------------------------------------------------------------
// 3. On-Foot / Vehicle Mouse Look + Z-Targeting (`kiu_mouse_look.c`)
// ---------------------------------------------------------------------------

#[inline]
unsafe fn read_f32(ctx: &mut Context, addr: u32) -> f32 {
    f32::from_bits(unsafe { mem_read32(ctx, addr) })
}

#[inline]
fn valid_heap_ptr(p: u32) -> bool {
    (0x0800_0000..0x1800_0000).contains(&p) && (p & 3) == 0
}

#[inline]
fn wrap_pi(a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    const TWO_PI: f32 = 2.0 * PI_F;
    a - (a / TWO_PI).round_ties_even() * TWO_PI
}

#[inline]
fn slew_step(err: f32) -> f32 {
    let a = err.abs();
    if a <= ZTARGET_MIN_STEP {
        return err;
    }
    let step = (a * ZTARGET_SLEW_GAIN).clamp(ZTARGET_MIN_STEP, ZTARGET_MAX_STEP);
    if err < 0.0 { -step } else { step }
}

unsafe fn find_target_holder(ctx: &mut Context, entity: u32) -> u32 {
    let holder = unsafe { mem_read32(ctx, entity + 0x1F0) };
    if valid_heap_ptr(holder) {
        return holder;
    }
    let end_node = entity + 0x1A8;
    let mut cur = unsafe { mem_read32(ctx, entity + 0x1AC) };
    let mut best = end_node;
    for _ in 0..64 {
        if !valid_heap_ptr(cur) {
            break;
        }
        if unsafe { mem_read8(ctx, cur + 0x10) } == 0 {
            break;
        }
        let k = unsafe { mem_read32(ctx, cur + 0x14) };
        if k < TARGET_COMP_KEY {
            cur = unsafe { mem_read32(ctx, cur + 0x0C) };
        } else {
            best = cur;
            cur = unsafe { mem_read32(ctx, cur + 0x08) };
        }
    }
    if best == end_node || !valid_heap_ptr(best) {
        return 0;
    }
    if unsafe { mem_read32(ctx, best + 0x14) } > TARGET_COMP_KEY {
        return 0;
    }
    let found = unsafe { mem_read32(ctx, best) };
    let end_val = unsafe { mem_read32(ctx, end_node) };
    if found == end_val || !valid_heap_ptr(found) {
        return 0;
    }
    found
}

/// Computes KIU's on-foot vertical crosshair elevation angle in radians from
/// `f_005928B0` (`code420.c:2406..2570`) and `f_00575F3C` (`code400.c:5416..5431`).
///
/// KIU shifts the on-foot crosshair vertically to `y = 120 * (1 - ratio)` where
/// `ratio = clamp(offset / (vfov * 0.5), -0.2, +0.3)`. Positive `ratio` places
/// the crosshair ABOVE screen center (`y < 120`), so adding the returned angle
/// to `cam + 0x38` (where positive pitch tilts DOWN) places a world target
/// squarely under Pit's crosshair rather than at screen center.
#[inline]
fn crosshair_pitch_offset_from_fields(vfov_raw: f32, offset_raw: f32, disabled: bool) -> f32 {
    if disabled || !vfov_raw.is_finite() || !offset_raw.is_finite() || vfov_raw <= 1e-4 {
        return 0.0;
    }
    let ratio = (offset_raw / (vfov_raw * 0.5)).clamp(-0.2, 0.3);
    // `cam_mgr + 0x358` is vertical FOV (convert degrees -> radians if > 3.2).
    let vfov_rad = if vfov_raw > 3.2 { vfov_raw.to_radians() } else { vfov_raw };
    let vfov_rad = if (0.25..=2.2).contains(&vfov_rad) { vfov_rad } else { 60.0_f32.to_radians() };
    (ratio * (vfov_rad * 0.5).tan()).atan()
}

unsafe fn read_crosshair_pitch_offset(ctx: &mut Context, cam: u32) -> f32 {
    let Some(cam_mgr) = cam.checked_sub(0x278) else {
        return 0.0;
    };
    if !valid_heap_ptr(cam_mgr) {
        return 0.0;
    }
    let disabled = unsafe { mem_read8(ctx, cam_mgr + 0xC08) == 1 && mem_read8(ctx, cam_mgr + 0xC04) == 1 };
    let vfov_raw = unsafe { read_f32(ctx, cam_mgr + 0x358) };
    let offset_raw = unsafe { read_f32(ctx, cam_mgr + 0x44C) };
    crosshair_pitch_offset_from_fields(vfov_raw, offset_raw, disabled)
}

/// Clears KIU's native L-tap behind-Pit camera recenter state (`cam_mgr + 0x804 + 0x11`,
/// consumed by `f_00822390` immediately after `f_007462A4`) so quick-tapping Fire while
/// Z-targeting never hijacks the camera away from the locked target.
unsafe fn suppress_native_recenter(ctx: &mut Context, cam: u32) {
    if let Some(cam_mgr) = cam.checked_sub(0x278) {
        if valid_heap_ptr(cam_mgr) {
            unsafe { mem_write8(ctx, cam_mgr + 0x815, 0) };
        }
    }
}

/// Resolves the best world-space aim point `(x, y, z)` and fine hitbox slot (`0..63`,
/// or `ZTARGET_NO_PART` for coarse AABB fallback) for `entity`.
///
/// Uses KIU's live multi-part hitbox array (`comp + 0x4C`, `0x34`-byte entries, active
/// bitset at `comp + 0x50`, world center at `hb + 0x0C..0x14`) when available so bosses
/// and wide/curved enemies aim squarely at an actual active weakpoint/body part instead
/// of empty air at the enclosing AABB midpoint.
#[allow(clippy::too_many_arguments)]
unsafe fn get_target_point(
    ctx: &mut Context,
    entity: u32,
    px: f32,
    py: f32,
    pz: f32,
    aim_fx: f32,
    aim_fy: f32,
    aim_fz: f32,
    preferred_part: u8,
    require_targetable: bool,
) -> Option<((f32, f32, f32), u8)> {
    if !valid_heap_ptr(entity) {
        return None;
    }
    let holder = unsafe { find_target_holder(ctx, entity) };
    if !valid_heap_ptr(holder) {
        return None;
    }
    let comp = unsafe { mem_read32(ctx, holder + 0x20) };
    if !valid_heap_ptr(comp) {
        return None;
    }
    if (unsafe { mem_read32(ctx, comp + 0x38) } & 1) == 0 {
        return None;
    }
    let flags34 = unsafe { mem_read32(ctx, comp + 0x34) };
    if require_targetable && (flags34 & 4) == 0 {
        return None;
    }

    // Prefer active fine hitboxes (`f_00AEDD2C`, `code734.c:20451`) when present.
    if (flags34 & 2) != 0 {
        let cap = unsafe { mem_read32(ctx, comp + 0x40) };
        let count = unsafe { mem_read32(ctx, comp + 0x44) };
        let hb_arr = unsafe { mem_read32(ctx, comp + 0x4C) };
        let bitset = unsafe { mem_read32(ctx, comp + 0x50) };
        if count > 0 && cap > 0 && cap <= 64 && valid_heap_ptr(hb_arr) && valid_heap_ptr(bitset) {
            // Fast path: keep tracking the previously locked part while still active.
            if (preferred_part as u32) < cap {
                let idx = preferred_part as u32;
                let word = unsafe { mem_read32(ctx, bitset + ((idx >> 5) << 2)) };
                if ((word >> (idx & 31)) & 1) != 0 {
                    let hb = hb_arr + idx * 0x34;
                    let hx = unsafe { read_f32(ctx, hb + 0x0C) };
                    let hy = unsafe { read_f32(ctx, hb + 0x10) };
                    let hz = unsafe { read_f32(ctx, hb + 0x14) };
                    if hx.is_finite() && hy.is_finite() && hz.is_finite() {
                        return Some(((hx, hy, hz), preferred_part));
                    }
                }
            }
            // Otherwise pick the active hitbox best aligned with the crosshair aim ray.
            let mut best_pos: Option<(f32, f32, f32)> = None;
            let mut best_part = ZTARGET_NO_PART;
            let mut best_dot = -2.0_f32;
            for idx in 0..cap {
                let word = unsafe { mem_read32(ctx, bitset + ((idx >> 5) << 2)) };
                if ((word >> (idx & 31)) & 1) == 0 {
                    continue;
                }
                let hb = hb_arr + idx * 0x34;
                let hx = unsafe { read_f32(ctx, hb + 0x0C) };
                let hy = unsafe { read_f32(ctx, hb + 0x10) };
                let hz = unsafe { read_f32(ctx, hb + 0x14) };
                if !(hx.is_finite() && hy.is_finite() && hz.is_finite()) {
                    continue;
                }
                let (dx, dy, dz) = (hx - px, hy - py, hz - pz);
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < 1e-4 {
                    continue;
                }
                let dot = (dx * aim_fx + dy * aim_fy + dz * aim_fz) / d2.sqrt();
                if dot > best_dot {
                    best_dot = dot;
                    best_pos = Some((hx, hy, hz));
                    best_part = idx as u8;
                }
            }
            if let Some(pos) = best_pos {
                return Some((pos, best_part));
            }
        }
    }

    // Coarse AABB midpoint fallback (`f_00602430`).
    let x0 = unsafe { read_f32(ctx, comp + 0x54) };
    let y0 = unsafe { read_f32(ctx, comp + 0x58) };
    let z0 = unsafe { read_f32(ctx, comp + 0x5C) };
    let x1 = unsafe { read_f32(ctx, comp + 0x60) };
    let y1 = unsafe { read_f32(ctx, comp + 0x64) };
    let z1 = unsafe { read_f32(ctx, comp + 0x68) };
    if !(x0.is_finite() && y0.is_finite() && z0.is_finite() && x1.is_finite() && y1.is_finite() && z1.is_finite()) {
        return None;
    }
    Some((((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5), ZTARGET_NO_PART))
}

unsafe fn entity_in_list(ctx: &mut Context, guard_addr: u32, head_addr: u32, needle: u32) -> bool {
    if (unsafe { mem_read32(ctx, guard_addr) } & 1) == 0 {
        return false;
    }
    let sentinel = unsafe { mem_read32(ctx, head_addr) };
    let mut node = unsafe { mem_read32(ctx, head_addr + 4) };
    for _ in 0..256 {
        if !valid_heap_ptr(node) {
            break;
        }
        let ent = unsafe { mem_read32(ctx, node) };
        if ent == sentinel {
            break;
        }
        if ent == needle {
            return true;
        }
        node = unsafe { mem_read32(ctx, node + 4) };
    }
    false
}

#[allow(clippy::too_many_arguments)]
unsafe fn is_live_target(
    ctx: &mut Context,
    entity: u32,
    pit_entity: u32,
    px: f32,
    py: f32,
    pz: f32,
    aim_fx: f32,
    aim_fy: f32,
    aim_fz: f32,
    preferred_part: u8,
    min_dist: f32,
    max_dist: f32,
    require_targetable: bool,
) -> Option<((f32, f32, f32), u8)> {
    if !valid_heap_ptr(entity) || entity == pit_entity {
        return None;
    }
    let in_lists = unsafe {
        entity_in_list(ctx, ENEMY_LIST_GUARD, ENEMY_LIST_HEAD, entity)
            || entity_in_list(ctx, FIGHTER_LIST_GUARD, FIGHTER_LIST_HEAD, entity)
    };
    if !in_lists {
        return None;
    }
    let ((tx, ty, tz), part) = unsafe {
        get_target_point(ctx, entity, px, py, pz, aim_fx, aim_fy, aim_fz, preferred_part, require_targetable)
    }?;
    let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
    let d2 = dx * dx + dy * dy + dz * dz;
    if d2 < min_dist * min_dist || d2 > max_dist * max_dist {
        return None;
    }
    Some(((tx, ty, tz), part))
}

#[inline]
fn screen_right_vec(yaw: f32) -> (f32, f32) {
    // In KIU (`YAW_PER_PIXEL < 0`), turning right decreases `yaw`, shifting
    // `(sin(yaw), cos(yaw))` toward `(-cos(yaw), sin(yaw))`.
    let (sy, cy) = yaw.sin_cos();
    (-cy, sy)
}

#[allow(clippy::too_many_arguments)]
unsafe fn scan_entity_list(
    ctx: &mut Context,
    guard_addr: u32,
    head_addr: u32,
    pit_entity: u32,
    exclude_entity: u32,
    switch_dir: i8,
    require_front_cone: bool,
    px: f32,
    py: f32,
    pz: f32,
    aim_fx: f32,
    aim_fy: f32,
    aim_fz: f32,
    right_x: f32,
    right_z: f32,
    min_dist: f32,
    max_dist: f32,
    best_entity: &mut u32,
    best_part: &mut u8,
    best_score: &mut f32,
) {
    if (unsafe { mem_read32(ctx, guard_addr) } & 1) == 0 {
        return;
    }
    let sentinel = unsafe { mem_read32(ctx, head_addr) };
    let mut node = unsafe { mem_read32(ctx, head_addr + 4) };
    let min_d2 = min_dist * min_dist;
    let max_d2 = max_dist * max_dist;
    for _ in 0..256 {
        if !valid_heap_ptr(node) {
            break;
        }
        let ent = unsafe { mem_read32(ctx, node) };
        if ent == sentinel {
            break;
        }
        node = unsafe { mem_read32(ctx, node + 4) };
        if ent == pit_entity || ent == exclude_entity {
            continue;
        }
        let Some(((tx, ty, tz), part)) = (unsafe {
            get_target_point(ctx, ent, px, py, pz, aim_fx, aim_fy, aim_fz, ZTARGET_NO_PART, true)
        }) else {
            continue;
        };
        let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < min_d2 || d2 > max_d2 {
            continue;
        }
        let dist = d2.sqrt();
        let inv_dist = 1.0 / dist;
        let fwd_dot = (dx * aim_fx + dy * aim_fy + dz * aim_fz) * inv_dist;
        if require_front_cone && fwd_dot < ZTARGET_MIN_FWD_DOT {
            continue;
        }
        let score = if switch_dir != 0 {
            let right_proj = (dx * right_x + dz * right_z) * inv_dist;
            let dir_proj = if switch_dir > 0 { right_proj } else { -right_proj };
            if dir_proj < 0.05 {
                continue;
            }
            dir_proj * (1.0 + dist / 80.0)
        } else if require_front_cone {
            // Tier 1 (on-screen ±65° up to 160u): prioritize the enemy closest to the
            // player's crosshair ray, with a gentle distance weight so closer enemies
            // win ties while sniping across an arena still locks cleanly.
            let ang_err = fwd_dot.clamp(-1.0, 1.0).acos();
            (ang_err / ZTARGET_FOV_RAD) * (1.0 + dist / 80.0) + dist * 0.0025
        } else {
            // Tier 2 (360° fallback up to 80u when nothing is ahead): pick the nearest
            // threat (preferring flanking enemies over directly behind).
            let ang_err = fwd_dot.clamp(-1.0, 1.0).acos();
            dist * (0.65 + 0.35 * (ang_err / PI_F))
        };
        if score < *best_score {
            *best_score = score;
            *best_entity = ent;
            *best_part = part;
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn find_best_target(
    ctx: &mut Context,
    pit_entity: u32,
    exclude_entity: u32,
    switch_dir: i8,
    px: f32,
    py: f32,
    pz: f32,
    cur_yaw: f32,
    aim_pitch: f32,
    min_dist: f32,
    max_dist: f32,
) -> (u32, u8) {
    let (sy, cy) = cur_yaw.sin_cos();
    let (sp, cp) = aim_pitch.sin_cos();
    let aim_fx = sy * cp;
    let aim_fy = -sp;
    let aim_fz = cy * cp;
    let (right_x, right_z) = screen_right_vec(cur_yaw);

    // Tier 1: forward on-screen cone (±65°, up to `max_dist` = 160u).
    let mut best = 0u32;
    let mut best_part = ZTARGET_NO_PART;
    let mut best_score = 1.0e30_f32;
    unsafe {
        scan_entity_list(
            ctx,
            ENEMY_LIST_GUARD,
            ENEMY_LIST_HEAD,
            pit_entity,
            exclude_entity,
            switch_dir,
            true,
            px,
            py,
            pz,
            aim_fx,
            aim_fy,
            aim_fz,
            right_x,
            right_z,
            min_dist,
            max_dist,
            &mut best,
            &mut best_part,
            &mut best_score,
        );
        scan_entity_list(
            ctx,
            FIGHTER_LIST_GUARD,
            FIGHTER_LIST_HEAD,
            pit_entity,
            exclude_entity,
            switch_dir,
            true,
            px,
            py,
            pz,
            aim_fx,
            aim_fy,
            aim_fz,
            right_x,
            right_z,
            min_dist,
            max_dist,
            &mut best,
            &mut best_part,
            &mut best_score,
        );
    }
    if best != 0 || switch_dir != 0 {
        return (best, best_part);
    }

    // Tier 2: nothing visible in front — fall back to a 360° scan within 80u so a
    // boss or flanking enemy that dashed past Pit still locks on immediately.
    let rear_max = max_dist.min(ZTARGET_REAR_FALLBACK_RANGE);
    let mut rear_score = 1.0e30_f32;
    unsafe {
        scan_entity_list(
            ctx,
            ENEMY_LIST_GUARD,
            ENEMY_LIST_HEAD,
            pit_entity,
            exclude_entity,
            0,
            false,
            px,
            py,
            pz,
            aim_fx,
            aim_fy,
            aim_fz,
            right_x,
            right_z,
            min_dist,
            rear_max,
            &mut best,
            &mut best_part,
            &mut rear_score,
        );
        scan_entity_list(
            ctx,
            FIGHTER_LIST_GUARD,
            FIGHTER_LIST_HEAD,
            pit_entity,
            exclude_entity,
            0,
            false,
            px,
            py,
            pz,
            aim_fx,
            aim_fy,
            aim_fz,
            right_x,
            right_z,
            min_dist,
            rear_max,
            &mut best,
            &mut best_part,
            &mut rear_score,
        );
    }
    (best, best_part)
}

/// Applies `(yaw, pitch)` directly to KIU's on-foot camera struct, matching the
/// exact 8-float commit performed by `f_006063E8` (`cam+0x34/0x38`, `cam+0x0C..0x14`
/// forward vector, and `cam+0x00..0x08` eye position) without `f_0090AE6C` spring
/// damping or deadzone stalls.
unsafe fn apply_cam_angles(ctx: &mut Context, cam: u32, yaw: f32, pitch: f32) {
    let yaw = wrap_pi(yaw);
    let pitch = pitch.clamp(-ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let fx = sy * cp;
    let fy = -sp;
    let fz = cy * cp;
    unsafe {
        mem_write32(ctx, cam + 0x34, yaw.to_bits());
        mem_write32(ctx, cam + 0x38, pitch.to_bits());
        mem_write32(ctx, cam + 0x0C, fx.to_bits());
        mem_write32(ctx, cam + 0x10, fy.to_bits());
        mem_write32(ctx, cam + 0x14, fz.to_bits());
        let dist = read_f32(ctx, cam + 0x30);
        let px = read_f32(ctx, cam + 0x24);
        let py = read_f32(ctx, cam + 0x28);
        let pz = read_f32(ctx, cam + 0x2C);
        if dist.is_finite() && px.is_finite() && py.is_finite() && pz.is_finite() {
            mem_write32(ctx, cam + 0x00, (px - fx * dist).to_bits());
            mem_write32(ctx, cam + 0x04, (py - fy * dist).to_bits());
            mem_write32(ctx, cam + 0x08, (pz - fz * dist).to_bits());
        }
    }
}

unsafe extern "C" fn override_007462a4(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        let ec = &mut *aot_ctx;
        let ctrl = ec.ctx.r[0];
        let cam = ec.ctx.r[1];
        let cam_ok = (0x0800_0278..0x1800_0000).contains(&cam) && (cam & 3) == 0;
        let in_vehicle = cam_ok && mem_read8(&mut ec.ctx, cam - 0xFF) != 0xFF;
        mem_write32(&mut ec.ctx, MAILBOX + 12, if in_vehicle { 3 } else { 1 });

        let mut direct_angles: Option<(f32, f32)> = None;
        let mark = mem_read32(&mut ec.ctx, MAILBOX);
        if mark == MOUSE_LOOK
            || mark == MOUSE_LOOK_RECENTER
            || mark == MOUSE_LOOK_HOLD
            || mark == MOUSE_LOOK_CENTER
        {
            let mut across = f32::from_bits(mem_read32(&mut ec.ctx, MAILBOX + 4));
            let mut down = f32::from_bits(mem_read32(&mut ec.ctx, MAILBOX + 8));
            mem_write32(&mut ec.ctx, MAILBOX + 4, 0);
            mem_write32(&mut ec.ctx, MAILBOX + 8, 0);
            if !across.is_finite() {
                across = 0.0;
            }
            if !down.is_finite() {
                down = 0.0;
            }

            if !cam_ok {
                ec.ztarget = ZTargetState::default();
                let s0 = vfp_s(&ec.ctx, 0);
                let s1 = vfp_s(&ec.ctx, 1);
                vfp_set_s(&mut ec.ctx, 0, s0 + across * YAW_PER_PIXEL);
                vfp_set_s(&mut ec.ctx, 1, s1 + down * PITCH_PER_PIXEL);
                call_original(aot_ctx, orig_body, 0x0074_62A4);
                return;
            }

            let pit_entity = mem_read32(&mut ec.ctx, cam - 0x1B8);
            let mut pit_yaw = read_f32(&mut ec.ctx, cam - 0x178);
            let px = read_f32(&mut ec.ctx, cam + 0x24);
            let py = read_f32(&mut ec.ctx, cam + 0x28);
            let pz = read_f32(&mut ec.ctx, cam + 0x2C);
            let mut cur_yaw = read_f32(&mut ec.ctx, cam + 0x34);
            let mut cur_pitch = read_f32(&mut ec.ctx, cam + 0x38);
            let min_dist = if in_vehicle { ZTARGET_MIN_DIST_VEHICLE } else { ZTARGET_MIN_DIST_FOOT };

            if !cur_yaw.is_finite() {
                cur_yaw = 0.0;
            }
            if !cur_pitch.is_finite() {
                cur_pitch = 0.0;
            }
            if !pit_yaw.is_finite() {
                pit_yaw = cur_yaw;
            }

            let pitch_off = read_crosshair_pitch_offset(&mut ec.ctx, cam);
            let aim_pitch = cur_pitch - pitch_off;
            let (sy, cy) = cur_yaw.sin_cos();
            let (sp, cp) = aim_pitch.sin_cos();
            let (aim_fx, aim_fy, aim_fz) = (sy * cp, -sp, cy * cp);

            let mut holding = mark == MOUSE_LOOK_HOLD;
            let pos_ok = px.is_finite() && py.is_finite() && pz.is_finite();
            if ec.ztarget.switch_cd > 0 {
                ec.ztarget.switch_cd -= 1;
            }

            if mark == MOUSE_LOOK_CENTER {
                // Dedicated behind-Pit recenter (`L3`): always snap cleanly behind Pit,
                // never hijacked by any nearby enemy.
                mem_write32(&mut ec.ctx, MAILBOX, MOUSE_LOOK);
                holding = false;
                ec.ztarget.locked_entity = 0;
                ec.ztarget.locked_part = ZTARGET_NO_PART;
                ec.ztarget.fallback_yaw = pit_yaw;
                ec.ztarget.active = 1;
                ec.ztarget.locked_on = 0;
                ec.ztarget.lost_ticks = 0;
                ec.ztarget.switch_cd = 0;
                ec.ztarget.snap_ticks = ZTARGET_TAP_TICKS;
            } else if mark == MOUSE_LOOK_RECENTER {
                mem_write32(&mut ec.ctx, MAILBOX, MOUSE_LOOK_HOLD);
                holding = true;
                let (ent, part) = if pos_ok {
                    find_best_target(
                        &mut ec.ctx,
                        pit_entity,
                        0,
                        0,
                        px,
                        py,
                        pz,
                        cur_yaw,
                        aim_pitch,
                        min_dist,
                        ZTARGET_MAX_RANGE,
                    )
                } else {
                    (0, ZTARGET_NO_PART)
                };
                ec.ztarget.locked_entity = ent;
                ec.ztarget.locked_part = part;
                ec.ztarget.fallback_yaw = pit_yaw;
                ec.ztarget.active = 1;
                ec.ztarget.locked_on = 0;
                ec.ztarget.lost_ticks = 0;
                ec.ztarget.switch_cd = 0;
                ec.ztarget.snap_ticks = ZTARGET_TAP_TICKS;
            } else if !holding && ec.ztarget.active != 0 {
                if across.abs() > ZTARGET_CANCEL_STICK_PX || down.abs() > ZTARGET_CANCEL_STICK_PX {
                    ec.ztarget = ZTargetState::default();
                }
            }

            // While holding R/L3 locked onto an enemy, flicking the right stick left or
            // right switches lock to the next visible enemy in that direction.
            if holding
                && ec.ztarget.locked_entity != 0
                && ec.ztarget.switch_cd == 0
                && pos_ok
                && across.abs() >= ZTARGET_SWITCH_FLICK_PX
            {
                let dir: i8 = if across > 0.0 { 1 } else { -1 };
                let (next_ent, next_part) = find_best_target(
                    &mut ec.ctx,
                    pit_entity,
                    ec.ztarget.locked_entity,
                    dir,
                    px,
                    py,
                    pz,
                    cur_yaw,
                    aim_pitch,
                    min_dist,
                    ZTARGET_MAX_RANGE,
                );
                if next_ent != 0 {
                    ec.ztarget.locked_entity = next_ent;
                    ec.ztarget.locked_part = next_part;
                    ec.ztarget.active = 1;
                    ec.ztarget.locked_on = 0;
                    ec.ztarget.lost_ticks = 0;
                    ec.ztarget.switch_cd = ZTARGET_SWITCH_COOLDOWN;
                    ec.ztarget.snap_ticks = ZTARGET_TAP_TICKS;
                }
            }

            // While R/L3 is held without a locked target, keep scanning the forward cone
            // so approaching or turning toward an enemy locks on automatically.
            if holding && ec.ztarget.locked_entity == 0 && pos_ok {
                let (found, part) = find_best_target(
                    &mut ec.ctx,
                    pit_entity,
                    0,
                    0,
                    px,
                    py,
                    pz,
                    cur_yaw,
                    aim_pitch,
                    min_dist,
                    ZTARGET_MAX_RANGE,
                );
                if found != 0 {
                    ec.ztarget.locked_entity = found;
                    ec.ztarget.locked_part = part;
                    ec.ztarget.active = 1;
                    ec.ztarget.locked_on = 0;
                    ec.ztarget.lost_ticks = 0;
                    ec.ztarget.snap_ticks = ZTARGET_TAP_TICKS;
                }
            }

            if ec.ztarget.active != 0 && (holding || ec.ztarget.snap_ticks > 0) {
                let mut goal_yaw = ec.ztarget.fallback_yaw;
                let mut goal_pitch = 0.0_f32;
                let mut have_goal = false;

                if ec.ztarget.locked_entity != 0 {
                    let range = if holding { ZTARGET_HOLD_MAX_RANGE } else { ZTARGET_MAX_RANGE };
                    let live = if pos_ok {
                        match is_live_target(
                            &mut ec.ctx,
                            ec.ztarget.locked_entity,
                            pit_entity,
                            px,
                            py,
                            pz,
                            aim_fx,
                            aim_fy,
                            aim_fz,
                            ec.ztarget.locked_part,
                            min_dist,
                            range,
                            true,
                        ) {
                            Some((pos, part)) => {
                                ec.ztarget.lost_ticks = 0;
                                ec.ztarget.locked_part = part;
                                Some(pos)
                            }
                            None if holding && ec.ztarget.lost_ticks < ZTARGET_FLICKER_GRACE => {
                                let soft = is_live_target(
                                    &mut ec.ctx,
                                    ec.ztarget.locked_entity,
                                    pit_entity,
                                    px,
                                    py,
                                    pz,
                                    aim_fx,
                                    aim_fy,
                                    aim_fz,
                                    ec.ztarget.locked_part,
                                    min_dist,
                                    range,
                                    false,
                                );
                                if let Some((pos, part)) = soft {
                                    ec.ztarget.lost_ticks = ec.ztarget.lost_ticks.saturating_add(1);
                                    ec.ztarget.locked_part = part;
                                    Some(pos)
                                } else {
                                    None
                                }
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    if let Some((tx, ty, tz)) = live {
                        let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
                        goal_yaw = dx.atan2(dz);
                        // Compensate for KIU's vertical crosshair offset (`f_005928B0`) so the
                        // locked hitbox lands squarely inside Pit's reticle rather than below it.
                        goal_pitch = ((-dy).atan2(dx.hypot(dz)) + pitch_off)
                            .clamp(-ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
                        have_goal = true;
                    } else if holding && pos_ok {
                        let (next_ent, next_part) = find_best_target(
                            &mut ec.ctx,
                            pit_entity,
                            0,
                            0,
                            px,
                            py,
                            pz,
                            cur_yaw,
                            aim_pitch,
                            min_dist,
                            ZTARGET_MAX_RANGE,
                        );
                        ec.ztarget.locked_entity = next_ent;
                        ec.ztarget.locked_part = next_part;
                        ec.ztarget.locked_on = 0;
                        ec.ztarget.lost_ticks = 0;
                        let next_center = if next_ent != 0 {
                            get_target_point(
                                &mut ec.ctx,
                                next_ent,
                                px,
                                py,
                                pz,
                                aim_fx,
                                aim_fy,
                                aim_fz,
                                next_part,
                                true,
                            )
                            .map(|(p, _)| p)
                        } else {
                            None
                        };
                        if let Some((tx, ty, tz)) = next_center {
                            let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
                            goal_yaw = dx.atan2(dz);
                            goal_pitch = ((-dy).atan2(dx.hypot(dz)) + pitch_off)
                                .clamp(-ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
                            have_goal = true;
                        } else {
                            ec.ztarget.active = 0;
                            ec.ztarget.snap_ticks = 0;
                        }
                    } else {
                        ec.ztarget = ZTargetState::default();
                    }
                } else {
                    have_goal = true;
                }

                if have_goal {
                    let yaw_err = wrap_pi(goal_yaw - cur_yaw);
                    let pitch_err = goal_pitch - cur_pitch;
                    let tight_hold = holding
                        && ec.ztarget.locked_entity != 0
                        && (ec.ztarget.locked_on != 0
                            || (yaw_err.abs() < ZTARGET_TIGHT_LOCK_RAD
                                && pitch_err.abs() < ZTARGET_TIGHT_LOCK_RAD));
                    let (dyaw, dpitch) = if tight_hold {
                        ec.ztarget.locked_on = 1;
                        (
                            yaw_err.clamp(-ZTARGET_MAX_STEP, ZTARGET_MAX_STEP),
                            pitch_err.clamp(-ZTARGET_MAX_STEP, ZTARGET_MAX_STEP),
                        )
                    } else {
                        (slew_step(yaw_err), slew_step(pitch_err))
                    };

                    if valid_heap_ptr(ctrl) {
                        mem_write32(&mut ec.ctx, ctrl + 0x24, 0);
                        mem_write32(&mut ec.ctx, ctrl + 0x38, 0);
                    }
                    suppress_native_recenter(&mut ec.ctx, cam);

                    // Pass zero delta to `f_007462A4` so its spring damper (`f_0090AE6C`)
                    // does not undershoot or stall our Z-target step; we commit the exact
                    // camera angles immediately after `call_original` returns.
                    vfp_set_s(&mut ec.ctx, 0, 0.0);
                    vfp_set_s(&mut ec.ctx, 1, 0.0);
                    direct_angles = Some((cur_yaw + dyaw, cur_pitch + dpitch));

                    let rem_yaw = wrap_pi(goal_yaw - (cur_yaw + dyaw)).abs();
                    let rem_pitch = (goal_pitch - (cur_pitch + dpitch)).abs();
                    let close_enough = rem_yaw < ZTARGET_DONE_EPS && rem_pitch < ZTARGET_DONE_EPS;
                    if ec.ztarget.snap_ticks > 0 {
                        ec.ztarget.snap_ticks -= 1;
                    }
                    if !holding {
                        if close_enough || ec.ztarget.snap_ticks <= 0 {
                            ec.ztarget = ZTargetState::default();
                        }
                    } else if ec.ztarget.locked_entity == 0 && (close_enough || ec.ztarget.snap_ticks <= 0) {
                        ec.ztarget.active = 0;
                        ec.ztarget.snap_ticks = 0;
                    }
                }
            } else if !holding {
                ec.ztarget = ZTargetState::default();
            }

            if direct_angles.is_none() {
                let s0 = vfp_s(&ec.ctx, 0);
                let s1 = vfp_s(&ec.ctx, 1);
                vfp_set_s(&mut ec.ctx, 0, s0 + across * YAW_PER_PIXEL);
                vfp_set_s(&mut ec.ctx, 1, s1 + down * PITCH_PER_PIXEL);
            }
        } else {
            ec.ztarget = ZTargetState::default();
        }

        if call_original(aot_ctx, orig_body, 0x0074_62A4) {
            if let Some((new_yaw, new_pitch)) = direct_angles {
                let ec = &mut *aot_ctx;
                if valid_heap_ptr(ctrl) {
                    mem_write32(&mut ec.ctx, ctrl + 0x24, 0);
                    mem_write32(&mut ec.ctx, ctrl + 0x38, 0);
                }
                suppress_native_recenter(&mut ec.ctx, cam);
                apply_cam_angles(&mut ec.ctx, cam, new_yaw, new_pitch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ztarget_state_abi_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<ZTargetState>(), 16);
        assert_eq!(ZTargetState::default().locked_part, ZTARGET_NO_PART);
    }

    #[test]
    fn crosshair_pitch_offset_matches_kiu_formula() {
        // 60 deg vertical FOV, offset = +0.3 * half_fov -> ratio = +0.3 -> ~+9.83 deg (+0.1715 rad)
        let vfov_deg = 60.0_f32;
        let off_pos = crosshair_pitch_offset_from_fields(vfov_deg, 0.3 * (vfov_deg * 0.5), false);
        assert!((off_pos - (0.3 * (30.0_f32.to_radians()).tan()).atan()).abs() < 1e-5);
        assert!(off_pos > 0.16 && off_pos < 0.18);

        // Same angle in radians produces identical result
        let vfov_rad = vfov_deg.to_radians();
        let off_rad = crosshair_pitch_offset_from_fields(vfov_rad, 0.3 * (vfov_rad * 0.5), false);
        assert!((off_pos - off_rad).abs() < 1e-5);

        // Clamps above +0.3 and below -0.2, returns 0 when disabled
        let off_hi = crosshair_pitch_offset_from_fields(vfov_deg, 100.0, false);
        assert!((off_hi - off_pos).abs() < 1e-5);
        let off_lo = crosshair_pitch_offset_from_fields(vfov_deg, -100.0, false);
        assert!(off_lo < -0.10 && off_lo > -0.13);
        assert_eq!(crosshair_pitch_offset_from_fields(vfov_deg, 10.0, true), 0.0);
    }

    #[test]
    fn screen_right_vec_matches_negative_yaw_right_turn() {
        // At yaw = 0, cam_fwd = (0, 0, +1). Turning right decreases yaw -> x becomes negative.
        let (rx, rz) = screen_right_vec(0.0);
        assert!((rx - (-1.0)).abs() < 1e-6);
        assert!(rz.abs() < 1e-6);
        // At yaw = +pi/2, cam_fwd = (+1, 0, 0). Decreasing yaw moves toward +z.
        let (rx2, rz2) = screen_right_vec(PI_F * 0.5);
        assert!(rx2.abs() < 1e-6);
        assert!((rz2 - 1.0).abs() < 1e-6);
    }
}
