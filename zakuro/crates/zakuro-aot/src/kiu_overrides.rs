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
const MOUSE_LOOK_RECENTER: u32 = 0x524B_4C4D; // "MLKR"
const MOUSE_LOOK_HOLD: u32 = 0x484B_4C4D; // "MLKH"

const YAW_PER_PIXEL: f32 = -0.00408;
const PITCH_PER_PIXEL: f32 = 0.00442;

const ENEMY_LIST_GUARD: u32 = 0x00D6_3710;
const ENEMY_LIST_HEAD: u32 = 0x00F2_5A78;
const FIGHTER_LIST_GUARD: u32 = 0x00D6_3708;
const FIGHTER_LIST_HEAD: u32 = 0x00F2_5A68;
const TARGET_COMP_KEY: u32 = 0x00D7_6244;

const PI_F: f32 = std::f32::consts::PI;
const ZTARGET_MAX_RANGE: f32 = 45.0;
const ZTARGET_HOLD_MAX_RANGE: f32 = 60.0;
const ZTARGET_MIN_DIST_FOOT: f32 = 0.6;
const ZTARGET_MIN_DIST_VEHICLE: f32 = 1.8;
/// Initial lock-on cone half-angle: 65 deg from camera forward ray (`cos(65 deg)`).
/// Anything outside this front cone cleanly falls back to behind-Pit recenter.
const ZTARGET_MAX_FOV_RAD: f32 = 65.0 * PI_F / 180.0;
const ZTARGET_MIN_FWD_DOT: f32 = 0.422_618_27;
const ZTARGET_TAP_TICKS: i32 = 14;
const ZTARGET_SLEW_GAIN: f32 = 0.55;
const ZTARGET_MIN_STEP: f32 = 0.08;
const ZTARGET_MAX_STEP: f32 = 0.85;
const ZTARGET_DONE_EPS: f32 = 0.005;
const ZTARGET_PITCH_LIMIT: f32 = 0.75;
const ZTARGET_CANCEL_STICK_PX: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ZTargetState {
    pub locked_entity: u32,
    pub active: i32,
    pub snap_ticks: i32,
    pub fallback_yaw: f32,
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
// 2. Air-Battle Reticle Overrides (`kiu_air_aim.c`)
// ---------------------------------------------------------------------------

unsafe extern "C" fn override_006d269c(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        let ctx = &mut (*aot_ctx).ctx;
        let reticle = ctx.r[0];
        let player = ctx.r[2];

        if (0x0800_0000..0x1800_0000).contains(&reticle) && mem_read8(ctx, reticle + 0xD5) == 0 {
            let dx = f32::from_bits(mem_read32(ctx, MAILBOX + 4));
            let dy = f32::from_bits(mem_read32(ctx, MAILBOX + 8));
            if dx.is_finite() && dy.is_finite() && (dx != 0.0 || dy != 0.0) {
                mem_write32(ctx, reticle + 0x208, 0);
                if (0x0800_0000..0x1800_0000).contains(&player) {
                    let p_aim = player + 0x98;
                    mem_write32(ctx, p_aim + 0xD8, 0);
                    mem_write32(ctx, p_aim + 0xDC, 0x3F00_0000); // 0.5f
                    mem_write8(ctx, p_aim + 0xE0, 0);
                }
            }
        }

        call_original(aot_ctx, orig_body, 0x006D_269C);
    }
}

unsafe extern "C" fn override_006d2abc(aot_ctx: *mut AotExecContext, orig_body: *const u8) {
    unsafe {
        let (reticle, xmin, ymin, xmax, ymax) = {
            let ctx = &(*aot_ctx).ctx;
            (
                ctx.r[0],
                vfp_s(ctx, 0),
                vfp_s(ctx, 1),
                vfp_s(ctx, 2),
                vfp_s(ctx, 3),
            )
        };

        if !call_original(aot_ctx, orig_body, 0x006D_2ABC) {
            return;
        }

        if !(0x0800_0000..0x1800_0000).contains(&reticle) {
            return;
        }

        let ctx = &mut (*aot_ctx).ctx;
        mem_write32(ctx, MAILBOX + 12, 2);

        if mem_read8(ctx, reticle + 0xD5) != 0 {
            return;
        }

        let flags84 = mem_read32(ctx, reticle + 0x84);
        mem_write32(ctx, reticle + 0x84, flags84 | 2);

        let mark = mem_read32(ctx, MAILBOX);
        let dx = f32::from_bits(mem_read32(ctx, MAILBOX + 4));
        let dy = f32::from_bits(mem_read32(ctx, MAILBOX + 8));
        if dx != 0.0 || dy != 0.0 {
            mem_write32(ctx, MAILBOX + 4, 0);
            mem_write32(ctx, MAILBOX + 8, 0);
        }

        let mut x = f32::from_bits(mem_read32(ctx, reticle + 0xA4));
        let mut y = f32::from_bits(mem_read32(ctx, reticle + 0xA8));

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

        x = if xmin.is_finite() && xmax.is_finite() && xmax > xmin {
            x.clamp(xmin, xmax)
        } else {
            x.clamp(0.0, 400.0)
        };
        y = if ymin.is_finite() && ymax.is_finite() && ymax > ymin {
            y.clamp(ymin, ymax)
        } else {
            y.clamp(0.0, 240.0)
        };

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

unsafe fn get_target_center(ctx: &mut Context, entity: u32) -> Option<(f32, f32, f32)> {
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
    if (unsafe { mem_read32(ctx, comp + 0x34) } & 4) == 0 {
        return None;
    }
    let x0 = unsafe { read_f32(ctx, comp + 0x54) };
    let y0 = unsafe { read_f32(ctx, comp + 0x58) };
    let z0 = unsafe { read_f32(ctx, comp + 0x5C) };
    let x1 = unsafe { read_f32(ctx, comp + 0x60) };
    let y1 = unsafe { read_f32(ctx, comp + 0x64) };
    let z1 = unsafe { read_f32(ctx, comp + 0x68) };
    if !(x0.is_finite() && y0.is_finite() && z0.is_finite() && x1.is_finite() && y1.is_finite() && z1.is_finite()) {
        return None;
    }
    Some(((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5))
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

unsafe fn is_live_target(
    ctx: &mut Context,
    entity: u32,
    pit_entity: u32,
    px: f32,
    py: f32,
    pz: f32,
    min_dist: f32,
    max_dist: f32,
) -> Option<(f32, f32, f32)> {
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
    let (tx, ty, tz) = unsafe { get_target_center(ctx, entity) }?;
    let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
    let d2 = dx * dx + dy * dy + dz * dz;
    if d2 < min_dist * min_dist || d2 > max_dist * max_dist {
        return None;
    }
    Some((tx, ty, tz))
}

#[allow(clippy::too_many_arguments)]
unsafe fn scan_entity_list(
    ctx: &mut Context,
    guard_addr: u32,
    head_addr: u32,
    pit_entity: u32,
    px: f32,
    py: f32,
    pz: f32,
    cam_fx: f32,
    cam_fy: f32,
    cam_fz: f32,
    min_dist: f32,
    max_dist: f32,
    best_entity: &mut u32,
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
        if ent == pit_entity {
            continue;
        }
        let Some((tx, ty, tz)) = (unsafe { get_target_center(ctx, ent) }) else {
            continue;
        };
        let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < min_d2 || d2 > max_d2 {
            continue;
        }
        let dist = d2.sqrt();
        // Require the target to sit inside the +/-65 deg 3D front cone of the
        // camera so off-screen/rear enemies never hijack behind-Pit recenter.
        let inv_dist = 1.0 / dist;
        let fwd_dot = (dx * cam_fx + dy * cam_fy + dz * cam_fz) * inv_dist;
        if fwd_dot < ZTARGET_MIN_FWD_DOT {
            continue;
        }
        let ang_err = fwd_dot.clamp(-1.0, 1.0).acos();
        // Strongly prioritize enemies closest to the crosshair, using distance
        // as a secondary tie-breaker.
        let score = dist * (0.20 + 2.80 * (ang_err / ZTARGET_MAX_FOV_RAD));
        if score < *best_score {
            *best_score = score;
            *best_entity = ent;
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn find_best_target(
    ctx: &mut Context,
    pit_entity: u32,
    px: f32,
    py: f32,
    pz: f32,
    cur_yaw: f32,
    cur_pitch: f32,
    min_dist: f32,
    max_dist: f32,
) -> u32 {
    let (sy, cy) = cur_yaw.sin_cos();
    let (sp, cp) = cur_pitch.sin_cos();
    let cam_fx = sy * cp;
    let cam_fy = -sp;
    let cam_fz = cy * cp;

    let mut best = 0u32;
    let mut best_score = 1.0e30_f32;
    unsafe {
        scan_entity_list(
            ctx,
            ENEMY_LIST_GUARD,
            ENEMY_LIST_HEAD,
            pit_entity,
            px,
            py,
            pz,
            cam_fx,
            cam_fy,
            cam_fz,
            min_dist,
            max_dist,
            &mut best,
            &mut best_score,
        );
        scan_entity_list(
            ctx,
            FIGHTER_LIST_GUARD,
            FIGHTER_LIST_HEAD,
            pit_entity,
            px,
            py,
            pz,
            cam_fx,
            cam_fy,
            cam_fz,
            min_dist,
            max_dist,
            &mut best,
            &mut best_score,
        );
    }
    best
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
        let cam_ok = (0x0800_0200..0x1800_0000).contains(&cam) && (cam & 3) == 0;
        let in_vehicle = cam_ok && mem_read8(&mut ec.ctx, cam - 0xFF) != 0xFF;
        mem_write32(&mut ec.ctx, MAILBOX + 12, if in_vehicle { 3 } else { 1 });

        let mut direct_angles: Option<(f32, f32)> = None;
        let mark = mem_read32(&mut ec.ctx, MAILBOX);
        if mark == MOUSE_LOOK || mark == MOUSE_LOOK_RECENTER || mark == MOUSE_LOOK_HOLD {
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

            let mut holding = mark == MOUSE_LOOK_HOLD;
            if mark == MOUSE_LOOK_RECENTER {
                mem_write32(&mut ec.ctx, MAILBOX, MOUSE_LOOK_HOLD);
                holding = true;
                ec.ztarget.locked_entity = if px.is_finite() && py.is_finite() && pz.is_finite() {
                    find_best_target(
                        &mut ec.ctx,
                        pit_entity,
                        px,
                        py,
                        pz,
                        cur_yaw,
                        cur_pitch,
                        min_dist,
                        ZTARGET_MAX_RANGE,
                    )
                } else {
                    0
                };
                ec.ztarget.fallback_yaw = pit_yaw;
                ec.ztarget.active = 1;
                ec.ztarget.snap_ticks = ZTARGET_TAP_TICKS;
            } else if !holding && ec.ztarget.active != 0 {
                if across.abs() > ZTARGET_CANCEL_STICK_PX || down.abs() > ZTARGET_CANCEL_STICK_PX {
                    ec.ztarget = ZTargetState::default();
                }
            }

            if ec.ztarget.active != 0 && (holding || ec.ztarget.snap_ticks > 0) {
                let mut goal_yaw = ec.ztarget.fallback_yaw;
                let mut goal_pitch = 0.0_f32;
                let mut have_goal = false;

                if ec.ztarget.locked_entity != 0 {
                    let range = if holding { ZTARGET_HOLD_MAX_RANGE } else { ZTARGET_MAX_RANGE };
                    let live = if px.is_finite() && py.is_finite() && pz.is_finite() {
                        is_live_target(
                            &mut ec.ctx,
                            ec.ztarget.locked_entity,
                            pit_entity,
                            px,
                            py,
                            pz,
                            min_dist,
                            range,
                        )
                    } else {
                        None
                    };
                    if let Some((tx, ty, tz)) = live {
                        let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
                        goal_yaw = dx.atan2(dz);
                        goal_pitch = (-dy.atan2(dx.hypot(dz))).clamp(-ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
                        have_goal = true;
                    } else if holding && px.is_finite() && py.is_finite() && pz.is_finite() {
                        ec.ztarget.locked_entity = find_best_target(
                            &mut ec.ctx,
                            pit_entity,
                            px,
                            py,
                            pz,
                            cur_yaw,
                            cur_pitch,
                            min_dist,
                            ZTARGET_MAX_RANGE,
                        );
                        let next_center = if ec.ztarget.locked_entity != 0 {
                            get_target_center(&mut ec.ctx, ec.ztarget.locked_entity)
                        } else {
                            None
                        };
                        if let Some((tx, ty, tz)) = next_center {
                            let (dx, dy, dz) = (tx - px, ty - py, tz - pz);
                            goal_yaw = dx.atan2(dz);
                            goal_pitch = (-dy.atan2(dx.hypot(dz))).clamp(-ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
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
                    let dyaw = slew_step(yaw_err);
                    let dpitch = slew_step(pitch_err);

                    if valid_heap_ptr(ctrl) {
                        mem_write32(&mut ec.ctx, ctrl + 0x24, 0);
                        mem_write32(&mut ec.ctx, ctrl + 0x38, 0);
                    }

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
            } else {
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
                apply_cam_angles(&mut ec.ctx, cam, new_yaw, new_pitch);
            }
        }
    }
}
