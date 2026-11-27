/*
 * Mouse look, behind-Pit recenter, and gamepad Z-target on foot/vehicle
 * for Kid Icarus: Uprising (USA, 0004000000030100).
 *
 * Mailbox protocol (HOST_MAILBOX_VADDR = 0x07000000):
 *   - "MLK1": direct 1:1 free-look (mouse or gamepad right stick).
 *   - "MLKR": gamepad R rising edge — acquire best targetable enemy in range
 *             or snap camera behind Pit if none is in range.
 *   - "MLKH": gamepad R held — smoothly track the locked enemy.
 *   - "MLKC": gamepad L3 — dedicated behind-Pit recenter.
 */
#include "overrides.h"
#include <math.h>
#include <string.h>

#define MAILBOX             0x07000000u /* Zakuro's HOST_MAILBOX_VADDR */
#define MOUSE_LOOK          0x314B4C4Du /* "MLK1" */
#define MOUSE_LOOK_RECENTER 0x524B4C4Du /* "MLKR" */
#define MOUSE_LOOK_HOLD     0x484B4C4Du /* "MLKH" */
#define MOUSE_LOOK_CENTER   0x434B4C4Du /* "MLKC" */

#define YAW_PER_PIXEL   (-0.00408f)
#define PITCH_PER_PIXEL 0.00442f

#define ENEMY_LIST_GUARD   0x00D63710u
#define ENEMY_LIST_HEAD    0x00F25A78u
#define FIGHTER_LIST_GUARD 0x00D63708u
#define FIGHTER_LIST_HEAD  0x00F25A68u
#define TARGET_COMP_KEY    0x00D76244u

#define PI_F                     3.14159265358979323846f
#define ZTARGET_MAX_RANGE        160.0f
#define ZTARGET_HOLD_MAX_RANGE   220.0f
#define ZTARGET_MIN_DIST_FOOT    0.6f
#define ZTARGET_MIN_DIST_VEHICLE 1.8f
#define ZTARGET_FRONT_WEIGHT     0.65f
#define ZTARGET_TAP_TICKS        14
#define ZTARGET_SLEW_GAIN        0.65f
#define ZTARGET_MIN_STEP         0.09f
#define ZTARGET_MAX_STEP         1.00f
#define ZTARGET_DONE_EPS         0.01f
#define ZTARGET_PITCH_LIMIT      0.75f
#define ZTARGET_CANCEL_STICK_PX  1.5f

static uint32_t z_locked_entity = 0;
static int z_active = 0;
static int z_snap_ticks = 0;
static float z_fallback_yaw = 0.0f;

static inline float as_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static inline float read_f32(Context *ctx, uint32_t addr) {
    return as_float(mem_read32(ctx, addr));
}

static inline int valid_heap_ptr(uint32_t p) {
    return p >= 0x08000000u && p < 0x18000000u && (p & 3u) == 0u;
}

static inline float wrap_pi(float a) {
    if (!isfinite(a)) {
        return 0.0f;
    }
    return remainderf(a, 2.0f * PI_F);
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float slew_step(float err) {
    float a = fabsf(err);
    if (a <= ZTARGET_MIN_STEP) {
        return err;
    }
    float step = a * ZTARGET_SLEW_GAIN;
    if (step < ZTARGET_MIN_STEP) {
        step = ZTARGET_MIN_STEP;
    } else if (step > ZTARGET_MAX_STEP) {
        step = ZTARGET_MAX_STEP;
    }
    return err < 0.0f ? -step : step;
}

static float read_crosshair_pitch_offset(Context *ctx, uint32_t cam) {
    if (cam < 0x08000278u) {
        return 0.0f;
    }
    uint32_t cam_mgr = cam - 0x278u;
    if (!valid_heap_ptr(cam_mgr)) {
        return 0.0f;
    }
    if (mem_read8(ctx, cam_mgr + 0xC08u) == 1u && mem_read8(ctx, cam_mgr + 0xC04u) == 1u) {
        return 0.0f;
    }
    float vfov_raw = read_f32(ctx, cam_mgr + 0x358u);
    float offset_raw = read_f32(ctx, cam_mgr + 0x44Cu);
    if (!isfinite(vfov_raw) || !isfinite(offset_raw) || vfov_raw <= 1e-4f) {
        return 0.0f;
    }
    float ratio = clampf(offset_raw / (vfov_raw * 0.5f), -0.2f, 0.3f);
    if (ratio == 0.0f) {
        return 0.0f;
    }
    float vfov_rad = vfov_raw > 3.2f ? (vfov_raw * PI_F / 180.0f) : vfov_raw;
    if (vfov_rad < 0.25f || vfov_rad > 2.2f) {
        vfov_rad = 60.0f * PI_F / 180.0f;
    }
    return atanf(ratio * tanf(vfov_rad * 0.5f));
}

static uint32_t find_target_holder(Context *ctx, uint32_t entity) {
    uint32_t holder = mem_read32(ctx, entity + 0x1F0u);
    if (valid_heap_ptr(holder)) {
        return holder;
    }
    uint32_t end_node = entity + 0x1A8u;
    uint32_t cur = mem_read32(ctx, entity + 0x1ACu);
    uint32_t best = end_node;
    for (int steps = 0; steps < 64 && valid_heap_ptr(cur); steps++) {
        if (mem_read8(ctx, cur + 0x10u) == 0u) {
            break;
        }
        uint32_t k = mem_read32(ctx, cur + 0x14u);
        if (k < TARGET_COMP_KEY) {
            cur = mem_read32(ctx, cur + 0xCu);
        } else {
            best = cur;
            cur = mem_read32(ctx, cur + 0x8u);
        }
    }
    if (best == end_node || !valid_heap_ptr(best)) {
        return 0;
    }
    if (mem_read32(ctx, best + 0x14u) > TARGET_COMP_KEY) {
        return 0;
    }
    uint32_t found = mem_read32(ctx, best + 0x0u);
    uint32_t end_val = mem_read32(ctx, end_node);
    if (found == end_val || !valid_heap_ptr(found)) {
        return 0;
    }
    return found;
}

static int get_target_center(Context *ctx, uint32_t entity, float *ox, float *oy, float *oz) {
    if (!valid_heap_ptr(entity)) {
        return 0;
    }
    uint32_t holder = find_target_holder(ctx, entity);
    if (!valid_heap_ptr(holder)) {
        return 0;
    }
    uint32_t comp = mem_read32(ctx, holder + 0x20u);
    if (!valid_heap_ptr(comp)) {
        return 0;
    }
    if ((mem_read32(ctx, comp + 0x38u) & 1u) == 0u) {
        return 0;
    }
    if ((mem_read32(ctx, comp + 0x34u) & 4u) == 0u) {
        return 0;
    }
    float x0 = read_f32(ctx, comp + 0x54u);
    float y0 = read_f32(ctx, comp + 0x58u);
    float z0 = read_f32(ctx, comp + 0x5Cu);
    float x1 = read_f32(ctx, comp + 0x60u);
    float y1 = read_f32(ctx, comp + 0x64u);
    float z1 = read_f32(ctx, comp + 0x68u);
    if (!isfinite(x0) || !isfinite(y0) || !isfinite(z0) ||
        !isfinite(x1) || !isfinite(y1) || !isfinite(z1)) {
        return 0;
    }
    *ox = (x0 + x1) * 0.5f;
    *oy = (y0 + y1) * 0.5f;
    *oz = (z0 + z1) * 0.5f;
    return 1;
}

static int entity_in_list(Context *ctx, uint32_t guard_addr, uint32_t head_addr, uint32_t needle) {
    if ((mem_read32(ctx, guard_addr) & 1u) == 0u) {
        return 0;
    }
    uint32_t sentinel = mem_read32(ctx, head_addr);
    uint32_t node = mem_read32(ctx, head_addr + 4u);
    for (int i = 0; i < 256 && valid_heap_ptr(node); i++) {
        uint32_t ent = mem_read32(ctx, node);
        if (ent == sentinel) {
            break;
        }
        if (ent == needle) {
            return 1;
        }
        node = mem_read32(ctx, node + 4u);
    }
    return 0;
}

static int is_live_target(Context *ctx, uint32_t entity, uint32_t pit_entity,
                          float px, float py, float pz, float min_dist, float max_dist,
                          float *ox, float *oy, float *oz) {
    if (!valid_heap_ptr(entity) || entity == pit_entity) {
        return 0;
    }
    if (!entity_in_list(ctx, ENEMY_LIST_GUARD, ENEMY_LIST_HEAD, entity) &&
        !entity_in_list(ctx, FIGHTER_LIST_GUARD, FIGHTER_LIST_HEAD, entity)) {
        return 0;
    }
    float tx, ty, tz;
    if (!get_target_center(ctx, entity, &tx, &ty, &tz)) {
        return 0;
    }
    float dx = tx - px, dy = ty - py, dz = tz - pz;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < min_dist * min_dist || d2 > max_dist * max_dist) {
        return 0;
    }
    *ox = tx;
    *oy = ty;
    *oz = tz;
    return 1;
}

static void scan_entity_list(Context *ctx, uint32_t guard_addr, uint32_t head_addr,
                             uint32_t pit_entity, float px, float py, float pz, float cur_yaw,
                             float min_dist, float max_dist,
                             uint32_t *best_entity, float *best_score) {
    if ((mem_read32(ctx, guard_addr) & 1u) == 0u) {
        return;
    }
    uint32_t sentinel = mem_read32(ctx, head_addr);
    uint32_t node = mem_read32(ctx, head_addr + 4u);
    float min_d2 = min_dist * min_dist;
    float max_d2 = max_dist * max_dist;
    for (int i = 0; i < 256 && valid_heap_ptr(node); i++) {
        uint32_t ent = mem_read32(ctx, node);
        if (ent == sentinel) {
            break;
        }
        node = mem_read32(ctx, node + 4u);
        if (ent == pit_entity) {
            continue;
        }
        float tx, ty, tz;
        if (!get_target_center(ctx, ent, &tx, &ty, &tz)) {
            continue;
        }
        float dx = tx - px, dy = ty - py, dz = tz - pz;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < min_d2 || d2 > max_d2) {
            continue;
        }
        float dist = sqrtf(d2);
        float ang_err = fabsf(wrap_pi(atan2f(dx, dz) - cur_yaw));
        float score = dist * (1.0f + ZTARGET_FRONT_WEIGHT * (ang_err / PI_F));
        if (score < *best_score) {
            *best_score = score;
            *best_entity = ent;
        }
    }
}

static uint32_t find_best_target(Context *ctx, uint32_t pit_entity,
                                 float px, float py, float pz, float cur_yaw,
                                 float min_dist, float max_dist) {
    uint32_t best = 0;
    float best_score = 1e30f;
    scan_entity_list(ctx, ENEMY_LIST_GUARD, ENEMY_LIST_HEAD,
                     pit_entity, px, py, pz, cur_yaw, min_dist, max_dist, &best, &best_score);
    scan_entity_list(ctx, FIGHTER_LIST_GUARD, FIGHTER_LIST_HEAD,
                     pit_entity, px, py, pz, cur_yaw, min_dist, max_dist, &best, &best_score);
    return best;
}

RECOMP_OVERRIDE(0x007462A4) {
    uint32_t ctrl = ctx->r[0];
    uint32_t cam = ctx->r[1];
    int cam_ok = (cam >= 0x08000200u && cam < 0x18000000u && (cam & 3u) == 0u);
    uint32_t in_vehicle = cam_ok && (mem_read8(ctx, cam - 0xFFu) != 0xFFu);
    mem_write32(ctx, MAILBOX + 12, in_vehicle ? 3u : 1u);

    uint32_t mark = mem_read32(ctx, MAILBOX);
    if (mark == MOUSE_LOOK || mark == MOUSE_LOOK_RECENTER ||
        mark == MOUSE_LOOK_HOLD || mark == MOUSE_LOOK_CENTER) {
        float across = as_float(mem_read32(ctx, MAILBOX + 4));
        float down = as_float(mem_read32(ctx, MAILBOX + 8));
        mem_write32(ctx, MAILBOX + 4, 0);
        mem_write32(ctx, MAILBOX + 8, 0);
        if (!isfinite(across)) across = 0.0f;
        if (!isfinite(down)) down = 0.0f;

        if (!cam_ok) {
            z_active = 0;
            z_locked_entity = 0;
            z_snap_ticks = 0;
            vfp_set_s(ctx, 0, vfp_s(ctx, 0) + across * YAW_PER_PIXEL);
            vfp_set_s(ctx, 1, vfp_s(ctx, 1) + down * PITCH_PER_PIXEL);
            CALL(RECOMP_ORIGINAL(0x007462A4));
            return;
        }

        uint32_t pit_entity = mem_read32(ctx, cam - 0x1B8u);
        float pit_yaw = read_f32(ctx, cam - 0x178u);
        float px = read_f32(ctx, cam + 0x24u);
        float py = read_f32(ctx, cam + 0x28u);
        float pz = read_f32(ctx, cam + 0x2Cu);
        float cur_yaw = read_f32(ctx, cam + 0x34u);
        float cur_pitch = read_f32(ctx, cam + 0x38u);
        float pitch_off = read_crosshair_pitch_offset(ctx, cam);
        float min_dist = in_vehicle ? ZTARGET_MIN_DIST_VEHICLE : ZTARGET_MIN_DIST_FOOT;

        if (!isfinite(cur_yaw)) cur_yaw = 0.0f;
        if (!isfinite(cur_pitch)) cur_pitch = 0.0f;
        if (!isfinite(pit_yaw)) pit_yaw = cur_yaw;

        int holding = (mark == MOUSE_LOOK_HOLD);
        if (mark == MOUSE_LOOK_CENTER) {
            mem_write32(ctx, MAILBOX, MOUSE_LOOK);
            z_locked_entity = 0;
            z_fallback_yaw = pit_yaw;
            z_active = 1;
            z_snap_ticks = ZTARGET_TAP_TICKS;
        } else if (mark == MOUSE_LOOK_RECENTER) {
            mem_write32(ctx, MAILBOX, MOUSE_LOOK_HOLD);
            holding = 1;
            z_locked_entity = (isfinite(px) && isfinite(py) && isfinite(pz))
                ? find_best_target(ctx, pit_entity, px, py, pz, cur_yaw, min_dist, ZTARGET_MAX_RANGE)
                : 0;
            z_fallback_yaw = pit_yaw;
            z_active = 1;
            z_snap_ticks = ZTARGET_TAP_TICKS;
        } else if (!holding && z_active) {
            if (fabsf(across) > ZTARGET_CANCEL_STICK_PX || fabsf(down) > ZTARGET_CANCEL_STICK_PX) {
                z_active = 0;
                z_locked_entity = 0;
                z_snap_ticks = 0;
            }
        }

        int applied_z = 0;
        if (z_active && (holding || z_snap_ticks > 0)) {
            float goal_yaw = z_fallback_yaw;
            float goal_pitch = 0.0f;
            int have_goal = 0;

            if (z_locked_entity != 0u) {
                float tx, ty, tz;
                float range = holding ? ZTARGET_HOLD_MAX_RANGE : ZTARGET_MAX_RANGE;
                if (isfinite(px) && isfinite(py) && isfinite(pz) &&
                    is_live_target(ctx, z_locked_entity, pit_entity, px, py, pz, min_dist, range, &tx, &ty, &tz)) {
                    float dx = tx - px, dy = ty - py, dz = tz - pz;
                    goal_yaw = atan2f(dx, dz);
                    goal_pitch = clampf(-atan2f(dy, hypotf(dx, dz)) + pitch_off, -ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
                    have_goal = 1;
                } else if (holding && isfinite(px) && isfinite(py) && isfinite(pz)) {
                    z_locked_entity = find_best_target(ctx, pit_entity, px, py, pz, cur_yaw, min_dist, ZTARGET_MAX_RANGE);
                    if (z_locked_entity != 0u &&
                        get_target_center(ctx, z_locked_entity, &tx, &ty, &tz)) {
                        float dx = tx - px, dy = ty - py, dz = tz - pz;
                        goal_yaw = atan2f(dx, dz);
                        goal_pitch = clampf(-atan2f(dy, hypotf(dx, dz)) + pitch_off, -ZTARGET_PITCH_LIMIT, ZTARGET_PITCH_LIMIT);
                        have_goal = 1;
                    } else {
                        z_active = 0;
                        z_snap_ticks = 0;
                    }
                } else {
                    z_active = 0;
                    z_locked_entity = 0;
                    z_snap_ticks = 0;
                }
            } else {
                have_goal = 1;
            }

            if (have_goal) {
                float yaw_err = wrap_pi(goal_yaw - cur_yaw);
                float pitch_err = goal_pitch - cur_pitch;
                float dyaw = slew_step(yaw_err);
                float dpitch = slew_step(pitch_err);

                if (valid_heap_ptr(ctrl)) {
                    mem_write32(ctx, ctrl + 0x24u, 0);
                    mem_write32(ctx, ctrl + 0x38u, 0);
                }

                vfp_set_s(ctx, 0, dyaw);
                vfp_set_s(ctx, 1, dpitch);
                applied_z = 1;

                int close_enough = fabsf(yaw_err) < ZTARGET_DONE_EPS && fabsf(pitch_err) < ZTARGET_DONE_EPS;
                if (z_snap_ticks > 0) {
                    z_snap_ticks--;
                }
                if (!holding) {
                    if (close_enough || z_snap_ticks <= 0) {
                        z_active = 0;
                        z_locked_entity = 0;
                        z_snap_ticks = 0;
                    }
                } else if (z_locked_entity == 0u && (close_enough || z_snap_ticks <= 0)) {
                    z_active = 0;
                    z_snap_ticks = 0;
                }
            }
        } else {
            z_active = 0;
            z_locked_entity = 0;
            z_snap_ticks = 0;
        }

        if (!applied_z) {
            vfp_set_s(ctx, 0, vfp_s(ctx, 0) + across * YAW_PER_PIXEL);
            vfp_set_s(ctx, 1, vfp_s(ctx, 1) + down * PITCH_PER_PIXEL);
        }
    } else {
        z_active = 0;
        z_locked_entity = 0;
        z_snap_ticks = 0;
    }

    CALL(RECOMP_ORIGINAL(0x007462A4));
}
