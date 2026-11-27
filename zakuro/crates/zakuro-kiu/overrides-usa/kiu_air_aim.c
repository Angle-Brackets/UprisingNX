/* Kid Icarus: Uprising (USA, 0004000000030100) — air-battle reticle delta override
 * and on-foot crosshair centering for PC Mouse + Keyboard and Gamepad.
 *
 * - In Air Battle / Orbit phases ([reticle + 0x81] == 0) when MAILBOX holds
 *   "MAM1" / "MAMR", suppresses KIU's idle auto-centering & Pit/orbit screen-follow
 *   drift and applies raw 1:1 (dx, dy) deltas directly to [reticle + 0xA4, 0xA8].
 * - In Land Battle ([reticle + 0x81] != 0) when MAILBOX holds "MLK1"/"MLKR"/"MLKH"/
 *   "MLKC"/"MLRC", leaves MAILBOX + 4/8 untouched for 0x007462A4 (1:1 camera mouse-look)
 *   and holds the crosshair cleanly at KIU's neutral center (200, 120 * (1 - ratio)).
 */
#include "overrides.h"
#include <math.h>
#include <string.h>

#define MAILBOX              0x07000000u
#define MOUSE_AIM            0x314D414Du /* "MAM1" */
#define MOUSE_AIM_RECENTER   0x524D414Du /* "MAMR" */
#define MOUSE_LOOK           0x314B4C4Du /* "MLK1" */
#define MOUSE_LOOK_RECENTER  0x524B4C4Du /* "MLKR" */
#define MOUSE_LOOK_HOLD      0x484B4C4Du /* "MLKH" */
#define MOUSE_LOOK_CENTER    0x434B4C4Du /* "MLKC" */
#define MOUSE_RETICLE_CENTER 0x43524C4Du /* "MLRC" */

#define AIR_RETICLE_VTABLE   0x00CF1B40u

static inline float bits_to_f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint32_t f_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int valid_heap_ptr(uint32_t p) {
    return p >= 0x08000000u && p < 0x18000000u && (p & 3u) == 0u;
}

static float read_crosshair_ratio(Context *ctx, uint32_t cam_mgr) {
    if (!valid_heap_ptr(cam_mgr)) {
        return 0.0f;
    }
    if (mem_read8(ctx, cam_mgr + 0xC08u) == 1u && mem_read8(ctx, cam_mgr + 0xC04u) == 1u) {
        return 0.0f;
    }
    float vfov = bits_to_f(mem_read32(ctx, cam_mgr + 0x358u));
    float offset = bits_to_f(mem_read32(ctx, cam_mgr + 0x44Cu));
    if (!isfinite(vfov) || !isfinite(offset) || vfov <= 1e-4f) {
        return 0.0f;
    }
    return clampf(offset / (vfov * 0.5f), -0.2f, 0.3f);
}

RECOMP_OVERRIDE(0x006D269C) {
    uint32_t reticle = ctx->r[0];
    uint32_t player = ctx->r[2];

    int preserve = 0;
    uint32_t orig_x = 0, orig_y = 0;

    if (valid_heap_ptr(reticle) && mem_read8(ctx, reticle + 0xD5u) == 0u) {
        uint32_t mark = mem_read32(ctx, MAILBOX);
        if (mark == MOUSE_AIM || mark == MOUSE_AIM_RECENTER) {
            /* Suppress KIU's 0.5s idle auto-centering while mouse/stick aim is active. */
            mem_write32(ctx, reticle + 0x208u, 0);
            if (valid_heap_ptr(player)) {
                uint32_t p_aim = player + 0x98u;
                mem_write32(ctx, p_aim + 0xD8u, 0);
                mem_write32(ctx, p_aim + 0xDCu, 0x3F000000u); /* 0.5f */
                mem_write8(ctx, p_aim + 0xE0u, 0);
            }
            /* Zero Pit/orbit screen-follow velocity so rail turns & Ch. 25 orbit
             * never drag the crosshair off target. */
            if (mem_read32(ctx, reticle) == AIR_RETICLE_VTABLE) {
                mem_write32(ctx, reticle + 0x2F4u, 0);
                mem_write32(ctx, reticle + 0x2F8u, 0);
            }
            if (mem_read8(ctx, reticle + 0x1D0u) == 0u) {
                orig_x = mem_read32(ctx, reticle + 0xA4u);
                orig_y = mem_read32(ctx, reticle + 0xA8u);
                preserve = 1;
            }
        }
    }

    CALL(RECOMP_ORIGINAL(0x006D269C));

    if (preserve) {
        mem_write32(ctx, reticle + 0xA4u, orig_x);
        mem_write32(ctx, reticle + 0xA8u, orig_y);
    }
}

RECOMP_OVERRIDE(0x006D2ABC) {
    uint32_t reticle = ctx->r[0];
    uint32_t cam_mgr = ctx->r[1];
    float xmin = bits_to_f(ctx->vfp[0]);
    float ymin = bits_to_f(ctx->vfp[1]);
    float xmax = bits_to_f(ctx->vfp[2]);
    float ymax = bits_to_f(ctx->vfp[3]);

    CALL(RECOMP_ORIGINAL(0x006D2ABC));

    if (!valid_heap_ptr(reticle)) {
        return;
    }

    /* Report GAME_FLYING only for full-screen Air/Orbit modes without clobbering Land/Vehicle. */
    if (mem_read8(ctx, reticle + 0x81u) == 0u) {
        mem_write32(ctx, MAILBOX + 12, 2u);
    }

    if (mem_read8(ctx, reticle + 0xD5u) != 0u) {
        return;
    }

    uint32_t mark = mem_read32(ctx, MAILBOX);

    /* On foot / vehicle: keep MAILBOX + 4/8 for 0x007462A4 (1:1 camera mouse-look)
     * and hold the reticle centered at KIU's neutral crosshair position. */
    if (mark == MOUSE_LOOK || mark == MOUSE_LOOK_RECENTER || mark == MOUSE_LOOK_HOLD ||
        mark == MOUSE_LOOK_CENTER || mark == MOUSE_RETICLE_CENTER) {
        if (mark == MOUSE_RETICLE_CENTER) {
            mem_write32(ctx, MAILBOX, MOUSE_LOOK);
            mem_write32(ctx, MAILBOX + 4, 0);
            mem_write32(ctx, MAILBOX + 8, 0);
        }
        uint32_t flags84 = mem_read32(ctx, reticle + 0x84u);
        mem_write32(ctx, reticle + 0x84u, flags84 | 2u);
        float cx_raw = bits_to_f(mem_read32(ctx, reticle + 0xE4u));
        float cy_base = bits_to_f(mem_read32(ctx, reticle + 0xECu));
        float cx = (isfinite(cx_raw) && cx_raw > 0.0f && cx_raw < 400.0f) ? cx_raw : 200.0f;
        float base_y = (isfinite(cy_base) && cy_base > 0.0f && cy_base < 240.0f) ? cy_base : 120.0f;
        float ratio = read_crosshair_ratio(ctx, cam_mgr);
        float cy = clampf(base_y * (1.0f - ratio), 0.0f, 240.0f);
        mem_write32(ctx, reticle + 0xA4u, f_to_bits(cx));
        mem_write32(ctx, reticle + 0xA8u, f_to_bits(cy));
        mem_write32(ctx, reticle + 0x208u, 0);
        mem_write32(ctx, reticle + 0x20Cu, 0);
        return;
    }

    if (mark != MOUSE_AIM && mark != MOUSE_AIM_RECENTER) {
        return;
    }

    uint32_t flags84 = mem_read32(ctx, reticle + 0x84u);
    mem_write32(ctx, reticle + 0x84u, flags84 | 2u);

    float dx = bits_to_f(mem_read32(ctx, MAILBOX + 4));
    float dy = bits_to_f(mem_read32(ctx, MAILBOX + 8));
    if (dx != 0.0f || dy != 0.0f) {
        mem_write32(ctx, MAILBOX + 4, 0);
        mem_write32(ctx, MAILBOX + 8, 0);
    }

    float x = bits_to_f(mem_read32(ctx, reticle + 0xA4u));
    float y = bits_to_f(mem_read32(ctx, reticle + 0xA8u));

    if (mark == MOUSE_AIM_RECENTER) {
        mem_write32(ctx, MAILBOX, MOUSE_AIM);
        float cx = bits_to_f(mem_read32(ctx, reticle + 0x2ECu));
        float cy = bits_to_f(mem_read32(ctx, reticle + 0x2F0u));
        x = (isfinite(cx) && cx > 0.0f && cx < 400.0f) ? cx : 200.0f;
        y = (isfinite(cy) && cy > 0.0f && cy < 240.0f) ? cy : 120.5f;
        mem_write32(ctx, reticle + 0x208u, 0);
        mem_write32(ctx, reticle + 0x20Cu, 0);
    } else if (isfinite(dx) && isfinite(dy)) {
        x += dx;
        y += dy;
    }

    if (isfinite(xmin) && isfinite(xmax) && xmax > xmin) {
        x = clampf(x, xmin, xmax);
    } else {
        x = clampf(x, 0.0f, 400.0f);
    }
    if (isfinite(ymin) && isfinite(ymax) && ymax > ymin) {
        y = clampf(y, ymin, ymax);
    } else {
        y = clampf(y, 0.0f, 240.0f);
    }

    mem_write32(ctx, reticle + 0xA4u, f_to_bits(x));
    mem_write32(ctx, reticle + 0xA8u, f_to_bits(y));
}
