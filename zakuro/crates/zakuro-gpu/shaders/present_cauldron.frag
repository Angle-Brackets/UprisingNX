#version 450
// Fiend's Cauldron 16:9 Unified Mural Stage + Smooth Cross-Dissolve + Floating Intensity Cursor.
// Compiled via naga (glsl-in -> spv-out):
//   binding 0: top_screen (SAMPLED_IMAGE)
//   binding 1: bot_screen (SAMPLED_IMAGE)
//   binding 2: screen_sampler (SAMPLER)

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D top_screen;
layout(set = 0, binding = 1) uniform texture2D bot_screen;
layout(set = 0, binding = 2) uniform sampler screen_sampler;

layout(push_constant) uniform CauldronCrop {
    vec4 top_area;
    vec4 top_bounds;
    vec4 bot_area;
    vec4 bot_bounds;
    float cursor_x;
    float cursor_opacity;
    int hud_visible;
    int mode;
} crop;

vec3 sample_top(vec2 local_uv, vec2 local_duv) {
    vec2 at = crop.top_area.xy + local_uv * crop.top_area.zw;
    vec2 size = vec2(textureSize(sampler2D(top_screen, screen_sampler), 0));
    vec2 covered = abs(local_duv * crop.top_area.zw) * size;
    if (max(covered.x, covered.y) <= 1.001) {
        vec2 texel = at * size;
        if (crop.mode == 1) {
            at = (floor(texel) + 0.5) / size;
        } else if (crop.mode == 2) {
            vec2 scale = 1.0 / max(covered, vec2(1e-6));
            vec2 centred = texel - 0.5;
            vec2 blend = clamp((fract(centred) - 0.5) * scale + 0.5, 0.0, 1.0);
            at = (floor(centred) + 0.5 + blend) / size;
        }
        return textureLod(sampler2D(top_screen, screen_sampler), clamp(at, crop.top_bounds.xy, crop.top_bounds.zw), 0.0).rgb;
    }
    ivec2 taps = ivec2(clamp(ceil(covered), 1.0, 4.0));
    vec3 sum = vec3(0.0);
    for (int y = 0; y < taps.y; y++) {
        for (int x = 0; x < taps.x; x++) {
            vec2 offset = ((vec2(x, y) + 0.5) / vec2(taps) - 0.5) * covered / size;
            sum += textureLod(sampler2D(top_screen, screen_sampler), clamp(at + offset, crop.top_bounds.xy, crop.top_bounds.zw), 0.0).rgb;
        }
    }
    return sum / float(taps.x * taps.y);
}

vec3 sample_bot(vec2 local_uv, vec2 local_duv) {
    vec2 at = crop.bot_area.xy + local_uv * crop.bot_area.zw;
    vec2 size = vec2(textureSize(sampler2D(bot_screen, screen_sampler), 0));
    vec2 covered = abs(local_duv * crop.bot_area.zw) * size;
    if (max(covered.x, covered.y) <= 1.001) {
        vec2 texel = at * size;
        if (crop.mode == 1) {
            at = (floor(texel) + 0.5) / size;
        } else if (crop.mode == 2) {
            vec2 scale = 1.0 / max(covered, vec2(1e-6));
            vec2 centred = texel - 0.5;
            vec2 blend = clamp((fract(centred) - 0.5) * scale + 0.5, 0.0, 1.0);
            at = (floor(centred) + 0.5 + blend) / size;
        }
        return textureLod(sampler2D(bot_screen, screen_sampler), clamp(at, crop.bot_bounds.xy, crop.bot_bounds.zw), 0.0).rgb;
    }
    ivec2 taps = ivec2(clamp(ceil(covered), 1.0, 4.0));
    vec3 sum = vec3(0.0);
    for (int y = 0; y < taps.y; y++) {
        for (int x = 0; x < taps.x; x++) {
            vec2 offset = ((vec2(x, y) + 0.5) / vec2(taps) - 0.5) * covered / size;
            sum += textureLod(sampler2D(bot_screen, screen_sampler), clamp(at + offset, crop.bot_bounds.xy, crop.bot_bounds.zw), 0.0).rgb;
        }
    }
    return sum / float(taps.x * taps.y);
}

vec3 sample_top_fast(vec2 local_uv) {
    vec2 at = clamp(crop.top_area.xy + local_uv * crop.top_area.zw, crop.top_bounds.xy, crop.top_bounds.zw);
    return textureLod(sampler2D(top_screen, screen_sampler), at, 0.0).rgb;
}

vec3 sample_bot_fast(vec2 local_uv) {
    vec2 at = clamp(crop.bot_area.xy + local_uv * crop.bot_area.zw, crop.bot_bounds.xy, crop.bot_bounds.zw);
    return textureLod(sampler2D(bot_screen, screen_sampler), at, 0.0).rgb;
}

float max3(vec3 c) {
    return max(c.r, max(c.g, c.b));
}

float sd_round_box(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

float sd_segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    return length(pa - ba * h);
}

float sd_chevron_x(vec2 p, float dir) {
    float u = p.x * dir;
    float half_h = 4.9 * (1.0 - u / 5.0);
    float dx_base = -u;
    float dy_side = (abs(p.y) - half_h) * 0.714;
    float dx_tip = u - 5.0;
    return max(dx_base, max(dy_side, dx_tip));
}

float sd_pointer_up(vec2 p) {
    float u = -p.y;
    float half_w = 4.4 * (1.0 - u / 8.6);
    float dy_base = -u;
    float dx_side = (abs(p.x) - half_w) * 0.888;
    float dy_tip = u - 8.6;
    return max(dy_base, max(dx_side, dy_tip));
}

// Reconstructs the exact cartoon silhouette (cream interior + ~1.7px dark outline)
// of the parked Total Hearts pouch (`[16..102.4, 166..233.8]` on `#000000` background).
float pouch_alpha(vec2 t_px) {
    if (t_px.x < 14.5 || t_px.x > 104.0 || t_px.y < 163.2 || t_px.y > 235.5) {
        return 0.0;
    }
    const vec2 inv = vec2(1.0 / 400.0, 1.0 / 240.0);
    float m = max3(sample_top_fast(t_px * inv));
    m = max(m, max3(sample_top_fast((t_px + vec2( 1.7,  0.0)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2(-1.7,  0.0)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2( 0.0,  1.7)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2( 0.0, -1.7)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2( 1.2,  1.2)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2(-1.2,  1.2)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2( 1.2, -1.2)) * inv)));
    m = max(m, max3(sample_top_fast((t_px + vec2(-1.2, -1.2)) * inv)));
    return smoothstep(0.08, 0.24, m);
}

void main() {
    const float W = 426.66666;
    const float H = 240.0;
    vec2 p = uv * vec2(W, H);
    vec2 dp = max(fwidth(p), vec2(0.05));
    float fw = dp.x;
    float px_scale = 1.0 / fw;
    float aa = fw;

    // 1. Dynamic obsidian-ember room stage (smoothly cross-dissolves slate-blue <-> crimson)
    uint uflags = uint(crop.hud_visible);
    bool hud_vis = (uflags & 1u) != 0u;
    uint packed_rgb = uflags >> 8u;
    vec3 room_bg = (packed_rgb != 0u)
        ? vec3(
            float(packed_rgb & 255u),
            float((packed_rgb >> 8u) & 255u),
            float((packed_rgb >> 16u) & 255u)
        ) / 255.0
        : vec3(0.13, 0.02, 0.015);
    vec2 n = uv * 2.0 - 1.0;
    float edge = clamp(length(vec2(n.x * 0.9, n.y * 0.5)) - 0.62, 0.0, 1.0);
    float top_glow = clamp(1.0 - p.y / 53.0, 0.0, 1.0) * 0.55;
    float warmth = clamp(edge * 0.85 + top_glow * 0.42, 0.0, 1.0);
    vec3 col = room_bg * warmth;

    // 2. Full Un-Occluded Parchment Mural (Winged Pit, Reapers & Treasure Chests 100% visible!)
    if (hud_vis) {
        vec2 m_min = vec2(19.5, 43.0);
        vec2 m_max = vec2(407.2, 203.5);
        vec2 m_size = m_max - m_min;
        vec2 m_center = 0.5 * (m_min + m_max);
        vec2 m_half = 0.5 * m_size;
        float d_mural = sd_round_box(p - m_center, m_half, 3.1);
        if (d_mural < aa) {
            vec2 m_uv = clamp((p - m_min) / m_size, 0.0, 1.0);
            vec2 t_px = vec2(30.0, 25.0) + m_uv * vec2(340.0, 136.0);
            vec2 m_duv = (dp / m_size) * (vec2(340.0, 136.0) / vec2(400.0, 240.0));
            vec3 mural_rgb = sample_top(t_px / vec2(400.0, 240.0), m_duv) * 0.88;

            // Gentle soft contact shadow right behind the iron pot belly so the Cauldron
            // feels grounded while Winged Pit & all center art stay bright and legible.
            float pot_dist = length((m_uv - vec2(0.5, 0.73)) / vec2(0.165, 0.235));
            float pot_shade = mix(0.75, 1.0, smoothstep(0.18, 1.0, pot_dist));
            vec3 staged = mural_rgb * pot_shade;
            float fill_a = clamp(0.5 - d_mural * px_scale, 0.0, 1.0);
            col = mix(col, staged, fill_a);
        }
        float border_d = abs(d_mural);
        if (border_d < 0.36 + aa) {
            float border_a = clamp(1.0 - (border_d - 0.33) * px_scale, 0.0, 1.0) * 0.84;
            col = mix(col, vec3(0.533, 0.345, 0.212), border_a);
        }

        // 3. Top instruction header ("Bet hearts..." / "Spend hearts...") — full 388x21 at 1:1 aspect
        vec2 h_min = vec2(96.9, 0.8);
        vec2 h_max = vec2(329.7, 13.4);
        vec2 h_size = h_max - h_min;
        if (p.x >= h_min.x && p.x <= h_max.x && p.y >= h_min.y && p.y <= h_max.y) {
            vec2 h_uv = (p - h_min) / h_size;
            vec2 t_px = vec2(6.0, 1.5) + h_uv * vec2(388.0, 21.0);
            vec2 h_duv = (dp / h_size) * (vec2(388.0, 21.0) / vec2(400.0, 240.0));
            vec3 h_col = sample_top(t_px / vec2(400.0, 240.0), h_duv);
            vec3 h_sh = sample_top_fast((t_px - vec2(0.0, 1.0)) / vec2(400.0, 240.0));
            float sh_a = smoothstep(0.08, 0.34, max3(h_sh));
            float h_a = smoothstep(0.08, 0.34, max3(h_col));
            col = mix(col, vec3(0.0), sh_a * 0.68);
            col = mix(col, h_col, h_a);
        }
    }

    // 4. Interactive Bottom Screen (0..9 slider, blazing Cauldron, Heart Pan, [Wager], [Move Out!])
    vec2 b_min = vec2(62.22222, 13.0);
    vec2 b_max = vec2(364.44444, 240.0);
    vec2 b_size = b_max - b_min;
    if (p.x >= b_min.x && p.x <= b_max.x && p.y >= b_min.y && p.y <= b_max.y) {
        vec2 b_uv = (p - b_min) / b_size;
        vec2 b_px = b_uv * vec2(320.0, 240.0);
        bool in_orig_back = (b_px.x < 38.0 && b_px.y > 202.0);
        vec2 sample_uv = in_orig_back ? vec2(1.0 - b_uv.x, b_uv.y) : b_uv;
        vec2 b_duv = dp / b_size;
        vec3 bot_col = sample_bot(sample_uv, b_duv);
        float bot_lum = max3(bot_col);
        float bot_min = min(bot_col.r, min(bot_col.g, bot_col.b));
        float chroma = bot_lum - bot_min;
        float fx = clamp(min(b_px.x / 18.0, (320.0 - b_px.x) / 18.0), 0.0, 1.0);
        float fy_top = clamp(b_px.y / 14.0, 0.0, 1.0);
        float fy_bot = (abs(b_px.x - 163.5) > 118.0) ? clamp((236.0 - b_px.y) / 18.0, 0.0, 1.0) : 1.0;
        float m_fade = fx * fy_top * fy_bot;

        if (!hud_vis) {
            // Heart-dump launch animation over dark stage
            vec3 bot_glow = bot_col * mix(m_fade * m_fade, 1.0, smoothstep(0.04, 0.12, bot_lum));
            col = 1.0 - (1.0 - col) * (1.0 - bot_glow);
            if (!in_orig_back) {
                bool in_launch_wager = (b_px.x >= 224.0 && b_px.y <= 30.0);
                bool in_moveout = (b_px.x >= 48.0 && b_px.x <= 279.0 && b_px.y >= 184.0);
                if (in_launch_wager || in_moveout) {
                    col = mix(col, bot_col, smoothstep(0.035, 0.13, bot_lum));
                }
            }
        } else {
            // Prep stage over bright visible Parchment Mural:
            // (a) Premultiplied Fire Over blend keeps crimson/orange/cyan flames rich and saturated
            //     over bright parchment without washing out to white or creating dark fringes.
            float fade_w = mix(m_fade * m_fade, 1.0, smoothstep(0.04, 0.12, bot_lum));
            float fire_a = smoothstep(0.035, 0.52, bot_lum) * fade_w;
            col = clamp(col * (1.0 - fire_a * 0.84) + bot_col * fade_w, 0.0, 1.0);

            // (b) Crisp 100%-solid 3D Cauldron pot, Copper Pan, Handles, Piled Hearts, Pointer Line & UI
            if (!in_orig_back) {
                bool in_top_strip = (b_px.x >= 8.0 && b_px.x <= 312.0 && b_px.y >= 0.0 && b_px.y <= 36.0);
                bool in_slider_core = (b_px.x >= 14.0 && b_px.x <= 303.0 && b_px.y >= 15.2 && b_px.y <= 28.8);
                bool in_wager = (b_px.x >= 221.0 && b_px.x <= 320.0 && b_px.y >= 143.5 && b_px.y <= 172.0);
                bool in_wager_core = (b_px.x >= 223.5 && b_px.x <= 318.5 && b_px.y >= 146.0 && b_px.y <= 169.5);
                bool in_moveout = (b_px.x >= 48.0 && b_px.x <= 279.0 && b_px.y >= 202.0)
                    || (b_px.x >= 72.0 && b_px.x <= 255.0 && b_px.y >= 184.0 && (bot_col.b > bot_col.r + 0.035 || bot_min > 0.20));

                bool in_pot = (length((b_px - vec2(160.0, 148.0)) / vec2(48.0, 38.5)) <= 1.0)
                    || (length((b_px - vec2(160.0, 117.0)) / vec2(51.5, 11.5)) <= 1.0);
                float pan_d = length((b_px - vec2(160.0, 97.5)) / vec2(65.0, 21.0));
                bool in_handles = (abs(b_px.x - 160.0) <= 86.5 && b_px.y >= 79.0 && b_px.y <= 95.5);
                bool in_hearts = (abs(b_px.x - 160.0) <= 74.0 && b_px.y >= 34.0 && b_px.y <= 98.0);
                bool in_pointer = (sd_segment(b_px, vec2(160.0, 96.0), vec2(226.0, 158.0)) <= 5.5);
                bool in_3d_core = in_pot || (pan_d <= 1.0) || in_handles;

                if (in_top_strip || in_wager || in_moveout || in_3d_core || in_hearts || in_pointer) {
                    float thresh_lo = (in_hearts && !in_3d_core) ? 0.12 : 0.045;
                    float thresh_hi = (in_hearts && !in_3d_core) ? 0.25 : 0.14;
                    float solid_a = (in_slider_core || in_wager_core)
                        ? 1.0
                        : smoothstep(thresh_lo, thresh_hi, bot_lum);
                    col = mix(col, bot_col, solid_a);
                }
            }
        }
    }

    // 5. Total Hearts Pouch crowning top-left of mural (sampled from parked [16..102.4, 166..233.8])
    if (hud_vis) {
        vec2 p_min = vec2(9.0, 41.5);
        vec2 p_max = vec2(82.5, 99.0);
        vec2 p_size = p_max - p_min;
        if (p.x >= p_min.x - 3.0 && p.x <= p_max.x + 3.0 && p.y >= p_min.y - 2.0 && p.y <= p_max.y + 4.0) {
            vec2 sh_uv = (p - vec2(0.0, 1.4) - p_min) / p_size;
            vec2 sh_t = vec2(16.0, 166.0) + sh_uv * vec2(86.4, 67.76);
            float sh_a = pouch_alpha(sh_t);
            col = mix(col, vec3(0.0), sh_a * 0.58);

            vec2 p_uv = (p - p_min) / p_size;
            vec2 t_px = vec2(16.0, 166.0) + p_uv * vec2(86.4, 67.76);
            vec2 p_duv = (dp / p_size) * (vec2(86.4, 67.76) / vec2(400.0, 240.0));
            vec3 p_col = sample_top(t_px / vec2(400.0, 240.0), p_duv);
            float p_a = pouch_alpha(t_px);
            col = mix(col, p_col, p_a);
        }
    }

    // 6. Back button tucked cleanly into 16:9 bottom-left corner
    vec2 bk_min = vec2(10.5, 201.0);
    vec2 bk_max = vec2(43.0, 232.0);
    vec2 bk_size = bk_max - bk_min;
    if (p.x >= bk_min.x - 2.0 && p.x <= bk_max.x + 2.0 && p.y >= bk_min.y - 2.0 && p.y <= bk_max.y + 3.0) {
        vec2 sh_uv = (p - vec2(0.0, 1.2) - bk_min) / bk_size;
        if (sh_uv.x >= 0.0 && sh_uv.x <= 1.0 && sh_uv.y >= 0.0 && sh_uv.y <= 1.0) {
            vec2 sh_b = vec2(1.3, 204.0) + sh_uv * vec2(34.7, 33.3);
            vec3 sh_c = sample_bot_fast(sh_b / vec2(320.0, 240.0));
            float sh_a = smoothstep(0.09, 0.22, max3(sh_c));
            col = mix(col, vec3(0.0), sh_a * 0.65);
        }
        vec2 bk_uv = (p - bk_min) / bk_size;
        if (bk_uv.x >= 0.0 && bk_uv.x <= 1.0 && bk_uv.y >= 0.0 && bk_uv.y <= 1.0) {
            vec2 b_px = vec2(1.3, 204.0) + bk_uv * vec2(34.7, 33.3);
            vec2 bk_duv = (dp / bk_size) * (vec2(34.7, 33.3) / vec2(320.0, 240.0));
            vec3 bk_col = sample_bot(b_px / vec2(320.0, 240.0), bk_duv);
            float bk_a = smoothstep(0.09, 0.22, max3(bk_col));
            col = mix(col, bk_col, bk_a);
        }
    }

    // 7. Floating Intensity Cursor Window (< [Infernal | Intensity 7.1] >)
    if (hud_vis) {
        float ticker_x = 62.22222 + crop.cursor_x * (302.22222 / 320.0);
        float cx = clamp(ticker_x, 48.0, 378.0);
        vec2 bc = vec2(cx, 66.4);
        vec2 b_half = vec2(32.5, 16.3);
        if (abs(p.x - bc.x) <= b_half.x + 14.0 && p.y >= bc.y - b_half.y - 11.0 && p.y <= bc.y + b_half.y + 8.0) {
            float opacity = clamp(crop.cursor_opacity, 0.0, 1.0);
            float active = clamp((opacity - 0.48) / 0.52, 0.0, 1.0);

            float d_box = sd_round_box(p - bc, b_half, 4.6);
            float d_sh = sd_round_box(p - (bc + vec2(0.0, 1.3)), b_half, 4.6);

            // Soft drop shadow + warm golden active halo while dragging
            float sh_soft = clamp(1.0 - max(d_sh, 0.0) / 2.6, 0.0, 1.0);
            col = mix(col, vec3(0.0), sh_soft * sh_soft * mix(0.24, 0.68, active));
            if (active > 0.01 && d_box > 0.0) {
                float halo = exp(-d_box * 0.55) * active * 0.38;
                col = mix(col, vec3(1.0, 0.58, 0.22), halo);
            }

            // Upward blue/indigo slider pointer arrow pointing straight at `ticker_x` on the 0..9 bar
            float d_up = sd_pointer_up(p - vec2(ticker_x, bc.y - b_half.y - 0.5));
            if (d_up < aa) {
                float up_fill_a = clamp(0.5 - d_up * px_scale, 0.0, 1.0);
                float up_edge_a = clamp(1.0 - abs(d_up) * px_scale, 0.0, 1.0);
                vec3 up_fill = mix(vec3(0.42, 0.55, 1.0), vec3(0.52, 0.68, 1.0), active);
                vec3 up_rim = mix(vec3(0.20, 0.28, 0.78), vec3(1.0, 0.85, 0.52), active * 0.85);
                vec3 up_col = mix(up_fill, up_rim, up_edge_a * 0.85);
                col = mix(col, up_col, up_fill_a * mix(0.75, 1.0, active));
            }

            // Left & right draggable chevron ears (< and >)
            float d_left = sd_chevron_x(p - vec2(bc.x - b_half.x - 1.4, bc.y), -1.0);
            float d_right = sd_chevron_x(p - vec2(bc.x + b_half.x + 1.4, bc.y), 1.0);
            float d_chev = min(d_left, d_right);
            if (d_chev < aa) {
                float chev_fill_a = clamp(0.5 - d_chev * px_scale, 0.0, 1.0);
                float chev_edge_a = clamp(1.0 - abs(d_chev) * px_scale, 0.0, 1.0);
                vec3 ear_fill = mix(vec3(0.80, 0.24, 0.17), vec3(0.92, 0.20, 0.14), active);
                vec3 ear_rim = mix(vec3(0.75, 0.53, 0.37), vec3(1.0, 0.84, 0.55), active);
                vec3 ear_col = mix(ear_fill, ear_rim, chev_edge_a * 0.85);
                col = mix(col, ear_col, chev_fill_a * opacity);
            }

            // Rounded Intensity badge sampled from our parked slot at x_center = 255 on top_screen
            if (d_box < aa) {
                vec2 c_size = 2.0 * b_half;
                vec2 c_uv = clamp((p - (bc - b_half)) / c_size, 0.0, 1.0);
                vec2 t_px = vec2(188.2, 167.4) + c_uv * vec2(133.6, 70.6);
                vec2 c_duv = (dp / c_size) * (vec2(133.6, 70.6) / vec2(400.0, 240.0));
                vec3 badge_col = sample_top(t_px / vec2(400.0, 240.0), c_duv);

                // Active warm gold border while dragging / nudging
                float border_band = clamp(1.0 - abs(d_box + 0.32) * px_scale * 0.9, 0.0, 1.0);
                vec3 rim_col = mix(vec3(0.72, 0.50, 0.35), vec3(1.0, 0.83, 0.48), active);
                badge_col = mix(badge_col, rim_col, border_band * mix(0.25, 0.95, active));

                float box_a = clamp(0.5 - d_box * px_scale, 0.0, 1.0);
                col = mix(col, badge_col, box_a * opacity);
            }
        }
    }

    colour = vec4(clamp(col, 0.0, 1.0), 1.0);
}
