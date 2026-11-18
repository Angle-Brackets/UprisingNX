#version 450
// Chapter Select 16:9 Single-Screen Vertical Showcase:
//   - Centered [Laurel + Chapter XX + Title] over a full-width pastel ribbon
//   - Upper showcase: [Panoramic Stage Artwork | High Score & Intensity Records]
//   - Lower showcase: Full-width unstretched 25-chapter panoramic carousel + Back / Scrollbar / X: Gear
// Compiled via naga (glsl-in -> spv-out):
//   binding 0: top_screen (SAMPLED_IMAGE)
//   binding 1: bot_screen (SAMPLED_IMAGE)
//   binding 2: screen_sampler (SAMPLER)

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D top_screen;
layout(set = 0, binding = 1) uniform texture2D bot_screen;
layout(set = 0, binding = 2) uniform sampler screen_sampler;

layout(push_constant) uniform ChapterCrop {
    vec4 top_area;
    vec4 top_bounds;
    vec4 bot_area;
    vec4 bot_bounds;
    float title_right;
    float pad0;
    int pad1;
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
    ivec2 taps = (crop.mode == 0)
        ? ivec2(clamp(ceil(covered * 0.5), 1.0, 4.0))
        : ivec2(clamp(ceil(covered), 1.0, 4.0));
    if (taps == ivec2(1, 1)) {
        return textureLod(sampler2D(top_screen, screen_sampler), clamp(at, crop.top_bounds.xy, crop.top_bounds.zw), 0.0).rgb;
    }
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
    ivec2 taps = (crop.mode == 0)
        ? ivec2(clamp(ceil(covered * 0.5), 1.0, 4.0))
        : ivec2(clamp(ceil(covered), 1.0, 4.0));
    if (taps == ivec2(1, 1)) {
        return textureLod(sampler2D(bot_screen, screen_sampler), clamp(at, crop.bot_bounds.xy, crop.bot_bounds.zw), 0.0).rgb;
    }
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

float min3(vec3 c) {
    return min(c.r, min(c.g, c.b));
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

vec3 blend_parchment(vec3 bg, vec3 src_col, vec3 paper_est, float edge_fade) {
    vec3 p_safe = max(paper_est, vec3(0.08));
    vec3 ratio = clamp(src_col / p_safe, 0.0, 1.0);
    vec3 mul_c = bg * ratio;

    float luma = dot(src_col, vec3(0.299, 0.587, 0.114));
    float p_luma = max(dot(p_safe, vec3(0.299, 0.587, 0.114)), 0.08);
    float rel_luma = luma / p_luma;
    float sat = (max3(src_col) - min3(src_col)) / (max3(src_col) + 0.004);
    float direct_w = max(
        smoothstep(0.68, 0.38, rel_luma),
        smoothstep(0.15, 0.34, sat) * smoothstep(0.95, 0.80, rel_luma)
    );
    vec3 out_c = mix(mul_c, src_col, direct_w);
    return mix(bg, out_c, clamp(edge_fade, 0.0, 1.0));
}

void main() {
    const float W = 426.66666;
    const float H = 240.0;
    vec2 p = uv * vec2(W, H);
    vec2 dp = max(fwidth(p), vec2(0.05));

    // Track screen fade-in/fade-out from a clean parchment patch on top_screen.
    vec3 top_paper = sample_top_fast(vec2(380.0 / 400.0, 100.0 / 240.0));
    vec3 fade_scale = clamp(top_paper / vec3(0.935, 0.895, 0.825), 0.0, 1.06);

    // 1. Full-bleed 16:9 laid-parchment page
    float g1 = vnoise(p * 1.65) - 0.5;
    float g2 = vnoise(p * 0.42) - 0.5;
    float grain = g1 * 0.014 + g2 * 0.020;
    vec2 n = (uv - vec2(0.46, 0.48)) / vec2(0.55, 0.55);
    float rad = clamp(dot(n * vec2(0.80, 0.92), n * vec2(0.80, 0.92)), 0.0, 1.45);
    vec3 center_p = vec3(0.980, 0.961, 0.922);
    vec3 edge_p = vec3(0.906, 0.855, 0.772);
    vec3 col = clamp((mix(center_p, edge_p, 0.54 * rad) + grain) * fade_scale, 0.0, 1.0);

    // 2. Top Header: Rose-Thorn Vines + Full-Width Pastel Ribbon + Centered [Laurel + Chapter XX + Title]
    const float S_HDR = 0.56;
    if (p.y <= 13.8) {
        float t_y = p.y / S_HDR;
        float fb = smoothstep(24.6, 19.5, t_y);
        if (p.x <= 190.4) {
            float t_x = p.x / S_HDR;
            vec2 duv = (dp / S_HDR) / vec2(400.0, 240.0);
            vec3 v_col = sample_top(vec2(t_x / 400.0, t_y / 240.0), duv);
            float fr = smoothstep(340.0, 316.0, t_x);
            col = blend_parchment(col, v_col, top_paper * 0.98, fb * fr);
        } else if (p.x >= 306.0 && p.x <= 416.3) {
            float t_x = 332.0 - (p.x - 306.0) / S_HDR;
            vec2 duv = (dp / S_HDR) / vec2(400.0, 240.0);
            vec3 v_col = sample_top(vec2(t_x / 400.0, t_y / 240.0), duv);
            float fx = smoothstep(332.0, 314.0, t_x) * smoothstep(135.0, 155.0, t_x);
            col = blend_parchment(col, v_col, top_paper * 0.98, fb * fx);
        }
    }

    if (p.y >= 13.8 && p.y <= 37.0) {
        float t_y = 25.0 + (p.y - 14.0) / S_HDR;
        float title_r = (crop.title_right > 80.0) ? crop.title_right : 302.0;
        float src_x0 = 18.0;
        float src_x1 = clamp(title_r + 6.0, 175.0, 390.0);
        float dst_w = (src_x1 - src_x0) * S_HDR;
        float dst_x0 = 0.5 * (W - dst_w);
        float dst_x1 = 0.5 * (W + dst_w);
        vec2 duv = (dp / S_HDR) / vec2(400.0, 240.0);

        // (a) Full-width continuous rainbow ribbon spanning the ENTIRE 16:9 top bar (0 .. W)
        if (t_y >= 25.0 && t_y <= 59.5) {
            float rib_fy = smoothstep(25.0, 26.8, t_y) * smoothstep(59.5, 57.8, t_y);
            float sample_y = clamp(t_y, 26.0, 58.0);
            vec3 v_prof = sample_top_fast(vec2(394.0 / 400.0, sample_y / 240.0))
                / max(sample_top_fast(vec2(394.0 / 400.0, 26.6 / 240.0)), vec3(0.08));
            v_prof = clamp(v_prof, 0.88, 1.06);

            // Sample KIU's pure un-occluded rainbow gradient at t_y = 26.6 stretched smoothly across 0..W
            float r_x = mix(20.0, 368.0, clamp(p.x / W, 0.0, 1.0));
            vec3 full_rainbow = clamp(sample_top_fast(vec2(r_x / 400.0, 26.6 / 240.0)) * v_prof, 0.0, 1.0);
            col = mix(col, full_rainbow, rib_fy * 0.95);

            // Overlay ONLY the centered [Laurel + Chapter XX + Title] ink & leaves onto full_rainbow!
            if (p.x >= dst_x0 && p.x <= dst_x1) {
                float t_x = src_x0 + (p.x - dst_x0) / S_HDR;
                vec3 hdr_c = sample_top(vec2(t_x / 400.0, t_y / 240.0), duv);
                vec3 orig_rib = clamp(
                    sample_top_fast(vec2(clamp(t_x, 20.0, 368.0) / 400.0, 26.6 / 240.0)) * v_prof,
                    vec3(0.08),
                    vec3(1.0)
                );
                float edge_x = smoothstep(dst_x0, dst_x0 + 2.5, p.x) * smoothstep(dst_x1, dst_x1 - 2.5, p.x);

                // Dark sepia text & laurel drop shadow multiplied seamlessly over full_rainbow
                float hdr_lum = dot(hdr_c, vec3(0.299, 0.587, 0.114));
                float rib_lum = max(dot(orig_rib, vec3(0.299, 0.587, 0.114)), 0.08);
                float ink_a = smoothstep(0.94, 0.52, hdr_lum / rib_lum);
                vec3 ink_mul = col * clamp(hdr_c / orig_rib, 0.0, 1.0);
                col = mix(col, ink_mul, ink_a * edge_x * rib_fy);

                // Gold or Green Laurel Wreath leaves composited over full_rainbow
                float hdr_warm = max(hdr_c.r, hdr_c.g) - hdr_c.b;
                float rib_warm = max(orig_rib.r, orig_rib.g) - orig_rib.b;
                float wreath_a = smoothstep(0.06, 0.18, hdr_warm - rib_warm)
                    * smoothstep(0.20, 0.40, max(hdr_c.r, hdr_c.g));
                col = mix(col, hdr_c, wreath_a * edge_x * rib_fy);
            }
        }

        // (b) Crisp full-width dark-sepia underline bar (t_y = 60.0 .. 64.4 across x = 18.67 .. 408.0)
        if (p.x >= 18.67 && p.x <= 408.0 && t_y >= 59.8 && t_y <= 64.5) {
            float ul_a = smoothstep(59.6, 60.5, t_y) * smoothstep(64.6, 63.7, t_y);
            col = mix(col, vec3(0.212, 0.145, 0.098) * fade_scale, ul_a);
        }
    }

    // 3. Upper-Left Panoramic Stage Artwork (1:1 aspect: 193.33 x 64.0 <- 302.0 x 100.0)
    vec2 a_min = vec2(18.67, 40.5);
    vec2 a_max = vec2(212.0, 104.5);
    if (p.x >= a_min.x && p.x <= a_max.x && p.y >= a_min.y && p.y <= a_max.y) {
        vec2 a_size = a_max - a_min;
        vec2 a_uv = (p - a_min) / a_size;
        vec2 t_px = vec2(55.0, 68.0) + a_uv * vec2(302.0, 100.0);
        vec2 a_duv = (dp / a_size) * (vec2(302.0, 100.0) / vec2(400.0, 240.0));
        vec3 a_col = sample_top(t_px / vec2(400.0, 240.0), a_duv);
        float fx = smoothstep(0.0, 3.5, p.x - a_min.x) * smoothstep(0.0, 4.5, a_max.x - p.x);
        float fy = smoothstep(0.0, 2.8, p.y - a_min.y) * smoothstep(0.0, 3.5, a_max.y - p.y);
        col = blend_parchment(col, a_col, top_paper * 0.98, fx * fy);
    }

    // 4. Upper-Right Chapter Records (High Score / Enemies Defeated / Intensity, 1:1 aspect, zero box!)
    vec2 s_min = vec2(226.0, 52.5);
    vec2 s_max = vec2(405.8, 89.8);
    if (p.x >= s_min.x && p.x <= s_max.x && p.y >= s_min.y && p.y <= s_max.y) {
        vec2 s_size = s_max - s_min;
        vec2 s_uv = (p - s_min) / s_size;
        // Start at y = 172.2 (below the stage-art drop shadow!) and key purely on sepia ink darkness
        vec2 t_px = vec2(14.0, 172.2) + s_uv * vec2(310.0, 64.3);
        vec2 s_duv = (dp / s_size) * (vec2(310.0, 64.3) / vec2(400.0, 240.0));
        vec3 s_col = sample_top(t_px / vec2(400.0, 240.0), s_duv);
        // Sample local paper right next to the text row so background paper cancels to 0.0 alpha!
        vec3 local_p = sample_top_fast(vec2(10.0 / 400.0, t_px.y / 240.0));
        float p_lum = max(dot(local_p, vec3(0.299, 0.587, 0.114)), 0.08);
        float s_lum = dot(s_col, vec3(0.299, 0.587, 0.114));
        float ink_a = smoothstep(0.92, 0.45, s_lum / p_lum);
        vec3 ratio = clamp(s_col / max(local_p, vec3(0.08)), 0.0, 1.0);
        float fx = smoothstep(0.0, 3.0, p.x - s_min.x) * smoothstep(0.0, 3.0, s_max.x - p.x);
        float fy = smoothstep(0.0, 2.0, p.y - s_min.y) * smoothstep(0.0, 2.0, s_max.y - p.y);
        col = mix(col, col * ratio, ink_a * fx * fy);
    }

    // 5. Horizontal Double Sepia / Antique-Gold Showcase Divider (p.y = 108.7 .. 110.2)
    if (p.x >= 18.67 && p.x <= 408.0 && p.y >= 108.0 && p.y <= 110.8) {
        float d1 = clamp(1.0 - abs(p.y - 108.6) / max(dp.y, 0.35), 0.0, 1.0);
        float d2 = clamp(1.0 - abs(p.y - 109.9) / max(dp.y, 0.35), 0.0, 1.0);
        col = mix(col, vec3(0.455, 0.345, 0.227) * fade_scale, d1 * 0.68);
        col = mix(col, vec3(0.776, 0.643, 0.416) * fade_scale, d2 * 0.58);
    }

    // 6. Full-Width Lower Panoramic Chapter Selector (x = 18.67 .. 408.0, matching top showcase width!)
    vec2 b_min = vec2(18.67, 111.0);
    vec2 b_max = vec2(408.0, 238.5);
    if (p.x >= b_min.x && p.x <= b_max.x && p.y >= b_min.y && p.y <= b_max.y) {
        vec2 b_size = b_max - b_min;
        vec2 b_uv = (p - b_min) / b_size;
        vec2 b_px = vec2(b_uv.x * 320.0, 4.0 + b_uv.y * 234.5);
        vec2 b_duv = (dp / b_size) * vec2(1.0, 234.5 / 240.0);
        vec3 b_col = sample_bot(b_px / vec2(320.0, 240.0), b_duv);
        vec3 bot_paper = sample_bot_fast(vec2(160.0 / 320.0, 102.0 / 240.0));
        // Guard against pink rose-thorn frame touching (160, 102): take max with (20, 104)
        bot_paper = max(bot_paper, sample_bot_fast(vec2(20.0 / 320.0, 104.0 / 240.0)));

        float fx = smoothstep(0.0, 5.0, p.x - b_min.x) * smoothstep(0.0, 5.0, b_max.x - p.x);
        float fy = smoothstep(0.0, 3.0, p.y - b_min.y) * smoothstep(0.0, 2.0, b_max.y - p.y);
        col = blend_parchment(col, b_col, bot_paper * 0.985, fx * fy);
    }

    // 7. Authentic Golden Pit & Crimson Rose Crest tucked cleanly into bottom-right corner
    vec2 c_min = vec2(393.5, 190.0);
    vec2 c_max = vec2(426.67, 240.0);
    if (p.x >= c_min.x && p.x <= c_max.x && p.y >= c_min.y && p.y <= c_max.y) {
        vec2 c_size = c_max - c_min;
        vec2 c_uv = (p - c_min) / c_size;
        vec2 t_px = vec2(326.0, 134.0) + c_uv * vec2(74.0, 106.0);
        // Smooth feather out top-left corner of crop where the stage-art right corner sat
        float corner_mask = max(smoothstep(361.0, 366.0, t_px.x), smoothstep(167.0, 172.0, t_px.y));
        if (corner_mask > 0.001) {
            vec2 c_duv = (dp / c_size) * (vec2(74.0, 106.0) / vec2(400.0, 240.0));
            vec3 c_col = sample_top(t_px / vec2(400.0, 240.0), c_duv);
            float fx = smoothstep(0.0, 4.5, p.x - c_min.x);
            float fy = smoothstep(0.0, 3.5, p.y - c_min.y);
            col = blend_parchment(col, c_col, top_paper * 0.98, fx * fy * corner_mask);
        }
    }

    colour = vec4(clamp(col, 0.0, 1.0), 1.0);
}
