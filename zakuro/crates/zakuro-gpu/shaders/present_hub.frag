#version 450
// Unified 16:9 Single-Screen Frontend Hub Presenter:
//   layout_kind == 0 (ModeSelect):
//     Lifts KIU's native bottom-screen interlocking mode buttons (`Solo / StreetPass /
//     Vault / Together / Options / Back / AR / Manual`) at 1:1 pixel aspect over its
//     full-width pastel backdrop, and composites KIU's native top-screen undulating
//     black-wave description banner along the bottom.
//   layout_kind == 1 (SoloHub — Option B Perimeter Frame):
//     Displays the full-bleed 16:9 3D Sanctuary camera (with KIU's native Total Hearts
//     pouch relocated at 1:1 aspect to top-left and lower-right black swirl omitted),
//     framed around the perimeter by KIU's 7 repurposed bottom-screen station sprites.
//   layout_kind == 2 (FileSelect — Option A Vertical Connected Speech-Bubble Stack):
//     Displays KIU's full-bleed lined rainbow parchment + outer ink filigree + centered
//     1:1 Speech-Bubble Summary above the 3 leather-bound Tomes (`I / II / III`), with
//     KIU's native speech-bubble pointer tail connecting into the selected Tome.
//   layout_kind == 3 (WeaponSelect — Unified 16:9 Gear Hub, Category Select, & Armory):
//     Displays KIU's full-bleed 16:9 rose-damask & crimson-thorn stage with right-aligned
//     weapon inspector / showcase and left-aligned 1:1 interactive Armory panel (`pad0` =
//     `bot_x0`, `pad1` = `bot_scale`), analytically chroma-keyed from `#FF00FF`.
// Compiled via naga (glsl-in -> spv-out):
//   binding 0: top_screen (SAMPLED_IMAGE)
//   binding 1: bot_screen (SAMPLED_IMAGE)
//   binding 2: screen_sampler (SAMPLER)

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 colour;

layout(set = 0, binding = 0) uniform texture2D top_screen;
layout(set = 0, binding = 1) uniform texture2D bot_screen;
layout(set = 0, binding = 2) uniform sampler screen_sampler;

layout(push_constant) uniform HubCrop {
    vec4 top_area;
    vec4 top_bounds;
    vec4 bot_area;
    vec4 bot_bounds;
    float pad0;
    float pad1;
    int layout_kind;
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
    vec2 smooth_taps = vec2(
        covered.x > 1.001 ? clamp(ceil(covered.x * 0.5), 2.0, 4.0) : 1.0,
        covered.y > 1.001 ? clamp(ceil(covered.y * 0.5), 2.0, 4.0) : 1.0
    );
    ivec2 taps = (crop.mode == 0)
        ? ivec2(smooth_taps)
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
    vec2 smooth_taps = vec2(
        covered.x > 1.001 ? clamp(ceil(covered.x * 0.5), 2.0, 4.0) : 1.0,
        covered.y > 1.001 ? clamp(ceil(covered.y * 0.5), 2.0, 4.0) : 1.0
    );
    ivec2 taps = (crop.mode == 0)
        ? ivec2(smooth_taps)
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

vec3 sample_bot_fast(vec2 local_uv) {
    vec2 at = clamp(crop.bot_area.xy + local_uv * crop.bot_area.zw, crop.bot_bounds.xy, crop.bot_bounds.zw);
    return textureLod(sampler2D(bot_screen, screen_sampler), at, 0.0).rgb;
}

float max3(vec3 c) {
    return max(c.r, max(c.g, c.b));
}

void main() {
    const float W = 426.66666;
    const float H = 240.0;
    vec2 p = uv * vec2(W, H);
    vec2 dp = max(fwidth(p), vec2(0.05));
    vec2 duv = dp / vec2(W, H);

    if (crop.layout_kind == 0) {
        // ModeSelect:
        // 1. Native Top-Screen lined parchment + color-shifting wings/vines + undulating
        //    black bottom wave + 1:1 white description blurb (with redundant top-left
        //    header omitted in `adjust_mode_top`).
        vec3 col = sample_top(uv, duv);

        // 2. Superimpose KIU's lifted 1:1 bottom-screen mode buttons keyed from `#0000FF`,
        //    normalized against `(4, 4)` so full-screen fades transition seamlessly.
        vec3 ref_bg = sample_bot_fast(vec2(4.0 / W, 4.0 / H));
        float ref_blue = ref_bg.b - max(ref_bg.r, ref_bg.g);
        if (ref_blue > 0.05 && p.y < 202.0) {
            vec3 b_col = sample_bot(uv, duv);
            vec3 c_norm = clamp((b_col - ref_bg) / ref_blue + vec3(0.0, 0.0, 1.0), 0.0, 1.0);
            float bg_amt = clamp(
                (c_norm.b - max(1.45 * c_norm.r, 1.65 * c_norm.g) - 0.06) / 0.94,
                0.0,
                1.0
            );
            float fg_a = (1.0 - bg_amt) * clamp(ref_blue, 0.0, 1.0);
            vec3 premul_fg = clamp(c_norm - bg_amt * vec3(0.0, 0.0, 1.0), 0.0, 1.0) * clamp(ref_blue, 0.0, 1.0);
            col = clamp(col * (1.0 - fg_a) + premul_fg, 0.0, 1.0);
        }
        colour = vec4(clamp(col, 0.0, 1.0), 1.0);
        return;
    }

    if (crop.layout_kind == 2) {
        // FileSelect (Option A — Vertical Connected Speech-Bubble Stack):
        // 1. Native Top-Screen lined rainbow parchment + outer corner vines/wings/wave +
        //    1:1 Speech-Bubble Save Summary in the upper half.
        vec3 col = sample_top(uv, duv);

        // 2. Superimpose KIU's 3 leather-bound Tomes (`I / II / III`), connected speech-bubble
        //    pointer tail (`p.y >= 114.0`, emerging from the black Speech Bubble below text),
        //    `Back`, `How to Play`, and sub-prompts keyed analytically from `#00FFFF` (`0, 1, 1`).
        vec3 ref_bg = sample_bot_fast(vec2(4.0 / W, 4.0 / H));
        float ref_cyan = min(ref_bg.g, ref_bg.b) - ref_bg.r;
        if (ref_cyan > 0.05 && p.y >= 114.0) {
            vec3 b_col = sample_bot(uv, duv);
            vec3 c_norm = clamp((b_col - ref_bg) / ref_cyan + vec3(0.0, 1.0, 1.0), 0.0, 1.0);
            float bg_amt = clamp(
                min(c_norm.g, c_norm.b) - c_norm.r - 1.2 * abs(c_norm.g - c_norm.b),
                0.0,
                1.0
            );
            float fg_a = (1.0 - bg_amt) * clamp(ref_cyan, 0.0, 1.0);
            vec3 premul_fg = clamp(c_norm - bg_amt * vec3(0.0, 1.0, 1.0), 0.0, 1.0) * clamp(ref_cyan, 0.0, 1.0);
            col = clamp(col * (1.0 - fg_a) + premul_fg, 0.0, 1.0);
        }
        colour = vec4(clamp(col, 0.0, 1.0), 1.0);
        return;
    }

    if (crop.layout_kind == 3) {
        // WeaponSelect (Gear Hub, Weapon Category Select, Weapon List, & Power Configurator):
        vec3 col = sample_top(uv, duv);
        float bot_x0 = crop.pad0;
        bool is_power = crop.pad1 < 0.0;
        bool is_list = crop.pad1 > 0.0 && crop.pad1 < 0.775;
        float bot_s = max(abs(crop.pad1), 0.01);
        float bot_w = 320.0 * bot_s;
        float bot_h = 240.0 * bot_s;
        float bot_y0 = 240.0 - bot_h;
        vec3 ref_bg = sample_bot_fast(vec2(318.5 / 320.0, 238.5 / 240.0));
        float ref_mag = min(ref_bg.r, ref_bg.b) - ref_bg.g;
        const vec2 probes[3] = vec2[3](
            vec2(318.5 / 320.0, 2.0 / 240.0),
            vec2(72.0 / 320.0, 234.0 / 240.0),
            vec2(1.5 / 320.0, 2.0 / 240.0)
        );
        for (int i = 0; i < 3; i++) {
            vec3 cand = sample_bot_fast(probes[i]);
            float m = min(cand.r, cand.b) - cand.g;
            if (m > ref_mag) {
                ref_bg = cand;
                ref_mag = m;
            }
        }
        if (ref_mag > 0.04 && p.x >= bot_x0 && p.x <= bot_x0 + bot_w && p.y >= bot_y0 && p.y <= bot_y0 + bot_h) {
            vec2 local_uv = (p - vec2(bot_x0, bot_y0)) / vec2(bot_w, bot_h);
            vec2 local_duv = dp / vec2(bot_w, bot_h);
            vec3 b_col = sample_bot(local_uv, local_duv);
            vec3 c_norm = clamp((b_col - ref_bg) / ref_mag + vec3(1.0, 0.0, 1.0), 0.0, 1.0);
            vec2 bp = local_uv * vec2(320.0, 240.0);
            bool in_opaque_well = (is_power && (
                (bp.x >= 1.0 && bp.x <= 148.0 && bp.y >= 50.0 && bp.y <= 200.0) ||
                (bp.x >= 147.0 && bp.x <= 307.0 && bp.y >= 26.0 && bp.y <= 200.0)
            )) || (is_list && bp.x >= 3.5 && bp.x <= 314.5 && bp.y >= 31.0 && bp.y <= 192.0);
            float raw_mag = min(c_norm.r, c_norm.b) - c_norm.g - 3.2 * abs(c_norm.r - c_norm.b);
            float bg_amt = in_opaque_well
                ? smoothstep(0.90, 0.97, raw_mag)
                : clamp((raw_mag - 0.03) / 0.97, 0.0, 1.0);
            float fg_a = 1.0 - bg_amt;
            vec3 premul_fg = clamp(b_col - bg_amt * ref_bg, 0.0, 1.0);
            col = clamp(col * (1.0 - fg_a) + premul_fg, 0.0, 1.0);
        }
        colour = vec4(clamp(col, 0.0, 1.0), 1.0);
        return;
    }

    if (crop.layout_kind == 4) {
        // Pause (In-Battle Pause Menu, Options/Controls/Sound, Quit Modal, & Remove Display):
        //   pad0 = PauseMenu state:
        //     1, 2, 4 -> Main Pause (centered card, faded top strip cropped: by=42..240)
        //     3       -> Quit Confirmation modal (exact dialog box cropped: bx=16.5..303.5, by=18.5..221.5)
        //     5, 6, 7 -> Options / Controls / Volume (centered full 4:3 card above 54px bottom help bar)
        //     9       -> Remove Display / Photo Mode (clean 16:9 view + bottom-right DISPLAY tab only)
        vec3 col = sample_top(uv, duv);

        bool is_quit = abs(crop.pad0 - 3.0) < 0.5;
        bool is_photo = crop.pad0 >= 8.5;
        bool is_options = !is_photo && crop.pad0 >= 4.5;

        if (is_photo) {
            // Clean 16:9 screenshot view: show only KIU's bottom-right `DISPLAY` tab
            // (`bx = 222..320, by = 207..240` = 98x33) anchored cleanly at bottom-right.
            vec2 tab_min = vec2(W - 88.0, H - 30.0);
            vec2 tab_max = vec2(W, H);
            if (p.x >= tab_min.x && p.x <= tab_max.x && p.y >= tab_min.y && p.y <= tab_max.y) {
                vec2 local = (p - tab_min) / (tab_max - tab_min);
                vec2 buv = vec2(mix(222.0 / 320.0, 1.0, local.x), mix(207.0 / 240.0, 1.0, local.y));
                vec2 bduv = dp / (tab_max - tab_min) * vec2(98.0 / 320.0, 33.0 / 240.0);
                col = sample_bot(buv, bduv);
            }
            colour = vec4(clamp(col, 0.0, 1.0), 1.0);
            return;
        }

        // Source UV sub-rect on the 320x240 bottom screen and destination rect in 426.67x240 space.
        vec2 src_min = is_quit
            ? vec2(16.5 / 320.0, 18.5 / 240.0)
            : (is_options ? vec2(0.0, 0.0) : vec2(0.0, 42.0 / 240.0));
        vec2 src_max = is_quit
            ? vec2(303.5 / 320.0, 221.5 / 240.0)
            : vec2(1.0, 1.0);

        vec2 card_size = is_quit
            ? vec2(246.0, 174.0)
            : (is_options ? vec2(224.0, 168.0) : vec2(256.0, 158.4));
        float card_y0 = is_quit
            ? (H - 174.0) * 0.5
            : (is_options ? 10.0 : 50.0);
        vec2 card_min = vec2((W - card_size.x) * 0.5, card_y0);
        vec2 card_max = card_min + card_size;
        float radius = is_quit ? 7.5 : (is_options ? 5.0 : 6.0);

        // Signed distance to the rounded card rectangle (`<= 0` inside).
        vec2 card_center = 0.5 * (card_min + card_max);
        vec2 card_half = 0.5 * card_size;
        vec2 q = abs(p - card_center) - (card_half - vec2(radius));
        float dist = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;

        // Soft drop shadow behind the card (offset +2.5px downward).
        vec2 qs = abs((p - vec2(0.0, 2.5)) - card_center) - (card_half - vec2(radius));
        float sdist = length(max(qs, vec2(0.0))) + min(max(qs.x, qs.y), 0.0) - radius;
        if (sdist > 0.0 && sdist < 11.0) {
            float shadow = (1.0 - smoothstep(0.0, 11.0, sdist)) * 0.58;
            col *= (1.0 - shadow);
        }

        // Anti-aliased rounded card + optional warm gold hairline frame (Quit dialog already has its own black border).
        float aa = max(dp.x, dp.y) * 0.75;
        if (!is_quit) {
            float border_w = 1.4;
            float border_mask = 1.0 - smoothstep(border_w - aa, border_w + aa, dist);
            if (border_mask > 0.0) {
                vec3 frame_col = vec3(0.88, 0.76, 0.42);
                col = mix(col, frame_col, border_mask * 0.88);
            }
        }
        float card_mask = 1.0 - smoothstep(-aa, aa, dist);
        if (card_mask > 0.0) {
            vec2 local_uv = clamp((p - card_min) / card_size, vec2(0.0), vec2(1.0));
            vec2 buv = mix(src_min, src_max, local_uv);
            vec2 bduv = (dp / card_size) * (src_max - src_min);
            vec3 card_col = sample_bot(buv, bduv);
            col = mix(col, card_col, card_mask);
        }

        colour = vec4(clamp(col, 0.0, 1.0), 1.0);
        return;
    }

    // SoloHub (Option B — Perimeter Sanctuary Frame):
    // 1. Live 3D Sanctuary camera + relocated top-left Total Hearts + bottom-center blurb
    vec3 col = sample_top(uv, duv);

    // 2. Repurposed 1:1 bottom-screen station sprites keyed cleanly from `#FF00FF`,
    //    normalized against the top-center reference pixel so full-screen black/white
    //    transitions fade seamlessly with zero tint.
    vec3 ref_bg = sample_bot_fast(vec2(0.5, 8.0 / H));
    float ref_mag = min(ref_bg.r, ref_bg.b) - ref_bg.g;
    if (ref_mag > 0.05) {
        vec3 b_col = sample_bot(uv, duv);
        vec3 c_norm = clamp((b_col - ref_bg) / ref_mag + vec3(1.0, 0.0, 1.0), 0.0, 1.0);
        float bg_amt = clamp(
            min(c_norm.r, c_norm.b) - c_norm.g - 1.2 * abs(c_norm.r - c_norm.b),
            0.0,
            1.0
        );
        float fg_a = (1.0 - bg_amt) * clamp(ref_mag, 0.0, 1.0);
        vec3 premul_fg = clamp(c_norm - bg_amt * vec3(1.0, 0.0, 1.0), 0.0, 1.0) * clamp(ref_mag, 0.0, 1.0);
        col = clamp(col * (1.0 - fg_a) + premul_fg, 0.0, 1.0);
    }

    colour = vec4(clamp(col, 0.0, 1.0), 1.0);
}
