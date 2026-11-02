#include <switch.h>
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define NUM_BUFFERS 8
#define SAMPLES_PER_BUFFER 800 // 800 stereo samples @ 48 kHz = 16.666 ms (1 frame @ 60fps)
#define BUFFER_MEM_SIZE 0x1000 // 4096 bytes (aligned to 0x1000)
#define MAX_INPUT_SAMPLES 16384

typedef struct {
    AudioOutBuffer audout_buf;
    int16_t *data;
    bool in_flight;
} AudioBuffer;

static AudioBuffer g_buffers[NUM_BUFFERS];
static int16_t g_input[MAX_INPUT_SAMPLES][2];
static int g_input_count = 0;
static double g_input_pos = 0.0;
static const double BASE_STEP = 32728.498 / 48000.0; // ~0.6818437
static double s_step = 32728.498 / 48000.0;
static int16_t s_last_sample[2] = {0, 0};
static bool g_initialized = false;

void nx_audio_init(void) {
    if (g_initialized) return;

    memset(g_buffers, 0, sizeof(g_buffers));
    for (int i = 0; i < NUM_BUFFERS; i++) {
        g_buffers[i].data = (int16_t *)memalign(0x1000, BUFFER_MEM_SIZE);
        if (!g_buffers[i].data) {
            printf("nx_audio: failed to allocate audio buffer %d\n", i);
            return;
        }
        memset(g_buffers[i].data, 0, BUFFER_MEM_SIZE);
    }

    Result rc = audoutInitialize();
    if (R_FAILED(rc)) {
        printf("nx_audio: audoutInitialize failed: 0x%x\n", rc);
        return;
    }

    rc = audoutStartAudioOut();
    if (R_FAILED(rc)) {
        printf("nx_audio: audoutStartAudioOut failed: 0x%x\n", rc);
        audoutExit();
        return;
    }

    g_input_count = 0;
    g_input_pos = 0.0;
    s_step = BASE_STEP;
    s_last_sample[0] = 0;
    s_last_sample[1] = 0;
    g_initialized = true;
    printf("nx_audio: initialized audout (48kHz stereo, %d buffers)\n", NUM_BUFFERS);
}

void nx_audio_play(const int16_t *samples, uint32_t count) {
    if (!g_initialized || !samples || count == 0) return;

    // Reclaim finished buffers from hardware
    AudioOutBuffer *released = NULL;
    u32 released_count = 0;
    while (R_SUCCEEDED(audoutGetReleasedAudioOutBuffer(&released, &released_count)) && released != NULL) {
        for (int i = 0; i < NUM_BUFFERS; i++) {
            if (&g_buffers[i].audout_buf == released) {
                g_buffers[i].in_flight = false;
                break;
            }
        }
        released = NULL;
    }

    // Append incoming samples (stereo pairs)
    if (g_input_count + (int)count <= MAX_INPUT_SAMPLES) {
        memcpy(&g_input[g_input_count], samples, count * sizeof(int16_t) * 2);
        g_input_count += (int)count;
    } else {
        // Drop oldest if overflowing
        int drop = (g_input_count + (int)count) - MAX_INPUT_SAMPLES;
        if (drop < g_input_count) {
            memmove(&g_input[0], &g_input[drop], (g_input_count - drop) * sizeof(g_input[0]));
            g_input_count -= drop;
            g_input_pos = (g_input_pos > drop) ? (g_input_pos - drop) : 0.0;
            memcpy(&g_input[g_input_count], samples, count * sizeof(int16_t) * 2);
            g_input_count += (int)count;
        }
    }

    // Count in-flight buffers
    int in_flight = 0;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (g_buffers[i].in_flight) in_flight++;
    }

    // Drift compensation: Target 3 in-flight hardware buffers (50 ms) + 400 input samples (~12 ms reserve)
    // Keep pitch adjustments smoothly within ±2.0% (imperceptible, standard emulator rate matching)
    int total_input_available = g_input_count - (int)g_input_pos;
    double current_buffered = (double)(in_flight * SAMPLES_PER_BUFFER) * BASE_STEP + (double)total_input_available;
    const double TARGET_BUFFERED = (3.0 * (double)SAMPLES_PER_BUFFER) * BASE_STEP + 400.0; // ~2036 samples
    double error = current_buffered - TARGET_BUFFERED;

    double target_scale = 1.0 + (error / 4000.0) * 0.02;
    if (target_scale < 0.98) target_scale = 0.98;
    if (target_scale > 1.02) target_scale = 1.02;

    s_step = s_step * 0.95 + (BASE_STEP * target_scale) * 0.05;

    const int needed = (int)((double)SAMPLES_PER_BUFFER * s_step) + 2;

    // Fill available buffers
    for (int b = 0; b < NUM_BUFFERS; b++) {
        if (g_buffers[b].in_flight) continue;

        // If hardware DAC already has 4 buffers queued (66 ms), hold off until next frame
        if (in_flight >= 4) {
            break;
        }

        int available = g_input_count - (int)g_input_pos;

        // If we already have 1+ buffer playing in DAC and not enough samples to fill a complete buffer,
        // wait for next frame rather than playing truncated/silence-padded audio
        if (in_flight >= 1 && available < needed) {
            break;
        }

        // If input is completely exhausted:
        if (available <= 0) {
            if (in_flight == 0) {
                // Emergency DAC underrun prevention: smoothly fade out to silence
                int16_t *out = g_buffers[b].data;
                for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
                    float fade = 1.0f - (float)i / 32.0f;
                    if (fade < 0.0f) fade = 0.0f;
                    out[i * 2 + 0] = (int16_t)(s_last_sample[0] * fade);
                    out[i * 2 + 1] = (int16_t)(s_last_sample[1] * fade);
                }
                s_last_sample[0] = 0;
                s_last_sample[1] = 0;
                g_buffers[b].audout_buf.next = NULL;
                g_buffers[b].audout_buf.buffer = g_buffers[b].data;
                g_buffers[b].audout_buf.buffer_size = BUFFER_MEM_SIZE;
                g_buffers[b].audout_buf.data_size = SAMPLES_PER_BUFFER * 2 * sizeof(int16_t);
                g_buffers[b].audout_buf.data_offset = 0;
                g_buffers[b].in_flight = true;
                audoutAppendAudioOutBuffer(&g_buffers[b].audout_buf);
                in_flight++;
            }
            break;
        }

        // Dynamic step for this buffer: if in_flight == 0 and available < needed,
        // smoothly stretch available samples across SAMPLES_PER_BUFFER to avoid ANY dead silence
        double step = s_step;
        if (in_flight == 0 && available < needed && available >= 64) {
            step = (double)(available - 1) / (double)SAMPLES_PER_BUFFER;
        }

        int16_t *out = g_buffers[b].data;
        for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
            int idx = (int)g_input_pos;
            if (idx + 1 >= g_input_count) {
                idx = g_input_count - 2;
                if (idx < 0) idx = 0;
            }
            float frac = (float)(g_input_pos - idx);
            if (frac < 0.0f) frac = 0.0f;
            if (frac > 1.0f) frac = 1.0f;
            int16_t l0 = g_input[idx][0];
            int16_t l1 = g_input[idx + 1][0];
            int16_t r0 = g_input[idx][1];
            int16_t r1 = g_input[idx + 1][1];

            int16_t sample_l = (int16_t)(l0 + frac * (l1 - l0));
            int16_t sample_r = (int16_t)(r0 + frac * (r1 - r0));
            out[i * 2 + 0] = sample_l;
            out[i * 2 + 1] = sample_r;
            s_last_sample[0] = sample_l;
            s_last_sample[1] = sample_r;

            g_input_pos += step;
        }

        // Enqueue to audout
        g_buffers[b].audout_buf.next = NULL;
        g_buffers[b].audout_buf.buffer = g_buffers[b].data;
        g_buffers[b].audout_buf.buffer_size = BUFFER_MEM_SIZE;
        g_buffers[b].audout_buf.data_size = SAMPLES_PER_BUFFER * 2 * sizeof(int16_t);
        g_buffers[b].audout_buf.data_offset = 0;
        g_buffers[b].in_flight = true;

        audoutAppendAudioOutBuffer(&g_buffers[b].audout_buf);
        in_flight++;
    }

    // Compact input buffer
    int consumed = (int)g_input_pos;
    if (consumed > 0 && consumed <= g_input_count) {
        int remaining = g_input_count - consumed;
        if (remaining > 0) {
            memmove(&g_input[0], &g_input[consumed], remaining * sizeof(g_input[0]));
        }
        g_input_count = remaining;
        g_input_pos -= consumed;
    }
}

void nx_audio_exit(void) {
    if (!g_initialized) return;

    audoutStopAudioOut();
    audoutExit();

    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (g_buffers[i].data) {
            free(g_buffers[i].data);
            g_buffers[i].data = NULL;
        }
    }

    g_initialized = false;
    printf("nx_audio: exited\n");
}
