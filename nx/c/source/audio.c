#include <switch.h>
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define NUM_BUFFERS 6
#define SAMPLES_PER_BUFFER 800 // 800 stereo samples @ 48 kHz = 16.666 ms (1 frame @ 60fps)
#define BUFFER_MEM_SIZE 0x1000 // 4096 bytes (aligned to 0x1000)
#define MAX_INPUT_SAMPLES 8192

typedef struct {
    AudioOutBuffer audout_buf;
    int16_t *data;
    bool in_flight;
} AudioBuffer;

static AudioBuffer g_buffers[NUM_BUFFERS];
static int16_t g_input[MAX_INPUT_SAMPLES][2];
static int g_input_count = 0;
static double g_input_pos = 0.0;
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
    s_step = 32728.498 / 48000.0;
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

    const double BASE_STEP = 32728.498 / 48000.0; // ~0.6818437

    // Dynamic rate control: Target 2 in-flight hardware buffers + 200 input samples in reserve.
    // Proportional control with low-pass filter ensures seamless audio playback across frame rate variations.
    int total_input_available = g_input_count - (int)g_input_pos;
    double target_input = 200.0 + (double)(2 - in_flight) * (SAMPLES_PER_BUFFER * BASE_STEP);
    double error = (double)total_input_available - target_input;

    double target_scale = 1.0 + (error / 1500.0) * 0.08;
    if (target_scale < 0.88) target_scale = 0.88;
    if (target_scale > 1.12) target_scale = 1.12;

    s_step = s_step * 0.90 + (BASE_STEP * target_scale) * 0.10;

    // Fill available buffers
    for (int b = 0; b < NUM_BUFFERS; b++) {
        if (g_buffers[b].in_flight) continue;

        int available = g_input_count - (int)g_input_pos;
        if (available < 16) {
            // Truly exhausted
            if (in_flight == 0) {
                // If nothing is playing on DAC, write a smooth fade to silence
                int16_t *out = g_buffers[b].data;
                for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
                    float fade = 1.0f - (float)i / 64.0f;
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

        // If available is lower than needed, adapt step for this buffer so we stretch what we have
        double buf_step = s_step;
        int needed = (int)((double)SAMPLES_PER_BUFFER * buf_step) + 2;
        if (available < needed) {
            buf_step = (double)(available - 2) / (double)SAMPLES_PER_BUFFER;
            if (buf_step < 0.25) buf_step = 0.25;
        }

        int16_t *out = g_buffers[b].data;
        for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
            int idx = (int)g_input_pos;
            if (idx + 1 >= g_input_count) {
                out[i * 2 + 0] = s_last_sample[0];
                out[i * 2 + 1] = s_last_sample[1];
                continue;
            }
            float frac = (float)(g_input_pos - idx);
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

            g_input_pos += buf_step;
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
