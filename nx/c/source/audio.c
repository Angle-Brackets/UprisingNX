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

    // Dynamic pitch/drift adjustment: 3DS rate is ~32728.5 Hz, Switch is 48000 Hz
    const double BASE_STEP = 32728.498 / 48000.0; // ~0.6818437
    double step = BASE_STEP;
    if (in_flight > 3) {
        step *= 1.005; // Speed up slightly to drain queue
    } else if (in_flight < 2) {
        step *= 0.995; // Slow down slightly to prevent underrun
    }

    // Fill available buffers
    for (int b = 0; b < NUM_BUFFERS; b++) {
        if (g_buffers[b].in_flight) continue;

        // Check if we have enough input samples to produce SAMPLES_PER_BUFFER output samples
        double end_pos = g_input_pos + (SAMPLES_PER_BUFFER - 1) * step;
        if ((int)end_pos + 1 >= g_input_count) {
            break; // Wait for more samples
        }

        int16_t *out = g_buffers[b].data;
        for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
            int idx = (int)g_input_pos;
            float frac = (float)(g_input_pos - idx);
            int16_t l0 = g_input[idx][0];
            int16_t l1 = g_input[idx + 1][0];
            int16_t r0 = g_input[idx][1];
            int16_t r1 = g_input[idx + 1][1];

            out[i * 2 + 0] = (int16_t)(l0 + frac * (l1 - l0));
            out[i * 2 + 1] = (int16_t)(r0 + frac * (r1 - r0));

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
