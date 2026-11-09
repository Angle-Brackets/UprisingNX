// Audio output on a thread of its own, with time stretching.
//
// The game makes its sound in emulated time: one 3DS audio frame (160 samples at
// 32728 Hz) per 4.9 emulated ms, and emulated time only goes as fast as frames are
// emulated. At 38 fps that is ~63% of a second of sound per real second, so a plain
// queue runs dry and the sound stutters. Here the game's frame only pushes samples
// into a ring (nx_audio_play never blocks), and an audio thread keeps the DAC fed
// at the real rate. It plays the ring through a WSOLA time stretcher whose tempo
// follows how full the ring is, so slow frames slow the sound's tempo (as the game
// itself is in slow motion) without lowering its pitch or leaving gaps.
//
// UNX_AUDIO_STRETCH=0 keeps the tempo near 1 (drift correction only); underruns
// then fade to silence.

#include <switch.h>
#include <malloc.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN_RATE 32728.498
#define OUT_RATE 48000.0
#define NUM_BUFFERS 3
#define SAMPLES_PER_BUFFER 512 // 10.7 ms at 48 kHz
#define BUFFER_MEM_SIZE 0x1000

// the ring of the game's samples, single producer (the game's frame), single consumer
#define RING_FRAMES 32768
#define RING_MASK (RING_FRAMES - 1)
static int16_t g_ring[RING_FRAMES][2];
static volatile uint64_t g_write; // frames the game has written, ever

// WSOLA: grains of N input frames, Hann windowed, laid down every HS frames (50%)
#define N 640    // 19.6 ms
#define HS (N / 2)
#define SEARCH 160 // +-4.9 ms around where the tempo says the next grain starts

static float g_window[N];
static float g_tail[HS][2];   // the windowed second half of the last grain
static double g_nominal;      // where the tempo says the next grain starts (input frames)
static uint64_t g_prev;       // where the last grain started
static double g_tempo = 1.0;

// stretched frames waiting to be resampled to 48 kHz
#define ST_MAX 4096
static float g_st[ST_MAX][2];
static int g_st_count;
static double g_st_pos;

static AudioOutBuffer g_audout[NUM_BUFFERS];
static int16_t *g_data[NUM_BUFFERS];
static Thread g_thread;
static volatile bool g_running;
static bool g_initialized;
static bool g_stretch = true;
static uint64_t g_underruns, g_grains;
static float g_last[2];

static inline float in_at(uint64_t i, int ch) { return (float)g_ring[i & RING_MASK][ch]; }

// target ring fill (input frames) ahead of the next grain's end: ~90 ms
#define TARGET (IN_RATE * 0.090)

static void set_tempo(void) {
    uint64_t w = __atomic_load_n(&g_write, __ATOMIC_ACQUIRE);
    double fill = (double)w - (g_nominal + N + SEARCH);
    double want = fill / TARGET;
    double lo = g_stretch ? 0.35 : 0.97;
    double hi = g_stretch ? 1.30 : 1.03;
    if (want < lo) want = lo;
    if (want > hi) want = hi;
    // down fast when the ring drains (a late frame), back up slowly
    g_tempo += (want - g_tempo) * (want < g_tempo ? 0.3 : 0.05);
}

// lays down one grain, HS stretched frames. false when the ring has too little.
static bool wsola_step(void) {
    set_tempo();
    double next = g_nominal + HS * g_tempo;
    uint64_t w = __atomic_load_n(&g_write, __ATOMIC_ACQUIRE);
    int64_t base = (int64_t)next - SEARCH;
    if (base < 0) base = 0;
    if ((double)base + 2 * SEARCH + N > (double)w) return false;
    // the stretch is falling too far behind what the game wrote: jump ahead
    if ((double)w - next > RING_FRAMES - 4096) {
        next = (double)w - TARGET - N - SEARCH;
        base = (int64_t)next - SEARCH;
        g_prev = (uint64_t)base;
    }
    // the start whose first half best continues the last grain's natural next frames
    uint64_t natural = g_prev + HS;
    int best = SEARCH;
    if (g_grains > 0) {
        float bestc = -1e30f;
        for (int d = 0; d <= 2 * SEARCH; d += 2) {
            uint64_t a = (uint64_t)base + d;
            float c = 0, e = 1e-3f;
            for (int i = 0; i < HS; i += 4) {
                float x = in_at(a + i, 0) + in_at(a + i, 1);
                float y = in_at(natural + i, 0) + in_at(natural + i, 1);
                c += x * y;
                e += x * x;
            }
            c /= sqrtf(e);
            if (c > bestc) { bestc = c; best = d; }
        }
    }
    uint64_t a = (uint64_t)base + best;
    if (g_st_count + HS > ST_MAX) return true; // full, the resampler has plenty
    for (int i = 0; i < HS; i++) {
        for (int ch = 0; ch < 2; ch++) {
            g_st[g_st_count + i][ch] = g_tail[i][ch] + in_at(a + i, ch) * g_window[i];
            g_tail[i][ch] = in_at(a + HS + i, ch) * g_window[HS + i];
        }
    }
    g_st_count += HS;
    g_prev = a;
    g_nominal = next;
    g_grains++;
    return true;
}

static void fill_buffer(int16_t *out) {
    const double step = IN_RATE / OUT_RATE;
    for (int i = 0; i < SAMPLES_PER_BUFFER; i++) {
        while ((int)g_st_pos + 1 >= g_st_count) {
            if (!wsola_step()) break;
        }
        int idx = (int)g_st_pos;
        if (idx + 1 >= g_st_count) {
            // nothing to play: fade what was last played to silence
            g_underruns++;
            for (; i < SAMPLES_PER_BUFFER; i++) {
                g_last[0] *= 0.97f;
                g_last[1] *= 0.97f;
                out[i * 2] = (int16_t)g_last[0];
                out[i * 2 + 1] = (int16_t)g_last[1];
            }
            break;
        }
        float f = (float)(g_st_pos - idx);
        for (int ch = 0; ch < 2; ch++) {
            float v = g_st[idx][ch] + f * (g_st[idx + 1][ch] - g_st[idx][ch]);
            if (v > 32767.f) v = 32767.f;
            if (v < -32768.f) v = -32768.f;
            g_last[ch] = v;
            out[i * 2 + ch] = (int16_t)v;
        }
        g_st_pos += step;
    }
    int used = (int)g_st_pos;
    if (used > 0) {
        if (used > g_st_count) used = g_st_count;
        memmove(&g_st[0], &g_st[used], (size_t)(g_st_count - used) * sizeof(g_st[0]));
        g_st_count -= used;
        g_st_pos -= used;
    }
}

static void submit(int b) {
    fill_buffer(g_data[b]);
    g_audout[b].next = NULL;
    g_audout[b].buffer = g_data[b];
    g_audout[b].buffer_size = BUFFER_MEM_SIZE;
    g_audout[b].data_size = SAMPLES_PER_BUFFER * 2 * sizeof(int16_t);
    g_audout[b].data_offset = 0;
    audoutAppendAudioOutBuffer(&g_audout[b]);
}

static void audio_thread(void *arg) {
    (void)arg;
    for (int b = 0; b < NUM_BUFFERS; b++) submit(b);
    uint64_t report = 0;
    while (g_running) {
        AudioOutBuffer *released = NULL;
        u32 count = 0;
        Result rc = audoutWaitPlayFinish(&released, &count, 100000000ULL);
        if (R_FAILED(rc) || !released) continue;
        for (int b = 0; b < NUM_BUFFERS; b++) {
            if (&g_audout[b] == released) submit(b);
        }
        // others released at the same time
        while (R_SUCCEEDED(audoutGetReleasedAudioOutBuffer(&released, &count)) && released && count) {
            for (int b = 0; b < NUM_BUFFERS; b++) {
                if (&g_audout[b] == released) submit(b);
            }
        }
        if (++report % 940 == 0) { // ~10 s
            printf("nx_audio: tempo %.2f, %llu underruns, ring %lld frames ahead\n", g_tempo,
                   (unsigned long long)g_underruns,
                   (long long)((double)__atomic_load_n(&g_write, __ATOMIC_ACQUIRE) - g_nominal));
        }
    }
}

void nx_audio_init(void) {
    if (g_initialized) return;
    const char *e = getenv("UNX_AUDIO_STRETCH");
    g_stretch = !(e && e[0] == '0');
    for (int i = 0; i < N; i++) g_window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)N);
    for (int b = 0; b < NUM_BUFFERS; b++) {
        g_data[b] = (int16_t *)memalign(0x1000, BUFFER_MEM_SIZE);
        if (!g_data[b]) { printf("nx_audio: no memory for buffer %d\n", b); return; }
        memset(g_data[b], 0, BUFFER_MEM_SIZE);
    }
    Result rc = audoutInitialize();
    if (R_FAILED(rc)) { printf("nx_audio: audoutInitialize failed: 0x%x\n", rc); return; }
    rc = audoutStartAudioOut();
    if (R_FAILED(rc)) { printf("nx_audio: audoutStartAudioOut failed: 0x%x\n", rc); audoutExit(); return; }
    g_running = true;
    // above the game's priority, on core 0 next to the game's mostly waiting thread
    rc = threadCreate(&g_thread, audio_thread, NULL, NULL, 0x8000, 0x2A, 0);
    if (R_SUCCEEDED(rc)) rc = threadStart(&g_thread);
    if (R_FAILED(rc)) { printf("nx_audio: no audio thread: 0x%x\n", rc); g_running = false; return; }
    g_initialized = true;
    printf("nx_audio: audout 48kHz stereo on its own thread, %d x %d samples, time stretch %s\n", NUM_BUFFERS,
           SAMPLES_PER_BUFFER, g_stretch ? "on" : "off");
}

// the game's samples (stereo, IN_RATE) since its last frame. never blocks.
void nx_audio_play(const int16_t *samples, uint32_t count) {
    if (!g_initialized || !samples || count == 0) return;
    uint64_t w = g_write;
    for (uint32_t i = 0; i < count; i++) {
        g_ring[(w + i) & RING_MASK][0] = samples[i * 2];
        g_ring[(w + i) & RING_MASK][1] = samples[i * 2 + 1];
    }
    __atomic_store_n(&g_write, w + count, __ATOMIC_RELEASE);
}

void nx_audio_exit(void) {
    if (!g_initialized) return;
    g_running = false;
    threadWaitForExit(&g_thread);
    threadClose(&g_thread);
    audoutStopAudioOut();
    audoutExit();
    for (int b = 0; b < NUM_BUFFERS; b++) { free(g_data[b]); g_data[b] = NULL; }
    g_initialized = false;
    printf("nx_audio: exited\n");
}
