// A sampling profiler for the game thread. A thread on another core pauses
// it about every millisecond (svcSetThreadActivity), reads its PC and LR
// (svcGetThreadContext3) and lets it go. Samples, as raw addresses with the
// runtime address of main() to find the load base by, go to
// sdmc:/switch/uprisingnx/prof.bin every 10 s, over the last file. The buffer
// is a ring of the last ~65 s, so the file shows what is on screen now, not
// the boot; scripts/prof.py symbolizes and ranks them against the ELF.
#include <stdio.h>
#include <string.h>
#include <switch.h>

#define PROF_PATH "sdmc:/switch/uprisingnx/prof.bin"
#define PROF_MAX (1 << 15)  // 32k samples, ~32 s at 1 kHz over two threads, 1.5 MB
#define PROF_DUMP_EVERY 10000

int main(int argc, char **argv);

// the pc, lr and up to four callers further up by the frame pointers
typedef struct { u64 pc, lr, up[4]; } Sample;
static Sample samples[PROF_MAX];
static u32 count, failed;  // count: samples ever taken; the ring holds the last PROF_MAX
#define PROF_THREADS 4
static Handle targets[PROF_THREADS];
static u32 ntargets, nextra;
static Thread sampler;
static volatile bool running;

static void dump(void) {
    FILE *f = fopen(PROF_PATH ".tmp", "wb");
    if (!f) return;
    u32 kept = count < PROF_MAX ? count : PROF_MAX;
    u32 header[4] = {0x464f5250 /* PROF */, 3, kept, failed};
    u64 anchor = (u64)&main;
    fwrite(header, sizeof header, 1, f);
    fwrite(&anchor, sizeof anchor, 1, f);
    fwrite(samples, sizeof(Sample), kept, f);  // ring order, which a histogram doesn't mind
    fclose(f);
    remove(PROF_PATH);
    rename(PROF_PATH ".tmp", PROF_PATH);
}

static void sample_loop(void *arg) {
    (void)arg;
    u32 since_dump = 0;
    while (running) {
        svcSleepThread(1000000);
        // every thread registered, one after another; the thread's number goes
        // in the top bits of the pc (addresses are below 2^48), see prof.py
        for (u32 t = 0; t < ntargets; t++) {
            if (R_FAILED(svcSetThreadActivity(targets[t], ThreadActivity_Paused))) { failed++; continue; }
            ThreadContext ctx;
            Result rc = svcGetThreadContext3(&ctx, targets[t]);
            svcSetThreadActivity(targets[t], ThreadActivity_Runnable);
            if (R_FAILED(rc)) { failed++; continue; }
            Sample *out = &samples[count++ % PROF_MAX];
            *out = (Sample){ctx.pc.x | ((u64)t << 56), ctx.lr, {0, 0, 0, 0}};
            // up the frame records ({previous fp, return address}) while they stay on the
            // thread's stack, which is paused, to say who called what it was in
            u64 fp = ctx.fp, sp = ctx.sp;
            for (int d = 0; d < 4 && fp >= sp && fp < sp + (4u << 20) && (fp & 15) == 0; d++) {
                u64 *frame = (u64 *)fp;
                out->up[d] = frame[1];
                if (frame[0] <= fp) break;
                fp = frame[0];
            }
        }
        if (++since_dump >= PROF_DUMP_EVERY) { since_dump = 0; dump(); }
    }
    dump();
}

// called on a thread to profile besides the one that starts the profiler: the
// GPU's, from zakuro-core. thread 0 is the game's, the others come after
void prof_add_thread(void) {
    u32 i = 1 + __atomic_fetch_add(&nextra, 1, __ATOMIC_RELAXED);
    if (i < PROF_THREADS) {
        targets[i] = threadGetCurHandle();
        u32 n = __atomic_load_n(&ntargets, __ATOMIC_RELAXED);
        while (n < i + 1 && !__atomic_compare_exchange_n(&ntargets, &n, i + 1, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
    }
}

// called on the thread to profile
void prof_start(void) {
    targets[0] = threadGetCurHandle();
    u32 n = __atomic_load_n(&ntargets, __ATOMIC_RELAXED);
    while (n < 1 && !__atomic_compare_exchange_n(&ntargets, &n, 1, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
    running = true;
    // core 2, out of the way of the game (core 0) and the renderer's threads
    if (R_FAILED(threadCreate(&sampler, sample_loop, NULL, NULL, 0x10000, 0x2C, 2)) || R_FAILED(threadStart(&sampler))) {
        printf("prof: could not start the sampler\n");
        running = false;
        return;
    }
    printf("prof: sampling the game thread every 1 ms into %s\n", PROF_PATH);
}

void prof_stop(void) {
    if (!running) return;
    running = false;
    threadWaitForExit(&sampler);
    threadClose(&sampler);
    printf("prof: %u samples, %u failed\n", count, failed);
}
