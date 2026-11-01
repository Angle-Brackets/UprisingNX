// A sampling profiler for the game thread. A thread on another core pauses
// it about every millisecond (svcSetThreadActivity), reads its PC and LR
// (svcGetThreadContext3) and lets it go. Samples, as raw addresses with the
// runtime address of main() to find the load base by, go to sdmc:/switch/uprisingnx/prof.bin every 10 s, over the last
// file; scripts/prof.py symbolizes and ranks them against the ELF.
#include <stdio.h>
#include <string.h>
#include <switch.h>

#define PROF_PATH "sdmc:/switch/uprisingnx/prof.bin"
#define PROF_MAX (1 << 18)  // 256k samples, 8 MB
#define PROF_DUMP_EVERY 10000

int main(int argc, char **argv);

typedef struct { u64 pc, lr; } Sample;
static Sample samples[PROF_MAX];
static u32 count, failed;
static Handle target;
static Thread sampler;
static volatile bool running;

static void dump(void) {
    FILE *f = fopen(PROF_PATH ".tmp", "wb");
    if (!f) return;
    u32 header[4] = {0x464f5250 /* PROF */, 2, count, failed};
    u64 anchor = (u64)&main;
    fwrite(header, sizeof header, 1, f);
    fwrite(&anchor, sizeof anchor, 1, f);
    fwrite(samples, sizeof(Sample), count, f);
    fclose(f);
    remove(PROF_PATH);
    rename(PROF_PATH ".tmp", PROF_PATH);
}

static void sample_loop(void *arg) {
    (void)arg;
    u32 since_dump = 0;
    while (running) {
        svcSleepThread(1000000);
        if (R_FAILED(svcSetThreadActivity(target, ThreadActivity_Paused))) { failed++; continue; }
        ThreadContext ctx;
        Result rc = svcGetThreadContext3(&ctx, target);
        svcSetThreadActivity(target, ThreadActivity_Runnable);
        if (R_FAILED(rc)) { failed++; continue; }
        if (count < PROF_MAX) samples[count++] = (Sample){ctx.pc.x, ctx.lr};
        if (++since_dump >= PROF_DUMP_EVERY) { since_dump = 0; dump(); }
    }
    dump();
}

// called on the thread to profile
void prof_start(void) {
    target = threadGetCurHandle();
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
