// Code memory for zakuro-gpu's shader JIT (zakuro/crates/zakuro-gpu/src/shader/jit.rs).
//
// The kernel lets a process have ten code memories, so programs are packed
// into a few large ones. A code memory is either writable or executable, never
// both, so while one is being written the code in it cannot run: the caller
// (jit.rs) holds a write lock around unx_jit_make, and every run of compiled
// code holds a read lock.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <switch.h>

#define ARENA (4u << 20)

static Jit *arena;      // the one programs are being put in
static size_t used;     // bytes of it taken
static int arenas;

const void *unx_jit_make(const void *code, size_t len) {
    size_t take = (len + 63) & ~(size_t)63;
    if (take > ARENA) {
        printf("jit: a program of %zu bytes does not fit an arena\n", len);
        return NULL;
    }
    if (!arena || used + take > ARENA) {
        Jit *next = malloc(sizeof *next);
        if (!next) return NULL;
        Result rc = jitCreate(next, ARENA);
        if (R_FAILED(rc)) {
            printf("jit: jitCreate failed after %d arenas, rc 0x%x (module %u desc %u)\n", arenas, rc, R_MODULE(rc), R_DESCRIPTION(rc));
            free(next);
            return NULL;
        }
        arena = next;
        used = 0;
        arenas++;
        printf("jit: arena %d, 4 MiB of code memory\n", arenas);
        // a new code memory is writable
    } else {
        Result rc = jitTransitionToWritable(arena);
        if (R_FAILED(rc)) {
            printf("jit: jitTransitionToWritable failed, 0x%x\n", rc);
            return NULL;
        }
    }
    memcpy((char *)jitGetRwAddr(arena) + used, code, len);
    const void *at = (const char *)jitGetRxAddr(arena) + used;
    used += take;
    Result rc = jitTransitionToExecutable(arena);
    if (R_FAILED(rc)) {
        printf("jit: jitTransitionToExecutable failed, 0x%x\n", rc);
        return NULL;
    }
    return at;
}

// Single contiguous CodeMemory object for zakuro-aot's ahead-of-time ARMv6K -> AArch64
// game binary (~60-88 MiB, within ±128 MiB B/BL range). Uses only 1 of Horizon's 10
// CodeMemory slots, leaving 9 slots for PICA200 shader JIT arenas above.
static Jit s_aot_jit;
static bool s_aot_allocated;

int unx_aot_alloc(size_t len, void **rw_out, const void **rx_out) {
    if (s_aot_allocated) {
        jitClose(&s_aot_jit);
        s_aot_allocated = false;
    }
    size_t aligned = (len + 0xFFFu) & ~(size_t)0xFFFu;
    if (aligned == 0) aligned = 0x1000u;
    Result rc = jitCreate(&s_aot_jit, aligned);
    if (R_FAILED(rc)) {
        printf("aot: jitCreate(%zu bytes) failed, rc 0x%x (module %u desc %u)\n",
               aligned, rc, R_MODULE(rc), R_DESCRIPTION(rc));
        return -1;
    }
    s_aot_allocated = true;
    *rw_out = jitGetRwAddr(&s_aot_jit);
    *rx_out = jitGetRxAddr(&s_aot_jit);
    printf("aot: allocated %.2f MiB CodeMemory (rw=%p rx=%p)\n",
           (double)aligned / (1024.0 * 1024.0), *rw_out, *rx_out);
    return 0;
}

int unx_aot_finish(void) {
    if (!s_aot_allocated) return -1;
    Result rc = jitTransitionToExecutable(&s_aot_jit);
    if (R_FAILED(rc)) {
        printf("aot: jitTransitionToExecutable failed, rc 0x%x\n", rc);
        return -1;
    }
    return 0;
}

void unx_aot_free(void) {
    if (s_aot_allocated) {
        jitClose(&s_aot_jit);
        s_aot_allocated = false;
    }
}
