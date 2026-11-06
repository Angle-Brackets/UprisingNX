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
