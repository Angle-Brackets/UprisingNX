// POSIX pieces Rust std calls that libnx/newlib don't provide.
#include <errno.h>
#include <unistd.h>
#include <switch.h>

// std reads the page size (thread stacks) and the processor count
// (available_parallelism, so rayon's default pool size). Applications get
// cores 0-2; core 3 belongs to the system.
long sysconf(int name) {
    switch (name) {
    case _SC_PAGESIZE:
        return 0x1000;
    case _SC_NPROCESSORS_CONF:
    case _SC_NPROCESSORS_ONLN:
        return 3;
    default:
        errno = EINVAL;
        return -1;
    }
}

// ---- ELF TLS for Rust ------------------------------------------------------
// LLVM reads the thread pointer from tpidr_el0 (`mrs xN, tpidr_el0`) for
// #[thread_local] and std's thread_local!. Horizon leaves tpidr_el0 at 0; libnx
// keeps the real pointer in its thread vars and hands it to GCC code through
// the soft helper __aarch64_read_tp (devkitA64 builds with -mtp=soft). Both
// use the same AArch64 variant-I layout, so copying libnx's pointer into
// tpidr_el0 makes Rust's TLS land on libnx's per-thread TLS block. The kernel
// saves tpidr_el0 per thread (ThreadContext.tpidr), so it sticks.
//
// Every thread that runs Rust code needs it once: the main thread calls
// nx_tls_init(), and pthread_create is wrapped (-Wl,--wrap=pthread_create) so
// threads std/rayon spawn get it before their start routine. Threads made with
// libnx threadCreate that call into Rust must call nx_tls_init() themselves.
#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void *__aarch64_read_tp(void);

void nx_tls_init(void) {
    void *tp = __aarch64_read_tp();
    __asm__ volatile("msr tpidr_el0, %0" ::"r"(tp));
}

struct nx_thread_start {
    void *(*fn)(void *);
    void *arg;
};

static void *nx_thread_trampoline(void *p) {
    struct nx_thread_start start = *(struct nx_thread_start *)p;
    free(p);
    nx_tls_init();
    // Core 0 is dedicated to the main emulation thread. Worker and background
    // threads alternate between Core 1 and Core 2 (0x7 allows migration if needed).
    static int s_next_core = 0;
    // UNX_CORES=3 spreads the threads over all three cores (the game's thread is idle
    // most of the time, so core 0 is mostly wasted when it is kept for it alone)
    static int s_spread = -1;
    if (s_spread < 0) { const char *e = getenv("UNX_CORES"); s_spread = (e && e[0] == '3') ? 1 : 0; }
    int slot = __atomic_fetch_add(&s_next_core, 1, __ATOMIC_RELAXED);
    int core = s_spread ? slot % 3 : 1 + (slot % 2);
    svcSetThreadCoreMask(CUR_THREAD_HANDLE, core, 0x7);
    return start.fn(start.arg);
}

// UNX_PLACE=1: zakuro names its busy threads as they start (zakuro_gpu::thread_role),
// and each goes on a core of its own choosing here, the GPU's thread alone on
// UNX_PLACE_GPU (2) a step above the others' priority, the shader pool and the
// shading helpers on the cores listed in UNX_PLACE_SHADER / UNX_PLACE_SHADE ("10").
static int place_core(const char *list, int index, int fallback) {
    if (!list || !list[0]) return fallback;
    size_t n = strlen(list);
    int c = list[index % n] - '0';
    return (c >= 0 && c <= 2) ? c : fallback;
}

#include <malloc.h>
/* the process's memory for the frontend's report: used and total from the
 * kernel, then malloc's bytes in use and the arena it took */
void unx_memory(unsigned long long *out) {
    u64 used = 0, total = 0;
    svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0);
    svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
    struct mallinfo info = mallinfo();
    out[0] = used;
    out[1] = total;
    out[2] = (unsigned long long)info.uordblks;
    out[3] = (unsigned long long)info.arena;
}

void unx_thread_role(const char *role, int index) {
    // the pipeline compilers: a step below every other thread's priority, so
    // they never take a core from the game or the GPU, free to run on any
    // core and on core 0 first, which the game's thread leaves idle most of
    // each frame. at the drawing threads' priority on cores 1 and 2 they
    // waited seconds for a turn (Horizon does not share a core between
    // threads of one priority), the scene drawing through the generic
    // shader all along. UNX_COMPILER_PRIO (hex) overrides the priority.
    if (!strcmp(role, "compiler")) {
        const char *p = getenv("UNX_COMPILER_PRIO");
        int priority = (p && p[0]) ? (int)strtol(p, NULL, 16) : 0x2D;
        svcSetThreadPriority(CUR_THREAD_HANDLE, priority);
        svcSetThreadCoreMask(CUR_THREAD_HANDLE, 0, 0x7);
        printf("place: compiler %d at priority 0x%X, core 0 first, any core\n", index, priority);
        return;
    }
    const char *on = getenv("UNX_PLACE");
    if (!on || on[0] != '1') return;
    int core;
    if (!strcmp(role, "gpu")) {
        core = place_core(getenv("UNX_PLACE_GPU"), 0, 2);
        svcSetThreadPriority(CUR_THREAD_HANDLE, 0x2B);
    } else if (!strcmp(role, "shader")) {
        core = place_core(getenv("UNX_PLACE_SHADER"), index, index % 2 ? 0 : 1);
    } else if (!strcmp(role, "shade")) {
        core = place_core(getenv("UNX_PLACE_SHADE"), index, index % 2 ? 0 : 1);
    } else {
        return;
    }
    svcSetThreadCoreMask(CUR_THREAD_HANDLE, core, 1u << core);
    printf("place: %s %d on core %d\n", role, index, core);
}

int __real_pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*fn)(void *), void *arg);

int __wrap_pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*fn)(void *), void *arg) {
    struct nx_thread_start *start = malloc(sizeof *start);
    if (!start)
        return EAGAIN;
    start->fn = fn;
    start->arg = arg;
    int r = __real_pthread_create(thread, attr, nx_thread_trampoline, start);
    if (r != 0)
        free(start);
    return r;
}

// ---- no dynamic loading on Horizon ----------------------------------------
// recomp-abi links libloading to open a recompiled library at run time. The
// Switch build links the recompiled code statically, so these only report
// failure.
#include <stddef.h>
void *dlopen(const char *file, int mode) { (void)file; (void)mode; return NULL; }
void *dlsym(void *handle, const char *name) { (void)handle; (void)name; return NULL; }
int dlclose(void *handle) { (void)handle; return 0; }
char *dlerror(void) { return (char *)"no dynamic loading on Horizon"; }

// ---- getrandom from the kernel's CSPRNG ------------------------------------
#include <sys/types.h>
#include <switch.h>
ssize_t getrandom(void *buf, size_t len, unsigned flags) {
    (void)flags;
    static int ready;
    if (!ready) {
        if (R_FAILED(csrngInitialize())) { errno = EIO; return -1; }
        ready = 1;
    }
    if (R_FAILED(csrngGetRandomBytes(buf, len))) { errno = EIO; return -1; }
    return (ssize_t)len;
}
