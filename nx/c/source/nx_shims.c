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

static int s_disp_dirty = 1;

static void on_display_applet_hook(AppletHookType hook, void *param) {
    (void)param;
    if (hook == AppletHookType_OnFocusState
        || hook == AppletHookType_OnOperationMode
        || hook == AppletHookType_OnResume) {
        __atomic_store_n(&s_disp_dirty, 1, __ATOMIC_RELAXED);
    }
}

/* Switch TV / panel active pixel dimensions: queries default display resolution
 * plus docked TV screen-size underscan (System Settings -> TV Settings -> Adjust
 * Screen Size) so the Vulkan swapchain matches the ViLayer 1:1 without a second
 * vi hardware resample. Refreshed via appletHook + display-resolution event so
 * per-frame polls avoid IPC. */
void unx_display_resolution(unsigned int *out_w, unsigned int *out_h, int force) {
    static int s_init = 0;
    static int s_has_event = 0;
    static Event s_event;
    static AppletHookCookie s_hook;
    static unsigned int s_w = 1280, s_h = 720;
    bool refresh = !s_init || force != 0 || __atomic_exchange_n(&s_disp_dirty, 0, __ATOMIC_RELAXED) != 0;
    if (!s_init) {
        s_init = 1;
        appletHook(&s_hook, on_display_applet_hook, NULL);
        if (R_SUCCEEDED(appletGetDefaultDisplayResolutionChangeEvent(&s_event))) {
            s_has_event = 1;
        }
    } else if (s_has_event && R_SUCCEEDED(eventWait(&s_event, 0))) {
        refresh = true;
    }
    if (refresh) {
        s32 w = 0, h = 0;
        bool docked = appletGetOperationMode() == AppletOperationMode_Console;
        if (!R_SUCCEEDED(appletGetDefaultDisplayResolution(&w, &h)) || w <= 0 || h <= 0) {
            w = docked ? 1920 : 1280;
            h = docked ? 1080 : 720;
        }
        unsigned int pct = 100;
        if (docked && R_SUCCEEDED(setsysInitialize())) {
            SetSysTvSettings tv;
            if (R_SUCCEEDED(setsysGetTvSettings(&tv))) {
                if (tv.underscan >= 1 && tv.underscan <= 20) {
                    pct = 100u - tv.underscan;
                } else if (tv.underscan >= 80 && tv.underscan <= 99) {
                    pct = tv.underscan;
                }
            }
            setsysExit();
        }
        unsigned int nw = (unsigned int)w;
        unsigned int nh = (unsigned int)h;
        if (pct < 100) {
            nw = ((nw * pct + 50u) / 100u) & ~1u;
            nh = ((nh * pct + 50u) / 100u) & ~1u;
        }
        if (nw != s_w || nh != s_h) {
            printf("display: %dx%d at %u%% TV screen area -> swapchain %ux%u\n", (int)w, (int)h, pct, nw, nh);
        }
        s_w = nw;
        s_h = nh;
    }
    *out_w = s_w;
    *out_h = s_h;
}

void unx_thread_role(const char *role, int index) {
    // the pipeline compilers: one step below the drawing/game threads' 0x2C
    // priority (not 0x32, which lets background workers starve shader compiles
    // on Horizon's strict fixed-priority scheduler), free to run on any core
    // and on core 0 first. UNX_COMPILER_PRIO (hex) overrides the priority.
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

// ---- getrandom from libnx's own generator ----------------------------------
// randomGet is a ChaCha generator in the process, seeded from the kernel's
// entropy at start and behind a mutex: it cannot fail. Rust's std reads this
// for every HashMap's keys and aborts the whole program on -1, which the
// csrng service once gave mid-battle (a failed IPC call), so nothing here
// asks a service any more
#include <sys/types.h>
#include <switch.h>
ssize_t getrandom(void *buf, size_t len, unsigned flags) {
    (void)flags;
    randomGet(buf, len);
    return (ssize_t)len;
}

// ---- abort leaves a note ---------------------------------------------------
// abort (a Rust panic, an allocation failure, an assert) runs exit, which
// closes HID under the game's thread: Atmosphère then reports that thread's
// fault in padUpdate, not the abort. so the aborting thread's return addresses
// go to crash.txt first, as offsets for addr2line, using raw open/write and a
// stack buffer so an out-of-memory or heap-corrupted abort never touches malloc.
#include <fcntl.h>

static int s_panic_logged = 0;
static int s_in_abort = 0;

void unx_note_panic_logged(void) {
    __atomic_store_n(&s_panic_logged, 1, __ATOMIC_RELEASE);
}

void unx_write_crash(const char *buf, size_t len) {
    int fd = open("sdmc:/switch/uprisingnx/crash.txt", O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd >= 0) {
        size_t off = 0;
        while (off < len) {
            ssize_t n = write(fd, buf + off, len - off);
            if (n <= 0) break;
            off += (size_t)n;
        }
        fsync(fd);
        close(fd);
    }
}

void unx_kernel_memory_mib(unsigned long long *used_mib, unsigned long long *total_mib) {
    u64 used = 0, total = 0;
    svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0);
    svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
    *used_mib = (unsigned long long)(used >> 20);
    *total_mib = (unsigned long long)(total >> 20);
}

void __real_abort(void) __attribute__((noreturn));
void _start(void);
void __wrap_abort(void) {
    if (!__atomic_load_n(&s_panic_logged, __ATOMIC_ACQUIRE)
        && __atomic_exchange_n(&s_in_abort, 1, __ATOMIC_SEQ_CST) == 0) {
        char buf[2048];
        unsigned long long used_mib = 0, total_mib = 0;
        unx_kernel_memory_mib(&used_mib, &total_mib);
        uintptr_t base = (uintptr_t)&_start;
        int n = snprintf(
            buf, sizeof buf,
            "ABORT on thread %p (mem %llu/%llu MiB, base 0x%lx), return addresses:\n",
            (void *)threadGetSelf(), used_mib, total_mib, (unsigned long)base
        );
        size_t len = (n > 0 && (size_t)n < sizeof buf) ? (size_t)n : 0;
        uintptr_t *fp = __builtin_frame_address(0);
        for (int i = 0; i < 48 && fp && ((uintptr_t)fp & 7) == 0 && len + 64 < sizeof buf; i++) {
            uintptr_t lr = fp[1];
            if (lr == 0) break;
            int w = (lr >= base)
                ? snprintf(buf + len, sizeof buf - len, "    uprising_nx+0x%lx\n", (unsigned long)(lr - base))
                : snprintf(buf + len, sizeof buf - len, "    0x%lx\n", (unsigned long)lr);
            if (w > 0 && len + (size_t)w < sizeof buf) len += (size_t)w;
            uintptr_t *next = (uintptr_t *)fp[0];
            if (next <= fp) break;
            fp = next;
        }
        unx_write_crash(buf, len);
    }
    __real_abort();
}
