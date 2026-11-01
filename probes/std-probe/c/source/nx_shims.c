// POSIX pieces Rust std calls that libnx/newlib don't provide.
#include <errno.h>
#include <unistd.h>

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
    return start.fn(start.arg);
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
