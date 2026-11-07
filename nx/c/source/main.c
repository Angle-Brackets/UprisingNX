// UprisingNX: the libnx shell. Owns the process, the window, input and nxlink;
// the Rust side (uprising_nx) runs Zakuro and presents through NVK.
//   argv[1] ROM path (default: the copy Dekopon already has on the SD card)
//   argv[2] "interpret" to skip recompiled code
// Built with RECOMP=1 the NRO links 3dsrecomp's librecomp.a and runs it.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <switch.h>

typedef struct {
    u32 buttons;
    float circle_x, circle_y, c_stick_x, c_stick_y;
    u32 touching;
    u16 touch_x, touch_y;
} UnxInput;

typedef struct {
    u32 abi;
    const void *entries;
    u32 entry_count;
    const void *modules;
    u32 module_count;
} UnxRecomp;

#ifdef UNX_RECOMP
extern const u32 recomp_abi, recomp_entry_count, recomp_module_count;
extern const u8 recomp_entries, recomp_modules;
#endif

void *unx_create(void *nwindow, const char *rom, const char *data_dir, const UnxRecomp *recomp);
u32 unx_frame(void *unx, const UnxInput *input);
void unx_destroy(void *unx);
void unx_set_font(const u8 *data, size_t len);
void nx_tls_init(void);
void prof_start(void);
void prof_stop(void);

// 3DS PadState bits, from zakuro-core services/hid.rs
enum {
    P_A = 1 << 0, P_B = 1 << 1, P_SELECT = 1 << 2, P_START = 1 << 3,
    P_RIGHT = 1 << 4, P_LEFT = 1 << 5, P_UP = 1 << 6, P_DOWN = 1 << 7,
    P_R = 1 << 8, P_L = 1 << 9, P_X = 1 << 10, P_Y = 1 << 11,
    P_ZL = 1 << 14, P_ZR = 1 << 15,
};

static float axis(s32 v) { return fmaxf(-1.0f, fminf(1.0f, v / 32767.0f)); }

static void read_input(PadState *pad, UnxInput *in) {
    padUpdate(pad);
    u64 held = padGetButtons(pad);
    // Nintendo layout matches the 3DS's: A right, B bottom, X top, Y left
    static const struct { u64 nx; u32 ds; } map[] = {
        {HidNpadButton_A, P_A}, {HidNpadButton_B, P_B}, {HidNpadButton_X, P_X}, {HidNpadButton_Y, P_Y},
        {HidNpadButton_L, P_L}, {HidNpadButton_R, P_R}, {HidNpadButton_ZL, P_ZL}, {HidNpadButton_ZR, P_ZR},
        {HidNpadButton_Plus, P_START}, {HidNpadButton_Minus, P_SELECT},
        {HidNpadButton_Up, P_UP}, {HidNpadButton_Down, P_DOWN}, {HidNpadButton_Left, P_LEFT}, {HidNpadButton_Right, P_RIGHT},
    };
    memset(in, 0, sizeof *in);
    for (size_t i = 0; i < sizeof map / sizeof *map; i++)
        if (held & map[i].nx) in->buttons |= map[i].ds;
    HidAnalogStickState l = padGetStickPos(pad, 0), r = padGetStickPos(pad, 1);
    in->circle_x = axis(l.x); in->circle_y = axis(l.y);
    in->c_stick_x = axis(r.x); in->c_stick_y = axis(r.y);
    // the touchscreen's first finger, in pixels of the 1280x720 screen; the
    // Rust side maps it onto the 3DS bottom screen for the layout on show
    HidTouchScreenState touch = {0};
    if (hidGetTouchScreenStates(&touch, 1) && touch.count > 0) {
        in->touching = 1;
        in->touch_x = (u16)touch.touches[0].x;
        in->touch_y = (u16)touch.touches[0].y;
    }
}

struct args { const char *rom; u32 interpret; };

// KEY=VALUE lines from sdmc:/switch/uprisingnx/env.txt become environment
// variables, which Zakuro reads for its switches (ZAKURO_GPU_TIMES,
// ZAKURO_GPU_SHADERS, RUST_LOG...), to try them without a rebuild. # starts a
// comment.
static void load_env(void) {
    // the GPU on a thread of its own, unless env.txt says ZAKURO_GPU_THREAD=0
    setenv("ZAKURO_GPU_THREAD", "1", 0);
    // the top screen's read-back to guest memory is skipped (the picture is shown
    // from the GPU), 22 -> 30 fps in the land battle; ZAKURO_SKIP_SCREEN_SYNC is
    // read by presence, so env.txt cannot turn it off: remove this line to
    setenv("ZAKURO_SKIP_SCREEN_SYNC", "1", 0);
    // self-sampling draws (the water/ripple moire) issued in 8 runs so each
    // run sees the image the earlier ones drew; ZAKURO_SELF_CHUNKS=1 turns it off
    setenv("ZAKURO_SELF_CHUNKS", "8", 0);
    // aiming with the left stick, see nx/rust/src/aim.rs: 0 off, 1 on (default),
    // 2 flying aims only while the modifier (UNX_AIM_MODIFIER, default ZL) is held
    setenv("UNX_LEFT_STICK_AIM", "1", 0);
    FILE *f = fopen("sdmc:/switch/uprisingnx/env.txt", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        char *eq = strchr(line, '=');
        if (line[0] == '#' || !eq) continue;
        *eq = 0;
        setenv(line, eq + 1, 1);
        printf("env: %s=%s\n", line, eq + 1);
    }
    fclose(f);
}

// the game loop, on a thread with a big stack: recompiled code nests a C
// frame per guest call (up to RECOMP_DEPTH_LIMIT deep), and the desktop port
// gives its main thread 8 MiB for that
static void *game(void *p) {
    const struct args *args = p;
    const UnxRecomp *recomp = NULL;
#ifdef UNX_RECOMP
    static UnxRecomp linked;
    linked = (UnxRecomp){recomp_abi, &recomp_entries, recomp_entry_count, &recomp_modules, recomp_module_count};
    if (!args->interpret) recomp = &linked;
#endif

    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    svcSetThreadCoreMask(CUR_THREAD_HANDLE, 0, 0x7);
    PadState pad;
    padInitializeDefault(&pad);
    hidInitializeTouchScreen();
    appletSetFocusHandlingMode(AppletFocusHandlingMode_SuspendHomeSleepNotify);

    load_env();
    // UNX_CLOCKS=docked (or a hex PerformanceConfiguration such as 0x92220008) asks apm for
    // the docked clocks (CPU 1020 MHz, GPU 768 MHz, stock docked values) while running
    // handheld; the session stays open for the run, since the setting may go with it
    {
        const char *want = getenv("UNX_CLOCKS");
        if (want && *want) {
            u32 config = !strcmp(want, "docked") ? 0x92220008u : (u32)strtoul(want, NULL, 0);
            Result rc = apmInitialize();
            printf("clocks: apmInitialize %08x\n", (unsigned)rc);
            if (R_SUCCEEDED(rc)) {
                Result a = apmSetPerformanceConfiguration(ApmPerformanceMode_Normal, config);
                Result b = apmSetPerformanceConfiguration(ApmPerformanceMode_Boost, config);
                printf("clocks: asked for config %08x: normal %08x, boost %08x\n", (unsigned)config, (unsigned)a, (unsigned)b);
                u32 got = 0;
                for (int m = 0; m < 2; m++) {
                    Result g = apmGetPerformanceConfiguration((ApmPerformanceMode)m, &got);
                    printf("clocks: mode %d config now %08x (%08x)\n", m, (unsigned)got, (unsigned)g);
                }
                u32 cur = 0;
                if (R_SUCCEEDED(appletGetCurrentPerformanceConfiguration(&cur))) printf("clocks: applet current config %08x\n", (unsigned)cur);
            }
        }
    }
    // the Switch's own system font stands in for the 3DS shared font unless
    // the user placed a shared_font.bin or font.ttf (UNX_SYSTEM_FONT=0 to skip)
    if (!getenv("UNX_SYSTEM_FONT") || strcmp(getenv("UNX_SYSTEM_FONT"), "0")) {
        if (R_SUCCEEDED(plInitialize(PlServiceType_User))) {
            PlFontData font;
            if (R_SUCCEEDED(plGetSharedFontByType(&font, PlSharedFontType_Standard))) {
                printf("font: Switch standard system font, %u bytes\n", font.size);
                unx_set_font(font.address, font.size);
            } else printf("font: plGetSharedFontByType failed\n");
            plExit();
        } else printf("font: pl:u unavailable\n");
    }
    void *unx = unx_create(nwindowGetDefault(), args->rom, "sdmc:/switch/uprisingnx/data", recomp);
    if (getenv("UNX_PROF")) prof_start();
    if (!unx) {
        printf("uprising_nx: could not start, see above\n");
    } else {
        UnxInput in;
        while (appletMainLoop()) {
            read_input(&pad, &in);
            // + and - together quits (the game uses both alone)
            if ((in.buttons & (P_START | P_SELECT)) == (P_START | P_SELECT)) break;
            if (unx_frame(unx, &in)) break;
        }
        if (getenv("UNX_PROF")) prof_stop();
        unx_destroy(unx);
    }
    return NULL;
}

int main(int argc, char **argv) {
    socketInitializeDefault();
    nxlinkStdio();
    nx_tls_init();
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("uprising_nx: starting\n");
    // read-only: which mode and clocks this run has
    {
        printf("clocks: operation mode %d\n", (int)appletGetOperationMode());
        if (R_SUCCEEDED(apmInitialize())) {
            ApmPerformanceMode pm;
            if (R_SUCCEEDED(apmGetPerformanceMode(&pm))) printf("clocks: apm performance mode %d (0 handheld, 1 docked)\n", (int)pm);
            apmExit();
        }
        if (R_SUCCEEDED(clkrstInitialize())) {
            const PcvModule mods[] = {PcvModule_CpuBus, PcvModule_GPU, PcvModule_EMC};
            const char *names[] = {"cpu", "gpu", "emc"};
            for (int i = 0; i < 3; i++) {
                ClkrstSession sess;
                u32 hz = 0;
                if (R_SUCCEEDED(clkrstOpenSession(&sess, mods[i], 3)) ) {
                    clkrstGetClockRate(&sess, &hz);
                    clkrstCloseSession(&sess);
                    printf("clocks: %s %u MHz\n", names[i], hz / 1000000);
                } else printf("clocks: %s session refused\n", names[i]);
            }
            clkrstExit();
        } else printf("clocks: clkrst unavailable\n");
    }

    struct args args = {
        argc > 1 ? argv[1] : "sdmc:/switch/dekopon/roms/Kid Icarus Uprising (USA).cci",
        argc > 2 && !strcmp(argv[2], "interpret"),
    };
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 << 20);
    pthread_t thread;
    if (pthread_create(&thread, &attr, game, &args) != 0) {
        printf("uprising_nx: could not start the game thread\n");
    } else {
        pthread_join(thread, NULL);
    }
    printf("uprising_nx: exiting\n");
    fflush(stdout);
    socketExit();
    return 0;
}
