// UprisingNX: the libnx shell. Owns the process, the window, input and nxlink;
// the Rust side (uprising_nx) runs Zakuro and presents through NVK.
//   argv[1] ROM path (default: the copy Dekopon already has on the SD card)
//   argv[2] "interpret" to skip recompiled code
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <switch.h>

typedef struct {
    u32 buttons;
    float circle_x, circle_y, c_stick_x, c_stick_y;
    u32 touching;
    u16 touch_x, touch_y;
} UnxInput;

void *unx_create(void *nwindow, const char *rom, const char *data_dir, u32 interpret);
u32 unx_frame(void *unx, const UnxInput *input);
void unx_destroy(void *unx);
void nx_tls_init(void);

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
    // handheld touchscreen as the 3DS bottom screen, 320x240 centered for now
    HidTouchScreenState touch = {0};
    if (hidGetTouchScreenStates(&touch, 1) && touch.count > 0) {
        int x = (int)touch.touches[0].x - (1280 - 960) / 2, y = (int)touch.touches[0].y;
        if (x >= 0 && x < 960 && y >= 0 && y < 720) {
            in->touching = 1; in->touch_x = (u16)(x / 3); in->touch_y = (u16)(y / 3);
        }
    }
}

int main(int argc, char **argv) {
    socketInitializeDefault();
    nxlinkStdio();
    nx_tls_init();
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("uprising_nx: starting\n");

    const char *rom = argc > 1 ? argv[1] : "sdmc:/switch/dekopon/roms/Kid Icarus Uprising (USA).cci";
    u32 interpret = argc > 2 && !strcmp(argv[2], "interpret");

    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);
    hidInitializeTouchScreen();
    appletSetFocusHandlingMode(AppletFocusHandlingMode_SuspendHomeSleepNotify);

    void *unx = unx_create(nwindowGetDefault(), rom, "sdmc:/switch/uprisingnx/data", interpret);
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
        unx_destroy(unx);
    }
    printf("uprising_nx: exiting\n");
    fflush(stdout);
    socketExit();
    return 0;
}
