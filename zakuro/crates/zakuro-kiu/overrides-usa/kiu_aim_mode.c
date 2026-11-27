/*
 * Automatic mouse/gamepad aim mode reporting for Kid Icarus: Uprising (USA, 0004000000030100).
 *
 * Every tick the game dispatches touch/aiming input through a virtual call to either
 * the on-foot controller (0x006EA040) or the air-battle controller (0x00A3EDCC).
 * Writing GAME_ON_FOOT / GAME_FLYING to HOST_MAILBOX_VADDR + 12 lets the host
 * frontend switch between 1:1 mouse-look, 1:1 air reticle aim, and unlocked menu pointer.
 */
#include "overrides.h"

#define GAME_MODE    0x0700000Cu /* Zakuro's HOST_MAILBOX_VADDR + 12 */
#define GAME_ON_FOOT 1u
#define GAME_FLYING  2u

RECOMP_OVERRIDE(0x006EA040) {
    mem_write32(ctx, GAME_MODE, GAME_ON_FOOT);
    CALL(RECOMP_ORIGINAL(0x006EA040));
}

RECOMP_OVERRIDE(0x00A3EDCC) {
    mem_write32(ctx, GAME_MODE, GAME_FLYING);
    CALL(RECOMP_ORIGINAL(0x00A3EDCC));
}
