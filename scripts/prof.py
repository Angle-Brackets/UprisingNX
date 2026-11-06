#!/usr/bin/env python3
"""Rank the samples nx/c's profiler wrote (prof.bin) against the NRO's ELF.

  scripts/prof.py prof.bin nx/c/uprising_nx.elf [top=40]

Each sample is a thread's PC and LR as offsets into the image; the game thread's
by default, the GPU's with a fourth argument, 1. PCs are
mapped to the enclosing symbol from `nm` (the real, non-inlined function;
recompiled guest code shows as f_XXXXXXXX), grouped by component, and the
hottest few get their inline chain from addr2line.
"""
import bisect
import collections
import struct
import subprocess
import sys

BIN = "/opt/devkitpro/devkitA64/bin/aarch64-none-elf-"


def load(path, main_addr):
    """samples as offsets into the image: v2 files hold raw addresses and the
    runtime address of main(), whose ELF address gives the load base."""
    d = open(path, "rb").read()
    magic, ver, count, failed = struct.unpack_from("<4I", d, 0)
    assert magic == 0x464F5250 and ver == 2, "not a v2 prof.bin"
    base = struct.unpack_from("<Q", d, 16)[0] - main_addr
    off = lambda a: (a - base) & 0xFFFFFFFFFFFFFFFF
    raw = [struct.unpack_from("<QQ", d, 24 + 16 * i) for i in range(count)]
    # the thread's number is in the top bits of the pc, 0 for the game's
    return [((pc >> 56, off(pc & ((1 << 56) - 1))), off(lr)) for pc, lr in raw], failed


def symbols(elf):
    out = subprocess.run([BIN + "nm", "-C", "--defined-only", "-n", elf], capture_output=True, text=True).stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] in "tTwW":
            addrs.append(int(parts[0], 16))
            names.append(parts[2])
    return addrs, names


def component(name):
    if name.startswith("f_") or name.startswith("kiu_"):
        return "game code (recompiled)"
    for prefix, label in [
        ("zakuro_gpu", "zakuro-gpu (PICA emulation)"), ("zakuro_core::services", "zakuro services"),
        ("zakuro_core::kernel", "zakuro kernel"), ("zakuro_core::memory", "zakuro memory"),
        ("zakuro_core::recompiled", "zakuro recomp glue"), ("zakuro_cpu", "zakuro interpreter"),
        ("zakuro_core", "zakuro core (other)"), ("naga", "naga (shader translation)"),
        ("hashbrown", "hashbrown"), ("alloc::", "Rust alloc"), ("core::", "Rust core"), ("std::", "Rust std"),
        ("nvk_", "NVK"), ("nv_push", "NVK"), ("vk_", "Mesa Vulkan runtime"), ("nir_", "Mesa NIR"),
        ("nak", "Mesa NAK"), ("nouveau_ws", "NVK winsys"), ("mem", "libc mem*"), ("svc", "libnx svc"),
        ("_ZN", "C++"),
    ]:
        if name.startswith(prefix) or name.lstrip("<").startswith(prefix):
            return label
    return "other"


def main():
    elf = sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    addrs, names = symbols(elf)
    samples, failed = load(sys.argv[1], addrs[names.index("main")])
    # prof.py prof.bin elf [top] [thread]: thread 0 is the game's, 1 the GPU's
    want = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    samples = [(pc, lr) for (tid, pc), lr in samples if tid == want]

    def sym(a):
        i = bisect.bisect_right(addrs, a) - 1
        return names[i] if i >= 0 else "?"

    n = len(samples)
    print(f"{n} samples ({failed} failed)\n")
    by_fn = collections.Counter(sym(pc) for pc, _ in samples)
    by_comp = collections.Counter(component(sym(pc)) for pc, _ in samples)
    print("by component:")
    for name, c in by_comp.most_common():
        print(f"  {100 * c / n:5.1f}%  {name}")
    print(f"\ntop {top} functions:")
    for name, c in by_fn.most_common(top):
        print(f"  {100 * c / n:5.1f}%  {name[:150]}")

    # callers of the hottest few, by LR
    print("\ncallers of the top 12:")
    callers = collections.defaultdict(collections.Counter)
    for pc, lr in samples:
        callers[sym(pc)][sym(lr)] += 1
    for name, c in by_fn.most_common(12):
        print(f"  {100 * c / n:5.1f}%  {name[:110]}")
        for caller, cc in callers[name].most_common(3):
            print(f"           {100 * cc / c:4.0f}% from {caller[:110]}")

    # inline chains for the hottest PCs
    hot = collections.Counter(pc for pc, _ in samples).most_common(10)
    print("\nhottest PCs, inline chains:")
    res = subprocess.run([BIN + "addr2line", "-f", "-C", "-i", "-e", elf] + [hex(pc) for pc, _ in hot], capture_output=True, text=True).stdout
    # addr2line -i prints a variable number of lines per address; re-run one at a time for clarity
    for pc, c in hot:
        chain = subprocess.run([BIN + "addr2line", "-f", "-C", "-i", "-e", elf, hex(pc)], capture_output=True, text=True).stdout.splitlines()
        funcs = [chain[i] for i in range(0, len(chain), 2)]
        print(f"  {100 * c / n:5.1f}%  {hex(pc)}  " + " <- ".join(f[:60] for f in funcs[:5]))


if __name__ == "__main__":
    main()
