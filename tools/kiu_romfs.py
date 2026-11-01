#!/usr/bin/env python3
"""Inspect KIU's RomFS containers: .zrc (LZ11-compressed darc) and .arc (darc).

  kiu_romfs.py ls <file.zrc|file.arc> [glob]      list entries (path, size, texture info)
  kiu_romfs.py get <file> <entry path> <out>      write one entry out

Texture info is read from a CTR image's footer (bclim/bflim "CLIM"/"FLIM" with an
"imag" block): width, height, format. Run with python3 -I on the user's own
extracted RomFS; nothing here ships game data.
"""
import fnmatch
import struct
import sys


def lz11(d):
    if d[0] != 0x11:
        raise ValueError("not LZ11")
    size = d[1] | d[2] << 8 | d[3] << 16
    pos = 4
    if size == 0:
        size = struct.unpack_from("<I", d, 4)[0]
        pos = 8
    out = bytearray()
    while len(out) < size:
        flags = d[pos]
        pos += 1
        for bit in range(8):
            if len(out) >= size:
                break
            if not flags & (0x80 >> bit):
                out.append(d[pos])
                pos += 1
                continue
            b = d[pos]
            ind = b >> 4
            if ind == 0:
                length = (((b & 0xF) << 4) | (d[pos + 1] >> 4)) + 0x11
                disp = ((d[pos + 1] & 0xF) << 8 | d[pos + 2]) + 1
                pos += 3
            elif ind == 1:
                length = (((b & 0xF) << 12) | (d[pos + 1] << 4) | (d[pos + 2] >> 4)) + 0x111
                disp = ((d[pos + 2] & 0xF) << 8 | d[pos + 3]) + 1
                pos += 4
            else:
                length = ind + 1
                disp = ((b & 0xF) << 8 | d[pos + 1]) + 1
                pos += 2
            start = len(out) - disp
            for i in range(length):
                out.append(out[start + i])
    return bytes(out)


def darc_entries(d):
    """Yields (path, offset, size) for every file in a darc."""
    assert d[:4] == b"darc", d[:4]
    table_off = struct.unpack_from("<I", d, 0x10)[0]
    # entry 0 is the root dir; its size field is the total entry count
    root_count = struct.unpack_from("<I", d, table_off + 8)[0]
    names_off = table_off + root_count * 12
    entries = [struct.unpack_from("<III", d, table_off + i * 12) for i in range(root_count)]

    def name(i):
        o = names_off + (entries[i][0] & 0xFFFFFF)
        end = o
        while d[end:end + 2] != b"\0\0":
            end += 2
        return d[o:end].decode("utf-16-le")

    stack = []  # (end index, path)
    for i in range(1, root_count):
        while stack and i >= stack[-1][0]:
            stack.pop()
        word, a, b = entries[i]
        nm = name(i)
        prefix = "/".join(p for _, p in stack)
        path = f"{prefix}/{nm}" if prefix else nm
        if word & 0x01000000:  # directory: b = index one past its last entry
            stack.append((b, nm))
        else:
            yield path, a, b


FORMATS = {0: "L8", 1: "A8", 2: "LA4", 3: "LA8", 4: "HILO8", 5: "RGB565", 6: "RGB8",
           7: "RGBA5551", 8: "RGBA4", 9: "RGBA8", 10: "ETC1", 11: "ETC1A4", 12: "L4", 13: "A4"}


def tex_info(blob):
    for magic in (b"CLIM", b"FLIM"):
        i = blob.rfind(magic)
        if i >= 0 and blob.rfind(b"imag") > i:
            j = blob.rfind(b"imag")
            w, h = struct.unpack_from("<HH", blob, j + 8)
            fmt = blob[j + 12] if magic == b"CLIM" else blob[j + 12]
            return f"{magic.decode()} {w}x{h} {FORMATS.get(fmt, fmt)}"
    return ""


def load(path):
    d = open(path, "rb").read()
    return lz11(d) if d[:1] == b"\x11" else d


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    d = load(path)
    if cmd == "ls":
        pat = sys.argv[3] if len(sys.argv) > 3 else "*"
        for p, off, size in darc_entries(d):
            if fnmatch.fnmatch(p, pat):
                print(f"{size:9d}  {p}  {tex_info(d[off:off + size])}")
    elif cmd == "get":
        for p, off, size in darc_entries(d):
            if p == sys.argv[3]:
                open(sys.argv[4], "wb").write(d[off:off + size])
                return
        sys.exit("no such entry")


if __name__ == "__main__":
    main()
