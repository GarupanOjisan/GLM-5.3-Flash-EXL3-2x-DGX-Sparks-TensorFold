#!/usr/bin/env python3
"""CPU check of the MPO guard (patch 0120, upstream #201).

An iPhone camera JPEG is an MPO: Pillow reports n_frames == 2, so the animated/multipage guard refused it
with HTTP 400. Frame 0 is the picture. Animated WebP stays refused.

  python3 tools/mpo_jpeg_check.py --source-root /path/to/patched/src
  python3 tools/mpo_jpeg_check.py --source-root /path/to/unpatched/src --expect-stock

No server, no GPU, no checkpoint. Needs Pillow.
"""
from __future__ import annotations

import argparse
import io
import sys

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--expect-stock", action="store_true",
                    help="the unpatched guard: a 2-frame MPO is refused")
    args = ap.parse_args()
    sys.path.insert(0, args.source_root)
    from tensorfold.vision.images import DEFAULT_LIMITS, ImageInputError, _decode

    primary = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(primary)
    for x in (140, 320, 500):
        draw.ellipse((x - 50, 190, x + 50, 290), fill="red")
    aux = Image.new("RGB", (160, 120), "gray")
    extra = Image.new("RGB", (80, 60), "blue")

    def save(fmt, frames, **kw):
        buf = io.BytesIO()
        frames[0].save(buf, format=fmt, save_all=True, append_images=list(frames[1:]), **kw)
        return buf.getvalue()

    mpo = save("MPO", [primary, aux])
    mpo3 = save("MPO", [primary.resize((100, 80)), aux, extra])
    jpeg = io.BytesIO()
    primary.save(jpeg, format="JPEG")
    jpeg = jpeg.getvalue()
    webp = save("WEBP", [Image.new("RGB", (32, 24), "red"), Image.new("RGB", (32, 24), "blue")], duration=100)

    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
        fails += not ok

    opened = Image.open(io.BytesIO(mpo))
    check("synthetic MPO is format MPO with 2 frames", opened.format == "MPO" and opened.n_frames == 2,
          f"{opened.format} n_frames={getattr(opened, 'n_frames', None)}")

    def decode(data):
        return _decode(data, "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels)

    if args.expect_stock:
        try:
            decode(mpo)
            check("stock guard refuses a 2-frame MPO", False, "accepted")
        except ImageInputError as exc:
            check("stock guard refuses a 2-frame MPO", "single frame" in str(exc), str(exc))
    else:
        try:
            got = decode(mpo)
        except ImageInputError as exc:
            got = None
            check("2-frame MPO is accepted at the first frame's size", False, str(exc))
        if got is not None:
            # 640x480 is frame 0. Frame 1 is the 160x120 auxiliary image. JPEG is lossy, so the bytes
            # are not the canvas we drew; the size is what says which frame was decoded.
            check("2-frame MPO is accepted at the first frame's size",
                  (got.width, got.height) == (640, 480), f"{got.width}x{got.height}")
        try:
            got3 = decode(mpo3)
        except ImageInputError as exc:
            got3 = None
            check("3-frame MPO is accepted at frame 0's size", False, str(exc))
        if got3 is not None:
            check("3-frame MPO is accepted at frame 0's size", (got3.width, got3.height) == (100, 80),
                  f"{got3.width}x{got3.height}")

    got_j = decode(jpeg)
    check("single-frame JPEG is accepted at the same size", (got_j.width, got_j.height) == (640, 480))

    try:
        decode(webp)
        check("animated WebP is still refused", False, "accepted")
    except ImageInputError as exc:
        check("animated WebP is still refused", "single frame" in str(exc), str(exc))

    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
