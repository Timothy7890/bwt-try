"""Pack the device's collision.ply (triangle mesh) into a compact binary for the front-end.

Layout (little-endian):
    uint32 nv, uint32 nf, float32[nv*3] positions, uint32[nf*3] triangle indices
Coordinates stay in the source (Z-up) frame, same as point_cloud.ply.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData

from gs_common import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DATA_DIR / "collision.ply")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "labels" / "collision.bin")
    args = ap.parse_args()

    ply = PlyData.read(str(args.src))
    v = ply["vertex"].data
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype("<f4")
    faces = ply["face"].data["vertex_indices"]
    tris = []
    for f in faces:  # fan-triangulate anything that is not a triangle
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    idx = np.asarray(tris, dtype="<u4")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<II", len(pos), len(idx)))
        fh.write(pos.tobytes())
        fh.write(idx.tobytes())
    print(f"{len(pos)} vertices, {len(idx)} triangles -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print("bbox", pos.min(0).round(2), pos.max(0).round(2))


if __name__ == "__main__":
    main()
