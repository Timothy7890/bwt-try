"""Export the collision mesh as a GLB for the viewer's walk mode.

The viewer reads raw vertex positions as world coordinates, so the display rotation of the
splat (source Z-up -> viewer Y-up: (x, y, z) -> (x, z, -y)) is baked into the vertices here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

from gs_common import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DATA_DIR / "collision.ply")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "labels" / "collision.glb")
    args = ap.parse_args()

    ply = PlyData.read(str(args.src))
    v = ply["vertex"].data
    src = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    world = np.stack([src[:, 0], src[:, 2], -src[:, 1]], axis=1)  # X -90deg
    faces = np.asarray([list(f) for f in ply["face"].data["vertex_indices"]])

    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.out))
    print(f"{len(world)} vertices, {len(faces)} faces -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print("world bbox min", world.min(0).round(2), "max", world.max(0).round(2))


if __name__ == "__main__":
    main()
