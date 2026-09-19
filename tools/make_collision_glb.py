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

    # The device mesh has dips/holes in the floor; add a flat floor at the Gaussian floor height
    # (labels/objects.json: floor_z) over the whole hall so the walk camera never drops through.
    objects_json = DATA_DIR / "labels" / "objects.json"
    if objects_json.exists():
        import json
        meta = json.load(open(objects_json, encoding="utf-8"))
        fz = meta.get("floor_z")
        if fz is not None:
            lo = np.array(meta["scene_bbox_min"]) - 1.0
            hi = np.array(meta["scene_bbox_max"]) + 1.0
            # source frame corners (x, y, z=fz) -> world (x, fz, -y); two triangles, facing up
            quad = np.array([[lo[0], lo[1], fz], [hi[0], lo[1], fz], [hi[0], hi[1], fz], [lo[0], hi[1], fz]])
            quad_w = np.stack([quad[:, 0], quad[:, 2], -quad[:, 1]], axis=1)
            floor = trimesh.Trimesh(vertices=quad_w, faces=[[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]], process=False)  # double-sided
            mesh = trimesh.util.concatenate([mesh, floor])
            print(f"added flat floor at z={fz:.3f} (world y) covering x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.out))
    print(f"{len(world)} vertices, {len(faces)} faces -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print("world bbox min", world.min(0).round(2), "max", world.max(0).round(2))


if __name__ == "__main__":
    main()
