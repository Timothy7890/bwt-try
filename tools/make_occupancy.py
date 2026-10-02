"""Build a voxel occupancy grid from the Gaussians themselves, for overlay occlusion tests.

The device's collision mesh has gaps (walls missing in places), so the front-end instead
asks "is there solid scene between the camera and this point?" against a bit-packed grid of
voxels that contain enough opaque Gaussians.

Output occupancy.bin (little-endian):
    float32 origin[3], float32 voxel_size, uint32 dims[3], uint8 bits[ceil(nx*ny*nz/8)]
    index = x + nx * (y + ny * z), bit (index & 7) of byte (index >> 3)
Coordinates are the source (Z-up) frame.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from gs_common import DATA_DIR, SCENE_PLY, load_scene, scene_xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE_PLY)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "labels" / "occupancy.bin")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="voxel size in metres; grown automatically if the grid would exceed --max-voxels")
    ap.add_argument("--max-voxels", type=float, default=8e7,
                    help="voxel budget (bits); keeps large outdoor scenes to ~10 MB")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--min-count", type=int, default=1, help="Gaussians per voxel to count as solid")
    ap.add_argument("--max-scale", type=float, default=2.0,
                    help="ignore Gaussians whose largest axis exceeds this (metres); they are fog/floaters")
    ap.add_argument("--dilate", type=int, default=0, help="binary dilation iterations (closes small holes)")
    args = ap.parse_args()

    _, vertex = load_scene(args.scene)
    xyz = scene_xyz(vertex)
    opacity = 1 / (1 + np.exp(-vertex["opacity"].astype(np.float64)))
    scale = np.exp(np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], 1).astype(np.float64)).max(1)
    keep = (opacity > args.min_opacity) & (scale < args.max_scale)
    pts = xyz[keep]
    print(f"{keep.sum()}/{len(xyz)} Gaussians used")

    # bounds from the solid Gaussians only, so far-away floaters do not blow up the grid
    ext = pts.max(0) - pts.min(0)
    voxel = args.voxel
    if np.prod(ext / voxel + 2) > args.max_voxels:
        voxel = float(np.cbrt(np.prod(ext) / args.max_voxels))
        voxel = float(np.ceil(voxel / 0.05) * 0.05)
        print(f"scene extent {np.round(ext, 1)} m too large for {args.voxel} m voxels -> using {voxel:.2f} m")
    lo = pts.min(0) - voxel
    hi = pts.max(0) + voxel
    dims = np.ceil((hi - lo) / voxel).astype(np.int64)
    nx, ny, nz = (int(d) for d in dims)
    print("grid", (nx, ny, nz), "=", nx * ny * nz, "voxels", f"@ {voxel:.2f} m")

    ijk = np.minimum(np.floor((pts - lo) / voxel).astype(np.int64), dims - 1)
    flat = ijk[:, 0] + nx * (ijk[:, 1] + ny * ijk[:, 2])
    uniq, counts = np.unique(flat, return_counts=True)
    solid = uniq[counts >= args.min_count]
    print(f"{len(solid)} solid voxels ({len(solid) / (nx * ny * nz):.1%})")

    if args.dilate > 0:
        from scipy.ndimage import binary_dilation
        grid = np.zeros(nx * ny * nz, dtype=bool)
        grid[solid] = True
        grid = grid.reshape(nz, ny, nx)  # index = x + nx*(y + ny*z)
        grid = binary_dilation(grid, iterations=args.dilate)
        solid = np.nonzero(grid.ravel())[0]
        print(f"{len(solid)} solid voxels after dilation")

    bits = np.zeros((nx * ny * nz + 7) // 8, dtype=np.uint8)
    np.bitwise_or.at(bits, solid >> 3, (1 << (solid & 7)).astype(np.uint8))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(struct.pack("<3ff3I", *lo.astype(np.float32), voxel, nx, ny, nz))
        fh.write(bits.tobytes())
    print("wrote", args.out, f"{args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
