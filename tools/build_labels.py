"""Build labels.npy / classes.json / metadata.json from manual segment exports.

Usage:
    python tools/build_labels.py [--label-dir DIR] [--tol 1e-4]

Every file in the label dir (plant.ply, floor.txt, ...) becomes one class named
after the file stem. Points are mapped back to original PLY rows either by an
"index" scalar field (preferred) or by nearest-neighbour xyz matching.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gs_common import LABEL_DIR, OUT_DIR, SCENE_PLY, load_scene, read_segment, scene_xyz

SEG_SUFFIXES = {".ply", ".txt", ".asc", ".csv", ".xyz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE_PLY)
    ap.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tol", type=float, default=1e-4, help="xyz match tolerance (metres)")
    args = ap.parse_args()

    seg_files = sorted(p for p in args.label_dir.iterdir()
                       if p.suffix.lower() in SEG_SUFFIXES and not p.name.startswith("."))
    if not seg_files:
        raise SystemExit(f"no segment files in {args.label_dir}")

    print(f"loading scene {args.scene} ...")
    _, vertex = load_scene(args.scene)
    n = len(vertex)
    xyz = scene_xyz(vertex)
    tree = None

    labels = np.full(n, -1, dtype=np.int32)
    classes = {}
    report = {}

    for cid, seg in enumerate(seg_files):
        name = seg.stem
        sxyz, sidx = read_segment(seg)
        m = len(sxyz)
        if sidx is not None and sidx.min() >= 0 and sidx.max() < n:
            rows = sidx
            method = "index"
            matched = m
        else:
            if tree is None:
                print("building KD-tree over scene xyz ...")
                tree = cKDTree(xyz)
            d, rows = tree.query(sxyz, k=1, distance_upper_bound=args.tol)
            ok = np.isfinite(d)
            rows = rows[ok]
            matched = int(ok.sum())
            method = "xyz"
        rows = np.unique(rows)
        conflict = int((labels[rows] >= 0).sum())
        labels[rows] = cid
        classes[cid] = name
        report[name] = {"file": seg.name, "points_in_file": m, "matched": matched,
                        "unique_rows": int(len(rows)), "overwrote": conflict, "method": method}
        print(f"[{cid}] {name:<20} {matched}/{m} matched ({method}), "
              f"{len(rows)} rows, overwrote {conflict}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "labels.npy", labels)
    with open(args.out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in classes.items()}, f, ensure_ascii=False, indent=2)
    meta = {
        "scene_ply": str(args.scene),
        "num_gaussians": n,
        "labelled": int((labels >= 0).sum()),
        "source_up_axis": "Z",
        "supersplat_display_rotation_deg": {"x": -90, "y": 0, "z": 0},
        "generated": datetime.now().isoformat(timespec="seconds"),
        "segments": report,
    }
    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out_dir}/labels.npy  ({meta['labelled']}/{n} labelled)")


if __name__ == "__main__":
    main()
