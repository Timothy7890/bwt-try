"""Write one small recoloured splat file per class for the front-end "mask" mode.

Each masks/<name>.ply holds only that class's Gaussians (original order preserved), with a
flat class colour, view-dependent SH removed and opacity scaled. The viewer draws these on
top of the untouched scene, giving a tinted overlay that can be toggled per object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gs_common import OUT_DIR, SCENE_PLY, load_scene, rgb_to_f_dc, write_vertex_ply

PALETTE_HEX = ["#ff4d33", "#33e666", "#4088ff", "#ffd933", "#e64de6", "#33e6e6", "#ff8c1a", "#99cc33"]


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE_PLY)
    ap.add_argument("--label-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "masks")
    ap.add_argument("--opacity", type=float, default=1.0,
                    help="multiply opacity of mask Gaussians (1 = as original)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply Gaussian size so the tint fully covers the original")
    args = ap.parse_args()

    labels = np.load(args.label_dir / "labels.npy")
    classes = json.load(open(args.label_dir / "classes.json", encoding="utf-8"))
    objects = {o["name"]: o for o in json.load(open(args.label_dir / "objects.json", encoding="utf-8"))["objects"]}
    ply, vertex = load_scene(args.scene)
    rest_names = [n for n in vertex.dtype.names if n.startswith("f_rest_")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for cid_str, name in classes.items():
        m = labels == int(cid_str)
        if not m.any():
            continue
        v = vertex[m].copy()
        colour = hex_to_rgb(objects.get(name, {}).get("color", PALETTE_HEX[int(cid_str) % len(PALETTE_HEX)]))
        dc = rgb_to_f_dc(colour)
        v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = dc[0], dc[1], dc[2]
        for n in rest_names:
            v[n] = 0
        if args.opacity != 1.0:
            a = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
            a = np.clip(a * args.opacity, 1e-4, 1 - 1e-4)
            v["opacity"] = np.log(a / (1 - a)).astype(np.float32)
        if args.scale != 1.0:
            for n in ("scale_0", "scale_1", "scale_2"):
                v[n] = (v[n] + np.log(args.scale)).astype(np.float32)
        path = args.out_dir / f"{name}.ply"
        write_vertex_ply(v, path, comments=ply.comments)
        manifest[name] = {"file": f"masks/{name}.ply", "count": int(m.sum())}
        print(f"{name:<20} {m.sum():>8} gaussians -> {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    with open(args.out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("wrote", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
