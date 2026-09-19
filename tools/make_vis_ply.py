"""Create derived 3DGS PLYs to visualise labels in SuperSplat.

Outputs (in vis/):
    highlight.ply        whole scene, labelled classes recoloured, rest greyed
    <class>_only.ply     only the Gaussians of that class (original colours)
    unlabelled.ply       everything that has no label

The original point_cloud.ply is never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gs_common import OUT_DIR, SCENE_PLY, VIS_DIR, f_dc_to_rgb, load_scene, rgb_to_f_dc, write_vertex_ply

PALETTE = np.array([
    [1.00, 0.30, 0.20], [0.20, 0.90, 0.40], [0.25, 0.55, 1.00], [1.00, 0.85, 0.20],
    [0.90, 0.30, 0.90], [0.20, 0.90, 0.90], [1.00, 0.55, 0.10], [0.60, 0.80, 0.20],
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE_PLY)
    ap.add_argument("--label-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-dir", type=Path, default=VIS_DIR)
    ap.add_argument("--grey-opacity", type=float, default=1.0,
                    help="multiply opacity of unlabelled Gaussians by this (1 = unchanged). "
                         "Values < 1 expose individual Gaussians and look noisy; dim in the renderer instead.")
    ap.add_argument("--keep-colour", action="store_true",
                    help="leave unlabelled Gaussians in their original colour instead of grey")
    ap.add_argument("--grey-strength", type=float, default=1.0,
                    help="0 = original colour, 1 = fully grey (default 1)")
    ap.add_argument("--no-split", action="store_true", help="skip per-class / unlabelled PLYs")
    args = ap.parse_args()

    labels = np.load(args.label_dir / "labels.npy")
    classes = json.load(open(args.label_dir / "classes.json", encoding="utf-8"))
    ply, vertex = load_scene(args.scene)
    assert len(vertex) == len(labels), "labels.npy length != number of Gaussians"

    v = vertex.copy()
    rest_names = [n for n in v.dtype.names if n.startswith("f_rest_")]
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgb = f_dc_to_rgb(f_dc)

    # unlabelled -> desaturate only. Keep opacity and higher-order SH so the
    # surfaces stay smooth; dimming belongs in the renderer, not in the PLY.
    un = labels < 0
    if not args.keep_colour and args.grey_strength > 0:
        lum = (rgb[un] @ np.array([0.299, 0.587, 0.114]))[:, None]
        s = float(np.clip(args.grey_strength, 0, 1))
        rgb[un] = (1 - s) * rgb[un] + s * lum
        # higher-order SH are colour offsets; scale their chroma the same way
        # by averaging the three channels of each coefficient
        if rest_names and s > 0:
            k = len(rest_names) // 3
            coeffs = np.stack([v[n][un] for n in rest_names], axis=1).reshape(-1, 3, k)
            mean = coeffs.mean(axis=1, keepdims=True)
            coeffs = (1 - s) * coeffs + s * mean
            coeffs = coeffs.reshape(-1, len(rest_names))
            for i, n in enumerate(rest_names):
                v[n][un] = coeffs[:, i]
    if args.grey_opacity != 1.0:
        a = 1 / (1 + np.exp(-v["opacity"][un].astype(np.float64)))
        a = np.clip(a * args.grey_opacity, 1e-4, 1 - 1e-4)
        v["opacity"][un] = np.log(a / (1 - a)).astype(np.float32)

    # labelled -> flat palette colour, kill view-dependent SH so colour is stable
    lab = labels >= 0
    for cid_str in classes:
        cid = int(cid_str)
        m = labels == cid
        rgb[m] = PALETTE[cid % len(PALETTE)]
    new_dc = rgb_to_f_dc(rgb)
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = new_dc[:, 0], new_dc[:, 1], new_dc[:, 2]
    for n in rest_names:
        v[n][lab] = 0
    write_vertex_ply(v, args.out_dir / "highlight.ply", comments=ply.comments)
    print("wrote", args.out_dir / "highlight.ply")

    if not args.no_split:
        for cid_str, name in classes.items():
            m = labels == int(cid_str)
            write_vertex_ply(vertex[m], args.out_dir / f"{name}_only.ply", comments=ply.comments)
            print(f"wrote {name}_only.ply  ({m.sum()} gaussians)")
        write_vertex_ply(vertex[un], args.out_dir / "unlabelled.ply", comments=ply.comments)
        print(f"wrote unlabelled.ply  ({un.sum()} gaussians)")


if __name__ == "__main__":
    main()
