"""Compute per-class geometry (bounding box, centre, anchor) for the front-end overlay.

Reads labels/labels.npy + classes.json, writes labels/objects.json in the ORIGINAL PLY
coordinate frame (Z-up). The front-end parents the overlay to the splat entity so the
same display rotation applies automatically.

Usage:
    python tools/make_objects.py [--pad 0.03] [--trim 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gs_common import OUT_DIR, SCENE_PLY, load_scene, scene_xyz

PALETTE_HEX = ["#ff4d33", "#33e666", "#4088ff", "#ffd933", "#e64de6", "#33e6e6", "#ff8c1a", "#99cc33"]

import re

NAME_RE = re.compile(r"^(?P<category>.+?)[-_ ]+(?P<instance>\d+)$")


def split_name(name: str):
    """'面板-2' -> ('面板', 2); '植物' -> ('植物', None)."""
    m = NAME_RE.match(name)
    if not m:
        return name, None
    return m.group("category"), int(m.group("instance"))


def robust_bounds(pts: np.ndarray, trim_pct: float, pad: float):
    """Bounding box after trimming `trim_pct` percent of outliers on each side."""
    lo = np.percentile(pts, trim_pct, axis=0) - pad
    hi = np.percentile(pts, 100 - trim_pct, axis=0) + pad
    return lo, hi


def oriented_box(pts: np.ndarray, trim_pct: float, pad: float):
    """Yaw-aligned oriented box (Z stays up): PCA of the xy footprint gives the two horizontal
    axes, so a panel that is 2° off the world axes does not get a fat axis-aligned box.

    Returns (axes[3,3] rows = unit vectors, centre[3], half_size[3], corners[8,3]).
    Corner i: bit0 -> axis0 max, bit1 -> axis1 max, bit2 -> z max.
    """
    xy = pts[:, :2]
    mean = xy.mean(0)
    cov = np.cov((xy - mean).T)
    _, vec = np.linalg.eigh(cov)
    a0 = vec[:, 1]  # longest horizontal direction
    a1 = np.array([-a0[1], a0[0]])
    axes = np.array([[a0[0], a0[1], 0.0], [a1[0], a1[1], 0.0], [0.0, 0.0, 1.0]])
    proj = pts @ axes.T
    lo = np.percentile(proj, trim_pct, axis=0) - pad
    hi = np.percentile(proj, 100 - trim_pct, axis=0) + pad
    centre_local = (lo + hi) / 2
    half = (hi - lo) / 2
    centre = centre_local @ axes
    corners = []
    for i in range(8):
        s = np.array([1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1])
        corners.append(centre + (s * half) @ axes)
    return axes, centre, half, np.array(corners)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, default=SCENE_PLY)
    ap.add_argument("--label-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--pad", type=float, default=0.01, help="box padding in metres")
    ap.add_argument("--trim", type=float, default=1.0, help="percentile trimmed per side (outliers)")
    ap.add_argument("--display-names", type=Path, default=None,
                    help="optional json {class_name: display_name}")
    args = ap.parse_args()

    display = json.load(open(args.display_names, encoding="utf-8")) if args.display_names else {}

    _, vertex = load_scene(args.scene)
    xyz = scene_xyz(vertex)
    if (args.label_dir / "labels.npy").exists():
        labels = np.load(args.label_dir / "labels.npy")
        classes = json.load(open(args.label_dir / "classes.json", encoding="utf-8"))
    else:
        # scene not labelled yet: still emit floor / bounds so the viewer and walk mesh work
        print("no labels.npy - writing objects.json without objects")
        labels = np.full(len(xyz), -1, dtype=np.int32)
        classes = {}
    args.label_dir.mkdir(parents=True, exist_ok=True)
    opacity = 1 / (1 + np.exp(-vertex["opacity"].astype(np.float64)))

    objects = []
    # stable colours: category -> hex, persisted so adding a class never recolours old ones
    colors_path = args.label_dir / "colors.json"
    colors: dict[str, str] = json.load(open(colors_path, encoding="utf-8")) if colors_path.exists() else {}
    category_ids: dict[str, int] = {}
    for cid_str, name in classes.items():
        cid = int(cid_str)
        m = labels == cid
        if not m.any():
            continue
        category, instance = split_name(name)
        cat_id = category_ids.setdefault(category, len(category_ids))
        if category not in colors:
            used = set(colors.values())
            colors[category] = next((c for c in PALETTE_HEX if c not in used), PALETTE_HEX[len(colors) % len(PALETTE_HEX)])
        shown = display.get(name) or display.get(category) or category
        if instance is not None:
            shown = f"{shown} #{instance}"
        pts = xyz[m]
        w = opacity[m]
        axes, box_centre, half, corners = oriented_box(pts, args.trim, args.pad)
        lo, hi = corners.min(0), corners.max(0)  # axis-aligned hull of the oriented box
        centre = (pts * w[:, None]).sum(0) / w.sum()
        # anchor for the label / leader line: top centre of the box (Z-up source frame)
        anchor = [float(box_centre[0]), float(box_centre[1]), float(box_centre[2] + half[2])]
        yaw = float(np.degrees(np.arctan2(axes[0, 1], axes[0, 0])))
        objects.append({
            "id": cid,
            "name": name,
            "category": category,
            "instance": instance,
            "display_name": shown,
            "color": colors[category],
            "count": int(m.sum()),
            "bbox_min": [float(v) for v in lo],
            "bbox_max": [float(v) for v in hi],
            "size": [float(v) for v in (2 * half)],
            "yaw_deg": yaw,
            "axes": axes.round(6).tolist(),
            "box_centre": [float(v) for v in box_centre],
            "corners": corners.round(5).tolist(),
            "centre": [float(v) for v in centre],
            "anchor": anchor,
        })
        print(f"[{cid}] {name:<20} n={m.sum():>8}  size={np.round(2 * half, 3)}  yaw={yaw:6.1f}°  centre={np.round(centre, 2)}")

    lo_all, hi_all = xyz.min(0), xyz.max(0)
    # floor height: strongest 2 cm z-bin in the lower half of the scene
    z = xyz[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.02))
    i = int(np.argmax(hist[: len(hist) // 2]))
    floor_z = float((edges[i] + edges[i + 1]) / 2)
    j = len(hist) // 2 + int(np.argmax(hist[len(hist) // 2:]))
    ceiling_z = float((edges[j] + edges[j + 1]) / 2)
    print(f"floor z = {floor_z:.3f}, ceiling z = {ceiling_z:.3f} (room height {ceiling_z - floor_z:.2f} m)")
    out = {
        "categories": list(category_ids),
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "frame": "source (Z-up, original PLY coordinates)",
        "scene_bbox_min": [float(v) for v in lo_all],
        "scene_bbox_max": [float(v) for v in hi_all],
        "objects": objects,
    }
    with open(colors_path, "w", encoding="utf-8") as f:
        json.dump(colors, f, ensure_ascii=False, indent=2)
    path = args.label_dir / "objects.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
