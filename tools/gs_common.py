"""Shared helpers for 3DGS PLY label tooling."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814

# One scene per directory: <scene>/point_cloud.ply, <scene>/manual_labels/, <scene>/labels/ ...
# Set BWT_SCENE_DIR (tools/pipeline.py does this) to pick the scene; the old server path is
# only a fallback for existing workflows there.
_LEGACY_DATA_DIR = Path("/g12213021yx/Xin_Yang/project_data/3DGS/BWT/0916展厅数据第二次测试")
DATA_DIR = Path(os.environ.get("BWT_SCENE_DIR") or _LEGACY_DATA_DIR).expanduser().resolve()
SCENE_PLY = DATA_DIR / "point_cloud.ply"
LABEL_DIR = DATA_DIR / "manual_labels"   # drop CloudCompare / SuperSplat exports here
OUT_DIR = DATA_DIR / "labels"            # labels.npy / classes.json / metadata.json
VIS_DIR = DATA_DIR / "vis"               # derived PLYs for SuperSplat

INDEX_NAME_RE = re.compile(r"index|idx", re.I)


def load_scene(path: Path = SCENE_PLY):
    """Return (PlyData, vertex structured array) of the original 3DGS PLY."""
    ply = PlyData.read(str(path))
    return ply, ply["vertex"].data


def scene_xyz(vertex) -> np.ndarray:
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def read_segment(path: Path):
    """Read a segment exported from CloudCompare/SuperSplat.

    Returns (xyz float64 [N,3], index int64 [N] or None).
    Supports .ply (binary/ascii) and CloudCompare ASCII (.txt/.asc/.csv/.xyz).
    """
    suf = path.suffix.lower()
    if suf == ".ply":
        v = PlyData.read(str(path))["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        idx = None
        for name in v.dtype.names:
            if INDEX_NAME_RE.search(name):
                idx = np.rint(v[name]).astype(np.int64)
                break
        return xyz, idx

    header_names = None
    with open(path, "r", errors="replace") as f:
        first = f.readline().strip()
    if first.startswith("//") or first.startswith("#"):
        header_names = re.split(r"[,\s]+", first.lstrip("/# ").strip())
    data = np.loadtxt(path, comments=("//", "#"), delimiter=None if "," not in first else ",", ndmin=2)
    xyz = data[:, :3].astype(np.float64)
    idx = None
    if header_names:
        for i, name in enumerate(header_names):
            if INDEX_NAME_RE.search(name) and i < data.shape[1]:
                idx = np.rint(data[:, i]).astype(np.int64)
                break
    return xyz, idx


def f_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    return np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)


def rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / SH_C0


def write_vertex_ply(vertex, path: Path, comments=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False, byte_order="<", comments=list(comments or [])).write(str(path))
