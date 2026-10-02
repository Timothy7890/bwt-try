"""Prepare one scene directory for the exhibit viewer, on any machine (macOS / Linux / Windows).

Scene directory layout (only point_cloud.ply is required):

    <scene>/point_cloud.ply     3DGS scene (Z-up)
    <scene>/manual_labels/      one CloudCompare / SuperSplat export per object (奖杯-1.txt, 植物.ply ...)
    <scene>/collision.ply       optional device collision mesh (only for ?occluder=mesh)
    <scene>/labels/             everything generated here; the viewer reads from it

Usage:
    python3 tools/pipeline.py <scene_dir>                       # check + all steps + link to viewer
    python3 tools/pipeline.py <scene_dir> --steps check         # only report what is there / missing
    python3 tools/pipeline.py <scene_dir> --steps labels,link   # after editing manual_labels/
    python3 tools/pipeline.py <scene_dir> --force               # re-encode scene.sog / walk voxels

Steps (in order): check, labels, occupancy, walk, collision, sog, link.
  walk: voxel collision built from the Gaussians (splat-transform .voxel.json) for walk mode
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIEWER_DATA = REPO / "exhibit-viewer" / "data"
STEPS = ["check", "labels", "occupancy", "walk", "collision", "sog", "link"]
SEG_SUFFIXES = {".ply", ".txt", ".asc", ".csv", ".xyz"}
PY_DEPS = ["numpy", "scipy", "plyfile"]

# viewer file name -> path inside <scene>/labels
LINKED = ["scene.sog", "objects.json", "occupancy.bin", "collision.bin",
          "walk.voxel.json", "walk.voxel.bin", "masks"]
# files app.js would otherwise pick up from a previous scene / the old flat-floor walk mesh
STALE_SCENE_FILES = ["scene_fast.sog", "scene.compressed.ply", "scene.ply", "collision.glb"]
# app.js turns the Z-up source into Y-up with Euler (-90, 0, 0). splat-transform writes voxels
# in the viewer's default frame (which assumes a 180° Z roll), so undo that roll first.
WALK_ROTATION = ["-r", "0,0,180", "-r", "-90,0,0"]


class Scene:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.ply = self.root / "point_cloud.ply"
        self.manual = self.root / "manual_labels"
        self.collision_ply = self.root / "collision.ply"
        self.out = self.root / "labels"
        self.sog = self.out / "scene.sog"
        self.walk = self.out / "walk.voxel.json"

    def segments(self):
        if not self.manual.is_dir():
            return []
        return sorted(p for p in self.manual.iterdir()
                      if p.suffix.lower() in SEG_SUFFIXES and not p.name.startswith("."))


def run_tool(scene: Scene, script: str, *args: str):
    print(f"\n=== {script} {' '.join(args)}".rstrip())
    env = dict(os.environ, BWT_SCENE_DIR=str(scene.root), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    subprocess.run([sys.executable, str(HERE / script), *args], check=True, env=env)


def mark(ok: bool | None):
    return {True: "[ok]  ", False: "[缺失]", None: "[可选]"}[ok]


def step_check(scene: Scene) -> bool:
    print(f"场景目录: {scene.root}")
    segs = scene.segments()
    rows = [
        (scene.ply.exists(), "point_cloud.ply", "必需"),
        (bool(segs) or None, f"manual_labels/ ({len(segs)} 个文件)", "没有则只能看原始场景，无对象框/Mask"),
        (scene.collision_ply.exists() or None, "collision.ply", "只用于 ?occluder=mesh，默认遮挡用体素"),
    ]
    for ok, name, note in rows:
        print(f"  {mark(ok)} {name:<32} {note if ok is not True else ''}")

    print("  已生成 (labels/):")
    for name in ["labels.npy", "objects.json", "masks/manifest.json", "occupancy.bin",
                 "walk.voxel.json", "collision.bin", "scene.sog"]:
        p = scene.out / name
        print(f"    {'[ok]  ' if p.exists() else '[未生成]'} {name}")

    missing = []
    for m in PY_DEPS:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        print(f"  [缺失] Python 包: {' '.join(missing)}  ->  {sys.executable} -m pip install -r tools/requirements.txt")
    if not shutil.which("npx"):
        print("  [缺失] npx (Node.js)，生成 scene.sog 和启动服务都需要")
    return scene.ply.exists() and not missing


def step_labels(scene: Scene):
    if scene.segments():
        run_tool(scene, "build_labels.py")
    elif (scene.out / "labels.npy").exists():
        print("\nmanual_labels/ 为空，沿用已有的 labels/labels.npy")
    else:
        print("\nmanual_labels/ 为空：只生成不含对象的 objects.json（地面高度、场景范围）")
    run_tool(scene, "make_objects.py")
    if (scene.out / "labels.npy").exists():
        run_tool(scene, "make_mask_splats.py")


def step_occupancy(scene: Scene):
    run_tool(scene, "make_occupancy.py")


def step_collision(scene: Scene):
    if scene.collision_ply.exists():
        run_tool(scene, "make_collision.py")
    else:
        print("\n=== collision: 没有 collision.ply，跳过（不影响默认的体素遮挡和步行）")


def splat_transform(scene: Scene, step: str, out: Path, actions: list[str], force: bool):
    if not force and out.exists() and out.stat().st_mtime >= scene.ply.stat().st_mtime:
        print(f"\n=== {step}: {out.name} 已是最新，跳过（--force 重新生成）")
        return
    npx = shutil.which("npx")
    if not npx:
        print(f"\n=== {step}: 找不到 npx，跳过。安装 Node.js 后重跑")
        return
    scene.out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {step}: {scene.ply.name} -> {out}")
    subprocess.run([npx, "-y", "@playcanvas/splat-transform", "-w", str(scene.ply), *actions, str(out)], check=True)


def auto_walk_voxel(scene: Scene) -> float:
    """0.05 m indoors; coarser for large sites so walk.voxel.bin stays around 20 MB
    (measured: ~1.6 KB per m² of footprint at 0.05 m, size ∝ 1/voxel²)."""
    import json
    objects = scene.out / "objects.json"
    if not objects.exists():
        return 0.05
    meta = json.load(open(objects, encoding="utf-8"))
    lo, hi = meta["scene_bbox_min"], meta["scene_bbox_max"]
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    voxel = 0.05 * (area * 1.6e-3 / 20) ** 0.5
    return max(0.05, round(-(-voxel // 0.05) * 0.05, 2))


def step_walk(scene: Scene, force: bool, voxel: float | None):
    voxel = voxel or auto_walk_voxel(scene)
    print(f"\n=== walk: voxel {voxel:.2f} m")
    splat_transform(scene, "walk", scene.walk, [*WALK_ROTATION, "--voxel-size", str(voxel)], force)


def step_sog(scene: Scene, force: bool):
    splat_transform(scene, "sog", scene.sog, [], force)


def _remove(p: Path):
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def _link(target: Path, link: Path):
    """Relative symlink so the repo can move; falls back to a copy where symlinks are not
    allowed (Windows without developer mode)."""
    try:
        link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=target.is_dir())
        return "link"
    except OSError:
        (shutil.copytree if target.is_dir() else shutil.copy2)(target, link)
        return "copy"


def step_link(scene: Scene):
    print(f"\n=== link: exhibit-viewer/data -> {scene.out}")
    VIEWER_DATA.mkdir(parents=True, exist_ok=True)
    stash = VIEWER_DATA / "_stash"
    for name in LINKED + STALE_SCENE_FILES:
        dst = VIEWER_DATA / name
        if dst.is_symlink() or not dst.exists():
            _remove(dst)  # also clears dangling links
            continue
        # a real file from an older manual setup: keep it, just move it out of the way
        stash.mkdir(exist_ok=True)
        moved = stash / name
        _remove(moved)
        shutil.move(str(dst), str(moved))
        print(f"  moved existing {name} -> _stash/{name}")
    for name in LINKED:
        src = scene.out / name
        if not src.exists():
            print(f"  [跳过] {name}（未生成）")
            continue
        how = _link(src, VIEWER_DATA / name)
        print(f"  [{how}] {name}")
    # robot GLBs are shared by all scenes: <data root>/glb next to the scene directories
    robot_src = scene.root.parent / "glb"
    if robot_src.is_dir():
        _remove(VIEWER_DATA / "robot")
        print(f"  [{_link(robot_src, VIEWER_DATA / 'robot')}] robot")
    elif not (VIEWER_DATA / "robot").exists():
        print(f"  [跳过] robot（没有 {robot_src}，角色不显示）")
    (VIEWER_DATA / "CURRENT_SCENE.txt").write_text(str(scene.root) + "\n", encoding="utf-8")
    if not (scene.out / "scene.sog").exists():
        print("  注意：没有 scene.sog，前端会找不到场景文件")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma separated subset of {STEPS}")
    ap.add_argument("--force", action="store_true", help="re-encode scene.sog / walk voxels even if up to date")
    ap.add_argument("--walk-voxel", type=float, default=None,
                    help="walk collision voxel size in metres (default: 0.05, coarser for large sites)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    scene = Scene(args.scene_dir)
    if not scene.root.is_dir():
        ap.error(f"not a directory: {scene.root}")

    ready = step_check(scene)
    if steps == ["check"]:
        return
    if not ready and any(s in steps for s in STEPS[1:-1]):
        raise SystemExit("\n缺少必需文件或 Python 包，先按上面的提示补齐")
    for s in STEPS[1:]:
        if s not in steps:
            continue
        if s == "sog":
            step_sog(scene, args.force)
        elif s == "walk":
            step_walk(scene, args.force, args.walk_voxel)
        else:
            globals()[f"step_{s}"](scene)
    print("\ndone. 启动服务: npx serve exhibit-viewer --symlinks -l 3000，然后刷新浏览器")


if __name__ == "__main__":
    main()
