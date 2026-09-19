"""URDF + joint-angle CSV  ->  animated GLB (node animation, no skinning).

Robots are rigid link chains, so every URDF link becomes a glTF node carrying its mesh and
every revolute joint becomes an animated rotation on the child node. The Z-up robot frame
is converted to glTF Y-up by a wrapper node.

CSV format (LAFAN1 retargeting / your own teleop logs), one row per frame:
    x y z qx qy qz qw   q_1 ... q_N        (root pose in a Z-up world, then joint angles in
                                             URDF traversal order = the dataset's README order)

Usage examples:
    python3 robot_to_glb.py --robot g1 --clip walk=g1/walk1_subject1.csv:loop \\
                            --clip idle=g1/walk1_subject1.csv:idle --out g1_character.glb
    python3 robot_to_glb.py --robot g1 --clip replay=g1/walk1_subject1.csv:full --out g1_walk1.glb

Clip modes:  loop  = in-place walking cycle (root XY/yaw removed, seamless loop)
             idle  = calmest ~1 s window, held as a short loop
             full  = whole take with root motion (for replaying a recorded route)
"""
from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pygltflib as gl
import trimesh

ROBOT_DIR = Path("/g12213021yx/Xin_Yang/project_data/3DGS/BWT/robot/lafan1")
ROBOTS = {
    "g1": ROBOT_DIR / "robot_description/g1/g1_29dof_rev_1_0.urdf",
    "h1_2": ROBOT_DIR / "robot_description/h1_2/h1_2_wo_hand.urdf",
    "h2": ROBOT_DIR.parent / "unitree_ros/robots/h2_description/H2.urdf",
}
FPS = 30.0

# Default standing stances (radians), keyed by joint name. Legs follow Unitree's controller
# defaults (hip pitch -0.1/-0.16, knee 0.3/0.36, ankle -0.2); arms hang with a slight bend.
# Anything not listed is 0. root_z = pelvis height above the floor in that stance.
STANCES = {
    "g1": {
        "root_z": 0.775,
        "left_hip_pitch_joint": -0.10, "left_knee_joint": 0.30, "left_ankle_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.10, "right_knee_joint": 0.30, "right_ankle_pitch_joint": -0.20,
        "left_shoulder_pitch_joint": 0.15, "left_shoulder_roll_joint": 0.12, "left_elbow_joint": 0.45,
        "right_shoulder_pitch_joint": 0.15, "right_shoulder_roll_joint": -0.12, "right_elbow_joint": 0.45,
    },
    "h2": {  # root_z is derived from the URDF leg length at build time
        "root_z": None,
        "left_hip_pitch_joint": -0.12, "left_knee_joint": 0.30, "left_ankle_pitch_joint": -0.18,
        "right_hip_pitch_joint": -0.12, "right_knee_joint": 0.30, "right_ankle_pitch_joint": -0.18,
        "left_shoulder_pitch_joint": 0.15, "left_shoulder_roll_joint": 0.12, "left_elbow_joint": 0.45,
        "right_shoulder_pitch_joint": 0.15, "right_shoulder_roll_joint": -0.12, "right_elbow_joint": 0.45,
    },
    "h1_2": {
        "root_z": 1.00,
        "left_hip_pitch_joint": -0.16, "left_knee_joint": 0.36, "left_ankle_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.16, "right_knee_joint": 0.36, "right_ankle_pitch_joint": -0.20,
        "left_shoulder_pitch_joint": 0.15, "left_shoulder_roll_joint": 0.12, "left_elbow_joint": 0.45,
        "right_shoulder_pitch_joint": 0.15, "right_shoulder_roll_joint": -0.12, "right_elbow_joint": 0.45,
    },
}
# joints that breathe: (amplitude rad, phase)
BREATH = {"shoulder_pitch": 0.025, "elbow": 0.03, "waist_pitch": 0.008, "torso": 0.0}


def stance_clip(robot, revolute, fps=FPS, seconds=3.0, stance_root_z=None):
    """Hand-authored standing pose with a gentle breathing cycle. Returns root_xyz, root_q, qj."""
    st = dict(STANCES[robot])
    if st.get("root_z") is None:
        st["root_z"] = stance_root_z
    n = int(seconds * fps)
    t = np.arange(n) / fps
    phase = np.sin(2 * np.pi * t / seconds)
    qj = np.zeros((n, len(revolute)))
    for k, j in enumerate(revolute):
        base = st.get(j.name, 0.0)
        amp = 0.0
        for key, a in BREATH.items():
            if key in j.name:
                amp = a
        sign = -1 if ("right" in j.name and "roll" in j.name) else 1
        qj[:, k] = base + sign * amp * phase
    root_xyz = np.zeros((n, 3))
    root_xyz[:, 2] = st["root_z"] + 0.004 * phase
    root_q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
    return root_xyz, root_q, qj


# ----------------------------------------------------------------- quaternion helpers (x, y, z, w)

def quat_from_rpy(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle / 2)
    return np.concatenate([axis[None, :] * s[:, None], np.cos(angle / 2)[:, None]], axis=1)


def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_yaw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_continuous(q):
    """Flip signs so consecutive quaternions stay on the same hemisphere (smooth slerp)."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


# ----------------------------------------------------------------- URDF

class Link:
    def __init__(self, name):
        self.name = name
        self.visuals = []  # (mesh_path, xyz, rpy, scale, rgba)


class Joint:
    def __init__(self, name, jtype, parent, child, xyz, rpy, axis):
        self.name, self.type, self.parent, self.child = name, jtype, parent, child
        self.xyz, self.rpy, self.axis = xyz, rpy, axis


def parse_urdf(path: Path):
    root = ET.parse(path).getroot()
    base = path.parent
    materials = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None:
            materials[m.get("name")] = [float(v) for v in c.get("rgba").split()]
    links = {}
    for l in root.findall("link"):
        link = Link(l.get("name"))
        for v in l.findall("visual"):
            geo = v.find("geometry/mesh")
            if geo is None:
                continue
            o = v.find("origin")
            xyz = [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            scale = [float(x) for x in geo.get("scale", "1 1 1").split()]
            rgba = [0.6, 0.6, 0.6, 1.0]
            mat = v.find("material")
            if mat is not None:
                c = mat.find("color")
                if c is not None:
                    rgba = [float(x) for x in c.get("rgba").split()]
                elif mat.get("name") in materials:
                    rgba = materials[mat.get("name")]
            fn = geo.get("filename").replace("package://", "")
            mesh_path = (base / fn) if not fn.startswith("/") else Path(fn)
            if not mesh_path.exists():  # package-style paths: search by basename
                hits = list(base.rglob(Path(fn).name))
                mesh_path = hits[0] if hits else mesh_path
            link.visuals.append((mesh_path, xyz, rpy, scale, rgba))
        links[link.name] = link
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        a = j.find("axis")
        axis = [float(x) for x in (a.get("xyz") if a is not None else "1 0 0").split()]
        joints.append(Joint(j.get("name"), j.get("type"), j.find("parent").get("link"),
                            j.find("child").get("link"), xyz, rpy, axis))
    return links, joints


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def zero_pose_pelvis_height(links, joints, root_link):
    """Height of the root link origin above the lowest mesh vertex with all joints at zero."""
    lowest = [np.inf]

    def walk(link_name, R, t):
        for mesh_path, xyz, rpy, scale, _ in links[link_name].visuals:
            if not mesh_path.exists():
                continue
            tm = trimesh.load(str(mesh_path), force="mesh")
            if tm.is_empty:
                continue
            v = tm.vertices * np.asarray(scale)
            v = (rpy_matrix(*rpy) @ v.T).T + np.asarray(xyz)
            v = (R @ v.T).T + t
            lowest[0] = min(lowest[0], float(v[:, 2].min()))
        for j in joints:
            if j.parent == link_name and j.type != "floating":
                Rj = rpy_matrix(*j.rpy)
                walk(j.child, R @ Rj, R @ np.asarray(j.xyz) + t)

    walk(root_link, np.eye(3), np.zeros(3))
    return -lowest[0]


# ----------------------------------------------------------------- clips

def load_csv(path: Path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :3], data[:, 3:7], data[:, 7:]


def pose_distance(qj, root_z, i, j):
    return np.linalg.norm(qj[i] - qj[j]) + 3.0 * abs(root_z[i] - root_z[j])


def find_walk_loop(qj, root_z, root_xy, fps=FPS):
    """Best (start, end) for a seamless cycle: one full gait period apart, similar pose, and
    the robot is actually moving (so we don't pick a standing segment)."""
    n = len(qj)
    speed = np.linalg.norm(np.gradient(root_xy, axis=0), axis=1) * fps
    best, best_pair = np.inf, None
    for period in range(int(0.8 * fps), int(1.6 * fps)):
        for i in range(0, n - period, 2):
            j = i + period
            if speed[i:j].mean() < 0.4:
                continue
            d = pose_distance(qj, root_z, i, j) + 0.5 * np.linalg.norm(qj[i + 1] - qj[i] - (qj[j + 1] - qj[j])) if j + 1 < n else np.inf
            if d < best:
                best, best_pair = d, (i, j)
    if best_pair is None:
        raise SystemExit("no moving segment found for a walk loop")
    return best_pair


def find_idle_window(qj, root_xy, fps=FPS, seconds=1.5, arm_roll_cols=(16, 23)):
    """Calmest window where the robot stands still with arms down (LAFAN takes start with a
    T-pose calibration, which must not be picked as 'idle')."""
    w = int(seconds * fps)
    vel = np.linalg.norm(np.diff(qj, axis=0), axis=1)
    speed = np.linalg.norm(np.gradient(root_xy, axis=0), axis=1) * fps
    best = None
    for i in range(0, len(qj) - w, 3):
        arms = np.abs(qj[i:i + w][:, list(arm_roll_cols)]).mean() if qj.shape[1] > max(arm_roll_cols) else 0.0
        if arms > 0.5 or speed[i:i + w].mean() > 0.15:
            continue
        v = vel[i:i + w - 1].sum()
        if best is None or v < best[0]:
            best = (v, i)
    if best is None:  # fall back to the calmest window regardless of pose
        sums = np.convolve(vel, np.ones(w), mode="valid")
        i = int(np.argmin(sums))
        return i, i + w
    return best[1], best[1] + w


def strip_root_motion(root_xyz, root_q):
    """Remove horizontal travel and heading so the clip plays in place, keep bob and lean."""
    yaw = quat_yaw(root_q)
    unyaw = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(-yaw / 2), np.cos(-yaw / 2)], axis=1)
    q = quat_mul(unyaw, root_q)
    xyz = root_xyz.copy()
    xyz[:, 0] = 0.0
    xyz[:, 1] = 0.0
    return xyz, q


def make_seamless(arr, blend=6):
    """Cross-fade the last `blend` frames toward the first frame so the loop has no jump."""
    arr = arr.copy()
    n = len(arr)
    for k in range(blend):
        t = (k + 1) / (blend + 1)
        idx = n - blend + k
        arr[idx] = (1 - t) * arr[idx] + t * arr[0]
    return arr


# ----------------------------------------------------------------- GLB writer

class GlbBuilder:
    def __init__(self):
        self.g = gl.GLTF2(asset=gl.Asset(version="2.0", generator="bwt robot_to_glb"))
        self.blob = bytearray()
        self.g.scene = 0
        self.g.scenes = [gl.Scene(nodes=[])]

    def _accessor(self, arr, ctype, atype, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        offset = len(self.blob)
        self.blob.extend(arr.tobytes())
        while len(self.blob) % 4:
            self.blob.append(0)
        self.g.bufferViews.append(gl.BufferView(buffer=0, byteOffset=offset, byteLength=arr.nbytes, target=target))
        acc = gl.Accessor(bufferView=len(self.g.bufferViews) - 1, componentType=ctype, count=len(arr), type=atype)
        if minmax:
            acc.min = arr.min(axis=0).tolist() if arr.ndim > 1 else [float(arr.min())]
            acc.max = arr.max(axis=0).tolist() if arr.ndim > 1 else [float(arr.max())]
        self.g.accessors.append(acc)
        return len(self.g.accessors) - 1

    def add_material(self, rgba):
        self.g.materials.append(gl.Material(
            name=f"m{len(self.g.materials)}",
            pbrMetallicRoughness=gl.PbrMetallicRoughness(baseColorFactor=[float(v) for v in rgba], metallicFactor=0.3, roughnessFactor=0.6)))
        return len(self.g.materials) - 1

    def add_mesh(self, tm: trimesh.Trimesh, material):
        v = tm.vertices.astype(np.float32)
        nrm = tm.vertex_normals.astype(np.float32)
        idx = tm.faces.astype(np.uint32).ravel()
        p = self._accessor(v, gl.FLOAT, gl.VEC3, gl.ARRAY_BUFFER, minmax=True)
        n = self._accessor(nrm, gl.FLOAT, gl.VEC3, gl.ARRAY_BUFFER)
        i = self._accessor(idx, gl.UNSIGNED_INT, gl.SCALAR, gl.ELEMENT_ARRAY_BUFFER)
        self.g.meshes.append(gl.Mesh(primitives=[gl.Primitive(attributes=gl.Attributes(POSITION=p, NORMAL=n), indices=i, material=material)]))
        return len(self.g.meshes) - 1

    def add_node(self, name, translation=None, rotation=None, scale=None, mesh=None, parent=None):
        node = gl.Node(name=name, mesh=mesh)
        if translation is not None:
            node.translation = [float(x) for x in translation]
        if rotation is not None:
            node.rotation = [float(x) for x in rotation]
        if scale is not None and any(abs(s - 1) > 1e-9 for s in scale):
            node.scale = [float(x) for x in scale]
        self.g.nodes.append(node)
        idx = len(self.g.nodes) - 1
        if parent is None:
            self.g.scenes[0].nodes.append(idx)
        else:
            self.g.nodes[parent].children = (self.g.nodes[parent].children or []) + [idx]
        return idx

    def add_animation(self, name, times, channels):
        """channels: list of (node_index, path, values[N, k])"""
        t = self._accessor(times.astype(np.float32), gl.FLOAT, gl.SCALAR, minmax=True)
        anim = gl.Animation(name=name, samplers=[], channels=[])
        for node, path, values in channels:
            v = self._accessor(values.astype(np.float32), gl.FLOAT, gl.VEC4 if path == "rotation" else gl.VEC3)
            anim.samplers.append(gl.AnimationSampler(input=t, output=v, interpolation="LINEAR"))
            anim.channels.append(gl.AnimationChannel(sampler=len(anim.samplers) - 1,
                                                     target=gl.AnimationChannelTarget(node=node, path=path)))
        self.g.animations.append(anim)

    def save(self, path: Path):
        self.g.buffers = [gl.Buffer(byteLength=len(self.blob))]
        self.g.set_binary_blob(bytes(self.blob))
        self.g.save_binary(str(path))


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS), default="g1")
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--clip", action="append", default=[], help="name=csv[:loop|idle|full]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--zero", default="",
                    help="comma list of joint-name substrings to force to 0 (e.g. hip_yaw,ankle_roll)")
    ap.add_argument("--damp", default="",
                    help="comma list of substring=factor; scales that joint's motion about its mean "
                         "(e.g. shoulder_pitch=0.5,elbow=0.6 to calm arm swing)")
    ap.add_argument("--csv-robot", choices=list(ROBOTS), default=None,
                    help="the CSV's joint order follows this robot's URDF; angles are mapped by joint "
                         "name onto --robot and the root height is scaled by leg length (e.g. G1 data on H2)")
    args = ap.parse_args()

    urdf = args.urdf or ROBOTS[args.robot]
    links, joints = parse_urdf(urdf)
    revolute = [j for j in joints if j.type in ("revolute", "continuous")]
    print(f"{urdf.name}: {len(links)} links, {len(revolute)} actuated joints")

    child_of = {j.child: j for j in joints}
    floating = next((j for j in joints if j.type == "floating"), None)
    root_link = floating.child if floating else next(l for l in links if l not in child_of)
    pelvis_h = zero_pose_pelvis_height(links, joints, root_link)
    print(f"zero-pose pelvis height {pelvis_h:.3f} m")

    # optional: CSV columns belong to another robot -> map by joint name, scale root height
    col_map = None
    root_scale = 1.0
    if args.csv_robot and args.csv_robot != args.robot:
        src_links, src_joints = parse_urdf(ROBOTS[args.csv_robot])
        src_rev = [j for j in src_joints if j.type in ("revolute", "continuous")]
        src_idx = {j.name: k for k, j in enumerate(src_rev)}
        # equivalent joints with different names across Unitree humanoids
        ALIASES = {"waist_yaw_joint": "torso_joint", "torso_joint": "waist_yaw_joint"}
        col_map = [src_idx.get(j.name, src_idx.get(ALIASES.get(j.name, ""), -1)) for j in revolute]
        missing = [j.name for j, c in zip(revolute, col_map) if c < 0]
        src_child_of = {j.child: j for j in src_joints}
        src_float = next((j for j in src_joints if j.type == "floating"), None)
        src_root = src_float.child if src_float else next(l for l in src_links if l not in src_child_of)
        src_h = zero_pose_pelvis_height(src_links, src_joints, src_root)
        root_scale = pelvis_h / src_h
        print(f"mapping {args.csv_robot} columns by name: {len(revolute) - len(missing)} matched, "
              f"unmatched -> 0: {missing}; root height x{root_scale:.3f}")

    b = GlbBuilder()
    # wrapper: robot Z-up -> glTF Y-up
    s = math.sin(-math.pi / 4)
    c = math.cos(-math.pi / 4)
    wrapper = b.add_node(f"{args.robot}_root", rotation=[s, 0, 0, c])
    node_of = {}

    def build(link_name, parent_node, joint):
        t = joint.xyz if joint else [0, 0, 0]
        q = quat_from_rpy(*joint.rpy) if joint else [0, 0, 0, 1]
        node = b.add_node(link_name, translation=t, rotation=q, parent=parent_node)
        node_of[link_name] = node
        for mesh_path, xyz, rpy, scale, rgba in links[link_name].visuals:
            if not mesh_path.exists():
                print("  missing mesh", mesh_path)
                continue
            tm = trimesh.load(str(mesh_path), force="mesh")
            if tm.is_empty:
                continue
            mat = b.add_material(rgba)
            m = b.add_mesh(tm, mat)
            b.add_node(f"{link_name}_vis", translation=xyz, rotation=quat_from_rpy(*rpy), scale=scale, mesh=m, parent=node)
        for j in joints:
            if j.parent == link_name and j.type != "floating":
                build(j.child, node, j)

    build(root_link, wrapper, None)
    root_node = node_of[root_link]
    print(f"root link {root_link}, {len(b.g.meshes)} meshes")

    for spec in args.clip:
        name, rest = spec.split("=", 1)
        if rest == "stance":
            root_xyz, root_q, qj = stance_clip(args.robot, revolute, args.fps, stance_root_z=pelvis_h * 0.985)
            mode = "stance"
            print(f"  {name}: hand-authored stance with breathing ({len(qj) / args.fps:.1f} s loop)")
        else:
            csv, _, mode = rest.partition(":")
            mode = mode or "full"
            csv_path = Path(csv) if Path(csv).is_absolute() else ROBOT_DIR / csv
            root_xyz, root_q, qj = load_csv(csv_path)
            if col_map is not None:
                qj = np.stack([qj[:, c] if c >= 0 else np.zeros(len(qj)) for c in col_map], axis=1)
                root_xyz = root_xyz * np.array([root_scale, root_scale, root_scale])
            if qj.shape[1] != len(revolute):
                raise SystemExit(f"{csv_path.name}: {qj.shape[1]} joint columns but URDF has {len(revolute)} joints")

        # optional per-joint clean-up of transferred motion
        if mode != "stance" and (args.zero or args.damp):
            zero_keys = [z for z in args.zero.split(",") if z]
            damp = {}
            for item in args.damp.split(","):
                if "=" in item:
                    k, v = item.split("=")
                    damp[k] = float(v)
            for k, j in enumerate(revolute):
                if any(z in j.name for z in zero_keys):
                    qj[:, k] = 0.0
                    continue
                for key, f in damp.items():
                    if key in j.name:
                        m = qj[:, k].mean()
                        qj[:, k] = m + (qj[:, k] - m) * f

        if mode == "stance":
            pass
        elif mode == "loop":
            i, j = find_walk_loop(qj, root_xyz[:, 2], root_xyz[:, :2], args.fps)
            seg_speed = float(np.linalg.norm(np.gradient(root_xyz[i:j, :2], axis=0), axis=1).mean() * args.fps)
            print(f"  {name}: walk loop frames {i}-{j} ({(j - i) / args.fps:.2f} s), travel speed {seg_speed:.2f} m/s")
            root_xyz, root_q, qj = root_xyz[i:j], root_q[i:j], qj[i:j]
            root_xyz, root_q = strip_root_motion(root_xyz, root_q)
            qj, root_xyz, root_q = make_seamless(qj), make_seamless(root_xyz), make_seamless(root_q)
        elif mode == "idle":
            # a static standing pose: the median of the calmest arms-down window, held for 1 s.
            # (looping a raw 1.5 s window twitches at the seam because start != end pose)
            i, j = find_idle_window(qj, root_xyz[:, :2], args.fps)
            print(f"  {name}: idle pose = median of frames {i}-{j}")
            root_xyz, root_q = strip_root_motion(root_xyz[i:j], root_q[i:j])
            pose_q = np.median(qj[i:j], axis=0)
            pose_root = np.median(root_xyz, axis=0)
            pose_rq = root_q[len(root_q) // 2]
            pose_rq = pose_rq / np.linalg.norm(pose_rq)
            qj = np.stack([pose_q, pose_q])
            root_xyz = np.stack([pose_root, pose_root])
            root_q = np.stack([pose_rq, pose_rq])
            times_override = np.array([0.0, 1.0])
        else:
            print(f"  {name}: full take, {len(qj)} frames ({len(qj) / args.fps:.1f} s), with root motion")

        n = len(qj)
        times = np.arange(n) / args.fps
        if mode == "idle":
            times = times_override
        channels = [(root_node, "translation", root_xyz), (root_node, "rotation", quat_continuous(root_q))]
        for k, j in enumerate(revolute):
            q_origin = quat_from_rpy(*j.rpy)
            q_joint = quat_axis_angle(j.axis, qj[:, k])
            q = quat_mul(np.broadcast_to(q_origin, q_joint.shape), q_joint)
            channels.append((node_of[j.child], "rotation", quat_continuous(q)))
        b.add_animation(name, times, channels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    b.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), animations: {[a.name for a in b.g.animations]}")


if __name__ == "__main__":
    main()
