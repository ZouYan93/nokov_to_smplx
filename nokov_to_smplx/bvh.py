"""Strict common23 hierarchy admission and native per-joint intrinsic Euler FK."""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
SOURCE_PARENTS = {
    "Hips": None, "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1",
    "Spine3": "Spine2", "Neck": "Spine3", "Head": "Neck",
}
for _side in ("Left", "Right"):
    for _name, _parent in (
        ("Shoulder", "Spine3"), ("Arm", _side+"Shoulder"),
        ("ForeArm", _side+"Arm"), ("Hand", _side+"ForeArm"),
        ("UpLeg", "Hips"), ("Leg", _side+"UpLeg"),
        ("Foot", _side+"Leg"), ("ToeBase", _side+"Foot"),
    ):
        SOURCE_PARENTS[_side+_name] = _parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class SourceMotion:
    bones: list
    positions: np.ndarray
    quaternions_wxyz: np.ndarray
    frame_time: float
    provenance: dict

    @property
    def fps(self):
        return 1 / self.frame_time

    @property
    def names(self):
        return tuple(b["name"] for b in self.bones)

    @property
    def world_rotations(self):
        # Retain the source-reader quaternion / fixed-converter matrix boundary.
        return Rotation.from_quat(
            self.quaternions_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]
        ).as_matrix().reshape(len(self.positions), 23, 3, 3)


def _validate_geometry(bones):
    actual = {b["name"]: None if b["parent"] < 0 else bones[b["parent"]]["name"]
              for b in bones}
    require(len(actual) == 23 and actual == SOURCE_PARENTS,
            "Unsupported NOKOV names/parents: common23 four-spine body required")
    lookup = {b["name"]: b for b in bones}
    for name in ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"):
        x, y, z = lookup[name]["offset"]
        require(y > 0 and abs(x) <= .002*y and abs(z) <= .4*y,
                "Unsupported source axes: expected observed Y-up sagittal torso")
    for side, sign in (("Left", 1), ("Right", -1)):
        for name in ("Arm", "ForeArm", "Hand"):
            x, y, z = lookup[side+name]["offset"]
            require(sign*x > 0 and np.hypot(y, z) <= .4*abs(x),
                    "Unsupported source axes: arm ray exceeds signed X-lateral cone")
        x, y, z = lookup[side+"UpLeg"]["offset"]
        require(sign*x > 0 and abs(y) <= .4*abs(x) and abs(z) <= .002*abs(x),
                "Unsupported source axes: expected observed X-lateral hips")
        for name in ("Leg", "Foot"):
            x, y, z = lookup[side+name]["offset"]
            require(y < 0 and abs(x) <= .002*abs(y) and abs(z) <= .1*abs(y),
                    "Unsupported source axes: expected observed downward Y legs")
            require(.15 < np.linalg.norm([x, y, z])/100 < .7,
                    "Source dimensions do not support centimetre convention")
        for key in ("offset", "endOffset"):
            require(key in lookup[side+"ToeBase"], "Toe End Site required")
            x, _, z = lookup[side+"ToeBase"][key]
            require(z > 0 and abs(x) <= .002*z,
                    "Unsupported source axes: expected observed forward Z toes")


def _parse_header(header):
    declared = re.findall(r"(?:ROOT|JOINT)\s+(\S+)", header)
    require(all(n in declared for n in ("LeftToeBase", "RightToeBase")),
            "SOURCE_TOE_ROTATION_UNAVAILABLE: both articulated ToeBase joints required")
    require(len(declared) == 23, "SOURCE_BODY23_REQUIRED: common23 hierarchy required")
    tokens = re.findall(r"\{|\}|[^\s{}]+", header)
    cursor, channel_index, bones = 0, 0, []

    def take(expected=None):
        nonlocal cursor
        require(cursor < len(tokens), "Truncated BVH hierarchy")
        token = tokens[cursor]
        cursor += 1
        require(expected is None or token == expected, "Malformed BVH hierarchy")
        return token

    def vector():
        values = [float(take()) for _ in range(3)]
        require(np.isfinite(values).all(), "Nonfinite rest offsets")
        return values

    def joint(parent):
        nonlocal channel_index
        require(len(bones) < 23, "SOURCE_BODY23_REQUIRED")
        take("ROOT" if parent < 0 else "JOINT")
        bone = dict(name=take(), parent=parent)
        index = len(bones)
        bones.append(bone)
        take("{")
        take("OFFSET")
        bone["offset"] = vector()
        take("CHANNELS")
        width = int(take())
        require(width == (6 if parent < 0 else 3),
                "Non-root translation or missing rotation unsupported")
        channels = [take() for _ in range(width)]
        rotation = channels[3:] if parent < 0 else channels
        require(set(rotation) == {a+"rotation" for a in "XYZ"}, "Invalid Euler channels")
        require(parent >= 0 or channels[:3] == [a+"position" for a in "XYZ"],
                "Root XYZ translation required")
        bone.update(channels=channels, indices=list(range(channel_index, channel_index+width)))
        channel_index += width
        while cursor < len(tokens) and tokens[cursor] != "}":
            if tokens[cursor] == "JOINT":
                joint(index)
            else:
                require("endOffset" not in bone, "Duplicate End Site")
                take("End"); take("Site"); take("{"); take("OFFSET")
                bone["endOffset"] = vector()
                take("}")
        take("}")

    take("HIERARCHY")
    joint(-1)
    require(cursor == len(tokens) and len(bones) == 23, "SOURCE_BODY23_REQUIRED")
    _validate_geometry(bones)
    return bones, channel_index


def load_bvh(file):
    """Decode all native frames into metres/Z-up/WXYZ without action edits."""
    path = Path(file)
    before = path.stat()
    require(before.st_size <= 32*1024**2, "NOKOV BVH exceeds 32 MiB")
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino, s.st_ctime_ns)
    require(identity(before) == identity(after) and len(raw) == before.st_size,
            "NOKOV source changed during reading")
    lines = list(io.StringIO(raw.decode("utf-8-sig"), newline=None))
    motion = next((i for i, line in enumerate(lines) if line.strip() == "MOTION"), None)
    require(motion is not None, "Missing BVH MOTION section")
    header = "".join(lines[:motion])
    require(len(header) <= 65536, "NOKOV hierarchy exceeds 64 KiB")
    bones, channels = _parse_header(header)
    rows = [line.strip() for line in lines[motion+1:] if line.strip()]
    require(len(rows) >= 2, "Missing BVH clock")
    count = re.fullmatch(r"Frames:\s*(\d+)", rows[0])
    step = re.fullmatch(r"Frame Time:\s*(\S+)", rows[1])
    require(count is not None and step is not None, "Invalid BVH clock")
    count, dt = int(count[1]), float(step[1])
    require(np.isfinite(dt) and 1/240 <= dt <= 1 and 2 <= count <= 72000
            and count*dt <= 300.1, "Unsupported BVH clock/range")
    require(len(rows) == count+2, "NOKOV declared and actual frame counts differ")
    values = np.empty((count, channels))
    for i, row in enumerate(rows[2:]):
        fields = row.split()
        require(len(fields) == channels, f"NOKOV channel count differs at native frame {i}")
        values[i] = fields
    require(np.isfinite(values).all(), "NOKOV motion contains nonfinite values")
    world = np.empty((count, 23, 3, 3))
    points = np.empty((count, 23, 3))
    rest = np.zeros((23, 3))
    orders = {}
    for j, bone in enumerate(bones):
        rotation = [(c[0], i) for c, i in zip(bone["channels"], bone["indices"])
                    if c.endswith("rotation")]
        order = "".join(axis for axis, _ in rotation)
        orders[bone["name"]] = order
        local = Rotation.from_euler(
            order, values[:, [i for _, i in rotation]], degrees=True
        ).as_matrix()
        parent = bone["parent"]
        if parent < 0:
            world[:, j] = local
            points[:, j] = values[:, bone["indices"][:3]]
        else:
            world[:, j] = world[:, parent] @ local
            points[:, j] = points[:, parent]+world[:, parent] @ bone["offset"]
        rest[j] = np.asarray(bone["offset"])+(rest[parent] if parent >= 0 else 0)
    points = points @ UP.T / 100
    xyzw = Rotation.from_matrix((UP @ world).reshape(-1, 3, 3)).as_quat()
    quaternions = xyzw[:, [3, 0, 1, 2]].reshape(count, 23, 4)
    rest_up = Rotation.from_euler("x", 90, degrees=True)
    rest = rest_up.apply(rest)/100
    provenance = dict(
        schema="nokov-common23-source/1", sourceSha256=hashlib.sha256(raw).hexdigest(),
        headerSha256=hashlib.sha256(header.encode()).hexdigest(),
        nativeFrames=count, nativeFrameTime=dt, nativeFps=1/dt,
        nativeStartFrame=0, nativeEndFrameExclusive=count,
        nativeTimeRange=[0., count*dt], lastSampleTime=(count-1)*dt,
        declaredEulerOrders=orders, fullHierarchyFk=True, joints=23,
        sourceRestPositions={b["name"]: rest[j].tolist() for j, b in enumerate(bones)},
        sourceRestQuaternions={b["name"]: rest_up.as_quat()[[3, 0, 1, 2]].tolist()
                               for b in bones},
        sourceConvention="cm/Y-up/right-handed; root XYZ replaces OFFSET; nonroot OFFSET-only",
        targetConvention="m/Z-up/right-handed/WXYZ; Rx90 applied once to globals",
        officialVersion="unknown", handOrientationRepair=False,
        smoothing=False, resampled=False, sourceChanged=False,
    )
    return SourceMotion(bones, points, quaternions, dt, provenance)
