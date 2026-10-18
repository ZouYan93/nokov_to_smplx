"""Retained profile02 rest mapping; all target geometry is built from user data."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import SourceMotion, load_bvh, require

NAMES = (
    "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
    "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase",
    "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
    "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
MODEL_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
LAFAN_COMMIT = "d465aa7202ddc94f2cd557b682854437573e0dca"
# Published lafan_to_smplx constants, MIT, Joao Pedro Araujo (2025).
BETAS = np.array([
    1.4775, .6674, -1.1742, .4731, 1.2984, -.2159, 1.5276, -.3152,
    -.6441, -.2986, .5089, -.6354, .3321, -.1099, -.3060, -.7330,
], dtype=np.float32).astype(float)
# Topology identifiers only: heel, big toe, small toe. Coordinates stay in user data.
FOOT_VERTICES = {"Left": (8846, 5770, 5780), "Right": (8635, 8463, 8474)}
FEET = (7, 8, 10, 11)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    require(np.isfinite(vector).all() and np.all(norm > 1e-12), "Degenerate rest ray")
    return vector/norm


def _swing(a, b):
    a, b = _unit(a), _unit(b)
    axis, dot = np.cross(a, b), float(a @ b)
    size = np.linalg.norm(axis)
    if size > 1e-12:
        return Rotation.from_rotvec(axis/size*np.arctan2(size, dot)).as_matrix()
    if dot > 0:
        return np.eye(3)
    return Rotation.from_rotvec(
        _unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))*np.pi
    ).as_matrix()


def _foot_basis(up, forward):
    y = _unit(up)
    z = _unit(forward-y*np.dot(y, forward))
    x = _unit(np.cross(y, z))
    return np.column_stack((x, y, np.cross(x, y)))


def _model_foot_frame(heel, big_toe, small_toe):
    forward = (big_toe+small_toe)/2-heel
    lateral = np.cross([0., 1., 0.], forward)
    require(np.isfinite(forward).all() and np.linalg.norm(forward) >= 1e-8
            and np.linalg.norm(lateral) >= 1e-8, "Degenerate model foot landmarks")
    forward = forward/np.linalg.norm(forward)
    lateral = lateral/np.linalg.norm(lateral)
    up = np.cross(forward, lateral)
    return Rotation.from_matrix(np.column_stack((lateral, up, forward))).as_matrix()


@dataclass
class TargetRest:
    joints: np.ndarray
    foot_frames: dict
    model_sha256: str


def _target_from_arrays(template, directions, regressor, digest):
    """Match the retained model-buffer float32 then float64 arithmetic."""
    template = np.asarray(template, dtype=np.float32).astype(float)
    directions = np.asarray(directions[..., :16], dtype=np.float32).astype(float)
    regressor = np.asarray(regressor, dtype=np.float32).astype(float)
    require(template.shape == (10475, 3) and directions.shape == (10475, 3, 16)
            and regressor.shape == (55, 10475), "Expected neutral SMPL-X 16-shape geometry")
    require(all(np.isfinite(a).all() for a in (template, directions, regressor)),
            "Nonfinite model geometry")
    vertices = template+np.einsum("vck,k->vc", directions, BETAS)
    joints = (regressor @ vertices)[:22]
    feet = {side: _model_foot_frame(*vertices[list(indices)])
            for side, indices in FOOT_VERTICES.items()}
    return TargetRest(joints, feet, digest)


def load_target(model_file):
    """Read the licensed neutral NPZ explicitly; no model package or pickle."""
    path = Path(model_file).resolve()
    require(path.is_file(), "Licensed neutral SMPL-X NPZ is missing")
    with path.open("rb") as stream:
        before = path.stat()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
        require(digest.hexdigest() == MODEL_SHA256,
                "SMPL-X SHA256 differs from the retained neutral model")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as model:
            parents = model["kintree_table"][0, :22].copy()
            require(np.array_equal(parents[1:], PARENTS[1:]), "Unexpected SMPL-X hierarchy")
            target = _target_from_arrays(
                model["v_template"], model["shapedirs"], model["J_regressor"],
                digest.hexdigest(),
            )
        after = path.stat()
    identity = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino, s.st_ctime_ns)
    require(identity(before) == identity(after), "SMPL-X model changed during reading")
    return target


def _axis_prior(target):
    # These are the original straight-rest single-ray directions. Rebuilding
    # their shortest swings reproduces the sealed prior without storing any
    # target-derived matrix. Multi-ray joints are recomputed per source below.
    prior = np.repeat(np.eye(3)[None], 22, axis=0)
    for indices, axis in (
        ((1, 2, 4, 5), [0., -1., 0.]),
        ((3, 6, 12), [0., 1., 0.]),
        ((13, 16, 18), [1., 0., 0.]),
        ((14, 17, 19), [-1., 0., 0.]),
    ):
        for j in indices:
            child = next(i for i, parent in enumerate(PARENTS) if parent == j)
            prior[j] = _swing(target[child]-target[j], axis)
    return prior


def fixed_profile(source, target):
    """Construct constant rest rotations; no source motion values enter here."""
    lookup = {b["name"]: b for b in source.bones}
    rest = np.asarray([b["offset"] for b in source.bones], dtype=float)
    for j, bone in enumerate(source.bones):
        if bone["parent"] >= 0:
            rest[j] += rest[bone["parent"]]
    source_rest = rest[[source.names.index(n) for n in NAMES]]/100
    correction = _axis_prior(target.joints)
    feet = {}
    for side, ankle, toe in (("Left", 7, 10), ("Right", 8, 11)):
        up = -np.asarray(lookup[side+"Foot"]["offset"])
        ray = np.asarray(lookup[side+"ToeBase"]["offset"])
        tip = np.asarray(lookup[side+"ToeBase"]["endOffset"])
        basis, projected = _foot_basis(up, tip), _foot_basis(up, ray)
        sine = min(np.linalg.norm(np.cross(_unit(up), _unit(v))) for v in (ray, tip))
        require(sine > 1e-6 and Rotation.from_matrix(basis @ projected.T).magnitude() <= 1e-6,
                "Toe/End Site forward axes disagree or are degenerate")
        correction[[ankle, toe]] = basis @ target.foot_frames[side].T
        feet[side] = dict(frame=basis.tolist(), minimumShinForwardSine=float(sine))
    for j in range(22):
        children = [i for i, parent in enumerate(PARENTS) if parent == j]
        if j in FEET or not children:
            continue
        a = _unit(target.joints[children]-target.joints[j])
        b = _unit(source_rest[children]-source_rest[j])
        u, singular, vh = np.linalg.svd(b.T @ a/len(a))
        rank = np.count_nonzero(singular > max(singular[0]*1e-10, 1e-12))
        if rank < 2:
            require(len(a) == 1, "Multiple collinear rest rays require review")
            correction[j] = _swing(correction[j] @ a[0], b[0]) @ correction[j]
        else:
            correction[j] = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
    require(np.max(np.abs(correction.swapaxes(-1, -2) @ correction-np.eye(3))) < 1e-10
            and np.max(np.abs(np.linalg.det(correction)-1)) < 1e-10,
            "Invalid proper rest rotation")
    axial = ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head")
    straight = all(np.linalg.norm(np.asarray(lookup[n]["offset"])[[0, 2]])
                   <= abs(lookup[n]["offset"][1])*.002 for n in axial)
    arms_straight = all(
        np.linalg.norm(np.asarray(lookup[side+n]["offset"])[1:])
        <= .002*abs(lookup[side+n]["offset"][0])
        for side in ("Left", "Right") for n in ("Arm", "ForeArm", "Hand")
    )
    geometry_json = json.dumps(source.bones, sort_keys=True)
    return correction, dict(
        structure="body23-four-spine-rigid",
        geometry="straight-axial" if straight else "curved-axial",
        armGeometry="straight-lateral" if arms_straight else "curved-lateral",
        armRestTransverseRatioLimit=.4, sourceFeet=feet,
        geometrySha256=hashlib.sha256(geometry_json.encode()).hexdigest(),
        officialVersion="unknown",
        sourceConventionEvidence="rest-geometry compatibility; file-level export version unverified",
        staticMethod="equal-unit-ray proper Wahba; minimum swing from retained axis prior; leaf identity",
        targetGeometry="runtime from licensed neutral NPZ; fixed published16 shape",
    )


@dataclass
class Conversion:
    root_orient: np.ndarray
    pose_body: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    mocap_frame_rate: float
    provenance: dict


def convert_motion(source: SourceMotion, target: TargetRest):
    """Transfer all native frames; source Hips = shaped pelvis + translation."""
    require(target.model_sha256 == MODEL_SHA256, "Expected validated neutral target")
    require(target.joints.shape == (22, 3) and np.isfinite(target.joints).all(),
            "Invalid target rest joints")
    correction, profile = fixed_profile(source, target)
    indices = [source.names.index(n) for n in NAMES]
    world = source.world_rotations[:, indices]
    desired = world @ correction
    local = desired.copy()
    local[:, 1:] = desired[:, PARENTS[1:]].swapaxes(-1, -2) @ desired[:, 1:]
    pose = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 22, 3)
    provenance = dict(
        schema="nokov-to-smplx/1", method="nokov-fixed-rest/1",
        source=source.provenance, profile=profile,
        targetModel=dict(sha256=target.model_sha256, gender="neutral",
                         geometryDerivedAtRuntime=True),
        shape=dict(source="published lafan_to_smplx fixed neutral16", commit=LAFAN_COMMIT,
                   subjectShapeMeasured=False, shapeFitted=False),
        fit=dict(status="mapped-unmeasured", targetActionPoseFitted=False,
                 anatomicalGroundTruth=False),
        rootPolicy="source Hips minus shaped SMPL-X pelvis pivot; floor/centroid/hip shift absent",
        omittedSpine3="retained in full source FK and descendant globals",
        upAxis="Z", lengthUnit="m", officialVersion="unknown",
        nativeClockPreserved=True, motionFramesUsedToCalibrate=0, smoothing=False,
        handOrientationRepair=False, resampled=False,
        anatomyStatus="unverified; joint centres, axial twist and skin/contact unmeasured",
    )
    return Conversion(
        pose[:, 0].copy(), pose[:, 1:].reshape(-1, 63).copy(),
        source.positions[:, source.names.index("Hips")]-target.joints[0],
        BETAS.copy(), source.fps, provenance,
    )


def convert_bvh(bvh_file, smplx_model_path):
    return convert_motion(load_bvh(bvh_file), load_target(smplx_model_path))


def save_npz(result, output_file):
    """Write only plain numeric/string arrays; load with allow_pickle=False."""
    path = Path(output_file)
    require(path.suffix.lower() == ".npz", "Output must have .npz extension")
    require(not path.exists(), "Output already exists; choose a new output path")
    metadata = json.dumps(result.provenance, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, root_orient=result.root_orient, pose_body=result.pose_body,
            trans=result.trans, betas=result.betas,
            mocap_frame_rate=np.asarray(result.mocap_frame_rate),
            up_axis=np.asarray("Z"), gender=np.asarray("neutral"),
            provenance=np.asarray(metadata),
        )
    return path
