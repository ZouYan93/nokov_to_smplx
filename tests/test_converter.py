"""Authored synthetic data only; no model or capture fixture is distributed."""

import hashlib
import itertools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nokov_to_smplx import convert_motion, load_bvh, save_npz
from nokov_to_smplx.bvh import UP
from nokov_to_smplx.converter import (
    BETAS, FOOT_VERTICES, MODEL_SHA256, NAMES, PARENTS, TargetRest,
    _swing, _target_from_arrays, fixed_profile, load_target,
)

TREE = {
    "Hips": ["Spine", "LeftUpLeg", "RightUpLeg"],
    "Spine": ["Spine1"], "Spine1": ["Spine2"], "Spine2": ["Spine3"],
    "Spine3": ["LeftShoulder", "RightShoulder", "Neck"],
    "Neck": ["Head"],
}
for side in ("Left", "Right"):
    for parent, child in (
        ("Shoulder", "Arm"), ("Arm", "ForeArm"), ("ForeArm", "Hand"),
        ("UpLeg", "Leg"), ("Leg", "Foot"), ("Foot", "ToeBase"),
    ):
        TREE[side+parent] = [side+child]


def make_bvh(path, *, curved=False, order=None, values=None, dt=.0111111):
    orders = itertools.cycle(itertools.permutations("XYZ"))
    bones = []

    def joint(name, parent):
        sign = -1 if name.startswith("Right") else 1
        if name == "Hips":
            offset = [83., 92., -19.]
        elif name.endswith("UpLeg"):
            offset = [sign*12., 0., 0.]
        elif name.endswith(("Leg", "Foot")):
            offset = [0., -40., 1. if curved else 0.]
        elif name.endswith("ToeBase"):
            offset = [0., -4., 15.]
        elif name.endswith("Shoulder"):
            offset = [sign*4., 5., 0.]
        elif name.endswith(("Arm", "ForeArm", "Hand")):
            offset = [sign*20., -3. if curved else 0., 4. if curved else 0.]
        else:
            offset = [0., 10., 2. if curved else 0.]
        axes = order or "".join(next(orders))
        channels = (["Xposition", "Yposition", "Zposition"] if parent < 0 else [])
        channels += [a+"rotation" for a in axes]
        idx = len(bones)
        start = sum(len(b["channels"]) for b in bones)
        bones.append(dict(name=name, parent=parent, offset=offset, channels=channels,
                          indices=list(range(start, start+len(channels)))))
        lines = [f"{'ROOT' if parent < 0 else 'JOINT'} {name}", "{",
                 "OFFSET "+" ".join(map(str, offset)),
                 f"CHANNELS {len(channels)} "+" ".join(channels)]
        for child in TREE.get(name, []):
            lines += joint(child, idx)
        if name not in TREE:
            lines += ["End Site", "{", "OFFSET 0 0 5", "}"]
        return lines+["}"]

    lines = ["HIERARCHY"]+joint("Hips", -1)
    if values is None:
        values = np.random.default_rng(182).uniform(-40, 40, (8, 72))
        values[:, :3] = np.arange(24).reshape(8, 3)+[7, 101, -13]
        values[0, 3:] = 0
    text = "\n".join(lines+["MOTION", f"Frames: {len(values)}", f"Frame Time: {dt}"])
    text += "\n"+"\n".join(" ".join(format(x, ".17g") for x in row) for row in values)+"\n"
    path.write_text(text, encoding="utf-8")
    return bones, values


def explicit_fk(bones, values):
    """Independent scalar-axis composition, including noncommuting mixed orders."""
    world = np.empty((len(values), 23, 3, 3))
    points = np.empty((len(values), 23, 3))
    for j, bone in enumerate(bones):
        local = np.repeat(np.eye(3)[None], len(values), axis=0)
        for channel, index in zip(bone["channels"], bone["indices"]):
            if not channel.endswith("rotation"):
                continue
            c, s = np.cos(np.deg2rad(values[:, index])), np.sin(np.deg2rad(values[:, index]))
            r = np.repeat(np.eye(3)[None], len(values), axis=0)
            u, v = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[channel[0]]
            r[:, u, u] = c; r[:, v, v] = c; r[:, u, v] = -s; r[:, v, u] = s
            local = local @ r
        parent = bone["parent"]
        world[:, j] = local if parent < 0 else world[:, parent] @ local
        points[:, j] = values[:, :3] if parent < 0 else (
            points[:, parent]+world[:, parent] @ bone["offset"]
        )
    return UP @ world, points @ UP.T / 100


def synthetic_target(source):
    rest = np.zeros((23, 3))
    for j, b in enumerate(source.bones):
        rest[j] = np.asarray(b["offset"])+(rest[b["parent"]] if b["parent"] >= 0 else 0)
    joints = rest[[source.names.index(n) for n in NAMES]]/100
    joints -= joints[0].copy()
    joints += np.random.default_rng(7).normal(0, .012, joints.shape)
    frames = {s: Rotation.from_euler("xyz", a, degrees=True).as_matrix()
              for s, a in (("Left", [3, 7, 2]), ("Right", [-2, -5, 1]))}
    return TargetRest(joints, frames, MODEL_SHA256)


def output_matrices(result):
    pose = np.concatenate((result.root_orient[:, None], result.pose_body.reshape(-1, 21, 3)), axis=1)
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    world = local.copy()
    for j, p in enumerate(PARENTS[1:], 1):
        world[:, j] = world[:, p] @ local[:, j]
    return local, world


@pytest.mark.parametrize("curved", [False, True])
def test_full23_declared_intrinsic_fk_and_single_axis_conversion(tmp_path, curved):
    path = tmp_path/"synthetic.bvh"
    bones, values = make_bvh(path, curved=curved)
    before = path.read_bytes()
    source = load_bvh(path)
    rotations, points = explicit_fk(bones, values)
    np.testing.assert_allclose(source.world_rotations, rotations, atol=4e-15, rtol=0)
    np.testing.assert_allclose(source.positions, points, atol=4e-15, rtol=0)
    np.testing.assert_array_equal(source.positions[:, 0], values[:, :3] @ UP.T/100)
    np.testing.assert_allclose(source.world_rotations[0], np.repeat(UP[None], 23, axis=0),
                               atol=1e-15, rtol=0)
    assert len(set(source.provenance["declaredEulerOrders"].values())) == 6
    assert source.fps == 1/.0111111
    assert source.provenance["nativeTimeRange"] == [0, 8*.0111111]
    assert source.provenance["sourceSha256"] == hashlib.sha256(before).hexdigest()
    assert before == path.read_bytes()


@pytest.mark.parametrize("order", ["".join(p) for p in itertools.permutations("XYZ")])
def test_all_uniform_orders_have_independent_matrix_oracle(tmp_path, order):
    path = tmp_path/"synthetic.bvh"
    bones, values = make_bvh(path, order=order)
    source = load_bvh(path)
    np.testing.assert_allclose(source.world_rotations, explicit_fk(bones, values)[0],
                               atol=4e-15, rtol=0)


def test_fixed_mapping_pelvis_pivot_and_rotation_increments(tmp_path):
    path = tmp_path/"synthetic.bvh"
    make_bvh(path, curved=True)
    source = load_bvh(path)
    target = synthetic_target(source)
    correction, profile = fixed_profile(source, target)
    result = convert_motion(source, target)
    local, world = output_matrices(result)
    ids = [source.names.index(n) for n in NAMES]
    expected = source.world_rotations[:, ids] @ correction
    np.testing.assert_allclose(world, expected, atol=5e-15, rtol=0)
    np.testing.assert_allclose(local[:, 1:],
                               expected[:, PARENTS[1:]].swapaxes(-1, -2) @ expected[:, 1:],
                               atol=5e-15, rtol=0)
    np.testing.assert_allclose(result.trans+target.joints[0], source.positions[:, 0],
                               atol=3e-16, rtol=0)
    original = source.world_rotations[:, ids]
    np.testing.assert_allclose(
        world[1:] @ world[:-1].swapaxes(-1, -2),
        original[1:] @ original[:-1].swapaxes(-1, -2), atol=8e-15, rtol=0,
    )
    np.testing.assert_array_equal(correction[[15, 20, 21]],
                                  np.repeat(np.eye(3)[None], 3, axis=0))
    np.testing.assert_array_equal(result.betas, BETAS)
    assert profile["geometry"] == "curved-axial"
    assert profile["armGeometry"] == "curved-lateral"
    assert result.provenance["fit"]["status"] == "mapped-unmeasured"
    assert result.provenance["officialVersion"] == "unknown"
    assert result.provenance["smoothing"] is False
    assert result.provenance["handOrientationRepair"] is False
    assert result.provenance["resampled"] is False
    json.dumps(result.provenance, allow_nan=False)


def test_spine3_composition_toe_articulation_and_rest_only_profile(tmp_path):
    a = tmp_path/"a.bvh"; b = tmp_path/"b.bvh"
    bones, values = make_bvh(a, values=np.zeros((3, 72)))
    spine = next(x for x in bones if x["name"] == "Spine3")
    toe = next(x for x in bones if x["name"] == "LeftToeBase")
    values[1, spine["indices"][0]] = 33
    values[2, toe["indices"][1]] = 24
    make_bvh(b, values=values)
    base, changed = load_bvh(a), load_bvh(b)
    target = synthetic_target(base)
    np.testing.assert_array_equal(fixed_profile(base, target)[0], fixed_profile(changed, target)[0])
    _, world = output_matrices(convert_motion(changed, target))
    assert np.max(np.abs(world[0, 15]-world[1, 15])) > .1
    np.testing.assert_allclose(world[0, 9], world[1, 9], atol=1e-15, rtol=0)
    assert np.max(np.abs(world[0, 10]-world[2, 10])) > .1
    np.testing.assert_allclose(world[0, 7], world[2, 7], atol=1e-15, rtol=0)


@pytest.mark.parametrize("bad", [
    "truncated", "extra", "width", "nan", "clock", "toe", "parent",
    "translations", "units", "arm", "torso", "end-site", "disagree", "duplicate-axis",
])
def test_bad_or_unsupported_source_is_refused(tmp_path, bad):
    path = tmp_path/"synthetic.bvh"
    make_bvh(path)
    text = path.read_text()
    if bad == "truncated": text = "\n".join(text.splitlines()[:-1])+"\n"
    if bad == "extra": text += text.splitlines()[-1]+"\n"
    if bad == "width": text = text.rsplit(" ", 1)[0]+"\n"
    if bad == "nan": text = text.replace("Frame Time: 0.0111111\n7", "Frame Time: 0.0111111\nnan")
    if bad == "clock": text = text.replace("Frame Time: 0.0111111", "Frame Time: 0")
    if bad == "toe": text = text.replace("LeftToeBase", "MissingToe")
    if bad == "parent": text = text.replace("LeftShoulder", "RightShoulder", 1)
    if bad == "translations": text = text.replace("CHANNELS 3", "CHANNELS 6 Xposition Yposition Zposition", 1)
    if bad == "units": text = text.replace("OFFSET 0.0 -40.0 0.0", "OFFSET 0.0 -0.4 0.0")
    if bad == "arm": text = text.replace("OFFSET 20.0 0.0 0.0", "OFFSET 20.0 20.0 0.0")
    if bad == "torso": text = text.replace("OFFSET 0.0 10.0 0.0", "OFFSET 0.0 10.0 5.0")
    if bad == "end-site": text = text.replace("OFFSET 0 0 5", "OFFSET 0 0 0")
    if bad == "disagree": text = text.replace("OFFSET 0 0 5", "OFFSET 1 0 5")
    if bad == "duplicate-axis": text = text.replace("Xrotation Yrotation Zrotation", "Xrotation Yrotation Yrotation", 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        source = load_bvh(path)
        convert_motion(source, synthetic_target(source))


@pytest.mark.parametrize("newline,bom", [("\n", False), ("\r\n", False), ("\r", True)])
def test_header_bom_newlines_preserve_clock_and_metadata(tmp_path, newline, bom):
    path = tmp_path/"synthetic.bvh"
    make_bvh(path)
    baseline = load_bvh(path)
    text = path.read_text()
    path.write_bytes((("\ufeff" if bom else "")+text.replace("\n", newline)).encode("utf-8"))
    source = load_bvh(path)
    np.testing.assert_array_equal(source.positions, baseline.positions)
    np.testing.assert_array_equal(source.quaternions_wxyz, baseline.quaternions_wxyz)
    assert source.provenance["headerSha256"] == baseline.provenance["headerSha256"]


def test_runtime_shaped_geometry_uses_rounded16_components_and_landmarks():
    template = np.zeros((10475, 3), dtype=np.float64)
    for indices in FOOT_VERTICES.values():
        template[list(indices)] = [[0, 0, 0], [.03, .01, .2], [-.02, .01, .2]]
    directions = np.zeros((10475, 3, 17), dtype=np.float64)
    directions[:22, 1, :16] = np.arange(16)*.000012345678901
    directions[..., 16] = 1e6
    regressor = np.zeros((55, 10475))
    regressor[np.arange(22), np.arange(22)] = 1
    target = _target_from_arrays(template, directions, regressor, MODEL_SHA256)
    expected = template[:22].astype(np.float32).astype(float)+(
        directions[:22, :, :16].astype(np.float32).astype(float)*BETAS
    ).sum(axis=-1)
    np.testing.assert_allclose(target.joints, expected, atol=1e-17, rtol=0)
    for frame in target.foot_frames.values():
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-15, rtol=0)
        forward = np.array([.005, .01, .2])
        np.testing.assert_allclose(frame[:, 2], forward/np.linalg.norm(forward), atol=1e-8, rtol=0)


def test_model_pin_is_mandatory_before_npz_loading(tmp_path):
    model = tmp_path/"wrong.npz"
    model.write_bytes(b"opaque wrong model")
    with pytest.raises(ValueError, match="SHA256"):
        load_target(model)


@pytest.mark.parametrize("a,b", [([1, 0, 0], [-1, 0, 0]), ([1, 0, 0], [1, 0, 0])])
def test_shortest_swing_parallel_and_antiparallel(a, b):
    rotation = _swing(a, b)
    np.testing.assert_allclose(rotation @ a, b, atol=1e-15, rtol=0)
    assert np.linalg.det(rotation) == pytest.approx(1)


def test_cli_npz_contract_plain_arrays_and_output_collision(tmp_path, monkeypatch, capsys):
    from nokov_to_smplx import __main__ as cli
    source_file = tmp_path/"synthetic.bvh"
    make_bvh(source_file)
    source = load_bvh(source_file)
    result = convert_motion(source, synthetic_target(source))
    output = tmp_path/"out.npz"
    monkeypatch.setattr(cli, "convert_bvh", lambda *_: result)
    cli.main(["--bvh", str(source_file), "--model", "licensed.npz", "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["frames"] == 8
    with np.load(output, allow_pickle=False) as data:
        assert set(data.files) == {
            "root_orient", "pose_body", "trans", "betas", "mocap_frame_rate",
            "up_axis", "gender", "provenance",
        }
        assert data["up_axis"].item() == "Z" and data["gender"].item() == "neutral"
        assert data["root_orient"].shape == (8, 3) and data["pose_body"].shape == (8, 63)
        assert data["trans"].shape == (8, 3) and data["betas"].shape == (16,)
        assert data["mocap_frame_rate"].item() == source.fps
        assert all(not data[k].dtype.hasobject for k in data.files)
        assert json.loads(data["provenance"].item()) == result.provenance
    with pytest.raises(ValueError, match="already exists"):
        save_npz(result, output)
