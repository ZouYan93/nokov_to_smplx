"""Optional original captures plus independently frozen Worker numeric outputs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nokov_to_smplx import convert_motion, load_bvh
from nokov_to_smplx.converter import (
    FEET, NAMES, PARENTS, _axis_prior, fixed_profile, load_target,
)

CASES = (
    ("deep-bow", "nokov-version-coverage-20261018/originals/deep-robotics/bow.bvh",
     "7278f2deb24fe9333596cd7cad6bd6374eb0c48da8e2bb65e4bfe986d3cc13be", 1542),
    ("deep-boxing", "nokov-version-coverage-20261018/originals/deep-robotics/boxing.bvh",
     "b942057fa087341eed8eb54447055e8a349ff6363822ff7ea7d498e0dcf2b0e8", 2301),
    ("deep-wave", "nokov-version-coverage-20261018/originals/deep-robotics/wave.bvh",
     "6f409582b4e0a45bbd8150549c43b87f3b612d293c56068b6e87163463ab91e6", 2283),
    ("magiclab-test1", "nokov-version-coverage-20261018/originals/magiclab/Test1-Body1.bvh",
     "a1685b420ba2d47ce06c48515cde755b632881a7f7d4c27be3cba1f62a6b34c9", 2672),
    ("magiclab-take107", "nokov-public-coverage-20261018/samples-01/Take_107_Skeleton0.bvh",
     "f0722be90e357cbcb01548d46e4b08cf33f163141f3f00ecaf17a70039148860", 5121),
)


@pytest.fixture(scope="module")
def external():
    keys = ("NOKOV_TEST_DATA", "NOKOV_TEST_MODEL", "NOKOV_TEST_REFERENCE")
    if not all(os.environ.get(k) for k in keys):
        pytest.skip("External captures, licensed model and frozen Worker arrays are optional; see README")
    return tuple(Path(os.environ[k]) for k in keys)


@pytest.fixture(scope="module")
def target(external):
    return load_target(external[1])


@pytest.mark.parametrize("action,relative,digest,count", CASES, ids=[c[0] for c in CASES])
def test_whole_native_motion_matches_frozen_worker(external, target, action, relative, digest, count):
    capture = external[0]/relative
    expected = external[2]/f"{action}.npz"
    if not capture.is_file() or not expected.is_file():
        pytest.skip(f"Optional external asset missing for {action}")
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == digest
    source = load_bvh(capture)
    result = convert_motion(source, target)
    correction, _ = fixed_profile(source, target)
    with np.load(expected, allow_pickle=False) as reference:
        np.testing.assert_allclose(target.joints, reference["target_rest"], atol=5e-15, rtol=0)
        np.testing.assert_allclose(
            [target.foot_frames[s] for s in ("Left", "Right")],
            reference["target_foot_frames"], atol=5e-15, rtol=0,
        )
        prior_ids = [j for j in range(22) if j not in FEET and sum(p == j for p in PARENTS) <= 1]
        prior = _axis_prior(target.joints)[prior_ids]
        old_prior = reference["prior_corrections"][prior_ids]
        np.testing.assert_allclose(prior, old_prior, atol=2e-14, rtol=0)
        full_prior_angle = Rotation.from_matrix(prior @ old_prior.swapaxes(-1, -2)).magnitude()
        assert float(full_prior_angle.max()) < 2e-14
        correction_error = float(np.max(np.abs(correction-reference["corrections"])))
        np.testing.assert_allclose(correction, reference["corrections"], atol=2e-14, rtol=0)
        np.testing.assert_array_equal(source.positions, reference["source_positions"])
        np.testing.assert_array_equal(source.quaternions_wxyz, reference["source_quaternions"])
        errors = {}
        for key in ("root_orient", "pose_body", "trans"):
            actual = getattr(result, key)
            errors[key+"Max"] = float(np.max(np.abs(actual-reference[key])))
            np.testing.assert_allclose(actual, reference[key], atol=8e-14, rtol=0)
        np.testing.assert_array_equal(result.betas, reference["betas"])
        assert result.mocap_frame_rate == reference["mocap_frame_rate"].item()
    assert len(result.trans) == count
    assert source.provenance["sourceSha256"] == digest
    assert source.provenance["officialVersion"] == "unknown"
    print(json.dumps(dict(action=action, frames=count, priorFullRotationMaxRad=float(full_prior_angle.max()),
                          correctionsMax=correction_error, **errors)))
