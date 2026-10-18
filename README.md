# nokov_to_smplx

A small NumPy/SciPy CLI and library for common23 NOKOV BVH exports. It retains the
`nokov-fixed-rest/1` profile02 mapping, full source FK, source Hips trajectory,
and original frame timing. Output has `fit.status = "mapped-unmeasured"`.

## Run

Python 3.10+ and a separately licensed neutral SMPL-X NPZ are required.

```sh
python -m pip install .
python -m nokov_to_smplx --bvh captures/action.bvh \
  --model models/smplx/SMPLX_NEUTRAL.npz --output results/action.npz
```

The installed command is `nokov-to-smplx`. The aliases `--bvh_file`,
`--smplx_model_path`, and `--output_file` are also accepted. `--model` names the
NPZ file itself. Existing output files are refused.

The model SHA256 must be:

```text
376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992
```

There are no model downloads, Torch, GMR, application imports, or runtime
calibration files. `nokov_fixed_rest.json` and its licensed target geometry are
not distributed. Model files, captures, and generated results are ignored by Git.

## Library and output

```python
import json
import numpy as np
from nokov_to_smplx import convert_bvh, save_npz

result = convert_bvh("captures/action.bvh", "models/smplx/SMPLX_NEUTRAL.npz")
save_npz(result, "results/action.npz")
with np.load("results/action.npz", allow_pickle=False) as motion:
    metadata = json.loads(motion["provenance"].item())
```

| NPZ array | Shape | Meaning |
| --- | --- | --- |
| `root_orient` | `(F, 3)` | Root axis-angle, radians |
| `pose_body` | `(F, 63)` | 21 local body axis-angles in SMPL-X order |
| `trans` | `(F, 3)` | SMPL-X translation, metres |
| `betas` | `(16,)` | Published fixed neutral shape, float32 values promoted to float64 |
| `mocap_frame_rate` | scalar | Exact reciprocal of BVH Frame Time |
| `up_axis` | scalar string | `Z` |
| `gender` | scalar string | `neutral` |
| `provenance` | scalar string | Finite JSON with source/model SHA256, clock, per-joint Euler orders, geometry and fit status |

All arrays load with `allow_pickle=False`. Provenance is also available as the
ordinary JSON-compatible `result.provenance` dictionary. Pose and translation
arrays use float64. Face and finger poses are absent.

## Supported source contract

The 23 named joints and their parents must be exactly:

```text
Hips
  Spine -> Spine1 -> Spine2 -> Spine3
    [Spine3] Neck -> Head
    [Spine3] LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand
    [Spine3] RightShoulder -> RightArm -> RightForeArm -> RightHand
  LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToeBase
  RightUpLeg -> RightLeg -> RightFoot -> RightToeBase
```

Both toes need a meaningful End Site. Each joint declares its own three
intrinsic Euler channels, in any permutation of XYZ; mixed orders are supported.
Root channels begin with XYZ position. Root position replaces root OFFSET.
All other joints use OFFSET and three rotation channels.

Supported observed raw axes are right-handed centimetres/Y-up, signed X-lateral
arms/hips, downward Y legs, and forward Z toes. The retained geometric admission
rules allow sagittal curved torso offsets up to `abs(z)/y = 0.4`, arm transverse
ratio up to `0.4`, small lateral axial roundoff (`0.002`), and sagittal leg ratio
up to `0.1`. Hip vertical ratio is at most `0.4`, and toe lateral ratio is at most
`0.002`. Thigh and shin offsets must each be between 0.15 and 0.7 m under the cm
convention. Toe and End Site forwards must agree after projection against the
shin-up direction and form nondegenerate frames.

`straight-axial` / `curved-axial` and `straight-lateral` / `curved-lateral` describe
rest geometry. They do not establish official vendor V1/V2/V3 labels.
`officialVersion` is always `unknown`. Custom8, unrelated 59-joint exports,
missing toes, nonroot translation channels, arbitrary axes, and malformed
motion tables are rejected. Limits match the retained reader: 32 MiB file,
64 KiB hierarchy, 2..72000 frames, 1..240 Hz, and at most 300.1 seconds.

## Mapping and limits

The reader evaluates all 23 source joints before selecting semantic body22.
The extra Spine3 contributes to every descendant global rotation and position.
Positions change from cm/Y-up to m/Z-up once; `Rx(+90 degrees)` left-multiplies
source global rotations once.

Target rest joints and shaped heel/forefoot frames are rebuilt from the licensed
NPZ using the fixed published16 betas. Historical float32 model-buffer rounding
is retained before float64 geometry arithmetic. Single-ray straight-rest priors
are rebuilt by shortest swings to signed source axes, leaves retain identity,
and per-source rest rays construct the same proper-Wahba/minimum-swing mapping
as profile02. Feet use source shin/End Site frames and shaped target foot frames.
The resulting constant correction right-multiplies each source global rotation;
body22 parent-relative rotations produce the output axis-angles.

`trans = source_Hips_m_Zup - shaped_model_pelvis_native_m`. Standard SMPL-X root
rotation pivots about its shaped pelvis, so `trans + shaped_pelvis` retains the
source Hips path. There is no floor shift, centroid shift, hip anchor adjustment,
shape or motion-pose fit, smoothing, hand repair, interpolation, or resampling.
Deterministic rest-orientation construction uses only hierarchy offsets and
target rest geometry, never action frames.

This is a conditional rig mapping. Joint centres, axial anatomy, skin/contact,
subject shape, and anatomical accuracy remain unmeasured. Fixed rotations
preserve rotation increments; source jumps and collapsed-spine articulation
remain. No anatomical quality pass is asserted.

## Tests

```sh
python -m pip install ".[test]"
python -m pytest -q
```

Synthetic tests construct their BVH/model data in temporary directories. The
optional five-original regression uses external captures and independently
frozen Worker output NPZs; neither is shipped:

```sh
NOKOV_TEST_DATA=/external/audit-assets \
NOKOV_TEST_MODEL=/licensed/SMPLX_NEUTRAL.npz \
NOKOV_TEST_REFERENCE=/external/frozen-worker-npz \
python -m pytest -q -s tests/test_external.py
```

Fixture relative paths and SHA256 pins are listed in that test. Reference files
are named `<action>.npz` and contain `root_orient`, `pose_body`, `trans`, `betas`,
`mocap_frame_rate`, `source_positions`, `source_quaternions`, `corrections`,
`target_rest`, `target_foot_frames`, and `prior_corrections`. They are local audit
material containing licensed target-derived data and must remain external or
inside ignored `results/`. Missing external assets produce explicit skips.

Code is MIT-licensed; attribution and model/data boundaries are in
`THIRD_PARTY_NOTICES.md`. A short Chinese example is in `README.zh-CN.md`.
