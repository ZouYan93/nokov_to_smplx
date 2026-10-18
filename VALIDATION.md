# Local numerical validation

2026-10-18; Python 3.13, NumPy 2.1.3, SciPy 1.15.3.

- Synthetic suite: 32 passed.
- With the external original captures, licensed model and frozen Worker arrays:
  37 passed, zero skips.
- CLI conversion: original `deep-bow`, 1542 frames at 200 Hz; all output arrays
  load with `allow_pickle=False`, including explicit `up_axis=Z` and
  `gender=neutral`.

The local audit executed a SHA256-bound snapshot of the retained source reader
and fixed converter, with application imports removed and only the final
application envelope validator replaced by finite-array assertions. Expected
arrays were frozen before standalone conversion. Its profiles were additionally
checked against immutable profile02 evidence:
`0c77b929dad15d539ef93b0a844050688c791fd7d79f65abac4808d1da136d11`.
The retained Worker calibration JSON SHA256 was
`526ea16d730b252553ab5ab86ff506832687a113be061dac62b35907ace0aa9f`.
The standalone package imports none of these reference modules or files.

| Original | Native frames | Root axis-angle max difference (rad) | Body axis-angle max difference (rad) | Translation max difference (m) |
| --- | ---: | ---: | ---: | ---: |
| deep-bow | 1542 | 8.88e-16 | 8.88e-16 | 0 |
| deep-boxing | 2301 | 1.33e-15 | 1.55e-15 | 1.39e-17 |
| deep-wave | 2283 | 1.11e-15 | 1.33e-15 | 0 |
| magiclab-test1 | 2672 | 8.88e-16 | 1.55e-15 | 1.39e-17 |
| magiclab-take107 | 5121 | 1.78e-15 | 1.55e-15 | 1.39e-17 |

Across all 13,919 frames, source positions and WXYZ quaternions, output betas and
frame rates are elementwise equal to the retained reference. Maximum local
rotation-matrix difference is 1.84e-15; maximum global rotation-matrix difference
is 2.50e-15. Per-source correction matrices differ by at most 7.78e-16.

Rebuilt target joints differ from the retained target cache by at most
2.23e-16 m; shaped foot frames are elementwise equal. All thirteen consumed
single-ray body priors and all three leaf priors were compared as complete
rotations, including twist: maximum matrix difference 7.78e-16 and maximum
rotation difference 2.50e-16 rad. Ankle/toe corrections use their separate
runtime foot construction and pass the complete correction comparison.

Captures, model data, frozen reference arrays and the local audit receipt remain
external or under ignored `results/`. They are excluded from the distribution.
This verifies retained conversion parity and the declared source geometry
contract. Anatomical accuracy and vendor export versions remain unmeasured.
