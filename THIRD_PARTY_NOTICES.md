# Third-party notices

The fixed sixteen shape coefficients and the global-rotation transfer convention
are derived from Joao Pedro Araujo's MIT-licensed `lafan_to_smplx`:
https://github.com/jaraujo98/lafan_to_smplx
at commit `d465aa7202ddc94f2cd557b682854437573e0dca`.
The copyright and full MIT terms are retained in `LICENSE`.

The NOKOV hierarchy reader and deterministic profile02 rest construction were
extracted from the owner's retained converter. Target coordinates, shape bases,
joint regressors, foot frames, and target-derived correction tables are excluded
from this distribution. They are rebuilt from the user's licensed model.

SMPL-X is a separately licensed model. This repository grants no model or motion
capture rights and includes no model data or original captures. Six integer foot
vertex identifiers refer to the user's SMPL-X topology; their coordinates are
read only at runtime. NumPy and SciPy remain under their respective licenses.
