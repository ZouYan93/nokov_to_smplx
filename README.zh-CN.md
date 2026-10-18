# NOKOV 转 SMPL-X

依赖 Python 3.10+、NumPy、SciPy，以及自行获得授权的中性 SMPL-X NPZ。

```sh
python -m pip install .
python -m nokov_to_smplx --bvh captures/action.bvh --model models/smplx/SMPLX_NEUTRAL.npz --output results/action.npz
```

支持 common23 四段脊柱骨架，每个关节遵循其声明的内禀 Euler 顺序。
厘米/Y 向上转换为米/Z 向上，保留原始帧率与 Hips 轨迹。
模型静态关节与足部坐标系在运行时生成；仓库仅含代码与公开形状参数。

输出状态为 `mapped-unmeasured`，人体解剖精度尚无测量证据。
流程保留原动作，省略平滑、手部修复、动作拟合与重采样。
几何描述采用直脊柱、弯曲脊柱、直臂、弯曲臂；官方版本字段为 `unknown`。
详细骨架约束、模型 SHA256、输出字段及测试命令见 `README.md`。
