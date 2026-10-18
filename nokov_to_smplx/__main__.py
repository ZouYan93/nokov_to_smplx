import argparse
import json

from .converter import convert_bvh, save_npz


def main(argv=None):
    parser = argparse.ArgumentParser(description="Common23 NOKOV BVH to native-clock SMPL-X NPZ")
    parser.add_argument("--bvh", "--bvh_file", required=True, dest="bvh")
    parser.add_argument("--model", "--smplx_model_path", required=True, dest="model",
                        help="User-licensed SMPLX_NEUTRAL.npz file")
    parser.add_argument("--output", "--output_file", required=True, dest="output")
    args = parser.parse_args(argv)
    try:
        result = convert_bvh(args.bvh, args.model)
        path = save_npz(result, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Conversion failed: {exc}\n")
    print(json.dumps(dict(
        output=str(path.resolve()), frames=len(result.trans),
        mocap_frame_rate=result.mocap_frame_rate,
        fit_status=result.provenance["fit"]["status"],
    )))


if __name__ == "__main__":
    main()
