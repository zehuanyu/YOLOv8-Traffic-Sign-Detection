import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo_traffic_sign.legacy import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained traffic sign model.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    args = parser.parse_args()
    run_eval(weights=args.weights, data=args.data)


if __name__ == "__main__":
    main()
