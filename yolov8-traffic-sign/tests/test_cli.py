from yolo_traffic_sign.cli import build_parser


def test_cli_train_defaults():
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.variant == "baseline"
    assert args.epochs == 100


def test_cli_eval_command():
    parser = build_parser()
    args = parser.parse_args(["eval", "--weights", "best.pt", "--data", "configs/datasets/lisa_day.yaml"])
    assert args.command == "eval"
    assert args.weights == "best.pt"


def test_cli_predict_command():
    parser = build_parser()
    args = parser.parse_args(
        ["predict", "--weights", "best.pt", "--source", "assets/sample_predictions", "--variant", "cbam"]
    )
    assert args.command == "predict"
    assert args.variant == "cbam"
