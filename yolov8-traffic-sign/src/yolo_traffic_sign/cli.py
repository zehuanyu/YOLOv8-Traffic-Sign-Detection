from __future__ import annotations

import argparse

from yolo_traffic_sign.inference import run_prediction
from yolo_traffic_sign.legacy import (
    run_baseline_train,
    run_cbam_train,
    run_eval,
    run_imcmd_train,
    run_lcfe_train,
    run_lcfe_v2_train,
    run_ts_train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for YOLO traffic sign experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training workflow.")
    train_parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "cbam", "lcfe", "lcfe_v2", "imcmd", "imcmd_ts", "ts"],
        help="Training variant to run.",
    )
    train_parser.add_argument("--data", default="configs/datasets/lisa_day.yaml")
    train_parser.add_argument("--model-size", default="s", choices=["n", "s", "m", "l", "x"])
    train_parser.add_argument("--model-type", default="small", choices=["small", "large"])
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--device", default="")
    train_parser.add_argument("--project", default="runs/traffic_sign")
    train_parser.add_argument("--name", default="experiment")
    train_parser.add_argument("--yaml-path", default="")
    train_parser.add_argument("--resume", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Run evaluation with legacy evaluator.")
    eval_parser.add_argument("--weights", required=True)
    eval_parser.add_argument("--data", required=True)

    predict_parser = subparsers.add_parser("predict", help="Run prediction and save visual outputs.")
    predict_parser.add_argument("--weights", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "cbam", "lcfe", "lcfe_v2", "imcmd", "imcmd_ts", "ts"],
    )
    predict_parser.add_argument("--project", default="outputs/predict")
    predict_parser.add_argument("--name", default="demo")
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--imgsz", type=int, default=640)
    predict_parser.add_argument("--device", default="")
    predict_parser.add_argument("--save-txt", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.variant == "baseline":
            run_baseline_train(
                data=args.data,
                model_size=args.model_size,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                project=args.project,
                name=args.name,
                resume=args.resume,
            )
            return

        if args.variant in {"imcmd", "imcmd_ts"}:
            run_imcmd_train(
                variant=args.variant,
                model_type=args.model_type,
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device or "0",
                name=args.name,
                resume=args.resume,
            )
            return

        if args.variant == "cbam":
            run_cbam_train(
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device or "0",
                name=args.name,
            )
            return

        if args.variant == "lcfe":
            yaml_path = args.yaml_path or "yolov8s_lcfe.yaml"
            run_lcfe_train(
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device or "0",
                name=args.name,
                yaml_path=yaml_path,
            )
            return

        if args.variant == "lcfe_v2":
            yaml_path = args.yaml_path or "yolov8s_lcfe_v2.yaml"
            run_lcfe_v2_train(
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device or "0",
                name=args.name,
                yaml_path=yaml_path,
            )
            return

        if args.variant == "ts":
            device = args.device if args.device != "" else 0
            run_ts_train(
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device,
            )
            return

    if args.command == "eval":
        run_eval(weights=args.weights, data=args.data)
        return

    if args.command == "predict":
        output_dir = run_prediction(
            weights=args.weights,
            source=args.source,
            variant=args.variant,
            project=args.project,
            name=args.name,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save_txt=args.save_txt,
        )
        print(f"Saved predictions to: {output_dir}")
        return

    parser.error("Unsupported command.")
