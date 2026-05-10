"""Wrapper helpers around legacy root-level training and evaluation scripts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_root_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_module(module_name: str):
    _ensure_root_on_path()
    return importlib.import_module(module_name)


def run_baseline_train(*, data: str, model_size: str, epochs: int, batch: int, imgsz: int,
                       device: str, project: str, name: str, resume: bool):
    module = load_module("train_yolo")
    return module.train_traffic_sign_detector(
        data_yaml=data,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch,
        img_size=imgsz,
        device=device,
        project=project,
        name=name,
        resume=resume,
    )


def run_imcmd_train(*, variant: str, model_type: str, data: str, epochs: int,
                    batch: int, device: str, name: str | None, resume: bool):
    module = load_module("train_imcmd")
    return module.train_imcmd(
        model_type=model_type,
        variant=variant,
        data_yaml=data,
        epochs=epochs,
        batch=batch,
        device=device,
        name=name,
        resume=resume,
    )


def run_ts_train(*, epochs: int, batch: int, imgsz: int, device: int | str):
    module = load_module("train_ts")
    return module.train_ts(epochs=epochs, batch=batch, imgsz=imgsz, device=device)


def run_cbam_train(*, data: str, epochs: int, batch: int, device: str, name: str):
    module = load_module("train_cbam")
    return module.train_cbam(
        data_yaml=data,
        epochs=epochs,
        batch=batch,
        device=device,
        name=name,
    )


def run_lcfe_train(*, data: str, epochs: int, batch: int, imgsz: int, device: str,
                   name: str, yaml_path: str):
    module = load_module("train_lcfe")
    return module.train_with_lcfe(
        yaml_path=yaml_path,
        data_yaml=data,
        epochs=epochs,
        batch_size=batch,
        img_size=imgsz,
        device=device,
        name=name,
    )


def run_lcfe_v2_train(*, data: str, epochs: int, batch: int, device: str,
                      name: str, yaml_path: str):
    module = load_module("train_lcfe_v2")
    return module.train_v2(
        yaml_path=yaml_path,
        data_yaml=data,
        epochs=epochs,
        batch_size=batch,
        device=device,
        name=name,
    )


def run_eval(*, weights: str, data: str):
    module = load_module("eval_model")
    if hasattr(module, "evaluate_model"):
        return module.evaluate_model(weights_path=weights, data_yaml=data)
    if hasattr(module, "main"):
        old_argv = sys.argv[:]
        try:
            sys.argv = ["eval_model.py", "--model", weights, "--data", data]
            return module.main()
        finally:
            sys.argv = old_argv
    raise AttributeError("Legacy eval_model.py does not expose evaluate_model() or main().")
