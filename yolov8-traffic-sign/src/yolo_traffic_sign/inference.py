from __future__ import annotations

from pathlib import Path


def run_prediction(
    *,
    weights: str,
    source: str,
    variant: str = "baseline",
    project: str = "outputs/predict",
    name: str = "demo",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "",
    save_txt: bool = False,
) -> Path:
    from ultralytics import YOLO

    from yolo_traffic_sign.models import register_variant_modules

    register_variant_modules(variant)

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        save_txt=save_txt,
        verbose=True,
    )

    if not results:
        raise RuntimeError("Prediction finished but returned no results.")

    output_dir = Path(project) / name
    return output_dir
