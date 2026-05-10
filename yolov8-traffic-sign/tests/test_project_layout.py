from pathlib import Path

import yaml

from yolo_traffic_sign.paths import CONFIGS_DIR, DATA_DIR, DOCS_DIR, PROJECT_ROOT, REPORTS_DIR


def test_core_directories_exist():
    assert PROJECT_ROOT.exists()
    assert CONFIGS_DIR.exists()
    assert DATA_DIR.exists()
    assert DOCS_DIR.exists()
    assert REPORTS_DIR.exists()


def test_dataset_config_is_valid_yaml():
    dataset_cfg = CONFIGS_DIR / "datasets" / "lisa_day.yaml"
    with dataset_cfg.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    assert parsed["nc"] == 7
    assert len(parsed["names"]) == 7
