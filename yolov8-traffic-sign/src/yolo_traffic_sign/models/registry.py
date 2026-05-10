from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _register_module(name: str, obj) -> None:
    from ultralytics.nn import modules
    import ultralytics.nn.tasks as tasks

    setattr(modules, name, obj)
    setattr(tasks, name, obj)


def register_variant_modules(variant: str) -> None:
    """Register custom modules required by a given experiment variant."""
    if variant == "cbam":
        from cbam import CBAM

        _register_module("CBAM", CBAM)
        return

    if variant == "lcfe":
        from cca_light import CCA_Light

        _register_module("CCA_Light", CCA_Light)
        return

    if variant == "lcfe_v2":
        from cca_light_v2 import CCA_Light_v2

        _register_module("CCA_Light_v2", CCA_Light_v2)
        return

    if variant in {"imcmd", "imcmd_ts", "ts"}:
        import imcmd_modules

        for name in [
            "C2f_CA",
            "AMFF",
            "AGRFM",
            "Detect_IMCMD",
            "CoordinateAttention",
            "SKAttention",
        ]:
            _register_module(name, getattr(imcmd_modules, name))
        return
