# GitHub Release Checklist

## Before first push

- replace placeholder dataset paths in `configs/datasets/*.yaml`
- confirm `README.md` reflects the final experiment story
- keep large datasets and weights out of git
- include 2 to 4 representative visual results in the repository
- verify `requirements.txt` matches the environment you actually used

## Good portfolio additions

- a benchmark comparison table
- one ablation study summary
- qualitative prediction images
- a short explanation of what each custom module contributes
- final best-model metrics in the README

## Recommended first GitHub release assets

- `best.pt` only if file size is acceptable and licensing allows it
- experiment report PDF
- result charts from `charts/`
- sample predictions
