# Contributing

## Development setup

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .[dev]
```

## Recommended workflow

- keep experimental outputs out of git
- prefer adding reusable code under `src/`
- keep quick one-off scripts inside `scripts/`
- document new experiments in `docs/` or `reports/`

## Before opening a PR

- run `pytest`
- update README or docs if behavior changes
- avoid hardcoding absolute local paths
