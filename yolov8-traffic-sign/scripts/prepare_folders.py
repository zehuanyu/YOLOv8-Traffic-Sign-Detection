from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    folders = [
        project_root / "data" / "external",
        project_root / "data" / "interim",
        project_root / "data" / "processed",
        project_root / "reports" / "figures",
        project_root / "artifacts",
        project_root / "outputs",
        project_root / "checkpoints",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"created: {folder}")


if __name__ == "__main__":
    main()
