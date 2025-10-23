from pathlib import Path

DS_DIR = Path(__file__).parent / "datasets"


def get_dataset_names():
    return [d.name for d in DS_DIR.iterdir() if d.is_dir()]
