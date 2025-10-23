from pathlib import Path

import gradio as gr
from extra import get_dataset_names

DS_DIR = Path(__file__).parent / "datasets"
DS_DIR.mkdir(exist_ok=True)


def create_dataset(name):
    dataset_path = DS_DIR / name
    dataset_path.mkdir(exist_ok=True)
    (dataset_path / "originals").mkdir(exist_ok=True)
    (dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_path / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\n")
    (dataset_path / "README.md").write_text(f"# {name}\n\nThis is the README for the {name} dataset.")

    return f"Dataset {name} created successfully.", get_dataset_names()


def refresh_dataset(name):
    # Search for the dataset and refresh its contents or metadata
    dataset_path = DS_DIR / name
    if not dataset_path.exists():
        return f"Dataset {name} does not exist."

    # Count images in originals folder
    originals_path = dataset_path / "originals"
    num_originals = len(list(originals_path.glob("*")))
    return f"Dataset {name} refreshed successfully. Found {num_originals} original images."


def load_dataset(name):
    dataset_path = DS_DIR / name
    if not dataset_path.exists():
        return f"Dataset {name} does not exist.", None

    # Load dataset information
    readme_path = dataset_path / "README.md"
    if readme_path.exists():
        info = readme_path.read_text()
    else:
        info = f"No README found for dataset {name}."

    # Prepare dataset preview (this is a placeholder, actual implementation may vary)
    originals_path = dataset_path / "originals"
    image_paths = list(originals_path.glob("*"))

    return info, gr.Dataset(samples=[[str(p)] for p in image_paths])
