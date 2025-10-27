from pathlib import Path

import gradio as gr
from PIL import Image

DS_DIR = Path(__file__).parent / "datasets"
DS_DIR.mkdir(exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def get_dataset_images(dataset_name: str):
    dataset_path = DS_DIR / dataset_name
    if not dataset_path.exists():
        return []

    originals_path = dataset_path / "originals"
    image_paths = list(originals_path.glob("*"))

    # Filter images by allowed extensions
    image_paths = [(p, p.name) for p in image_paths if p.suffix in ALLOWED_IMAGE_EXTENSIONS]

    return image_paths


def get_dataset_summary(dataset_name: str):
    dataset_path = DS_DIR / dataset_name
    if not dataset_path.exists():
        return f"Dataset **{dataset_name}** does not exist."

    originals_path = dataset_path / "originals"

    total_files = list(originals_path.glob("*"))
    image_paths = [p for p in total_files if p.suffix in ALLOWED_IMAGE_EXTENSIONS]
    num_originals = len(image_paths)

    # Create summary in markdown format
    summary_msg = "# Dataset Summary\n"
    summary_msg += "## Original Images\n"
    summary_msg += f"- Total: {len(image_paths)}\n"
    summary_msg += f"- Allowed: {num_originals}\n"

    return summary_msg


def _get_image_metadata(image_path: Path):
    temp_img = Image.open(image_path)
    width, height = temp_img.size
    img_format = temp_img.format
    img_size = image_path.stat().st_size
    temp_img.close()

    return f"""
    - Format: {img_format}
    - Dimensions: {width}x{height}
    - Size: {img_size} bytes
    """


# ============================== Handlers ==============================


def get_dataset_names():
    datasets = [d.name for d in DS_DIR.iterdir() if d.is_dir()]
    datasets.sort()

    return datasets


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

    dropdown_update = gr.update(choices=get_dataset_names())

    return dropdown_update


def refresh_dataset(name):
    # Search for the dataset and refresh its contents or metadata
    dataset_path = DS_DIR / name
    if not dataset_path.exists():
        return f"Dataset **{name}** does not exist."

    # Results
    summary_msg = get_dataset_summary(name)
    gallery_update = get_dataset_images(name)

    return summary_msg, gallery_update


def select_image(name, evt: gr.SelectData):
    dataset_path = DS_DIR / name
    if not dataset_path.exists():
        return f"Dataset **{name}** does not exist."

    image_caption = evt.value["caption"]
    selected_image_path = dataset_path / "originals" / image_caption

    if not selected_image_path.exists():
        return f"Image **{image_caption}** does not exist in dataset **{name}** or has been deleted."

    # Here you can add more details about the image if needed
    summary_msg = f"# Selected Image\n- Path: {selected_image_path}"
    summary_msg += _get_image_metadata(selected_image_path)

    return summary_msg
