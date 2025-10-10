import base64
import io
import os

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# =============================

MODEL_DIR = "/mnt/data/personal/cencosud/ia_ft/meta-sam2/checkpoints"
CONFIG_DIR = "configs/sam2.1"

model_to_cfg = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
}


def search_image(image_name):
    IMAGE_DIR = "/mnt/data/personal/cencosud/ia_ft/images"
    for root, dirs, files in os.walk(IMAGE_DIR):
        for filename in files:
            if filename == image_name:
                return os.path.join(root, filename)
    return None


def mask_to_base64(mask: np.ndarray) -> str:
    # mask: 2D numpy array, 0/1 or bool
    mask_img = (mask * 255).astype(np.uint8)
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[..., 0:3] = 30, 144, 255  # blue color (BGR)
    rgba[..., 3] = (mask_img * 0.6).astype(np.uint8)  # alpha 0.6*255
    img = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def detect(model_name, image_path, positive_points, negative_points):
    image_full_path = search_image(image_path)
    if image_full_path is None:
        print(f"Imagen no encontrada: {image_path}")
        return {"error": "Imagen no encontrada"}
    print(f"Usando imagen: {image_full_path}")

    image = Image.open(image_full_path)
    image = np.array(image.convert("RGB"))
    print(f"Imagen cargada con shape: {image.shape}")

    sam2_checkpoint = os.path.join(MODEL_DIR, f"{model_name}.pt")
    if not os.path.exists(sam2_checkpoint):
        print(f"Modelo no encontrado: {sam2_checkpoint}")
        return {"error": "Modelo no encontrado"}
    print(f"Usando modelo: {sam2_checkpoint}")

    sam2_config = os.path.join(CONFIG_DIR, model_to_cfg[model_name])
    # if not os.path.exists(sam2_config):
    #     print(f"Configuración no encontrada: {sam2_config}")
    #     return {"error": "Configuración no encontrada"}
    # print(f"Usando configuración: {sam2_config}")

    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    print("Modelo SAM 2 cargado.")
    predictor = SAM2ImagePredictor(sam2_model)
    print("Predictor SAM 2 creado.")

    print("Procesando la imagen...")
    predictor.set_image(image)
    print("Imagen procesada.")

    input_array_point = []
    input_label_point = []

    print("Procesando puntos...")

    if len(positive_points) == 0:
        print("No se proporcionaron puntos positivos.")
        return {"error": "Se requieren al menos un punto positivo."}

    for point in positive_points:
        input_array_point.append([point["x"], point["y"]])
        input_label_point.append(1)

    for point in negative_points:
        input_array_point.append([point["x"], point["y"]])
        input_label_point.append(0)

    input_array_point = np.array(input_array_point)
    input_label_point = np.array(input_label_point)

    print("Prediciendo máscaras...")
    masks, scores, logits = predictor.predict(
        point_coords=input_array_point,
        point_labels=input_label_point,
        multimask_output=True,
    )

    print("Ordenando máscaras por score...")
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    MASK_COLOR = (30 / 255, 144 / 255, 255 / 255, 0.6)

    response = {}
    print("Preparando respuesta...")
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_b64 = mask_to_base64(mask)
        response[f"mask_{i + 1}"] = {
            "score": float(score),
            "mask_base64": mask_b64,
        }

    return response
