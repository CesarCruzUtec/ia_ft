import base64
import gc
import io
import os

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

model_to_cfg = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1_hiera_l.yaml",
}


def print_dict(d):
    # Iterate over all elements in dictionary recursively and limit the length of strings to 50 characters
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{key}:")
            print_dict(value)
        elif isinstance(value, str):
            if len(value) > 50:
                print(f"{key}: {value[:50]}... (length: {len(value)})")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")


class ImageAnalyzer:
    def __init__(self):
        # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        self.model_dir = "../meta-sam2/checkpoints"
        self.config_dir = "configs/sam2.1"

        self.image_name = None
        self.image = None

        self.model_name = None
        self.model = None
        self.predictor = None

        self.yolo_model_name = None
        self.yolo_model = None

    def get_points(self, model, image_name):
        # Load YOLO model
        self.load_model_yolo(model)

        # Load image
        self.load_image(image_name)

        # Obtain coordinates (Currently only one YOLO custom model is used for detection)
        results = self.yolo_model(self.image)

        detection_list = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]

                detection_list.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

        return {"detections": detection_list}

    def analyze_image(self, model, image_name, boxes):
        if len(boxes) == 0:
            print("No se proporcionaron cajas.")
            raise ValueError("Se requieren al menos una caja.")

        # Load model if different from current
        self.load_model(model)

        # Search image and save it as numpy array
        self.load_image(image_name)

        # Process image
        if self.predictor is None:
            raise ValueError("El predictor no está inicializado. Carga un modelo primero.")

        print("Procesando la imagen...")
        self.predictor.set_image(self.image)
        print("Imagen procesada.")

        # Prepare input boxes
        print("Preparando cajas...")
        input_boxes = []
        for box in boxes:
            input_boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
        input_boxes = np.array(input_boxes)

        # Predict masks
        print("Prediciendo máscaras...")
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            print(f"Mask {i + 1}: score {score}")
            mask_b64 = self.mask_to_base64(mask)
            boxes[i]["mask_base64"] = mask_b64
            boxes[i]["score"] = float(score)

        return {"masks": boxes}

    def load_model(self, model_name):
        if self.model_name == model_name:
            print(f"Modelo {model_name} ya cargado.")
            return

        print("Liberando memoria del modelo anterior...")
        del self.model
        del self.predictor
        gc.collect()
        torch.cuda.empty_cache()

        self.model_name = model_name
        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        print(f"Cargando modelo: {model_path}")
        config_path = os.path.join(self.config_dir, model_to_cfg[model_name])

        sam_model = build_sam2(config_path, model_path, device=self.device)
        self.model = sam_model

        predictor = SAM2ImagePredictor(sam_model)
        self.predictor = predictor
        print("Modelo cargado y predictor creado.")

    def load_model_yolo(self, model_name):
        if self.yolo_model_name == model_name:
            print(f"Modelo YOLO {model_name} ya cargado.")
            return

        print("Liberando memoria del modelo YOLO anterior...")
        del self.yolo_model
        gc.collect()
        torch.cuda.empty_cache()

        self.yolo_model_name = model_name
        model_path = os.path.join("../models", f"{model_name}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {model_path}")

        print(f"Cargando modelo YOLO: {model_path}")
        yolo_model = YOLO(model_path)
        self.yolo_model = yolo_model
        print("Modelo YOLO cargado.")

    def load_image(self, image_name):
        if self.image_name == image_name and self.image is not None:
            print(f"Imagen {image_name} ya cargada.")
            return

        image_full_path = self.search_image(image_name)
        if image_full_path is None:
            print(f"Imagen no encontrada: {image_name}")
            raise FileNotFoundError(f"Imagen no encontrada: {image_name}")
        print(f"Usando imagen: {image_full_path}")

        image = Image.open(image_full_path)
        image = np.array(image.convert("RGB"))
        print(f"Imagen cargada con shape: {image.shape}")

        self.image_name = image_name
        self.image = image

    def search_image(self, image_name):
        image_dir = "../images"
        for root, dirs, files in os.walk(image_dir):
            for filename in files:
                if filename == image_name:
                    return os.path.join(root, filename)
        return None

    def mask_to_base64(self, mask: np.ndarray) -> str:
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
