from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils import ImageAnalyzer, print_dict

app = FastAPI()

image_analyzer = ImageAnalyzer()

# Permitir CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GetBoxesRequest(BaseModel):
    model: str
    image_name: str


@app.post("/get_boxes")
async def get_boxes(req: GetBoxesRequest):
    # Aquí podrías procesar la imagen y los puntos
    print_dict(req.model_dump())
    response = image_analyzer.get_boxes(
        model=req.model,
        image_name=req.image_name,
    )

    return response


class AnalyzeRequest(BaseModel):
    model: str
    image_name: str
    boxes: list = []  # Lista de cajas


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    print_dict(req.model_dump())
    response = image_analyzer.analyze_image(
        model=req.model,
        image_name=req.image_name,
        boxes=req.boxes,
    )

    return response


class MeasureRequest(BaseModel):
    image_name: str
    boxes: list = []  # Lista de cajas con mascaras


@app.post("/measure")
async def measure(req: MeasureRequest):
    print_dict(req.model_dump())
    response = image_analyzer.measure(
        image_name=req.image_name,
        boxes=req.boxes,
    )

    return response
