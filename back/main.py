import utils
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Permitir CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    model: str
    image_path: str
    points: dict
    realtime: bool = False


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # Aquí podrías procesar la imagen y los puntos
    print(f"Recibido: {req}")

    model = req.model
    image_path = req.image_path
    positive_points = req.points.get("positive", [])
    negative_points = req.points.get("negative", [])

    response = utils.detect(model, image_path, positive_points, negative_points)

    return response
