import cv2
import numpy as np
import base64
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Union
import io
from PIL import Image

from detector import detectar_botella_y_nivel
from utils import image_to_base64

app = FastAPI(title="Liquor Level Analyzer API")

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Tu URL de frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageResponse(BaseModel):
    liquid_level: float
    processed_image: str  # Base64 encoded image
    debug_images: Dict[str, str] = {}  # Diccionario de imágenes de debug (nombre: Base64)
    
class ErrorResponse(BaseModel):
    detail: str
    
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error interno: {str(exc)}"}
    )

@app.post("/analyze/", response_model=Union[ImageResponse, ErrorResponse])
async def analyze_liquor_level(
    file: UploadFile = File(...),
    liquor_type: str = Form("whisky")
):
    try:
        # Leer la imagen desde el archivo subido
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Archivo de imagen vacío")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        # Verificar dimensiones y tipo de imagen
        print(f"Imagen recibida: {img.shape}, tipo: {img.dtype}")
        
        # Dictionary para guardar las imágenes de debug
        debug_images = {}
        
        # Asegurarse de que la imagen tenga el formato adecuado
        if len(img.shape) != 3 or img.shape[2] != 3:
            print("Convirtiendo imagen a BGR...")
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar imagen si es muy grande
        max_size = 1280  # Tamaño máximo para evitar problemas de memoria
        if img.shape[0] > max_size or img.shape[1] > max_size:
            scale = max_size / max(img.shape[0], img.shape[1])
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        
        # Validar tipo de licor
        valid_liquor_types = ["whisky", "vodka", "rum", "tequila"]
        if liquor_type not in valid_liquor_types:
            liquor_type = "whisky"  # Valor predeterminado
        
        # Procesar imagen usando la función de detector.py
        resultado, roi, mask_viz, nivel_detectado = detectar_botella_y_nivel(img, liquor_type=liquor_type)
        
        # Guardar imágenes de debug
        debug_images["grayscale"] = image_to_base64(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
        if roi is not None and roi.size > 10:
            debug_images["roi"] = image_to_base64(roi)
        
        if mask_viz is not None and mask_viz.size > 10:
            debug_images["mask"] = image_to_base64(mask_viz)
        
        # Convertir la imagen procesada a base64
        if resultado is not None:
            processed_image_base64 = image_to_base64(resultado)
        else:
            processed_image_base64 = ""
            nivel_detectado = 0
        
        return ImageResponse(
            liquid_level=round(nivel_detectado, 1),
            processed_image=processed_image_base64,
            debug_images=debug_images
        )
    except HTTPException as e:
        # Re-lanzar excepciones HTTP
        raise
    except Exception as e:
        # Registrar el error y devolver una respuesta amigable
        print(f"Error en analyze_liquor_level: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API de análisis de nivel de líquido en botellas"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)