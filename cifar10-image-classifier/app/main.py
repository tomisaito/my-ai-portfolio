import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
import io
from typing import Dict

from app.model import CIFAR10_CLASSES, get_transform, load_model

ROOT_PATH = os.environ.get("ROOT_PATH", "")

app = FastAPI(
    title="CIFAR-10 Image Classification API",
    version="2.2.0",
    root_path=ROOT_PATH or ""
)

model, model_loaded = load_model()
transform = get_transform()

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    image_size: tuple
    status: str = "success"

@app.get("/")
def read_root():
    status_badge = "✅ ONLINE" if model_loaded else "⚠️ MODEL ERROR"
    status_color = "#4caf50" if model_loaded else "#ff9800"
    root_path = app.root_path or ""

    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIFAR-10 API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto;
                    padding: 20px; background: #f0f2f5; }}
            .container {{ background: white; padding: 40px; border-radius: 16px;
                          box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
            .badge {{ background: {status_color}; color: white; padding: 6px 12px;
                      border-radius: 12px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>CIFAR-10 Image Classification <span style="background:{status_color}; color:#fff; padding:4px 8px; border-radius:6px;">{status_badge}</span></h1>

        <form id="uploadForm">
            <input type="file" name="file" accept="image/*" required {"disabled" if not model_loaded else ""}>
            <button type="submit" {"disabled" if not model_loaded else ""}>Classify</button>
        </form>
        <pre id="result"></pre>

        <script>
            const ROOT_PATH = "{root_path}";
            document.getElementById('uploadForm').onsubmit = async (e) => {{
                e.preventDefault();
                const resultEl = document.getElementById('result');
                const data = new FormData(e.target);
                resultEl.textContent = "Processing...";
                try {{
                    const res = await fetch(ROOT_PATH + '/predict', {{ method: 'POST', body: data }});
                    const json = await res.json();
                    resultEl.textContent = JSON.stringify(json, null, 2);
                }} catch (err) {{
                    resultEl.textContent = "Error: " + err.message;
                }}
            }};
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": app.version
    }

@app.get("/classes")
def get_classes():
    return {"classes": CIFAR10_CLASSES, "total": len(CIFAR10_CLASSES)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.max(probabilities, 0)

    return {
        "predicted_class": CIFAR10_CLASSES[top_class.item()],
        "confidence": float(top_prob),
        "all_probabilities": { CIFAR10_CLASSES[i]: float(probabilities[i]) for i in range(len(CIFAR10_CLASSES)) },
        "image_size": original_size,
        "status": "success"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(exc), "status": "error"})