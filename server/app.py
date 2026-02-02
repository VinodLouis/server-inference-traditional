import time
import psutil
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from scipy.special import softmax
from pathlib import Path
import sys
import logging
import io
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.models import ModelLoader

from server.transformer import Transfrormer

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


app = FastAPI(title="ML Inference Benchmark API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Server-Timing"]
)

model_loader = ModelLoader()
#metrics_calculator = MetricsCalculator()
data_generator = Transfrormer()
process = psutil.Process()
start_time = time.time()

@app.get("/api/health")
@app.head("/api/health") 
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "timestamp": time.time()
    }



@app.post("/api/benchmark")
async def benchmark(
    file: Optional[UploadFile] = File(None),
    model: str = Form("mobilenetv2"),
    text: Optional[str] = Form(None),      
    device: str = Form("auto")
):
    """
    Runs inference benchmark on specified model and input
    
    :param file: Input file for vison models analysis
    :type file: binary
    :param model: Description
    :type model: distilbert | mobilenetv2 | resnet20
    :param text: Input text description for NLP models analysis
    :type text: string
    :param device: Device to run inference on 
    :type device: mps | cuda | cpu | auto
    """
    try:
        target_device = model_loader.get_device(device) 

        if model in ["mobilenetv2", "resnet20"]:
            # Load model 
            if model == "mobilenetv2": 
                model = model_loader.load_mobilenetv2(device=device) 
            else: 
                model = model_loader.load_resnet20(device=device) 

            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Preprocess
            transform = data_generator.cifar_transform
            input_tensor = transform(image).unsqueeze(0).to(target_device)

            # Run inference
            with torch.no_grad():
                output = model(input_tensor)

            probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
            predicted_class = int(np.argmax(probs))

            return {
                "output": output,
                "predictions": predicted_class,
            }

        elif model == "distilbert":
            # Load model + tokenizer
            model, tokenizer = model_loader.load_distilbert(device=device)
            model.to(target_device)
            model.eval()

            # Encode single text
            enc = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(target_device) for k, v in enc.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits.cpu().numpy()
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            predicted_class = int(np.argmax(logits))

            return {
                "predictions": predicted_class,
                "probabilities": probs.tolist(),
            }
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model}"})
    except Exception as e: 
        return JSONResponse(status_code=500, content={"error": str(e)})

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
