# Server Inference Traditional

This project provides a **server-based inference pipeline** for both **vision models** (MobileNetV2, ResNet-20 trained on CIFAR-10) and an **NLP model** (DistilBERT fine-tuned on AG News). It includes utilities for model loading, caching, dataset generation, ONNX model conversion, and REST API endpoints for benchmarking.

---

## 🚀 Features

- **Server Inference** using FastAPI + Uvicorn
- **Vision Models**:
  - MobileNetV2 (CIFAR-10)
  - ResNet-20 (CIFAR-10)
- **NLP Model**:
  - DistilBERT fine-tuned on AG News classification
- **Flexible Device Support**: CPU, CUDA, MPS (Apple Silicon)
- **Model Caching** for efficient reuse
- **ONNX Conversion** for optimized deployment in browser
- **Dataset Utilities** for CIFAR-10

---

## ⚙️ Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv inference_env
   source ./venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Generation

Generate CIFAR-10 dataset samples for testing:

```bash
python generate-data/cifar10.py
```

This script prepares the dataset for inference with vision models and populates data under [data](./data/) folder.

## 🧩 Model Generation & Conversion

Convert PyTorch models to ONNX format for deployment:

```bash
python model-conversion/model_gen.py
```

This step ensures models can be exported and run efficiently in different environments. Models will be generate inside [models](./models/) folder.

## 🌐 Running the Server

```bash
python -m uvicorn server.app:app --reload
```

The server exposes endpoints for:

- Vision inference (MobileNetV2 / ResNet-20)

- NLP inference (DistilBERT AG News classification)

## 🧠 Model Explanation

### Vision Models

- MobileNetV2: Lightweight CNN optimized for mobile/edge devices, trained on CIFAR-10.

- ResNet-20: Residual network with skip connections, trained on CIFAR-10 for robust image classification.

### NLP Model

- DistilBERT (AG News): A distilled version of BERT fine-tuned for text classification on the AG News dataset (4 categories: World, Sports, Business, Sci/Tech).

## 📂 Dataset Explanation

### CIFAR-10

60,000 32×32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

| Class Number | Class Label |
| ------------ | ----------- |
| 0            | Airplane    |
| 1            | Automobile  |
| 2            | Bird        |
| 3            | Cat         |
| 4            | Deer        |
| 5            | Dog         |
| 6            | Frog        |
| 7            | Horse       |
| 8            | Ship        |
| 9            | Truck       |

### AG News Classes

News articles categorized into 4 classes (World, Sports, Business, Sci/Tech).

| Class Number | Class Label |
| ------------ | ----------- |
| 0            | World       |
| 1            | Sports      |
| 2            | Business    |
| 3            | Sci/Tech    |

## Benchmark Inference

Endpoint:

```
POST /api/benchmark
```

Parameters:

- `file`: (optional) Image file for vision models

- `model`: Model to run (`mobilenetv2` | `resnet20` | `distilbert`)

- `text`: (optional) Input text for NLP models

- `device`: Target device (`cpu` | `cuda` | `mps` | `auto`)

Example Request (Vision):

```bash
curl 'http://localhost:8000/api/benchmark' \

  -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryzuRB9vSBLpwTNE0p' \
  --data-raw $'------WebKitFormBoundaryzuRB9vSBLpwTNE0p\r\nContent-Disposition: form-data; name="file"; filename="9_1.jpg"\r\nContent-Type: image/jpeg\r\n\r\nÿØÿà\u0000\u0010JFIF\u0000\u0001\u0001\u0000\u0000\u0001\u0000\u0001\u0000\u0000ÿÛ\u0000C\u0000------WebKitFormBoundaryzuRB9vSBLpwTNE0p--\r\n'
```

Example Response (Vision):

```json
{
  "output": {...},
  "predictions": 9
}
```

Example Request (Nlp):

```bash
curl 'http://localhost:8000/api/benchmark' \
  -H 'Content-Type: multipart/form-data; boundary=----WebKitFormBoundaryNAvmLsDcBEeDPZIL' \
  --data-raw $'------WebKitFormBoundaryNAvmLsDcBEeDPZIL\r\nContent-Disposition: form-data; name="text"\r\n\r\nundefined Canadian Press - VANCOUVER (CP) - The sister of a man who died after a violent confrontation with police has demanded the city\'s chief constable resign for defending the officer involved.\r\n------WebKitFormBoundaryNAvmLsDcBEeDPZIL\r\nContent-Disposition: form-data; name="model"\r\n\r\ndistilbert\r\n------WebKitFormBoundaryNAvmLsDcBEeDPZIL\r\nContent-Disposition: form-data; name="device"\r\n\r\ncpu\r\n------WebKitFormBoundaryNAvmLsDcBEeDPZIL--\r\n'
```

Example Response (Nlp):

```json
{
  "predictions": 0,
  "probabilities": [
    [
      0.9998769760131836, 6.800624396419153e-5, 2.9212240406195633e-5,
      2.576130282250233e-5
    ]
  ]
}
```

## ✅ Cache Management

The `ModelLoader` class:

- Loads models on CPU/MPS/CUDA

- Caches models & tokenizers

- Provides cache info and clearing utilities
