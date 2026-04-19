"""FastAPI endpoint for plant disease prediction.

Serves a ConvNeXt-Tiny model trained on the PlantDoc dataset.
Accepts plant/leaf image uploads and returns top-5 disease predictions
with confidence scores.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import json
import io

app = FastAPI(
    title="Plant Disease Classifier",
    description="Identifies plant disease from leaf and plant images",
)

# gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and label map
with open("label_map.json") as f:
    label_map = json.load(f)
# Reverse mapping: index to disease name for readable output
idx_to_disease = {int(v): k for k, v in label_map.items()}

# Load ConvNeXt-Tiny architecture without pretrained weights,
# then load our trained checkpoint
model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(label_map))
model.load_state_dict(torch.load("convnext_tiny_best.pth", map_location=device))
model = model.to(device)
model.eval() #set to inference mode (disables dropout, batchnorm updates)

# Same preprocessing as validation. resize, convert to tensor, normalize with ImageNet stats
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predicts plant disease from an uploaded image.

    Args:
        file (UploadFile): Image file (jpeg,png) of a plant or leaf.

    Returns:
        dict: {"predictions": list of top-5 predictions}, where each prediction
              contains "disease" (str) and "confidence" (float, percentage).
    """
    # Read uploaded image bytes and convert to RGB PIL Image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Preprocessing. resize, normalize, add batch dimension
    x = transform(img).unsqueeze(0).to(device)

    # Inference without computing gradients
    with torch.no_grad():
        out = model(x) #raw logits, (1, 39)
        probs = torch.softmax(out, dim=1) #convert to probabilities
        top5 = torch.topk(probs, 5)

    # Results as list of {disease, confidence} dicts
    results = []
    for i in range(5):
        results.append(
            {
                "disease": idx_to_disease[top5.indices[0][i].item()],
                "confidence": round(top5.values[0][i].item() * 100, 2),
            }
        )

    return {"predictions": results}
