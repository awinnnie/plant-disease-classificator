import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import json
import io

app = FastAPI(title="Plant Disease Classifier", description="Identifies plant disease from leaf and plant images")

# Load model and label map
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('label_map.json') as f:
    label_map = json.load(f)
idx_to_disease = {int(v): k for k, v in label_map.items()}

model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(label_map))
model.load_state_dict(torch.load('convnext_tiny_best.pth', map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        top5 = torch.topk(probs, 5)

    results = []
    for i in range(5):
        results.append({
            "disease": idx_to_disease[top5.indices[0][i].item()],
            "confidence": round(top5.values[0][i].item() * 100, 2)
        })

    return {"predictions": results}