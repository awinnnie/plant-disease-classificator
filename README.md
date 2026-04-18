# Plant Disease Classification

An AI system that identifies plant diseases from leaf images regardless of the host plant species. Given an image of a plant or a plant leaf, the model predicts the disease (e.g., "late blight" for both tomato and potato).

## Results

| Model | Params | Accuracy | mAP |
|---|---|---|---|
| ResNet-18 Baseline | 11M | 71.0% | 79.3% |
| ResNet-18 + Augmentation | 11M | 75.9% | 84.0% |
| EfficientNet-B0 + Augmentation | 5.3M | 77.7% | 84.7% |
| EfficientNet-B0 + MixUp | 5.3M | 79.2% | 86.2% |
| **ConvNeXt-Tiny** (deployed) | **28M** | **82.1%** | **90.3%** |

## Project Structure

├── src/
│   ├── dataset.py       # Dataset class, disease label extraction
│   ├── model.py         # Model creation and loading
│   ├── train.py         # Training loop with W&B logging
│   └── evaluate.py      # mAP, classification report, confusion matrix
├── api/
│   └── main.py          # FastAPI inference endpoint
├── report/
│   └── report.pdf       # Technical report
├── label_map.json       # Disease class mapping
├── requirements.txt
└── README.md

## Setup

```bash
git clone https://github.com/awinnnie/plant-disease-classificator.git
cd plant-disease-classificator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Training was done on Google Colab (T4 GPU). To reproduce:

1. Download the PlantDoc dataset from [Google Drive](https://drive.google.com/drive/folders/1gaVEvJyVG2sQNDgJ_0gSyGHx3s2S5DjQ)
2. Use `src/dataset.py` to build the dataset with disease-only labels
3. Use `src/model.py` to create the model
4. Use `src/train.py` to train with your desired config:

```python
from src.model import create_model
from src.train import train_model

model = create_model(num_classes=39, backbone='convnext_tiny')
train_model(model, train_loader, val_loader, config={
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'epochs': 25,
    'patience': 5,
})
```

## API

Run locally:

```bash
uvicorn api.main:app --reload
```

Swagger docs available at `http://localhost:8000/docs`

**Live API:** [https://awinnie-plant-disease-classification.hf.space/docs](https://awinnie-plant-disease-classification.hf.space/docs)

## Links

- **W&B Experiments:** [https://wandb.ai/awinnnie_/plant-disease-classification](https://wandb.ai/awinnnie_/plant-disease-classification)
- **Model Weights:** Hosted on [HuggingFace Spaces](https://huggingface.co/spaces/awinnie/plant-disease-classification)
- **Live API:** [https://awinnie-plant-disease-classification.hf.space/docs](https://awinnie-plant-disease-classification.hf.space/docs)