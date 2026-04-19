# Plant Disease Classification: A Comparative Study of CNN Architectures

**Anna Khurshudyan**

## 1. Abstract

This report presents a plant disease classification system that identifies diseases from leaf images regardless of the host plant species. The system classifies images into 39 disease categories derived from 82 plant-disease folder combinations in the PlantDoc dataset (~8k images). We evaluate multiple CNN architectures including ResNet-18, EfficientNet-B0, and ConvNeXt-Tiny, with various training strategies. Our best model, ConvNeXt-Tiny with label smoothing and strong augmentation, achieves 82.1% accuracy and 90.3% mAP on the validation set. We also demonstrate that EfficientNet-B0 with MixUp achieves comparable performance (86.2% mAP) with 5x fewer parameters, offering a better efficiency-accuracy tradeoff for deployment.

## 2. Introduction

Plant disease identification is critical for agriculture, yet manual diagnosis requires expert knowledge and is not scalable. The goal of this project is to build a classifier that, given a leaf image, predicts the disease name independent of the plant species. For example, both "tomato late blight" and "potato late blight" should map to the same output class: "late blight." This is a harder task than standard plant disease classification where plant+disease pairs are treated as separate classes, because the model must learn disease-invariant features that generalize across visually different plant species.

## 3. Dataset

We use the PlantDoc dataset containing 7,783 training images and 390 validation images organized into 82 folders named by plant species and disease. We extract disease-only labels by automatically detecting the plant prefix: for each folder name, we find the longest prefix shared with at least one other folder, treating it as the plant name and the remainder as the disease. This approach handles multi-word plant names (e.g., "bell pepper") without hardcoding. After merging, we obtain 39 unique disease classes.

The dataset is small and imbalanced: class sizes range from 6 images (coffee black rot) to 330 (citrus canker) in training, and from 1 to 10 in validation. This imbalance motivated our use of strong data augmentation and regularization techniques. The small validation set (10 images per class maximum) means accuracy can fluctuate by ~0.25% from a single image, making mAP a more reliable metric.

## 4. Methodology

### 4.1 General Approach

We adopt a transfer learning approach: take a CNN pretrained on ImageNet (1.4M images, 1000 classes) and fine-tune it on our plant disease dataset. The pretrained early layers already detect universal visual features (edges, textures, color patterns), while the later layers and classifier head are adapted to recognize disease-specific patterns such as spots, discoloration, and powdery coatings. All models are trained on a single NVIDIA T4 GPU via Google Colab's free tier.

### 4.2 Data Augmentation

Given the small dataset size (~200 images per class on average), augmentation is essential to prevent overfitting. Our baseline uses only horizontal flips and resize to 224x224. The enhanced augmentation pipeline includes: resize to 256x256 followed by random crop to 224x224 (introducing spatial variation), random horizontal and vertical flips, RandAugment (num_ops=2, magnitude=9) which applies random combinations of photometric and geometric transforms, and ImageNet normalization. For the EfficientNet-B0 MixUp experiment, we additionally use ColorJitter (brightness, contrast, saturation perturbations of 0.2).

### 4.3 Training Configuration

All experiments share a common training framework: AdamW optimizer, cosine annealing learning rate schedule, and early stopping based on validation accuracy with a patience of 5-7 epochs. The best model checkpoint (by validation accuracy) is saved during training. All experiments are tracked with Weights & Biases for reproducibility and comparison.

## 5. Experiments and Ablations

### 5.1 Experiment 1: ResNet-18 Baseline

Architecture: ResNet-18 pretrained on ImageNet (~11M parameters). The final fully connected layer is replaced with a linear layer mapping to 39 classes. Training uses Adam optimizer with lr=1e-4, batch size 32, basic augmentation (resize + horizontal flip), and CrossEntropyLoss. Trained for 15 epochs. Result: 71.0% accuracy, 79.3% mAP. The model reaches 98% train accuracy by epoch 10 while validation plateaus at 71%, indicating severe overfitting due to the small dataset.

### 5.2 Experiment 2: ResNet-18 + Strong Augmentation

Same ResNet-18 architecture, but with enhanced augmentation (RandAugment, random crop, vertical flip), weight decay of 1e-4, and cosine annealing scheduler. Trained for 20 epochs. Result: 75.9% accuracy, 84.0% mAP. The augmentation reduced the overfitting gap significantly (train accuracy 94.6% vs. 98.8% baseline) and improved validation accuracy by 4.9 percentage points, confirming that data augmentation is the most impactful single intervention for this dataset size.

### 5.3 Experiment 3: EfficientNet-B0

Architecture: EfficientNet-B0 pretrained on ImageNet (~5.3M parameters). EfficientNet uses compound scaling to balance network depth, width, and resolution, achieving better accuracy per parameter than ResNet. The classifier head is replaced with Dropout(0.3) followed by a linear layer. Uses the same enhanced augmentation pipeline, AdamW optimizer (lr=1e-4, weight_decay=1e-4), and cosine annealing. Early stopping with patience=5. Result: 77.7% accuracy, 84.7% mAP. Despite having half the parameters of ResNet-18, EfficientNet-B0 outperforms it by 6.7% in accuracy, validating the architecture's efficiency.

### 5.4 Experiment 4: EfficientNet-B0 + MixUp

Same EfficientNet-B0 architecture with additional techniques: MixUp augmentation (alpha=0.4) which blends pairs of training images and their labels during training, forcing the model to learn more robust features; ColorJitter augmentation; increased dropout to 0.4; label smoothing of 0.1; and stronger weight decay of 1e-3. Patience increased to 7 epochs. Trained for 30 epochs. Result: 79.2% accuracy, 86.2% mAP. MixUp improved mAP by 1.5 points over the base EfficientNet. Note that train accuracy appears artificially low (~50-70%) because MixUp trains on blended images while validation uses clean images.

### 5.5 Experiment 5: ConvNeXt-Tiny (Best Model)

Architecture: ConvNeXt-Tiny pretrained on ImageNet (~28M parameters). ConvNeXt modernizes the standard CNN design by incorporating ideas from Vision Transformers: it uses depthwise convolutions, LayerNorm instead of BatchNorm, GELU activations, and a 4-stage architecture with channel sizes 96-192-384-768. The final classifier layer is replaced with a linear layer mapping to 39 classes. Training uses AdamW (lr=1e-4, weight_decay=1e-4), label smoothing of 0.1, cosine annealing, enhanced augmentation (RandAugment + crops + flips), and early stopping with patience=5. Trained for 25 epochs, early stopped at epoch 19. Result: 82.1% accuracy, 90.3% mAP. This is our best model and the one deployed in production.

### 5.6 Failed Experiments

We also attempted two approaches that did not improve results. First, freezing all layers except the last two blocks of EfficientNet-B0 with label smoothing: this was too restrictive, reaching only ~48% validation accuracy before being stopped early. The frozen layers could not adapt enough to the plant disease domain. Second, a two-stage training strategy (5 epochs with frozen backbone at lr=1e-3, then full fine-tuning at lr=1e-5): the second stage learning rate was too conservative, reaching only 65% validation accuracy. These negative results confirm that full fine-tuning with appropriate regularization is superior to partial freezing for this dataset size.

### 5.7 Considered but Not Pursued

We considered a two-stage pipeline: first classify the plant species, then predict the disease conditioned on the plant. This was rejected for three reasons: (1) the task requires disease identification regardless of host plant, so the model should generalize across species; (2) a pipeline introduces cascading errors where plant misclassification guarantees wrong disease prediction; (3) splitting the already small dataset across two models would reduce per-model training data. We also considered ensemble methods but rejected them as contradicting the efficiency requirements and complicating deployment.

## 6. Results

| Model | Parameters | Accuracy | mAP | Key Changes |
|---|---|---|---|---|
| ResNet-18 Baseline | 11M | 71.0% | 79.3% | Basic augmentation |
| ResNet-18 Augmented | 11M | 75.9% | 84.0% | + RandAugment, weight decay |
| EfficientNet-B0 | 5.3M | 77.7% | 84.7% | + Better backbone, dropout |
| EfficientNet-B0 MixUp | 5.3M | 79.2% | 86.2% | + MixUp, ColorJitter |
| **ConvNeXt-Tiny** | **28M** | **82.1%** | **90.3%** | **+ Larger backbone, label smoothing** |

Each experiment builds on the previous one, allowing us to isolate the contribution of each technique. The progression shows that data augmentation provided the largest single improvement (+4.7% mAP from baseline), followed by backbone architecture (+0.7% from ResNet to EfficientNet, +6.6% from EfficientNet to ConvNeXt), and training techniques like MixUp and label smoothing (+1.5% for MixUp).

### 6.1 Per-Class Analysis

The ConvNeXt model achieves perfect AP (1.0) on 8 classes: alternaria leaf spot, angular leaf spot, black leaf streak, blossom end rot, bunchy top, leaf curl, smut, stem rust, stripe rust, and tar spot. The weakest classes are septoria leaf spot (AP=0.53), bacterial leaf spot (AP=0.52), and downy mildew (AP=0.76). Downy mildew's low AP is expected: it appears across 7 different plants in the training set, each with visually different manifestations, making it one of the hardest classes to unify. Bacterial leaf spot and septoria leaf spot are visually similar diseases (both present as small dark spots on leaves), which explains their confusion.

### 6.2 Model Selection

We deploy ConvNeXt-Tiny as the production model despite its larger size (28M vs 5.3M parameters) because the 4.1% mAP advantage directly impacts the primary evaluation criterion (50% weight). However, we note that EfficientNet-B0 with MixUp achieves 86.2% mAP with 5.3x fewer parameters, making it the superior choice for resource-constrained environments. The model file size is 106MB for ConvNeXt vs 21MB for EfficientNet, and inference time is comparable on CPU hardware. For true on-premise deployment on edge devices, EfficientNet-B0 would be the recommended model.

## 7. Deployment

The model is served via a FastAPI endpoint with a single POST /predict route that accepts an image file and returns the top-5 predicted diseases with confidence scores. The API is containerized with Docker and deployed on HuggingFace Spaces, providing automatic Swagger/OpenAPI documentation at the /docs endpoint. The model loads on startup and runs inference on CPU, requiring no GPU for serving. Model weights are hosted on HuggingFace alongside the application code.

## 8. Conclusion

We developed a plant disease classifier achieving 90.3% mAP across 39 disease classes using ConvNeXt-Tiny with transfer learning. Key findings: (1) strong data augmentation is the most impactful technique for small datasets; (2) modern architectures like ConvNeXt and EfficientNet significantly outperform older ones like ResNet-18; (3) techniques like MixUp and label smoothing provide meaningful but smaller gains; (4) partial layer freezing hurts performance when the target domain differs substantially from ImageNet. The system is deployed as a production-ready FastAPI service accessible via HuggingFace Spaces.

## 9. Links

- **GitHub Repository:** https://github.com/awinnnie/plant-disease-classificator
- **W&B Experiments:** https://wandb.ai/awinnnie_/plant-disease-classification
- **Live API:** https://awinnie-plant-disease-classification.hf.space/docs
- **Model Weights:** https://huggingface.co/spaces/awinnie/plant-disease-classification