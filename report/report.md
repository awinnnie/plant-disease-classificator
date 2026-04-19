# Plant Disease Classification

**Anna Khurshudyan**

## 1. Abstract

This report presents a plant disease classification system that identifies diseases from plant and plant leaf images regardless of the species. The system classifies images into 39 disease categories derived from 82 plant-disease folder combinations in the PlantDoc dataset (~8k images). 
Multiple CNN architectures were evaluated, including ResNet-18, EfficientNet-B0 and ConvNeXt-Tiny, with various training strategies. The best model, ConvNeXt-Tiny with label smoothing and strong augmentation, achieves 82.1% accuracy and 90.3% mAP on the validation set. However, EfficientNet-B0 with MixUp achieves comparable performance (86.2% mAP) with 5x fewer parameters, offering a better efficiency-accuracy tradeoff for deployment.

## 2. Dataset

PlantDoc dataset containing 7783 training images and 390 validation images organized into 82 folders named by plant species and disease. Disease-only labels were extracted automatically by detecting the plant prefix: for each folder name, the longest prefix shared with at least one other folder is found, treating it as the plant name and the remainder as the disease. This approach handled multi-word plant names (e.g. "bell pepper") without hardcoding. After merging, 39 unique disease classes were obtained.

The dataset is small and imbalanced. Class sizes range from 6 images (coffee black rot) to 330 (citrus canker) in training, and from 1 to 10 in validation. This imbalance motivated use of strong data augmentation and regularization techniques. The small validation set (10 images per class maximum) means accuracy can fluctuate by ~0.25% from a single image, making mAP a more reliable metric.

## 3. Methodology

### 3.1 General Approach

A transfer learning approach: take a CNN pretrained on ImageNet (1.4M images, 1000 classes) and fine-tune it on the plant disease dataset. The pretrained early layers already trained to detect universal visual features (edges, textures, color patterns), while the later layers and classifier head are adapted to recognize disease-specific patterns. All models are trained on a single NVIDIA T4 GPU via Google Colab's free tier.

### 3.2 Data Augmentation

Given the small dataset size (~200 images per class on average), augmentation is essential to prevent overfitting. The baseline uses only horizontal flips and resize to 224x224. The enhanced augmentation pipeline includes resizing to 256x256 followed by random crop to 224x224 (spatial variation), random horizontal and vertical flips, RandAugment (num_ops=2, magnitude=9) which applies random combinations of photometric and geometric transforms and ImageNet normalization. For the EfficientNet-B0 MixUp experiment, I additionally used ColorJitter (brightness, contrast, saturation perturbations of 0.2).

### 3.3 Training Configuration

- **Regularization:** Dropout was added to EfficientNet's classifier head (0.3 in Experiment 3, increased to 0.4 in Experiment 4). Weight decay ranged from 1e-4 to 1e-3.
- **Early stopping:** Added from Experiment 3 onward, with patience of 5 epochs (7 for Experiment 4). The best checkpoint by validation accuracy is saved during training.
- **Tracking:** All experiments logged to Weights & Biases (loss, accuracy, learning rate per epoch).


## 4. Experiments

### 4.1 Experiment 1: ResNet-18 Baseline

Architecture: ResNet-18 pretrained on ImageNet (~11M parameters). The final fully connected layer is replaced with a linear layer mapping to 39 classes. Training uses Adam optimizer with lr=1e-4, batch size 32, basic augmentation (resize to 224x224 + horizontal flip), and CrossEntropyLoss. No scheduler, no weight decay, no early stopping. Trained for 15 epochs. Result: 71.0% accuracy, 79.3% mAP. The model reaches 98% train accuracy by epoch 10 while validation plateaus at 71%, indicating severe overfitting due to the small dataset.

### 4.2 Experiment 2: ResNet-18 with strong augmentation

Same ResNet-18 architecture, but with enhanced augmentation (resize to 256, random crop to 224, horizontal and vertical flips, RandAugment num_ops=2 magnitude=9). Added weight decay of 1e-4 and cosine annealing scheduler (T_max=20). Still uses Adam optimizer and standard CrossEntropyLoss. Trained for 20 epochs. Result: 75.9% accuracy, 84.0% mAP. The augmentation reduced the overfitting gap significantly (train accuracy 94.6% vs. 98.8% baseline) and improved validation accuracy by 4.9 percentage points, confirming that data augmentation is the most impactful single intervention for this dataset size.

### 4.3 Experiment 3: EfficientNet-B0

Architecture: EfficientNet-B0 pretrained on ImageNet (~5.3M parameters). EfficientNet uses compound scaling to balance network depth, width, and resolution, achieved better accuracy per parameter than ResNet. The classifier head is replaced with Dropout(0.3) followed by a linear layer. Used the same enhanced augmentation as in Experiment 2, Adam optimizer (lr=1e-4, weight_decay=1e-4), cosine annealing (T_max=25), and standard CrossEntropyLoss without label smoothing. Early stopping was added with patience=5, stopped at epoch 16. Result: 77.7% accuracy, 84.7% mAP. 

### 4.4 Experiment 4: EfficientNet-B0 with MixUp

Same EfficientNet-B0 architecture but with several additions. MixUp augmentation (beta distribution alpha=0.4) which blends pairs of training images and their labels each batch, ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2), increased dropout to 0.4, label smoothing of 0.1, AdamW optimizer with stronger weight decay of 1e-3 and cosine annealing with T_max=30. Patience was increased to 7 epochs, even though it did not early stop. Trained for 30 epochs. Result: 79.2% accuracy, 86.2% mAP. MixUp improved mAP by 1.5 points over the base EfficientNet. Train accuracy looks very low (~50-70%) because MixUp trains on blended images while validation uses clean images.

### 4.5 Experiment 5: ConvNeXt-Tiny

Architecture: ConvNeXt-Tiny pretrained on ImageNet (~28M parameters). ConvNeXt modernizes the standard CNN design by using ideas from Vision Transformers- depthwise convolutions, LayerNorm instead of BatchNorm, GELU activations and a 4-stage architecture with channel sizes 96-192-384-768. The final classifier layer is directly replaced with a linear layer to 39 classes, no added dropout as ConvNeXt has built-in regularization. Training uses AdamW (lr=1e-4, weight_decay=1e-4), label smoothing of 0.1, cosine annealing (T_max=25), enhanced augmentation (same as Experiment 2, no ColorJitter or MixUp) and early stopping with patience=5. Trained for 25 epochs, early stopped at epoch 19. Result: 82.1% accuracy, 90.3% mAP. This is the best model and the one deployed in production.

### 4.6 Failed Experiments

I tried two approaches that did not improve results. First, freezing all layers except the last two blocks of EfficientNet-B0 with label smoothing. It was too restrictive, reaching only ~48% validation accuracy before being stopped early. The frozen layers could not adapt enough to the plant disease domain. Second, a two-stage training strategy (5 epochs with frozen backbone at lr=1e-3, then full fine-tuning at lr=1e-5): the second stage learning rate was too conservative, reaching only 65% validation accuracy. These results confirm that full fine-tuning with appropriate regularization is superior to partial freezing for this dataset size.

### 4.7 Considered but Not Pursued

Also, a two-stage pipeline was considered: first classify the plant species, then predict the disease conditioned on the plant. Did not implement for three reasons: 1. The task requires disease identification regardless of host plant, so the model should generalize across species. 2. A pipeline introduces cascading errors where plant misclassification guarantees wrong disease prediction. 3. Splitting the already small dataset across two models would reduce per-model training data. Ensemble methods were also considered but rejected as contradicting the efficiency requirements and complicating deployment.

## 5. Results

| Model | Parameters | Accuracy | mAP | Key Changes |
|---|---|---|---|---|
| ResNet-18 Baseline | 11M | 71.0% | 79.3% | Basic augmentation |
| ResNet-18 Augmented | 11M | 75.9% | 84.0% | + RandAugment, weight decay |
| EfficientNet-B0 | 5.3M | 77.7% | 84.7% | + Better backbone, dropout |
| EfficientNet-B0 MixUp | 5.3M | 79.2% | 86.2% | + MixUp, ColorJitter |
| **ConvNeXt-Tiny** | **28M** | **82.1%** | **90.3%** | **+ Larger backbone, label smoothing** |

The progression shows that data augmentation provided the largest single improvement (+4.7% mAP from baseline), followed by backbone architecture (+0.7% from ResNet to EfficientNet, +6.6% from EfficientNet to ConvNeXt) and training techniques like MixUp and label smoothing (+1.5% for MixUp).

### 5.1 Per-Class Analysis

The ConvNeXt model achieved perfect AP (1.0) on 8 classes: alternaria leaf spot, angular leaf spot, black leaf streak, blossom end rot, bunchy top, leaf curl, smut, stem rust, stripe rust and tar spot. The weakest classes are septoria leaf spot (AP=0.53), bacterial leaf spot (AP=0.52) and downy mildew (AP=0.76). Downy mildew's low AP is expected as it appears across 7 different plants in the training set, all of them with visually very different, making it one of the hardest classes to unify. Bacterial leaf spot and septoria leaf spot are visually similar diseases (both present as small dark spots on leaves), which explains their confusion.

### 5.2 Model Selection

ConvNeXt-Tiny was deployed as the production model despite its larger size (28M vs 5.3M parameters) because the 4.1% mAP advantage directly impacts the primary evaluation criterion (50% weight). However, EfficientNet-B0 with MixUp achieves 86.2% mAP with 5.3x fewer parameters, making it the superior choice for resource-constrained environments. The model file size is 106MB for ConvNeXt vs 21MB for EfficientNet, and inference time is comparable on CPU hardware. For true on-premise deployment on edge devices, EfficientNet-B0 would be the recommended model.

## 6. Deployment

The model is served via a FastAPI endpoint with a single POST /predict route that accepts an image file and returns the top-5 predicted diseases with confidence scores. The API is containerized with Docker and deployed on HuggingFace Spaces, providing automatic Swagger/OpenAPI documentation at the /docs endpoint. The model loads on startup and runs inference on CPU, requiring no GPU for serving. Model weights are hosted on HuggingFace alongside the application code.

## 8. Links

- **GitHub Repository:** https://github.com/awinnnie/plant-disease-classificator
- **W&B Experiments:** https://wandb.ai/awinnnie_/plant-disease-classification
- **Live API:** https://awinnie-plant-disease-classification.hf.space/docs
- **Model Weights:** https://huggingface.co/spaces/awinnie/plant-disease-classification