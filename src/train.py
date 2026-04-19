"""Training loop with WandB logging.
Provides a configurable training function that supports multiple
training strategies including MixUp augmentation, label smoothing,
cosine annealing LR scheduling and early stopping."""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, train_loader, val_loader, config, wandb_run=None):
    """Train a model with early stopping and checkpoint saving.

    Args:
        model (nn.Module): PyTorch model to train (from create_model()).
        train_loader (DataLoader): Training data loader with (image, label) batches.
        val_loader (DataLoader): Validation data loader with (image, label) batches.
        config (dict): Training configuration with keys:
            - 'lr' (float): Learning rate (1e-4).
            - 'epochs' (int): Maximum number of training epochs.
            - 'weight_decay' (float, optional): AdamW weight decay. Default 1e-4.
            - 'label_smoothing' (float, optional): Label smoothing factor. Default 0.0.
            - 'patience' (int, optional): Early stopping patience. Default 5.
            - 'mixup' (bool, optional): Whether to apply MixUp augmentation. Default False.
            - 'save_path' (str, optional): Path to save best checkpoint. Default 'best_model.pth'.
        wandb_run (wandb.Run, optional): Active W&B run for logging metrics. Default None.

    Returns:
        tuple: (model, best_val_acc)
            - model (nn.Module): Trained model (last epoch state, not best).
            - best_val_acc (float): Highest validation accuracy achieved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function with label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    
    # AdamW optimizer, handles weight decay correctly unlike Adam
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    # Cosine annealing gradually reduces LR from initial value to near zero over epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0 #tracking best validation accuracy for checkpointing
    patience_counter = 0 #counting epochs when no improvement
    patience = config.get("patience", 5) #max n of epochs without improvement before stopping
    use_mixup = config.get("mixup", False) #use mixup (blend image pairs during training) or not

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_mixup:
                # MixUp - blend two images and their labels with ratio lam
                lam = np.random.beta(0.4, 0.4) #sample mixing ratio from Beta distribution
                index = torch.randperm(images.size(0)).to(device) #random pairing
                mixed_images = lam * images + (1 - lam) * images[index]
                outputs = model(mixed_images)
                # Weighted loss from both original and paired labels
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(
                    outputs, labels[index]
                )
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        scheduler.step() #updating learning rate according to cosine annealing schedule
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Logging metrics to W&B if run is active
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / train_total,
                    "train_acc": train_acc,
                    "val_loss": val_loss / val_total,
                    "val_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        print(
            f"Epoch {epoch+1}/{config['epochs']} - Train: {train_acc:.4f} - Val: {val_acc:.4f}"
        )

        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.get("save_path", "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val acc: {best_val_acc:.4f}")
    return model, best_val_acc
