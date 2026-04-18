"""Training loop with WandB logging."""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(model, train_loader, val_loader, config, wandb_run=None):
    """Train model with early stopping and checkpoint saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0
    patience_counter = 0
    patience = config.get("patience", 5)
    use_mixup = config.get("mixup", False)

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if use_mixup:
                lam = np.random.beta(0.4, 0.4)
                index = torch.randperm(images.size(0)).to(device)
                mixed_images = lam * images + (1 - lam) * images[index]
                outputs = model(mixed_images)
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

        scheduler.step()
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

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
