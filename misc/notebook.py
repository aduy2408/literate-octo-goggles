# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kaggle==2.2.1",
#     "kagglehub==1.0.1",
#     "tqdm==4.68.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    # import os
    # from kaggle.api.kaggle_api_extended import KaggleApi


    # os.environ["KAGGLE_USERNAME"] = "bacahlam"
    # os.environ["KAGGLE_KEY"] = "6c1d3ecfec0e9b7fbbfd65d575f3b320"

    # api = KaggleApi()
    # api.authenticate()

    # api.dataset_download_files(
    #     "ctvmnn/ham10000",
    #     path="/marimo",
    #     unzip=True
    # )
    return


@app.cell
def _():
    print("hello")
    return


@app.cell
def _():
    from pathlib import Path

    import pandas as pd
    from PIL import Image

    import torch
    from torch.utils.data import Dataset


    class HAM10000Dataset(Dataset):
        def __init__(
            self,
            csv_file,
            image_dir,
            transform=None
        ):
            self.df = pd.read_csv(csv_file)

            self.image_dir = Path(image_dir)
            self.transform = transform

            self.classes = sorted(
                self.df["dx"].unique()
            )

            self.class_to_idx = {
                cls: idx
                for idx, cls in enumerate(self.classes)
            }

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            image_path = (
                self.image_dir /
                f"{row['image_id']}.jpg"
            )

            image = Image.open(image_path).convert("RGB")

            label = self.class_to_idx[row["dx"]]

            if self.transform:
                image = self.transform(image)

            return image, label

    return Dataset, Image, Path, pd, torch


@app.cell
def _():
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_tf, val_tf


@app.cell
def _(pd):
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np


    df = pd.read_csv(
        "HAM10000_metadata.csv"
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["dx"],
        random_state=42
    )
    return compute_class_weight, np, train_df, val_df


@app.cell
def _(Dataset, Image, Path):
    class HAMDataset(Dataset):
        def __init__(
            self,
            dataframe,
            image_dir,
            transform=None
        ):
            self.df = dataframe.reset_index(drop=True)
            self.image_dir = Path(image_dir)
            self.transform = transform

            self.classes = sorted(
                self.df["dx"].unique()
            )

            self.class_to_idx = {
                c: i
                for i, c in enumerate(self.classes)
            }

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            img = Image.open(
                self.image_dir /
                f"{row.image_id}.jpg"
            ).convert("RGB")

            label = self.class_to_idx[row.dx]

            if self.transform:
                img = self.transform(img)

            return img, label

    return (HAMDataset,)


@app.cell
def _(HAMDataset, train_df, train_tf, val_df, val_tf):
    train_dataset = HAMDataset(
        train_df,
        image_dir="HAM10000_images/HAM10000_images",
        transform=train_tf
    )

    val_dataset = HAMDataset(
        val_df,
        image_dir="HAM10000_images/HAM10000_images",
        transform=val_tf
    )
    return train_dataset, val_dataset


@app.cell
def _(train_dataset, val_dataset):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, val_loader


@app.cell
def _(train_dataset):
    print(train_dataset.class_to_idx)
    return


@app.cell
def _(compute_class_weight, np, torch, train_df):

    import torch.nn as nn

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["dx"]),
        y=train_df["dx"]
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32
        ).cuda()
    )
    return criterion, nn


@app.cell
def _():
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

    return (
        EfficientNet_B1_Weights,
        accuracy_score,
        balanced_accuracy_score,
        efficientnet_b1,
        f1_score,
        tqdm,
    )


@app.cell
def _(torch):
    config = {
        "num_classes": 7,
        "epochs": 100,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_path": "effnet_b1_ham10000_best.pth",
    }
    return (config,)


@app.class_definition
class EarlyStopping:
    def __init__(self, patience=10, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


@app.cell
def _():
    early_stopping = EarlyStopping(
        patience=10,
        mode="max"  # vì F1 càng lớn càng tốt
    )
    return


@app.cell
def _():
    # model_weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    # model = efficientnet_b1(weights=model_weights)

    # in_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(in_features, config["num_classes"])

    # model = model.to(config["device"])
    return


@app.cell
def _(EfficientNet_B1_Weights, config, efficientnet_b1, nn, torch):
    model_weights = EfficientNet_B1_Weights.IMAGENET1K_V1
    model = efficientnet_b1(weights=model_weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features,
        config["num_classes"]
    )

    checkpoint = torch.load(
       "/marimo/effnet_b1_ham10000_best.pth",
        map_location=config["device"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(config["device"])
    return checkpoint, model


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    checkpoint,
    config,
    f1_score,
    model,
    torch,
    tqdm,
):
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=config["lr"],
    #     weight_decay=config["weight_decay"]
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config["epochs"]
    # )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"]
    )
    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in tqdm(loader, desc="Train", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)

        acc = accuracy_score(all_targets, all_preds)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        return avg_loss, acc, bal_acc, macro_f1

    return optimizer, scheduler, train_one_epoch


@app.cell
def _(accuracy_score, balanced_accuracy_score, f1_score, torch, tqdm):
    @torch.no_grad()
    def validate_one_epoch(model, loader, criterion, device):
        model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)

        acc = accuracy_score(all_targets, all_preds)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        return avg_loss, acc, bal_acc, macro_f1

    return (validate_one_epoch,)


@app.cell
def _(
    config,
    criterion,
    model,
    optimizer,
    scheduler,
    torch,
    train_dataset,
    train_loader,
    train_one_epoch,
    val_loader,
    validate_one_epoch,
):
    best_val_f1 = 0.0
    patience = 10
    epochs_without_improve = 0
    min_delta = 1e-3

    for epoch in range(config["epochs"]):
        train_loss, train_acc, train_bal_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config["device"]
        )

        val_loss, val_acc, val_bal_acc, val_f1 = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=config["device"]
        )

        scheduler.step()

        print(
            f"Epoch [{epoch+1:02d}/{config['epochs']}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train BalAcc: {train_bal_acc:.4f} | "
            f"Train F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val BalAcc: {val_bal_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            epochs_without_improve = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": val_f1,
                    "class_to_idx": train_dataset.class_to_idx,
                    "config": config,
                },
                config["save_path"]
            )

            print(f"Saved best model with Val F1: {best_val_f1:.4f}")

        else:
            epochs_without_improve += 1
            print(
                f"No improvement for {epochs_without_improve}/{patience} epochs"
            )

        if epochs_without_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best Val F1: {best_val_f1:.4f}")
            break
    return


if __name__ == "__main__":
    app.run()
