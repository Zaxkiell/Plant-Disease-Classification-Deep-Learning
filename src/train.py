import os
import yaml
import torch
import timm
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Paths
    train_path = os.path.join(cfg["dataset"]["root_dir"], cfg["dataset"]["train_dir"])
    val_path = os.path.join(cfg["dataset"]["root_dir"], cfg["dataset"]["val_dir"])

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.ToTensor(),
    ])

    # Datasets
    train_ds = datasets.ImageFolder(train_path, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_path, transform=val_tfms)

    print("Num classes:", len(train_ds.classes))
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["dataset"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"]
    )

    # Model
    model = timm.create_model(
        cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=len(train_ds.classes)
    ).to(device)

    if cfg["model"].get("freeze_backbone", False):
        for _, param in model.named_parameters():
            param.requires_grad = False

        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif hasattr(model, "fc"):
            for p in model.fc.parameters():
                p.requires_grad = True
        else:
            head = model.get_classifier()
            for p in head.parameters():
                p.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )


    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["plot_dir"], exist_ok=True)

    best_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(cfg["training"]["epochs"]):
        # ---- Train ----
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- Val ----
        model.eval()
        correct, total, running_val_loss = 0, 0, 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct / total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(cfg["output"]["model_dir"], "best_model.pth"))
            print(f"✅ Saved best model (Val Acc={best_acc:.4f})")

    # Plot curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(cfg["output"]["plot_dir"], "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig(os.path.join(cfg["output"]["plot_dir"], "val_acc.png"))
    plt.close()

    print("Training finished. Best Val Acc:", best_acc)


if __name__ == "__main__":
    main()
