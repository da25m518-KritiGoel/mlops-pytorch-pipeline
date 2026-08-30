import argparse
import json
import os

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model

def evaluate(model, loader, criterion, device, max_samples=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            if max_samples is not None and total >= max_samples:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    model = get_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
    )

    os.makedirs(output_cfg["checkpoint_dir"], exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    metrics_path = os.path.join(
        output_cfg["checkpoint_dir"],
        "metrics.jsonl",
    )

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if total >= train_cfg.get("max_train_samples", 1000000):
                break
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
	    train_cfg.get("max_val_samples", 1000000),

        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        print(json.dumps(record), flush=True)

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(
                output_cfg["checkpoint_dir"],
                output_cfg["model_name"],
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model_cfg["architecture"],
                    "num_classes": model_cfg["num_classes"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"Saved checkpoint: {checkpoint_path}",
                flush=True,
            )
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print("Early stopping triggered.", flush=True)
            break


if __name__ == "__main__":
    main()