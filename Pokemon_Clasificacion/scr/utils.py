
"""utils.py: guardar modelos y graficar curvas."""

import torch
from pathlib import Path
import matplotlib.pyplot as plt


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Guardando modelo en: {model_save_path}")
    torch.save(model.state_dict(), f=model_save_path)


def plot_loss_curves(results: dict, save_dir: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, results["train_loss"], label="Train loss")
    plt.plot(epochs, results["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, results["train_acc"], label="Train acc")
    plt.plot(epochs, results["test_acc"], label="Test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png")
    plt.show()
