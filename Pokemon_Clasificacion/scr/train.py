
"""train.py: entrena TinyVGG y TinyVGG_2."""

from pathlib import Path
import torch

import data_setup
import engine
import model_builder
import utils


def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Usando dispositivo: {device}")
    return device


def train_experiment(
    model: torch.nn.Module,
    model_name: str,
    train_dataloader,
    test_dataloader,
    device: str,
    num_epochs: int,
    learning_rate: float,
    base_save_dir: str,
):
    save_dir = Path(base_save_dir) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\n[INFO] Iniciando entrenamiento de: {model_name}")
    results = engine.train(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs, device)
    utils.save_model(model, str(save_dir), f"{model_name}.pth")
    utils.plot_loss_curves(results, save_dir=str(save_dir))
    print(
        f"[RESULTADOS] {model_name} -> "
        f"Train Acc: {results['train_acc'][-1]:.2f} | Test Acc: {results['test_acc'][-1]:.2f}"
    )
    return results


def main():
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    HIDDEN_UNITS = 64
    LEARNING_RATE = 1e-3
    IMAGE_SIZE = 64

    train_dir = r"C:/Users/INIFAP-MOVIL/Documents/3 TERCER SEMESTRE/Topicos II/Trabajos/Topicos_II/Topicos_II/Pokemon_Clasificacion/data/train"
    test_dir  = r"C:/Users/INIFAP-MOVIL/Documents/3 TERCER SEMESTRE/Topicos II/Trabajos/Topicos_II/Topicos_II/Pokemon_Clasificacion/data/test"

    
    #train_dir = "data/train"
    #test_dir = "data/test"

    device = get_device()

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    num_classes = len(class_names)
    print(f"[INFO] Número de clases: {num_classes}")

    base_save_dir = "../models"

    tinyvgg = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=num_classes,
    )
    train_experiment(
        tinyvgg, "TinyVGG_base", train_dataloader, test_dataloader,
        device, NUM_EPOCHS, LEARNING_RATE, base_save_dir,
    )

    tinyvgg2 = model_builder.TinyVGG_2(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=num_classes,
        dropout_p=0.4,
    )
    train_experiment(
        tinyvgg2, "TinyVGG_2_mejorado", train_dataloader, test_dataloader,
        device, NUM_EPOCHS, LEARNING_RATE, base_save_dir,
    )


if __name__ == "__main__":
    main()
