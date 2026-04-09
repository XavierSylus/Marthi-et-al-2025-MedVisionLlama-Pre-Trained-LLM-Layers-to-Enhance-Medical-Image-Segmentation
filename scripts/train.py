# train.py
import os
import math
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import random
import logging
import dataclasses
from src.data.datasets import MyDataset, RotatedDataset, IntensityAdjustedDataset
from src.metrics.metrics import dice_score, nsd_score
from src.models.vit_baseline import ViT_Baseline
from src.models.medvisionllama import MedVisionLlama
from src.models.vit_variants import ViT_Depth, ViT_MLP
from src.models.medvisionllama_variants import (
    MedVisionLlama_Frozen,
    MedVision_BioBERT,
    MedVision_BioGPT,
    MedVision_ClinicalBERT,
)
from src.models.layers import ViTArgs, ViTArgs_Depth
from src.losses.losses import DiceBCELoss
from src.utils.utils import save_predictions, generate_save_path
from torchinfo import summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    data_dir,
    data_file,
    save_path,
    task_name,
    epochs=100,
    lr=0.01,
    batch_size=2,
    patch_size=(16, 16, 4),
    image_size=(64, 64, 16),
    dataset_fraction="full",
    model_label="MedVisionLlama",
):
    """
    Trains a Vision Transformer model for segmentation on the given dataset and
    evaluates performance on training, validation, and test sets.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)

    # Select the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset initialization
    original_dataset = MyDataset(data_dir, data_file, image_size)

    # Split dataset into test set (10%) and remaining set (90%)
    total_size = len(original_dataset)
    test_size = int(0.1 * total_size)
    remaining_size = total_size - test_size

    test_dataset, remaining_dataset = random_split(
        original_dataset, [test_size, remaining_size]
    )

    # [核心架构修复 1]：引入全量测试集 DataLoader，废弃单样本抽签
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Reduce remaining training dataset based on user input
    total_remaining_size = len(remaining_dataset)
    if dataset_fraction == "ten":
        reduced_size = int(0.1 * total_remaining_size)
    elif dataset_fraction == "thirty":
        reduced_size = int(0.3 * total_remaining_size)
    else:
        reduced_size = total_remaining_size

    selected_indices = random.sample(range(total_remaining_size), reduced_size)
    reduced_dataset = Subset(remaining_dataset, selected_indices)

    # Augmented datasets
    augmented_datasets = [
        RotatedDataset(reduced_dataset, rotation_range=(-5, 5)),
        IntensityAdjustedDataset(
            reduced_dataset,
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
        ),
    ]

    # Combine training dataset with augmentations
    combined_dataset_for_train_val = ConcatDataset(
        [reduced_dataset] + augmented_datasets
    )

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(combined_dataset_for_train_val))
    val_size = len(combined_dataset_for_train_val) - train_size
    train_dataset, val_dataset = random_split(
        combined_dataset_for_train_val, [train_size, val_size]
    )

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Logging dataset sizes
    logger.info(f"Total dataset size: {total_size}")
    logger.info(f"Test set size: {len(test_dataset)}")
    logger.info(
        f"Remaining dataset size (after test set split): {len(remaining_dataset)}"
    )
    logger.info(f"Reduced dataset size (for training): {len(reduced_dataset)}")
    logger.info(f"Train dataset size after augmentation: {len(train_dataset)}")
    logger.info(f"Validation dataset size after augmentation: {len(val_dataset)}")

    # Model Initialization
    if model_label == "ViT_Baseline":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = ViT_Baseline(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "ViT_MLP":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = ViT_MLP(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "ViT_Depth":
        vit_args = ViTArgs_Depth(image_size=image_size, patch_size=patch_size)
        model = ViT_Depth(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "MedVisionLlama":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = MedVisionLlama(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "MedVisionLlama_Frozen":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = MedVisionLlama_Frozen(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "MedVision_BioGPT":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = MedVision_BioGPT(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "MedVision_BioBERT":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = MedVision_BioBERT(**dataclasses.asdict(vit_args)).to(device)

    elif model_label == "MedVision_ClinicalBERT":
        vit_args = ViTArgs(image_size=image_size, patch_size=patch_size)
        model = MedVision_ClinicalBERT(**dataclasses.asdict(vit_args)).to(device)

    else:
        raise ValueError(f"Unknown model label: {model_label}")

    # Loss function and optimizer setup
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)

    # Count total parameters in millions and log it
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model Size: {num_params:.2f}M parameters")

    # Track best loss during training
    best_loss = float("inf")
    epoch_data = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, total_dice, total_nsd = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            video, target = batch
            video, target = video.to(device), target.to(device)

            optimizer.zero_grad()
            output, _ = model(video.float())
            output = output.squeeze(1)
            target = (target > 0.5).float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score((output > 0.5).float(), target)
            total_nsd += nsd_score((output > 0.5).float(), target)

        # Compute average metrics for the epoch
        epoch_loss = total_loss / len(train_dataloader)
        epoch_dice = total_dice / len(train_dataloader)
        epoch_nsd = total_nsd / len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss, val_dice, val_nsd = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_video, val_target = val_batch
                val_video, val_target = val_video.to(device), val_target.to(device)
                val_target = (val_target > 0.5).float()
                val_output, _ = model(val_video.float())
                val_output = val_output.squeeze(1)
                val_loss += criterion(val_output, val_target).item()
                val_dice += dice_score((val_output > 0.5).float(), val_target)
                val_nsd += nsd_score((val_output > 0.5).float(), val_target)

        # Compute average metrics for validation
        val_loss /= len(val_dataloader)
        val_dice /= len(val_dataloader)
        val_nsd /= len(val_dataloader)

        scheduler.step(epoch_loss)

        # Save model checkpoint if best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            }
            model_save_path = generate_save_path(task_name, patch_size, batch_size, lr)
            torch.save(state_dict, model_save_path)

        # ==========================================
        # [核心架构修复 2]：全量测试集公平评估引擎与 NaN 防御
        # ==========================================
        model.eval()
        epoch_test_loss, epoch_test_dice, epoch_test_nsd = 0.0, 0.0, 0.0
        valid_test_batches = 0
        
        last_test_video, last_test_target, last_test_output, last_activations = None, None, None, None

        with torch.no_grad():
            for test_batch in test_dataloader:
                test_video, test_target = test_batch
                test_video, test_target = test_video.to(device), test_target.to(device)
                
                test_output, activations = model(test_video.float(), visualize=True)
                test_output = test_output.squeeze(1)
                test_target = (test_target > 0.5).float()
                
                batch_loss = criterion(test_output, test_target).item()
                batch_dice = dice_score((test_output > 0.5).float(), test_target)
                batch_nsd = nsd_score((test_output > 0.5).float(), test_target)

                # 物理拦截器：剔除 0/0 导致的数学黑洞
                if not math.isnan(float(batch_dice)) and not math.isnan(float(batch_nsd)):
                    epoch_test_loss += batch_loss
                    epoch_test_dice += float(batch_dice)
                    epoch_test_nsd += float(batch_nsd)
                    valid_test_batches += 1
                
                # 缓存最后一个 batch 用于特定 epoch 的可视化存储
                last_test_video = test_video
                last_test_target = test_target
                last_test_output = test_output
                last_activations = activations

        # 计算全量测试集的绝对平均指标
        test_loss = epoch_test_loss / valid_test_batches if valid_test_batches > 0 else float('nan')
        test_dice = epoch_test_dice / valid_test_batches if valid_test_batches > 0 else float('nan')
        test_nsd = epoch_test_nsd / valid_test_batches if valid_test_batches > 0 else float('nan')

        # 只在特定 Epoch 保存可视化结果（避免 I/O 爆炸）
        if epoch % 5 == 0 and last_test_video is not None:
            prediction_save_path = generate_save_path(
                task_name, patch_size, batch_size, lr, epoch
            )
            save_predictions(
                epoch,
                last_test_video,
                last_test_target,
                last_test_output,
                last_activations,
                prediction_save_path,
            )
        # ==========================================

        # Store results for the epoch
        epoch_data.append(
            [
                epoch + 1,
                epoch_loss,
                val_loss,
                test_loss,
                epoch_dice,
                val_dice,
                test_dice,
                epoch_nsd,
                val_nsd,
                test_nsd,
            ]
        )

        # Log the results for each epoch
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train Dice: {epoch_dice:.4f}, Val Dice: {val_dice:.4f}, Test Dice: {test_dice:.4f}, "
            f"Train NSD: {epoch_nsd:.4f}, Val NSD: {val_nsd:.4f}, Test NSD: {test_nsd:.4f}"
        )

    # Convert the epoch data to a DataFrame and save as CSV
    df = pd.DataFrame(
        epoch_data,
        columns=[
            "Epoch",
            "Train Loss",
            "Val Loss",
            "Test Loss",
            "Train Dice",
            "Val Dice",
            "Test Dice",
            "Train NSD",
            "Val NSD",
            "Test NSD",
        ],
    )
    csv_path = os.path.join(save_path, f"{task_name}_training_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Training results saved to {csv_path}")

    # Plot the training, validation and test loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
    plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss")
    plt.plot(df["Epoch"], df["Test Loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, Validation and Test Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_path, f"{task_name}_loss_curve.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Loss curve saved to {loss_plot_path}")
    plt.close()

    # Plot the training, validation and test Dice curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["Train Dice"], label="Train Dice")
    plt.plot(df["Epoch"], df["Val Dice"], label="Val Dice")
    plt.plot(df["Epoch"], df["Test Dice"], label="Test Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training, Validation and Test Dice Score")
    plt.legend()
    plt.grid(True)
    dice_plot_path = os.path.join(save_path, f"{task_name}_dice_curve.png")
    plt.savefig(dice_plot_path)
    logger.info(f"Dice curve saved to {dice_plot_path}")
    plt.close()

    # Plot the training, validation and test NSD curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df["Train NSD"], label="Train NSD")
    plt.plot(df["Epoch"], df["Val NSD"], label="Val NSD")
    plt.plot(df["Epoch"], df["Test NSD"], label="Test NSD")
    plt.xlabel("Epoch")
    plt.ylabel("NSD Score")
    plt.title("Training, Validation and Test NSD Score")
    plt.legend()
    plt.grid(True)
    nsd_plot_path = os.path.join(save_path, f"{task_name}_nsd_curve.png")
    plt.savefig(nsd_plot_path)
    logger.info(f"NSD curve saved to {nsd_plot_path}")
    plt.close()