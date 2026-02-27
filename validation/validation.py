# -*- coding: utf-8 -*-
"""
validation.py
-------------
Validation and verification module for automated LV segmentation.
Plug-in compatible with the existing preprocess_functions.py and training pipeline.

Usage (in your Colab notebook):
    import validation as val

    # After training, evaluate on val_loader:
    results = val.evaluate(model, val_loader, device)
    val.print_report(results)
    val.plot_predictions(model, val_dataset, device, n_samples=4)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import directed_hausdorff
import pandas as pd


# ---------------------------------------------------------------------------
# 1. CORE METRICS
# ---------------------------------------------------------------------------

def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    """
    Dice similarity coefficient between two binary masks.
    Accepts numpy arrays or torch tensors (any shape).
    Returns a float in [0, 1].  1.0 = perfect overlap.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)

    intersection = np.logical_and(pred_flat, true_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)


def iou_score(pred_mask, true_mask, smooth=1e-6):
    """
    Intersection over Union (Jaccard index).
    Returns a float in [0, 1].  1.0 = perfect overlap.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)

    intersection = np.logical_and(pred_flat, true_flat).sum()
    union        = np.logical_or(pred_flat, true_flat).sum()
    return (intersection + smooth) / (union + smooth)


def hausdorff_distance(pred_mask, true_mask):
    """
    Symmetric Hausdorff distance (in pixels) between two binary masks.
    Measures worst-case boundary mismatch.
    Returns float (pixels); lower is better.
    Returns np.nan if either mask is empty.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_pts = np.argwhere(pred_mask > 0)
    true_pts = np.argwhere(true_mask > 0)

    if len(pred_pts) == 0 or len(true_pts) == 0:
        return np.nan

    d1 = directed_hausdorff(pred_pts, true_pts)[0]
    d2 = directed_hausdorff(true_pts, pred_pts)[0]
    return max(d1, d2)


def precision_recall(pred_mask, true_mask, smooth=1e-6):
    """
    Returns (precision, recall) for binary masks.
    Precision: of all predicted positives, how many were correct?
    Recall:    of all true positives, how many did we find?
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_flat = pred_mask.flatten().astype(bool)
    true_flat = true_mask.flatten().astype(bool)

    tp = np.logical_and(pred_flat, true_flat).sum()
    fp = np.logical_and(pred_flat, ~true_flat).sum()
    fn = np.logical_and(~pred_flat, true_flat).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    return float(precision), float(recall)


def pixel_accuracy(pred_mask, true_mask):
    """Fraction of pixels correctly classified (both foreground and background)."""
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_flat = (pred_mask.flatten() > 0.5).astype(bool)
    true_flat = (true_mask.flatten() > 0.5).astype(bool)
    return float((pred_flat == true_flat).mean())


def compute_all_metrics(pred_mask, true_mask):
    """
    Convenience wrapper: returns a dict with all metrics for one sample.
    pred_mask / true_mask should already be *binary* numpy or tensor arrays.
    """
    dice  = dice_coefficient(pred_mask, true_mask)
    iou   = iou_score(pred_mask, true_mask)
    hd    = hausdorff_distance(pred_mask, true_mask)
    prec, rec = precision_recall(pred_mask, true_mask)
    acc   = pixel_accuracy(pred_mask, true_mask)

    return {
        "dice":      round(float(dice),  4),
        "iou":       round(float(iou),   4),
        "hausdorff": round(float(hd),    4) if not np.isnan(hd) else np.nan,
        "precision": round(float(prec),  4),
        "recall":    round(float(rec),   4),
        "accuracy":  round(float(acc),   4),
    }


# ---------------------------------------------------------------------------
# 2. THRESHOLD HELPER
# ---------------------------------------------------------------------------

def threshold_predictions(logits, threshold=0.5):
    """
    Convert raw model logits (or sigmoid probabilities) to a binary mask.
    Applies sigmoid if values are outside [0, 1].
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits) if logits.min() < 0 or logits.max() > 1 else logits
        return (probs > threshold).float()
    else:
        probs = 1 / (1 + np.exp(-logits)) if logits.min() < 0 or logits.max() > 1 else logits
        return (probs > threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. BATCH / LOADER EVALUATION
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, threshold=0.5, compute_hausdorff=True):
    """
    Run the model over an entire DataLoader and collect per-sample metrics.

    Parameters
    ----------
    model        : trained UNet (or any model that accepts (B, 1, H, W) tensors)
    dataloader   : val_loader or test_loader from your training script
    device       : torch.device
    threshold    : binarization threshold for predictions (default 0.5)
    compute_hausdorff : set False to skip HD (slow on large batches)

    Returns
    -------
    results : dict with keys
        "per_sample" -> list of per-sample metric dicts
        "mean"       -> dict of mean metrics across all samples
        "std"        -> dict of std  metrics across all samples
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs  = imgs.to(device)
            masks = masks.to(device).float()

            logits = model(imgs)
            preds  = threshold_predictions(logits, threshold)

            # Iterate over items in the batch
            for i in range(imgs.shape[0]):
                pred_np = preds[i, 0].cpu().numpy()
                true_np = masks[i, 0].cpu().numpy()

                m = {
                    "dice":      float(dice_coefficient(pred_np, true_np)),
                    "iou":       float(iou_score(pred_np, true_np)),
                    "precision": float(precision_recall(pred_np, true_np)[0]),
                    "recall":    float(precision_recall(pred_np, true_np)[1]),
                    "accuracy":  float(pixel_accuracy(pred_np, true_np)),
                }
                if compute_hausdorff:
                    m["hausdorff"] = float(hausdorff_distance(pred_np, true_np))

                all_metrics.append(m)

    metric_keys = list(all_metrics[0].keys())
    means = {k: float(np.nanmean([m[k] for m in all_metrics])) for k in metric_keys}
    stds  = {k: float(np.nanstd( [m[k] for m in all_metrics])) for k in metric_keys}

    return {"per_sample": all_metrics, "mean": means, "std": stds}


# ---------------------------------------------------------------------------
# 4. TRAINING-LOOP VALIDATION STEP
# ---------------------------------------------------------------------------

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Run one validation epoch and return average loss + mean Dice.
    Drop-in replacement for the manual val loop in your training script.

    Example inside your training loop:
        val_loss, val_dice = val.validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}")
    """
    model.eval()
    total_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs  = imgs.to(device)
            masks = masks.to(device).float()

            logits = model(imgs)
            loss   = criterion(logits, masks)
            total_loss += loss.item()

            preds = threshold_predictions(logits)
            for i in range(imgs.shape[0]):
                dice_scores.append(
                    dice_coefficient(preds[i, 0].cpu().numpy(),
                                     masks[i, 0].cpu().numpy())
                )

    avg_loss = total_loss / len(dataloader)
    avg_dice = float(np.mean(dice_scores))
    return avg_loss, avg_dice


# ---------------------------------------------------------------------------
# 5. REPORTING
# ---------------------------------------------------------------------------

def print_report(results):
    """
    Pretty-print a summary table from the dict returned by evaluate().
    """
    means = results["mean"]
    stds  = results["std"]
    n     = len(results["per_sample"])

    print("=" * 52)
    print(f"  VALIDATION REPORT  ({n} samples)")
    print("=" * 52)
    print(f"  {'Metric':<20} {'Mean':>8}  {'Std':>8}")
    print("-" * 52)
    for k in means:
        print(f"  {k:<20} {means[k]:>8.4f}  {stds[k]:>8.4f}")
    print("=" * 52)


def metrics_dataframe(results):
    """
    Return a pandas DataFrame with one row per sample.
    Useful for deeper analysis (groupby contour type, patient, etc.)
    """
    return pd.DataFrame(results["per_sample"])


def save_metrics_csv(results, filepath="validation_metrics.csv"):
    """Save per-sample metrics to a CSV file."""
    df = metrics_dataframe(results)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


# ---------------------------------------------------------------------------
# 6. VISUALISATION
# ---------------------------------------------------------------------------

def plot_predictions(model, dataset, device, n_samples=4, threshold=0.5,
                     save_path=None):
    """
    Plot a grid of: original image | ground-truth mask | prediction | overlay.

    Parameters
    ----------
    model     : trained model
    dataset   : ContourDataset instance (val_dataset or train_dataset)
    device    : torch.device
    n_samples : how many random examples to show
    threshold : binarization threshold
    save_path : if provided, saves the figure to this path instead of showing it
    """
    model.eval()
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)),
                               replace=False)

    fig, axes = plt.subplots(len(indices), 4,
                             figsize=(14, 3.5 * len(indices)))

    if len(indices) == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing

    col_titles = ["MRI Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, idx in enumerate(indices):
        image, true_mask = dataset[idx]   # (1, H, W) tensors

        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device))  # (1, 1, H, W)
        pred_mask = threshold_predictions(logit)[0, 0].cpu().numpy()

        img_np  = image[0].cpu().numpy()
        true_np = true_mask[0].cpu().numpy()

        # Dice for this sample
        dice = dice_coefficient(pred_mask, true_np)

        # Column 0 – raw image
        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9)

        # Column 1 – ground truth mask
        axes[row, 1].imshow(img_np, cmap="gray")
        axes[row, 1].contour(true_np, colors="lime", linewidths=1)

        # Column 2 – prediction
        axes[row, 2].imshow(img_np, cmap="gray")
        axes[row, 2].contour(pred_mask, colors="red", linewidths=1)
        axes[row, 2].set_xlabel(f"Dice = {dice:.3f}", fontsize=9)

        # Column 3 – overlay (both contours)
        axes[row, 3].imshow(img_np, cmap="gray")
        axes[row, 3].contour(true_np,  colors="lime", linewidths=1, linestyles="--")
        axes[row, 3].contour(pred_mask, colors="red",  linewidths=1)

        gt_patch   = mpatches.Patch(color="lime", label="Ground Truth")
        pred_patch = mpatches.Patch(color="red",  label="Prediction")
        axes[row, 3].legend(handles=[gt_patch, pred_patch],
                            loc="lower right", fontsize=7)

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle("LV Segmentation – Validation Predictions", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_training_curves(train_losses, val_losses, val_dices, save_path=None):
    """
    Plot training/validation loss and validation Dice over epochs.

    Parameters
    ----------
    train_losses : list of floats (one per epoch)
    val_losses   : list of floats (one per epoch)
    val_dices    : list of floats (one per epoch)
    save_path    : optional filepath to save the figure

    Example usage (modify your training loop):
        train_losses, val_losses, val_dices = [], [], []
        for epoch in range(num_epochs):
            # ... training step ...
            val_loss, val_dice = val.validate_one_epoch(model, val_loader, criterion, device)
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            val_dices.append(val_dice)
        val.plot_training_curves(train_losses, val_losses, val_dices)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss",  markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",    markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, "g-o", label="Val Dice", markersize=4)
    ax2.axhline(y=0.80, color="gray", linestyle="--", alpha=0.6, label="0.80 target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Coefficient")
    ax2.set_title("Validation Dice Over Training")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_metric_distribution(results, save_path=None):
    """
    Box plots showing the distribution of each metric across all val samples.
    Useful for spotting outliers and understanding model consistency.
    """
    df = metrics_dataframe(results)

    fig, axes = plt.subplots(1, len(df.columns), figsize=(3 * len(df.columns), 4))
    if len(df.columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax.set_title(col.capitalize(), fontsize=10)
        ax.set_ylabel("Score")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

    plt.suptitle("Metric Distributions Across Validation Set", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 7. MODEL CHECKPOINT HELPERS
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_dice, filepath="best_model.pth"):
    """Save model + optimizer state with metadata."""
    torch.save({
        "epoch":     epoch,
        "val_dice":  val_dice,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filepath)
    print(f"Checkpoint saved: epoch={epoch}, val_dice={val_dice:.4f} → {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load a checkpoint and restore model + optimizer state."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch    = checkpoint.get("epoch",    0)
    val_dice = checkpoint.get("val_dice", 0.0)
    print(f"Checkpoint loaded: epoch={epoch}, val_dice={val_dice:.4f}")
    return epoch, val_dice


# ---------------------------------------------------------------------------
# 8. FULL TRAINING LOOP TEMPLATE
# ---------------------------------------------------------------------------

def run_training_with_validation(model, train_loader, val_loader,
                                  criterion, optimizer, device,
                                  num_epochs=20,
                                  checkpoint_path="best_model.pth"):
    """
    Complete training loop with per-epoch validation, early stopping on Dice,
    and checkpoint saving.  Replaces the training loop in your Colab notebook.

    Returns
    -------
    history : dict with "train_loss", "val_loss", "val_dice" lists
    """
    best_dice = 0.0
    history   = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device).float()

            preds = model(imgs)
            loss  = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        val_loss, val_dice = validate_one_epoch(model, val_loader,
                                                criterion, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Epoch {epoch:>3}/{num_epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={val_dice:.4f}"
              + (" ← best" if val_dice > best_dice else ""))

        # Save best checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_dice, checkpoint_path)

    print(f"\nTraining complete. Best val Dice = {best_dice:.4f}")
    return history
