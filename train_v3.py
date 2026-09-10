#!/usr/bin/env python3
"""
Handyman Handwriting Model v3 training.

Fresh:
    python train_v3.py --epochs 10

Resume:
    python train_v3.py --resume CHECKPOINTS/latest_v3.pt --epochs 30

When resuming from epoch 10 with --epochs 30, this runs epochs 11-30.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from handwriting_dataset import HandwritingDataset, handwriting_collate
from handwriting_model import HandwritingModel, count_parameters


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "TRAINING_DATA"
CHECKPOINT_DIR = PROJECT_ROOT / "CHECKPOINTS"

SEED = 42
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP = 1.0
POSITION_WEIGHT = 0.10
EOS_WEIGHT = 1.0


# ============================================================
# SEED / DEVICE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOSS
# ============================================================

def get_loss(model, batch):
    losses = model.calculate_loss(
        text_ids=batch["text_ids"],
        text_mask=batch["text_mask"],
        trajectory=batch["trajectory"],
        writer_ids=batch["writer_ids"],
        trajectory_mask=batch["trajectory_mask"],
        eos_weight=EOS_WEIGHT,
        position_weight=POSITION_WEIGHT,
    )

    stats = {
        "loss": float(losses["loss"].detach()),
        "mdn_loss": float(losses["mdn_loss"].detach()),
        "position_loss": float(losses["position_loss"].detach()),
    }
    return losses["loss"], stats


# ============================================================
# EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None,
    epoch=1,
    total_epochs=1,
    max_batches=None,
    split_name="TRAIN",
    grad_clip=1.0,
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_mdn = 0.0
    total_position = 0.0
    batches = 0
    start_time = time.time()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        batch = {
            key: value.to(DEVICE) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            loss, stats = get_loss(model, batch)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                optimizer.step()

        total_loss += stats["loss"]
        total_mdn += stats["mdn_loss"]
        total_position += stats["position_loss"]
        batches += 1

        elapsed = time.time() - start_time
        print(
            f"\r{split_name} epoch {epoch}/{total_epochs} "
            f"[{batch_index + 1:4d}/{max_batches or len(loader):4d}] "
            f"loss={stats['loss']:.4f} "
            f"mdn={stats['mdn_loss']:.4f} "
            f"pos={stats['position_loss']:.4f} "
            f"avg={total_loss / batches:.4f} "
            f"time={elapsed / 60:.1f}m",
            end="",
            flush=True,
        )

    print()

    if batches == 0:
        raise RuntimeError(f"No batches were processed for {split_name}.")

    return {
        "loss": total_loss / batches,
        "mdn_loss": total_mdn / batches,
        "position_loss": total_position / batches,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_loss,
    config,
    best_val_loss=None,
    best_epoch=None,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    print()
    print("=" * 75)
    print(f"RESUMING FROM CHECKPOINT: {path}")
    print("=" * 75)

    checkpoint = torch.load(path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0))
    old_val_loss = float(checkpoint.get("val_loss", float("inf")))

    # New checkpoints store the best-so-far values.
    # Older checkpoints fall back to their own validation loss.
    best_val_loss = checkpoint.get("best_val_loss", old_val_loss)
    if best_val_loss is None:
        best_val_loss = old_val_loss

    best_epoch = checkpoint.get("best_epoch", start_epoch)
    if best_epoch is None:
        best_epoch = start_epoch

    print(f"Checkpoint epoch:     {start_epoch}")
    print(f"Checkpoint val loss:  {old_val_loss:.6f}")
    print(f"Best validation loss: {float(best_val_loss):.6f}")
    print(f"Best epoch:           {int(best_epoch)}")
    print(f"Next epoch:           {start_epoch + 1}")
    print()

    return start_epoch, float(best_val_loss), int(best_epoch)


# ============================================================
# DATASET HELPERS
# ============================================================

def make_dataset(split):
    return HandwritingDataset(DATA_DIR / split)


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=handwriting_collate,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# SMOKE TEST
# ============================================================

def run_smoke_test(model, train_loader, val_loader, optimizer, epochs):
    print()
    print("SMOKE TEST: 2 train batches + 1 validation batch")
    print()

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(
            model, train_loader, optimizer,
            epoch, epochs, max_batches=2,
            split_name="TRAIN", grad_clip=DEFAULT_GRAD_CLIP,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model, val_loader, optimizer=None,
                epoch=epoch, total_epochs=epochs,
                max_batches=1, split_name="VALID",
                grad_clip=DEFAULT_GRAD_CLIP,
            )

        print()
        print(f"Epoch {epoch} summary:")
        print("  TRAIN")
        print(f"    total:    {train_stats['loss']:.6f}")
        print(f"    mdn:      {train_stats['mdn_loss']:.6f}")
        print(f"    position: {train_stats['position_loss']:.6f}")
        print("  VALIDATION")
        print(f"    total:    {val_stats['loss']:.6f}")
        print(f"    mdn:      {val_stats['mdn_loss']:.6f}")
        print(f"    position: {val_stats['position_loss']:.6f}")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            print(f"  New best model! validation={best_val:.6f}")

    return best_val


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Handyman handwriting model v3."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "FINAL epoch number. Example: resuming from epoch 10 "
            "with --epochs 30 runs epochs 11 through 30."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from, e.g. CHECKPOINTS/latest_v3.pt",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_GRAD_CLIP
    )
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    if args.smoke_test and args.resume:
        raise ValueError("--resume and --smoke-test cannot be used together.")

    set_seed(SEED)

    print("=" * 75)
    print("HANDYMAN HANDWRITING TRAINING v3")
    print("Causal BiGRU + monotonic attention + LSTM + MDN")
    print("Global-position auxiliary objective")
    print("No target leakage: input[t] -> target[t+1]")
    print("=" * 75)
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = make_dataset("train")
    validation_dataset = make_dataset("validation")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, shuffle=False
    )

    # --------------------------------------------------------
    # Dataset info
    # --------------------------------------------------------

    with open(DATA_DIR / "vocabulary.json", "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    with open(DATA_DIR / "writers.json", "r", encoding="utf-8") as f:
        writers = json.load(f)

    vocab_size = len(vocabulary["tokens"])
    num_writers = len(writers["writers"])

    print(f"Vocabulary: {vocab_size}")
    print(f"Writers:    {num_writers}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    ).to(DEVICE)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "model": "HandwritingModel_v3_causal",
        "vocab_size": vocab_size,
        "num_writers": num_writers,
        "position_scale": 100.0,
        "position_weight": POSITION_WEIGHT,
        "eos_weight": EOS_WEIGHT,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": SEED,
    }

    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------

    if args.smoke_test:
        best_val = run_smoke_test(
            model, train_loader, validation_loader,
            optimizer, args.epochs,
        )

        save_checkpoint(
            CHECKPOINT_DIR / "smoke_v3.pt",
            model, optimizer, args.epochs,
            best_val, config,
            best_val_loss=best_val,
            best_epoch=args.epochs,
        )

        print()
        print("=" * 75)
        print("SMOKE TEST COMPLETE")
        print("=" * 75)
        print(f"Best smoke-test validation loss: {best_val:.6f}")
        return

    # --------------------------------------------------------
    # Resume, if requested
    # --------------------------------------------------------

    start_epoch = 0
    best_val = float("inf")
    best_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        start_epoch, best_val, best_epoch = load_checkpoint(
            resume_path, model, optimizer
        )

        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint is already at epoch {start_epoch}, "
                f"but --epochs is {args.epochs}. "
                f"Set --epochs to a larger FINAL epoch number."
            )

    # --------------------------------------------------------
    # Full training
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = CHECKPOINT_DIR / "training_history_v3.json"
    history = []

    # Preserve previous history when resuming.
    if args.resume and history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                old_history = json.load(f)
            history = old_history.get("history", [])
            print(f"Loaded {len(history)} previous history records.")
        except (OSError, json.JSONDecodeError):
            print("Warning: could not read previous history; starting fresh history.")

    print()
    print(f"Training epochs {start_epoch + 1} -> {args.epochs}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            split_name="TRAIN",
            grad_clip=args.grad_clip,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=validation_loader,
                optimizer=None,
                epoch=epoch,
                total_epochs=args.epochs,
                split_name="VALID",
                grad_clip=args.grad_clip,
            )

        record = {
            "epoch": epoch,
            "train": train_stats,
            "validation": val_stats,
        }
        history.append(record)

        print()
        print(f"Epoch {epoch} summary:")
        print(f"  TRAIN total:       {train_stats['loss']:.6f}")
        print(f"  TRAIN mdn:         {train_stats['mdn_loss']:.6f}")
        print(f"  TRAIN position:    {train_stats['position_loss']:.6f}")
        print(f"  VALIDATION total:  {val_stats['loss']:.6f}")
        print(f"  VALIDATION mdn:    {val_stats['mdn_loss']:.6f}")
        print(f"  VALIDATION position: {val_stats['position_loss']:.6f}")

        # Always save the latest checkpoint.
        save_checkpoint(
            CHECKPOINT_DIR / "latest_v3.pt",
            model, optimizer, epoch,
            val_stats["loss"], config,
            best_val_loss=best_val,
            best_epoch=best_epoch,
        )

        # Save best checkpoint.
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_epoch = epoch

            save_checkpoint(
                CHECKPOINT_DIR / "best_v3.pt",
                model, optimizer, epoch,
                best_val, config,
                best_val_loss=best_val,
                best_epoch=best_epoch,
            )

            print(
                f"New best model! validation={best_val:.6f}"
            )

        # Save history after every epoch.
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val,
                    "history": history,
                },
                f,
                indent=2,
            )

    print()
    print("=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_v3.pt'}")
    print(f"Latest checkpoint: {CHECKPOINT_DIR / 'latest_v3.pt'}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
