#!/usr/bin/env python3

"""
Handyman Handwriting Model v3 training.

This trainer expects handwriting_dataset.py to return batches containing:

    text_ids
    text_mask
    trajectory
    trajectory_mask
    writer_ids

The model handles the causal shift internally:

    trajectory[:, :-1] -> predict trajectory[:, 1:]

Do not shift the trajectory again in this training script.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from handwriting_dataset import (
    HandwritingDataset,
    handwriting_collate,
)

from handwriting_model import (
    HandwritingModel,
    count_parameters,
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    "/home/aurlin/Projects/Handyman"
)

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
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
        "loss": float(
            losses["loss"].detach()
        ),
        "mdn_loss": float(
            losses["mdn_loss"].detach()
        ),
        "position_loss": float(
            losses["position_loss"].detach()
        ),
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

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mdn = 0.0
    total_position = 0.0
    batches = 0

    start_time = time.time()

    for batch_index, batch in enumerate(loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        batch = {
            key: (
                value.to(DEVICE)
                if torch.is_tensor(value)
                else value
            )
            for key, value in batch.items()
        }

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):
            loss, stats = get_loss(
                model,
                batch,
            )

            if is_training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

                optimizer.step()

        total_loss += stats["loss"]
        total_mdn += stats["mdn_loss"]
        total_position += stats["position_loss"]

        batches += 1

        elapsed = (
            time.time() - start_time
        )

        print(
            f"\r{split_name} epoch "
            f"{epoch}/{total_epochs} "
            f"[{batch_index + 1:4d}/"
            f"{max_batches or len(loader):4d}] "
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
        raise RuntimeError(
            f"No batches were processed for {split_name}."
        )

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
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


# ============================================================
# DATASET HELPERS
# ============================================================

def make_dataset(split):
    """
    Uses the existing HandwritingDataset API.

    If your dataset constructor has additional optional
    arguments, the defaults should remain compatible with
    the project dataset created earlier.
    """

    return HandwritingDataset(
        DATA_DIR / split,
    )


def make_loader(
    dataset,
    batch_size,
    shuffle,
):
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

def run_smoke_test(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
):
    print()
    print(
        "SMOKE TEST: "
        "2 train batches + 1 validation batch"
    )
    print()

    best_val = float("inf")

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            max_batches=2,
            split_name="TRAIN",
            grad_clip=DEFAULT_GRAD_CLIP,
        )

        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                epoch=epoch,
                total_epochs=epochs,
                max_batches=1,
                split_name="VALID",
                grad_clip=DEFAULT_GRAD_CLIP,
            )

        print()
        print(
            f"Epoch {epoch} summary:"
        )
        print(
            "  TRAIN"
        )
        print(
            f"    total:    "
            f"{train_stats['loss']:.6f}"
        )
        print(
            f"    mdn:      "
            f"{train_stats['mdn_loss']:.6f}"
        )
        print(
            f"    position: "
            f"{train_stats['position_loss']:.6f}"
        )

        print(
            "  VALIDATION"
        )
        print(
            f"    total:    "
            f"{val_stats['loss']:.6f}"
        )
        print(
            f"    mdn:      "
            f"{val_stats['mdn_loss']:.6f}"
        )
        print(
            f"    position: "
            f"{val_stats['position_loss']:.6f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            print(
                f"  New best model! "
                f"validation={best_val:.6f}"
            )

    return best_val


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Handyman handwriting "
            "model v3."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed(SEED)

    print("=" * 75)
    print(
        "HANDYMAN HANDWRITING TRAINING v3"
    )
    print(
        "Causal BiGRU + monotonic attention + LSTM + MDN"
    )
    print(
        "Global-position auxiliary objective"
    )
    print(
        "No target leakage: input[t] -> target[t+1]"
    )
    print("=" * 75)

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = make_dataset(
        "train"
    )

    validation_dataset = make_dataset(
        "validation"
    )

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
    )

    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Dataset info
    # --------------------------------------------------------

    with open(
        DATA_DIR / "vocabulary.json",
        "r",
        encoding="utf-8",
    ) as f:
        vocabulary = json.load(f)

    with open(
        DATA_DIR / "writers.json",
        "r",
        encoding="utf-8",
    ) as f:
        writers = json.load(f)

    vocab_size = len(
        vocabulary["tokens"]
    )

    num_writers = len(
        writers["writers"]
    )

    print(
        f"Vocabulary: {vocab_size}"
    )

    print(
        f"Writers:    {num_writers}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    ).to(DEVICE)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

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
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            optimizer=optimizer,
            epochs=args.epochs,
        )
        save_checkpoint(
            CHECKPOINT_DIR / "smoke_v3.pt",
            model,
            optimizer,
            args.epochs,
            best_val,
            config,
        )

        print()
        print("=" * 75)
        print("SMOKE TEST COMPLETE")
        print("=" * 75)
        print(
            f"Best smoke-test validation loss: "
            f"{best_val:.6f}"
        )
        print()
        print(
            "No full-training checkpoint was "
            "written by the smoke test."
        )
        print()

        return

    # --------------------------------------------------------
    # Full training
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):

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
        print(
            f"Epoch {epoch} summary:"
        )
        print(
            f"  TRAIN total: "
            f"{train_stats['loss']:.6f}"
        )
        print(
            f"  TRAIN mdn: "
            f"{train_stats['mdn_loss']:.6f}"
        )
        print(
            f"  TRAIN position: "
            f"{train_stats['position_loss']:.6f}"
        )
        print(
            f"  VALIDATION total: "
            f"{val_stats['loss']:.6f}"
        )
        print(
            f"  VALIDATION mdn: "
            f"{val_stats['mdn_loss']:.6f}"
        )
        print(
            f"  VALIDATION position: "
            f"{val_stats['position_loss']:.6f}"
        )

        save_checkpoint(
            CHECKPOINT_DIR / "latest_v3.pt",
            model,
            optimizer,
            epoch,
            val_stats["loss"],
            config,
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_epoch = epoch

            save_checkpoint(
                CHECKPOINT_DIR / "best_v3.pt",
                model,
                optimizer,
                epoch,
                best_val,
                config,
            )

            print(
                f"New best model! "
                f"validation={best_val:.6f}"
            )

    with open(
        CHECKPOINT_DIR
        / "training_history_v3.json",
        "w",
        encoding="utf-8",
    ) as f:
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
    print(
        f"Best validation loss: "
        f"{best_val:.6f}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        "Best checkpoint: "
        f"{CHECKPOINT_DIR / 'best_v3.pt'}"
    )
    print(
        "Latest checkpoint: "
        f"{CHECKPOINT_DIR / 'latest_v3.pt'}"
    )
    print(
        "Training history: "
        f"{CHECKPOINT_DIR / 'training_history_v3.json'}"
    )


if __name__ == "__main__":
    main()
