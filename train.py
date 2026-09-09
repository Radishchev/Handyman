#!/usr/bin/env python3
"""
Train the Handyman Graves-style handwriting model.

Uses the existing:
    handwriting_dataset.py
    TRAINING_DATA/

and saves:
    CHECKPOINTS/best.pt
    CHECKPOINTS/latest.pt
    CHECKPOINTS/training_history.json

IMPORTANT:
The old deterministic model checkpoint is NOT compatible with this model.
This script intentionally starts a new model from scratch unless --resume
is explicitly supplied with a compatible v2 checkpoint.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from handwriting_dataset import HandwritingDataset, handwriting_collate
from handwriting_model import (
    HandwritingModel,
    count_parameters,
    mdn_loss,
)


PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
TRAINING_DATA = PROJECT_ROOT / "TRAINING_DATA"
CHECKPOINT_DIR = PROJECT_ROOT / "CHECKPOINTS"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_info() -> Tuple[int, int]:
    with open(
        TRAINING_DATA / "vocabulary.json",
        "r",
        encoding="utf-8",
    ) as f:
        vocabulary = json.load(f)

    with open(
        TRAINING_DATA / "writers.json",
        "r",
        encoding="utf-8",
    ) as f:
        writers = json.load(f)

    return len(vocabulary["tokens"]), len(writers["writers"])


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def get_loss(
    model: HandwritingModel,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    params = model(
        batch["text_ids"],
        batch["text_mask"],
        batch["trajectory"],
        batch["writer_ids"],
    )

    targets = batch["trajectory"][:, 1:, :]
    mask = batch["trajectory_mask"][:, 1:].float()

    loss = mdn_loss(
        params,
        targets[:, :, 0],
        targets[:, :, 1],
        targets[:, :, 2],
        mask,
    )

    metrics = {
        "loss": float(loss.detach().item()),
    }
    return loss, metrics


def run_epoch(
    model: HandwritingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int = 50,
    grad_clip: float = 1.0,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        loss, _ = get_loss(model, batch)

        if training:
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_idx}."
                )

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().item())
        total_batches += 1

        if batch_idx % log_every == 0 or batch_idx == 1 or batch_idx == len(loader):
            avg = total_loss / total_batches
            mode = "TRAIN" if training else "VALID"
            elapsed = time.time() - start_time

            print(
                f"{mode} epoch {epoch}/{epochs} "
                f"[{batch_idx:4d}/{len(loader):4d}] "
                f"loss={loss.item():.4f} "
                f"avg={avg:.4f} "
                f"time={elapsed/60:.1f}m"
            )

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    path: Path,
    model: HandwritingModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history: Dict,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": config,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Handyman handwriting model v2."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a compatible v2 checkpoint.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only a few batches to verify the new model/trainer.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 75)
    print("HANDYMAN HANDWRITING TRAINING v2")
    print("Gaussian window attention + MDN")
    print("=" * 75)
    print(f"Device: {device}")

    vocab_size, num_writers = load_dataset_info()
    print(f"Vocabulary: {vocab_size}")
    print(f"Writers:    {num_writers}")

    train_dataset = HandwritingDataset(split="train")
    val_dataset = HandwritingDataset(split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=handwriting_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=handwriting_collate,
    )

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
    }
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        checkpoint_path = Path(args.resume)
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(
            checkpoint.get("best_val_loss", float("inf"))
        )
        history = checkpoint.get("history", history)

        print(f"Resuming at epoch {start_epoch}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": "HandwritingModel_v2",
        "vocab_size": vocab_size,
        "num_writers": num_writers,
        "trajectory_format": "[dx, dy, eos]",
        "text_embedding_dim": 64,
        "text_hidden_dim": 128,
        "writer_embedding_dim": 32,
        "rnn_hidden_dim": 400,
        "num_window_components": 10,
        "num_mixtures": 20,
        "dropout": 0.1,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }

    if args.smoke_test:
        print("\nSMOKE TEST: 2 train batches + 1 validation batch")
        original_train = train_loader
        original_val = val_loader

        train_batches = []
        for i, batch in enumerate(original_train):
            train_batches.append(batch)
            if i >= 1:
                break

        val_batches = []
        for i, batch in enumerate(original_val):
            val_batches.append(batch)
            if i >= 0:
                break

        class SmallLoader:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        train_loader = SmallLoader(train_batches)
        val_loader = SmallLoader(val_batches)

    final_epochs = (
        start_epoch + args.epochs - 1
        if not args.resume
        else start_epoch + args.epochs - 1
    )

    for epoch in range(start_epoch, final_epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            final_epochs,
            grad_clip=args.grad_clip,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                None,
                device,
                epoch,
                final_epochs,
                grad_clip=args.grad_clip,
            )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)

        print(
            f"\nEpoch {epoch} summary:"
            f"\n  TRAIN      {train_loss:.6f}"
            f"\n  VALIDATION {val_loss:.6f}"
        )

        save_checkpoint(
            CHECKPOINT_DIR / "latest_v2.pt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            history,
            config,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                CHECKPOINT_DIR / "best_v2.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                history,
                config,
            )

            print(
                f"  New best model! "
                f"validation={best_val_loss:.6f}"
            )

        with open(
            CHECKPOINT_DIR / "training_history_v2.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "config": config,
                    "history": history,
                    "best_validation_loss": best_val_loss,
                },
                f,
                indent=2,
            )

    print("\n" + "=" * 75)
    print("TRAINING COMPLETE")
    print("=" * 75)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_v2.pt'}")
    print(f"Latest checkpoint: {CHECKPOINT_DIR / 'latest_v2.pt'}")


if __name__ == "__main__":
    main()
