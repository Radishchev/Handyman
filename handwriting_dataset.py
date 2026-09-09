#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
DATA_DIR = PROJECT_ROOT / "TRAINING_DATA"

BATCH_SIZE = 32
NUM_WORKERS = 0  # Keep 0 for now; increase later if needed.


# ============================================================
# DATASET
# ============================================================

class HandwritingDataset(Dataset):
    """
    PyTorch dataset for BRUSH handwriting trajectories.

    Each item contains:

        text
        text_ids
        trajectory
        char_ids
        writer_id
        sample_id
    """

    def __init__(self, split):

        self.split = split

        self.split_dir = DATA_DIR / split
        self.metadata_file = (
            DATA_DIR / f"{split}_metadata.json"
        )

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found:\n{self.split_dir}"
            )

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n{self.metadata_file}"
            )

        # ----------------------------------------------------
        # Load vocabulary
        # ----------------------------------------------------

        vocabulary_file = DATA_DIR / "vocabulary.json"

        with open(
            vocabulary_file,
            "r",
            encoding="utf-8"
        ) as f:
            vocabulary_data = json.load(f)

        self.vocabulary = vocabulary_data["tokens"]

        # JSON stores dictionary values as numbers,
        # but convert explicitly to int.
        self.char_to_id = {
            char: int(idx)
            for char, idx in self.vocabulary.items()
        }

        self.pad_id = self.char_to_id["<PAD>"]
        self.unk_id = self.char_to_id["<UNK>"]
        self.start_id = self.char_to_id["<START>"]
        self.end_id = self.char_to_id["<END>"]

        # ----------------------------------------------------
        # Load writer vocabulary
        # ----------------------------------------------------

        writers_file = DATA_DIR / "writers.json"

        with open(
            writers_file,
            "r",
            encoding="utf-8"
        ) as f:
            writer_data = json.load(f)

        self.writer_to_id = {
            writer: int(idx)
            for writer, idx
            in writer_data["writers"].items()
        }

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as f:
            self.metadata = json.load(f)

        if len(self.metadata) == 0:
            raise RuntimeError(
                f"No samples found in {split}"
            )

        print(
            f"{split}: {len(self.metadata):,} samples"
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.metadata)

    # ========================================================
    # TEXT ENCODING
    # ========================================================

    def encode_text(self, text):
        """
        Convert text characters into vocabulary IDs.

        Example:

            "abc"

        becomes something like:

            [12, 13, 14]
        """

        ids = []

        for char in text:

            if char in self.char_to_id:
                ids.append(
                    self.char_to_id[char]
                )
            else:
                ids.append(
                    self.unk_id
                )

        return torch.tensor(
            ids,
            dtype=torch.long
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        item = self.metadata[index]

        file_path = DATA_DIR / item["file"]

        data = np.load(file_path)

        # ----------------------------------------------------
        # Trajectory
        # ----------------------------------------------------

        trajectory = data["trajectory"].astype(
            np.float32
        )

        trajectory = torch.from_numpy(
            trajectory
        )

        # Shape:
        #
        # [trajectory_length, 3]
        #
        # columns:
        #
        # 0 = dx
        # 1 = dy
        # 2 = eos

        # ----------------------------------------------------
        # Character IDs
        # ----------------------------------------------------

        char_ids = data["char_id"].astype(
            np.int64
        )

        char_ids = torch.from_numpy(
            char_ids
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = str(
            data["sentence"].item()
        )

        text_ids = self.encode_text(
            text
        )

        # ----------------------------------------------------
        # Writer
        # ----------------------------------------------------

        writer_id = str(
            data["writer_id"].item()
        )

        if writer_id not in self.writer_to_id:
            raise ValueError(
                f"Unknown writer ID: {writer_id}"
            )

        writer_index = self.writer_to_id[
            writer_id
        ]

        writer_index = torch.tensor(
            writer_index,
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "text": text,

            "text_ids": text_ids,

            "trajectory": trajectory,

            "char_ids": char_ids,

            "writer_id": writer_index,

            "writer_name": writer_id,

            "sample_id": str(
                data["sample_id"].item()
            ),
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def handwriting_collate(batch):
    """
    Convert variable-length samples into a padded batch.

    Text lengths can differ.
    Trajectory lengths can differ.

    Output shapes:

        text_ids:
            [batch, max_text_length]

        text_mask:
            [batch, max_text_length]

        trajectory:
            [batch, max_trajectory_length, 3]

        trajectory_mask:
            [batch, max_trajectory_length]

        char_ids:
            [batch, max_trajectory_length]
    """

    batch_size = len(batch)

    # ========================================================
    # TEXT
    # ========================================================

    text_lengths = torch.tensor(
        [
            len(item["text_ids"])
            for item in batch
        ],
        dtype=torch.long
    )

    max_text_length = int(
        text_lengths.max().item()
    )

    text_ids = torch.full(
        (
            batch_size,
            max_text_length,
        ),
        fill_value=0,
        dtype=torch.long
    )

    text_mask = torch.zeros(
        (
            batch_size,
            max_text_length,
        ),
        dtype=torch.bool
    )

    for i, item in enumerate(batch):

        length = len(
            item["text_ids"]
        )

        text_ids[
            i,
            :length
        ] = item["text_ids"]

        text_mask[
            i,
            :length
        ] = True

    # ========================================================
    # TRAJECTORY
    # ========================================================

    trajectory_lengths = torch.tensor(
        [
            len(item["trajectory"])
            for item in batch
        ],
        dtype=torch.long
    )

    max_trajectory_length = int(
        trajectory_lengths.max().item()
    )

    trajectory = torch.zeros(
        (
            batch_size,
            max_trajectory_length,
            3,
        ),
        dtype=torch.float32
    )

    trajectory_mask = torch.zeros(
        (
            batch_size,
            max_trajectory_length,
        ),
        dtype=torch.bool
    )

    char_ids = torch.full(
        (
            batch_size,
            max_trajectory_length,
        ),
        fill_value=-1,
        dtype=torch.long
    )

    for i, item in enumerate(batch):

        length = len(
            item["trajectory"]
        )

        trajectory[
            i,
            :length
        ] = item["trajectory"]

        trajectory_mask[
            i,
            :length
        ] = True

        char_ids[
            i,
            :length
        ] = item["char_ids"]

    # ========================================================
    # WRITERS
    # ========================================================

    writer_ids = torch.stack(
        [
            item["writer_id"]
            for item in batch
        ]
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        # Text
        "text": [
            item["text"]
            for item in batch
        ],

        "text_ids": text_ids,

        "text_mask": text_mask,

        "text_lengths": text_lengths,

        # Trajectory
        "trajectory": trajectory,

        "trajectory_mask": trajectory_mask,

        "trajectory_lengths": trajectory_lengths,

        # Character alignment
        "char_ids": char_ids,

        # Writer/style
        "writer_ids": writer_ids,

        "writer_names": [
            item["writer_name"]
            for item in batch
        ],

        # Dataset identity
        "sample_ids": [
            item["sample_id"]
            for item in batch
        ],
    }


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(
    split,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
):

    dataset = HandwritingDataset(
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=handwriting_collate,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# ============================================================
# BATCH INSPECTION
# ============================================================

def inspect_batch(batch):

    print()
    print("=" * 75)
    print("BATCH INSPECTION")
    print("=" * 75)

    print()

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    print("TEXT")
    print("-" * 40)

    print(
        f"Shape:       {tuple(batch['text_ids'].shape)}"
    )

    print(
        f"Mask shape:  {tuple(batch['text_mask'].shape)}"
    )

    print(
        f"Lengths:     "
        f"{batch['text_lengths'].tolist()}"
    )

    for i in range(
        min(5, len(batch["text"]))
    ):

        print(
            f"  [{i}] "
            f"writer={batch['writer_names'][i]} "
            f"text={batch['text'][i]!r}"
        )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    print()
    print("TRAJECTORY")
    print("-" * 40)

    print(
        f"Shape:       "
        f"{tuple(batch['trajectory'].shape)}"
    )

    print(
        f"Mask shape:  "
        f"{tuple(batch['trajectory_mask'].shape)}"
    )

    print(
        f"Lengths:     "
        f"{batch['trajectory_lengths'].tolist()}"
    )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    print()
    print("WRITER")
    print("-" * 40)

    print(
        f"Writer IDs:  "
        f"{batch['writer_ids'].tolist()}"
    )

    print(
        f"Writer names:"
    )

    print(
        f"  {batch['writer_names']}"
    )

    # --------------------------------------------------------
    # Character IDs
    # --------------------------------------------------------

    print()
    print("CHARACTER ALIGNMENT")
    print("-" * 40)

    print(
        f"Shape:       "
        f"{tuple(batch['char_ids'].shape)}"
    )

    # --------------------------------------------------------
    # First sample
    # --------------------------------------------------------

    print()
    print("FIRST SAMPLE")
    print("-" * 40)

    first_length = int(
        batch["trajectory_lengths"][0]
    )

    print(
        f"Text:       {batch['text'][0]!r}"
    )

    print(
        f"Writer:     {batch['writer_names'][0]}"
    )

    print(
        f"Points:     {first_length}"
    )

    print()
    print("First 10 trajectory points:")

    print(
        batch["trajectory"][0, :10]
    )

    print()
    print("First 10 character IDs:")

    print(
        batch["char_ids"][0, :10]
    )

    # --------------------------------------------------------
    # EOS
    # --------------------------------------------------------

    eos = batch["trajectory"][:, :, 2]

    eos_count = (
        eos *
        batch["trajectory_mask"].float()
    ).sum()

    print()
    print(
        f"EOS points in batch: "
        f"{int(eos_count.item())}"
    )

    # --------------------------------------------------------
    # Padding check
    # --------------------------------------------------------

    padded_positions = ~batch[
        "trajectory_mask"
    ]

    if padded_positions.any():

        padded_values = batch[
            "trajectory"
        ][padded_positions]

        max_padding_value = float(
            torch.abs(
                padded_values
            ).max().item()
        )

        print(
            f"Maximum padding trajectory value: "
            f"{max_padding_value}"
        )

    print()
    print("=" * 75)
    print("BATCH INSPECTION COMPLETE")
    print("=" * 75)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("HANDWRITING DATASET TEST")
    print("=" * 75)

    # --------------------------------------------------------
    # Check train
    # --------------------------------------------------------

    train_dataset, train_loader = (
        create_dataloader(
            split="train",
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_dataset, val_loader = (
        create_dataloader(
            split="validation",
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_dataset, test_loader = (
        create_dataloader(
            split="test",
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    print()
    print("Dataset sizes:")
    print(
        f"  Train:      {len(train_dataset):,}"
    )
    print(
        f"  Validation: {len(val_dataset):,}"
    )
    print(
        f"  Test:       {len(test_dataset):,}"
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    print()
    print("Loading first training batch...")

    batch = next(
        iter(train_loader)
    )

    inspect_batch(batch)

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    assert batch["text_ids"].dtype == torch.long

    assert batch["trajectory"].dtype == torch.float32

    assert batch["trajectory"].shape[2] == 3

    assert batch["char_ids"].dtype == torch.long

    assert batch["writer_ids"].dtype == torch.long

    assert (
        batch["trajectory_mask"].dtype
        == torch.bool
    )

    assert (
        batch["text_mask"].dtype
        == torch.bool
    )

    # --------------------------------------------------------
    # Check trajectory values
    # --------------------------------------------------------

    real_points = batch[
        "trajectory"
    ][batch["trajectory_mask"]]

    assert torch.isfinite(
        real_points
    ).all()

    # --------------------------------------------------------
    # Check EOS
    # --------------------------------------------------------

    eos_values = real_points[:, 2]

    assert torch.all(
        (eos_values == 0)
        | (eos_values == 1)
    )

    print()
    print("=" * 75)
    print("ALL DATASET CHECKS PASSED")
    print("=" * 75)
    print()
    print("The data pipeline is ready for model construction.")
    print()


if __name__ == "__main__":
    main()