#!/usr/bin/env python3

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")

FEATURE_DIR = PROJECT_ROOT / "BRUSH_FEATURES"
OUTPUT_DIR = PROJECT_ROOT / "TRAINING_DATA"

# Dataset split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducibility
RANDOM_SEED = 42

# Minimum trajectory length
MIN_POINTS = 10


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
        )

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found:\n{FEATURE_DIR}"
        )


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample(sample_file):

    data = np.load(sample_file)

    # --------------------------------------------------------
    # Required arrays
    # --------------------------------------------------------

    required = [
        "x",
        "y",
        "dx",
        "dy",
        "distance",
        "speed",
        "direction",
        "eos",
        "pen_down",
        "char_id",
        "char_fraction",
    ]

    for key in required:

        if key not in data:
            raise ValueError(
                f"{sample_file} is missing required feature: {key}"
            )

    # --------------------------------------------------------
    # Sentence
    # --------------------------------------------------------

    if "sentence" not in data:
        raise ValueError(
            f"{sample_file} does not contain sentence metadata"
        )

    sentence = str(data["sentence"].item())

    # --------------------------------------------------------
    # Core trajectory
    # --------------------------------------------------------

    dx = data["dx"].astype(np.float32)
    dy = data["dy"].astype(np.float32)
    eos = data["eos"].astype(np.float32)

    if not (
        len(dx) == len(dy) == len(eos)
    ):
        raise ValueError(
            f"Trajectory arrays have different lengths: {sample_file}"
        )

    if len(dx) < MIN_POINTS:
        return None

    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    if not np.all(np.isfinite(dx)):
        raise ValueError(f"Non-finite dx values: {sample_file}")

    if not np.all(np.isfinite(dy)):
        raise ValueError(f"Non-finite dy values: {sample_file}")

    if not np.all(np.isfinite(eos)):
        raise ValueError(f"Non-finite eos values: {sample_file}")

    # EOS should only contain 0 or 1.
    unique_eos = np.unique(eos)

    if not np.all(
        np.isin(unique_eos, [0.0, 1.0])
    ):
        raise ValueError(
            f"Invalid EOS values in {sample_file}: {unique_eos}"
        )

    # --------------------------------------------------------
    # Create core trajectory
    #
    # This is what the first model will predict:
    #
    # [dx, dy, eos]
    # --------------------------------------------------------

    trajectory = np.column_stack(
        [
            dx,
            dy,
            eos,
        ]
    ).astype(np.float32)

    # --------------------------------------------------------
    # Character IDs
    #
    # These are useful for later analysis/debugging.
    # They are NOT part of the first model target.
    # --------------------------------------------------------

    char_id = data["char_id"].astype(np.int16)

    if len(char_id) != len(trajectory):
        raise ValueError(
            f"char_id length mismatch: {sample_file}"
        )

    return {
        "sentence": sentence,
        "trajectory": trajectory,
        "char_id": char_id,
    }


# ============================================================
# FIND ALL SAMPLES
# ============================================================

def find_samples():

    writer_dirs = sorted(
        [
            p for p in FEATURE_DIR.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda p: int(p.name)
    )

    samples = []

    for writer_dir in writer_dirs:

        writer_id = writer_dir.name

        sample_files = sorted(
            writer_dir.glob("*.npz"),
            key=lambda p: int(p.stem)
        )

        for sample_file in sample_files:

            samples.append(
                {
                    "writer_id": writer_id,
                    "sample_id": sample_file.stem,
                    "path": sample_file,
                }
            )

    return samples


# ============================================================
# SPLIT DATA
# ============================================================

def split_samples(samples):

    rng = random.Random(RANDOM_SEED)

    samples = samples.copy()

    rng.shuffle(samples)

    total = len(samples)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train = samples[:train_end]
    validation = samples[train_end:val_end]
    test = samples[val_end:]

    return train, validation, test


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(samples, split_name):

    split_dir = OUTPUT_DIR / split_name

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = []

    total_points = 0
    total_strokes = 0

    for index, sample in enumerate(samples):

        loaded = load_sample(
            sample["path"]
        )

        if loaded is None:
            continue

        trajectory = loaded["trajectory"]
        char_id = loaded["char_id"]
        sentence = loaded["sentence"]

        # ----------------------------------------------------
        # Unique output ID
        #
        # Include writer ID so samples cannot collide.
        # ----------------------------------------------------

        output_id = (
            f"writer_{sample['writer_id']}"
            f"__sample_{sample['sample_id']}"
        )

        output_file = split_dir / f"{output_id}.npz"

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        np.savez_compressed(
            output_file,
            trajectory=trajectory,
            char_id=char_id,
            sentence=np.array(sentence),
            writer_id=np.array(sample["writer_id"]),
            sample_id=np.array(sample["sample_id"]),
        )

        points = len(trajectory)

        strokes = int(
            np.sum(trajectory[:, 2] == 1)
        )

        total_points += points
        total_strokes += strokes

        metadata.append(
            {
                "id": output_id,
                "writer_id": sample["writer_id"],
                "sample_id": sample["sample_id"],
                "sentence": sentence,
                "points": points,
                "strokes": strokes,
                "file": f"{split_name}/{output_file.name}",
            }
        )

        if (
            (index + 1) % 2000 == 0
            or index + 1 == len(samples)
        ):
            print(
                f"  {split_name}: "
                f"{index + 1:,}/{len(samples):,}"
            )

    # --------------------------------------------------------
    # Metadata CSV-style JSON
    # --------------------------------------------------------

    metadata_file = (
        OUTPUT_DIR /
        f"{split_name}_metadata.json"
    )

    with open(metadata_file, "w") as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return {
        "samples": len(metadata),
        "points": total_points,
        "strokes": total_strokes,
        "writers": len(
            set(m["writer_id"] for m in metadata)
        ),
    }


# ============================================================
# VOCABULARY
# ============================================================

def build_vocabulary(samples):

    characters = Counter()

    for sample in samples:

        try:
            data = np.load(sample["path"])

            sentence = str(
                data["sentence"].item()
            )

        except Exception:
            continue

        for char in sentence:
            characters[char] += 1

    # Sort by Unicode codepoint for reproducibility.
    unique_chars = sorted(
        characters.keys()
    )

    # Reserve special tokens.
    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<START>": 2,
        "<END>": 3,
    }

    next_id = len(vocabulary)

    for char in unique_chars:

        if char not in vocabulary:

            vocabulary[char] = next_id
            next_id += 1

    return vocabulary, characters


# ============================================================
# WRITER VOCABULARY
# ============================================================

def build_writer_vocabulary(samples):

    writers = sorted(
        set(
            sample["writer_id"]
            for sample in samples
        ),
        key=lambda x: int(x)
    )

    writer_to_id = {
        writer: index
        for index, writer in enumerate(writers)
    }

    return writer_to_id


# ============================================================
# DATASET STATISTICS
# ============================================================

def calculate_statistics(samples):

    points = []
    strokes = []

    for sample in samples:

        try:
            data = np.load(sample["path"])

            trajectory = np.column_stack(
                [
                    data["dx"],
                    data["dy"],
                    data["eos"],
                ]
            )

        except Exception:
            continue

        points.append(
            len(trajectory)
        )

        strokes.append(
            int(
                np.sum(
                    trajectory[:, 2] == 1
                )
            )
        )

    if not points:
        return {}

    return {
        "samples": len(points),

        "points": {
            "mean": float(np.mean(points)),
            "median": float(np.median(points)),
            "min": int(np.min(points)),
            "max": int(np.max(points)),
            "p95": float(np.percentile(points, 95)),
        },

        "strokes": {
            "mean": float(np.mean(strokes)),
            "median": float(np.median(strokes)),
            "min": int(np.min(strokes)),
            "max": int(np.max(strokes)),
            "p95": float(np.percentile(strokes, 95)),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("BRUSH TRAINING DATA PREPARATION")
    print("=" * 75)

    validate_config()

    # --------------------------------------------------------
    # Clean/create output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find samples
    # --------------------------------------------------------

    print()
    print("Finding samples...")

    samples = find_samples()

    print(
        f"Samples found: {len(samples):,}"
    )

    if not samples:
        raise RuntimeError(
            "No samples found."
        )

    # --------------------------------------------------------
    # Verify writer count
    # --------------------------------------------------------

    writers = sorted(
        set(
            sample["writer_id"]
            for sample in samples
        ),
        key=lambda x: int(x)
    )

    print(
        f"Writers found: {len(writers)}"
    )

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    print()
    print("Building text vocabulary...")

    vocabulary, character_counts = (
        build_vocabulary(samples)
    )

    vocabulary_file = (
        OUTPUT_DIR / "vocabulary.json"
    )

    with open(
        vocabulary_file,
        "w"
    ) as f:

        json.dump(
            {
                "size": len(vocabulary),
                "tokens": vocabulary,
                "character_counts": dict(
                    character_counts
                ),
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    # --------------------------------------------------------
    # Writer vocabulary
    # --------------------------------------------------------

    writer_to_id = build_writer_vocabulary(
        samples
    )

    writer_file = (
        OUTPUT_DIR / "writers.json"
    )

    with open(
        writer_file,
        "w"
    ) as f:

        json.dump(
            {
                "num_writers": len(writer_to_id),
                "writers": writer_to_id,
            },
            f,
            indent=2
        )

    print(
        f"Writer IDs: {len(writer_to_id)}"
    )

    # --------------------------------------------------------
    # Train/validation/test split
    # --------------------------------------------------------

    print()
    print("Creating dataset split...")

    train, validation, test = (
        split_samples(samples)
    )

    print(
        f"Train:      {len(train):,}"
    )

    print(
        f"Validation: {len(validation):,}"
    )

    print(
        f"Test:       {len(test):,}"
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    print()
    print("Saving training data...")
    print()

    train_stats = save_split(
        train,
        "train"
    )

    print()

    val_stats = save_split(
        validation,
        "validation"
    )

    print()

    test_stats = save_split(
        test,
        "test"
    )

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    print()
    print("Calculating dataset statistics...")

    train_detail = calculate_statistics(train)
    val_detail = calculate_statistics(validation)
    test_detail = calculate_statistics(test)

    # --------------------------------------------------------
    # Full summary
    # --------------------------------------------------------

    summary = {
        "dataset": "BRUSH",

        "random_seed": RANDOM_SEED,

        "source_directory": str(
            FEATURE_DIR
        ),

        "output_directory": str(
            OUTPUT_DIR
        ),

        "total_samples": len(samples),

        "writers": len(writers),

        "vocabulary_size": len(vocabulary),

        "trajectory_format": [
            "dx",
            "dy",
            "eos",
        ],

        "trajectory_dtype": "float32",

        "minimum_points": MIN_POINTS,

        "sampling_rate_hz": 100,

        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },

        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },

        "detailed_statistics": {
            "train": train_detail,
            "validation": val_detail,
            "test": test_detail,
        },

        "notes": [
            "Raw BRUSH data is not modified.",
            "BRUSH_FEATURES is not modified.",
            "The first model target is [dx, dy, eos].",
            "Writer identity is preserved for style conditioning.",
            "Character IDs are preserved for analysis/debugging.",
            "Text vocabulary is stored separately.",
            "Train/validation/test samples are randomly split.",
        ],
    }

    summary_file = (
        OUTPUT_DIR / "dataset_summary.json"
    )

    with open(
        summary_file,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print final statistics
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("DATA PREPARATION COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Total samples:     {len(samples):,}"
    )

    print(
        f"Writers:            {len(writers):,}"
    )

    print(
        f"Vocabulary:         {len(vocabulary):,}"
    )

    print()

    for name, stats in [
        ("TRAIN", train_stats),
        ("VALIDATION", val_stats),
        ("TEST", test_stats),
    ]:

        print(name)
        print("-" * 40)

        print(
            f"Samples: {stats['samples']:,}"
        )

        print(
            f"Points:  {stats['points']:,}"
        )

        print(
            f"Strokes: {stats['strokes']:,}"
        )

        print(
            f"Writers: {stats['writers']:,}"
        )

        print()

    print("Trajectory format:")
    print("  [dx, dy, eos]")

    print()
    print("Output:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Files:")
    print("  train/")
    print("  validation/")
    print("  test/")
    print("  train_metadata.json")
    print("  validation_metadata.json")
    print("  test_metadata.json")
    print("  vocabulary.json")
    print("  writers.json")
    print("  dataset_summary.json")

    print()
    print("Ready for model construction.")
    print()


if __name__ == "__main__":
    main()