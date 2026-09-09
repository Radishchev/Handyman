#!/usr/bin/env python3

"""
BRUSH feature engineering
==========================

Input:
    BRUSH_JSON/<writer_id>/<sample_id>.json

Output:
    BRUSH_FEATURES/<writer_id>/<sample_id>.npz
    BRUSH_FEATURES/metadata.csv
    BRUSH_FEATURES/summary.json

Raw BRUSH_JSON data is never modified.

Derived features:
    x, y
    dx, dy
    distance
    speed
    direction
    eos
    pen_down
    char_id
    char_fraction

Recommended initial model representation:
    dx, dy, eos
"""

from pathlib import Path
import csv
import json
import sys

import numpy as np


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "BRUSH_JSON"
OUTPUT_DIR = PROJECT_ROOT / "BRUSH_FEATURES"

# BRUSH original data was sampled every 10 ms.
DT = 0.01

# Don't regenerate files that already exist.
OVERWRITE = False


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print(f"ERROR: {message}")
    sys.exit(1)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_sample(data, path):
    if not isinstance(data, dict):
        fail(f"{path} is not a JSON object.")

    for key in ("sentence", "drawing", "labels"):
        if key not in data:
            fail(f"{path} is missing '{key}'.")

    drawing = np.asarray(data["drawing"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.float32)

    if drawing.ndim != 2 or drawing.shape[1] != 3:
        fail(
            f"{path}: drawing must have shape (N, 3), "
            f"got {drawing.shape}"
        )

    if labels.ndim != 2:
        fail(
            f"{path}: labels must be 2-D, "
            f"got {labels.shape}"
        )

    if len(drawing) != len(labels):
        fail(
            f"{path}: drawing and labels have different lengths: "
            f"{len(drawing)} vs {len(labels)}"
        )

    if not np.all(np.isfinite(drawing)):
        fail(f"{path}: drawing contains NaN or infinity.")

    if not np.all(np.isfinite(labels)):
        fail(f"{path}: labels contain NaN or infinity.")

    return drawing, labels


def labels_to_char_ids(labels):
    """
    Convert one-hot labels into a character ID per trajectory point.

    - 0, 1, 2, ... = character position in sentence
    - -1 = point has no character label
    """

    n = len(labels)

    char_ids = np.full(
        n,
        -1,
        dtype=np.int16
    )

    if labels.shape[1] == 0:
        return char_ids

    active = labels > 0

    has_label = active.any(axis=1)

    if np.any(has_label):
        char_ids[has_label] = np.argmax(
            labels[has_label],
            axis=1
        ).astype(np.int16)

    return char_ids


def compute_features(drawing, labels):
    """
    Calculate point-level handwriting features.

    Important:
    BRUSH eos=1 means the current stroke ends at that point.

    Therefore, movement from point i-1 -> i is considered connected
    only if eos[i-1] == 0.
    """

    x = drawing[:, 0].astype(np.float32)
    y = drawing[:, 1].astype(np.float32)

    eos = (
        drawing[:, 2] > 0.5
    ).astype(np.uint8)

    n = len(drawing)

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    dx = np.zeros(
        n,
        dtype=np.float32
    )

    dy = np.zeros(
        n,
        dtype=np.float32
    )

    if n > 1:

        connected = eos[:-1] == 0

        dx[1:][connected] = (
            x[1:][connected] -
            x[:-1][connected]
        )

        dy[1:][connected] = (
            y[1:][connected] -
            y[:-1][connected]
        )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = np.hypot(
        dx,
        dy
    ).astype(np.float32)

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speed = (
        distance / DT
    ).astype(np.float32)

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = np.zeros(
        n,
        dtype=np.float32
    )

    moving = distance > 0

    direction[moving] = np.arctan2(
        dy[moving],
        dx[moving]
    ).astype(np.float32)

    # --------------------------------------------------------
    # Character IDs
    # --------------------------------------------------------

    char_ids = labels_to_char_ids(
        labels
    )

    # --------------------------------------------------------
    # Character progress
    # --------------------------------------------------------

    char_fraction = np.zeros(
        n,
        dtype=np.float32
    )

    for char_id in np.unique(char_ids):

        if char_id < 0:
            continue

        indices = np.flatnonzero(
            char_ids == char_id
        )

        if len(indices) == 1:

            char_fraction[
                indices[0]
            ] = 0.0

        else:

            char_fraction[indices] = np.linspace(
                0.0,
                1.0,
                len(indices),
                dtype=np.float32
            )

    # --------------------------------------------------------
    # Pen state
    # --------------------------------------------------------

    pen_down = np.ones(
        n,
        dtype=np.uint8
    )

    pen_down[eos == 1] = 0

    return {
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "speed": speed,
        "direction": direction,
        "eos": eos,
        "pen_down": pen_down,
        "char_id": char_ids,
        "char_fraction": char_fraction,
    }


def compute_metadata(
    writer_id,
    sample_id,
    sentence,
    drawing,
    features
):
    x = features["x"]
    y = features["y"]

    distance = features["distance"]
    eos = features["eos"]
    char_ids = features["char_id"]

    return {
        "writer_id": int(writer_id),
        "sample_id": int(sample_id),
        "sentence": sentence,

        "text_length": len(sentence),

        "points": int(len(drawing)),

        "strokes": int(
            np.count_nonzero(eos)
        ),

        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),

        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),

        "width": float(
            np.max(x) - np.min(x)
        ),

        "height": float(
            np.max(y) - np.min(y)
        ),

        "path_length": float(
            np.sum(distance)
        ),

        "mean_step": float(
            np.mean(distance)
        ),

        "max_step": float(
            np.max(distance)
        ),

        "mean_speed": float(
            np.mean(features["speed"])
        ),

        "max_speed": float(
            np.max(features["speed"])
        ),

        "labeled_points": int(
            np.count_nonzero(char_ids >= 0)
        ),

        "unlabeled_points": int(
            np.count_nonzero(char_ids < 0)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_DIR.exists():

        fail(
            f"Input directory not found:\n"
            f"  {INPUT_DIR}\n\n"
            f"Make sure BRUSH_JSON exists first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    writer_dirs = sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.is_dir()
            and p.name.isdigit()
        ],
        key=lambda p: int(p.name)
    )

    if not writer_dirs:

        fail(
            f"No numeric writer directories found in:\n"
            f"  {INPUT_DIR}"
        )

    print("=" * 75)
    print("BRUSH FEATURE ENGINEERING")
    print("=" * 75)

    print()
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    metadata_rows = []

    total_samples = 0
    total_points = 0
    total_strokes = 0

    # ========================================================
    # PROCESS WRITERS
    # ========================================================

    for writer_dir in writer_dirs:

        writer_id = int(
            writer_dir.name
        )

        json_files = sorted(
            [
                p
                for p in writer_dir.glob("*.json")
                if p.stem.isdigit()
            ],
            key=lambda p: int(p.stem)
        )

        if not json_files:
            continue

        output_writer_dir = (
            OUTPUT_DIR /
            writer_dir.name
        )

        output_writer_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Writer {writer_id:3d}: "
            f"{len(json_files):4d} samples"
        )

        for json_path in json_files:

            sample_id = int(
                json_path.stem
            )

            output_path = (
                output_writer_dir /
                f"{sample_id}.npz"
            )

            # ------------------------------------------------
            # Read JSON
            # ------------------------------------------------

            data = load_json(
                json_path
            )

            drawing, labels = validate_sample(
                data,
                json_path
            )

            # ------------------------------------------------
            # Feature engineering
            # ------------------------------------------------

            features = compute_features(
                drawing,
                labels
            )

            # ------------------------------------------------
            # Save features
            # ------------------------------------------------

            if (
                not output_path.exists()
                or OVERWRITE
            ):

                np.savez_compressed(

                    output_path,

                    # Original coordinates
                    x=features["x"],
                    y=features["y"],

                    # Movement
                    dx=features["dx"],
                    dy=features["dy"],

                    # Derived motion
                    distance=features["distance"],
                    speed=features["speed"],
                    direction=features["direction"],

                    # Pen information
                    eos=features["eos"],
                    pen_down=features["pen_down"],

                    # Character information
                    char_id=features["char_id"],
                    char_fraction=features["char_fraction"],

                    # Original text
                    sentence=np.array(
                        data["sentence"]
                    )
                )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            row = compute_metadata(
                writer_id,
                sample_id,
                data["sentence"],
                drawing,
                features
            )

            metadata_rows.append(row)

            total_samples += 1
            total_points += len(drawing)

            total_strokes += int(
                np.count_nonzero(
                    features["eos"]
                )
            )

    # ========================================================
    # SAVE METADATA CSV
    # ========================================================

    metadata_path = (
        OUTPUT_DIR /
        "metadata.csv"
    )

    fieldnames = [
        "writer_id",
        "sample_id",
        "sentence",
        "text_length",
        "points",
        "strokes",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "width",
        "height",
        "path_length",
        "mean_step",
        "max_step",
        "mean_speed",
        "max_speed",
        "labeled_points",
        "unlabeled_points",
    ]

    with metadata_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            metadata_rows
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "input_directory": str(
            INPUT_DIR
        ),

        "output_directory": str(
            OUTPUT_DIR
        ),

        "writers": len(
            writer_dirs
        ),

        "samples": total_samples,

        "trajectory_points": total_points,

        "strokes": total_strokes,

        "dt_seconds": DT,

        "core_training_features": [
            "dx",
            "dy",
            "eos",
        ],

        "all_features": [
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
        ],

        "notes": [
            "Raw BRUSH_JSON files are not modified.",
            "Connected movement is reset after an EOS point.",
            "Coordinates are intentionally not normalized in this first pass.",
            "Normalization will be evaluated separately.",
        ],
    }

    summary_path = (
        OUTPUT_DIR /
        "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # REPORT
    # ========================================================

    if total_samples > 0:

        average_points = (
            total_points /
            total_samples
        )

        average_strokes = (
            total_strokes /
            total_samples
        )

    else:

        average_points = 0
        average_strokes = 0

    print()
    print("-" * 75)
    print("COMPLETE")
    print("-" * 75)

    print(
        f"Writers processed:        "
        f"{len(writer_dirs):,}"
    )

    print(
        f"Samples processed:        "
        f"{total_samples:,}"
    )

    print(
        f"Trajectory points:        "
        f"{total_points:,}"
    )

    print(
        f"Strokes:                  "
        f"{total_strokes:,}"
    )

    print(
        f"Average points/sample:    "
        f"{average_points:.2f}"
    )

    print(
        f"Average strokes/sample:   "
        f"{average_strokes:.2f}"
    )

    print()
    print("Output:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {metadata_path}")
    print(f"  {summary_path}")

    print()
    print("Recommended initial model input:")
    print("  [dx, dy, eos]")

    print()
    print("Example:")
    print(
        f"  {OUTPUT_DIR}/0/0.npz"
    )


if __name__ == "__main__":
    main()