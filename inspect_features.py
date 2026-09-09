#!/usr/bin/env python3

"""
BRUSH feature inspection
========================

Reads:
    BRUSH_FEATURES/<writer_id>/<sample_id>.npz

Writes:
    FEATURE_ANALYSIS/
        summary.json
        sample_statistics.csv
        dx_distribution.png
        dy_distribution.png
        distance_distribution.png
        speed_distribution.png
        direction_distribution.png
        points_per_sample.png
        strokes_per_sample.png
        writer_points.png
        writer_strokes.png

This script does NOT modify BRUSH_FEATURES.
"""

from pathlib import Path
import csv
import json
import math
import sys

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "BRUSH_FEATURES"
OUTPUT_DIR = PROJECT_ROOT / "FEATURE_ANALYSIS"

# Histogram samples.
# We don't need to keep every point in memory.
HISTOGRAM_SAMPLE_LIMIT = 2_000_000

# Random seed makes sampling reproducible.
RNG = np.random.default_rng(42)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print(f"ERROR: {message}")
    sys.exit(1)


def percentile(values, p):
    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, p))


def basic_stats(values):
    values = np.asarray(values)

    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(np.max(values)),
    }


def add_histogram_sample(storage, values, limit):
    """
    Add values to a reservoir-like sample.

    This is intentionally simple. We keep at most `limit`
    values for plotting/statistics so the script doesn't need
    to hold all 7.6 million trajectory points in memory.
    """

    values = np.asarray(
        values,
        dtype=np.float32
    ).ravel()

    if len(values) == 0:
        return

    remaining = limit - len(storage)

    if remaining <= 0:
        return

    if len(values) <= remaining:
        storage.extend(values.tolist())
    else:
        indices = RNG.choice(
            len(values),
            size=remaining,
            replace=False
        )

        storage.extend(
            values[indices].tolist()
        )


def save_histogram(
    values,
    title,
    xlabel,
    filename,
    bins=100,
    log_x=False,
):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[np.isfinite(values)]

    if log_x:
        values = values[values > 0]

    if len(values) == 0:
        return

    plt.figure(figsize=(10, 6))

    if log_x:

        positive = values[values > 0]

        low = max(
            float(np.min(positive)),
            1e-6
        )

        high = float(
            np.max(positive)
        )

        if high > low:
            bins_edges = np.logspace(
                np.log10(low),
                np.log10(high),
                bins
            )

            plt.hist(
                positive,
                bins=bins_edges
            )

            plt.xscale("log")

        else:
            plt.hist(
                positive,
                bins=50
            )

    else:

        plt.hist(
            values,
            bins=bins
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close()


def save_simple_histogram(
    values,
    title,
    xlabel,
    filename,
    bins=50,
):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=bins
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Samples")
    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_DIR.exists():

        fail(
            f"Input directory not found:\n"
            f"  {INPUT_DIR}\n\n"
            f"Run feature_engineering.py first."
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
            f"No writer directories found in:\n"
            f"  {INPUT_DIR}"
        )

    print("=" * 75)
    print("BRUSH FEATURE INSPECTION")
    print("=" * 75)

    print()
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # ========================================================
    # STORAGE
    # ========================================================

    dx_values = []
    dy_values = []
    distance_values = []
    speed_values = []
    direction_values = []

    points_per_sample = []
    strokes_per_sample = []

    sample_rows = []

    writer_points = {}
    writer_strokes = {}
    writer_samples = {}

    total_points = 0
    total_strokes = 0
    total_samples = 0

    eos_count = 0

    # ========================================================
    # PROCESS DATA
    # ========================================================

    for writer_dir in writer_dirs:

        writer_id = int(
            writer_dir.name
        )

        npz_files = sorted(
            [
                p
                for p in writer_dir.glob("*.npz")
                if p.stem.isdigit()
            ],
            key=lambda p: int(p.stem)
        )

        if not npz_files:
            continue

        writer_points[writer_id] = 0
        writer_strokes[writer_id] = 0
        writer_samples[writer_id] = 0

        print(
            f"Writer {writer_id:3d}: "
            f"{len(npz_files):4d} samples"
        )

        for npz_path in npz_files:

            sample_id = int(
                npz_path.stem
            )

            data = np.load(
                npz_path,
                allow_pickle=False
            )

            required = [
                "x",
                "y",
                "dx",
                "dy",
                "distance",
                "speed",
                "direction",
                "eos",
                "char_id",
            ]

            missing = [
                key
                for key in required
                if key not in data
            ]

            if missing:

                fail(
                    f"{npz_path} is missing features: "
                    f"{missing}"
                )

            x = data["x"]
            y = data["y"]

            dx = data["dx"]
            dy = data["dy"]

            distance = data["distance"]
            speed = data["speed"]
            direction = data["direction"]

            eos = data["eos"]

            sentence = str(
                data["sentence"].item()
            )

            n = len(x)

            if not (
                len(y)
                == len(dx)
                == len(dy)
                == len(distance)
                == len(speed)
                == len(direction)
                == len(eos)
                == n
            ):

                fail(
                    f"{npz_path}: feature arrays "
                    f"have inconsistent lengths."
                )

            # ------------------------------------------------
            # Basic sample statistics
            # ------------------------------------------------

            strokes = int(
                np.count_nonzero(eos)
            )

            path_length = float(
                np.sum(distance)
            )

            width = float(
                np.max(x) -
                np.min(x)
            )

            height = float(
                np.max(y) -
                np.min(y)
            )

            row = {
                "writer_id": writer_id,
                "sample_id": sample_id,
                "sentence": sentence,
                "text_length": len(sentence),
                "points": n,
                "strokes": strokes,
                "width": width,
                "height": height,
                "path_length": path_length,
                "mean_distance": float(
                    np.mean(distance)
                ),
                "max_distance": float(
                    np.max(distance)
                ),
                "mean_speed": float(
                    np.mean(speed)
                ),
                "max_speed": float(
                    np.max(speed)
                ),
            }

            sample_rows.append(row)

            # ------------------------------------------------
            # Global counters
            # ------------------------------------------------

            total_samples += 1
            total_points += n
            total_strokes += strokes
            eos_count += strokes

            writer_samples[
                writer_id
            ] += 1

            writer_points[
                writer_id
            ] += n

            writer_strokes[
                writer_id
            ] += strokes

            points_per_sample.append(n)
            strokes_per_sample.append(strokes)

            # ------------------------------------------------
            # Histogram samples
            # ------------------------------------------------

            add_histogram_sample(
                dx_values,
                dx,
                HISTOGRAM_SAMPLE_LIMIT
            )

            add_histogram_sample(
                dy_values,
                dy,
                HISTOGRAM_SAMPLE_LIMIT
            )

            add_histogram_sample(
                distance_values,
                distance,
                HISTOGRAM_SAMPLE_LIMIT
            )

            add_histogram_sample(
                speed_values,
                speed,
                HISTOGRAM_SAMPLE_LIMIT
            )

            add_histogram_sample(
                direction_values,
                direction,
                HISTOGRAM_SAMPLE_LIMIT
            )

            data.close()

    # ========================================================
    # PRINT FEATURE STATISTICS
    # ========================================================

    print()
    print("=" * 75)
    print("POINT-LEVEL FEATURES")
    print("=" * 75)

    feature_stats = {}

    for name, values in [
        ("dx", dx_values),
        ("dy", dy_values),
        ("distance", distance_values),
        ("speed", speed_values),
        ("direction", direction_values),
    ]:

        stats = basic_stats(values)

        feature_stats[name] = stats

        print()
        print(name.upper())
        print("-" * 40)

        print(
            f"Mean:       {stats['mean']:.6f}"
        )

        print(
            f"Std:        {stats['std']:.6f}"
        )

        print(
            f"Min:        {stats['min']:.6f}"
        )

        print(
            f"1%:         {stats['p01']:.6f}"
        )

        print(
            f"5%:         {stats['p05']:.6f}"
        )

        print(
            f"25%:        {stats['p25']:.6f}"
        )

        print(
            f"Median:     {stats['median']:.6f}"
        )

        print(
            f"75%:        {stats['p75']:.6f}"
        )

        print(
            f"95%:        {stats['p95']:.6f}"
        )

        print(
            f"99%:        {stats['p99']:.6f}"
        )

        print(
            f"Max:        {stats['max']:.6f}"
        )

    # ========================================================
    # SAMPLE STATISTICS
    # ========================================================

    print()
    print("=" * 75)
    print("SAMPLE-LEVEL FEATURES")
    print("=" * 75)

    sample_stats = {

        "points_per_sample": basic_stats(
            points_per_sample
        ),

        "strokes_per_sample": basic_stats(
            strokes_per_sample
        ),
    }

    print()
    print("POINTS PER SAMPLE")
    print("-" * 40)

    stats = sample_stats[
        "points_per_sample"
    ]

    print(
        f"Mean:       {stats['mean']:.2f}"
    )

    print(
        f"Median:     {stats['median']:.2f}"
    )

    print(
        f"5%:         {stats['p05']:.2f}"
    )

    print(
        f"95%:        {stats['p95']:.2f}"
    )

    print(
        f"99%:        {stats['p99']:.2f}"
    )

    print(
        f"Max:        {stats['max']:.2f}"
    )

    print()
    print("STROKES PER SAMPLE")
    print("-" * 40)

    stats = sample_stats[
        "strokes_per_sample"
    ]

    print(
        f"Mean:       {stats['mean']:.2f}"
    )

    print(
        f"Median:     {stats['median']:.2f}"
    )

    print(
        f"5%:         {stats['p05']:.2f}"
    )

    print(
        f"95%:        {stats['p95']:.2f}"
    )

    print(
        f"99%:        {stats['p99']:.2f}"
    )

    print(
        f"Max:        {stats['max']:.2f}"
    )

    # ========================================================
    # OUTLIER ANALYSIS
    # ========================================================

    distance_array = np.asarray(
        distance_values,
        dtype=np.float32
    )

    speed_array = np.asarray(
        speed_values,
        dtype=np.float32
    )

    dx_array = np.asarray(
        dx_values,
        dtype=np.float32
    )

    dy_array = np.asarray(
        dy_values,
        dtype=np.float32
    )

    distance_p99 = np.percentile(
        distance_array,
        99
    )

    speed_p99 = np.percentile(
        speed_array,
        99
    )

    dx_abs_p99 = np.percentile(
        np.abs(dx_array),
        99
    )

    dy_abs_p99 = np.percentile(
        np.abs(dy_array),
        99
    )

    print()
    print("=" * 75)
    print("OUTLIER CHECK")
    print("=" * 75)

    print()
    print(
        f"Distance > 99th percentile: "
        f"{distance_p99:.4f}"
    )

    print(
        f"Speed > 99th percentile:    "
        f"{speed_p99:.4f}"
    )

    print(
        f"|dx| 99th percentile:        "
        f"{dx_abs_p99:.4f}"
    )

    print(
        f"|dy| 99th percentile:        "
        f"{dy_abs_p99:.4f}"
    )

    # ========================================================
    # WRITER STATISTICS
    # ========================================================

    writer_stats_path = (
        OUTPUT_DIR /
        "writer_statistics.csv"
    )

    with writer_stats_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "writer_id",
            "samples",
            "points",
            "strokes",
            "mean_points",
            "mean_strokes",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for writer_id in sorted(
            writer_samples
        ):

            samples = writer_samples[
                writer_id
            ]

            points = writer_points[
                writer_id
            ]

            strokes = writer_strokes[
                writer_id
            ]

            writer.writerow({

                "writer_id": writer_id,

                "samples": samples,

                "points": points,

                "strokes": strokes,

                "mean_points": (
                    points / samples
                ),

                "mean_strokes": (
                    strokes / samples
                ),
            })

    # ========================================================
    # SAMPLE CSV
    # ========================================================

    sample_csv_path = (
        OUTPUT_DIR /
        "sample_statistics.csv"
    )

    sample_fields = [
        "writer_id",
        "sample_id",
        "sentence",
        "text_length",
        "points",
        "strokes",
        "width",
        "height",
        "path_length",
        "mean_distance",
        "max_distance",
        "mean_speed",
        "max_speed",
    ]

    with sample_csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=sample_fields
        )

        writer.writeheader()
        writer.writerows(
            sample_rows
        )

    # ========================================================
    # PLOTS
    # ========================================================

    print()
    print("=" * 75)
    print("GENERATING PLOTS")
    print("=" * 75)

    save_histogram(
        dx_values,
        "BRUSH dx Distribution",
        "dx (pixels)",
        "dx_distribution.png",
    )

    save_histogram(
        dy_values,
        "BRUSH dy Distribution",
        "dy (pixels)",
        "dy_distribution.png",
    )

    save_histogram(
        distance_values,
        "BRUSH Movement Distance",
        "Distance (pixels)",
        "distance_distribution.png",
        log_x=True,
    )

    save_histogram(
        speed_values,
        "BRUSH Speed Distribution",
        "Speed (pixels/second)",
        "speed_distribution.png",
        log_x=True,
    )

    save_histogram(
        direction_values,
        "BRUSH Direction Distribution",
        "Direction (radians)",
        "direction_distribution.png",
    )

    save_simple_histogram(
        points_per_sample,
        "Trajectory Points per Sample",
        "Points",
        "points_per_sample.png",
    )

    save_simple_histogram(
        strokes_per_sample,
        "Strokes per Sample",
        "Strokes",
        "strokes_per_sample.png",
    )

    # --------------------------------------------------------
    # Writer plots
    # --------------------------------------------------------

    writer_ids = sorted(
        writer_samples
    )

    writer_point_values = [
        writer_points[w]
        for w in writer_ids
    ]

    writer_stroke_values = [
        writer_strokes[w]
        for w in writer_ids
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        writer_ids,
        writer_point_values
    )

    plt.title(
        "Trajectory Points by Writer"
    )

    plt.xlabel(
        "Writer ID"
    )

    plt.ylabel(
        "Trajectory Points"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "writer_points.png",
        dpi=150
    )

    plt.close()

    plt.figure(figsize=(12, 6))

    plt.bar(
        writer_ids,
        writer_stroke_values
    )

    plt.title(
        "Strokes by Writer"
    )

    plt.xlabel(
        "Writer ID"
    )

    plt.ylabel(
        "Strokes"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "writer_strokes.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "dataset": {
            "writers": len(writer_dirs),
            "samples": total_samples,
            "trajectory_points": total_points,
            "strokes": total_strokes,
        },

        "features": feature_stats,

        "sample_statistics": sample_stats,

        "outlier_thresholds": {
            "distance_p99": float(
                distance_p99
            ),
            "speed_p99": float(
                speed_p99
            ),
            "abs_dx_p99": float(
                dx_abs_p99
            ),
            "abs_dy_p99": float(
                dy_abs_p99
            ),
        },

        "histogram_sample_limit": (
            HISTOGRAM_SAMPLE_LIMIT
        ),

        "random_seed": 42,

        "recommended_next_step": (
            "Use these statistics to decide "
            "trajectory normalization and "
            "training representation."
        ),
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
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print("INSPECTION COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Samples inspected: "
        f"{total_samples:,}"
    )

    print(
        f"Points inspected:  "
        f"{total_points:,}"
    )

    print(
        f"Strokes:            "
        f"{total_strokes:,}"
    )

    print()
    print("Output:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("Important files:")

    print(
        f"  {OUTPUT_DIR}/summary.json"
    )

    print(
        f"  {OUTPUT_DIR}/sample_statistics.csv"
    )

    print(
        f"  {OUTPUT_DIR}/writer_statistics.csv"
    )

    print()
    print("Plots:")
    print(
        "  dx_distribution.png"
    )
    print(
        "  dy_distribution.png"
    )
    print(
        "  distance_distribution.png"
    )
    print(
        "  speed_distribution.png"
    )
    print(
        "  direction_distribution.png"
    )
    print(
        "  points_per_sample.png"
    )
    print(
        "  strokes_per_sample.png"
    )
    print(
        "  writer_points.png"
    )
    print(
        "  writer_strokes.png"
    )


if __name__ == "__main__":
    main()