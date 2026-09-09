#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
FEATURE_DIR = PROJECT_ROOT / "BRUSH_FEATURES"
OUTPUT_DIR = PROJECT_ROOT / "WRITER_ANALYSIS"

POINTS_PER_SECOND = 100.0  # BRUSH original sampling rate


# ============================================================
# HELPERS
# ============================================================

def percentile(values, q):
    """Safe percentile helper."""
    values = np.asarray(values)

    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, q))


def mean_or_zero(values):
    """Safe mean helper."""
    values = np.asarray(values)

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def std_or_zero(values):
    """Safe standard deviation helper."""
    values = np.asarray(values)

    if len(values) == 0:
        return 0.0

    return float(np.std(values))


# ============================================================
# ANALYZE ONE WRITER
# ============================================================

def analyze_writer(writer_dir):
    writer_id = writer_dir.name

    sample_files = sorted(writer_dir.glob("*.npz"))

    if not sample_files:
        return None

    sample_points = []
    sample_strokes = []
    sample_widths = []
    sample_heights = []
    sample_path_lengths = []

    all_dx = []
    all_dy = []
    all_distance = []
    all_speed = []

    total_points = 0
    total_strokes = 0

    for sample_file in sample_files:

        try:
            data = np.load(sample_file)

            x = data["x"].astype(np.float64)
            y = data["y"].astype(np.float64)
            dx = data["dx"].astype(np.float64)
            dy = data["dy"].astype(np.float64)
            distance = data["distance"].astype(np.float64)
            speed = data["speed"].astype(np.float64)
            eos = data["eos"]

        except Exception as e:
            print(f"WARNING: Could not read {sample_file}: {e}")
            continue

        if len(x) == 0:
            continue

        # ----------------------------------------------------
        # Basic sample statistics
        # ----------------------------------------------------

        points = len(x)

        # EOS=1 means the current stroke ends here.
        strokes = int(np.sum(eos == 1))

        width = float(np.max(x) - np.min(x))
        height = float(np.max(y) - np.min(y))

        # Path length is the sum of connected movement distances.
        path_length = float(np.sum(distance))

        sample_points.append(points)
        sample_strokes.append(strokes)
        sample_widths.append(width)
        sample_heights.append(height)
        sample_path_lengths.append(path_length)

        total_points += points
        total_strokes += strokes

        # ----------------------------------------------------
        # Point-level statistics
        # ----------------------------------------------------

        all_dx.extend(dx.tolist())
        all_dy.extend(dy.tolist())
        all_distance.extend(distance.tolist())
        all_speed.extend(speed.tolist())

    if total_points == 0:
        return None

    # Convert to arrays
    all_dx = np.asarray(all_dx)
    all_dy = np.asarray(all_dy)
    all_distance = np.asarray(all_distance)
    all_speed = np.asarray(all_speed)

    # --------------------------------------------------------
    # Writer-level record
    # --------------------------------------------------------

    result = {
        "writer_id": writer_id,

        # Dataset size
        "samples": len(sample_points),
        "points": total_points,
        "strokes": total_strokes,

        # Sample-level
        "mean_points": mean_or_zero(sample_points),
        "median_points": percentile(sample_points, 50),
        "p95_points": percentile(sample_points, 95),

        "mean_strokes": mean_or_zero(sample_strokes),
        "median_strokes": percentile(sample_strokes, 50),
        "p95_strokes": percentile(sample_strokes, 95),

        # Spatial size
        "mean_width": mean_or_zero(sample_widths),
        "median_width": percentile(sample_widths, 50),
        "p95_width": percentile(sample_widths, 95),

        "mean_height": mean_or_zero(sample_heights),
        "median_height": percentile(sample_heights, 50),
        "p95_height": percentile(sample_heights, 95),

        # Path length
        "mean_path_length": mean_or_zero(sample_path_lengths),
        "median_path_length": percentile(sample_path_lengths, 50),
        "p95_path_length": percentile(sample_path_lengths, 95),

        # Movement
        "mean_dx": mean_or_zero(all_dx),
        "std_dx": std_or_zero(all_dx),

        "mean_dy": mean_or_zero(all_dy),
        "std_dy": std_or_zero(all_dy),

        "mean_distance": mean_or_zero(all_distance),
        "median_distance": percentile(all_distance, 50),
        "p95_distance": percentile(all_distance, 95),

        # BRUSH has 100 samples/sec, so this is pixel/sec.
        "mean_speed": mean_or_zero(all_speed),
        "median_speed": percentile(all_speed, 50),
        "p95_speed": percentile(all_speed, 95),
    }

    return result


# ============================================================
# CREATE PLOT
# ============================================================

def save_bar_plot(df, column, title, ylabel, output_file):
    """Create a writer-by-writer bar chart."""

    plot_df = df.sort_values(column).reset_index(drop=True)

    plt.figure(figsize=(16, 6))

    plt.bar(
        np.arange(len(plot_df)),
        plot_df[column]
    )

    plt.xlabel("Writer")
    plt.ylabel(ylabel)
    plt.title(title)

    # Avoid putting 170 labels on the x-axis.
    if len(plot_df) <= 30:
        plt.xticks(
            np.arange(len(plot_df)),
            plot_df["writer_id"],
            rotation=90
        )
    else:
        positions = np.linspace(
            0,
            len(plot_df) - 1,
            min(20, len(plot_df)),
            dtype=int
        )

        plt.xticks(
            positions,
            plot_df.iloc[positions]["writer_id"],
            rotation=90
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# ============================================================
# DISTRIBUTION PLOT
# ============================================================

def save_distribution_plot(df, column, title, xlabel, output_file):
    """Plot the distribution of a writer-level metric."""

    values = df[column].dropna().values

    plt.figure(figsize=(10, 6))

    plt.hist(
        values,
        bins=30
    )

    plt.xlabel(xlabel)
    plt.ylabel("Number of writers")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# ============================================================
# CORRELATION MATRIX
# ============================================================

def save_correlation_matrix(df, output_file):

    columns = [
        "mean_points",
        "mean_strokes",
        "mean_width",
        "mean_height",
        "mean_path_length",
        "mean_dx",
        "mean_dy",
        "mean_distance",
        "mean_speed",
    ]

    corr = df[columns].corr()

    plt.figure(figsize=(11, 9))

    plt.imshow(
        corr.values,
        aspect="auto"
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        np.arange(len(columns)),
        columns,
        rotation=90
    )

    plt.yticks(
        np.arange(len(columns)),
        columns
    )

    plt.title("Writer-Level Feature Correlations")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("WRITER STYLE ANALYSIS")
    print("=" * 75)

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found:\n{FEATURE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find writers
    # --------------------------------------------------------

    writer_dirs = sorted(
        [
            p for p in FEATURE_DIR.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda p: int(p.name)
    )

    print()
    print(f"Writers found: {len(writer_dirs)}")
    print()

    # --------------------------------------------------------
    # Analyze writers
    # --------------------------------------------------------

    results = []

    for index, writer_dir in enumerate(writer_dirs, start=1):

        result = analyze_writer(writer_dir)

        if result is not None:
            results.append(result)

        if index % 10 == 0 or index == len(writer_dirs):
            print(
                f"Processed {index:3d}/{len(writer_dirs)} writers"
            )

    if not results:
        raise RuntimeError("No writer data was successfully analyzed.")

    df = pd.DataFrame(results)

    # Ensure numeric writer ordering
    df["writer_number"] = df["writer_id"].astype(int)

    df = df.sort_values("writer_number")
    df = df.drop(columns=["writer_number"])

    # --------------------------------------------------------
    # Save complete CSV
    # --------------------------------------------------------

    csv_file = OUTPUT_DIR / "writer_statistics.csv"

    df.to_csv(
        csv_file,
        index=False
    )

    # --------------------------------------------------------
    # Dataset-wide statistics
    # --------------------------------------------------------

    metric_columns = [
        "mean_points",
        "mean_strokes",
        "mean_width",
        "mean_height",
        "mean_path_length",
        "mean_dx",
        "mean_dy",
        "mean_distance",
        "mean_speed",
    ]

    between_writer_stats = {}

    for column in metric_columns:

        values = df[column].values

        between_writer_stats[column] = {
            "mean_across_writers": float(np.mean(values)),
            "std_across_writers": float(np.std(values)),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "p05": percentile(values, 5),
            "p95": percentile(values, 95),
        }

    # --------------------------------------------------------
    # Coefficient of variation
    # --------------------------------------------------------

    coefficient_of_variation = {}

    for column in metric_columns:

        mean_value = float(np.mean(df[column].values))
        std_value = float(np.std(df[column].values))

        if abs(mean_value) > 1e-12:
            cv = std_value / abs(mean_value)
        else:
            cv = 0.0

        coefficient_of_variation[column] = float(cv)

    # --------------------------------------------------------
    # Most extreme writers
    # --------------------------------------------------------

    extreme_writers = {}

    for column in metric_columns:

        min_row = df.loc[df[column].idxmin()]
        max_row = df.loc[df[column].idxmax()]

        extreme_writers[column] = {
            "lowest_writer": str(min_row["writer_id"]),
            "lowest_value": float(min_row[column]),

            "highest_writer": str(max_row["writer_id"]),
            "highest_value": float(max_row[column]),
        }

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "writers_analyzed": int(len(df)),

        "total_samples": int(df["samples"].sum()),
        "total_points": int(df["points"].sum()),
        "total_strokes": int(df["strokes"].sum()),

        "points_per_writer": {
            "mean": float(df["points"].mean()),
            "median": float(df["points"].median()),
            "min": int(df["points"].min()),
            "max": int(df["points"].max()),
        },

        "samples_per_writer": {
            "mean": float(df["samples"].mean()),
            "median": float(df["samples"].median()),
            "min": int(df["samples"].min()),
            "max": int(df["samples"].max()),
        },

        "between_writer_statistics": between_writer_stats,

        "coefficient_of_variation": coefficient_of_variation,

        "extreme_writers": extreme_writers,
    }

    summary_file = OUTPUT_DIR / "summary.json"

    with open(summary_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print()
    print("Generating plots...")
    print()

    plots = [
        (
            "mean_width",
            "Average Writing Width by Writer",
            "Width (pixels)",
            "mean_width_by_writer.png"
        ),
        (
            "mean_height",
            "Average Writing Height by Writer",
            "Height (pixels)",
            "mean_height_by_writer.png"
        ),
        (
            "mean_path_length",
            "Average Path Length by Writer",
            "Path length (pixels)",
            "mean_path_length_by_writer.png"
        ),
        (
            "mean_points",
            "Average Points per Sample by Writer",
            "Points per sample",
            "mean_points_by_writer.png"
        ),
        (
            "mean_strokes",
            "Average Strokes per Sample by Writer",
            "Strokes per sample",
            "mean_strokes_by_writer.png"
        ),
        (
            "mean_distance",
            "Average Movement Distance by Writer",
            "Distance (pixels / sample)",
            "mean_distance_by_writer.png"
        ),
        (
            "mean_speed",
            "Average Speed by Writer",
            "Speed (pixels / second)",
            "mean_speed_by_writer.png"
        ),
    ]

    for column, title, ylabel, filename in plots:

        save_bar_plot(
            df,
            column,
            title,
            ylabel,
            OUTPUT_DIR / filename
        )

    # --------------------------------------------------------
    # Distribution plots
    # --------------------------------------------------------

    distributions = [
        (
            "mean_width",
            "Distribution of Writer Average Width",
            "Average width (pixels)",
            "width_distribution.png"
        ),
        (
            "mean_height",
            "Distribution of Writer Average Height",
            "Average height (pixels)",
            "height_distribution.png"
        ),
        (
            "mean_path_length",
            "Distribution of Writer Average Path Length",
            "Average path length (pixels)",
            "path_length_distribution.png"
        ),
        (
            "mean_distance",
            "Distribution of Writer Average Movement",
            "Average distance (pixels / sample)",
            "distance_distribution.png"
        ),
        (
            "mean_speed",
            "Distribution of Writer Average Speed",
            "Average speed (pixels / second)",
            "speed_distribution.png"
        ),
    ]

    for column, title, xlabel, filename in distributions:

        save_distribution_plot(
            df,
            column,
            title,
            xlabel,
            OUTPUT_DIR / filename
        )

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    save_correlation_matrix(
        df,
        OUTPUT_DIR / "writer_feature_correlations.png"
    )

    # --------------------------------------------------------
    # Print important results
    # --------------------------------------------------------

    print("=" * 75)
    print("WRITER VARIATION")
    print("=" * 75)

    print()

    important_metrics = [
        ("mean_width", "Writing width"),
        ("mean_height", "Writing height"),
        ("mean_path_length", "Path length"),
        ("mean_points", "Points/sample"),
        ("mean_strokes", "Strokes/sample"),
        ("mean_distance", "Movement distance"),
        ("mean_speed", "Speed"),
    ]

    for column, name in important_metrics:

        values = df[column].values

        print(f"{name}")
        print("-" * 40)
        print(f"Mean across writers:   {np.mean(values):.4f}")
        print(f"Std across writers:    {np.std(values):.4f}")
        print(f"Minimum:               {np.min(values):.4f}")
        print(f"Median:                {np.median(values):.4f}")
        print(f"Maximum:               {np.max(values):.4f}")
        print(
            f"Coefficient variation: "
            f"{coefficient_of_variation[column]:.4f}"
        )
        print()

    # --------------------------------------------------------
    # Identify writers with extreme style measurements
    # --------------------------------------------------------

    print("=" * 75)
    print("EXTREME WRITERS")
    print("=" * 75)

    for column, name in important_metrics:

        lowest = df.loc[df[column].idxmin()]
        highest = df.loc[df[column].idxmax()]

        print()
        print(name)
        print(
            f"  Lowest:  writer {lowest['writer_id']} "
            f"({lowest[column]:.3f})"
        )
        print(
            f"  Highest: writer {highest['writer_id']} "
            f"({highest[column]:.3f})"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print(f"Writers analyzed: {len(df)}")
    print(f"Samples:           {df['samples'].sum():,}")
    print(f"Points:            {df['points'].sum():,}")
    print(f"Strokes:           {df['strokes'].sum():,}")

    print()
    print("Output:")
    print(f"  {OUTPUT_DIR}")
    print()
    print(f"  {csv_file}")
    print(f"  {summary_file}")
    print()
    print("Plots:")
    print("  width / height")
    print("  path length")
    print("  points / strokes")
    print("  movement distance")
    print("  speed")
    print("  distributions")
    print("  writer feature correlations")
    print()


if __name__ == "__main__":
    main()