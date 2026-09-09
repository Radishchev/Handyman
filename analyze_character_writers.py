#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
FEATURE_DIR = PROJECT_ROOT / "BRUSH_FEATURES"
OUTPUT_DIR = PROJECT_ROOT / "CHARACTER_WRITER_ANALYSIS"


# ============================================================
# CHARACTER ANALYSIS
# ============================================================

def analyze_sample(data, sentence):
    """
    Extract statistics for each character occurrence.

    BRUSH provides char_id for every trajectory point.
    Spaces normally have no trajectory points.
    """

    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    dx = data["dx"].astype(np.float64)
    dy = data["dy"].astype(np.float64)
    distance = data["distance"].astype(np.float64)
    eos = data["eos"].astype(np.int8)
    char_id = data["char_id"].astype(np.int32)

    results = []

    # Character IDs correspond to positions in the sentence.
    for cid in np.unique(char_id):

        if cid < 0 or cid >= len(sentence):
            continue

        mask = char_id == cid

        if not np.any(mask):
            continue

        char = sentence[cid]

        cx = x[mask]
        cy = y[mask]
        cdx = dx[mask]
        cdy = dy[mask]
        cdistance = distance[mask]
        ceos = eos[mask]

        # Ignore empty/non-drawn characters.
        if len(cx) == 0:
            continue

        width = np.max(cx) - np.min(cx)
        height = np.max(cy) - np.min(cy)

        path_length = np.sum(cdistance)

        strokes = np.sum(ceos == 1)

        results.append({
            "char": char,
            "char_id": int(cid),
            "points": int(len(cx)),
            "strokes": int(strokes),
            "width": float(width),
            "height": float(height),
            "path_length": float(path_length),
            "mean_dx": float(np.mean(cdx)),
            "mean_dy": float(np.mean(cdy)),
            "mean_distance": float(np.mean(cdistance)),
        })

    return results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("CHARACTER-LEVEL WRITER ANALYSIS")
    print("=" * 75)

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Feature directory not found:\n{FEATURE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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

    records = []

    # ========================================================
    # READ ALL SAMPLES
    # ========================================================

    for writer_index, writer_dir in enumerate(writer_dirs, start=1):

        writer_id = writer_dir.name

        sample_files = sorted(writer_dir.glob("*.npz"))

        for sample_file in sample_files:

            try:
                data = np.load(sample_file)

                # Sentence was saved into the NPZ metadata.
                sentence = str(data["sentence"].item())

            except Exception as e:
                print(
                    f"WARNING: Could not read {sample_file}: {e}"
                )
                continue

            character_results = analyze_sample(
                data,
                sentence
            )

            for result in character_results:

                result["writer_id"] = writer_id
                result["sample_id"] = sample_file.stem

                records.append(result)

        if writer_index % 10 == 0 or writer_index == len(writer_dirs):
            print(
                f"Processed {writer_index:3d}/{len(writer_dirs)} writers"
            )

    if not records:
        raise RuntimeError("No character data was extracted.")

    df = pd.DataFrame(records)

    # ========================================================
    # SAVE RAW CHARACTER OCCURRENCES
    # ========================================================

    occurrence_file = OUTPUT_DIR / "character_occurrences.csv"

    df.to_csv(
        occurrence_file,
        index=False
    )

    # ========================================================
    # CHARACTER SUMMARY
    # ========================================================

    print()
    print("Calculating character statistics...")
    print()

    summary_records = []

    for char, group in df.groupby("char"):

        summary_records.append({
            "char": char,
            "occurrences": len(group),
            "writers": group["writer_id"].nunique(),

            "mean_points": group["points"].mean(),
            "std_points": group["points"].std(),
            "median_points": group["points"].median(),

            "mean_strokes": group["strokes"].mean(),
            "std_strokes": group["strokes"].std(),

            "mean_width": group["width"].mean(),
            "std_width": group["width"].std(),

            "mean_height": group["height"].mean(),
            "std_height": group["height"].std(),

            "mean_path_length": group["path_length"].mean(),
            "std_path_length": group["path_length"].std(),

            "mean_distance": group["mean_distance"].mean(),
            "std_distance": group["mean_distance"].std(),
        })

    char_summary = pd.DataFrame(summary_records)

    char_summary = char_summary.sort_values(
        "occurrences",
        ascending=False
    )

    char_summary.to_csv(
        OUTPUT_DIR / "character_summary.csv",
        index=False
    )

    # ========================================================
    # WRITER × CHARACTER SUMMARY
    # ========================================================

    writer_character = (
        df.groupby(["writer_id", "char"])
        .agg(
            occurrences=("char", "size"),
            mean_points=("points", "mean"),
            mean_strokes=("strokes", "mean"),
            mean_width=("width", "mean"),
            mean_height=("height", "mean"),
            mean_path_length=("path_length", "mean"),
            mean_distance=("mean_distance", "mean"),
        )
        .reset_index()
    )

    writer_character.to_csv(
        OUTPUT_DIR / "writer_character_statistics.csv",
        index=False
    )

    # ========================================================
    # WRITER VARIATION FOR EACH CHARACTER
    # ========================================================

    variation_records = []

    for char, group in df.groupby("char"):

        writers = group["writer_id"].nunique()

        # Need at least two writers for meaningful variation.
        if writers < 2:
            continue

        metrics = [
            "points",
            "strokes",
            "width",
            "height",
            "path_length",
            "mean_distance",
        ]

        record = {
            "char": char,
            "writers": writers,
            "occurrences": len(group),
        }

        for metric in metrics:

            writer_means = (
                group.groupby("writer_id")[metric]
                .mean()
                .values
            )

            record[f"{metric}_between_writer_std"] = float(
                np.std(writer_means)
            )

            record[f"{metric}_between_writer_mean"] = float(
                np.mean(writer_means)
            )

            mean_value = np.mean(writer_means)

            if abs(mean_value) > 1e-12:
                record[f"{metric}_cv"] = float(
                    np.std(writer_means) / abs(mean_value)
                )
            else:
                record[f"{metric}_cv"] = 0.0

        variation_records.append(record)

    variation_df = pd.DataFrame(variation_records)

    variation_df = variation_df.sort_values(
        "occurrences",
        ascending=False
    )

    variation_df.to_csv(
        OUTPUT_DIR / "character_writer_variation.csv",
        index=False
    )

    # ========================================================
    # PRINT IMPORTANT CHARACTERS
    # ========================================================

    print("=" * 75)
    print("CHARACTER WRITER VARIATION")
    print("=" * 75)

    # Focus on common characters so rare symbols don't dominate.
    common = variation_df[
        variation_df["occurrences"] >= 1000
    ].copy()

    if len(common) > 0:

        common = common.sort_values(
            "path_length_cv",
            ascending=False
        )

        print()
        print("Characters with highest path-length variation:")
        print()

        for _, row in common.head(15).iterrows():

            print(
                f"'{row['char']}': "
                f"writers={int(row['writers'])}, "
                f"occurrences={int(row['occurrences'])}, "
                f"path CV={row['path_length_cv']:.3f}"
            )

        print()

        common = common.sort_values(
            "width_cv",
            ascending=False
        )

        print("Characters with highest width variation:")
        print()

        for _, row in common.head(15).iterrows():

            print(
                f"'{row['char']}': "
                f"writers={int(row['writers'])}, "
                f"occurrences={int(row['occurrences'])}, "
                f"width CV={row['width_cv']:.3f}"
            )

    # ========================================================
    # PLOTS
    # ========================================================

    print()
    print("Generating plots...")
    print()

    # --------------------------------------------------------
    # Character point count
    # --------------------------------------------------------

    top_chars = char_summary.head(30)

    plt.figure(figsize=(14, 7))

    plt.bar(
        np.arange(len(top_chars)),
        top_chars["mean_points"]
    )

    plt.xticks(
        np.arange(len(top_chars)),
        top_chars["char"],
        rotation=90
    )

    plt.xlabel("Character")
    plt.ylabel("Average trajectory points")
    plt.title("Average Trajectory Points per Character")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "character_points.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Character width variation
    # --------------------------------------------------------

    common_plot = variation_df[
        variation_df["occurrences"] >= 1000
    ].sort_values(
        "width_cv",
        ascending=False
    ).head(30)

    if len(common_plot) > 0:

        plt.figure(figsize=(14, 7))

        plt.bar(
            np.arange(len(common_plot)),
            common_plot["width_cv"]
        )

        plt.xticks(
            np.arange(len(common_plot)),
            common_plot["char"],
            rotation=90
        )

        plt.xlabel("Character")
        plt.ylabel("Coefficient of variation")
        plt.title("Writer Variation in Character Width")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "character_width_variation.png",
            dpi=150
        )

        plt.close()

    # --------------------------------------------------------
    # Path length variation
    # --------------------------------------------------------

    common_plot = variation_df[
        variation_df["occurrences"] >= 1000
    ].sort_values(
        "path_length_cv",
        ascending=False
    ).head(30)

    if len(common_plot) > 0:

        plt.figure(figsize=(14, 7))

        plt.bar(
            np.arange(len(common_plot)),
            common_plot["path_length_cv"]
        )

        plt.xticks(
            np.arange(len(common_plot)),
            common_plot["char"],
            rotation=90
        )

        plt.xlabel("Character")
        plt.ylabel("Coefficient of variation")
        plt.title("Writer Variation in Character Path Length")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "character_path_variation.png",
            dpi=150
        )

        plt.close()

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {
        "writers": int(df["writer_id"].nunique()),
        "samples": int(df["sample_id"].nunique()),
        "character_occurrences_with_trajectory": int(len(df)),
        "unique_characters": int(df["char"].nunique()),

        "most_common_characters": (
            char_summary.head(20)
            .to_dict(orient="records")
        ),

        "highest_path_variation": (
            variation_df
            .sort_values("path_length_cv", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),

        "highest_width_variation": (
            variation_df
            .sort_values("width_cv", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
    }

    with open(
        OUTPUT_DIR / "summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)

    print()
    print(
        f"Writers:               {df['writer_id'].nunique():,}"
    )

    print(
        f"Samples:               {df['sample_id'].nunique():,}"
    )

    print(
        f"Character trajectories: {len(df):,}"
    )

    print(
        f"Unique characters:     {df['char'].nunique():,}"
    )

    print()
    print("Output:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("Files:")
    print("  character_occurrences.csv")
    print("  character_summary.csv")
    print("  writer_character_statistics.csv")
    print("  character_writer_variation.csv")
    print("  summary.json")
    print()
    print("Plots:")
    print("  character_points.png")
    print("  character_width_variation.png")
    print("  character_path_variation.png")
    print()


if __name__ == "__main__":
    main()