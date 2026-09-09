import json
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# We will analyze the human-readable JSON dataset.
DATASET_ROOT = PROJECT_ROOT / "BRUSH_JSON"

# All analysis results will go here.
OUTPUT_ROOT = PROJECT_ROOT / "BRUSH_ANALYSIS"

# BRUSH has 170 writers: 0 through 169.
EXPECTED_WRITERS = 170

# Set to True to create PNG visualizations.
CREATE_PLOTS = True


# ============================================================
# JSON LOADING
# ============================================================

def load_sample(json_path):
    """
    Load one BRUSH JSON sample.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    sentence = data["sentence"]

    drawing = np.asarray(
        data["drawing"],
        dtype=np.float64
    )

    labels = np.asarray(
        data["labels"]
    )

    return sentence, drawing, labels


# ============================================================
# FIND WRITERS
# ============================================================

def find_writers():

    writers = []

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{DATASET_ROOT}\n\n"
            "Make sure BRUSH_JSON is next to analyze_brush.py."
        )

    for path in DATASET_ROOT.iterdir():

        if not path.is_dir():
            continue

        if not path.name.isdigit():
            continue

        writer_id = int(path.name)

        if 0 <= writer_id < EXPECTED_WRITERS:
            writers.append(path)

    writers.sort(
        key=lambda p: int(p.name)
    )

    return writers


# ============================================================
# FIND JSON SAMPLES
# ============================================================

def find_samples(writer_path):

    samples = []

    for path in writer_path.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() != ".json":
            continue

        if not path.stem.isdigit():
            continue

        samples.append(path)

    samples.sort(
        key=lambda p: int(p.stem)
    )

    return samples


# ============================================================
# CHARACTER LABEL DECODING
# ============================================================

def get_character_indices(labels):

    """
    Convert one-hot labels:

        [0,1,0,0]
        [0,1,0,0]
        [0,0,1,0]

    into:

        1
        1
        2
    """

    if labels.ndim != 2:
        return np.array([], dtype=int)

    if labels.shape[1] == 0:
        return np.full(
            labels.shape[0],
            -1,
            dtype=int
        )

    indices = np.argmax(
        labels,
        axis=1
    )

    valid = (
        np.max(
            labels,
            axis=1
        ) > 0
    )

    indices = indices.astype(int)

    indices[~valid] = -1

    return indices


# ============================================================
# SAMPLE STATISTICS
# ============================================================

def analyze_sample(
    sentence,
    drawing,
    labels,
    writer_id,
    sample_id
):

    n_points = len(drawing)

    # --------------------------------------------------------
    # Basic shape checks
    # --------------------------------------------------------

    drawing_shape_ok = (
        drawing.ndim == 2
        and drawing.shape[1] == 3
    )

    labels_shape_ok = (
        labels.ndim == 2
        and labels.shape[0] == n_points
    )

    # --------------------------------------------------------
    # EOS / strokes
    # --------------------------------------------------------

    if n_points > 0:

        eos = drawing[:, 2]

        eos_count = int(
            np.sum(eos == 1)
        )

    else:

        eos_count = 0


    # EOS marks the end of a stroke.
    stroke_count = eos_count


    # If the final point doesn't have EOS,
    # there is still a final unfinished stroke.
    if (
        n_points > 0
        and drawing[-1, 2] != 1
    ):
        stroke_count += 1


    # --------------------------------------------------------
    # Coordinate statistics
    # --------------------------------------------------------

    if n_points > 0:

        x = drawing[:, 0]
        y = drawing[:, 1]

        x_min = float(np.min(x))
        x_max = float(np.max(x))

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        width = x_max - x_min
        height = y_max - y_min

    else:

        x_min = 0.0
        x_max = 0.0
        y_min = 0.0
        y_max = 0.0

        width = 0.0
        height = 0.0


    # --------------------------------------------------------
    # Point-to-point movement
    # --------------------------------------------------------

    if n_points >= 2:

        dx = np.diff(
            drawing[:, 0]
        )

        dy = np.diff(
            drawing[:, 1]
        )

        distances = np.sqrt(
            dx * dx + dy * dy
        )

        # Do not treat pen-up jumps as handwriting movement.
        valid_movement = (
            drawing[:-1, 2] != 1
        )

        pen_down_distances = (
            distances[valid_movement]
        )

        if len(pen_down_distances) > 0:

            total_pen_distance = float(
                np.sum(
                    pen_down_distances
                )
            )

            mean_step = float(
                np.mean(
                    pen_down_distances
                )
            )

            median_step = float(
                np.median(
                    pen_down_distances
                )
            )

            max_step = float(
                np.max(
                    pen_down_distances
                )
            )

        else:

            total_pen_distance = 0.0
            mean_step = 0.0
            median_step = 0.0
            max_step = 0.0

    else:

        total_pen_distance = 0.0
        mean_step = 0.0
        median_step = 0.0
        max_step = 0.0


    # --------------------------------------------------------
    # Character labels
    # --------------------------------------------------------

    character_indices = get_character_indices(
        labels
    )


    valid_character_indices = (
        character_indices[
            character_indices >= 0
        ]
    )


    labeled_points = len(
        valid_character_indices
    )


    # Number of distinct character positions
    # actually represented by trajectory points.
    distinct_character_positions = (
        len(
            np.unique(
                valid_character_indices
            )
        )
        if labeled_points > 0
        else 0
    )


    # --------------------------------------------------------
    # Text statistics
    # --------------------------------------------------------

    character_count = len(sentence)

    space_count = sentence.count(" ")

    unique_characters = len(
        set(sentence)
    )


    return {

        "writer_id": writer_id,

        "sample_id": sample_id,

        "sentence": sentence,

        "character_count": character_count,

        "space_count": space_count,

        "unique_characters": unique_characters,

        "point_count": n_points,

        "stroke_count": stroke_count,

        "eos_count": eos_count,

        "labeled_point_count": labeled_points,

        "distinct_labeled_character_positions":
            distinct_character_positions,

        "x_min": x_min,

        "x_max": x_max,

        "y_min": y_min,

        "y_max": y_max,

        "width": width,

        "height": height,

        "total_pen_distance":
            total_pen_distance,

        "mean_step":
            mean_step,

        "median_step":
            median_step,

        "max_step":
            max_step,

        "drawing_shape_ok":
            drawing_shape_ok,

        "labels_shape_ok":
            labels_shape_ok,

    }


# ============================================================
# CHARACTER STATISTICS
# ============================================================

def analyze_characters(
    sentence,
    labels,
    character_statistics
):

    character_indices = get_character_indices(
        labels
    )

    for point_index, char_index in enumerate(
        character_indices
    ):

        if char_index < 0:
            continue

        if char_index >= len(sentence):
            continue

        character = sentence[char_index]

        character_statistics[character]["points"] += 1

        character_statistics[character]["samples"] += 1


# ============================================================
# WRITER STATISTICS
# ============================================================

def calculate_writer_summary(
    sample_rows
):

    if not sample_rows:
        return {}


    point_counts = np.array(
        [
            row["point_count"]
            for row in sample_rows
        ],
        dtype=float
    )


    stroke_counts = np.array(
        [
            row["stroke_count"]
            for row in sample_rows
        ],
        dtype=float
    )


    character_counts = np.array(
        [
            row["character_count"]
            for row in sample_rows
        ],
        dtype=float
    )


    return {

        "samples": len(sample_rows),

        "total_points":
            int(np.sum(point_counts)),

        "mean_points":
            float(np.mean(point_counts)),

        "median_points":
            float(np.median(point_counts)),

        "min_points":
            int(np.min(point_counts)),

        "max_points":
            int(np.max(point_counts)),

        "mean_strokes":
            float(np.mean(stroke_counts)),

        "median_strokes":
            float(np.median(stroke_counts)),

        "mean_characters":
            float(np.mean(character_counts)),

    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
    output_path
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# PLOT: POINT COUNT DISTRIBUTION
# ============================================================

def plot_point_distribution(
    rows
):

    values = [
        row["point_count"]
        for row in rows
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=50
    )

    plt.xlabel(
        "Trajectory points per sample"
    )

    plt.ylabel(
        "Number of samples"
    )

    plt.title(
        "BRUSH Trajectory Length Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "trajectory_length_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT: STROKE COUNT DISTRIBUTION
# ============================================================

def plot_stroke_distribution(
    rows
):

    values = [
        row["stroke_count"]
        for row in rows
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        values,
        bins=30
    )

    plt.xlabel(
        "Strokes per sample"
    )

    plt.ylabel(
        "Number of samples"
    )

    plt.title(
        "BRUSH Stroke Count Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "stroke_count_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT: CHARACTER FREQUENCY
# ============================================================

def plot_character_frequency(
    character_statistics
):

    sorted_chars = sorted(
        character_statistics.items(),
        key=lambda item: item[1]["points"],
        reverse=True
    )


    # Limit plot to the 50 most common characters.

    sorted_chars = sorted_chars[:50]


    characters = [
        item[0]
        for item in sorted_chars
    ]

    counts = [
        item[1]["points"]
        for item in sorted_chars
    ]


    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        range(len(characters)),
        counts
    )

    plt.xticks(
        range(len(characters)),
        characters
    )

    plt.xlabel(
        "Character"
    )

    plt.ylabel(
        "Trajectory points"
    )

    plt.title(
        "Character Frequency by Labeled Trajectory Points"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "character_frequency.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PLOT: POINTS PER CHARACTER
# ============================================================

def plot_points_per_character(
    character_statistics
):

    values = []


    for character, stats in (
        character_statistics.items()
    ):

        if stats["samples"] == 0:
            continue

        values.append(
            (
                character,
                stats["points"]
                / stats["samples"]
            )
        )


    values.sort(
        key=lambda x: x[1],
        reverse=True
    )


    values = values[:50]


    characters = [
        item[0]
        for item in values
    ]

    points = [
        item[1]
        for item in values
    ]


    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        range(len(characters)),
        points
    )

    plt.xticks(
        range(len(characters)),
        characters
    )

    plt.xlabel(
        "Character"
    )

    plt.ylabel(
        "Average labeled points"
    )

    plt.title(
        "Average Trajectory Points per Character"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_ROOT
        / "points_per_character.png",
        dpi=150
    )

    plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    writers,
    rows,
    character_statistics
):

    print()
    print("=" * 75)
    print("BRUSH DATASET ANALYSIS")
    print("=" * 75)
    print()


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("DATASET")
    print("-" * 75)

    print(
        f"Writers found:              {len(writers)}"
    )

    print(
        f"Total samples:              {len(rows)}"
    )


    total_points = sum(
        row["point_count"]
        for row in rows
    )


    total_strokes = sum(
        row["stroke_count"]
        for row in rows
    )


    print(
        f"Total trajectory points:    {total_points:,}"
    )

    print(
        f"Total strokes:              {total_strokes:,}"
    )


    if rows:

        point_counts = np.array(
            [
                row["point_count"]
                for row in rows
            ]
        )


        stroke_counts = np.array(
            [
                row["stroke_count"]
                for row in rows
            ]
        )


        character_counts = np.array(
            [
                row["character_count"]
                for row in rows
            ]
        )


        print()

        print(
            f"Average points/sample:      "
            f"{np.mean(point_counts):.2f}"
        )

        print(
            f"Median points/sample:       "
            f"{np.median(point_counts):.2f}"
        )

        print(
            f"Shortest sample:            "
            f"{np.min(point_counts):,} points"
        )

        print(
            f"Longest sample:             "
            f"{np.max(point_counts):,} points"
        )

        print()

        print(
            f"Average strokes/sample:     "
            f"{np.mean(stroke_counts):.2f}"
        )

        print(
            f"Median strokes/sample:      "
            f"{np.median(stroke_counts):.2f}"
        )

        print()

        print(
            f"Average text length:        "
            f"{np.mean(character_counts):.2f}"
        )


    # --------------------------------------------------------
    # Character statistics
    # --------------------------------------------------------

    print()
    print("CHARACTERS")
    print("-" * 75)


    print(
        f"Unique characters in text:  "
        f"{len(character_statistics)}"
    )


    sorted_chars = sorted(
        character_statistics.items(),
        key=lambda item: item[1]["points"],
        reverse=True
    )


    print()

    print(
        "Top characters by trajectory points:"
    )

    print()


    for character, stats in sorted_chars[:30]:

        display_character = character

        if character == " ":
            display_character = "<SPACE>"

        print(
            f"  {display_character!r:12} "
            f"{stats['points']:10,} points"
        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("VALIDATION")
    print("-" * 75)


    invalid_drawing = sum(
        not row["drawing_shape_ok"]
        for row in rows
    )


    invalid_labels = sum(
        not row["labels_shape_ok"]
        for row in rows
    )


    print(
        f"Invalid drawing arrays:     "
        f"{invalid_drawing}"
    )

    print(
        f"Invalid label arrays:       "
        f"{invalid_labels}"
    )


    if (
        invalid_drawing == 0
        and invalid_labels == 0
    ):

        print()
        print(
            "✓ All analyzed samples have valid basic shapes."
        )


    print()
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print(
        "Loading BRUSH JSON dataset..."
    )

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print()


    # --------------------------------------------------------
    # Find writers
    # --------------------------------------------------------

    writers = find_writers()


    if len(writers) != EXPECTED_WRITERS:

        print(
            f"WARNING: expected "
            f"{EXPECTED_WRITERS} writers, "
            f"but found {len(writers)}."
        )

        print()


    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    all_rows = []


    character_statistics = defaultdict(
        lambda: {
            "points": 0,
            "samples": 0
        }
    )


    writer_rows = []


    # ========================================================
    # PROCESS ALL WRITERS
    # ========================================================

    for writer_index, writer_path in enumerate(
        writers,
        start=1
    ):

        writer_id = int(
            writer_path.name
        )


        samples = find_samples(
            writer_path
        )


        print(
            f"[{writer_index}/{len(writers)}] "
            f"Writer {writer_id}: "
            f"{len(samples)} samples"
        )


        writer_sample_rows = []


        # ----------------------------------------------------
        # Process every sample
        # ----------------------------------------------------

        for sample_path in samples:

            try:

                (
                    sentence,
                    drawing,
                    labels
                ) = load_sample(
                    sample_path
                )


                row = analyze_sample(
                    sentence,
                    drawing,
                    labels,
                    writer_id,
                    int(sample_path.stem)
                )


                all_rows.append(
                    row
                )


                writer_sample_rows.append(
                    row
                )


                analyze_characters(
                    sentence,
                    labels,
                    character_statistics
                )


            except Exception as e:

                print(
                    f"    ERROR: "
                    f"{sample_path.name}: "
                    f"{e}"
                )


        # ----------------------------------------------------
        # Writer summary
        # ----------------------------------------------------

        summary = calculate_writer_summary(
            writer_sample_rows
        )


        if summary:

            summary["writer_id"] = writer_id

            writer_rows.append(
                summary
            )


    # ========================================================
    # SAVE DATA
    # ========================================================

    print()
    print(
        "Saving analysis results..."
    )


    # --------------------------------------------------------
    # Sample-level CSV
    # --------------------------------------------------------

    save_csv(
        all_rows,
        OUTPUT_ROOT
        / "sample_statistics.csv"
    )


    # --------------------------------------------------------
    # Writer-level CSV
    # --------------------------------------------------------

    save_csv(
        writer_rows,
        OUTPUT_ROOT
        / "writer_statistics.csv"
    )


    # --------------------------------------------------------
    # Character CSV
    # --------------------------------------------------------

    character_rows = []


    for character, stats in sorted(
        character_statistics.items()
    ):

        character_rows.append({

            "character": character,

            "display_character":
                "<SPACE>"
                if character == " "
                else character,

            "trajectory_points":
                stats["points"],

            "samples":
                stats["samples"],

            "average_points_per_sample":
                (
                    stats["points"]
                    / stats["samples"]
                    if stats["samples"] > 0
                    else 0
                )

        })


    save_csv(
        character_rows,
        OUTPUT_ROOT
        / "character_statistics.csv"
    )


    # ========================================================
    # PLOTS
    # ========================================================

    if CREATE_PLOTS:

        print(
            "Creating plots..."
        )


        plot_point_distribution(
            all_rows
        )


        plot_stroke_distribution(
            all_rows
        )


        plot_character_frequency(
            character_statistics
        )


        plot_points_per_character(
            character_statistics
        )


    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print_summary(
        writers,
        all_rows,
        character_statistics
    )


    print()
    print(
        "Analysis files written to:"
    )

    print(
        f"    {OUTPUT_ROOT}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
