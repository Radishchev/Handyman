import json
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "BRUSH_JSON"

OUTPUT_ROOT = PROJECT_ROOT / "CHARACTER_ANALYSIS"

EXPECTED_WRITERS = 170

# Maximum number of example images to create per character.
EXAMPLES_PER_CHARACTER = 10

# Set to False if you only want CSV statistics.
CREATE_IMAGES = True


# ============================================================
# HELPERS
# ============================================================

def display_character(char):
    """
    Make invisible/special characters readable in filenames and plots.
    """

    if char == " ":
        return "<SPACE>"

    if char == "\t":
        return "<TAB>"

    if char == "\n":
        return "<NEWLINE>"

    return char


def safe_character_name(char):
    """
    Convert a character into a safe directory/file name.
    """

    if char == " ":
        return "SPACE"

    if char == "\t":
        return "TAB"

    if char == "\n":
        return "NEWLINE"

    # Keep simple ASCII characters readable.
    if char.isascii() and char.isalnum():
        return char

    if char.isascii():
        mapping = {
            "/": "SLASH",
            "\\": "BACKSLASH",
            ":": "COLON",
            "*": "ASTERISK",
            "?": "QUESTION",
            '"': "QUOTE",
            "<": "LESS_THAN",
            ">": "GREATER_THAN",
            "|": "PIPE",
        }

        if char in mapping:
            return mapping[char]

    # Unicode fallback.
    return f"U+{ord(char):04X}"


def load_sample(path):
    """
    Load one BRUSH JSON sample.
    """

    with open(
        path,
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
# FIND DATASET
# ============================================================

def find_writers():

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{DATASET_ROOT}\n\n"
            "Make sure BRUSH_JSON is next to "
            "character_analysis.py."
        )

    writers = []

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
# EXTRACT CHARACTER TRAJECTORY
# ============================================================

def extract_character_trajectory(
    sentence,
    drawing,
    labels,
    character_position
):
    """
    Extract the trajectory points belonging to one
    character position in the sentence.

    Example:

        sentence = "hello"

        character_position = 1

        character = "e"

    The labels matrix tells us which trajectory points
    belong to that particular 'e'.
    """

    if labels.ndim != 2:
        return np.empty(
            (0, 3),
            dtype=np.float64
        )

    if character_position >= labels.shape[1]:
        return np.empty(
            (0, 3),
            dtype=np.float64
        )

    mask = (
        labels[:, character_position] > 0
    )

    return drawing[mask]


# ============================================================
# CHARACTER TRAJECTORY STATISTICS
# ============================================================

def trajectory_length(points):

    if len(points) < 2:
        return 0.0

    total = 0.0

    for i in range(len(points) - 1):

        # Don't connect the end of one stroke
        # to the beginning of another stroke.
        if points[i, 2] == 1:
            continue

        dx = (
            points[i + 1, 0]
            - points[i, 0]
        )

        dy = (
            points[i + 1, 1]
            - points[i, 1]
        )

        total += np.sqrt(
            dx * dx + dy * dy
        )

    return float(total)


def count_strokes(points):

    if len(points) == 0:
        return 0

    eos_count = int(
        np.sum(points[:, 2] == 1)
    )

    # If the final point isn't EOS,
    # count the unfinished stroke too.
    if points[-1, 2] != 1:
        eos_count += 1

    return eos_count


def character_metrics(points):

    point_count = len(points)

    if point_count == 0:

        return {
            "points": 0,
            "strokes": 0,
            "width": 0.0,
            "height": 0.0,
            "path_length": 0.0,
        }


    x = points[:, 0]
    y = points[:, 1]


    return {

        "points":
            point_count,

        "strokes":
            count_strokes(points),

        "width":
            float(np.max(x) - np.min(x)),

        "height":
            float(np.max(y) - np.min(y)),

        "path_length":
            trajectory_length(points),

    }


# ============================================================
# SAVE CHARACTER IMAGE
# ============================================================

def save_character_image(
    points,
    character,
    writer_id,
    sample_id,
    output_path
):
    """
    Draw one character trajectory.

    This intentionally uses equal aspect ratio because
    handwriting geometry matters.
    """

    if len(points) == 0:
        return


    fig, ax = plt.subplots(
        figsize=(5, 3)
    )


    start = 0


    for i in range(len(points)):

        if points[i, 2] == 1:

            stroke = points[
                start:i + 1
            ]

            if len(stroke) > 0:

                ax.plot(
                    stroke[:, 0],
                    stroke[:, 1],
                    linewidth=2
                )

            start = i + 1


    # Handle final stroke if it has no EOS.
    if start < len(points):

        stroke = points[start:]

        ax.plot(
            stroke[:, 0],
            stroke[:, 1],
            linewidth=2
        )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    # BRUSH coordinates have y increasing downward,
    # so invert the plot to match the original image space.
    ax.invert_yaxis()


    ax.set_title(
        f"Character: {display_character(character)}\n"
        f"Writer {writer_id} / Sample {sample_id}"
    )


    ax.set_xlabel("X")
    ax.set_ylabel("Y")


    ax.grid(
        True,
        alpha=0.2
    )


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=150
    )


    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    examples_root = (
        OUTPUT_ROOT / "examples"
    )

    if CREATE_IMAGES:

        examples_root.mkdir(
            parents=True,
            exist_ok=True
        )


    print()
    print("=" * 75)
    print("BRUSH CHARACTER ANALYSIS")
    print("=" * 75)
    print()

    print(
        f"Dataset: {DATASET_ROOT}"
    )

    print(
        f"Output:  {OUTPUT_ROOT}"
    )

    print()


    # ========================================================
    # FIND WRITERS
    # ========================================================

    writers = find_writers()


    print(
        f"Writers found: {len(writers)}"
    )

    if len(writers) != EXPECTED_WRITERS:

        print(
            f"WARNING: expected "
            f"{EXPECTED_WRITERS} writers."
        )

    print()


    # ========================================================
    # GLOBAL CHARACTER DATA
    # ========================================================

    character_occurrences = defaultdict(list)

    writer_character_occurrences = defaultdict(
        lambda: defaultdict(list)
    )


    # Used to create example images without generating
    # thousands of images.
    character_example_count = defaultdict(int)


    # One row per character occurrence.
    occurrence_rows = []


    # ========================================================
    # PROCESS DATASET
    # ========================================================

    total_samples = 0
    total_character_occurrences = 0
    total_drawn_occurrences = 0


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


        for sample_path in samples:

            try:

                (
                    sentence,
                    drawing,
                    labels
                ) = load_sample(
                    sample_path
                )

            except Exception as e:

                print(
                    f"    ERROR loading "
                    f"{sample_path}: {e}"
                )

                continue


            total_samples += 1

            sample_id = int(
                sample_path.stem
            )


            # ------------------------------------------------
            # Each position in the sentence is one character
            # occurrence.
            # ------------------------------------------------

            for char_position, character in enumerate(
                sentence
            ):

                total_character_occurrences += 1


                points = extract_character_trajectory(
                    sentence,
                    drawing,
                    labels,
                    char_position
                )


                metrics = character_metrics(
                    points
                )


                point_count = metrics[
                    "points"
                ]


                if point_count > 0:

                    total_drawn_occurrences += 1


                # ------------------------------------------------
                # Store global occurrence information
                # ------------------------------------------------

                occurrence = {

                    "writer_id":
                        writer_id,

                    "sample_id":
                        sample_id,

                    "character_position":
                        char_position,

                    "character":
                        character,

                    "points":
                        metrics["points"],

                    "strokes":
                        metrics["strokes"],

                    "width":
                        metrics["width"],

                    "height":
                        metrics["height"],

                    "path_length":
                        metrics["path_length"],

                }


                occurrence_rows.append(
                    occurrence
                )


                # ------------------------------------------------
                # Aggregate character statistics
                # ------------------------------------------------

                character_occurrences[
                    character
                ].append(
                    metrics
                )


                writer_character_occurrences[
                    writer_id
                ][character].append(
                    metrics
                )


                # ------------------------------------------------
                # Save example images
                # ------------------------------------------------

                if (
                    CREATE_IMAGES
                    and point_count > 0
                    and character_example_count[
                        character
                    ] < EXAMPLES_PER_CHARACTER
                ):

                    character_dir = (
                        examples_root
                        / safe_character_name(
                            character
                        )
                    )


                    character_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )


                    example_number = (
                        character_example_count[
                            character
                        ] + 1
                    )


                    output_path = (
                        character_dir
                        / (
                            f"example_"
                            f"{example_number:02d}_"
                            f"writer_{writer_id:03d}_"
                            f"sample_{sample_id}.png"
                        )
                    )


                    save_character_image(
                        points,
                        character,
                        writer_id,
                        sample_id,
                        output_path
                    )


                    character_example_count[
                        character
                    ] += 1


    # ========================================================
    # CHARACTER SUMMARY
    # ========================================================

    character_rows = []


    for character in sorted(
        character_occurrences.keys()
    ):

        occurrences = (
            character_occurrences[
                character
            ]
        )


        point_values = np.array(
            [
                item["points"]
                for item in occurrences
            ],
            dtype=float
        )


        stroke_values = np.array(
            [
                item["strokes"]
                for item in occurrences
            ],
            dtype=float
        )


        width_values = np.array(
            [
                item["width"]
                for item in occurrences
                if item["points"] > 0
            ],
            dtype=float
        )


        height_values = np.array(
            [
                item["height"]
                for item in occurrences
                if item["points"] > 0
            ],
            dtype=float
        )


        path_values = np.array(
            [
                item["path_length"]
                for item in occurrences
                if item["points"] > 0
            ],
            dtype=float
        )


        drawn_mask = (
            point_values > 0
        )


        drawn_occurrences = int(
            np.sum(drawn_mask)
        )


        row = {

            "character":
                character,

            "display_character":
                display_character(character),

            "text_occurrences":
                len(occurrences),

            "drawn_occurrences":
                drawn_occurrences,

            "undrawn_occurrences":
                len(occurrences)
                - drawn_occurrences,

            "total_points":
                int(np.sum(point_values)),

            "mean_points":
                float(np.mean(point_values)),

            "median_points":
                float(np.median(point_values)),

            "min_points":
                int(np.min(point_values)),

            "max_points":
                int(np.max(point_values)),

            "mean_points_when_drawn":
                (
                    float(np.mean(point_values[drawn_mask]))
                    if drawn_occurrences > 0
                    else 0.0
                ),

            "mean_strokes_when_drawn":
                (
                    float(np.mean(stroke_values[drawn_mask]))
                    if drawn_occurrences > 0
                    else 0.0
                ),

            "mean_width":
                (
                    float(np.mean(width_values))
                    if len(width_values) > 0
                    else 0.0
                ),

            "mean_height":
                (
                    float(np.mean(height_values))
                    if len(height_values) > 0
                    else 0.0
                ),

            "mean_path_length":
                (
                    float(np.mean(path_values))
                    if len(path_values) > 0
                    else 0.0
                ),

        }


        character_rows.append(
            row
        )


    # ========================================================
    # WRITER + CHARACTER SUMMARY
    # ========================================================

    writer_character_rows = []


    for writer_id in sorted(
        writer_character_occurrences
    ):

        character_dict = (
            writer_character_occurrences[
                writer_id
            ]
        )


        for character in sorted(
            character_dict
        ):

            occurrences = (
                character_dict[
                    character
                ]
            )


            points = np.array(
                [
                    item["points"]
                    for item in occurrences
                ],
                dtype=float
            )


            drawn = points > 0


            writer_character_rows.append({

                "writer_id":
                    writer_id,

                "character":
                    character,

                "display_character":
                    display_character(character),

                "occurrences":
                    len(occurrences),

                "drawn_occurrences":
                    int(np.sum(drawn)),

                "mean_points":
                    (
                        float(np.mean(points[drawn]))
                        if np.any(drawn)
                        else 0.0
                    ),

                "median_points":
                    (
                        float(np.median(points[drawn]))
                        if np.any(drawn)
                        else 0.0
                    ),

                "mean_strokes":
                    (
                        float(
                            np.mean(
                                [
                                    item["strokes"]
                                    for item in occurrences
                                    if item["points"] > 0
                                ]
                            )
                        )
                        if np.any(drawn)
                        else 0.0
                    ),

            })


    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    print()
    print(
        "Saving CSV files..."
    )


    occurrence_csv = (
        OUTPUT_ROOT
        / "character_occurrences.csv"
    )


    if occurrence_rows:

        with open(
            occurrence_csv,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            fieldnames = list(
                occurrence_rows[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                occurrence_rows
            )


    summary_csv = (
        OUTPUT_ROOT
        / "character_statistics.csv"
    )


    if character_rows:

        with open(
            summary_csv,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            fieldnames = list(
                character_rows[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                character_rows
            )


    writer_csv = (
        OUTPUT_ROOT
        / "writer_character_statistics.csv"
    )


    if writer_character_rows:

        with open(
            writer_csv,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            fieldnames = list(
                writer_character_rows[0].keys()
            )

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                writer_character_rows
            )


    # ========================================================
    # PLOT 1: CHARACTER POINT COUNTS
    # ========================================================

    if CREATE_IMAGES:

        print(
            "Creating character plots..."
        )


        sorted_rows = sorted(
            character_rows,
            key=lambda row: row["total_points"],
            reverse=True
        )


        top_rows = sorted_rows[:40]


        labels = [
            row["display_character"]
            for row in top_rows
        ]


        values = [
            row["total_points"]
            for row in top_rows
        ]


        plt.figure(
            figsize=(14, 7)
        )


        plt.bar(
            range(len(labels)),
            values
        )


        plt.xticks(
            range(len(labels)),
            labels,
            rotation=45,
            ha="right"
        )


        plt.xlabel(
            "Character"
        )


        plt.ylabel(
            "Total trajectory points"
        )


        plt.title(
            "Top Characters by Trajectory Points"
        )


        plt.tight_layout()


        plt.savefig(
            OUTPUT_ROOT
            / "character_point_totals.png",
            dpi=150
        )


        plt.close()


    # ========================================================
    # PLOT 2: AVERAGE POINTS PER CHARACTER
    # ========================================================

    if CREATE_IMAGES:

        sorted_rows = sorted(
            character_rows,
            key=lambda row: row["mean_points_when_drawn"],
            reverse=True
        )


        top_rows = sorted_rows[:40]


        labels = [
            row["display_character"]
            for row in top_rows
        ]


        values = [
            row["mean_points_when_drawn"]
            for row in top_rows
        ]


        plt.figure(
            figsize=(14, 7)
        )


        plt.bar(
            range(len(labels)),
            values
        )


        plt.xticks(
            range(len(labels)),
            labels,
            rotation=45,
            ha="right"
        )


        plt.xlabel(
            "Character"
        )


        plt.ylabel(
            "Average trajectory points"
        )


        plt.title(
            "Characters by Average Trajectory Length"
        )


        plt.tight_layout()


        plt.savefig(
            OUTPUT_ROOT
            / "character_average_points.png",
            dpi=150
        )


        plt.close()


    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {

        "writers":
            len(writers),

        "samples":
            total_samples,

        "unique_characters":
            len(character_rows),

        "character_occurrences":
            total_character_occurrences,

        "drawn_character_occurrences":
            total_drawn_occurrences,

        "undrawn_character_occurrences":
            (
                total_character_occurrences
                - total_drawn_occurrences
            ),

        "examples_per_character":
            EXAMPLES_PER_CHARACTER,

        "dataset":
            str(DATASET_ROOT),

    }


    with open(
        OUTPUT_ROOT / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 75)
    print("CHARACTER ANALYSIS COMPLETE")
    print("=" * 75)
    print()


    print(
        f"Writers processed:              "
        f"{len(writers)}"
    )


    print(
        f"Samples processed:              "
        f"{total_samples:,}"
    )


    print(
        f"Unique characters:              "
        f"{len(character_rows)}"
    )


    print(
        f"Character occurrences:          "
        f"{total_character_occurrences:,}"
    )


    print(
        f"Drawn character occurrences:    "
        f"{total_drawn_occurrences:,}"
    )


    print(
        f"Undrawn occurrences:            "
        f"{total_character_occurrences - total_drawn_occurrences:,}"
    )


    print()
    print(
        "Output:"
    )


    print(
        f"  {OUTPUT_ROOT}"
    )


    print()
    print(
        "Files:"
    )


    print(
        "  character_statistics.csv"
    )


    print(
        "  character_occurrences.csv"
    )


    print(
        "  writer_character_statistics.csv"
    )


    print(
        "  summary.json"
    )


    if CREATE_IMAGES:

        print(
            "  character_point_totals.png"
        )

        print(
            "  character_average_points.png"
        )

        print(
            "  examples/"
        )


    print()
    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()