from manim import *
import pickle
import numpy as np
from pathlib import Path


# ============================================================
# BRUSH DATASET CONFIGURATION
# ============================================================

# Expected project structure:
#
# Handyman/
# ├── brush_animation.py
# └── BRUSH/
#     ├── 0/
#     │   ├── 0
#     │   ├── 1
#     │   ├── 2
#     │   ├── ...
#     │   ├── 0_resample20
#     │   ├── 0_resample25
#     │   └── ...
#     ├── 1/
#     ├── 2/
#     └── ...
#
# This script visualizes WRITER 0.


PROJECT_ROOT = Path(__file__).resolve().parent

BRUSH_ROOT = PROJECT_ROOT / "BRUSH"

WRITER_ID = 2

WRITER_FOLDER = BRUSH_ROOT / str(WRITER_ID)


# ============================================================
# ANIMATION SETTINGS
# ============================================================

# Approximate amount of time spent drawing each sample.
#
# Increase this if you want to study the handwriting more slowly.
#
# Example:
#
# 1.0 = fast
# 2.0 = normal
# 4.0 = slow
#
SECONDS_PER_SAMPLE = 4


# Pause after each completed sample.
PAUSE_AFTER_SAMPLE = 0.5


# Pause at the beginning.
INTRO_TIME = 1.0


# Pause at the end.
ENDING_TIME = 2.0


# Give each character a different color.
COLOR_CHARACTERS = True


# ============================================================
# CHARACTER COLORS
# ============================================================

CHARACTER_COLORS = [
    RED,
    BLUE,
    GREEN,
    ORANGE,
    PURPLE,
    TEAL,
    YELLOW,
    PINK,
    MAROON,
    GOLD,
    DARK_BLUE,
    GREY,
    WHITE,
    RED_E,
    BLUE_E,
    GREEN_E,
    ORANGE,
    PURPLE_E,
    TEAL_E,
    YELLOW_E,
]


# ============================================================
# FIND ORIGINAL BRUSH FILES
# ============================================================

def find_original_files():

    """
    Find only the original BRUSH files.

    We want:

        0
        1
        2
        3
        ...

    We do NOT want:

        0_resample20
        0_resample25
        1_resample20
        1_resample25
        ...

    Checking file.name.isdigit() handles this automatically.
    """

    if not BRUSH_ROOT.exists():

        raise FileNotFoundError(
            "\nBRUSH directory was not found.\n\n"
            f"Expected:\n"
            f"{BRUSH_ROOT}\n\n"
            "Your project should look like:\n\n"
            "Handyman/\n"
            "├── brush_animation.py\n"
            "└── BRUSH/\n"
            "    ├── 0/\n"
            "    ├── 1/\n"
            "    └── ...\n"
        )


    if not WRITER_FOLDER.exists():

        raise FileNotFoundError(
            "\nWriter directory was not found.\n\n"
            f"Expected:\n"
            f"{WRITER_FOLDER}\n\n"
            f"BRUSH root:\n"
            f"{BRUSH_ROOT}\n"
        )


    files = []


    for file_path in WRITER_FOLDER.iterdir():

        # Original BRUSH files have names like:
        #
        # 0
        # 1
        # 2
        #
        # Resampled files have names like:
        #
        # 0_resample20
        # 0_resample25
        #
        # Therefore only numeric filenames are accepted.

        if (
            file_path.is_file()
            and file_path.name.isdigit()
        ):

            files.append(file_path)


    # Sort numerically.
    #
    # Without this:
    #
    # 0, 1, 10, 11, 2, 3
    #
    # With this:
    #
    # 0, 1, 2, 3, ..., 10, 11

    files.sort(
        key=lambda p: int(p.name)
    )


    return files


# ============================================================
# LOAD ONE BRUSH SAMPLE
# ============================================================

def load_brush_file(file_path):

    """
    Load one BRUSH pickle file.

    BRUSH stores:

        sentence
        drawing
        label

    drawing:
        N x 3

        x
        y
        eos

    labels:
        N x M

        one-hot character assignment
    """

    with open(
        file_path,
        "rb"
    ) as f:

        sentence, drawing, labels = pickle.load(f)


    drawing = np.asarray(
        drawing,
        dtype=float
    )

    labels = np.asarray(
        labels
    )


    return (
        sentence,
        drawing,
        labels
    )


# ============================================================
# CHARACTER INDEX FOR EACH TRAJECTORY POINT
# ============================================================

def get_character_indices(labels):

    """
    Convert the BRUSH N x M one-hot label matrix into:

        point 0 -> character index
        point 1 -> character index
        point 2 -> character index
        ...

    Example:

        [1,0,0,0] -> character 0
        [1,0,0,0] -> character 0
        [0,1,0,0] -> character 1
        ...

    If a row contains no label, return -1.
    """

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


    indices[~valid] = -1


    return indices


# ============================================================
# CONVERT BRUSH COORDINATES TO MANIM COORDINATES
# ============================================================

def convert_coordinates(drawing):

    """
    BRUSH coordinates behave like image coordinates:

        x increases →
        y increases ↓

    Manim coordinates:

        x increases →
        y increases ↑

    Therefore we flip Y.
    """

    x = drawing[:, 0].copy()

    y = drawing[:, 1].copy()


    # Center the handwriting around origin.

    x -= np.mean(x)

    y -= np.mean(y)


    # Flip Y.

    y = -y


    return x, y


# ============================================================
# CREATE TRAJECTORY SEGMENTS
# ============================================================

def create_segments(
    drawing,
    labels
):

    """
    Create the individual line segments that make up
    the handwriting trajectory.

    IMPORTANT:

    If:

        eos[i] == 1

    then the stroke ends at point i.

    Therefore we do NOT connect:

        point i -> point i+1

    """

    x, y = convert_coordinates(
        drawing
    )


    character_indices = (
        get_character_indices(
            labels
        )
    )


    segments = []


    for i in range(
        len(drawing) - 1
    ):

        # --------------------------------------------
        # EOS
        # --------------------------------------------

        if drawing[i, 2] == 1:

            continue


        # --------------------------------------------
        # Character
        # --------------------------------------------

        char_index = (
            character_indices[i]
        )


        if char_index < 0:

            continue


        # --------------------------------------------
        # Don't connect different characters
        # --------------------------------------------

        if (
            character_indices[i + 1]
            != char_index
        ):

            continue


        # --------------------------------------------
        # Coordinates
        # --------------------------------------------

        p1 = np.array([
            x[i],
            y[i],
            0
        ])


        p2 = np.array([
            x[i + 1],
            y[i + 1],
            0
        ])


        segments.append(
            (
                i,
                char_index,
                p1,
                p2
            )
        )


    return segments


# ============================================================
# MAIN MANIM SCENE
# ============================================================

class BRUSHWriter0(Scene):


    def construct(self):


        # ====================================================
        # FIND ALL FILES
        # ====================================================

        files = find_original_files()


        print()
        print("=" * 70)
        print("BRUSH HANDWRITING VISUALIZATION")
        print("=" * 70)
        print()
        print(f"Project root:")
        print(f"    {PROJECT_ROOT}")
        print()
        print(f"BRUSH root:")
        print(f"    {BRUSH_ROOT}")
        print()
        print(f"Writer:")
        print(f"    {WRITER_ID}")
        print()
        print(f"Writer folder:")
        print(f"    {WRITER_FOLDER}")
        print()
        print(f"Original files found:")
        print(f"    {len(files)}")
        print()
        print("Files that will be animated:")
        print()


        for file_path in files:

            print(
                f"    {file_path.name}"
            )


        print()
        print("=" * 70)
        print()


        # ====================================================
        # INTRO
        # ====================================================

        title = Text(
            "BRUSH Dataset",
            font_size=44
        )


        subtitle = Text(
            f"Writer {WRITER_ID} — "
            "original handwriting trajectories",
            font_size=25
        )


        subtitle.next_to(
            title,
            DOWN,
            buff=0.25
        )


        self.play(
            Write(title),
            Write(subtitle),
            run_time=1.0
        )


        self.wait(
            INTRO_TIME
        )


        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            run_time=0.5
        )


        # ====================================================
        # PROCESS EVERY FILE
        # ====================================================

        for sample_number, file_path in enumerate(
            files
        ):


            # =================================================
            # LOAD SAMPLE
            # =================================================

            sentence, drawing, labels = (
                load_brush_file(
                    file_path
                )
            )


            print(
                f"[{sample_number + 1}/{len(files)}] "
                f"File {file_path.name} | "
                f"{len(drawing)} points | "
                f"{len(sentence)} characters | "
                f"{sentence!r}"
            )


            # =================================================
            # HEADER
            # =================================================

            file_label = Text(
                f"Writer {WRITER_ID}   |   "
                f"Sample {file_path.name}",
                font_size=24
            )


            sentence_label = Text(
                f'"{sentence}"',
                font_size=27
            )


            file_label.to_edge(
                UP
            )


            sentence_label.next_to(
                file_label,
                DOWN,
                buff=0.15
            )


            self.play(
                FadeIn(file_label),
                FadeIn(sentence_label),
                run_time=0.25
            )


            # =================================================
            # COORDINATES
            # =================================================

            x, y = convert_coordinates(
                drawing
            )


            # =================================================
            # SCALE TO MANIM SCREEN
            # =================================================

            width = np.ptp(x)

            height = np.ptp(y)


            max_dimension = max(
                width,
                height
            )


            if max_dimension > 0:

                scale = (
                    10.5
                    / max_dimension
                )

            else:

                scale = 0.01


            x *= scale

            y *= scale


            # =================================================
            # CHARACTER LABELS
            # =================================================

            character_indices = (
                get_character_indices(
                    labels
                )
            )


            # =================================================
            # CREATE TRAJECTORY
            # =================================================

            trajectory = VGroup()


            # We keep track of which line corresponds
            # to which character.

            segments = []


            for i in range(
                len(drawing) - 1
            ):


                # ---------------------------------------------
                # EOS
                # ---------------------------------------------

                if drawing[i, 2] == 1:

                    continue


                # ---------------------------------------------
                # Character
                # ---------------------------------------------

                char_index = (
                    character_indices[i]
                )


                if char_index < 0:

                    continue


                # ---------------------------------------------
                # Don't connect different characters
                # ---------------------------------------------

                if (
                    character_indices[i + 1]
                    != char_index
                ):

                    continue


                # ---------------------------------------------
                # Create points
                # ---------------------------------------------

                p1 = np.array([
                    x[i],
                    y[i],
                    0
                ])


                p2 = np.array([
                    x[i + 1],
                    y[i + 1],
                    0
                ])


                # ---------------------------------------------
                # Create line
                # ---------------------------------------------

                line = Line(
                    p1,
                    p2
                )


                # ---------------------------------------------
                # Color
                # ---------------------------------------------

                if COLOR_CHARACTERS:

                    color = (
                        CHARACTER_COLORS[
                            char_index
                            % len(
                                CHARACTER_COLORS
                            )
                        ]
                    )

                else:

                    color = WHITE


                line.set_stroke(
                    color=color,
                    width=3
                )


                trajectory.add(
                    line
                )


                segments.append(
                    (
                        i,
                        line
                    )
                )


            # =================================================
            # REVEAL TRAJECTORY
            # =================================================
            #
            # IMPORTANT:
            #
            # We use ONE Manim animation here.
            #
            # We do NOT call:
            #
            #     self.play(...)
            #
            # thousands of times.
            #
            # This avoids the warnings you were getting about
            # 0.0029 second animations.
            # =================================================

            if len(trajectory) > 0:

                self.play(
                    Create(trajectory),
                    run_time=SECONDS_PER_SAMPLE,
                    rate_func=linear
                )


            # =================================================
            # HOLD COMPLETED SAMPLE
            # =================================================

            self.wait(
                PAUSE_AFTER_SAMPLE
            )


            # =================================================
            # REMOVE SAMPLE
            # =================================================

            self.play(
                FadeOut(
                    file_label
                ),
                FadeOut(
                    sentence_label
                ),
                FadeOut(
                    trajectory
                ),
                run_time=0.3
            )


        # ====================================================
        # END
        # ====================================================

        final_text = Text(
            f"Writer {WRITER_ID} complete",
            font_size=44
        )


        self.play(
            Write(final_text),
            run_time=1
        )


        self.wait(
            ENDING_TIME
        )