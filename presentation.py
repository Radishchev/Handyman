from pathlib import Path
import json
import random
import subprocess
import sys

import numpy as np

from manim import *


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

BRUSH_JSON = ROOT / "BRUSH_JSON"
PRESENTATION = ROOT / "Presentation"

FIRST_MODEL_OUTPUT = ROOT / "hello_world_writer_1.png"
LATEST_MODEL_OUTPUT = ROOT / "GENERATED" / "hello_world_writer_1.png"

PRESENTATION.mkdir(exist_ok=True)


# ============================================================
# STYLE
# ============================================================

BG = "#050505"
WHITE = "#F5F5F5"
GRAY = "#9A9A9A"
LIGHT_GRAY = "#C8C8C8"
ACCENT = "#7FDBFF"
GREEN = "#8BE28B"
RED = "#FF7777"
YELLOW = "#FFD866"

FONT = "DejaVu Sans"


# ============================================================
# GENERAL HELPERS
# ============================================================

def title(text, size=42):
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=WHITE,
    )


def subtitle(text, size=25):
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=GRAY,
    )


def small_text(text, size=20, color=LIGHT_GRAY):
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=color,
    )


def handwritten_text(text, size=42):
    """
    Uses Manim's Write animation so the text appears progressively,
    giving the presentation a hand-written feel.
    """
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=WHITE,
    )


def load_json_sample():
    """
    Find one usable BRUSH JSON sample.
    """
    if not BRUSH_JSON.exists():
        return None

    files = sorted(BRUSH_JSON.glob("*/*.json"))

    if not files:
        files = sorted(BRUSH_JSON.glob("**/*.json"))

    if not files:
        return None

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if (
                "sentence" in data
                and "drawing" in data
                and "labels" in data
            ):
                return data, path

        except Exception:
            continue

    return None


def dataset_stats():
    """
    Calculate a few lightweight statistics directly from BRUSH_JSON.
    """
    if not BRUSH_JSON.exists():
        return {
            "writers": 0,
            "samples": 0,
            "points": 0,
            "strokes": 0,
        }

    writers = []
    samples = 0
    points = 0
    strokes = 0

    for writer_dir in sorted(BRUSH_JSON.iterdir()):
        if not writer_dir.is_dir():
            continue

        json_files = list(writer_dir.glob("*.json"))

        if not json_files:
            continue

        writers.append(writer_dir.name)

        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                drawing = np.asarray(data["drawing"], dtype=float)

                samples += 1
                points += len(drawing)

                if len(drawing):
                    strokes += int(np.sum(drawing[:, 2] == 1))

            except Exception:
                continue

    return {
        "writers": len(writers),
        "samples": samples,
        "points": points,
        "strokes": strokes,
    }


def trajectory_mobject(
    drawing,
    scale=0.015,
    max_points=700,
    color=WHITE,
):
    """
    Convert BRUSH [x,y,eos] points into a Manim VMobject.

    EOS=1 means the current stroke ends.
    Separate strokes are represented as separate VMobjects.
    """

    drawing = np.asarray(drawing, dtype=float)

    if len(drawing) == 0:
        return VGroup()

    if len(drawing) > max_points:
        indices = np.linspace(
            0,
            len(drawing) - 1,
            max_points,
            dtype=int,
        )
        drawing = drawing[indices]

    xs = drawing[:, 0]
    ys = drawing[:, 1]

    center_x = (xs.min() + xs.max()) / 2
    center_y = (ys.min() + ys.max()) / 2

    strokes = []
    current = []

    for i, point in enumerate(drawing):
        x, y, eos = point

        px = (x - center_x) * scale
        py = -(y - center_y) * scale

        current.append(np.array([px, py, 0]))

        if eos == 1 or i == len(drawing) - 1:
            if len(current) >= 2:
                vm = VMobject()
                vm.set_points_smoothly(current)
                vm.set_stroke(
                    color=color,
                    width=2.5,
                )
                strokes.append(vm)

            current = []

    return VGroup(*strokes)


def image_with_backing(path, width=8.5):
    """
    ImageMobject must be inside Group, not VGroup.
    """

    img = ImageMobject(str(path))
    img.set_width(width)

    backing = Rectangle(
        width=img.width + 0.25,
        height=img.height + 0.25,
        stroke_width=1,
        stroke_color=GRAY,
        fill_opacity=0,
    )

    # ImageMobject is not a VMobject, therefore Group is required.
    group = Group(backing, img)

    return group


# ============================================================
# VIDEO 01
# TITLE
# ============================================================

class Video01Title(Scene):

    def construct(self):

        self.camera.background_color = BG

        word = handwritten_text(
            "handyman",
            size=72,
        )

        self.play(
            Write(word),
            run_time=3,
        )

        self.wait(1)


# ============================================================
# VIDEO 02
# DATASET INTRODUCTION
# ============================================================

class Video02DatasetIntro(Scene):

    def construct(self):

        self.camera.background_color = BG

        stats = dataset_stats()

        heading = handwritten_text(
            "the dataset",
            size=52,
        )

        heading.to_edge(UP, buff=0.7)

        self.play(
            Write(heading),
            run_time=1.5,
        )

        lines = VGroup(
            small_text(
                "BRUSH",
                size=30,
                color=ACCENT,
            ),
            small_text(
                f"{stats['writers']:,} writers",
                size=27,
            ),
            small_text(
                f"{stats['samples']:,} handwriting samples",
                size=27,
            ),
            small_text(
                "online pen trajectories",
                size=27,
            ),
        )

        lines.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.35,
        )

        lines.move_to(ORIGIN)

        self.play(
            Write(lines[0]),
            run_time=1,
        )

        self.play(
            Write(lines[1]),
            run_time=1,
        )

        self.play(
            Write(lines[2]),
            run_time=1,
        )

        self.play(
            Write(lines[3]),
            run_time=1,
        )

        self.wait(1)


# ============================================================
# VIDEO 03
# ACTUAL BRUSH SAMPLE
# ============================================================

class Video03DatasetSample(Scene):

    def construct(self):

        self.camera.background_color = BG

        result = load_json_sample()

        heading = handwritten_text(
            "one sample",
            size=46,
        )

        heading.to_edge(UP, buff=0.55)

        self.play(
            Write(heading),
            run_time=1.2,
        )

        if result is None:

            msg = small_text(
                "BRUSH_JSON sample not found",
                size=28,
                color=RED,
            )

            self.play(
                Write(msg),
                run_time=1,
            )

            self.wait(1)
            return

        data, path = result

        sentence = data["sentence"]

        text = Text(
            sentence,
            font=FONT,
            font_size=28,
            color=ACCENT,
        )

        text.next_to(
            heading,
            DOWN,
            buff=0.45,
        )

        self.play(
            Write(text),
            run_time=1.5,
        )

        drawing = trajectory_mobject(
            data["drawing"],
            scale=0.013,
            max_points=700,
            color=WHITE,
        )

        drawing.move_to(
            DOWN * 0.8
        )

        self.play(
            Create(drawing),
            run_time=5,
        )

        self.wait(1)


# ============================================================
# VIDEO 04
# DATA REPRESENTATION
# ============================================================

class Video04DatasetRepresentation(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "what the model sees",
            size=46,
        )

        heading.to_edge(UP, buff=0.55)

        self.play(
            Write(heading),
            run_time=1.2,
        )

        items = [
            ("sentence", '"hello world"', ACCENT),
            ("point", "[x, y, eos]", WHITE),
            ("movement", "[dx, dy, eos]", GREEN),
            ("label", "character ownership", YELLOW),
        ]

        groups = []

        for left, right, color in items:

            l = Text(
                left,
                font=FONT,
                font_size=27,
                color=color,
            )

            r = Text(
                right,
                font=FONT,
                font_size=27,
                color=WHITE,
            )

            row = VGroup(l, r)
            row.arrange(
                RIGHT,
                buff=0.5,
            )

            groups.append(row)

        table = VGroup(*groups)

        table.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.4,
        )

        table.move_to(ORIGIN)

        for row in table:

            self.play(
                Write(row),
                run_time=1,
            )

        self.wait(1)


# ============================================================
# VIDEO 05
# FEATURE ENGINEERING
# ============================================================

class Video05Features(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "feature engineering",
            size=48,
        )

        heading.to_edge(UP, buff=0.55)

        self.play(
            Write(heading),
            run_time=1.2,
        )

        features = [
            ("x, y", "position"),
            ("dx, dy", "movement"),
            ("distance", "step length"),
            ("speed", "movement rate"),
            ("direction", "stroke direction"),
            ("eos", "end of stroke"),
            ("char_id", "character"),
            ("char_fraction", "position in character"),
        ]

        rows = []

        for name, meaning in features:

            a = Text(
                name,
                font=FONT,
                font_size=25,
                color=ACCENT,
            )

            b = Text(
                meaning,
                font=FONT,
                font_size=23,
                color=LIGHT_GRAY,
            )

            row = VGroup(a, b)

            row.arrange(
                RIGHT,
                buff=0.45,
            )

            rows.append(row)

        group = VGroup(*rows)

        group.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.25,
        )

        group.move_to(ORIGIN)

        # Reveal only the first several at a time.
        for row in group:

            self.play(
                Write(row),
                run_time=0.65,
            )

        core = Text(
            "[ dx, dy, eos ]",
            font=FONT,
            font_size=35,
            color=GREEN,
        )

        core.to_edge(
            DOWN,
            buff=0.45,
        )

        self.play(
            Write(core),
            run_time=1,
        )

        self.wait(1)


# ============================================================
# VIDEO 06
# MODEL 1
# ============================================================

class Video06Model1(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "model 1",
            size=50,
        )

        heading.to_edge(
            UP,
            buff=0.45,
        )

        self.play(
            Write(heading),
            run_time=1.2,
        )

        # Architecture
        boxes = []

        labels = [
            ("TEXT", ACCENT),
            ("BiGRU", WHITE),
            ("WRITER", YELLOW),
            ("TRAJECTORY", GREEN),
            ("GRU", WHITE),
            ("dx / dy / eos", ACCENT),
        ]

        for text, color in labels:

            box = RoundedRectangle(
                width=2.1,
                height=0.65,
                corner_radius=0.12,
                stroke_width=1.5,
                stroke_color=color,
            )

            label = Text(
                text,
                font=FONT,
                font_size=19,
                color=color,
            )

            label.move_to(box)

            boxes.append(
                VGroup(box, label)
            )

        row = VGroup(*boxes)

        row.arrange(
            RIGHT,
            buff=0.28,
        )

        row.scale(0.95)

        row.move_to(
            ORIGIN + UP * 0.45
        )

        self.play(
            LaggedStart(
                *[Write(b) for b in boxes],
                lag_ratio=0.15,
            ),
            run_time=3,
        )

        # arrows
        arrows = VGroup()

        for i in range(len(boxes) - 1):

            start = boxes[i].get_right()
            end = boxes[i + 1].get_left()

            arrow = Arrow(
                start,
                end,
                buff=0.08,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.25,
            )

            arrows.add(arrow)

        self.play(
            Create(arrows),
            run_time=1.5,
        )

        note = small_text(
            "deterministic regression + teacher forcing",
            size=22,
            color=GRAY,
        )

        note.to_edge(
            DOWN,
            buff=0.5,
        )

        self.play(
            Write(note),
            run_time=1,
        )

        self.wait(1)


# ============================================================
# VIDEO 07
# MODEL 1 FAILURE
# ============================================================

class Video07Model1Failure(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "model 1 — result",
            size=46,
        )

        heading.to_edge(
            UP,
            buff=0.4,
        )

        self.play(
            Write(heading),
            run_time=1.2,
        )

        if not FIRST_MODEL_OUTPUT.exists():

            msg = small_text(
                "hello_world_writer_1.png not found",
                size=25,
                color=RED,
            )

            self.play(
                Write(msg),
                run_time=1,
            )

            self.wait(1)
            return

        image = image_with_backing(
            FIRST_MODEL_OUTPUT,
            width=8.4,
        )

        image.move_to(
            DOWN * 0.35
        )

        self.play(
            FadeIn(image),
            run_time=1.5,
        )

        failure = Text(
            "drift  •  averaged motion  •  weak alignment",
            font=FONT,
            font_size=22,
            color=RED,
        )

        failure.to_edge(
            DOWN,
            buff=0.35,
        )

        self.play(
            Write(failure),
            run_time=1.2,
        )

        self.wait(1)


# ============================================================
# VIDEO 08
# GRAVES-STYLE MODEL
# ============================================================

class Video08Model2(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "new model",
            size=50,
        )

        heading.to_edge(
            UP,
            buff=0.45,
        )

        self.play(
            Write(heading),
            run_time=1.2,
        )

        labels = [
            ("TEXT", ACCENT),
            ("BiGRU", WHITE),
            ("WINDOW", YELLOW),
            ("LSTM 1", GREEN),
            ("LSTM 2", GREEN),
            ("MDN", ACCENT),
            ("SAMPLE", WHITE),
        ]

        boxes = []

        for text, color in labels:

            box = RoundedRectangle(
                width=1.65,
                height=0.65,
                corner_radius=0.12,
                stroke_width=1.5,
                stroke_color=color,
            )

            label = Text(
                text,
                font=FONT,
                font_size=18,
                color=color,
            )

            label.move_to(box)

            boxes.append(
                VGroup(box, label)
            )

        row = VGroup(*boxes)

        row.arrange(
            RIGHT,
            buff=0.22,
        )

        row.scale(0.94)

        row.move_to(
            UP * 0.55
        )

        self.play(
            LaggedStart(
                *[Write(b) for b in boxes],
                lag_ratio=0.13,
            ),
            run_time=3,
        )

        arrows = VGroup()

        for i in range(len(boxes) - 1):

            arrows.add(
                Arrow(
                    boxes[i].get_right(),
                    boxes[i + 1].get_left(),
                    buff=0.06,
                    stroke_width=1.4,
                    max_tip_length_to_length_ratio=0.25,
                )
            )

        self.play(
            Create(arrows),
            run_time=1.5,
        )

        explanation = VGroup(
            small_text(
                "monotonic attention",
                size=23,
                color=YELLOW,
            ),
            small_text(
                "mixture density network",
                size=23,
                color=ACCENT,
            ),
            small_text(
                "stochastic trajectory generation",
                size=23,
                color=GREEN,
            ),
        )

        explanation.arrange(
            DOWN,
            buff=0.25,
        )

        explanation.move_to(
            DOWN * 0.8
        )

        for item in explanation:

            self.play(
                Write(item),
                run_time=0.8,
            )

        self.wait(1)


# ============================================================
# VIDEO 09
# MODEL 2 RESULT
# ============================================================

class Video09Model2Result(Scene):

    def construct(self):

        self.camera.background_color = BG

        heading = handwritten_text(
            "new model — result",
            size=46,
        )

        heading.to_edge(
            UP,
            buff=0.4,
        )

        self.play(
            Write(heading),
            run_time=1.2,
        )

        if not LATEST_MODEL_OUTPUT.exists():

            msg = small_text(
                "latest model output not found",
                size=25,
                color=RED,
            )

            self.play(
                Write(msg),
                run_time=1,
            )

            self.wait(1)
            return

        image = image_with_backing(
            LATEST_MODEL_OUTPUT,
            width=8.4,
        )

        image.move_to(
            DOWN * 0.35
        )

        self.play(
            FadeIn(image),
            run_time=1.5,
        )

        note = Text(
            "better stochastic modeling — but global drift remains",
            font=FONT,
            font_size=22,
            color=YELLOW,
        )

        note.to_edge(
            DOWN,
            buff=0.35,
        )

        self.play(
            Write(note),
            run_time=1.2,
        )

        self.wait(1)


# ============================================================
# VIDEO 10
# THANK YOU
# ============================================================

class Video10ThankYou(Scene):

    def construct(self):

        self.camera.background_color = BG

        thank = handwritten_text(
            "thank you",
            size=68,
        )

        self.play(
            Write(thank),
            run_time=2.5,
        )

        self.wait(1)


# ============================================================
# RENDERING
# ============================================================

SCENES = [
    ("01_title", "Video01Title"),
    ("02_dataset_intro", "Video02DatasetIntro"),
    ("03_dataset_sample", "Video03DatasetSample"),
    ("04_dataset_representation", "Video04DatasetRepresentation"),
    ("05_features", "Video05Features"),
    ("06_model1", "Video06Model1"),
    ("07_model1_result", "Video07Model1Failure"),
    ("08_model2", "Video08Model2"),
    ("09_model2_result", "Video09Model2Result"),
    ("10_thank_you", "Video10ThankYou"),
]


def render_scene(scene_name):

    print()
    print("=" * 70)
    print(f"RENDERING: {scene_name}")
    print("=" * 70)

    command = [
        sys.executable,
        "-m",
        "manim",
        "-qh",
        "--media_dir",
        str(PRESENTATION),
        str(Path(__file__).resolve()),
        scene_name,
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print()
        print(f"FAILED: {scene_name}")
        print()
        return False

    print()
    print(f"COMPLETED: {scene_name}")
    print()

    return True


def main():

    print()
    print("=" * 70)
    print("HANDYMAN PRESENTATION")
    print("=" * 70)
    print()
    print("This will render 10 separate videos.")
    print()
    print("Canva order:")
    print()

    for number, scene in SCENES:
        print(f"  {number}.mp4")

    print()
    print("=" * 70)
    print()

    failed = []

    for output_name, scene_name in SCENES:

        success = render_scene(scene_name)

        if not success:
            failed.append(scene_name)

    print()
    print("=" * 70)
    print("RENDERING COMPLETE")
    print("=" * 70)
    print()

    if failed:

        print("FAILED SCENES:")
        for scene in failed:
            print(f"  - {scene}")

        print()

    else:

        print("All scenes rendered successfully.")
        print()

    print("Manim files are inside:")
    print(PRESENTATION)
    print()

    print("IMPORTANT:")
    print(
        "Manim may place the actual MP4s inside Presentation/videos/..."
    )
    print(
        "The filenames correspond to the numbered scenes."
    )

    print()


if __name__ == "__main__":
    main()