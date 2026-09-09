"""
HANDYMAN — minimal Manim presentation
=====================================

Place this ONE file directly in:

    /home/aurlin/Projects/Handyman/handyman_presentation.py

Run all six slide videos:

    cd /home/aurlin/Projects/Handyman
    python handyman_presentation.py

Output:

    /home/aurlin/Projects/Handyman/Presentation/

The presentation is intentionally minimal:
- black background
- white / accent text
- Write() / Create() animations
- very little text
- actual BRUSH_JSON data
- actual model output screenshots
- one independent MP4 per slide

Slides:
    1. Title
    2. Dataset
    3. Feature Engineering
    4. Model 1 + its output
    5. Model 2 + its output + current issue
    6. Thank You

The two model-output paths are the paths supplied for this project.
"""

from manim import *
from pathlib import Path
import json
import sys
import subprocess
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

BRUSH_JSON = ROOT / "BRUSH_JSON"
PRESENTATION = ROOT / "Presentation"
PRESENTATION.mkdir(exist_ok=True)

# User-supplied model outputs
FIRST_MODEL_OUTPUT = ROOT / "hello_world_writer_1.png"
LATEST_MODEL_OUTPUT = ROOT / "GENERATED" / "hello_world_writer_1.png"


# ============================================================
# VISUAL STYLE
# ============================================================

BLACK = "#050505"
WHITE = "#F5F5F5"
GRAY = "#8A8A8A"
BLUE = "#4EA1FF"
GREEN = "#57D68D"
ORANGE = "#FF9D42"
RED = "#FF5C5C"
PURPLE = "#B68CFF"
CYAN = "#55DDE0"

# Handwriting-like accent palette for trajectory points.
TRAJ_COLORS = [
    RED, ORANGE, GREEN, BLUE,
    PURPLE, CYAN, "#FF69B4"
]


def setup(scene):
    scene.camera.background_color = BLACK


def write_text(scene, text, size=36, color=WHITE, run_time=1.0, **kwargs):
    obj = Text(
        text,
        font_size=size,
        color=color,
        **kwargs
    )
    scene.play(
        Write(obj, run_time=run_time)
    )
    return obj


def make_heading(text, subtitle=None):
    h = Text(
        text,
        font_size=36,
        weight=BOLD,
        color=WHITE
    )

    if subtitle is None:
        return h

    s = Text(
        subtitle,
        font_size=17,
        color=GRAY
    )

    return VGroup(
        h,
        s
    ).arrange(
        DOWN,
        aligned_edge=LEFT,
        buff=0.10
    )


def load_json_files():
    if not BRUSH_JSON.exists():
        return []

    return sorted(
        BRUSH_JSON.rglob("*.json")
    )


def load_first_json():
    files = load_json_files()

    if not files:
        return None, None

    path = files[0]

    try:
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f)
    except Exception:
        return path, None


def brush_stats():
    files = load_json_files()

    writers = set()
    total_points = 0
    total_strokes = 0
    point_counts = []

    for path in files:

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            relative = path.relative_to(
                BRUSH_JSON
            )

            if relative.parts:
                writers.add(
                    relative.parts[0]
                )

            drawing = data.get(
                "drawing",
                []
            )

            n = len(drawing)

            total_points += n
            point_counts.append(n)

            total_strokes += sum(
                1
                for row in drawing
                if len(row) >= 3
                and float(row[2]) == 1.0
            )

        except Exception:
            pass

    # Known project totals are used if the JSON directory
    # is temporarily unavailable.
    if not files:
        return {
            "samples": 27649,
            "writers": 170,
            "points": 7651614,
            "strokes": 532795,
            "mean_points": 276.74,
            "mean_strokes": 19.27,
        }

    return {
        "samples": len(files),
        "writers": len(writers) or 170,
        "points": total_points,
        "strokes": total_strokes,
        "mean_points": (
            np.mean(point_counts)
            if point_counts
            else 276.74
        ),
        "mean_strokes": (
            total_strokes / len(files)
            if files
            else 19.27
        ),
    }


def trajectory_mobject(
    data,
    width=5.6,
    height=2.5,
    max_points=260
):
    """
    Turn one BRUSH JSON sample into a clean animated trajectory.

    Character labels are used to color consecutive character
    segments when available.
    """

    drawing = np.asarray(
        data.get("drawing", []),
        dtype=float
    )

    labels = np.asarray(
        data.get(
            "labels",
            data.get("label", [])
        ),
        dtype=float
    )

    if len(drawing) < 2:
        return VGroup()

    original_n = len(drawing)

    if len(drawing) > max_points:

        indices = np.linspace(
            0,
            len(drawing) - 1,
            max_points
        ).astype(int)

        drawing = drawing[indices]

        if len(labels) == original_n:
            labels = labels[indices]

    points = drawing[:, :2].copy()

    # Normalize to fit.
    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].min()

    max_x = max(
        float(points[:, 0].max()),
        1.0
    )

    max_y = max(
        float(points[:, 1].max()),
        1.0
    )

    scale = min(
        width / max_x,
        height / max_y
    )

    points *= scale
    points[:, 1] *= -1

    points[:, 0] -= points[:, 0].mean()
    points[:, 1] -= points[:, 1].mean()

    result = VGroup()

    if len(labels) == len(points):

        character_ids = np.argmax(
            labels,
            axis=1
        )

        start = 0

        for i in range(
            1,
            len(points) + 1
        ):

            boundary = (
                i == len(points)
                or character_ids[i] != character_ids[start]
            )

            if boundary:

                if i - start >= 2:

                    path = VMobject()

                    path.set_points_as_corners(
                        [
                            np.array(
                                [x, y, 0]
                            )
                            for x, y in points[start:i]
                        ]
                    )

                    path.set_stroke(
                        color=TRAJ_COLORS[
                            int(character_ids[start])
                            % len(TRAJ_COLORS)
                        ],
                        width=3.2
                    )

                    result.add(path)

                start = i

    else:

        path = VMobject()

        path.set_points_as_corners(
            [
                np.array([x, y, 0])
                for x, y in points
            ]
        )

        path.set_stroke(
            color=BLUE,
            width=3
        )

        result.add(path)

    return result


def show_image(scene, path, width=5.3):
    """
    Put an actual project screenshot on the black slide.
    """
    if not path.exists():
        return None

    try:
        img = ImageMobject(str(path))
        img.width = width

        # White backing makes screenshots readable against black.
        backing = SurroundingRectangle(
            img,
            color=WHITE,
            stroke_width=1.5,
            buff=0.05
        )

        group = VGroup(
            backing,
            img
        )

        scene.play(
            FadeIn(group),
            run_time=0.8
        )

        return group

    except Exception:
        return None


# ============================================================
# SLIDE 1 — TITLE
# ============================================================

class TitleSlide(Scene):

    def construct(self):

        setup(self)

        word = Text(
            "handyman",
            font_size=86,
            weight=BOLD,
            color=WHITE
        )

        underline = Line(
            LEFT * 3.7,
            RIGHT * 3.7,
            color=BLUE,
            stroke_width=4
        )

        # The title itself is drawn onto the screen.
        self.play(
            Write(
                word,
                run_time=2.0
            )
        )

        self.play(
            Create(
                underline,
                run_time=0.8
            )
        )

        self.wait(3)


# ============================================================
# SLIDE 2 — DATASET
# ============================================================

class DatasetSlide(Scene):

    def construct(self):

        setup(self)

        heading = make_heading(
            "01  BRUSH",
            "The handwriting dataset"
        )

        heading.to_corner(
            UL,
            buff=0.55
        )

        self.play(
            Write(heading),
            run_time=1.1
        )

        stats = brush_stats()

        # Keep statistics large and sparse.
        stat_group = VGroup(
            Text(
                f"{stats['samples']:,}",
                font_size=34,
                weight=BOLD,
                color=BLUE
            ),
            Text(
                "samples",
                font_size=14,
                color=GRAY
            ),
            Text(
                f"{stats['writers']}",
                font_size=34,
                weight=BOLD,
                color=GREEN
            ),
            Text(
                "writers",
                font_size=14,
                color=GRAY
            ),
            Text(
                "86",
                font_size=34,
                weight=BOLD,
                color=ORANGE
            ),
            Text(
                "characters",
                font_size=14,
                color=GRAY
            )
        )

        # Pair value and label.
        stats_display = VGroup()

        for i in range(0, 6, 2):
            pair = VGroup(
                stat_group[i],
                stat_group[i + 1]
            ).arrange(
                DOWN,
                buff=0.04
            )

            stats_display.add(pair)

        stats_display.arrange(
            RIGHT,
            buff=1.0
        )

        stats_display.to_edge(
            UP,
            buff=1.55
        )

        self.play(
            LaggedStart(
                *[
                    FadeIn(x, shift=UP * 0.15)
                    for x in stats_display
                ],
                lag_ratio=0.15
            ),
            run_time=1.2
        )

        # Actual JSON example.
        path, data = load_first_json()

        json_title = Text(
            "one JSON sample",
            font_size=19,
            weight=BOLD,
            color=WHITE
        )

        json_title.move_to(
            LEFT * 3.7 + DOWN * 0.45
        )

        self.play(
            Write(json_title),
            run_time=0.7
        )

        if data is not None:

            sentence = str(
                data.get(
                    "sentence",
                    ""
                )
            )

            drawing = data.get(
                "drawing",
                []
            )

            labels = data.get(
                "labels",
                data.get(
                    "label",
                    []
                )
            )

            sample_text = VGroup(
                Text(
                    f'sentence: "{sentence[:30]}"',
                    font_size=15,
                    color=GREEN
                ),
                Text(
                    f"drawing: {len(drawing)} × 3",
                    font_size=15,
                    color=BLUE
                ),
                Text(
                    f"labels: {len(labels)} × {len(labels[0]) if labels else 0}",
                    font_size=15,
                    color=PURPLE
                ),
            ).arrange(
                DOWN,
                aligned_edge=LEFT,
                buff=0.12
            )

            sample_text.move_to(
                LEFT * 3.7 + DOWN * 1.45
            )

            self.play(
                LaggedStart(
                    *[
                        Write(x)
                        for x in sample_text
                    ],
                    lag_ratio=0.15
                ),
                run_time=1.4
            )

            trajectory = trajectory_mobject(
                data,
                width=5.5,
                height=2.25
            )

            trajectory.move_to(
                RIGHT * 3.0 + DOWN * 1.1
            )

            trajectory_title = Text(
                "actual pen trajectory",
                font_size=19,
                weight=BOLD,
                color=WHITE
            )

            trajectory_title.next_to(
                trajectory,
                UP,
                buff=0.20
            )

            self.play(
                Write(trajectory_title),
                run_time=0.6
            )

            self.play(
                Create(
                    trajectory,
                    run_time=2.5
                )
            )

        source = Text(
            "BRUSH • Brown University • ECCV 2020",
            font_size=12,
            color=GRAY
        )

        source.to_edge(
            DOWN,
            buff=0.35
        )

        self.play(
            Write(source),
            run_time=0.7
        )

        self.wait(3)


# ============================================================
# SLIDE 3 — FEATURE ENGINEERING
# ============================================================

class FeatureEngineeringSlide(Scene):

    def construct(self):

        setup(self)

        heading = make_heading(
            "02  FEATURE ENGINEERING",
            "Turning raw pen motion into a model input"
        )

        heading.to_corner(
            UL,
            buff=0.55
        )

        self.play(
            Write(heading),
            run_time=1.1
        )

        # Start with the original coordinates.
        raw = Text(
            "(x, y, eos)",
            font_size=36,
            weight=BOLD,
            color=WHITE
        )

        raw.move_to(
            LEFT * 4.2 + UP * 0.9
        )

        self.play(
            Write(raw),
            run_time=0.9
        )

        raw_note = Text(
            "raw online handwriting",
            font_size=14,
            color=GRAY
        )

        raw_note.next_to(
            raw,
            DOWN,
            buff=0.15
        )

        self.play(
            Write(raw_note),
            run_time=0.5
        )

        # Derive dx/dy.
        arrow1 = Arrow(
            raw.get_right(),
            RIGHT * 0.2 + UP * 0.9,
            color=BLUE,
            buff=0.2
        )

        self.play(
            GrowArrow(arrow1),
            run_time=0.6
        )

        deltas = VGroup(
            Text(
                "dx = x[t] − x[t−1]",
                font_size=22,
                color=BLUE
            ),
            Text(
                "dy = y[t] − y[t−1]",
                font_size=22,
                color=BLUE
            )
        ).arrange(
            DOWN,
            buff=0.12
        )

        deltas.move_to(
            RIGHT * 1.25 + UP * 0.9
        )

        self.play(
            LaggedStart(
                *[
                    Write(x)
                    for x in deltas
                ],
                lag_ratio=0.15
            ),
            run_time=1.1
        )

        # Show additional derived information briefly.
        derived = Text(
            "distance • speed • direction • char_id",
            font_size=16,
            color=GRAY
        )

        derived.move_to(
            RIGHT * 1.1 + DOWN * 0.15
        )

        self.play(
            Write(derived),
            run_time=0.8
        )

        # Final training representation.
        arrow2 = Arrow(
            deltas.get_right(),
            RIGHT * 5.0 + DOWN * 0.9,
            color=GREEN,
            buff=0.25
        )

        self.play(
            GrowArrow(arrow2),
            run_time=0.7
        )

        final = Text(
            "[ dx, dy, eos ]",
            font_size=40,
            weight=BOLD,
            color=GREEN
        )

        final.move_to(
            RIGHT * 4.2 + DOWN * 0.9
        )

        self.play(
            Write(final),
            run_time=1.0
        )

        final_note = Text(
            "training representation",
            font_size=14,
            color=GRAY
        )

        final_note.next_to(
            final,
            DOWN,
            buff=0.16
        )

        self.play(
            Write(final_note),
            run_time=0.5
        )

        # EOS explanation.
        eos = Text(
            "EOS = end of stroke",
            font_size=22,
            weight=BOLD,
            color=RED
        )

        eos.move_to(
            LEFT * 3.0 + DOWN * 2.2
        )

        eos_note = Text(
            "keeps separate pen strokes separate",
            font_size=14,
            color=GRAY
        )

        eos_note.next_to(
            eos,
            DOWN,
            buff=0.12
        )

        self.play(
            Write(eos),
            run_time=0.8
        )

        self.play(
            Write(eos_note),
            run_time=0.6
        )

        scale = Text(
            "27,649 samples  •  7.65M trajectory points",
            font_size=15,
            color=GRAY
        )

        scale.to_edge(
            DOWN,
            buff=0.35
        )

        self.play(
            Write(scale),
            run_time=0.7
        )

        self.wait(3)


# ============================================================
# SLIDE 4 — MODEL 1
# ============================================================

class Model1Slide(Scene):

    def construct(self):

        setup(self)

        heading = make_heading(
            "03  MODEL 1",
            "Deterministic baseline"
        )

        heading.to_corner(
            UL,
            buff=0.55
        )

        self.play(
            Write(heading),
            run_time=1.0
        )

        # Very simple architecture.
        labels = [
            ("TEXT", BLUE),
            ("BiGRU", GREEN),
            ("Writer", PURPLE),
            ("Trajectory GRU", ORANGE),
            ("dx / dy / EOS", RED),
        ]

        nodes = VGroup()

        for label, color in labels:

            box = RoundedRectangle(
                width=1.85,
                height=0.9,
                corner_radius=0.08,
                stroke_color=color,
                stroke_width=2,
                fill_opacity=0
            )

            txt = Text(
                label,
                font_size=16,
                weight=BOLD,
                color=WHITE
            )

            txt.move_to(
                box.get_center()
            )

            nodes.add(
                VGroup(box, txt)
            )

        nodes.arrange(
            RIGHT,
            buff=0.18
        )

        nodes.scale(0.82)
        nodes.to_edge(
            UP,
            buff=1.45
        )

        arrows = VGroup()

        for i in range(
            len(nodes) - 1
        ):

            arrows.add(
                Arrow(
                    nodes[i].get_right(),
                    nodes[i + 1].get_left(),
                    color=GRAY,
                    buff=0.04
                )
            )

        self.play(
            LaggedStart(
                *[
                    Write(n[1])
                    for n in nodes
                ],
                lag_ratio=0.12
            ),
            run_time=1.0
        )

        self.play(
            LaggedStart(
                *[
                    Create(a)
                    for a in arrows
                ],
                lag_ratio=0.1
            ),
            run_time=0.8
        )

        # Model result.
        output_label = Text(
            "hello world",
            font_size=22,
            weight=BOLD,
            color=WHITE
        )

        output_label.move_to(
            LEFT * 3.8 + DOWN * 0.65
        )

        self.play(
            Write(output_label),
            run_time=0.6
        )

        if FIRST_MODEL_OUTPUT.exists():

            img = ImageMobject(
                str(FIRST_MODEL_OUTPUT)
            )

            img.width = 5.7

            img.move_to(
                RIGHT * 3.2 + DOWN * 1.35
            )

            backing = SurroundingRectangle(
                img,
                color=WHITE,
                stroke_width=1,
                buff=0.04
            )

            self.play(
                FadeIn(
                    Group(
                        backing,
                        img
                    )
                ),
                run_time=0.8
            )

        else:

            missing = Text(
                "first model output not found",
                font_size=17,
                color=RED
            )

            missing.move_to(
                RIGHT * 3.2 + DOWN * 1.3
            )

            self.play(
                Write(missing),
                run_time=0.6
            )

        result = Text(
            "It produced a trajectory — but not recognizable handwriting.",
            font_size=18,
            color=RED
        )

        result.move_to(
            LEFT * 2.8 + DOWN * 1.65
        )

        self.play(
            Write(result),
            run_time=1.0
        )

        reason = Text(
            "Single deterministic prediction → averaged / unstable motion",
            font_size=15,
            color=GRAY
        )

        reason.move_to(
            LEFT * 2.4 + DOWN * 2.15
        )

        self.play(
            Write(reason),
            run_time=0.8
        )

        self.wait(3)


# ============================================================
# SLIDE 5 — MODEL 2
# ============================================================

class Model2Slide(Scene):

    def construct(self):

        setup(self)

        heading = make_heading(
            "04  MODEL 2",
            "Graves-style probabilistic handwriting synthesis"
        )

        heading.to_corner(
            UL,
            buff=0.55
        )

        self.play(
            Write(heading),
            run_time=1.0
        )

        # Architecture, kept sparse.
        labels = [
            ("Text\nBiGRU", BLUE),
            ("Gaussian\nwindow", GREEN),
            ("LSTM 1", PURPLE),
            ("LSTM 2", ORANGE),
            ("MDN\n20 mixtures", RED),
        ]

        nodes = VGroup()

        for label, color in labels:

            box = RoundedRectangle(
                width=1.8,
                height=0.95,
                corner_radius=0.08,
                stroke_color=color,
                stroke_width=2,
                fill_opacity=0
            )

            txt = Text(
                label,
                font_size=14,
                weight=BOLD,
                color=WHITE
            )

            txt.move_to(
                box.get_center()
            )

            nodes.add(
                VGroup(box, txt)
            )

        nodes.arrange(
            RIGHT,
            buff=0.17
        )

        nodes.scale(0.82)

        nodes.to_edge(
            UP,
            buff=1.45
        )

        arrows = VGroup()

        for i in range(
            len(nodes) - 1
        ):

            arrows.add(
                Arrow(
                    nodes[i].get_right(),
                    nodes[i + 1].get_left(),
                    color=GRAY,
                    buff=0.04
                )
            )

        self.play(
            LaggedStart(
                *[
                    Write(n[1])
                    for n in nodes
                ],
                lag_ratio=0.12
            ),
            run_time=1.0
        )

        self.play(
            LaggedStart(
                *[
                    Create(a)
                    for a in arrows
                ],
                lag_ratio=0.1
            ),
            run_time=0.8
        )

        # Three key ideas only.
        ideas = VGroup(
            Text(
                "alignment",
                font_size=19,
                weight=BOLD,
                color=GREEN
            ),
            Text(
                "writer style",
                font_size=19,
                weight=BOLD,
                color=PURPLE
            ),
            Text(
                "multiple possible motions",
                font_size=19,
                weight=BOLD,
                color=ORANGE
            )
        ).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.12
        )

        ideas.move_to(
            LEFT * 3.7 + DOWN * 1.2
        )

        self.play(
            LaggedStart(
                *[
                    Write(x)
                    for x in ideas
                ],
                lag_ratio=0.18
            ),
            run_time=1.2
        )

        # Latest actual output.
        if LATEST_MODEL_OUTPUT.exists():

            img = ImageMobject(
                str(LATEST_MODEL_OUTPUT)
            )

            img.width = 5.6

            img.move_to(
                RIGHT * 3.1 + DOWN * 1.35
            )

            backing = SurroundingRectangle(
                img,
                color=WHITE,
                stroke_width=1,
                buff=0.04
            )

            self.play(
                FadeIn(
                    Group(
                        backing,
                        img
                    )
                ),
                run_time=0.9
            )

        else:

            missing = Text(
                "latest model output not found",
                font_size=17,
                color=RED
            )

            missing.move_to(
                RIGHT * 3.1 + DOWN * 1.3
            )

            self.play(
                Write(missing),
                run_time=0.6
            )

        best = Text(
            "Best validation loss: 1.4252",
            font_size=18,
            weight=BOLD,
            color=GREEN
        )

        best.move_to(
            LEFT * 3.0 + DOWN * 2.15
        )

        self.play(
            Write(best),
            run_time=0.7
        )

        problem = Text(
            "Still drifts vertically / diagonally.",
            font_size=18,
            weight=BOLD,
            color=RED
        )

        problem.move_to(
            RIGHT * 2.8 + DOWN * 2.75
        )

        self.play(
            Write(problem),
            run_time=0.8
        )

        paper = Text(
            "Inspired by Graves (2013)",
            font_size=12,
            color=GRAY
        )

        paper.to_edge(
            DOWN,
            buff=0.30
        )

        self.play(
            Write(paper),
            run_time=0.5
        )

        self.wait(3)


# ============================================================
# SLIDE 6 — THANK YOU
# ============================================================

class ThankYouSlide(Scene):

    def construct(self):

        setup(self)

        word = Text(
            "thank you",
            font_size=72,
            weight=BOLD,
            color=WHITE
        )

        underline = Line(
            LEFT * 2.5,
            RIGHT * 2.5,
            color=BLUE,
            stroke_width=4
        )

        subtitle = Text(
            "handyman",
            font_size=20,
            color=GRAY
        )

        group = VGroup(
            word,
            underline,
            subtitle
        ).arrange(
            DOWN,
            buff=0.22
        )

        self.play(
            Write(word),
            run_time=1.5
        )

        self.play(
            Create(underline),
            run_time=0.7
        )

        self.play(
            Write(subtitle),
            run_time=0.7
        )

        self.wait(3)


# ============================================================
# RENDERING
# ============================================================

SCENES = [
    "TitleSlide",
    "DatasetSlide",
    "FeatureEngineeringSlide",
    "Model1Slide",
    "Model2Slide",
    "ThankYouSlide",
]


def render_all():

    print()
    print("=" * 65)
    print("HANDYMAN — MANIM PRESENTATION")
    print("=" * 65)
    print()

    print(f"Project:       {ROOT}")
    print(f"BRUSH_JSON:    {BRUSH_JSON}")
    print(f"Presentation:  {PRESENTATION}")
    print()

    stats = brush_stats()

    print("BRUSH statistics:")
    print(f"  Samples:        {stats['samples']:,}")
    print(f"  Writers:        {stats['writers']}")
    print(f"  Points:         {stats['points']:,}")
    print(f"  Strokes:        {stats['strokes']:,}")
    print(f"  Mean points:    {stats['mean_points']:.2f}")
    print(f"  Mean strokes:   {stats['mean_strokes']:.2f}")
    print()

    print("Model outputs:")
    print(f"  Model 1: {FIRST_MODEL_OUTPUT}")
    print(f"  Model 2: {LATEST_MODEL_OUTPUT}")
    print()

    for scene_name in SCENES:

        print("-" * 65)
        print(f"Rendering {scene_name}")
        print("-" * 65)

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

        result = subprocess.run(
            command
        )

        if result.returncode != 0:

            print()
            print(
                f"FAILED: {scene_name}"
            )
            sys.exit(
                result.returncode
            )

    print()
    print("=" * 65)
    print("ALL SLIDES RENDERED")
    print("=" * 65)
    print()
    print(
        f"Find the MP4 files inside:\n{PRESENTATION}"
    )
    print()
    print(
        "Run this to list them:"
    )
    print(
        f"find '{PRESENTATION}' -name '*.mp4'"
    )
    print()


if __name__ == "__main__":
    render_all()
