#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from handwriting_model import HandwritingModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/aurlin/Projects/Handyman")
DATA_DIR = PROJECT_ROOT / "TRAINING_DATA"
CHECKPOINT_DIR = PROJECT_ROOT / "CHECKPOINTS"
OUTPUT_DIR = PROJECT_ROOT / "GENERATED"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD VOCABULARY
# ============================================================

def load_vocabulary():
    vocabulary_file = DATA_DIR / "vocabulary.json"

    with open(vocabulary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = data["tokens"]

    return {
        char: int(idx)
        for char, idx in tokens.items()
    }


# ============================================================
# LOAD WRITERS
# ============================================================

def load_writers():
    writers_file = DATA_DIR / "writers.json"

    with open(writers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    writers = data["writers"]

    return {
        str(writer): int(idx)
        for writer, idx in writers.items()
    }


# ============================================================
# LOAD V3 MODEL
# ============================================================

def load_model(checkpoint_path):
    vocabulary = load_vocabulary()
    writers = load_writers()

    vocab_size = len(vocabulary)
    num_writers = len(writers)

    print()
    print("Loading v3 model...")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of writers: {num_writers}")

    model = HandwritingModel(
        vocab_size=vocab_size,
        num_writers=num_writers,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "val_loss" in checkpoint:
        print(
            f"Checkpoint validation loss: "
            f"{checkpoint['val_loss']:.6f}"
        )

    return model, vocabulary, writers


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(text, vocabulary):
    unk_id = vocabulary["<UNK>"]

    ids = []

    for char in text:
        if char in vocabulary:
            ids.append(vocabulary[char])
        else:
            print(
                f"Warning: character {char!r} "
                f"not in vocabulary. Using <UNK>."
            )
            ids.append(unk_id)

    if not ids:
        raise ValueError(
            "Text must contain at least one character."
        )

    return torch.tensor(
        ids,
        dtype=torch.long,
        device=DEVICE,
    )


# ============================================================
# GENERATE TRAJECTORY
# ============================================================

@torch.no_grad()
def generate_trajectory(
    model,
    text_ids,
    writer_id,
    max_points=1200,
    temperature=0.65,
    min_points=30,
    end_patience=25,
    eos_threshold=0.5,
    max_position=1000.0,
):
    """
    Generate handwriting using the v3 model.

    Returns:
        numpy array [T, 3]

    Columns:
        dx
        dy
        eos

    EOS means the current stroke ends.
    It does NOT mean the entire generation should stop.
    """

    # Model expects a batch of text IDs: [B, text_length].
    text_ids = text_ids.unsqueeze(0)

    text_mask = torch.ones(
        text_ids.shape,
        dtype=torch.bool,
        device=DEVICE,
    )

    # Model expects writer_ids: [B].
    writer_tensor = torch.tensor(
        [writer_id],
        dtype=torch.long,
        device=DEVICE,
    )

    trajectory = model.generate(
        text_ids=text_ids,
        text_mask=text_mask,
        writer_ids=writer_tensor,
        max_steps=max_points,
        temperature=temperature,
        eos_threshold=eos_threshold,
        end_patience=end_patience,
        max_position=max_position,
        min_steps=min_points,
    )

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()

    trajectory = np.asarray(
        trajectory,
        dtype=np.float32,
    )

    # The model returns [B,T,3].
    # This script generates one sample, so remove the batch dimension.
    if trajectory.ndim == 3:
        if trajectory.shape[0] != 1:
            raise ValueError(
                "Expected one generated sample, "
                f"but got shape {trajectory.shape}"
            )
        trajectory = trajectory[0]

    return trajectory


# ============================================================
# RELATIVE -> ABSOLUTE
# ============================================================

def trajectory_to_absolute(trajectory):
    """
    Convert:

        [dx, dy, eos]

    into:

        [x, y, eos]
    """

    points = []

    x = 0.0
    y = 0.0

    for dx, dy, eos in trajectory:
        x += float(dx)
        y += float(dy)

        points.append([
            x,
            y,
            eos,
        ])

    return np.asarray(
        points,
        dtype=np.float32,
    )


# ============================================================
# NORMALIZE FOR DISPLAY
# ============================================================

def normalize_for_display(
    absolute,
    target_height=100.0,
):
    """
    Scale generated handwriting to a convenient display
    height while preserving aspect ratio.
    """

    result = absolute.copy()

    x = result[:, 0]
    y = result[:, 1]

    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)

    if height > 1e-8:
        scale = target_height / height

        result[:, 0] *= scale
        result[:, 1] *= scale

    result[:, 0] -= np.min(result[:, 0])
    result[:, 1] -= np.min(result[:, 1])

    return result


# ============================================================
# PLOT TRAJECTORY
# ============================================================

def plot_trajectory(
    trajectory,
    output_path,
    title,
):
    """
    Render generated handwriting.

    EOS >= 0.5 means the current stroke ends
    at that point.
    """

    if len(trajectory) == 0:
        raise ValueError(
            "Cannot plot an empty trajectory."
        )

    eos = trajectory[:, 2]

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    stroke_start = 0

    for i in range(len(trajectory)):
        if eos[i] >= 0.5:
            stroke = trajectory[
                stroke_start:i + 1
            ]

            if len(stroke) >= 2:
                ax.plot(
                    stroke[:, 0],
                    -stroke[:, 1],
                    linewidth=1.5,
                )

            stroke_start = i + 1

    # Final unfinished stroke.
    if stroke_start < len(trajectory):
        stroke = trajectory[stroke_start:]

        if len(stroke) >= 2:
            ax.plot(
                stroke[:, 0],
                -stroke[:, 1],
                linewidth=1.5,
            )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SAVE JSON
# ============================================================

def save_trajectory_json(
    text,
    writer,
    relative_trajectory,
    absolute_trajectory,
    output_path,
):
    stroke_count = int(
        np.sum(
            relative_trajectory[:, 2] >= 0.5
        )
    )

    data = {
        "model": "Handyman_v3",
        "text": text,
        "writer": str(writer),
        "num_points": int(
            len(relative_trajectory)
        ),
        "num_strokes": stroke_count,
        "relative_trajectory": (
            relative_trajectory.tolist()
        ),
        "absolute_trajectory": (
            absolute_trajectory.tolist()
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate handwriting using "
            "the Handyman v3 model."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default="hello world",
        help="Text to generate.",
    )

    parser.add_argument(
        "--writer",
        type=str,
        default="1",
        help=(
            "BRUSH writer ID to use as "
            "handwriting style."
        ),
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=1200,
        help=(
            "Maximum number of trajectory "
            "points to generate."
        ),
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=30,
        help=(
            "Minimum number of trajectory "
            "points before stopping."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help=(
            "MDN sampling temperature. "
            "Lower = more conservative; "
            "higher = more varied."
        ),
    )

    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability threshold used to "
            "mark the end of a stroke."
        ),
    )

    parser.add_argument(
        "--end-patience",
        type=int,
        default=25,
        help=(
            "Number of steps attention must "
            "remain near the final text character "
            "before stopping."
        ),
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=1000.0,
        help=(
            "Maximum allowed absolute position "
            "magnitude during generation."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            CHECKPOINT_DIR / "best_v3.pt"
        ),
        help="V3 model checkpoint.",
    )

    args = parser.parse_args()

    print("=" * 75)
    print("HANDWRITING GENERATION — V3")
    print("=" * 75)
    print()
    print(f"Device:        {DEVICE}")
    print(f"Text:          {args.text!r}")
    print(f"Writer:        {args.writer}")
    print(f"Max points:    {args.max_points}")
    print(f"Min points:    {args.min_points}")
    print(f"Temperature:   {args.temperature}")
    print(f"EOS threshold: {args.eos_threshold}")
    print(f"End patience:  {args.end_patience}")
    print(f"Max position:  {args.max_position}")
    print(f"Checkpoint:    {args.checkpoint}")

    if args.max_points < 1:
        raise ValueError(
            "--max-points must be greater than 0."
        )

    if args.min_points < 1:
        raise ValueError(
            "--min-points must be greater than 0."
        )

    if args.min_points > args.max_points:
        raise ValueError(
            "--min-points cannot be larger than "
            "--max-points."
        )

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be greater than 0."
        )

    if not 0.0 < args.eos_threshold < 1.0:
        raise ValueError(
            "--eos-threshold must be between 0 and 1."
        )

    if args.end_patience < 1:
        raise ValueError(
            "--end-patience must be greater than 0."
        )

    if args.max_position <= 0:
        raise ValueError(
            "--max-position must be greater than 0."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, vocabulary, writers = load_model(
        Path(args.checkpoint)
    )

    # --------------------------------------------------------
    # Validate writer
    # --------------------------------------------------------

    if args.writer not in writers:
        print()
        print("Available writer IDs:")
        print(
            sorted(
                writers.keys(),
                key=lambda x: int(x),
            )
        )

        raise ValueError(
            f"Unknown writer ID: {args.writer}"
        )

    writer_id = writers[args.writer]

    # --------------------------------------------------------
    # Encode text
    # --------------------------------------------------------

    text_ids = encode_text(
        args.text,
        vocabulary,
    )

    print()
    print(
        f"Text token count: {len(text_ids)}"
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print("Generating trajectory...")
    print()

    relative_trajectory = generate_trajectory(
        model=model,
        text_ids=text_ids,
        writer_id=writer_id,
        max_points=args.max_points,
        temperature=args.temperature,
        min_points=args.min_points,
        end_patience=args.end_patience,
        eos_threshold=args.eos_threshold,
        max_position=args.max_position,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if relative_trajectory.ndim != 2:
        raise ValueError(
            "Generated trajectory must have "
            "shape [T, 3]. "
            f"Got {relative_trajectory.shape}"
        )

    if relative_trajectory.shape[1] != 3:
        raise ValueError(
            "Generated trajectory must have "
            "3 columns [dx, dy, eos]. "
            f"Got {relative_trajectory.shape}"
        )

    if len(relative_trajectory) == 0:
        raise ValueError(
            "Model generated an empty trajectory."
        )

    if not np.isfinite(
        relative_trajectory
    ).all():
        raise ValueError(
            "Generated trajectory contains "
            "NaN or Inf."
        )

    # --------------------------------------------------------
    # Count strokes
    # --------------------------------------------------------

    stroke_count = int(
        np.sum(
            relative_trajectory[:, 2] >= 0.5
        )
    )

    print(
        f"Generated points:         "
        f"{len(relative_trajectory)}"
    )

    print(
        f"Generated stroke endings: "
        f"{stroke_count}"
    )

    # --------------------------------------------------------
    # Convert to absolute coordinates
    # --------------------------------------------------------

    absolute_trajectory = (
        trajectory_to_absolute(
            relative_trajectory
        )
    )

    # --------------------------------------------------------
    # Normalize display version
    # --------------------------------------------------------

    display_trajectory = (
        normalize_for_display(
            absolute_trajectory
        )
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    safe_text = "".join(
        c if c.isalnum() else "_"
        for c in args.text
    ).strip("_")

    if not safe_text:
        safe_text = "handwriting"

    prefix = (
        f"{safe_text}"
        f"_writer_{args.writer}"
        f"_v3"
    )

    json_path = (
        OUTPUT_DIR /
        f"{prefix}.json"
    )

    png_path = (
        OUTPUT_DIR /
        f"{prefix}.png"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_trajectory_json(
        text=args.text,
        writer=args.writer,
        relative_trajectory=relative_trajectory,
        absolute_trajectory=absolute_trajectory,
        output_path=json_path,
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    plot_trajectory(
        trajectory=display_trajectory,
        output_path=png_path,
        title=(
            f"Generated handwriting: "
            f"{args.text!r} "
            f"(writer {args.writer}, v3)"
        ),
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    x = absolute_trajectory[:, 0]
    y = absolute_trajectory[:, 1]

    width = x.max() - x.min()
    height = y.max() - y.min()

    path_length = float(
        np.sum(
            np.sqrt(
                np.diff(x) ** 2
                + np.diff(y) ** 2
            )
        )
    )

    print()
    print("TRAJECTORY")
    print("-" * 40)
    print(
        f"X range:     "
        f"{x.min():.2f} -> {x.max():.2f}"
    )
    print(
        f"Y range:     "
        f"{y.min():.2f} -> {y.max():.2f}"
    )
    print(
        f"Width:       {width:.2f}"
    )
    print(
        f"Height:      {height:.2f}"
    )
    print(
        f"Path length: {path_length:.2f}"
    )

    print()
    print("OUTPUT")
    print("-" * 40)
    print(f"JSON: {json_path}")
    print(f"PNG:  {png_path}")

    print()
    print("=" * 75)
    print("GENERATION COMPLETE")
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
