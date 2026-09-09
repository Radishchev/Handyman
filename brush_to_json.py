import json
import pickle
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BRUSH_ROOT = PROJECT_ROOT / "BRUSH"
OUTPUT_ROOT = PROJECT_ROOT / "BRUSH_JSON"


# ============================================================
# HELPERS
# ============================================================

def numpy_to_list(array):
    """
    Convert a NumPy array into ordinary Python lists so that
    it can be serialized as JSON.
    """

    return array.tolist()


def load_pickle(file_path):
    """
    Load one original BRUSH pickle file.
    """

    with open(file_path, "rb") as f:
        sentence, drawing, labels = pickle.load(f)

    return sentence, drawing, labels


def save_json(
    output_path,
    sentence,
    drawing,
    labels
):
    """
    Save one BRUSH sample as JSON.
    """

    data = {
        "sentence": sentence,
        "drawing": numpy_to_list(drawing),
        "labels": numpy_to_list(labels),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def load_json(json_path):
    """
    Load one converted JSON file.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return (
        data["sentence"],
        np.asarray(
            data["drawing"],
            dtype=float
        ),
        np.asarray(
            data["labels"]
        )
    )


# ============================================================
# VERIFY CONVERSION
# ============================================================

def verify_conversion(
    sentence_original,
    drawing_original,
    labels_original,
    json_path
):
    """
    Read the JSON back and compare it with the original
    Pickle data.

    Returns True if everything matches.
    """

    (
        sentence_json,
        drawing_json,
        labels_json
    ) = load_json(json_path)


    # --------------------------------------------------------
    # Sentence
    # --------------------------------------------------------

    if sentence_original != sentence_json:

        print(
            "    ERROR: sentence mismatch"
        )

        return False


    # --------------------------------------------------------
    # Drawing shape
    # --------------------------------------------------------

    if drawing_original.shape != drawing_json.shape:

        print(
            "    ERROR: drawing shape mismatch"
        )

        print(
            f"    Original: {drawing_original.shape}"
        )

        print(
            f"    JSON:     {drawing_json.shape}"
        )

        return False


    # --------------------------------------------------------
    # Drawing values
    # --------------------------------------------------------

    if not np.array_equal(
        drawing_original,
        drawing_json
    ):

        print(
            "    ERROR: drawing values mismatch"
        )

        return False


    # --------------------------------------------------------
    # Label shape
    # --------------------------------------------------------

    if labels_original.shape != labels_json.shape:

        print(
            "    ERROR: labels shape mismatch"
        )

        print(
            f"    Original: {labels_original.shape}"
        )

        print(
            f"    JSON:     {labels_json.shape}"
        )

        return False


    # --------------------------------------------------------
    # Label values
    # --------------------------------------------------------

    if not np.array_equal(
        labels_original,
        labels_json
    ):

        print(
            "    ERROR: labels values mismatch"
        )

        return False


    return True


# ============================================================
# FIND WRITERS
# ============================================================

def find_writer_directories():

    """
    Find writer directories:

        0
        1
        2
        ...
        169

    """

    writers = []


    for path in BRUSH_ROOT.iterdir():

        if not path.is_dir():
            continue


        if not path.name.isdigit():
            continue


        writer_id = int(path.name)


        if 0 <= writer_id <= 169:

            writers.append(
                path
            )


    writers.sort(
        key=lambda p: int(p.name)
    )


    return writers


# ============================================================
# FIND ORIGINAL FILES
# ============================================================

def find_original_files(writer_directory):

    """
    Find original BRUSH files.

    Accepted:

        0
        1
        2
        ...

    Ignored:

        0_resample20
        0_resample25
        ...
    """

    files = []


    for path in writer_directory.iterdir():

        if not path.is_file():
            continue


        # Original files have purely numeric names.

        if not path.name.isdigit():
            continue


        files.append(path)


    files.sort(
        key=lambda p: int(p.name)
    )


    return files


# ============================================================
# CONVERT ONE FILE
# ============================================================

def convert_file(
    input_path,
    output_path
):

    # Load original Pickle.

    (
        sentence,
        drawing,
        labels
    ) = load_pickle(
        input_path
    )


    # Save JSON.

    save_json(
        output_path,
        sentence,
        drawing,
        labels
    )


    # Verify.

    success = verify_conversion(
        sentence,
        drawing,
        labels,
        output_path
    )


    return success


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BRUSH → JSON CONVERTER")
    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Check BRUSH directory
    # --------------------------------------------------------

    if not BRUSH_ROOT.exists():

        raise FileNotFoundError(
            f"\nBRUSH directory not found:\n"
            f"{BRUSH_ROOT}\n"
        )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Find writers
    # --------------------------------------------------------

    writers = find_writer_directories()


    print(
        f"BRUSH directory:"
    )

    print(
        f"    {BRUSH_ROOT}"
    )

    print()


    print(
        f"Output directory:"
    )

    print(
        f"    {OUTPUT_ROOT}"
    )

    print()


    print(
        f"Writers found:"
    )

    print(
        f"    {len(writers)}"
    )

    print()


    if len(writers) != 170:

        print(
            "WARNING:"
        )

        print(
            f"Expected 170 writers, "
            f"but found {len(writers)}."
        )

        print()


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_files = 0

    successful_files = 0

    failed_files = 0


    # ========================================================
    # PROCESS EVERY WRITER
    # ========================================================

    for writer_number, writer_directory in enumerate(
        writers,
        start=1
    ):

        writer_id = int(
            writer_directory.name
        )


        print(
            "-" * 70
        )

        print(
            f"Writer {writer_id} "
            f"({writer_number}/{len(writers)})"
        )

        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Create writer output directory
        # ----------------------------------------------------

        output_writer_directory = (
            OUTPUT_ROOT
            / str(writer_id)
        )


        output_writer_directory.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Find original files
        # ----------------------------------------------------

        files = find_original_files(
            writer_directory
        )


        print(
            f"Original samples: {len(files)}"
        )

        print()


        # ----------------------------------------------------
        # Process files
        # ----------------------------------------------------

        for index, input_path in enumerate(
            files,
            start=1
        ):

            total_files += 1


            output_path = (
                output_writer_directory
                / f"{input_path.name}.json"
            )


            print(
                f"  [{index}/{len(files)}] "
                f"{input_path.name} "
                f"→ "
                f"{output_path.name}",
                end=" "
            )


            try:

                success = convert_file(
                    input_path,
                    output_path
                )


                if success:

                    successful_files += 1

                    print(
                        "✓"
                    )

                else:

                    failed_files += 1

                    print(
                        "✗ VERIFICATION FAILED"
                    )


            except Exception as e:

                failed_files += 1

                print(
                    f"✗ ERROR: {e}"
                )


        print()


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print()


    print(
        f"Writers processed:"
    )

    print(
        f"    {len(writers)}"
    )

    print()


    print(
        f"Total original files:"
    )

    print(
        f"    {total_files}"
    )

    print()


    print(
        f"Successfully converted:"
    )

    print(
        f"    {successful_files}"
    )

    print()


    print(
        f"Failed:"
    )

    print(
        f"    {failed_files}"
    )

    print()


    print(
        f"JSON dataset:"
    )

    print(
        f"    {OUTPUT_ROOT}"
    )

    print()


    if failed_files == 0:

        print(
            "✓ Every converted file passed verification."
        )

    else:

        print(
            "⚠ Some files failed verification."
        )


    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
