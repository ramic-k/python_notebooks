#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_ROOT))

DETECTOR_ALIASES = {
    "tpx1": "tpx1",
    "tpx1-new": "tpx1",
    "tpx1_legacy": "tpx1_legacy",
    "tpx1-old": "tpx1_legacy",
    "tpx3": "tpx3",
}

REBIN_MODE_ALIASES = {
    "none": "none",
    "linear-tof": "linear_tof",
    "linear-lambda": "linear_lambda",
    "log-tof": "log_tof",
    "log-lambda": "log_lambda",
    "inverse-log-lambda": "inverse_log_lambda",
    "custom-schedule": "custom_schedule",
}

CUSTOM_BASIS_ALIASES = {
    "tof": "tof",
    "lambda": "lambda_",
    "lambda2": "lambda_squared",
    "lambda^2": "lambda_squared",
}

CUSTOM_SCALE_ALIASES = {
    "linear": "linear",
    "log": "log",
    "reverse-log": "reverse_log",
}


def load_runtime_dependencies():
    from __code._utilities.nexus import extract_file_path_from_nexus
    from __code.normalization_tof import (
        DetectorType,
        RebinCustomBasis,
        RebinCustomScale,
        RebinMode,
        Roi,
        autoreduce_dir,
        raw_dir,
    )
    from __code.normalization_tof.normalization_for_timepix1_timepix3 import (
        normalization_with_list_of_full_path,
    )
    from __code.normalization_tof.utilities import (
        DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
        retrieve_list_of_tif,
    )

    return {
        "extract_file_path_from_nexus": extract_file_path_from_nexus,
        "DetectorType": DetectorType,
        "RebinCustomBasis": RebinCustomBasis,
        "RebinCustomScale": RebinCustomScale,
        "RebinMode": RebinMode,
        "Roi": Roi,
        "autoreduce_dir": autoreduce_dir,
        "raw_dir": raw_dir,
        "normalization_with_list_of_full_path": normalization_with_list_of_full_path,
        "retrieve_list_of_tif": retrieve_list_of_tif,
        "DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE": DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run normalization_tof without the notebook widgets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./.pixi/envs/default/bin/python util/run_normalization_tof.py "
            "--ipts 36914 --detector tpx1 --sample-run 17174 --ob-path "
            "/SNS/VENUS/IPTS-36914/shared/autoreduce/images/tpx1/ob/... "
            "--output-folder /SNS/users/ykr/data/SNS/VENUS/IPTS-36914/shared/out\n"
            "  ./.pixi/envs/default/bin/python util/run_normalization_tof.py "
            "--working-dir /SNS/VENUS/IPTS-35167 --detector tpx3 "
            "--sample-path /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/images/tpx3/raw/... "
            "--ob-path /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/images/tpx3/raw/... "
            "--tof-bin-size-ns 700 --rebin-mode log-lambda --delta-lambda-over-lambda 0.05 "
            "--output-folder /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/shared/out\n"
        ),
    )
    parser.add_argument("--instrument", default="VENUS")
    parser.add_argument("--ipts", default="36914", help="IPTS number, with or without IPTS- prefix")
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Override working dir, e.g. /SNS/VENUS/IPTS-36914",
    )
    parser.add_argument(
        "--detector",
        default="tpx1",
        choices=sorted(DETECTOR_ALIASES.keys()),
        help="Detector workflow to use",
    )

    parser.add_argument("--sample-run", action="append", default=[], help="Sample run number")
    parser.add_argument("--sample-path", action="append", default=[], help="Full sample run folder path")
    parser.add_argument("--ob-run", action="append", default=[], help="OB run number")
    parser.add_argument("--ob-path", action="append", default=[], help="Full OB run folder path")
    parser.add_argument("--dc-run", action="append", default=[], help="Dark-current run number")
    parser.add_argument("--dc-path", action="append", default=[], help="Full dark-current run folder path")
    parser.add_argument(
        "--bragg-edge-cd-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Cd-filter background correction for Bragg edge mode. "
            "Requires sample and OB Cd-filter background runs."
        ),
    )
    parser.add_argument(
        "--closed-slits-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable closed-slits background correction for Bragg edge mode. "
            "Requires sample and OB closed-slits background runs."
        ),
    )
    parser.add_argument(
        "--sample-bg-run",
        action="append",
        default=[],
        help="Sample measured-background run number for the selected Bragg edge background mode",
    )
    parser.add_argument(
        "--sample-bg-path",
        action="append",
        default=[],
        help="Full sample measured-background run folder path for the selected Bragg edge background mode",
    )
    parser.add_argument(
        "--ob-bg-run",
        action="append",
        default=[],
        help="OB measured-background run number for the selected Bragg edge background mode",
    )
    parser.add_argument(
        "--ob-bg-path",
        action="append",
        default=[],
        help="Full OB measured-background run folder path for the selected Bragg edge background mode",
    )
    parser.add_argument(
        "--cd-sample-bg-run",
        action="append",
        default=[],
        help="Sample Cd-filter background run number",
    )
    parser.add_argument(
        "--cd-sample-bg-path",
        action="append",
        default=[],
        help="Full sample Cd-filter background run folder path",
    )
    parser.add_argument(
        "--cd-ob-bg-run",
        action="append",
        default=[],
        help="OB Cd-filter background run number",
    )
    parser.add_argument(
        "--cd-ob-bg-path",
        action="append",
        default=[],
        help="Full OB Cd-filter background run folder path",
    )
    parser.add_argument(
        "--closed-slits-sample-bg-run",
        action="append",
        default=[],
        help="Sample closed-slits background run number",
    )
    parser.add_argument(
        "--closed-slits-sample-bg-path",
        action="append",
        default=[],
        help="Full sample closed-slits background run folder path",
    )
    parser.add_argument(
        "--closed-slits-ob-bg-run",
        action="append",
        default=[],
        help="OB closed-slits background run number",
    )
    parser.add_argument(
        "--closed-slits-ob-bg-path",
        action="append",
        default=[],
        help="Full OB closed-slits background run folder path",
    )

    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--combine-samples", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--full-spectrum-roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute/export ROI spectrum normalization profile",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI as left,top,width,height",
    )
    parser.add_argument(
        "--default-roi-size",
        type=int,
        default=200,
        help="Centered square ROI size when --roi is not provided",
    )
    parser.add_argument(
        "--container-roi",
        default=None,
        help=(
            "Container-only reference ROI as left,top,width,height. "
            "When no OB run/path is provided, this enables container-only normalization."
        ),
    )
    parser.add_argument(
        "--container-roi-file",
        default=None,
        help="Previously exported container ROI JSON file to reuse for container normalization.",
    )

    parser.add_argument(
        "--proton-charge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--experimental-uncertainties",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default is enabled for TPX1, disabled for TPX3",
    )
    parser.add_argument(
        "--replace-ob-zeros-local-median",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--kernel-size", default="3,3,1", help="Local-median kernel as y,x,tof")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--distance-source-detector-m", type=float, default=25.0)
    parser.add_argument(
        "--correct-chips-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--rebin-mode",
        default="none",
        choices=sorted(REBIN_MODE_ALIASES.keys()),
    )
    parser.add_argument("--delta-tof-us", type=float, default=None)
    parser.add_argument("--delta-lambda-a", type=float, default=None)
    parser.add_argument("--delta-tof-over-tof", type=float, default=None)
    parser.add_argument("--delta-lambda-over-lambda", type=float, default=None)
    parser.add_argument("--delta-lambda-squared-a2", type=float, default=None)
    parser.add_argument(
        "--custom-basis",
        default="tof",
        choices=sorted(CUSTOM_BASIS_ALIASES.keys()),
    )
    parser.add_argument(
        "--custom-scale",
        default="linear",
        choices=sorted(CUSTOM_SCALE_ALIASES.keys()),
    )
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Custom schedule line as end_value,step. Repeat as needed. Final open segment: ',40'",
    )
    parser.add_argument(
        "--full-bins-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--snap-to-native-rebin-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For fixed-width rebin modes, round the requested bin width to an integer "
            "number of native source frames. Applies to linear TOF, linear lambda, and "
            "linear custom schedules on TOF/lambda."
        ),
    )

    parser.add_argument(
        "--black-filter-background",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable ROI-spectrum black-filter background correction. Use only for data containing "
            "the Ag black notch used to scale the fitted background shape."
        ),
    )
    parser.add_argument(
        "--black-filter-background-file",
        default=None,
        help="Background shape CSV. Defaults to the built-in HDPErpi degree-8 cdmax0.3 shape path.",
    )
    parser.add_argument(
        "--black-filter-anchor-energy-ev",
        type=float,
        default=5.1044,
        help="Nearest measured energy bin used to scale the sample and OB background shapes.",
    )

    parser.add_argument(
        "--tof-bin-size-ns",
        type=float,
        default=None,
        help="Manual fallback when no *_Spectra.txt exists",
    )

    parser.add_argument("--export-normalized-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-normalized-integrated", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-sample-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-sample-integrated", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-ob-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--export-ob-integrated", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def normalize_ipts(ipts_value: str) -> str:
    value = str(ipts_value).strip()
    return value if value.startswith("IPTS-") else f"IPTS-{value}"


def infer_working_dir(instrument: str, ipts: str, working_dir: str | None) -> Path:
    if working_dir:
        return Path(working_dir)
    return Path("/SNS") / instrument / ipts


def infer_run_number_from_path(full_path: str, detector_type: str) -> str:
    base_name = os.path.basename(os.path.normpath(full_path))
    tokens = base_name.split("_")

    if detector_type == "tpx1_legacy":
        if len(tokens) >= 2 and tokens[0] == "Run":
            return tokens[1]
    else:
        for index, token in enumerate(tokens):
            if token == "Run" and (index + 1) < len(tokens):
                return tokens[index + 1]

    raise ValueError(f"Unable to infer run number from path: {full_path}")


def resolve_path_from_run_number(
    instrument: str,
    ipts: str,
    working_dir: Path,
    detector_type: str,
    run_number: str,
    extract_file_path_from_nexus,
    autoreduce_dir,
    raw_dir,
    detector_type_constants,
) -> Path:
    run_number = str(run_number)

    if detector_type == detector_type_constants.tpx1_legacy:
        auto_root = (
            Path(autoreduce_dir[instrument][detector_type][0])
            / ipts
            / Path(autoreduce_dir[instrument][detector_type][1])
        )
        return auto_root / f"Run_{run_number}"

    nexus_path = working_dir / "nexus" / f"{instrument}_{run_number}.nxs.h5"
    if not nexus_path.exists():
        raise FileNotFoundError(f"NeXus file not found: {nexus_path}")

    relative_path = extract_file_path_from_nexus(nexus_path)
    if relative_path is None:
        raise ValueError(f"Could not extract file path from NeXus: {nexus_path}")

    extracted_path = Path(relative_path)
    if extracted_path.is_absolute():
        return extracted_path

    if detector_type == detector_type_constants.tpx1:
        auto_root = (
            Path(autoreduce_dir[instrument][detector_type][0])
            / ipts
            / Path(autoreduce_dir[instrument][detector_type][1])
        )
        return auto_root.parent.parent / extracted_path

    if detector_type == detector_type_constants.tpx3:
        raw_root = Path(raw_dir[instrument][detector_type][0]) / ipts / Path(raw_dir[instrument][detector_type][1])
        return raw_root / extracted_path

    raise ValueError(f"Unsupported detector type: {detector_type}")


def resolve_input_paths(
    run_numbers: list[str],
    explicit_paths: list[str],
    instrument: str,
    ipts: str,
    working_dir: Path,
    detector_type: str,
    label: str,
    extract_file_path_from_nexus,
    autoreduce_dir,
    raw_dir,
    detector_type_constants,
    allow_empty: bool = False,
) -> list[Path]:
    resolved = [Path(path) for path in explicit_paths]
    for run_number in run_numbers:
        resolved.append(
            resolve_path_from_run_number(
                instrument=instrument,
                ipts=ipts,
                working_dir=working_dir,
                detector_type=detector_type,
                run_number=run_number,
                extract_file_path_from_nexus=extract_file_path_from_nexus,
                autoreduce_dir=autoreduce_dir,
                raw_dir=raw_dir,
                detector_type_constants=detector_type_constants,
            )
        )

    if not resolved and not allow_empty:
        raise ValueError(f"No {label} inputs were provided.")

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {label} path(s): {missing}")

    return resolved


def build_data_dictionary(
    full_paths: list[Path],
    working_dir: Path,
    instrument: str,
    detector_type: str,
) -> dict:
    data_dictionary = {}
    for full_path in full_paths:
        run_number = infer_run_number_from_path(str(full_path), detector_type)
        nexus_path = working_dir / "nexus" / f"{instrument}_{run_number}.nxs.h5"
        data_dictionary[os.path.basename(full_path)] = {
            "full_path": str(full_path),
            "nexus": str(nexus_path) if nexus_path.exists() else None,
        }
    return data_dictionary


def first_spectra_file(full_path: Path) -> str | None:
    matches = sorted(glob.glob(os.path.join(full_path, "*_Spectra.txt")))
    return matches[0] if matches else None


def resolve_spectra_array(
    sample_paths: list[Path],
    ob_paths: list[Path],
    dc_paths: list[Path],
    tof_bin_size_ns: float | None,
    retrieve_list_of_tif,
) -> np.ndarray | None:
    import pandas as pd

    all_paths = sample_paths + ob_paths + dc_paths
    spectra_files = [first_spectra_file(path) for path in all_paths]
    found_spectra_files = [path for path in spectra_files if path is not None]

    if len(found_spectra_files) == len(all_paths):
        return None

    if found_spectra_files:
        spectra_file = found_spectra_files[0]
        pd_spectra = pd.read_csv(spectra_file, sep=",", header=0)
        if "shutter_time" in pd_spectra.columns:
            return np.asarray(pd_spectra["shutter_time"].values, dtype=np.float64)
        return np.asarray(pd_spectra.iloc[:, 0].values, dtype=np.float64)

    if tof_bin_size_ns is None:
        raise ValueError(
            "No *_Spectra.txt files were found. Provide --tof-bin-size-ns to create a manual spectra array."
        )

    first_sample_run = sample_paths[0]
    list_tif = retrieve_list_of_tif(first_sample_run)
    if not list_tif:
        raise ValueError(f"No TIFF files found in sample folder: {first_sample_run}")

    tof_bin_size_s = float(tof_bin_size_ns) * 1e-9
    return (np.arange(len(list_tif), dtype=np.float64) + 0.5) * tof_bin_size_s


def parse_roi(roi_text: str, Roi) -> object:
    parts = [part.strip() for part in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be left,top,width,height")
    left, top, width, height = [int(float(part)) for part in parts]
    return Roi(left=left, top=top, width=width, height=height)


def centered_roi_from_first_sample(sample_path: Path, size: int, retrieve_list_of_tif, Roi) -> object:
    list_tif = retrieve_list_of_tif(sample_path)
    if not list_tif:
        raise ValueError(f"No TIFF files found in sample folder: {sample_path}")

    with Image.open(list_tif[0]) as image:
        width, height = image.size

    roi_width = min(size, width)
    roi_height = min(size, height)
    left = max(0, (width - roi_width) // 2)
    top = max(0, (height - roi_height) // 2)
    return Roi(left=left, top=top, width=roi_width, height=roi_height)


def parse_kernel_size(kernel_text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in kernel_text.split(",")]
    if len(parts) != 3:
        raise ValueError("Kernel size must be y,x,tof")
    return tuple(int(part) for part in parts)


def parse_custom_segments(segment_texts: list[str]) -> list[dict] | None:
    if not segment_texts:
        return None

    parsed = []
    for raw_segment in segment_texts:
        parts = [part.strip() for part in raw_segment.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid --segment value: {raw_segment!r}. Expected end_value,step")
        end_value = None if parts[0] == "" else float(parts[0])
        step_value = float(parts[1])
        parsed.append({"end_value": end_value, "step": step_value})
    return parsed


def default_export_mode(args: argparse.Namespace) -> dict:
    return {
        "sample_stack": args.export_sample_stack,
        "ob_stack": args.export_ob_stack,
        "normalized_stack": args.export_normalized_stack,
        "sample_integrated": args.export_sample_integrated,
        "ob_integrated": args.export_ob_integrated,
        "normalized_integrated": args.export_normalized_integrated,
        "combined_normalized_integrated": False,
        "x_axis": True,
    }


def main() -> int:
    args = parse_args()
    runtime = load_runtime_dependencies()
    DetectorType = runtime["DetectorType"]
    RebinMode = runtime["RebinMode"]
    RebinCustomBasis = runtime["RebinCustomBasis"]
    RebinCustomScale = runtime["RebinCustomScale"]
    Roi = runtime["Roi"]
    autoreduce_dir = runtime["autoreduce_dir"]
    raw_dir = runtime["raw_dir"]
    extract_file_path_from_nexus = runtime["extract_file_path_from_nexus"]
    normalization_with_list_of_full_path = runtime["normalization_with_list_of_full_path"]
    retrieve_list_of_tif = runtime["retrieve_list_of_tif"]
    default_black_filter_background_shape_file = runtime["DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE"]

    instrument = args.instrument.upper()
    ipts = normalize_ipts(args.ipts)
    working_dir = infer_working_dir(instrument=instrument, ipts=ipts, working_dir=args.working_dir)
    detector_type = getattr(DetectorType, DETECTOR_ALIASES[args.detector])
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sample_paths = resolve_input_paths(
        run_numbers=args.sample_run,
        explicit_paths=args.sample_path,
        instrument=instrument,
        ipts=ipts,
        working_dir=working_dir,
        detector_type=detector_type,
        label="sample",
        extract_file_path_from_nexus=extract_file_path_from_nexus,
        autoreduce_dir=autoreduce_dir,
        raw_dir=raw_dir,
        detector_type_constants=DetectorType,
    )
    ob_paths = resolve_input_paths(
        run_numbers=args.ob_run,
        explicit_paths=args.ob_path,
        instrument=instrument,
        ipts=ipts,
        working_dir=working_dir,
        detector_type=detector_type,
        label="ob",
        extract_file_path_from_nexus=extract_file_path_from_nexus,
        autoreduce_dir=autoreduce_dir,
        raw_dir=raw_dir,
        detector_type_constants=DetectorType,
        allow_empty=bool(args.container_roi or args.container_roi_file),
    )
    dc_paths = resolve_input_paths(
        run_numbers=args.dc_run,
        explicit_paths=args.dc_path,
        instrument=instrument,
        ipts=ipts,
        working_dir=working_dir,
        detector_type=detector_type,
        label="dc",
        extract_file_path_from_nexus=extract_file_path_from_nexus,
        autoreduce_dir=autoreduce_dir,
        raw_dir=raw_dir,
        detector_type_constants=DetectorType,
    ) if (args.dc_run or args.dc_path) else []
    def resolve_optional_background_paths(run_numbers, explicit_paths, label):
        if not (run_numbers or explicit_paths):
            return []
        return resolve_input_paths(
            run_numbers=run_numbers,
            explicit_paths=explicit_paths,
            instrument=instrument,
            ipts=ipts,
            working_dir=working_dir,
            detector_type=detector_type,
            label=label,
            extract_file_path_from_nexus=extract_file_path_from_nexus,
            autoreduce_dir=autoreduce_dir,
            raw_dir=raw_dir,
            detector_type_constants=DetectorType,
        )

    generic_sample_bg_paths = resolve_optional_background_paths(
        args.sample_bg_run,
        args.sample_bg_path,
        "sample background",
    )
    generic_ob_bg_paths = resolve_optional_background_paths(
        args.ob_bg_run,
        args.ob_bg_path,
        "OB background",
    )
    measured_background_specs = [
        {
            "key": "cd",
            "key_prefix": "bragg_edge_cd",
            "flag": args.bragg_edge_cd_background,
            "mode": "Cd-filter background correction for Bragg edge mode",
            "column_label": "Cd-filter",
            "sample_paths": resolve_optional_background_paths(
                args.cd_sample_bg_run,
                args.cd_sample_bg_path,
                "sample Cd-filter background",
            ),
            "ob_paths": resolve_optional_background_paths(
                args.cd_ob_bg_run,
                args.cd_ob_bg_path,
                "OB Cd-filter background",
            ),
        },
        {
            "key": "closed_slits",
            "key_prefix": "closed_slits",
            "flag": args.closed_slits_background,
            "mode": "Closed-slits background correction for Bragg edge mode",
            "column_label": "closed-slits",
            "sample_paths": resolve_optional_background_paths(
                args.closed_slits_sample_bg_run,
                args.closed_slits_sample_bg_path,
                "sample closed-slits background",
            ),
            "ob_paths": resolve_optional_background_paths(
                args.closed_slits_ob_bg_run,
                args.closed_slits_ob_bg_path,
                "OB closed-slits background",
            ),
        },
    ]

    for background_spec in measured_background_specs:
        background_spec["has_dedicated_inputs"] = bool(
            background_spec["sample_paths"] or background_spec["ob_paths"]
        )
        background_spec["enabled"] = bool(
            background_spec["flag"] or background_spec["has_dedicated_inputs"]
        )

    enabled_background_specs = [
        background_spec
        for background_spec in measured_background_specs
        if background_spec["enabled"]
    ]
    if generic_sample_bg_paths or generic_ob_bg_paths:
        if len(enabled_background_specs) != 1:
            raise ValueError(
                "Generic --sample-bg-* and --ob-bg-* inputs can only be used when exactly one "
                "measured background mode is selected. For combined corrections, use the "
                "mode-specific --cd-* and --closed-slits-* inputs."
            )
        generic_target = enabled_background_specs[0]
        if generic_target["has_dedicated_inputs"]:
            raise ValueError(
                "Do not mix generic --sample-bg-* / --ob-bg-* with mode-specific measured "
                "background inputs for the same run."
            )
        generic_target["sample_paths"] = generic_sample_bg_paths
        generic_target["ob_paths"] = generic_ob_bg_paths
        generic_target["has_dedicated_inputs"] = True

    for background_spec in enabled_background_specs:
        if not background_spec["sample_paths"] or not background_spec["ob_paths"]:
            raise ValueError(
                f"{background_spec['mode']} requires both sample and OB background inputs."
            )

    for background_spec in enabled_background_specs:
        background_spec["weight"] = 1.0

    all_background_paths = []
    for background_spec in enabled_background_specs:
        all_background_paths += background_spec["sample_paths"]
        all_background_paths += background_spec["ob_paths"]

    spectra_array = resolve_spectra_array(
        sample_paths=sample_paths,
        ob_paths=ob_paths,
        dc_paths=dc_paths + all_background_paths,
        tof_bin_size_ns=args.tof_bin_size_ns,
        retrieve_list_of_tif=retrieve_list_of_tif,
    )

    if args.full_spectrum_roi:
        roi = (
            parse_roi(args.roi, Roi)
            if args.roi
            else centered_roi_from_first_sample(sample_paths[0], args.default_roi_size, retrieve_list_of_tif, Roi)
        )
    else:
        roi = None
    container_roi = parse_roi(args.container_roi, Roi) if args.container_roi else None
    container_roi_file = args.container_roi_file

    experimental_uncertainties_flag = (
        args.experimental_uncertainties
        if args.experimental_uncertainties is not None
        else detector_type != DetectorType.tpx3
    )

    rebin_mode = getattr(RebinMode, REBIN_MODE_ALIASES[args.rebin_mode])
    rebin_custom_schedule = parse_custom_segments(args.segment) if rebin_mode == RebinMode.custom_schedule else None
    rebin_custom_basis = (
        getattr(RebinCustomBasis, CUSTOM_BASIS_ALIASES[args.custom_basis])
        if rebin_mode == RebinMode.custom_schedule
        else None
    )
    rebin_custom_scale = (
        getattr(RebinCustomScale, CUSTOM_SCALE_ALIASES[args.custom_scale])
        if rebin_mode == RebinMode.custom_schedule
        else None
    )
    snap_to_native_rebin_grid = bool(
        args.snap_to_native_rebin_grid
        and (
            rebin_mode in [RebinMode.linear_tof, RebinMode.linear_lambda]
            or (
                rebin_mode == RebinMode.custom_schedule
                and rebin_custom_scale == RebinCustomScale.linear
                and rebin_custom_basis in [RebinCustomBasis.tof, RebinCustomBasis.lambda_]
            )
        )
    )
    export_mode = default_export_mode(args)
    kernel_size = parse_kernel_size(args.kernel_size)
    black_filter_background_config = None
    if args.black_filter_background:
        black_filter_background_config = {
            "enabled": True,
            "background_shape_file": args.black_filter_background_file
            or default_black_filter_background_shape_file,
            "anchor_energy_eV": args.black_filter_anchor_energy_ev,
        }
    sample_dict = build_data_dictionary(sample_paths, working_dir, instrument, detector_type)
    ob_dict = build_data_dictionary(ob_paths, working_dir, instrument, detector_type)
    dc_dict = build_data_dictionary(dc_paths, working_dir, instrument, detector_type) if dc_paths else {}
    measured_background_correction_configs = []
    for background_spec in enabled_background_specs:
        measured_background_correction_configs.append(
            {
                "enabled": True,
                "mode": background_spec["mode"],
                "column_label": background_spec["column_label"],
                "key_prefix": background_spec["key_prefix"],
                "weight": background_spec["weight"],
                "sample_background_dict": build_data_dictionary(
                    background_spec["sample_paths"],
                    working_dir,
                    instrument,
                    detector_type,
                ),
                "ob_background_dict": build_data_dictionary(
                    background_spec["ob_paths"],
                    working_dir,
                    instrument,
                    detector_type,
                ),
            }
        )

    print("Running normalization_tof with:")
    print(f"  working_dir: {working_dir}")
    print(f"  detector: {detector_type}")
    print(f"  sample paths: {[str(path) for path in sample_paths]}")
    print(f"  ob paths: {[str(path) for path in ob_paths]}")
    print(f"  dc paths: {[str(path) for path in dc_paths]}")
    for background_spec in enabled_background_specs:
        print(
            "  measured background: "
            f"{background_spec['column_label']} weight={background_spec['weight']:g} "
            f"sample={[str(path) for path in background_spec['sample_paths']]} "
            f"OB={[str(path) for path in background_spec['ob_paths']]}"
        )
    print(f"  output folder: {output_folder}")
    print(f"  roi: {roi}")
    print(f"  container roi: {container_roi}")
    print(f"  container roi file: {container_roi_file}")
    print(f"  rebin mode: {rebin_mode}")
    print(f"  custom schedule: {rebin_custom_schedule}")
    print(f"  snap fixed-width rebin to native grid: {snap_to_native_rebin_grid}")
    print(f"  experimental uncertainties: {experimental_uncertainties_flag}")
    print(f"  proton charge: {args.proton_charge}")
    print(f"  black-filter background correction: {black_filter_background_config}")
    print(
        "  measured background correction for Bragg edge mode: "
        f"{measured_background_correction_configs}"
    )

    normalized = normalization_with_list_of_full_path(
        sample_dict=sample_dict,
        combine_samples=args.combine_samples,
        ob_dict=ob_dict,
        dc_dict=dc_dict,
        spectra_array=spectra_array,
        output_folder=str(output_folder),
        verbose=False,
        proton_charge_flag=args.proton_charge,
        replace_ob_zeros_by_local_median_flag=args.replace_ob_zeros_local_median,
        kernel_size_for_local_median=kernel_size,
        max_iterations=args.max_iterations,
        instrument=instrument,
        preview=args.preview,
        distance_source_detector_m=args.distance_source_detector_m,
        correct_chips_alignment_flag=args.correct_chips_alignment,
        correct_chips_alignment_config=None,
        export_mode=export_mode,
        roi=roi,
        container_roi=container_roi,
        container_roi_file=container_roi_file,
        rebin_mode=rebin_mode,
        rebin_delta_tof_us=args.delta_tof_us if rebin_mode == RebinMode.linear_tof else None,
        rebin_delta_lambda_a=args.delta_lambda_a if rebin_mode == RebinMode.linear_lambda else None,
        rebin_delta_tof_over_tof=(
            args.delta_tof_over_tof if rebin_mode == RebinMode.log_tof else None
        ),
        rebin_delta_lambda_over_lambda=(
            args.delta_lambda_over_lambda if rebin_mode == RebinMode.log_lambda else None
        ),
        rebin_delta_lambda_squared_a2=(
            args.delta_lambda_squared_a2 if rebin_mode == RebinMode.inverse_log_lambda else None
        ),
        rebin_custom_basis=rebin_custom_basis,
        rebin_custom_scale=rebin_custom_scale,
        rebin_custom_schedule=rebin_custom_schedule,
        rebin_full_bins_only=args.full_bins_only if rebin_mode != RebinMode.none else False,
        rebin_snap_to_native_grid=snap_to_native_rebin_grid,
        experimental_uncertainties_flag=experimental_uncertainties_flag,
        black_filter_background_config=black_filter_background_config,
        measured_background_correction_configs=measured_background_correction_configs,
    )

    print("Normalization completed.")
    print(f"  output folder: {output_folder}")
    print(f"  normalized datasets: {list(normalized.data.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
