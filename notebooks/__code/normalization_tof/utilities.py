import argparse
import glob
import hashlib
import logging
import os
import re
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sqlite3 import Time
from typing import Tuple

import h5py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import iplot
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from PIL import Image
from skimage.io import imread
from scipy.ndimage import median_filter

TimepixGeometryCorrection = None

from __code.normalization_tof import RebinCustomBasis, RebinCustomScale, RebinMode, Roi
from __code._utilities.json import load_json, save_json

MARKERSIZE = 6
MAX_NORMALIZATION_OUTPUT_BASENAME_LENGTH = 220
FULL_STACK_IO_WORKERS = 4
TIFF_IO_PREFETCH_FACTOR = 2


def _compact_run_label(run_label) -> str:
    run_label = str(run_label)
    run_numbers = re.findall(r"Run_(\d+)", run_label)
    if not run_numbers:
        run_numbers = re.findall(r"\d{4,}", run_label)
    if run_numbers:
        return "Run_" + "_".join(run_numbers)
    digest = hashlib.sha1(run_label.encode("utf-8")).hexdigest()[:10]
    return f"hash_{digest}"


def _safe_normalization_output_basename(sample_label, ob_label, output_suffix: str = "") -> str:
    basename = f"normalized_sample_{sample_label}_obs_{ob_label}{output_suffix}"
    if len(basename) <= MAX_NORMALIZATION_OUTPUT_BASENAME_LENGTH:
        return basename

    compact_sample = _compact_run_label(sample_label)
    compact_ob = "no_open_beam" if str(ob_label) == "no_open_beam" else _compact_run_label(ob_label)
    compact_basename = f"normalized_sample_{compact_sample}_obs_{compact_ob}{output_suffix}"
    if len(compact_basename) <= MAX_NORMALIZATION_OUTPUT_BASENAME_LENGTH:
        logging.warning(
            "Normalization output directory name was too long; using compact run-number name: %s",
            compact_basename,
        )
        return compact_basename

    digest = hashlib.sha1(basename.encode("utf-8")).hexdigest()[:12]
    suffix_budget = 70
    compact_suffix = output_suffix[-suffix_budget:] if output_suffix else ""
    shortened = f"normalized_sample_{compact_sample}_obs_{compact_ob}_h{digest}{compact_suffix}"
    logging.warning(
        "Normalization output directory name was too long even after compaction; using hashed name: %s",
        shortened,
    )
    return shortened

class NormalizedData:
    data= {}
    lambda_array= None
    tof_array= None
    energy_array= None


from __code.normalization_tof.units import (
    DistanceUnitOptions,
    EnergyUnitOptions,
    TimeUnitOptions,
    convert_array_from_time_to_energy,
    convert_array_from_time_to_lambda,
)
# from __code.normalization_tof.normalization_for_timepix import create_master_dict

LOAD_DTYPE = np.uint16

PROTON_CHARGE_TOLERANCE = 0.1
DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE = (
    str(Path(__file__).resolve().parent / "data" / "hdperpi_background_constrained_poly_curve.csv")
)


class PLOT_SIZE:
    width = 8
    height = 5


SPECTRA_FILE_PREFIX = "Spectra.txt"

class DataType:
    sample = "sample"
    ob = "ob"
    dc = "dc"
    unknown = "unknown"
    sample_combined = "sample_combined"


class MasterDictKeys:
    frame_number = "frame_number"
    run_number = "run_number"
    proton_charge = "proton_charge"
    monitor_counts = "monitor_counts"
    matching_ob = "matching_ob"
    list_tif = "list_tif"
    data = "data"
    nexus_path = "nexus_path"
    data_path = "data_path"
    shutter_counts = "shutter_counts"
    list_spectra = "list_spectra"
    spectra_file_name = "spectra_file_name"
    detector_delay_us = "detector_delay_us"
    variance = "variance"


class StatusMetadata:
    all_shutter_counts_found = True
    all_monitor_counts_found = True
    all_spectra_found = True
    all_proton_charge_found = True


def _worker(fl):
#    return (imread(fl).astype(LOAD_DTYPE)).swapaxes(0, 1)
    return (imread(fl).astype(np.float32)).swapaxes(0, 1)
    #return (imread(fl).astype(np.float32))


def _bounded_ordered_thread_map(function, values, max_workers: int):
    """Yield ordered thread results without submitting the full TIFF list at once."""
    iterator = iter(values)
    worker_count = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = deque()
        for _ in range(worker_count * TIFF_IO_PREFETCH_FACTOR):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def _load_integrated_tof_data(list_tif: list = None) -> np.ndarray:
    if not list_tif:
        return np.array([], dtype=np.float32)

    max_workers = min(len(list_tif), os.cpu_count() or 1, 8)
    integrated_data = None

    for frame in _bounded_ordered_thread_map(_worker, list_tif, max_workers=max_workers):
        if integrated_data is None:
            integrated_data = frame
        else:
            integrated_data += frame

    return integrated_data


def _load_tof_stack(list_tif: list = None) -> np.ndarray:
    if not list_tif:
        return np.array([], dtype=np.float32)

    first_frame = _worker(list_tif[0])
    data = np.empty((len(list_tif), *first_frame.shape), dtype=np.float32)
    data[0] = first_frame

    remaining_files = list_tif[1:]
    if remaining_files:
        max_workers = min(len(remaining_files), FULL_STACK_IO_WORKERS)
        for index, frame in enumerate(
            _bounded_ordered_thread_map(_worker, remaining_files, max_workers=max_workers),
            start=1,
        ):
            data[index] = frame

    return data


def _frame_group_index_array(active_frame_groups, frame_count: int) -> np.ndarray:
    if not active_frame_groups:
        raise ValueError("Streaming rebinning requires at least one active output bin.")

    frame_to_group = np.full(frame_count, -1, dtype=int)
    for group_index, frame_group in enumerate(active_frame_groups):
        indices = np.asarray(frame_group, dtype=int)
        if np.any(indices < 0) or np.any(indices >= frame_count):
            raise ValueError("A streaming rebin group contains an out-of-range TIFF index.")
        if np.any(frame_to_group[indices] != -1):
            raise ValueError("A native TIFF index appears in more than one streaming rebin group.")
        frame_to_group[indices] = group_index
    return frame_to_group


def load_rebinned_tof_data(
    list_tif: list,
    active_frame_groups,
    shutter_counts=None,
    use_experimental_uncertainties: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream TIFFs directly into output-bin count and variance stacks."""
    if not list_tif:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float64)

    frame_to_group = _frame_group_index_array(active_frame_groups, len(list_tif))
    group_count = len(active_frame_groups)
    max_workers = min(len(list_tif), FULL_STACK_IO_WORKERS)
    primary_shutter_count = (
        _extract_primary_shutter_count(shutter_counts)
        if use_experimental_uncertainties
        else None
    )
    rebinned_data = None
    rebinned_variance = None
    cumulative_raw = None

    for frame_index, frame in enumerate(
        _bounded_ordered_thread_map(_worker, list_tif, max_workers=max_workers)
    ):
        if rebinned_data is None:
            rebinned_data = np.zeros((group_count, *frame.shape), dtype=np.float32)
            rebinned_variance = np.zeros((group_count, *frame.shape), dtype=np.float64)
            if primary_shutter_count is not None:
                cumulative_raw = np.zeros(frame.shape, dtype=np.float64)

        frame_float64 = np.asarray(frame, dtype=np.float64)
        if primary_shutter_count is not None:
            numerator = frame_float64 * (primary_shutter_count - cumulative_raw)
            denominator = primary_shutter_count + frame_float64
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_frame = np.where(
                    (denominator > 0) & (numerator >= 0),
                    numerator / denominator,
                    0.0,
                )
            cumulative_raw += raw_frame
            occupancy = cumulative_raw / primary_shutter_count
            with np.errstate(divide="ignore", invalid="ignore"):
                frame_variance = np.where(
                    1.0 - occupancy > 0,
                    frame_float64 / (1.0 - occupancy),
                    0.0,
                )
            frame_variance = np.maximum(frame_variance, 0.0)
        else:
            frame_variance = frame_float64

        group_index = frame_to_group[frame_index]
        if group_index >= 0:
            rebinned_data[group_index] += frame
            rebinned_variance[group_index] += frame_variance

    return rebinned_data, rebinned_variance


def load_rebinned_normalized_data_from_tiffs(
    sample_master_dict: dict,
    ob_master_dict: dict,
    sample_run_numbers: list,
    active_frame_groups,
    combine_samples: bool,
    use_proton_charge: bool,
    measured_background_runtime_configs=None,
) -> np.ndarray:
    """Stream and average corrected native pixel transmissions into output bins."""
    sample_infos = [sample_master_dict[run_number] for run_number in sample_run_numbers]
    ob_infos = list(ob_master_dict.values())
    if not sample_infos or not ob_infos:
        raise ValueError("Streaming native-ratio normalization requires sample and OB runs.")

    background_terms = []
    for config in measured_background_runtime_configs or []:
        sample_background_infos = list(config["sample_master_dict"].values())
        ob_background_infos = list(config["ob_master_dict"].values())
        if not sample_background_infos or not ob_background_infos:
            raise ValueError(
                "Streaming measured-background correction requires sample and OB background runs."
            )
        background_terms.append(
            {
                "weight": float(config.get("weight", 1.0)),
                "sample_infos": sample_background_infos,
                "ob_infos": ob_background_infos,
            }
        )

    all_infos = sample_infos + ob_infos
    for term in background_terms:
        all_infos.extend(term["sample_infos"])
        all_infos.extend(term["ob_infos"])
    file_lists = [info[MasterDictKeys.list_tif] for info in all_infos]
    frame_counts = {len(files) for files in file_lists}
    if len(frame_counts) != 1:
        raise ValueError(
            "Streaming native-ratio normalization requires equal TIFF counts in every run, "
            "including measured-background runs."
        )
    frame_count = frame_counts.pop()
    frame_to_group = _frame_group_index_array(active_frame_groups, frame_count)

    if use_proton_charge:
        sample_charges = np.asarray(
            [info[MasterDictKeys.proton_charge] for info in sample_infos],
            dtype=np.float64,
        )
        sample_charge_total = float(np.sum(sample_charges))
        ob_charge_total = float(
            np.sum([info[MasterDictKeys.proton_charge] for info in ob_infos])
        )
        for term in background_terms:
            term["sample_charge_total"] = float(
                np.sum(
                    [
                        info[MasterDictKeys.proton_charge]
                        for info in term["sample_infos"]
                    ]
                )
            )
            term["ob_charge_total"] = float(
                np.sum(
                    [
                        info[MasterDictKeys.proton_charge]
                        for info in term["ob_infos"]
                    ]
                )
            )
    else:
        sample_charges = None
        sample_charge_total = ob_charge_total = 1.0

    def combine_run_frame(infos, frame_index: int, charge_total: float) -> np.ndarray:
        frames = [_worker(info[MasterDictKeys.list_tif][frame_index]) for info in infos]
        combined = frames[0].copy()
        for frame in frames[1:]:
            combined += frame
        divisor = charge_total if use_proton_charge else len(frames)
        return combined / divisor

    def load_native_ratio(frame_index: int) -> np.ndarray:
        sample_frames = [_worker(info[MasterDictKeys.list_tif][frame_index]) for info in sample_infos]
        if combine_samples:
            sample_data = sample_frames[0].copy()
            for frame in sample_frames[1:]:
                sample_data += frame
            divisor = sample_charge_total if use_proton_charge else len(sample_frames)
            sample_data = sample_data / divisor
        else:
            sample_data = sample_frames[0]
            if use_proton_charge:
                sample_data = sample_data / sample_charges[0]

        ob_data = combine_run_frame(ob_infos, frame_index, ob_charge_total)

        for term in background_terms:
            sample_background = combine_run_frame(
                term["sample_infos"],
                frame_index,
                term.get("sample_charge_total", 1.0),
            )
            ob_background = combine_run_frame(
                term["ob_infos"],
                frame_index,
                term.get("ob_charge_total", 1.0),
            )
            sample_data = sample_data - term["weight"] * sample_background
            ob_data = ob_data - term["weight"] * ob_background

        normalized = np.divide(
            sample_data,
            ob_data,
            out=np.zeros_like(sample_data),
            where=ob_data != 0,
        )
        normalized[ob_data == 0] = 0
        return normalized

    rebinned_sum = None
    max_workers = min(frame_count, FULL_STACK_IO_WORKERS)
    for frame_index, normalized in enumerate(
        _bounded_ordered_thread_map(load_native_ratio, range(frame_count), max_workers=max_workers)
    ):
        if rebinned_sum is None:
            rebinned_sum = np.zeros((len(active_frame_groups), *normalized.shape), dtype=np.float32)
        group_index = frame_to_group[frame_index]
        if group_index >= 0:
            rebinned_sum[group_index] += normalized

    for group_index, frame_group in enumerate(active_frame_groups):
        rebinned_sum[group_index] /= len(frame_group)
    return rebinned_sum


def load_data_using_multithreading(list_tif: list = None, combine_tof: bool = False) -> np.ndarray:
    """Load a TIFF stack, or integrate it over TOF, with bounded I/O threads."""
    if combine_tof:
        return _load_integrated_tof_data(list_tif=list_tif)

    return _load_tof_stack(list_tif=list_tif)


def _get_timepix_geometry_correction():
    global TimepixGeometryCorrection

    if TimepixGeometryCorrection is None:
        try:
            from timepix_geometry_correction.correct import TimepixGeometryCorrection as _TimepixGeometryCorrection
        except ModuleNotFoundError:
            return None
        TimepixGeometryCorrection = _TimepixGeometryCorrection

    return TimepixGeometryCorrection


def retrieve_list_of_tif(folder: str) -> list:
    """retrieve list of tif files in the folder"""
    list_tif = glob.glob(os.path.join(folder, "*.tif*"))
    list_tif.sort()
    return list_tif


def create_x_axis_file(
    tof_array: np.ndarray = None,
    lambda_array: np.ndarray = None,
    energy_array: np.ndarray = None,
    bin_metadata: dict = None,
    output_folder: str = "./",
) -> str:
    """Create an x-axis file for normalized or rebinned data."""
    if bin_metadata:
        x_axis_data = {
            "file_index": bin_metadata["active_bin_index_array"],
            "source frame count": bin_metadata.get(
                "source_frame_count_array",
                np.ones(len(bin_metadata["active_bin_index_array"]), dtype=int),
            ),
            "starting tof (s)": bin_metadata["starting_tof_array"],
            "ending tof (s)": bin_metadata["ending_tof_array"],
            "mean tof (s)": bin_metadata["mean_tof_array"],
            "starting lambda (Angstroms)": bin_metadata["starting_lambda_array"],
            "ending lambda (Angstroms)": bin_metadata["ending_lambda_array"],
            "mean lambda (Angstroms)": bin_metadata["mean_lambda_array"],
            "starting energy (eV)": bin_metadata["starting_energy_array"],
            "ending energy (eV)": bin_metadata["ending_energy_array"],
            "mean energy (eV)": bin_metadata["mean_energy_array"],
        }
    else:
        x_axis_data = {
            "file_index": np.arange(len(lambda_array)),
            "mean tof (s)": tof_array,
            "mean lambda (Angstroms)": lambda_array,
            "mean energy (eV)": energy_array,
        }

    x_axis_file_name = os.path.join(output_folder, "x_axis.txt")
    pd.DataFrame(x_axis_data).to_csv(x_axis_file_name, index=False, sep=",")

    logging.info(f"X axis file created: {x_axis_file_name}")


def _format_rebin_value_for_output(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def create_rebin_output_suffix(
    rebin_mode: str = RebinMode.none,
    rebin_delta_tof_us: float = None,
    rebin_delta_lambda_a: float = None,
    rebin_delta_tof_over_tof: float = None,
    rebin_delta_lambda_over_lambda: float = None,
    rebin_delta_lambda_squared_a2: float = None,
    rebin_custom_basis: str = None,
    rebin_custom_scale: str = None,
    rebin_custom_schedule: list = None,
    rebin_full_bins_only: bool = False,
    rebin_snap_to_native_grid: bool = False,
) -> str:
    if rebin_mode == RebinMode.none:
        return ""

    output_suffix = None
    if rebin_mode == RebinMode.linear_tof:
        output_suffix = f"_rebin_lin_deltaTOF_{_format_rebin_value_for_output(rebin_delta_tof_us)}us"

    elif rebin_mode == RebinMode.linear_lambda:
        output_suffix = f"_rebin_lin_deltalambda_{_format_rebin_value_for_output(rebin_delta_lambda_a)}A"

    elif rebin_mode == RebinMode.log_tof:
        output_suffix = "_rebin_log_deltatof_over_tof_" + _format_rebin_value_for_output(rebin_delta_tof_over_tof)

    elif rebin_mode == RebinMode.log_lambda:
        output_suffix = "_rebin_log_deltalambdaoverlambda_" + _format_rebin_value_for_output(
            rebin_delta_lambda_over_lambda
        )

    elif rebin_mode == RebinMode.inverse_log_lambda:
        output_suffix = "_rebin_inverse_log_deltalambda2_" + _format_rebin_value_for_output(
            rebin_delta_lambda_squared_a2
        ) + "A2"

    elif rebin_mode == RebinMode.custom_schedule:
        basis_token = {
            RebinCustomBasis.energy_tof: "energyedges_tofwidths",
            RebinCustomBasis.tof: "tof",
            RebinCustomBasis.lambda_: "lambda",
            RebinCustomBasis.lambda_squared: "lambda2",
        }.get(rebin_custom_basis, "axis")
        scale_token = {
            RebinCustomScale.linear: "linear",
            RebinCustomScale.log: "log",
            RebinCustomScale.reverse_log: "reverse_log",
        }.get(rebin_custom_scale, "scale")
        segment_count = 0 if rebin_custom_schedule is None else len(rebin_custom_schedule)
        output_suffix = f"_rebin_custom_{basis_token}_{scale_token}_{segment_count}segments"
    else:
        raise ValueError(f"Unsupported rebin mode: {rebin_mode}")

    if rebin_full_bins_only:
        output_suffix += "_fullbins"
    snap_to_native_applies = (
        rebin_mode in [RebinMode.linear_tof, RebinMode.linear_lambda]
        or (
            rebin_mode == RebinMode.custom_schedule
            and rebin_custom_scale == RebinCustomScale.linear
            and rebin_custom_basis in [
                RebinCustomBasis.energy_tof,
                RebinCustomBasis.tof,
                RebinCustomBasis.lambda_,
            ]
        )
    )
    if rebin_snap_to_native_grid and snap_to_native_applies:
        output_suffix += "_snapnative"
    return output_suffix


def calculate_time_lambda_energy_arrays(
    time_spectra: np.ndarray = None,
    distance_source_detector_m: float = 25.0,
    detector_delay_us: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if time_spectra is None:
        return None, None, None

    if detector_delay_us is None:
        detector_delay_us = 0.0

    tof_array = np.asarray(time_spectra, dtype=np.float64)
    lambda_array = convert_array_from_time_to_lambda(
        time_array=tof_array,
        time_unit=TimeUnitOptions.s,
        distance_source_detector=distance_source_detector_m,
        distance_source_detector_unit=DistanceUnitOptions.m,
        detector_offset=detector_delay_us,
        detector_offset_unit=TimeUnitOptions.us,
        lambda_unit=DistanceUnitOptions.angstrom,
    )
    energy_array = convert_array_from_time_to_energy(
        time_array=tof_array,
        time_unit=TimeUnitOptions.s,
        distance_source_detector=distance_source_detector_m,
        distance_source_detector_unit=DistanceUnitOptions.m,
        detector_offset=detector_delay_us,
        detector_offset_unit=TimeUnitOptions.us,
        energy_unit=EnergyUnitOptions.eV,
    )
    return tof_array, lambda_array, energy_array


def _validate_rebin_axis(axis_values: np.ndarray, rebin_mode: str) -> np.ndarray:
    axis_values = np.asarray(axis_values, dtype=np.float64)
    if np.any(np.diff(axis_values) < 0):
        raise ValueError(f"{rebin_mode} rebinning requires a non-decreasing axis.")
    return axis_values


def _create_linear_bin_edges(
    axis_values: np.ndarray,
    bin_width: float,
    full_bins_only: bool = False,
    snap_to_native_grid: bool = False,
) -> np.ndarray:
    if bin_width is None or bin_width <= 0:
        raise ValueError("Linear rebinning requires a strictly positive bin width.")
    bin_width = _snap_linear_bin_width_to_native_axis(
        axis_values=axis_values,
        bin_width=bin_width,
        snap_to_native_grid=snap_to_native_grid,
    )
    new_axis = np.arange(axis_values[0], axis_values[-1], bin_width, dtype=np.float64)
    if len(new_axis) == 0:
        new_axis = np.array([axis_values[0]], dtype=np.float64)
    next_edge = new_axis[-1] + bin_width
    if (not full_bins_only) or (next_edge <= axis_values[-1] + 1e-12):
        new_axis = np.append(new_axis, next_edge)
    return new_axis


def _estimate_native_axis_step(axis_values: np.ndarray) -> float | None:
    axis_values = np.asarray(axis_values, dtype=np.float64)
    diffs = np.diff(axis_values)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return None

    median_step = float(np.median(positive_diffs))
    if median_step <= 0:
        return None

    # TPX1 axes can contain large chopper/frame gaps; those should not define
    # the native frame spacing used for fixed-width bin snapping.
    native_like_diffs = positive_diffs[positive_diffs <= (5.0 * median_step)]
    if len(native_like_diffs) == 0:
        native_like_diffs = positive_diffs
    native_step = float(np.median(native_like_diffs))
    return native_step if native_step > 0 else None


def _snap_linear_bin_width_to_native_axis(
    axis_values: np.ndarray,
    bin_width: float,
    snap_to_native_grid: bool = False,
) -> float:
    if not snap_to_native_grid:
        return bin_width

    native_step = _estimate_native_axis_step(axis_values)
    if native_step is None:
        return bin_width

    native_frame_count = max(1, int(round(bin_width / native_step)))
    return native_frame_count * native_step


def _snap_native_grid_full_bin_filter(
    bin_groups: list[list[int]],
    bin_edges: np.ndarray,
    axis_values: np.ndarray,
) -> list[list[int]]:
    native_step = _estimate_native_axis_step(axis_values)
    if native_step is None:
        return bin_groups

    filtered_groups = []
    for group, start_edge, end_edge in zip(bin_groups, bin_edges[:-1], bin_edges[1:]):
        expected_frame_count = max(1, int(round((end_edge - start_edge) / native_step)))
        filtered_groups.append(group if len(group) == expected_frame_count else [])
    return filtered_groups


def _create_log_bin_edges(
    axis_values: np.ndarray, relative_step: float, rebin_mode: str, full_bins_only: bool = False
) -> np.ndarray:
    if relative_step is None or relative_step <= 0:
        raise ValueError(f"{rebin_mode} requires a strictly positive logarithmic step.")
    if axis_values[0] <= 0:
        raise ValueError(f"{rebin_mode} requires strictly positive axis values.")

    start_parameter = float(axis_values[0])
    parameter_end = float(axis_values[-1])
    new_bin_array = [start_parameter]
    parameter = start_parameter
    while parameter < parameter_end:
        next_parameter = parameter + parameter * relative_step
        if next_parameter >= parameter_end:
            if (not full_bins_only) or np.isclose(next_parameter, parameter_end):
                new_bin_array.append(float(parameter_end if full_bins_only else next_parameter))
            break
        new_bin_array.append(next_parameter)
        parameter = next_parameter

    return np.asarray(new_bin_array, dtype=np.float64)


def _create_linear_bin_edges_between(
    start_value: float,
    end_value: float,
    bin_width: float,
    full_bins_only: bool = False,
    native_axis_step: float = None,
) -> np.ndarray:
    if bin_width is None or bin_width <= 0:
        raise ValueError("Linear rebinning requires a strictly positive bin width.")
    if end_value <= start_value:
        raise ValueError("Segment end must be greater than the segment start.")
    if native_axis_step is not None and native_axis_step > 0:
        native_frame_count = max(1, int(round(bin_width / native_axis_step)))
        bin_width = native_frame_count * native_axis_step

    new_bin_array = [float(start_value)]
    parameter = float(start_value)
    while parameter + bin_width < end_value:
        parameter += bin_width
        new_bin_array.append(parameter)

    next_edge = new_bin_array[-1] + bin_width
    if (not full_bins_only and new_bin_array[-1] < end_value) or np.isclose(next_edge, end_value):
        new_bin_array.append(float(end_value if full_bins_only else min(next_edge, end_value)))
    return np.asarray(new_bin_array, dtype=np.float64)


def _create_log_bin_edges_between(
    start_value: float, end_value: float, relative_step: float, rebin_mode: str, full_bins_only: bool = False
) -> np.ndarray:
    if relative_step is None or relative_step <= 0:
        raise ValueError(f"{rebin_mode} requires a strictly positive logarithmic step.")
    if start_value <= 0 or end_value <= 0:
        raise ValueError(f"{rebin_mode} requires strictly positive axis values.")
    if end_value <= start_value:
        raise ValueError("Segment end must be greater than the segment start.")

    new_bin_array = [float(start_value)]
    parameter = float(start_value)
    while parameter < end_value:
        next_parameter = parameter + parameter * relative_step
        if next_parameter >= end_value:
            if (not full_bins_only) or np.isclose(next_parameter, end_value):
                new_bin_array.append(float(end_value if full_bins_only else end_value))
            break
        new_bin_array.append(float(next_parameter))
        parameter = next_parameter

    if len(new_bin_array) == 1 and not full_bins_only:
        new_bin_array.append(float(end_value))
    return np.asarray(new_bin_array, dtype=np.float64)


def _create_reverse_log_bin_edges_between(
    start_value: float, end_value: float, relative_step: float, rebin_mode: str, full_bins_only: bool = False
) -> np.ndarray:
    forward_edges = _create_log_bin_edges_between(
        start_value, end_value, relative_step, rebin_mode, full_bins_only=full_bins_only
    )
    if len(forward_edges) < 2:
        return forward_edges
    forward_widths = np.diff(forward_edges)
    reverse_widths = forward_widths[::-1]

    if (not full_bins_only) and len(reverse_widths) >= 2 and reverse_widths[0] < reverse_widths[1]:
        reverse_widths[1] = reverse_widths[0] + reverse_widths[1]
        reverse_widths = reverse_widths[1:]

    new_bin_array = [float(start_value)]
    parameter = float(start_value)
    for width in reverse_widths:
        parameter += float(width)
        if parameter >= end_value:
            if not full_bins_only:
                new_bin_array.append(float(end_value))
            break
        new_bin_array.append(parameter)

    if (not full_bins_only) and new_bin_array[-1] < end_value:
        new_bin_array.append(float(end_value))
    return np.asarray(new_bin_array, dtype=np.float64)


def _normalize_custom_schedule(rebin_custom_schedule: list = None) -> list[dict]:
    if not rebin_custom_schedule:
        raise ValueError("custom_schedule requires at least one segment.")

    normalized_schedule = []
    for segment_index, segment in enumerate(rebin_custom_schedule):
        if isinstance(segment, dict):
            end_value = segment.get("end_value")
            step_value = segment.get("step")
        else:
            if len(segment) != 2:
                raise ValueError(
                    f"Invalid custom segment at index {segment_index}: expected (end_value, step)."
                )
            end_value, step_value = segment

        if end_value in ["", None]:
            normalized_end_value = None
        else:
            normalized_end_value = float(end_value)

        normalized_step_value = float(step_value)
        if normalized_step_value <= 0:
            raise ValueError(f"Custom segment {segment_index} step must be strictly positive.")

        normalized_schedule.append(
            {
                "end_value": normalized_end_value,
                "step": normalized_step_value,
            }
        )

    none_indices = [index for index, segment in enumerate(normalized_schedule) if segment["end_value"] is None]
    if len(none_indices) > 1:
        raise ValueError("Custom schedule can contain at most one open-ended segment.")
    if none_indices and none_indices[0] != len(normalized_schedule) - 1:
        raise ValueError("The open-ended custom schedule segment must be the final line.")

    specified_end_values = [segment["end_value"] for segment in normalized_schedule if segment["end_value"] is not None]
    if any(
        specified_end_values[index] >= specified_end_values[index + 1]
        for index in range(len(specified_end_values) - 1)
    ):
        raise ValueError("Custom schedule boundaries must be provided in strictly increasing order.")

    return normalized_schedule


def _create_edges_for_custom_segment(
    start_value: float = None,
    end_value: float = None,
    step_value: float = None,
    rebin_custom_scale: str = None,
    full_bins_only: bool = False,
    native_axis_step: float = None,
) -> np.ndarray:
    if rebin_custom_scale == RebinCustomScale.linear:
        return _create_linear_bin_edges_between(
            start_value,
            end_value,
            step_value,
            full_bins_only=full_bins_only,
            native_axis_step=native_axis_step,
        )
    if rebin_custom_scale == RebinCustomScale.log:
        return _create_log_bin_edges_between(
            start_value, end_value, step_value, RebinMode.custom_schedule, full_bins_only=full_bins_only
        )
    if rebin_custom_scale == RebinCustomScale.reverse_log:
        return _create_reverse_log_bin_edges_between(
            start_value, end_value, step_value, RebinMode.custom_schedule, full_bins_only=full_bins_only
        )

    raise ValueError(f"Unsupported custom schedule scale: {rebin_custom_scale}")


def _build_custom_schedule_bin_edges(
    tof_array: np.ndarray = None,
    lambda_array: np.ndarray = None,
    energy_array: np.ndarray = None,
    rebin_custom_basis: str = None,
    rebin_custom_scale: str = None,
    rebin_custom_schedule: list = None,
    rebin_full_bins_only: bool = False,
    rebin_snap_to_native_grid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_schedule = _normalize_custom_schedule(rebin_custom_schedule)

    if rebin_custom_basis == RebinCustomBasis.energy_tof:
        if rebin_custom_scale != RebinCustomScale.linear:
            raise ValueError("Energy-edge/TOF-width schedules only support linear TOF widths.")
        return _build_energy_edge_tof_width_bin_edges(
            tof_array=tof_array,
            energy_array=energy_array,
            normalized_schedule=normalized_schedule,
            rebin_full_bins_only=rebin_full_bins_only,
            rebin_snap_to_native_grid=rebin_snap_to_native_grid,
        )

    def _convert_custom_step(step_value: float) -> float:
        if rebin_custom_basis == RebinCustomBasis.tof and rebin_custom_scale == RebinCustomScale.linear:
            return step_value * 1e-6
        return step_value

    if rebin_custom_basis == RebinCustomBasis.tof:
        axis_values = _validate_rebin_axis(tof_array, RebinMode.custom_schedule)
        axis_label = "microseconds"
    elif rebin_custom_basis == RebinCustomBasis.lambda_:
        axis_values = _validate_rebin_axis(lambda_array, RebinMode.custom_schedule)
        axis_label = "Angstroms"
    elif rebin_custom_basis == RebinCustomBasis.lambda_squared:
        axis_values = _validate_rebin_axis(np.square(lambda_array), RebinMode.custom_schedule)
        if rebin_custom_scale != RebinCustomScale.linear:
            raise ValueError("lambda^2 custom schedules only support linear steps.")
        axis_label = "Angstroms^2"
    else:
        raise ValueError(f"Unsupported custom schedule basis: {rebin_custom_basis}")

    axis_start = float(axis_values[0])
    axis_end = float(axis_values[-1])
    native_axis_step = None
    if (
        rebin_snap_to_native_grid
        and rebin_custom_scale == RebinCustomScale.linear
        and rebin_custom_basis in [RebinCustomBasis.tof, RebinCustomBasis.lambda_]
    ):
        native_axis_step = _estimate_native_axis_step(axis_values)

    custom_bin_edges = [axis_start]
    current_start = axis_start
    for segment in normalized_schedule:
        end_value = segment["end_value"]
        segment_end = axis_end if end_value is None else float(end_value)
        if (rebin_custom_basis == RebinCustomBasis.tof) and (end_value is not None):
            segment_end *= 1e-6

        if segment_end < axis_start or segment_end > axis_end:
            display_start = axis_start * 1e6 if rebin_custom_basis == RebinCustomBasis.tof else axis_start
            display_end = axis_end * 1e6 if rebin_custom_basis == RebinCustomBasis.tof else axis_end
            requested_end = float(end_value) if end_value is not None else display_end
            raise ValueError(
                f"Custom schedule boundary {requested_end} is outside the data range "
                f"[{display_start}, {display_end}] {axis_label}."
            )

        if segment_end <= current_start:
            raise ValueError("Custom schedule boundaries must increase from the minimum axis upward.")

        segment_edges = _create_edges_for_custom_segment(
            start_value=current_start,
            end_value=segment_end,
            step_value=_convert_custom_step(segment["step"]),
            rebin_custom_scale=rebin_custom_scale,
            full_bins_only=rebin_full_bins_only,
            native_axis_step=native_axis_step,
        )
        custom_bin_edges.extend(segment_edges[1:])
        current_start = custom_bin_edges[-1]

        if current_start >= axis_end or end_value is None:
            break

    if current_start < axis_end:
        segment_edges = _create_edges_for_custom_segment(
            start_value=current_start,
            end_value=axis_end,
            step_value=_convert_custom_step(normalized_schedule[-1]["step"]),
            rebin_custom_scale=rebin_custom_scale,
            full_bins_only=rebin_full_bins_only,
            native_axis_step=native_axis_step,
        )
        custom_bin_edges.extend(segment_edges[1:])

    return axis_values, np.asarray(custom_bin_edges, dtype=np.float64)


def _build_energy_edge_tof_width_bin_edges(
    tof_array: np.ndarray,
    energy_array: np.ndarray,
    normalized_schedule: list[dict],
    rebin_full_bins_only: bool = False,
    rebin_snap_to_native_grid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build TOF bins from increasing energy-region edges and TOF widths.

    A row ``energy_upper_edge_eV, delta_tof_us`` applies that TOF width from
    the previous energy edge up to the listed edge. A final blank edge covers
    the remaining high-energy range. Since neutron energy decreases with TOF,
    the energy regions are reversed before constructing the ascending TOF bins.
    """
    tof_values = _validate_rebin_axis(tof_array, RebinMode.custom_schedule)
    energy_values = np.asarray(energy_array, dtype=np.float64)
    if energy_values.shape != tof_values.shape:
        raise ValueError("Energy-edge/TOF-width schedules require matching TOF and energy arrays.")
    if np.any(~np.isfinite(energy_values)) or np.any(energy_values <= 0):
        raise ValueError("Energy-edge/TOF-width schedules require finite positive energies.")
    if np.any(np.diff(energy_values) > 0):
        raise ValueError("Energy-edge/TOF-width schedules require energy to decrease as TOF increases.")

    finite_segments = [segment for segment in normalized_schedule if segment["end_value"] is not None]
    requested_energy_edges = np.asarray(
        [float(segment["end_value"]) for segment in finite_segments],
        dtype=np.float64,
    )
    energy_min = float(np.min(energy_values))
    energy_max = float(np.max(energy_values))
    if np.any(requested_energy_edges <= energy_min) or np.any(requested_energy_edges >= energy_max):
        raise ValueError(
            "Energy schedule edges must lie strictly inside this frame's energy range "
            f"({energy_min:.6g}, {energy_max:.6g}) eV."
        )

    low_to_high_widths_us = [float(segment["step"]) for segment in finite_segments]
    open_segment = next(
        (segment for segment in normalized_schedule if segment["end_value"] is None),
        None,
    )
    high_energy_width_us = (
        float(open_segment["step"])
        if open_segment is not None
        else float(normalized_schedule[-1]["step"])
    )
    low_to_high_widths_us.append(high_energy_width_us)

    requested_tof_edges = np.interp(
        requested_energy_edges,
        energy_values[::-1],
        tof_values[::-1],
    )
    tof_region_ends = [*requested_tof_edges[::-1], float(tof_values[-1])]
    high_to_low_widths_us = low_to_high_widths_us[::-1]
    native_axis_step = (
        _estimate_native_axis_step(tof_values)
        if rebin_snap_to_native_grid
        else None
    )

    custom_bin_edges = [float(tof_values[0])]
    current_start = custom_bin_edges[0]
    for segment_end, width_us in zip(tof_region_ends, high_to_low_widths_us):
        if segment_end <= current_start:
            continue
        segment_edges = _create_linear_bin_edges_between(
            start_value=current_start,
            end_value=float(segment_end),
            bin_width=float(width_us) * 1e-6,
            full_bins_only=rebin_full_bins_only,
            native_axis_step=native_axis_step,
        )
        custom_bin_edges.extend(segment_edges[1:])
        current_start = custom_bin_edges[-1]

    if len(custom_bin_edges) < 2:
        raise ValueError("The energy-edge/TOF-width schedule produced no complete output bins.")
    return tof_values, np.asarray(custom_bin_edges, dtype=np.float64)


def build_rebin_bin_groups(
    rebin_mode: str = RebinMode.none,
    tof_array: np.ndarray = None,
    lambda_array: np.ndarray = None,
    energy_array: np.ndarray = None,
    rebin_delta_tof_us: float = None,
    rebin_delta_lambda_a: float = None,
    rebin_delta_tof_over_tof: float = None,
    rebin_delta_lambda_over_lambda: float = None,
    rebin_delta_lambda_squared_a2: float = None,
    rebin_custom_basis: str = None,
    rebin_custom_scale: str = None,
    rebin_custom_schedule: list = None,
    rebin_full_bins_only: bool = False,
    rebin_snap_to_native_grid: bool = False,
) -> tuple[list[list[int]], np.ndarray]:
    if tof_array is None:
        return None, None

    nbr_frames = len(tof_array)
    if rebin_mode == RebinMode.none:
        return [[index] for index in range(nbr_frames)], np.arange(nbr_frames + 1, dtype=np.float64)

    if rebin_mode == RebinMode.linear_tof:
        axis_values = _validate_rebin_axis(tof_array, rebin_mode)
        bin_edges = _create_linear_bin_edges(
            axis_values,
            rebin_delta_tof_us * 1e-6,
            full_bins_only=rebin_full_bins_only,
            snap_to_native_grid=rebin_snap_to_native_grid,
        )
    elif rebin_mode == RebinMode.linear_lambda:
        axis_values = _validate_rebin_axis(lambda_array, rebin_mode)
        bin_edges = _create_linear_bin_edges(
            axis_values,
            rebin_delta_lambda_a,
            full_bins_only=rebin_full_bins_only,
            snap_to_native_grid=rebin_snap_to_native_grid,
        )
    elif rebin_mode == RebinMode.log_tof:
        axis_values = _validate_rebin_axis(tof_array, rebin_mode)
        bin_edges = _create_log_bin_edges(
            axis_values, rebin_delta_tof_over_tof, rebin_mode, full_bins_only=rebin_full_bins_only
        )
    elif rebin_mode == RebinMode.log_lambda:
        axis_values = _validate_rebin_axis(lambda_array, rebin_mode)
        bin_edges = _create_log_bin_edges(
            axis_values, rebin_delta_lambda_over_lambda, rebin_mode, full_bins_only=rebin_full_bins_only
        )
    elif rebin_mode == RebinMode.inverse_log_lambda:
        axis_values = _validate_rebin_axis(np.square(lambda_array), rebin_mode)
        bin_edges = _create_linear_bin_edges(
            axis_values, rebin_delta_lambda_squared_a2, full_bins_only=rebin_full_bins_only
        )
    elif rebin_mode == RebinMode.custom_schedule:
        axis_values, bin_edges = _build_custom_schedule_bin_edges(
            tof_array=tof_array,
            lambda_array=lambda_array,
            energy_array=energy_array,
            rebin_custom_basis=rebin_custom_basis,
            rebin_custom_scale=rebin_custom_scale,
            rebin_custom_schedule=rebin_custom_schedule,
            rebin_full_bins_only=rebin_full_bins_only,
            rebin_snap_to_native_grid=rebin_snap_to_native_grid,
        )
    else:
        raise ValueError(f"Unsupported rebin mode: {rebin_mode}")

    bin_groups = [[] for _ in np.arange(len(bin_edges) - 1)]
    for frame_index, axis_value in enumerate(axis_values):
        group_index = int(np.searchsorted(bin_edges, axis_value, side="right") - 1)
        if group_index < 0 or group_index >= len(bin_groups):
            continue
        bin_groups[group_index].append(frame_index)

    snap_to_native_applies = (
        rebin_mode in [RebinMode.linear_tof, RebinMode.linear_lambda]
        or (
            rebin_mode == RebinMode.custom_schedule
            and rebin_custom_scale == RebinCustomScale.linear
            and rebin_custom_basis in [
                RebinCustomBasis.energy_tof,
                RebinCustomBasis.tof,
                RebinCustomBasis.lambda_,
            ]
        )
    )
    if rebin_snap_to_native_grid and rebin_full_bins_only and snap_to_native_applies:
        bin_groups = _snap_native_grid_full_bin_filter(
            bin_groups=bin_groups,
            bin_edges=bin_edges,
            axis_values=axis_values,
        )

    return bin_groups, bin_edges


def build_rebin_bin_metadata(
    tof_array: np.ndarray = None,
    lambda_array: np.ndarray = None,
    energy_array: np.ndarray = None,
    bin_groups: list[list[int]] = None,
) -> dict:
    if bin_groups is None:
        return None

    active_bin_indices = []
    active_frame_groups = []
    source_frame_counts = []
    starting_tof = []
    ending_tof = []
    mean_tof = []
    starting_lambda = []
    ending_lambda = []
    mean_lambda = []
    starting_energy = []
    ending_energy = []
    mean_energy = []

    for full_bin_index, frame_group in enumerate(bin_groups):
        if not frame_group:
            continue

        frame_group = np.asarray(frame_group, dtype=int)
        active_bin_indices.append(full_bin_index)
        active_frame_groups.append(frame_group)
        source_frame_counts.append(len(frame_group))

        tof_values = np.asarray(tof_array[frame_group], dtype=np.float64)
        lambda_values = np.asarray(lambda_array[frame_group], dtype=np.float64)
        energy_values = np.asarray(energy_array[frame_group], dtype=np.float64)

        starting_tof.append(tof_values[0])
        ending_tof.append(tof_values[-1])
        mean_tof.append(np.mean(tof_values))

        starting_lambda.append(lambda_values[0])
        ending_lambda.append(lambda_values[-1])
        mean_lambda.append(np.mean(lambda_values))

        starting_energy.append(energy_values[0])
        ending_energy.append(energy_values[-1])
        mean_energy.append(np.mean(energy_values))

    return {
        "active_bin_index_array": np.asarray(active_bin_indices, dtype=int),
        "list_file_index_array": active_frame_groups,
        "source_frame_count_array": np.asarray(source_frame_counts, dtype=int),
        "starting_tof_array": np.asarray(starting_tof, dtype=np.float64),
        "ending_tof_array": np.asarray(ending_tof, dtype=np.float64),
        "mean_tof_array": np.asarray(mean_tof, dtype=np.float64),
        "starting_lambda_array": np.asarray(starting_lambda, dtype=np.float64),
        "ending_lambda_array": np.asarray(ending_lambda, dtype=np.float64),
        "mean_lambda_array": np.asarray(mean_lambda, dtype=np.float64),
        "starting_energy_array": np.asarray(starting_energy, dtype=np.float64),
        "ending_energy_array": np.asarray(ending_energy, dtype=np.float64),
        "mean_energy_array": np.asarray(mean_energy, dtype=np.float64),
    }


def rebin_array_from_bin_groups(
    data: np.ndarray = None, active_frame_groups: list[np.ndarray] = None, reducer: str = "sum"
) -> np.ndarray:
    if data is None:
        return None

    if active_frame_groups is None:
        return np.asarray(data).copy()

    array_data = np.asarray(data)
    if array_data.ndim == 0:
        return None

    if reducer == "sum":
        reduced_chunks = [np.sum(array_data[_group], axis=0) for _group in active_frame_groups]
    elif reducer == "mean":
        reduced_chunks = [np.mean(array_data[_group], axis=0) for _group in active_frame_groups]
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")

    return np.asarray(reduced_chunks)


def maybe_rebin_data_and_axes(
    sample_data: np.ndarray = None,
    sample_variance: np.ndarray = None,
    ob_data_combined: np.ndarray = None,
    ob_data_combined_variance: np.ndarray = None,
    dc_data_combined: np.ndarray = None,
    dc_data_combined_variance: np.ndarray = None,
    time_spectra: np.ndarray = None,
    distance_source_detector_m: float = 25.0,
    detector_delay_us: float = None,
    rebin_mode: str = RebinMode.none,
    rebin_delta_tof_us: float = None,
    rebin_delta_lambda_a: float = None,
    rebin_delta_tof_over_tof: float = None,
    rebin_delta_lambda_over_lambda: float = None,
    rebin_delta_lambda_squared_a2: float = None,
    rebin_custom_basis: str = None,
    rebin_custom_scale: str = None,
    rebin_custom_schedule: list = None,
    rebin_full_bins_only: bool = False,
    rebin_snap_to_native_grid: bool = False,
) -> dict:
    if (time_spectra is None) and (rebin_mode != RebinMode.none):
        raise ValueError(f"{rebin_mode} requires a valid spectra/time axis.")

    tof_array, lambda_array, energy_array = calculate_time_lambda_energy_arrays(
        time_spectra=time_spectra,
        distance_source_detector_m=distance_source_detector_m,
        detector_delay_us=detector_delay_us,
    )

    output_suffix = create_rebin_output_suffix(
        rebin_mode=rebin_mode,
        rebin_delta_tof_us=rebin_delta_tof_us,
        rebin_delta_lambda_a=rebin_delta_lambda_a,
        rebin_delta_tof_over_tof=rebin_delta_tof_over_tof,
        rebin_delta_lambda_over_lambda=rebin_delta_lambda_over_lambda,
        rebin_delta_lambda_squared_a2=rebin_delta_lambda_squared_a2,
        rebin_custom_basis=rebin_custom_basis,
        rebin_custom_scale=rebin_custom_scale,
        rebin_custom_schedule=rebin_custom_schedule,
        rebin_full_bins_only=rebin_full_bins_only,
        rebin_snap_to_native_grid=rebin_snap_to_native_grid,
    )

    if tof_array is None:
        return {
            "sample_data": sample_data,
            "sample_variance": sample_variance,
            "ob_data_combined": ob_data_combined,
            "ob_data_combined_variance": ob_data_combined_variance,
            "dc_data_combined": dc_data_combined,
            "dc_data_combined_variance": dc_data_combined_variance,
            "original_tof_array": tof_array,
            "original_lambda_array": lambda_array,
            "original_energy_array": energy_array,
            "active_frame_groups": None,
            "tof_array": None,
            "lambda_array": None,
            "energy_array": None,
            "output_suffix": output_suffix,
            "bin_metadata": None,
        }

    bin_groups, _ = build_rebin_bin_groups(
        rebin_mode=rebin_mode,
        tof_array=tof_array,
        lambda_array=lambda_array,
        energy_array=energy_array,
        rebin_delta_tof_us=rebin_delta_tof_us,
        rebin_delta_lambda_a=rebin_delta_lambda_a,
        rebin_delta_tof_over_tof=rebin_delta_tof_over_tof,
        rebin_delta_lambda_over_lambda=rebin_delta_lambda_over_lambda,
        rebin_delta_lambda_squared_a2=rebin_delta_lambda_squared_a2,
        rebin_custom_basis=rebin_custom_basis,
        rebin_custom_scale=rebin_custom_scale,
        rebin_custom_schedule=rebin_custom_schedule,
        rebin_full_bins_only=rebin_full_bins_only,
        rebin_snap_to_native_grid=rebin_snap_to_native_grid,
    )
    bin_metadata = build_rebin_bin_metadata(
        tof_array=tof_array,
        lambda_array=lambda_array,
        energy_array=energy_array,
        bin_groups=bin_groups,
    )
    active_frame_groups = None if bin_metadata is None else bin_metadata["list_file_index_array"]
    if bin_metadata is not None:
        bin_metadata["rebin_snap_to_native_grid"] = bool(rebin_snap_to_native_grid)

    return {
        "sample_data": rebin_array_from_bin_groups(sample_data, active_frame_groups, reducer="sum"),
        "sample_variance": rebin_array_from_bin_groups(sample_variance, active_frame_groups, reducer="sum"),
        "ob_data_combined": rebin_array_from_bin_groups(ob_data_combined, active_frame_groups, reducer="sum"),
        "ob_data_combined_variance": rebin_array_from_bin_groups(
            ob_data_combined_variance, active_frame_groups, reducer="sum"
        ),
        "dc_data_combined": rebin_array_from_bin_groups(dc_data_combined, active_frame_groups, reducer="sum"),
        "dc_data_combined_variance": rebin_array_from_bin_groups(
            dc_data_combined_variance, active_frame_groups, reducer="sum"
        ),
        "original_tof_array": tof_array,
        "original_lambda_array": lambda_array,
        "original_energy_array": energy_array,
        "active_frame_groups": active_frame_groups,
        "tof_array": None if bin_metadata is None else bin_metadata["mean_tof_array"],
        "lambda_array": None if bin_metadata is None else bin_metadata["mean_lambda_array"],
        "energy_array": None if bin_metadata is None else bin_metadata["mean_energy_array"],
        "output_suffix": output_suffix,
        "bin_metadata": bin_metadata,
    }


def load_images(
    master_dict=None,
    data_type=DataType.sample,
    verbose=False,
    active_frame_groups=None,
    use_experimental_uncertainties: bool = False,
):

    logging.info(f"Loading {data_type} data ...")
    for _run_number in master_dict.keys():
        logging.info(f"\tloading {data_type}# {_run_number} ... ")
        if verbose:
            display(HTML(f"Loading {data_type}# {_run_number} ..."))
        if active_frame_groups is None:
            master_dict[_run_number][MasterDictKeys.data] = load_data_using_multithreading(
                master_dict[_run_number][MasterDictKeys.list_tif], combine_tof=False
            )
            master_dict[_run_number][MasterDictKeys.variance] = None
        else:
            data, variance = load_rebinned_tof_data(
                list_tif=master_dict[_run_number][MasterDictKeys.list_tif],
                active_frame_groups=active_frame_groups,
                shutter_counts=master_dict[_run_number].get(MasterDictKeys.shutter_counts),
                use_experimental_uncertainties=use_experimental_uncertainties,
            )
            master_dict[_run_number][MasterDictKeys.data] = data
            master_dict[_run_number][MasterDictKeys.variance] = variance
        logging.info(f"\t{data_type}# {_run_number} loaded!")
        logging.info(f"\t{master_dict[_run_number][MasterDictKeys.data].shape = }")
        if verbose:
            display(HTML(f"{data_type}# {_run_number} loaded!"))
            display(HTML(f"{master_dict[_run_number][MasterDictKeys.data].shape = }"))


def calculate_roi_profile(data=None, roi=None):
    if (roi is None) or (data is None):
        return None

    array_data = np.asarray(data)
    if array_data.ndim < 3:
        return None

    x0 = roi.left
    y0 = roi.top
    width = roi.width
    height = roi.height
    return np.asarray(
        [np.sum(_data[y0 : y0 + height, x0 : x0 + width], dtype=np.float64) for _data in array_data],
        dtype=np.float64,
    )


def calculate_roi_intersection_profile(data=None, first_roi=None, second_roi=None):
    if data is None or first_roi is None or second_roi is None:
        return None
    left = max(first_roi.left, second_roi.left)
    top = max(first_roi.top, second_roi.top)
    right = min(first_roi.left + first_roi.width, second_roi.left + second_roi.width)
    bottom = min(first_roi.top + first_roi.height, second_roi.top + second_roi.height)
    if right <= left or bottom <= top:
        return np.zeros(np.asarray(data).shape[0], dtype=np.float64)
    return calculate_roi_profile(
        data=data,
        roi=Roi(left=left, top=top, width=right - left, height=bottom - top),
    )


def _load_black_filter_background_shape(background_shape_file: str) -> dict:
    if not background_shape_file:
        raise ValueError("Black-filter background correction requires a background shape CSV file.")

    background_shape_path = Path(background_shape_file).expanduser()
    if not background_shape_path.exists():
        raise FileNotFoundError(f"Black-filter background shape file not found: {background_shape_path}")

    background_shape = pd.read_csv(background_shape_path)
    required_columns = {
        "energy_eV",
        "sample_background_counts_fit",
        "ob_background_counts_fit",
    }
    missing_columns = required_columns.difference(background_shape.columns)
    if missing_columns:
        raise ValueError(
            f"Black-filter background shape file is missing required columns: {sorted(missing_columns)}"
        )

    energy = np.asarray(background_shape["energy_eV"], dtype=np.float64)
    sample_shape = np.asarray(background_shape["sample_background_counts_fit"], dtype=np.float64)
    ob_shape = np.asarray(background_shape["ob_background_counts_fit"], dtype=np.float64)

    valid = np.isfinite(energy) & np.isfinite(sample_shape) & np.isfinite(ob_shape)
    valid &= (energy > 0) & (sample_shape > 0) & (ob_shape > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Black-filter background shape file does not contain at least two valid positive rows.")

    order = np.argsort(energy[valid])
    return {
        "file": str(background_shape_path),
        "energy": energy[valid][order],
        "sample": sample_shape[valid][order],
        "ob": ob_shape[valid][order],
    }


def _evaluate_black_filter_background_shape(
    energy_array: np.ndarray,
    shape_energy: np.ndarray,
    shape_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a positive sampled background shape on the measured energy bins.

    The fitted background file is sampled on a dense energy grid. We interpolate
    that fitted model in log-log space. Energies outside the fitted range are left
    uncorrected by returning a zero background there.
    """
    measured_energy = np.asarray(energy_array, dtype=np.float64)
    evaluated_shape = np.zeros_like(measured_energy, dtype=np.float64)
    in_range = (
        np.isfinite(measured_energy)
        & (measured_energy >= shape_energy[0])
        & (measured_energy <= shape_energy[-1])
        & (measured_energy > 0)
    )
    if np.any(in_range):
        evaluated_shape[in_range] = np.exp(
            np.interp(
                np.log(measured_energy[in_range]),
                np.log(shape_energy),
                np.log(shape_counts),
            )
        )
    return evaluated_shape, in_range


def calculate_black_filter_background_corrected_spectrum(
    roi=None,
    sample_roi=None,
    ob_roi=None,
    sample_data=None,
    sample_variance=None,
    ob_data_combined=None,
    ob_data_combined_variance=None,
    energy_array=None,
    active_frame_groups=None,
    background_shape_file: str = None,
    anchor_energy_eV: float = 5.1044,
) -> dict:
    """Subtract a scaled black-filter background shape before ROI rebinning.

    Scaling uses one nearest measured energy bin:
        sample_scale = sample_roi(anchor) / sample_shape(anchor)
        ob_scale = ob_roi(anchor) / ob_shape(anchor)

    The returned uncertainty propagates only the measured sample/OB counting
    variance. The fitted background shape and single-bin scale factors are treated
    as exact model inputs.
    """
    sample_roi = sample_roi or roi
    ob_roi = ob_roi or sample_roi
    if sample_roi is None:
        raise ValueError("Black-filter background correction requires a ROI.")
    if sample_data is None or ob_data_combined is None:
        raise ValueError("Black-filter background correction requires sample and OB data.")
    if energy_array is None:
        raise ValueError("Black-filter background correction requires an energy axis.")

    measured_energy = np.asarray(energy_array, dtype=np.float64)
    sample_roi_counts = calculate_roi_profile(data=sample_data, roi=sample_roi)
    ob_roi_counts = calculate_roi_profile(data=ob_data_combined, roi=ob_roi)
    if sample_roi_counts is None or ob_roi_counts is None:
        raise ValueError("Unable to compute ROI counts for black-filter background correction.")
    if len(measured_energy) != len(sample_roi_counts):
        raise ValueError(
            "Black-filter background correction energy axis length does not match sample ROI profile length."
        )

    if sample_variance is None:
        sample_roi_variance = np.asarray(sample_roi_counts, dtype=np.float64)
    else:
        sample_roi_variance = calculate_roi_profile(data=sample_variance, roi=sample_roi)

    if ob_data_combined_variance is None:
        ob_roi_variance = np.asarray(ob_roi_counts, dtype=np.float64)
    else:
        ob_roi_variance = calculate_roi_profile(data=ob_data_combined_variance, roi=ob_roi)

    background_shape = _load_black_filter_background_shape(background_shape_file)
    sample_shape, sample_shape_in_range = _evaluate_black_filter_background_shape(
        measured_energy,
        background_shape["energy"],
        background_shape["sample"],
    )
    ob_shape, ob_shape_in_range = _evaluate_black_filter_background_shape(
        measured_energy,
        background_shape["energy"],
        background_shape["ob"],
    )

    finite_energy = np.isfinite(measured_energy)
    if not np.any(finite_energy):
        raise ValueError("Black-filter background correction found no finite measured energies.")

    anchor_index = int(np.nanargmin(np.abs(measured_energy - float(anchor_energy_eV))))
    anchor_energy = float(measured_energy[anchor_index])
    if not (sample_shape_in_range[anchor_index] and ob_shape_in_range[anchor_index]):
        raise ValueError(
            "Black-filter background anchor is outside the fitted background shape range: "
            f"anchor request {anchor_energy_eV} eV, nearest measured bin {anchor_energy} eV, "
            f"shape range [{background_shape['energy'][0]}, {background_shape['energy'][-1]}] eV."
        )
    if sample_shape[anchor_index] <= 0 or ob_shape[anchor_index] <= 0:
        raise ValueError("Black-filter background shape is non-positive at the selected anchor bin.")

    sample_scale = float(sample_roi_counts[anchor_index] / sample_shape[anchor_index])
    ob_scale = float(ob_roi_counts[anchor_index] / ob_shape[anchor_index])

    sample_background = sample_shape * sample_scale
    ob_background = ob_shape * ob_scale
    sample_corrected = sample_roi_counts - sample_background
    ob_corrected = ob_roi_counts - ob_background

    rebinned_sample_background = rebin_array_from_bin_groups(sample_background, active_frame_groups, reducer="sum")
    rebinned_ob_background = rebin_array_from_bin_groups(ob_background, active_frame_groups, reducer="sum")
    rebinned_sample_corrected = rebin_array_from_bin_groups(sample_corrected, active_frame_groups, reducer="sum")
    rebinned_ob_corrected = rebin_array_from_bin_groups(ob_corrected, active_frame_groups, reducer="sum")
    rebinned_sample_variance = rebin_array_from_bin_groups(sample_roi_variance, active_frame_groups, reducer="sum")
    rebinned_ob_variance = rebin_array_from_bin_groups(ob_roi_variance, active_frame_groups, reducer="sum")

    corrected_normalization = np.divide(
        rebinned_sample_corrected,
        rebinned_ob_corrected,
        out=np.zeros_like(rebinned_sample_corrected, dtype=np.float64),
        where=rebinned_ob_corrected != 0,
    )

    corrected_variance = np.zeros_like(rebinned_sample_corrected, dtype=np.float64)
    valid = rebinned_ob_corrected != 0
    corrected_variance[valid] = (
        rebinned_sample_variance[valid] / (rebinned_ob_corrected[valid] ** 2)
        + ((rebinned_sample_corrected[valid] ** 2) * rebinned_ob_variance[valid])
        / (rebinned_ob_corrected[valid] ** 4)
    )

    out_of_shape_range = int(
        np.count_nonzero(~(sample_shape_in_range & ob_shape_in_range) & finite_energy)
    )
    return {
        "black_filter_background_sample_roi_counts": rebinned_sample_background,
        "black_filter_background_ob_roi_counts": rebinned_ob_background,
        "black_filter_background_corrected_sample_roi_counts": rebinned_sample_corrected,
        "black_filter_background_corrected_sample_roi_uncertainty": np.sqrt(
            np.clip(rebinned_sample_variance, 0, None)
        ),
        "black_filter_background_corrected_ob_roi_counts": rebinned_ob_corrected,
        "black_filter_background_corrected_ob_roi_uncertainty": np.sqrt(
            np.clip(rebinned_ob_variance, 0, None)
        ),
        "black_filter_background_corrected_spectrum_normalization": corrected_normalization,
        "black_filter_background_corrected_spectrum_normalization_uncertainty": np.sqrt(
            np.clip(corrected_variance, 0, None)
        ),
        "black_filter_background_metadata": {
            "enabled": True,
            "shape_file": background_shape["file"],
            "scale_anchor_requested_eV": float(anchor_energy_eV),
            "scale_anchor_nearest_energy_eV": anchor_energy,
            "sample_scale_factor": sample_scale,
            "ob_scale_factor": ob_scale,
            "sample_anchor_roi_counts": float(sample_roi_counts[anchor_index]),
            "ob_anchor_roi_counts": float(ob_roi_counts[anchor_index]),
            "sample_shape_at_anchor": float(sample_shape[anchor_index]),
            "ob_shape_at_anchor": float(ob_shape[anchor_index]),
            "shape_energy_min_eV": float(background_shape["energy"][0]),
            "shape_energy_max_eV": float(background_shape["energy"][-1]),
            "uncertainty_note": (
                "Corrected uncertainty propagates measured sample/OB ROI counting variance only; "
                "background-shape and scale-factor uncertainty are not included."
            ),
            "out_of_shape_range_frame_count": out_of_shape_range,
        },
    }


def _extract_primary_shutter_count(shutter_counts=None) -> float:
    if shutter_counts is None:
        return None

    shutter_counts_array = np.asarray(shutter_counts, dtype=np.float64)
    shutter_counts_array = shutter_counts_array[shutter_counts_array > 0]
    if len(shutter_counts_array) == 0:
        return None
    return float(shutter_counts_array[0])


def recover_raw_counts_from_corrected_counts(corrected: np.ndarray, shutter_counts: float) -> tuple[np.ndarray, np.ndarray]:
    n_frames = corrected.shape[0]
    shutter_counts = float(shutter_counts)
    raw = np.zeros_like(corrected, dtype=np.float64)
    occupancy = np.zeros_like(corrected, dtype=np.float64)
    cumsum_raw = np.zeros(corrected.shape[1:], dtype=np.float64)

    for frame_index in range(n_frames):
        corrected_frame = corrected[frame_index].astype(np.float64)
        numerator = corrected_frame * (shutter_counts - cumsum_raw)
        denominator = shutter_counts + corrected_frame
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_frame = np.where(
                (denominator > 0) & (numerator >= 0),
                numerator / denominator,
                0.0,
            )
        raw[frame_index] = raw_frame
        cumsum_raw = cumsum_raw + raw_frame
        occupancy[frame_index] = cumsum_raw / shutter_counts

    return raw, occupancy


def calculate_detector_corrected_variance(data=None, shutter_counts=None):
    if data is None:
        return None

    primary_shutter_count = _extract_primary_shutter_count(shutter_counts)
    if primary_shutter_count is None:
        return None

    corrected = np.asarray(data, dtype=np.float64)
    _, occupancy = recover_raw_counts_from_corrected_counts(
        corrected=corrected,
        shutter_counts=primary_shutter_count,
    )

    denominator = 1.0 - occupancy
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(denominator > 0, corrected / denominator, 0.0)
    return np.maximum(variance, 0.0)


def calculate_data_variance(data=None, shutter_counts=None, use_experimental_uncertainties: bool = False):
    if data is None:
        return None

    if use_experimental_uncertainties:
        variance = calculate_detector_corrected_variance(data=data, shutter_counts=shutter_counts)
        if variance is not None:
            return variance

    return np.asarray(data, dtype=np.float64)


def calculate_combined_data_variance(
    master_dict: dict = None,
    use_proton_charge: bool = False,
    use_experimental_uncertainties: bool = False,
) -> np.ndarray:
    if not master_dict:
        return None

    run_numbers = list(master_dict.keys())
    if use_proton_charge:
        list_proton_charges = [master_dict[_run_number][MasterDictKeys.proton_charge] for _run_number in run_numbers]
        sum_proton_charge = np.sum(list_proton_charges)
        scale_factor = 1.0 / sum_proton_charge
    else:
        scale_factor = 1.0 / len(run_numbers)

    full_variance = []
    for _run_number in run_numbers:
        data = np.asarray(master_dict[_run_number][MasterDictKeys.data], dtype=np.float64)
        variance = master_dict[_run_number].get(MasterDictKeys.variance)
        if variance is None:
            variance = calculate_data_variance(
                data=data,
                shutter_counts=master_dict[_run_number].get(MasterDictKeys.shutter_counts),
                use_experimental_uncertainties=use_experimental_uncertainties,
            )
        full_variance.append(variance * scale_factor**2)

    return np.sum(np.asarray(full_variance), axis=0)


def calculate_dc_combined_variance(dc_master_dict: dict = None) -> np.ndarray:
    if not dc_master_dict:
        return None

    full_variance = [
        np.asarray(dc_master_dict[_dc_run_number][MasterDictKeys.data], dtype=np.float64)
        for _dc_run_number in dc_master_dict.keys()
    ]
    return np.sum(np.asarray(full_variance), axis=0) / (len(full_variance) ** 2)


def calculate_ratio_and_uncertainty(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_variance: np.ndarray,
    denominator_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator_variance = np.asarray(numerator_variance, dtype=np.float64)
    denominator_variance = np.asarray(denominator_variance, dtype=np.float64)

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )
    variance = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator != 0
    variance[valid] = (
        numerator_variance[valid] / (denominator[valid] ** 2)
        + ((numerator[valid] ** 2) * denominator_variance[valid]) / (denominator[valid] ** 4)
    )
    return ratio, np.sqrt(np.clip(variance, 0, None))


def calculate_bragg_edge_cd_background_profile(
    roi=None,
    sample_roi=None,
    ob_roi=None,
    raw_sample_data=None,
    raw_sample_variance=None,
    raw_ob_data=None,
    raw_ob_variance=None,
    sample_background_data=None,
    sample_background_variance=None,
    ob_background_data=None,
    ob_background_variance=None,
    mode: str = "Cd-filter background correction for Bragg edge mode",
    column_label: str = "Cd-filter",
    key_prefix: str = "bragg_edge_cd",
) -> dict:
    sample_roi = sample_roi or roi
    ob_roi = ob_roi or sample_roi
    if sample_roi is None:
        return None

    raw_sample_roi_counts = calculate_roi_profile(data=raw_sample_data, roi=sample_roi)
    raw_ob_roi_counts = calculate_roi_profile(data=raw_ob_data, roi=ob_roi)
    sample_background_roi_counts = calculate_roi_profile(data=sample_background_data, roi=sample_roi)
    ob_background_roi_counts = calculate_roi_profile(data=ob_background_data, roi=ob_roi)

    raw_sample_roi_variance = calculate_roi_profile(data=raw_sample_variance, roi=sample_roi)
    raw_ob_roi_variance = calculate_roi_profile(data=raw_ob_variance, roi=ob_roi)
    sample_background_roi_variance = calculate_roi_profile(data=sample_background_variance, roi=sample_roi)
    ob_background_roi_variance = calculate_roi_profile(data=ob_background_variance, roi=ob_roi)

    corrected_sample_roi_counts = raw_sample_roi_counts - sample_background_roi_counts
    corrected_ob_roi_counts = raw_ob_roi_counts - ob_background_roi_counts
    corrected_sample_roi_variance = raw_sample_roi_variance + sample_background_roi_variance
    corrected_ob_roi_variance = raw_ob_roi_variance + ob_background_roi_variance

    uncorrected_norm, uncorrected_unc = calculate_ratio_and_uncertainty(
        raw_sample_roi_counts,
        raw_ob_roi_counts,
        raw_sample_roi_variance,
        raw_ob_roi_variance,
    )
    corrected_norm, corrected_unc = calculate_ratio_and_uncertainty(
        corrected_sample_roi_counts,
        corrected_ob_roi_counts,
        corrected_sample_roi_variance,
        corrected_ob_roi_variance,
    )

    return {
        f"{key_prefix}_raw_sample_roi_counts": raw_sample_roi_counts,
        f"{key_prefix}_raw_sample_roi_uncertainty": np.sqrt(np.clip(raw_sample_roi_variance, 0, None)),
        f"{key_prefix}_raw_ob_roi_counts": raw_ob_roi_counts,
        f"{key_prefix}_raw_ob_roi_uncertainty": np.sqrt(np.clip(raw_ob_roi_variance, 0, None)),
        f"{key_prefix}_sample_background_roi_counts": sample_background_roi_counts,
        f"{key_prefix}_sample_background_roi_uncertainty": np.sqrt(
            np.clip(sample_background_roi_variance, 0, None)
        ),
        f"{key_prefix}_ob_background_roi_counts": ob_background_roi_counts,
        f"{key_prefix}_ob_background_roi_uncertainty": np.sqrt(np.clip(ob_background_roi_variance, 0, None)),
        f"{key_prefix}_corrected_sample_roi_counts": corrected_sample_roi_counts,
        f"{key_prefix}_corrected_sample_roi_uncertainty": np.sqrt(
            np.clip(corrected_sample_roi_variance, 0, None)
        ),
        f"{key_prefix}_corrected_ob_roi_counts": corrected_ob_roi_counts,
        f"{key_prefix}_corrected_ob_roi_uncertainty": np.sqrt(np.clip(corrected_ob_roi_variance, 0, None)),
        f"{key_prefix}_uncorrected_spectrum_normalization": uncorrected_norm,
        f"{key_prefix}_uncorrected_spectrum_normalization_uncertainty": uncorrected_unc,
        f"{key_prefix}_corrected_spectrum_normalization": corrected_norm,
        f"{key_prefix}_corrected_spectrum_normalization_uncertainty": corrected_unc,
        f"{key_prefix}_background_metadata": {
            "enabled": True,
            "mode": mode,
            "column_label": column_label,
            "key_prefix": key_prefix,
        },
    }


def extract_spectrum_normalization_values(spectrum_profile=None):
    if spectrum_profile is None:
        return None
    if isinstance(spectrum_profile, dict):
        return spectrum_profile.get("spectrum_normalization")
    return spectrum_profile


def extract_spectrum_normalization_uncertainty(spectrum_profile=None):
    if isinstance(spectrum_profile, dict):
        return spectrum_profile.get("spectrum_normalization_uncertainty")
    return None


def  calculate_ob_data_combined_used_by_spectrum_normalization(roi=None, ob_data_combined=None, verbose=False):

    logging.info(f"Calculating the ob_data_combined for spectrum normalization")
    if ob_data_combined is None:
        logging.info("\tno OB data provided. Skipping OB spectrum profile.")
        ob_data_combined_for_spectrum = None
    elif roi is not None:
        logging.info(f"\t{roi =}")
        ob_data_combined_for_spectrum = calculate_roi_profile(data=ob_data_combined, roi=roi)
        logging.info(f"\t{np.shape(ob_data_combined_for_spectrum) = }")
        logging.info(f"\t{np.shape(ob_data_combined) = }")

    else:
        logging.info(f"\tno roi provided! Skipping the normalization of spectrum.")
        ob_data_combined_for_spectrum = None

    if verbose and ob_data_combined is not None:
        display(HTML(f"{ob_data_combined.shape = }"))

    return ob_data_combined_for_spectrum


def correct_chips_alignment(data_combined=None, correct_chips_alignment_config=None, verbose=False):
    """
    correct the chips position (fill the gaps between the chips) using the dedicated library
    timepix_geometry_correction (https://github.com/ornlneutronimaging/timepix_geometry_correction)

    Args:
        data (np.ndarray): input data array
        config (dict): configuration dictionary for chips alignment
    Returns:
        np.ndarray: corrected data array
    """
    logging.info("Correcting chips alignment ...")
    if verbose:
        display(HTML("Correcting chips alignment ..."))

    timepix_geometry_correction_class = _get_timepix_geometry_correction()
    if timepix_geometry_correction_class is None:
        raise ModuleNotFoundError(
            "timepix_geometry_correction is required when chip alignment correction is enabled."
        )

    logging.info(f"\t{data_combined.shape = }")

    data_combined_corrected = np.zeros_like(data_combined)
    for _index, _data in enumerate(data_combined):
        o_corrector = timepix_geometry_correction_class(raw_images=_data,
                                                        config=correct_chips_alignment_config)
        data_corrected = o_corrector.correct()
    
        # remove useless dimension
        data = np.array([np.squeeze(_data) for _data in data_corrected])
        data_combined_corrected_squeezed = np.squeeze(data)
        
        data_combined_corrected[_index] = data_combined_corrected_squeezed

    logging.info(f"\t{data_combined_corrected.shape = }")
    
    logging.info("Chips alignment corrected!")
    
    if verbose:
        display(HTML("Chips alignment corrected!"))
  
    return data_combined_corrected


def correct_all_samples_chips_alignment(sample_master_dict=None, correct_chips_alignment_config=None, verbose=False):
    for _sample_run_number in sample_master_dict.keys():
        sample_master_dict[_sample_run_number][MasterDictKeys.data] = correct_chips_alignment(
            sample_master_dict[_sample_run_number][MasterDictKeys.data], 
            correct_chips_alignment_config,
            verbose=verbose
        )


def normalize_by_proton_charge(master_dict=None, run_number=None, data=None):
    if master_dict is not None and run_number is not None:
        logging.info("\t -> Normalized by proton charge")
        proton_charge = master_dict[run_number][MasterDictKeys.proton_charge]
        logging.info(f"\t\t proton charge: {proton_charge} C")
        logging.info(f"\t\t{type(proton_charge) = }")
        logging.info(f"\t\tbefore division: {data.dtype = }")
        data = data / proton_charge
        logging.info(f"\t\tafter division: {data.dtype = }")
        return data


def normalize_by_monitor_counts(master_dict=None, run_number=None, data=None):
    logging.info("\t -> Normalized by monitor counts")
    monitor_counts = master_dict[run_number][MasterDictKeys.monitor_counts]
    logging.info(f"\t\t monitor counts: {monitor_counts}")
    logging.info(f"\t\t{type(monitor_counts) = }")
    data = data / monitor_counts
    logging.info(f"{data.shape = }")
    return data



def preview_normalized_data(_sample_data, ob_data_combined, dc_data_combined, 
                            normalized_data, 
                            lambda_array, energy_array, 
                            detector_delay_us, _sample_run_number,
                            combine_samples=False,
                            _spectrum_normalized_data=None,
                            roi=None,
                            bin_metadata: dict = None):
   
    """preview normalized data"""
    def _source_frame_counts_for(data):
        if not bin_metadata or "source_frame_count_array" not in bin_metadata:
            return None
        source_frame_counts = np.asarray(bin_metadata["source_frame_count_array"], dtype=np.float64)
        if data is None or len(source_frame_counts) != np.asarray(data).shape[0]:
            return None
        return source_frame_counts

    def _data_per_source_frame(data):
        source_frame_counts = _source_frame_counts_for(data)
        if source_frame_counts is None:
            return data
        return np.asarray(data, dtype=np.float64) / source_frame_counts[:, None, None]

    def _profile_per_source_frame(profile, data):
        source_frame_counts = _source_frame_counts_for(data)
        if source_frame_counts is None:
            return profile, "Integrated counts"
        return profile / source_frame_counts, "Integrated counts per source frame"

    profile_xaxis_title = "Rebinned bin index" if bin_metadata is not None else "File image index"

    # display preview of normalized data
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=["Integrated Sample data", "Sample Profile"],
                       horizontal_spacing=0.15)
    sample_data_integrated = np.nanmean(_data_per_source_frame(_sample_data), axis=0)
    
    # Calculate 2-98% percentile range for better contrast
    vmin, vmax = np.percentile(sample_data_integrated, [2, 98])
    fig.add_trace(go.Heatmap(z=sample_data_integrated, 
                           colorscale="gray",
                           zmin=vmin,
                           zmax=vmax,
                           showscale=True,
                           showlegend=False,
                           colorbar=dict(x=0.45)), row=1, col=1)

    display(HTML(f"<h3>Preview of run {_sample_run_number}</h3>"))
    display(HTML(f"detector delay: {detector_delay_us:.2f} us"))

    sample_integrated1 = np.nansum(_sample_data, axis=1)
    sample_integrated = np.nansum(sample_integrated1, axis=1)
    sample_integrated, sample_profile_yaxis = _profile_per_source_frame(sample_integrated, _sample_data)
    fig.add_trace(go.Scatter(y=sample_integrated, 
                           mode='markers',
                           marker=dict(size=MARKERSIZE),
                           name="Sample"), row=1, col=2)
    fig.update_xaxes(title_text=profile_xaxis_title, row=1, col=2)
    fig.update_yaxes(title_text=sample_profile_yaxis, row=1, col=2)
    # Ensure equal aspect ratio for heatmap (square pixels)
    # Match the detector-image convention used by Matplotlib:
    # pixel row 0 is displayed at the top of the image.
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
    fig.show()

    if ob_data_combined is not None:
        fig2 = make_subplots(rows=1, cols=2,
                            subplot_titles=["OB integrated data", "OB Profile"],
                            horizontal_spacing=0.15)
        ob_data_integrated = np.nanmean(_data_per_source_frame(ob_data_combined), axis=0)

        # Calculate 2-98% percentile range for better contrast
        vmin, vmax = np.percentile(ob_data_integrated, [2, 98])
        fig2.add_trace(go.Heatmap(z=ob_data_integrated,
                                colorscale="gray",
                                zmin=vmin,
                                zmax=vmax,
                                showscale=True,
                                showlegend=False,
                                colorbar=dict(x=0.45)), row=1, col=1)

        ob_integrated1 = np.nansum(ob_data_combined, axis=1)
        ob_integrated = np.nansum(ob_integrated1, axis=1)
        ob_integrated, ob_profile_yaxis = _profile_per_source_frame(ob_integrated, ob_data_combined)
        fig2.add_trace(go.Scatter(y=ob_integrated,
                                mode='markers',
                                marker=dict(size=MARKERSIZE),
                                name="OB"), row=1, col=2)
        fig2.update_xaxes(title_text=profile_xaxis_title, row=1, col=2)
        fig2.update_yaxes(title_text=ob_profile_yaxis, row=1, col=2)
        # Ensure equal aspect ratio for heatmap (square pixels)
        fig2.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig2.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
        fig2.show()
    else:
        display(HTML("<span style='color:blue'>No OB run was provided; skipping OB preview.</span>"))

    if dc_data_combined is not None:
        fig3 = make_subplots(rows=1, cols=2, 
                           subplot_titles=["DC integrated data", "DC Profile"],
                           horizontal_spacing=0.15)
        dc_data_integrated = np.nanmean(_data_per_source_frame(dc_data_combined), axis=0)
        
        # Calculate 2-98% percentile range for better contrast
        vmin, vmax = np.percentile(dc_data_integrated, [2, 98])
        fig3.add_trace(go.Heatmap(z=dc_data_integrated, 
                                colorscale="gray",
                                zmin=vmin,
                                zmax=vmax,
                                showscale=True,
                                showlegend=False,
                                colorbar=dict(x=0.45)), row=1, col=1)

        dc_integrated1 = np.nansum(dc_data_combined, axis=1)
        dc_integrated = np.nansum(dc_integrated1, axis=1)
        dc_integrated, dc_profile_yaxis = _profile_per_source_frame(dc_integrated, dc_data_combined)
        fig3.add_trace(go.Scatter(y=dc_integrated, 
                                mode='markers',
                                marker=dict(size=MARKERSIZE),
                                name="DC"), row=1, col=2)
        fig3.update_xaxes(title_text=profile_xaxis_title, row=1, col=2)
        fig3.update_yaxes(title_text=dc_profile_yaxis, row=1, col=2)
        # Ensure equal aspect ratio for heatmap (square pixels)
        fig3.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig3.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
        fig3.show()

    # if not combine_samples:
    # Determine the profile label first
    if roi is not None:
        x0 = roi.left
        y0 = roi.top
        width = roi.width
        height = roi.height
        _label = "pixel by pixel normalization profile of ROI"
    else:
        _label = "pixel by pixel normalization profile of full image"
    
    fig4 = make_subplots(rows=1, cols=2, 
                        subplot_titles=["Integrated Normalized data", _label],
                        horizontal_spacing=0.15)
    normalized_data_integrated = np.nanmean(normalized_data[_sample_run_number], axis=0)
    
    # Use fixed range for normalized data (0-1) but apply percentile for better visualization
    vmin, vmax = np.percentile(normalized_data_integrated[~np.isnan(normalized_data_integrated)], [2, 98])
    vmin = max(vmin, 0)  # Keep lower bound at 0
    vmax = min(vmax, 1)  # Keep upper bound at 1
    
    fig4.add_trace(go.Heatmap(z=normalized_data_integrated, 
                            colorscale="gray",
                            zmin=vmin,
                            zmax=vmax,
                            showscale=True,
                            showlegend=False,
                            colorbar=dict(x=0.45)), row=1, col=1)

    if roi is not None:
        # Add rectangle overlay for ROI
        fig4.add_shape(type="rect",
                      x0=x0, y0=y0,
                      x1=x0+width, y1=y0+height,
                      line=dict(color="red", width=2),
                      fillcolor="rgba(0,0,0,0)",
                      row=1, col=1)

        profile_step1 = np.nanmean(normalized_data[_sample_run_number][:, y0:y0+height, x0:x0+width], axis=1)
        profile = np.nanmean(profile_step1, axis=1)

    else:
        profile_step1 = np.nanmean(normalized_data[_sample_run_number], axis=1)
        profile = np.nanmean(profile_step1, axis=1)

    fig4.add_trace(go.Scatter(y=profile, 
                            mode='markers',
                            marker=dict(size=MARKERSIZE),
                            showlegend=False), row=1, col=2)
    fig4.update_xaxes(title_text="File image index", row=1, col=2)
    fig4.update_yaxes(title_text="Transmission (a.u.)", row=1, col=2)
    # Ensure equal aspect ratio for heatmap (square pixels)
    fig4.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig4.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
    fig4.show()

    if lambda_array is not None:
        fig5 = make_subplots(rows=1, cols=2, 
                           subplot_titles=[f"{_label.split('profile of ')[-1]} counts vs Lambda", 
                                           f"{_label.split('profile of ')[-1]} counts vs energy"],
                           horizontal_spacing=0.15)
        logging.info(f"{np.shape(profile) = }")

        # Lambda plot
        fig5.add_trace(go.Scatter(x=lambda_array, y=profile, 
                                mode='markers',
                                marker=dict(symbol='star', size=MARKERSIZE),
                                showlegend=False), row=1, col=1)
        fig5.update_xaxes(title_text="Lambda (A)", row=1, col=1)
        fig5.update_yaxes(title_text="Transmission (a.u.)", row=1, col=1)

        # Energy plot
        fig5.add_trace(go.Scatter(x=energy_array, y=profile, 
                                mode='markers',
                                marker=dict(symbol='star', size=MARKERSIZE),
                                showlegend=False), row=1, col=2)
        fig5.update_xaxes(title_text="Energy (eV)", type="log", row=1, col=2)
        fig5.update_yaxes(title_text="Transmission (a.u.)", row=1, col=2)
        fig5.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
        fig5.show()

        spectrum_profile = extract_spectrum_normalization_values(_spectrum_normalized_data)
        spectrum_uncertainty = extract_spectrum_normalization_uncertainty(_spectrum_normalized_data)
        if spectrum_profile is not None:

            fig6 = make_subplots(rows=1, cols=2, 
                               subplot_titles=["Lambda vs ROI Spectrum", "Energy vs ROI Spectrum"],
                               horizontal_spacing=0.15)
            logging.info(f"{np.shape(profile) = }")

            error_y_dict = None
            if spectrum_uncertainty is not None:
                error_y_dict = dict(type='data', array=spectrum_uncertainty, visible=True)

            # Lambda plot
            fig6.add_trace(go.Scatter(x=lambda_array, y=spectrum_profile, 
                                    mode='markers',
                                    marker=dict(symbol='star', size=MARKERSIZE, color='red'),
                                    error_y=error_y_dict,
                                    name="spectrum normalization of ROI"), row=1, col=1)
            fig6.update_xaxes(title_text="Lambda (A)", row=1, col=1)
            fig6.update_yaxes(title_text="Transmission (a.u.)", row=1, col=1)
            logging.info(f"{lambda_array = }")

            # Energy plot
            fig6.add_trace(go.Scatter(x=energy_array, y=spectrum_profile, 
                                    mode='markers',
                                    marker=dict(symbol='star', size=MARKERSIZE, color='red'),
                                    error_y=error_y_dict,
                                    name="spectrum normalization of ROI", showlegend=False), row=1, col=2)
            fig6.update_xaxes(title_text="Energy (eV)", type="log", row=1, col=2)
            fig6.update_yaxes(title_text="Transmission (a.u.)", row=1, col=2)
            logging.info(f"{energy_array = }")
            fig6.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))
            fig6.show()



def get_detector_offset_from_nexus(nexus_path: str) -> float:
    """get the detector offset from the nexus file"""
    with h5py.File(nexus_path, "r") as hdf5_data:
        try:
            detector_offset_micros = hdf5_data["entry"]["DASlogs"]["BL10:Det:TH:DSPT1:TIDelay"]["value"][0]
            # detector_offset_micros = hdf5_data["entry"]["DASlogs"]["BL10:Det:DSP1:Trig2:Delay"]["value"][0]
        except KeyError:
            detector_offset_micros = None
    return detector_offset_micros


def get_run_number_from_nexus(nexus_path: str) -> int:
    """get the run number from the nexus file"""
    with h5py.File(nexus_path, "r") as hdf5_data:
        try:
            run_number = hdf5_data["entry"]["entry_identifier"][:][0].decode("utf8")
        except KeyError:
            run_number = None
    return run_number

def export_sample_images(
    output_folder,
    export_corrected_stack_of_sample_data,
    export_corrected_integrated_sample_data,
    _sample_run_number,
    _sample_data,
    spectra_file_name=None,
    spectra_array=None,
    output_suffix="",
    bin_metadata: dict = None,
):
    logging.info(f"> Exporting sample corrected images to {output_folder} ...")

    logging.info(f"\t{_sample_run_number = }")
    logging.info(f"\t{spectra_file_name = }")
    logging.info(f"\t{spectra_array = }")

    sample_output_folder = os.path.join(output_folder, f"sample_{_sample_run_number}{output_suffix}")
    os.makedirs(sample_output_folder, exist_ok=True)

    if export_corrected_stack_of_sample_data:
        output_stack_folder = os.path.join(sample_output_folder, "stack")
        logging.info(f"\tmaking folder {output_stack_folder}")
        os.makedirs(output_stack_folder, exist_ok=True)

        for _index, _data in enumerate(_sample_data):
            file_index = _index if bin_metadata is None else int(bin_metadata["active_bin_index_array"][_index])
            _output_file = os.path.join(output_stack_folder, f"image{file_index:04d}.tif")
            make_tiff(data=_data, filename=_output_file)
        logging.info(f"\t -> Exporting sample data to {output_stack_folder} is done!")

        if spectra_array is not None:
            # manually create the file for spectra
            spectra_file_name = os.path.join(output_stack_folder, f"manually_created_{SPECTRA_FILE_PREFIX}")
            _full_counts_array = np.empty_like(spectra_array)
            for _index, _data in enumerate(_sample_data):
                _full_counts_array[_index] = np.nansum(_data)
            pd_spectra = pd.DataFrame({
                "shutter_time": spectra_array,
                "counts": _full_counts_array
            })
            pd_spectra.to_csv(spectra_file_name, index=False, sep=",")
            logging.info(f"\t -> Exporting manually created spectra file to {spectra_file_name} is done!")

        else:
            shutil.copy(spectra_file_name, os.path.join(output_stack_folder))
            logging.info(f"\t -> Exporting spectra file {spectra_file_name} to {output_stack_folder} is done!")
        
        display(HTML(f"Created folder {output_stack_folder} for sample outputs!"))
    
    if export_corrected_integrated_sample_data:
        # making up the integrated sample data
        sample_data_integrated = np.nanmean(_sample_data, axis=0)
        full_file_name = os.path.join(sample_output_folder, "integrated.tif")
        logging.info(f"\t -> Exporting integrated sample data to {full_file_name} ...")
        make_tiff(data=sample_data_integrated, filename=full_file_name)
        logging.info(f"\t -> Exporting integrated sample data to {full_file_name} is done!")


def export_ob_images(
    ob_run_numbers,
    output_folder,
    export_corrected_stack_of_ob_data,
    export_corrected_integrated_ob_data,
    ob_data_combined,
    spectra_file_name=None,
    spectra_array=None,
    output_suffix="",
    bin_metadata: dict = None,
):
    """export ob images to the output folder"""
    logging.info(f"> Exporting combined ob images to {output_folder} ...")
    logging.info(f"\t{ob_run_numbers = }")
    list_ob_runs_number_only = [
        str(isolate_run_number_from_full_path(_ob_run_number)) for _ob_run_number in ob_run_numbers
    ]
    if len(list_ob_runs_number_only) == 1:
        ob_output_folder = os.path.join(output_folder, f"ob_{list_ob_runs_number_only[0]}{output_suffix}")
    else:
        str_list_ob_runs = "_".join(list_ob_runs_number_only)
        ob_output_folder = os.path.join(output_folder, f"ob_{str_list_ob_runs}{output_suffix}")
    os.makedirs(ob_output_folder, exist_ok=True)

    output_stack_folder = ""
    if export_corrected_stack_of_ob_data:
        output_stack_folder = os.path.join(ob_output_folder, "stack")
        logging.info(f"\tmaking folder {output_stack_folder}")
        os.makedirs(output_stack_folder, exist_ok=True)

    if export_corrected_integrated_ob_data:
        # making up the integrated ob data
        ob_data_integrated = np.nanmean(ob_data_combined, axis=0)
        full_file_name = os.path.join(ob_output_folder, "integrated.tif")
        logging.info(f"\t -> Exporting integrated ob data to {full_file_name} ...")
        make_tiff(data=ob_data_integrated, filename=full_file_name)
        logging.info(f"\t -> Exporting integrated ob data to {full_file_name} is done!")

    if export_corrected_stack_of_ob_data:
        logging.info(f"\t -> Exporting ob data to {output_stack_folder} ...")
        _list_data = ob_data_combined
        for _index, _data in enumerate(_list_data):
            file_index = _index if bin_metadata is None else int(bin_metadata["active_bin_index_array"][_index])
            _output_file = os.path.join(output_stack_folder, f"image{file_index:04d}.tif")
            make_tiff(data=_data, filename=_output_file)
        logging.info(f"\t -> Exporting ob data to {output_stack_folder} is done!")
        
        if spectra_array is not None:
            # manually create the file for spectra
            spectra_file_name = os.path.join(output_stack_folder, f"manually_created_{SPECTRA_FILE_PREFIX}")
            _full_counts_array = np.empty_like(spectra_array)
            for _index, _data in enumerate(ob_data_combined):
                _full_counts_array[_index] = np.nansum(_data)
            pd_spectra = pd.DataFrame({
                "shutter_time": spectra_array,
                "counts": _full_counts_array
            })
            pd_spectra.to_csv(spectra_file_name, index=False, sep=",")
            logging.info(f"\t -> Exporting manually created spectra file to {spectra_file_name} is done!")

        else:
            # copy spectra file to the output folder
            shutil.copy(spectra_file_name, os.path.join(output_stack_folder))
            logging.info(f"\t -> Exported spectra file {spectra_file_name} to {output_stack_folder}!")

    display(HTML(f"Created folder {output_stack_folder} for OB outputs!"))


# def normalization(sample_folder=None, ob_folder=None, output_folder="./", verbose=False):
#     pass


def make_tiff(data: list, filename: str = "", metadata: dict = None) -> None:
    new_image = Image.fromarray(np.array(data), mode="F")
    if metadata:
        new_image.save(filename, tiffinfo=metadata)
    else:
        new_image.save(filename)


def _safe_write_integrated_preview_png(
    output_file: str,
    integrated_image: np.ndarray,
    profile: np.ndarray = None,
    roi: Roi = None,
    vmin: float = None,
    vmax: float = None,
    profile_xaxis_title: str = "File image index",
    profile_title: str = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if profile is None:
            fig, axis = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
            image_axis = axis
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            image_axis = axes[0]

        image = image_axis.imshow(
            integrated_image,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
        )
        image_axis.set_title("Integrated Normalized data")
        fig.colorbar(image, ax=image_axis, fraction=0.046, pad=0.04)
        if roi is not None:
            image_axis.add_patch(
                Rectangle(
                    (roi.left, roi.top),
                    roi.width,
                    roi.height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )

        if profile is not None:
            profile_axis = axes[1]
            profile_axis.plot(np.arange(len(profile)), profile, ".", markersize=MARKERSIZE)
            profile_axis.set_title(profile_title or (
                "pixel by pixel normalization profile of ROI"
                if roi is not None
                else "pixel by pixel normalization profile of full image"
            ))
            profile_axis.set_xlabel(profile_xaxis_title)
            profile_axis.set_ylabel("Transmission (a.u.)")
            profile_axis.grid(True, alpha=0.25)

        fig.savefig(output_file, dpi=150)
        plt.close(fig)
    except Exception as error:
        logging.warning(
            "Unable to export integrated normalized preview PNG %s. Error: %s",
            output_file,
            error,
        )


def _extract_spectrum_normalization_profile(spectrum_normalized_data=None):
    if spectrum_normalized_data is None:
        return None

    if isinstance(spectrum_normalized_data, dict):
        for key in (
            "spectrum_normalization",
            "black_filter_background_corrected_spectrum_normalization",
        ):
            if spectrum_normalized_data.get(key) is not None:
                return np.asarray(spectrum_normalized_data[key], dtype=np.float64)

        for key in sorted(spectrum_normalized_data):
            if key.endswith("_corrected_spectrum_normalization"):
                values = spectrum_normalized_data.get(key)
                if values is not None:
                    return np.asarray(values, dtype=np.float64)
        return None

    return np.asarray(spectrum_normalized_data, dtype=np.float64)


def export_integrated_normalized_preview(
    output_folder: str,
    integrated_normalized_image: np.ndarray,
    normalized_stack: np.ndarray = None,
    roi: Roi = None,
    bin_metadata: dict = None,
    spectrum_normalized_data=None,
) -> None:
    """Export the integrated-normalized preview plot and its underlying arrays."""
    os.makedirs(output_folder, exist_ok=True)

    integrated_image = np.asarray(integrated_normalized_image, dtype=np.float64)
    data_file = os.path.join(output_folder, "normalized_integrated_data.txt")
    np.savetxt(
        data_file,
        integrated_image,
        fmt="%.8e",
        header="Integrated normalized image used for the preview heatmap.",
    )
    logging.info(f"\t -> Exported integrated normalized preview data to {data_file}")

    pixel_profile = None
    pixel_profile_label = None
    if normalized_stack is not None:
        stack = np.asarray(normalized_stack, dtype=np.float64)
        if roi is not None:
            x0 = roi.left
            y0 = roi.top
            width = roi.width
            height = roi.height
            profile_step1 = np.nanmean(stack[:, y0:y0 + height, x0:x0 + width], axis=1)
            pixel_profile = np.nanmean(profile_step1, axis=1)
            pixel_profile_label = "ROI mean pixel-normalized intensity"
        else:
            profile_step1 = np.nanmean(stack, axis=1)
            pixel_profile = np.nanmean(profile_step1, axis=1)
            pixel_profile_label = "full-image mean pixel-normalized intensity"

    spectrum_profile = _extract_spectrum_normalization_profile(spectrum_normalized_data)
    if spectrum_profile is not None and pixel_profile is not None and len(spectrum_profile) != len(pixel_profile):
        logging.warning(
            "Integrated preview spectrum profile length (%s) does not match pixel profile length (%s); "
            "falling back to the pixel profile.",
            len(spectrum_profile),
            len(pixel_profile),
        )
        spectrum_profile = None

    profile = pixel_profile if pixel_profile is not None else spectrum_profile
    profile_label = (
        pixel_profile_label
        if pixel_profile is not None
        else "ROI summed-count transmission"
    )
    if profile is not None:
        profile_xaxis_title = "Rebinned bin index" if bin_metadata is not None else "File image index"
        profile_dict = {
            "profile_index": np.arange(len(profile), dtype=int),
            "integrated_normalized_profile": profile,
        }
        if spectrum_profile is not None:
            profile_dict["roi_summed_count_transmission"] = spectrum_profile
        if pixel_profile is not None:
            profile_dict["pixel_mean_normalized_intensity"] = pixel_profile
        if bin_metadata is not None:
            profile_dict.update(
                {
                    "source_frame_count": bin_metadata.get("source_frame_count_array"),
                    "starting_tof (micros)": bin_metadata.get("starting_tof_array") * 1e6,
                    "ending_tof (micros)": bin_metadata.get("ending_tof_array") * 1e6,
                    "mean_tof (micros)": bin_metadata.get("mean_tof_array") * 1e6,
                    "mean_lambda (Angstroms)": bin_metadata.get("mean_lambda_array"),
                    "mean_energy (eV)": bin_metadata.get("mean_energy_array"),
                }
            )
        profile_dataframe = pd.DataFrame(profile_dict)
        profile_dataframe.attrs["profile description"] = profile_label
        if roi is not None:
            profile_dataframe.attrs["roi [left, top, width, height]"] = (
                f"{roi.left}, {roi.top}, {roi.width}, {roi.height}"
            )
        profile_file = os.path.join(output_folder, "normalized_integrated_profile.txt")
        with open(profile_file, "w") as profile_handle:
            for key, value in profile_dataframe.attrs.items():
                profile_handle.write(f"# {key}: {value}\n")
            profile_dataframe.to_csv(profile_handle, index=False)
        logging.info(f"\t -> Exported integrated normalized preview profile to {profile_file}")

    finite_values = integrated_image[np.isfinite(integrated_image)]
    if finite_values.size:
        vmin, vmax = np.percentile(finite_values, [2, 98])
        vmin = max(vmin, 0)
        vmax = min(vmax, 1)
    else:
        vmin, vmax = 0, 1

    if profile is None:
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Integrated Normalized data"])
        fig.add_trace(
            go.Heatmap(
                z=integrated_image,
                colorscale="gray",
                zmin=vmin,
                zmax=vmax,
                showscale=True,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        profile_title = "ROI normalization profiles" if roi is not None else (
            "full-image normalization profiles"
        )
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Integrated Normalized data", profile_title],
            horizontal_spacing=0.15,
        )
        fig.add_trace(
            go.Heatmap(
                z=integrated_image,
                colorscale="gray",
                zmin=vmin,
                zmax=vmax,
                showscale=True,
                showlegend=False,
                colorbar=dict(x=0.45),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                y=profile,
                mode="markers",
                marker=dict(size=MARKERSIZE),
                name=profile_label,
                showlegend=spectrum_profile is not None and pixel_profile is not None,
            ),
            row=1,
            col=2,
        )
        if spectrum_profile is not None and pixel_profile is not None:
            fig.add_trace(
                go.Scatter(
                    y=spectrum_profile,
                    mode="markers",
                    marker=dict(size=MARKERSIZE, opacity=0.45),
                    name="ROI summed-count transmission",
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(
            title_text=profile_xaxis_title,
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Transmission (a.u.)", row=1, col=2)

    if roi is not None:
        fig.add_shape(
            type="rect",
            x0=roi.left,
            y0=roi.top,
            x1=roi.left + roi.width,
            y1=roi.top + roi.height,
            line=dict(color="red", width=2),
            fillcolor="rgba(0,0,0,0)",
            row=1,
            col=1,
        )
    # Match the detector-image convention used by Matplotlib:
    # pixel row 0 is displayed at the top of the image.
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_layout(height=600, width=1200, margin=dict(l=50, r=50, t=80, b=50))

    html_file = os.path.join(output_folder, "normalized_integrated_preview.html")
    fig.write_html(html_file, include_plotlyjs="cdn")
    logging.info(f"\t -> Exported integrated normalized preview HTML to {html_file}")
    _safe_write_integrated_preview_png(
        output_file=os.path.join(output_folder, "normalized_integrated_preview.png"),
        integrated_image=integrated_image,
        profile=profile,
        roi=roi,
        vmin=vmin,
        vmax=vmax,
        profile_xaxis_title=profile_xaxis_title if profile is not None else "File image index",
        profile_title=profile_label if profile is not None else None,
    )


def isolate_run_number_from_full_path(run_number_full_path: str) -> str:
    """isolate the run number from the full path"""
    run_number = os.path.basename(run_number_full_path)
    return isolate_run_number(run_number)


def isolate_run_number(run_number_full_path: str) -> int:
    run_number = os.path.basename(run_number_full_path)
    # fixme needs to treat old data set and new data set
    # retrieve run number behind the string _Run_ in the file name
    split_1 = run_number.split("Run_")
    if len(split_1) == 2:
        run_number = split_1[1]
    else:
        split_2 = split_1[1].split("_")
        run_number = split_2[0]
    return int(run_number)


def init_master_dict(data_dictionary: dict) -> dict:
    master_dict = {}

    for _base_name in data_dictionary.keys():
        master_dict[_base_name] = {
            MasterDictKeys.nexus_path: data_dictionary[_base_name]["nexus"],
            MasterDictKeys.run_number: None,
            MasterDictKeys.frame_number: None,
            MasterDictKeys.data_path: data_dictionary[_base_name]["full_path"],
            MasterDictKeys.proton_charge: None,
            MasterDictKeys.shutter_counts: None,
            MasterDictKeys.matching_ob: [],
            MasterDictKeys.list_tif: [],
            MasterDictKeys.list_spectra: None,
            MasterDictKeys.spectra_file_name: None,
            MasterDictKeys.detector_delay_us: None,
            MasterDictKeys.data: None,
            MasterDictKeys.variance: None,
        }

    return master_dict


def retrieve_root_nexus_full_path(sample_folder: str) -> str:
    """retrieve the root nexus path from the sample folder"""
    clean_path = os.path.abspath(sample_folder)
    if clean_path[0] == "/":
        clean_path = clean_path[1:]

    path_splitted = clean_path.split("/")
    facility = path_splitted[0]
    instrument = path_splitted[1]
    ipts = path_splitted[2]

    return f"/{facility}/{instrument}/{ipts}/nexus/"


def update_dict_with_shutter_counts(master_dict: dict) -> tuple[dict, bool]:
    """update the master dict with shutter counts from shutter count file"""
    status_all_shutter_counts_found = True
    for run_number in master_dict.keys():
        data_path = master_dict[run_number][MasterDictKeys.data_path]
        _list_files = glob.glob(os.path.join(data_path, "*_ShutterCount.txt"))
        if len(_list_files) == 0:
            logging.info(f"Shutter count file not found for run {run_number}!")
            master_dict[run_number][MasterDictKeys.shutter_counts] = None
            status_all_shutter_counts_found = False
            continue
        else:
            shutter_count_file = _list_files[0]
            with open(shutter_count_file) as f:
                lines = f.readlines()
                list_shutter_counts = []
                for _line in lines:
                    _, _value = _line.strip().split("\t")
                    if _value == "0":
                        break
                    list_shutter_counts.append(float(_value))
                master_dict[run_number][MasterDictKeys.shutter_counts] = list_shutter_counts
    
    return master_dict, status_all_shutter_counts_found


def update_dict_with_spectra_files(master_dict: dict, spectra_array: np.ndarray = None) -> tuple[dict, bool]:
    """update the master dict with spectra values from spectra file"""
    status_all_spectra_found = True
    for _run_number in master_dict.keys():

        if spectra_array is not None:
            master_dict[_run_number][MasterDictKeys.list_spectra] = spectra_array
            master_dict[_run_number][MasterDictKeys.spectra_file_name] = "Provided array"

        else: 

            data_path = master_dict[_run_number][MasterDictKeys.data_path]
            _list_files = glob.glob(os.path.join(data_path, f"*_{SPECTRA_FILE_PREFIX}"))
            
            if len(_list_files) == 0:
                logging.info(f"Spectra file not found for run {_run_number}!")
                master_dict[_run_number][MasterDictKeys.list_spectra] = None
                status_all_spectra_found = False
                continue
            
            else:
                spectra_file = _list_files[0]
                master_dict[_run_number][MasterDictKeys.spectra_file_name] = spectra_file
                pd_spectra = pd.read_csv(spectra_file, sep=",", header=0)
                shutter_time = pd_spectra["shutter_time"].values
                master_dict[_run_number][MasterDictKeys.list_spectra] = shutter_time

    return master_dict, status_all_spectra_found


def update_dict_with_proton_charge(master_dict: dict) -> tuple[dict, bool]:
    """update the master dict with proton charge from nexus file"""
    status_all_proton_charge_found = True
    for _run_number in master_dict.keys():
        _nexus_path = master_dict[_run_number][MasterDictKeys.nexus_path]
        if _nexus_path is None or not os.path.exists(_nexus_path):
            logging.info(f"Nexus file not found for run {_run_number}!")
            master_dict[_run_number][MasterDictKeys.proton_charge] = None
            status_all_proton_charge_found = False
            continue

        try:
            with h5py.File(_nexus_path, "r") as hdf5_data:
                proton_charge = hdf5_data["entry"][MasterDictKeys.proton_charge][0] / 1e12
        except KeyError:
            proton_charge = None
            status_all_proton_charge_found = False
        master_dict[_run_number][MasterDictKeys.proton_charge] = np.float32(proton_charge)
    return status_all_proton_charge_found


def update_dict_with_monitor_counts(master_dict: dict) -> bool:
    """update the master dict with monitor counts from nexus file"""
    status_all_monitor_counts_found = True
    for _run_number in master_dict.keys():
        _nexus_path = master_dict[_run_number][MasterDictKeys.nexus_path]
        if _nexus_path is None or not os.path.exists(_nexus_path):
            logging.info(f"Nexus file not found for run {_run_number}!")
            master_dict[_run_number][MasterDictKeys.monitor_counts] = None
            status_all_monitor_counts_found = False
            continue

        try:
            with h5py.File(_nexus_path, "r") as hdf5_data:
                monitor_counts = hdf5_data["entry"]["monitor1"]["total_counts"][0]
        except KeyError:
            monitor_counts = None
            status_all_monitor_counts_found = False
        master_dict[_run_number][MasterDictKeys.monitor_counts] = np.float32(monitor_counts)
    return status_all_monitor_counts_found


def update_dict_with_list_of_images(master_dict: dict) -> dict:
    """update the master dict with list of images"""
    for _run_number in master_dict.keys():
        list_tif = retrieve_list_of_tif(master_dict[_run_number][MasterDictKeys.data_path])
        logging.info(f"Retrieved {len(list_tif)} tif files for run {_run_number}!")
        master_dict[_run_number][MasterDictKeys.list_tif] = list_tif


def get_list_run_number(data_folder: str) -> list:
    """get list of run numbers from the data folder"""
    list_runs = glob.glob(os.path.join(data_folder, "Run_*"))
    list_run_number = [int(os.path.basename(run).split("_")[1]) for run in list_runs]
    return list_run_number


def update_dict_with_nexus_full_path(nexus_root_path: str, instrument: str, master_dict: dict) -> dict:
    """create dict of nexus path for each run number"""
    for run_number in master_dict.keys():
        master_dict[run_number][MasterDictKeys.nexus_path] = os.path.join(
            nexus_root_path, f"{instrument}_{run_number}.nxs.h5"
        )


def update_with_nexus_metadata(master_dict: dict) -> dict:
    for run_number in master_dict.keys():
        nexus_path = master_dict[run_number][MasterDictKeys.nexus_path]
        if nexus_path is None or not os.path.exists(nexus_path):
            logging.info(f"Nexus file not found for run {run_number}!")
            continue
        detector_offset_us = get_detector_offset_from_nexus(nexus_path)
        master_dict[run_number][MasterDictKeys.detector_delay_us] = detector_offset_us

        _run_number = get_run_number_from_nexus(nexus_path)
        master_dict[run_number][MasterDictKeys.run_number] = _run_number


def update_dict_with_data_full_path(data_root_path: str, master_dict: dict) -> dict:
    """create dict of data path for each run number"""
    for run_number in master_dict.keys():
        master_dict[run_number][MasterDictKeys.data_path] = os.path.join(data_root_path, f"Run_{run_number}")


def create_master_dict(
    data_dictionary: dict = None,
    data_type: DataType = DataType.sample,
    data_root_path: str = None,
    instrument: str = "VENUS",
    spectra_array: np.ndarray = None,
) -> tuple[dict, StatusMetadata]:
    logging.info(f"Create {data_type} master dict of : {data_dictionary.keys()}")

    if len(list(data_dictionary.keys())) == 0:
        logging.warning("No run numbers found in data dictionary!")
        return {}, StatusMetadata()

    status_metadata = StatusMetadata()

    # retrieve metadata for each run number
    master_dict = init_master_dict(data_dictionary)

    logging.info("updating with nexus metadata")
    update_with_nexus_metadata(master_dict)

    logging.info("updating with shutter counts!")
    master_dict, all_shutter_counts_found = update_dict_with_shutter_counts(master_dict)
    if not all_shutter_counts_found:
        status_metadata.all_shutter_counts_found = False
    logging.info(f"{master_dict = }")

    # if all_shutter_counts_found:
    logging.info("updating with spectra values!")
    master_dict, all_spectra_found = update_dict_with_spectra_files(master_dict, spectra_array=spectra_array)
    if not all_spectra_found:
        status_metadata.all_spectra_found = False
    logging.info(f"{master_dict = }")

    logging.info("updating with monitor counts!")
    all_monitor_counts_found = update_dict_with_monitor_counts(master_dict)
    if not all_monitor_counts_found:
        status_metadata.all_monitor_counts_found = False
    logging.info(f"{master_dict = }")

    logging.info("updating with proton charge!")
    all_proton_charge_found = update_dict_with_proton_charge(master_dict)
    if not all_proton_charge_found:
        status_metadata.all_proton_charge_found = False
    logging.info(f"{master_dict = }")

    logging.info("updating with list of images!")
    update_dict_with_list_of_images(master_dict)

    return master_dict, status_metadata


def produce_list_shutter_for_each_image(list_time_spectra: list = None, list_shutter_counts: list = None) -> list:
    """produce list of shutter counts for each image"""

    delat_time_spectra = list_time_spectra[1] - list_time_spectra[0]
    list_index_jump = np.where(np.diff(list_time_spectra) > delat_time_spectra)[0]
    list_index_jump = np.where(np.diff(list_time_spectra) > 0.0001)[0]

    logging.info(f"\t{list_index_jump = }")
    logging.info(f"\t{list_shutter_counts = }")

    list_shutter_values_for_each_image = np.zeros(len(list_time_spectra), dtype=np.float32)
    if len(list_shutter_counts) == 1:  # resonance mode
        list_shutter_values_for_each_image.fill(list_shutter_counts[0])
        return list_shutter_values_for_each_image

    list_shutter_values_for_each_image[0 : list_index_jump[0] + 1].fill(list_shutter_counts[0])
    for _index in range(1, len(list_index_jump)):
        _start = list_index_jump[_index - 1]
        _end = list_index_jump[_index]
        list_shutter_values_for_each_image[_start + 1 : _end + 1].fill(list_shutter_counts[_index])

    list_shutter_values_for_each_image[list_index_jump[-1] + 1 :] = list_shutter_counts[-1]

    return list_shutter_values_for_each_image


def replace_zero_with_local_median(data: np.ndarray, 
                                  kernel_size: Tuple[int, int, int] = (3, 3, 3),
                                  max_iterations: int = 10) -> np.ndarray:
    """
    Replace 0 values in a 3D array using local median filtering.

    This function ONLY processes small neighborhoods around 0 pixels,
    avoiding expensive computation on the entire dataset.
    
    Parameters:
    -----------
    data : np.ndarray
        3D input array that may contain 0 values
    kernel_size : Tuple[int, int, int]
        Size of the kernel for median filtering in (height, width, depth) format
        Default is (3, 3, 3)
    max_iterations : int
        Maximum number of iterations to replace 0 values
        Default is 10
    
    Returns:
    --------
    np.ndarray
        Array with 0 values replaced by local median values
    """
    # Work on a copy to avoid modifying the original data
    result = data.copy()

    # Track initial 0 count
    initial_zero_count = np.sum(result == 0)
    if initial_zero_count == 0:
        return result

    logging.info(f"Starting efficient 0 replacement with kernel size {kernel_size}")
    logging.info(f"Initial 0 count: {initial_zero_count}")

    # Calculate padding for kernel
    pad_h, pad_w, pad_d = [k // 2 for k in kernel_size]
    
    for iteration in range(max_iterations):
        # Find current 0 locations
        zero_coords = np.argwhere(result == 0)
        current_zero_count = len(zero_coords)

        if current_zero_count == 0:
            logging.info(f"All 0 values replaced after {iteration} iterations")
            break

        logging.info(f"Iteration {iteration + 1}: {current_zero_count} 0 values remaining")

        # Process each 0 pixel individually
        replaced_count = 0
        for coord in zero_coords:
            y, x, z = coord
            
            # Define the local neighborhood bounds
            y_min = max(0, y - pad_h)
            y_max = min(result.shape[0], y + pad_h + 1)
            x_min = max(0, x - pad_w)
            x_max = min(result.shape[1], x + pad_w + 1)
            z_min = max(0, z - pad_d)
            z_max = min(result.shape[2], z + pad_d + 1)
            
            # Extract the local neighborhood
            neighborhood = result[y_min:y_max, x_min:x_max, z_min:z_max]
            
            # Get non-NaN values in the neighborhood
            valid_values = neighborhood[~np.isnan(neighborhood)]
            
            # If we have valid values, compute median and replace
            if len(valid_values) > 0:
                median_value = np.median(valid_values)
                result[y, x, z] = median_value
                replaced_count += 1

        logging.info(f"  Replaced {replaced_count} zero values in this iteration")

        # If no progress was made, break
        if replaced_count == 0:
            remaining_zero_count = np.sum(result == 0)
            logging.info(f"No progress made. {remaining_zero_count} zero values could not be replaced")
            logging.info("(These may be in regions with no valid neighbors)")
            break

    final_zero_count = np.sum(result == 0)
    logging.info(f"Final zero count: {final_zero_count}")
    logging.info(f"Successfully replaced {initial_zero_count - final_zero_count} zero values")

    return result


def combine_dc_images(dc_master_dict: dict) -> np.ndarray:
    """combine all dc images
    
    Parameters:
    -----------
    dc_master_dict : dict
        master dict of dc run numbers
    
    Returns:
    --------
    np.ndarray
        combined dc data
    
    """
    logging.info("Combining all dark current images")
    full_dc_data = []
    logging.info(f"dc_master_dict = {dc_master_dict}")

    if not dc_master_dict:
        return None

    for _dc_run_number in dc_master_dict.keys():
        logging.info(f"Combining dc# {_dc_run_number} ...")
        dc_data = np.array(dc_master_dict[_dc_run_number][MasterDictKeys.data], dtype=np.float32)
        full_dc_data.append(dc_data)
        logging.info(f"{np.shape(full_dc_data) = }")

    logging.info("Combining all dc images is done!")
    logging.info(f"\tbefore: {len(full_dc_data) = }")
    dc_data_combined = np.array(full_dc_data).mean(axis=0)
    logging.info(f"\tafter: {dc_data_combined.shape = }")

    return dc_data_combined


def combine_ob_images(
    ob_master_dict: dict,
    use_proton_charge: bool = False,
    # use_monitor_counts: bool = False,
    use_shutter_counts: bool = False,
    replace_ob_zeros_by_nan: bool = False,
    replace_ob_zeros_by_local_median: bool = False,
    kernel_size_for_local_median: Tuple[int, int, int] = (3, 3, 3), 
    max_iterations: int = 10,
) -> Tuple[np.ndarray, float]:
    """combine all ob images and correct by proton charge and shutter counts
    
    Parameters:
    -----------
    ob_master_dict : dict
        master dict of ob run numbers
    use_proton_charge : bool
        whether to correct by proton charge
    use_monitor_counts : bool
        whether to correct by monitor counts
    use_shutter_counts : bool
        whether to correct by shutter counts
    replace_ob_zeros_by_nan : bool
        whether to replace ob zeros by nan
    replace_ob_zeros_by_local_median : bool
        whether to replace ob zeros by local median
    kernel_size : Tuple[int, int, int]
        kernel size for local median filtering
    max_iterations : int
        maximum number of iterations for local median filtering
    
    Returns:
    --------
    np.ndarray
        combined ob data
    float
        total proton charge used for correction
    
    """

    logging.info("Combining all open beam images")
    logging.info(f"\tcorrecting by proton charge: {use_proton_charge}")
    # logging.info(f"\tcorrecting by monitor counts: {use_monitor_counts}")
    logging.info(f"\tshutter counts: {use_shutter_counts}")
    logging.info(f"\treplace ob zeros by nan: {replace_ob_zeros_by_nan}")
    logging.info(f"\treplace ob zeros by local median: {replace_ob_zeros_by_local_median}")
    logging.info(f"\tkernel size for local median: y:{kernel_size_for_local_median[0]}, "
                 f"x:{kernel_size_for_local_median[1]}, "
                 f"tof:{kernel_size_for_local_median[2]}")
    full_ob_data_corrected = []

    if use_proton_charge:
        # used for the weighted sum of the ob data
        logging.info("Getting proton charge for each ob run number:")
        list_proton_charges = []
        for _ob_run_number in ob_master_dict.keys():
            proton_charge = ob_master_dict[_ob_run_number][MasterDictKeys.proton_charge]
            list_proton_charges.append(proton_charge)
            logging.info(f"\t ob# {_ob_run_number}: proton charge = {proton_charge} C")

        sum_proton_charge = np.sum(list_proton_charges)
        logging.info(f"\t Total proton charge of all ob runs: {sum_proton_charge} C")
    else:
        sum_proton_charge = 1.0  # dummy value to avoid division by zero

    for _ob_run_number in ob_master_dict.keys():
        logging.info(f"Combining ob# {_ob_run_number} ...")
        ob_data = np.array(ob_master_dict[_ob_run_number][MasterDictKeys.data], dtype=np.float32)

        # get statistics of ob data
        data_shape = ob_data.shape
        nbr_pixels = data_shape[1] * data_shape[2]
        logging.info(" **** Statistics of ob data *****")
        number_of_zeros = np.sum(ob_data == 0)
        logging.info(f"\t ob data shape: {data_shape}")
        logging.info(f"\t Number of zeros in ob data: {number_of_zeros}")
        logging.info(f"\t Percentage of zeros in ob data: {number_of_zeros / (data_shape[0] * nbr_pixels) * 100:.2f}%")
        logging.info(f"\t Mean of ob data: {np.mean(ob_data)}")
        logging.info(f"\t maximum of ob data: {np.max(ob_data)}")
        logging.info(f"\t minimum of ob data: {np.min(ob_data)}")
        logging.info("**********************************")

        if use_proton_charge:
            logging.info("\t -> Normalized by proton charge")
            proton_charge = ob_master_dict[_ob_run_number][MasterDictKeys.proton_charge]
            logging.info(f"\t\t proton charge: {proton_charge} C")
            logging.info(f"\t\t{type(proton_charge) = }")
            logging.info(f"\t\tbefore division: {proton_charge.dtype = }")
            ob_data *= (proton_charge / sum_proton_charge) # weighted sum
            logging.info(f"\t\tafter division: {ob_data.dtype = }")
            logging.info(f"{ob_data.shape = }")

        # if use_monitor_counts:
        #     logging.info("\t -> Normalized by monitor counts")
        #     monitor_counts = ob_master_dict[_ob_run_number][MasterDictKeys.monitor_counts]
        #     logging.info(f"\t\t monitor counts: {monitor_counts}")
        #     logging.info(f"\t\t{type(monitor_counts) = }")
        #     ob_data = ob_data / monitor_counts
        #     logging.info(f"{ob_data.shape = }")

        if use_shutter_counts:
            logging.info("\t -> Normalized by shutter counts")

            list_shutter_values_for_each_image = produce_list_shutter_for_each_image(
                list_time_spectra=ob_master_dict[_ob_run_number][MasterDictKeys.list_spectra],
                list_shutter_counts=ob_master_dict[_ob_run_number][MasterDictKeys.shutter_counts],
            )

            logging.info(f"{list_shutter_values_for_each_image.shape = }")
            temp_ob_data = np.empty_like(ob_data, dtype=np.float32)
            for _index in range(len(list_shutter_values_for_each_image)):
                temp_ob_data[_index] = ob_data[_index] / list_shutter_values_for_each_image[_index]
            logging.info(f"{temp_ob_data.shape = }")
            ob_data = temp_ob_data.copy()

        # ob_data_combined = np.array(ob_data).mean(axis=0)
        # logging.info(f"{ob_data_combined.shape = }")

        if replace_ob_zeros_by_local_median:
            ob_data = replace_zero_with_local_median(ob_data, 
                                                     kernel_size=kernel_size_for_local_median, 
                                                     max_iterations=max_iterations)

        full_ob_data_corrected.append(ob_data)
        logging.info(f"{np.shape(full_ob_data_corrected) = }")

    logging.info("Combining all ob images is done!")
    logging.info(f"\tbefore: {len(full_ob_data_corrected) = }")
    if use_proton_charge:
        ob_data_combined = np.array(full_ob_data_corrected).sum(axis=0)
    else:
        ob_data_combined = np.array(full_ob_data_corrected).mean(axis=0)
        
    logging.info(f"\tafter: {ob_data_combined.shape = }")

    # remove zeros
    if replace_ob_zeros_by_nan:
        ob_data_combined[ob_data_combined == 0] = np.nan

    return ob_data_combined, sum_proton_charge


def combine_images(
    data_type: DataType.sample,
    master_dict: dict,
    use_proton_charge: bool = False,
    # use_monitor_counts: bool = False,
    # use_shutter_counts: bool = False,
    replace_zeros_by_nan: bool = False,
    replace_zeros_by_local_median: bool = False,
    kernel_size_for_local_median: Tuple[int, int, int] = (3, 3, 3), 
    max_iterations: int = 10,
) -> Tuple[np.ndarray, float]:
    """combine all images and correct by proton charge and shutter counts
    
    Parameters:
    -----------
    master_dict : dict
        master dict of run numbers
    use_proton_charge : bool
        whether to correct by proton charge
    use_monitor_counts : bool
        whether to correct by monitor counts
    use_shutter_counts : bool
        whether to correct by shutter counts
    replace_zeros_by_nan : bool
        whether to replace zeros by nan
    replace_zeros_by_local_median : bool
        whether to replace zeros by local median
    kernel_size : Tuple[int, int, int]
        kernel size for local median filtering
    max_iterations : int
        maximum number of iterations for local median filtering
    
    Returns:
    --------
    np.ndarray
        combined data
    float
        total proton charge used for correction
    
    """

    logging.info(f"Combining all {data_type} images")
    logging.info(f"\tcorrecting by proton charge: {use_proton_charge}")
    # logging.info(f"\tcorrecting by monitor counts: {use_monitor_counts}")
    # logging.info(f"\tshutter counts: {use_shutter_counts}")
    logging.info(f"\treplace zeros by nan: {replace_zeros_by_nan}")
    logging.info(f"\treplace zeros by local median: {replace_zeros_by_local_median}")
    logging.info(f"\tkernel size for local median: y:{kernel_size_for_local_median[0]}, "
                 f"x:{kernel_size_for_local_median[1]}, "
                 f"tof:{kernel_size_for_local_median[2]}")
    full_data_corrected = []

    if use_proton_charge:
        # Combined run is the total counts divided by total proton charge.
        # This is equivalent to exposure-time normalization for unequal-charge runs.
        logging.info(f"Getting proton charge for each {data_type} run number:")
        list_proton_charges = []
        for _run_number in master_dict.keys():
            proton_charge = master_dict[_run_number][MasterDictKeys.proton_charge]
            list_proton_charges.append(proton_charge)
            logging.info(f"\t {data_type}# {_run_number}: proton charge = {proton_charge} C")

        sum_proton_charge = np.sum(list_proton_charges)
        logging.info(f"\t Total proton charge of all {data_type} runs: {sum_proton_charge} C")
    else:
        sum_proton_charge = 1.0

    for _run_number in master_dict.keys():
        logging.info(f"Combining {data_type}# {_run_number} ...")
        data = np.array(master_dict[_run_number][MasterDictKeys.data], dtype=np.float32)

        # get statistics of data
        data_shape = data.shape
        nbr_pixels = data_shape[1] * data_shape[2]
        logging.info(f" **** Statistics of {data_type} data *****")
        number_of_zeros = np.sum(data == 0)
        logging.info(f"\t {data_type} data shape: {data_shape}")
        logging.info(f"\t Number of zeros in {data_type} data: {number_of_zeros}")
        logging.info(f"\t Percentage of zeros in {data_type} data: {number_of_zeros / (data_shape[0] * nbr_pixels) * 100:.2f}%")
        logging.info(f"\t Mean of {data_type} data: {np.mean(data)}")
        logging.info(f"\t maximum of {data_type} data: {np.max(data)}")
        logging.info(f"\t minimum of {data_type} data: {np.min(data)}")
        logging.info("**********************************")

        if replace_zeros_by_local_median:
            data = replace_zero_with_local_median(data, 
                                                kernel_size=kernel_size_for_local_median, 
                                                max_iterations=max_iterations)

        full_data_corrected.append(data)
        logging.info(f"{np.shape(full_data_corrected) = }")

    logging.info("Combining all ob images is done!")
    logging.info(f"\tbefore: {len(full_data_corrected) = }")
    if use_proton_charge:
        data_combined = np.array(full_data_corrected).sum(axis=0) / sum_proton_charge
    else:
        data_combined = np.array(full_data_corrected).mean(axis=0)
    
    # if use_proton_charge:
    #     data_combined = np.array(full_data_corrected).sum(axis=0)
    # else:
    #     data_combined = np.array(full_data_corrected).mean(axis=0)
        
    logging.info(f"\tafter: {data_combined.shape = }")

    # remove zeros
    if replace_zeros_by_nan:
        data_combined[data_combined == 0] = np.nan

    return data_combined, sum_proton_charge


# def normalization_by_shutter_counts(sample_master_dict=None,
#                 _sample_run_number=None,
#                 _sample_data=None,
#                 ob_master_dict=None,
#                 first_ob_run_number=None,
#             ):
#     """
#     Normalize sample data by shutter counts for each image.
    
#     This function normalizes sample data by dividing each image by its corresponding
#     shutter count value. The shutter count values are determined by mapping the time
#     spectra from the open beam data to the shutter counts recorded for the sample.
#     Images with zero shutter counts are replaced with NaN values to avoid division
#     by zero errors.
    
#     Parameters
#     ----------
#     sample_master_dict : dict, optional
#         Master dictionary containing sample run data and metadata including shutter counts.
#         Expected to have structure: {run_number: {MasterDictKeys.shutter_counts: list, ...}}
#     _sample_run_number : str or int, optional
#         The run number key to access the specific sample data in sample_master_dict
#     _sample_data : numpy.ndarray, optional
#         3D array of sample image data with shape (n_images, height, width)
#     ob_master_dict : dict, optional
#         Master dictionary containing open beam run data and metadata including time spectra.
#         Expected to have structure: {run_number: {MasterDictKeys.list_spectra: list, ...}}
#     first_ob_run_number : str or int, optional
#         The run number key to access the time spectra from the first open beam run
        
#     Returns
#     -------
#     numpy.ndarray
#         Normalized sample data array with same shape as input _sample_data.
#         Images corresponding to zero shutter counts are set to NaN.
        
#     Notes
#     -----
#     The normalization process involves:
#     1. Extracting time spectra from the open beam data
#     2. Extracting shutter counts from the sample data
#     3. Mapping shutter count values to each image based on time spectra
#     4. Dividing each sample image by its corresponding shutter count
#     5. Setting images with zero shutter counts to NaN
    
#     This function is typically used in neutron imaging data processing where
#     shutter counts represent the exposure time or beam intensity for each image.
    
#     Examples
#     --------
#     >>> normalized_data = normalization_by_shutter_counts(
#     ...     sample_master_dict=sample_dict,
#     ...     _sample_run_number="Run_12345",
#     ...     _sample_data=sample_images,
#     ...     ob_master_dict=ob_dict,
#     ...     first_ob_run_number="Run_12340"
#     ... )
#     """
        
#     list_shutter_values_for_each_image = produce_list_shutter_for_each_image(
#         list_time_spectra=ob_master_dict[first_ob_run_number][MasterDictKeys.list_spectra],
#         list_shutter_counts=sample_master_dict[_sample_run_number][MasterDictKeys.shutter_counts],
#     )

#     sample_data = []
#     for _sample, _shutter_value in zip(_sample_data, list_shutter_values_for_each_image, strict=False):
#         if _shutter_value != 0:
#             sample_data.append(_sample / _shutter_value)
#         else:
#             sample_data.append(np.nan)
#     _sample_data = np.array(sample_data)

#     return _sample_data


def perform_normalization(_sample_data=None, ob_data_combined=None, dc_data_combined=None):
    if ob_data_combined is None:
        if dc_data_combined is not None:
            raise ValueError("Dark-current subtraction requires an open-beam reference.")
        logging.info("normalization without OB: treating sample data as already normalized")
        _normalized_data = np.asarray(_sample_data).copy()
        _integrated_normalized_data = np.nanmean(_normalized_data, axis=0)
        return {
            'normalized_data': _normalized_data,
            'integrated_normalized_data': _integrated_normalized_data,
        }
    
    # working on each image (TOF) independently
    if dc_data_combined is not None:
        logging.info(f"normalization with DC subtraction")
        _normalized_data = np.divide(np.subtract(_sample_data, dc_data_combined), np.subtract(ob_data_combined, dc_data_combined), 
                                        out=np.zeros_like(_sample_data), 
                                        where=(ob_data_combined - dc_data_combined)!=0)
    else:
        logging.info(f"normalization without DC subtraction")
        _normalized_data = np.divide(_sample_data, ob_data_combined, 
                                        out=np.zeros_like(_sample_data), 
                                         where=ob_data_combined!=0)

    _normalized_data[ob_data_combined == 0] = 0
    
    # Integration of sample, dc and ob and then division
    if dc_data_combined is not None:
        logging.info(f"normalization with DC subtraction - integrated")
        _integrated_normalized_data = np.divide(np.subtract(np.sum(_sample_data, axis=0), np.sum(dc_data_combined, axis=0)),    
                                                np.subtract(np.sum(ob_data_combined, axis=0), np.sum(dc_data_combined, axis=0)), 
                                        out=np.zeros_like(np.sum(_sample_data, axis=0)), 
                                        where=(np.sum(ob_data_combined, axis=0) - np.sum(dc_data_combined, axis=0))!=0)
    else:
        logging.info(f"normalization without DC subtraction - integrated")
        _integrated_normalized_data = np.divide(np.sum(_sample_data, axis=0), np.sum(ob_data_combined, axis=0), 
                                        out=np.zeros_like(np.sum(_sample_data, axis=0)), 
                                         where=np.sum(ob_data_combined, axis=0)!=0)

    return {'normalized_data': _normalized_data,
            'integrated_normalized_data': _integrated_normalized_data}


def perform_spectrum_normalization(
    roi=None,
    sample_roi=None,
    ob_roi=None,
    sample_data=None,
    sample_variance=None,
    ob_data_combined_for_spectrum=None,
    ob_data_combined_variance_for_spectrum=None,
    dc_data_combined=None,
    dc_data_combined_for_spectrum=None,
    dc_data_combined_variance=None,
    dc_data_combined_variance_for_spectrum=None,
    sample_dc_data_combined_for_spectrum=None,
    ob_dc_data_combined_for_spectrum=None,
    sample_dc_data_combined_variance_for_spectrum=None,
    ob_dc_data_combined_variance_for_spectrum=None,
    sample_ob_dc_covariance_for_spectrum=None,
    black_filter_background_profile=None,
    bragg_edge_cd_background_profile=None,
    measured_background_profiles=None,
):
    sample_roi = sample_roi or roi
    ob_roi = ob_roi or sample_roi
    if sample_roi is None:
        return None

    sample_roi_counts = calculate_roi_profile(data=sample_data, roi=sample_roi)
    if sample_variance is None:
        sample_roi_variance = np.asarray(sample_roi_counts, dtype=np.float64)
    else:
        sample_roi_variance = calculate_roi_profile(data=sample_variance, roi=sample_roi)

    if ob_data_combined_for_spectrum is None:
        roi_pixel_count = float(sample_roi.width * sample_roi.height)
        ob_roi_counts = np.full_like(sample_roi_counts, roi_pixel_count, dtype=np.float64)
        ob_roi_variance = np.zeros_like(sample_roi_counts, dtype=np.float64)
    else:
        ob_roi_counts = np.asarray(ob_data_combined_for_spectrum, dtype=np.float64)
    if ob_data_combined_for_spectrum is not None and ob_data_combined_variance_for_spectrum is None:
        ob_roi_variance = np.asarray(ob_roi_counts, dtype=np.float64)
    elif ob_data_combined_for_spectrum is not None:
        ob_roi_variance = np.asarray(ob_data_combined_variance_for_spectrum, dtype=np.float64)

    spectrum_result = {
        "sample_roi_counts": sample_roi_counts,
        "sample_roi_uncertainty": np.sqrt(np.clip(sample_roi_variance, 0, None)),
        "ob_roi_counts": ob_roi_counts,
        "ob_roi_uncertainty": np.sqrt(np.clip(ob_roi_variance, 0, None)),
    }

    if dc_data_combined is not None:
        sample_dc_roi_counts = sample_dc_data_combined_for_spectrum
        if sample_dc_roi_counts is None:
            sample_dc_roi_counts = calculate_roi_profile(data=dc_data_combined, roi=sample_roi)
        if sample_dc_roi_counts is None:
            sample_dc_roi_counts = dc_data_combined_for_spectrum
        sample_dc_roi_counts = np.asarray(sample_dc_roi_counts, dtype=np.float64)

        ob_dc_roi_counts = ob_dc_data_combined_for_spectrum
        if ob_dc_roi_counts is None:
            ob_dc_roi_counts = calculate_roi_profile(data=dc_data_combined, roi=ob_roi)
        if ob_dc_roi_counts is None:
            ob_dc_roi_counts = dc_data_combined_for_spectrum
        ob_dc_roi_counts = np.asarray(ob_dc_roi_counts, dtype=np.float64)

        sample_dc_roi_variance = sample_dc_data_combined_variance_for_spectrum
        if sample_dc_roi_variance is None and dc_data_combined_variance is not None:
            sample_dc_roi_variance = calculate_roi_profile(
                data=dc_data_combined_variance,
                roi=sample_roi,
            )
        if sample_dc_roi_variance is None:
            sample_dc_roi_variance = dc_data_combined_variance_for_spectrum
        if sample_dc_roi_variance is None:
            sample_dc_roi_variance = sample_dc_roi_counts
        sample_dc_roi_variance = np.asarray(sample_dc_roi_variance, dtype=np.float64)

        ob_dc_roi_variance = ob_dc_data_combined_variance_for_spectrum
        if ob_dc_roi_variance is None and dc_data_combined_variance is not None:
            ob_dc_roi_variance = calculate_roi_profile(
                data=dc_data_combined_variance,
                roi=ob_roi,
            )
        if ob_dc_roi_variance is None:
            ob_dc_roi_variance = dc_data_combined_variance_for_spectrum
        if ob_dc_roi_variance is None:
            ob_dc_roi_variance = ob_dc_roi_counts
        ob_dc_roi_variance = np.asarray(ob_dc_roi_variance, dtype=np.float64)

        dc_covariance = sample_ob_dc_covariance_for_spectrum
        if dc_covariance is None and dc_data_combined_variance is not None:
            dc_covariance = calculate_roi_intersection_profile(
                data=dc_data_combined_variance,
                first_roi=sample_roi,
                second_roi=ob_roi,
            )
        if dc_covariance is None:
            same_roi = (
                sample_roi.left,
                sample_roi.top,
                sample_roi.width,
                sample_roi.height,
            ) == (ob_roi.left, ob_roi.top, ob_roi.width, ob_roi.height)
            dc_covariance = sample_dc_roi_variance if same_roi else np.zeros_like(sample_dc_roi_variance)
        dc_covariance = np.asarray(dc_covariance, dtype=np.float64)

        numerator = sample_roi_counts - sample_dc_roi_counts
        denominator = ob_roi_counts - ob_dc_roi_counts
        numerator_variance = sample_roi_variance + sample_dc_roi_variance
        denominator_variance = ob_roi_variance + ob_dc_roi_variance

        spectrum_normalization = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(sample_roi_counts, dtype=np.float64),
            where=denominator != 0,
        )

        spectrum_normalization_variance = np.zeros_like(sample_roi_counts, dtype=np.float64)
        valid = denominator != 0
        spectrum_normalization_variance[valid] = (
            numerator_variance[valid] / (denominator[valid] ** 2)
            + ((numerator[valid] ** 2) * denominator_variance[valid]) / (denominator[valid] ** 4)
            - (2.0 * numerator[valid] * dc_covariance[valid]) / (denominator[valid] ** 3)
        )

        spectrum_result.update(
            {
                "dc_roi_counts": sample_dc_roi_counts,
                "dc_roi_uncertainty": np.sqrt(np.clip(sample_dc_roi_variance, 0, None)),
                "sample_dc_roi_counts": sample_dc_roi_counts,
                "sample_dc_roi_uncertainty": np.sqrt(np.clip(sample_dc_roi_variance, 0, None)),
                "ob_dc_roi_counts": ob_dc_roi_counts,
                "ob_dc_roi_uncertainty": np.sqrt(np.clip(ob_dc_roi_variance, 0, None)),
                "sample_minus_dc_roi_counts": numerator,
                "sample_minus_dc_roi_uncertainty": np.sqrt(np.clip(numerator_variance, 0, None)),
                "ob_minus_dc_roi_counts": denominator,
                "ob_minus_dc_roi_uncertainty": np.sqrt(np.clip(denominator_variance, 0, None)),
            }
        )
    else:
        denominator = ob_roi_counts
        spectrum_normalization = np.divide(
            sample_roi_counts,
            denominator,
            out=np.zeros_like(sample_roi_counts, dtype=np.float64),
            where=denominator != 0,
        )

        spectrum_normalization_variance = np.zeros_like(sample_roi_counts, dtype=np.float64)
        valid = denominator != 0
        spectrum_normalization_variance[valid] = (
            sample_roi_variance[valid] / (denominator[valid] ** 2)
            + ((sample_roi_counts[valid] ** 2) * ob_roi_variance[valid]) / (denominator[valid] ** 4)
        )

    spectrum_result["spectrum_normalization"] = spectrum_normalization
    spectrum_result["spectrum_normalization_uncertainty"] = np.sqrt(
        np.clip(spectrum_normalization_variance, 0, None)
    )
    if black_filter_background_profile is not None:
        spectrum_result.update(black_filter_background_profile)
    if bragg_edge_cd_background_profile is not None:
        spectrum_result.update(bragg_edge_cd_background_profile)
    for measured_background_profile in measured_background_profiles or []:
        if measured_background_profile is not None:
            spectrum_result.update(measured_background_profile)
    logging.info(f"{np.shape(spectrum_result['spectrum_normalization']) = }")
    return spectrum_result


def export_normalized_data(ob_master_dict=None, 
                sample_master_dict=None, 
                _sample_run_number=None,
                normalized_data=None, 
                integrated_normalized_data=None,
                _spectrum_normalized_data=None,
                tof_array=None,
                lambda_array=None, 
                energy_array=None, 
                output_folder="./", 
                export_corrected_stack_of_normalized_data=False,
                export_corrected_integrated_normalized_data=False,
                roi=None,
                sample_roi=None,
                ob_roi=None,
                spectra_array=None,
                spectra_file=None,
                output_suffix="",
                bin_metadata: dict = None,
                uncertainty_model_label: str = None):

    logging.info("Exporting normalized data ...")

    list_ob_runs = list(ob_master_dict.keys())
    str_ob_runs = "_".join([str(_ob_run_number) for _ob_run_number in list_ob_runs])
    if not str_ob_runs:
        str_ob_runs = "no_open_beam"
    full_output_folder = os.path.join(
        output_folder,
        _safe_normalization_output_basename(_sample_run_number, str_ob_runs, output_suffix),
    )
    full_output_folder = os.path.abspath(full_output_folder)
    os.makedirs(full_output_folder, exist_ok=True)

    sample_roi = sample_roi or roi
    ob_roi = ob_roi or sample_roi
    if sample_roi is not None:
        logging.info(f"\t -> exporting the spectrum normalization")
        logging.info(f"{sample_roi =}")
        logging.info(f"{ob_roi =}")
        x0 = sample_roi.left
        y0 = sample_roi.top
        width = sample_roi.width
        height = sample_roi.height
        full_file_name = os.path.join(full_output_folder, "spectrum_normalization_profile.txt")
        if bin_metadata is not None:
            bin_index_array = bin_metadata["active_bin_index_array"]
            source_frame_count_array = bin_metadata.get(
                "source_frame_count_array",
                np.ones(len(bin_index_array), dtype=int),
            )
            starting_tof_micros = bin_metadata["starting_tof_array"] * 1e6
            ending_tof_micros = bin_metadata["ending_tof_array"] * 1e6
            effective_tof_span_micros = ending_tof_micros - starting_tof_micros
            mean_tof_micros = bin_metadata["mean_tof_array"] * 1e6
            mean_lambda = bin_metadata["mean_lambda_array"]
            mean_energy = bin_metadata["mean_energy_array"]
        else:
            bin_index_array = np.arange(len(lambda_array))
            source_frame_count_array = np.ones(len(bin_index_array), dtype=int)
            starting_tof_micros = None if tof_array is None else np.asarray(tof_array, dtype=np.float64) * 1e6
            ending_tof_micros = None if tof_array is None else np.asarray(tof_array, dtype=np.float64) * 1e6
            effective_tof_span_micros = np.zeros(len(bin_index_array), dtype=np.float64)
            mean_tof_micros = None if tof_array is None else np.asarray(tof_array, dtype=np.float64) * 1e6
            mean_lambda = lambda_array
            mean_energy = energy_array

        pd_dataframe_dict = {
            "bin index": bin_index_array,
            "source frame count": source_frame_count_array,
            "starting_tof (micros)": starting_tof_micros,
            "ending_tof (micros)": ending_tof_micros,
            "effective_tof_span (micros)": effective_tof_span_micros,
            "mean_tof (micros)": mean_tof_micros,
            "mean_lambda (Angstroms)": mean_lambda,
            "mean_energy (eV)": mean_energy,
        }

        if isinstance(_spectrum_normalized_data, dict):
            ordered_columns = [
                ("sample_roi_counts", "sample ROI counts"),
                ("sample_roi_uncertainty", "sample ROI uncertainty"),
                ("ob_roi_counts", "ob ROI counts"),
                ("ob_roi_uncertainty", "ob ROI uncertainty"),
                ("dc_roi_counts", "dc ROI counts"),
                ("dc_roi_uncertainty", "dc ROI uncertainty"),
                ("sample_dc_roi_counts", "sample DC ROI counts"),
                ("sample_dc_roi_uncertainty", "sample DC ROI uncertainty"),
                ("ob_dc_roi_counts", "OB DC ROI counts"),
                ("ob_dc_roi_uncertainty", "OB DC ROI uncertainty"),
                ("sample_minus_dc_roi_counts", "sample minus DC ROI counts"),
                ("sample_minus_dc_roi_uncertainty", "sample minus DC ROI uncertainty"),
                ("ob_minus_dc_roi_counts", "OB minus DC ROI counts"),
                ("ob_minus_dc_roi_uncertainty", "OB minus DC ROI uncertainty"),
                ("spectrum_normalization", "spectrum normalization"),
                ("spectrum_normalization_uncertainty", "spectrum normalization uncertainty"),
                ("black_filter_background_sample_roi_counts", "black-filter background sample ROI counts"),
                ("black_filter_background_ob_roi_counts", "black-filter background OB ROI counts"),
                (
                    "black_filter_background_corrected_sample_roi_counts",
                    "black-filter corrected sample ROI counts",
                ),
                (
                    "black_filter_background_corrected_sample_roi_uncertainty",
                    "black-filter corrected sample ROI uncertainty",
                ),
                (
                    "black_filter_background_corrected_ob_roi_counts",
                    "black-filter corrected OB ROI counts",
                ),
                (
                    "black_filter_background_corrected_ob_roi_uncertainty",
                    "black-filter corrected OB ROI uncertainty",
                ),
                (
                    "black_filter_background_corrected_spectrum_normalization",
                    "black-filter corrected spectrum normalization",
                ),
                (
                    "black_filter_background_corrected_spectrum_normalization_uncertainty",
                    "black-filter corrected spectrum normalization uncertainty",
                ),
            ]
            measured_background_metadata = []
            for metadata_key, metadata in _spectrum_normalized_data.items():
                if (
                    metadata_key.endswith("_background_metadata")
                    and metadata_key != "black_filter_background_metadata"
                    and isinstance(metadata, dict)
                ):
                    prefix = metadata.get("key_prefix") or metadata_key.removesuffix("_background_metadata")
                    measured_background_metadata.append((prefix, metadata))

            for background_prefix, background_metadata in measured_background_metadata:
                background_column_label = background_metadata.get("column_label", "Cd-filter")
                ordered_columns += [
                    (
                        f"{background_prefix}_raw_sample_roi_counts",
                        f"{background_column_label} raw sample ROI counts",
                    ),
                    (
                        f"{background_prefix}_raw_sample_roi_uncertainty",
                        f"{background_column_label} raw sample ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_raw_ob_roi_counts",
                        f"{background_column_label} raw OB ROI counts",
                    ),
                    (
                        f"{background_prefix}_raw_ob_roi_uncertainty",
                        f"{background_column_label} raw OB ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_sample_background_roi_counts",
                        f"{background_column_label} sample background ROI counts",
                    ),
                    (
                        f"{background_prefix}_sample_background_roi_uncertainty",
                        f"{background_column_label} sample background ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_ob_background_roi_counts",
                        f"{background_column_label} OB background ROI counts",
                    ),
                    (
                        f"{background_prefix}_ob_background_roi_uncertainty",
                        f"{background_column_label} OB background ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_corrected_sample_roi_counts",
                        f"{background_column_label} corrected sample ROI counts",
                    ),
                    (
                        f"{background_prefix}_corrected_sample_roi_uncertainty",
                        f"{background_column_label} corrected sample ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_corrected_ob_roi_counts",
                        f"{background_column_label} corrected OB ROI counts",
                    ),
                    (
                        f"{background_prefix}_corrected_ob_roi_uncertainty",
                        f"{background_column_label} corrected OB ROI uncertainty",
                    ),
                    (
                        f"{background_prefix}_uncorrected_spectrum_normalization",
                        f"{background_column_label} uncorrected spectrum normalization",
                    ),
                    (
                        f"{background_prefix}_uncorrected_spectrum_normalization_uncertainty",
                        f"{background_column_label} uncorrected spectrum normalization uncertainty",
                    ),
                    (
                        f"{background_prefix}_corrected_spectrum_normalization",
                        f"{background_column_label} corrected spectrum normalization",
                    ),
                    (
                        f"{background_prefix}_corrected_spectrum_normalization_uncertainty",
                        f"{background_column_label} corrected spectrum normalization uncertainty",
                    ),
                ]
            for source_key, output_key in ordered_columns:
                values = _spectrum_normalized_data.get(source_key)
                if values is not None:
                    pd_dataframe_dict[output_key] = values
        else:
            pd_dataframe_dict["spectrum normalization"] = _spectrum_normalized_data

        pd_dataframe = pd.DataFrame(pd_dataframe_dict)
        pd_dataframe.attrs['roi [left, top, width, height]'] = f"{x0}, {y0}, {width}, {height}"
        pd_dataframe.attrs['sample roi [left, top, width, height]'] = (
            f"{sample_roi.left}, {sample_roi.top}, {sample_roi.width}, {sample_roi.height}"
        )
        pd_dataframe.attrs['ob roi [left, top, width, height]'] = (
            f"{ob_roi.left}, {ob_roi.top}, {ob_roi.width}, {ob_roi.height}"
        )
        pd_dataframe.attrs['uncertainty model'] = uncertainty_model_label or (
            "Poisson counting statistics; proton charge treated as an exact scale factor"
        )
        if bin_metadata is not None:
            pd_dataframe.attrs["rebin snap to native grid"] = bin_metadata.get(
                "rebin_snap_to_native_grid",
                False,
            )
        if isinstance(_spectrum_normalized_data, dict):
            black_filter_background_metadata = _spectrum_normalized_data.get("black_filter_background_metadata")
            if black_filter_background_metadata:
                for key, value in black_filter_background_metadata.items():
                    pd_dataframe.attrs[f"black-filter background {key}"] = value
            for metadata_key, metadata in _spectrum_normalized_data.items():
                if (
                    metadata_key.endswith("_background_metadata")
                    and metadata_key != "black_filter_background_metadata"
                    and isinstance(metadata, dict)
                ):
                    metadata_prefix = metadata.get(
                        "mode",
                        "Cd-filter background correction for Bragg edge mode",
                    )
                    for key, value in metadata.items():
                        pd_dataframe.attrs[f"{metadata_prefix} {key}"] = value
                        
        with open(full_file_name, 'w') as f:
            for key, value in pd_dataframe.attrs.items():
                f.write(f"# {key}: {value}\n")
            pd_dataframe.to_csv(f, index=False)
        logging.info(f"\t -> Exporting the spectrum normalization profile to {full_file_name}")

    if export_corrected_integrated_normalized_data:
        # making up the integrated sample data
        full_file_name = os.path.join(full_output_folder, "normalized_integrated.tif")
        logging.info(f"\t -> Exporting integrated normalized data to {full_file_name} ...")
        make_tiff(data=integrated_normalized_data[_sample_run_number], filename=full_file_name)
        logging.info(f"\t -> Exporting integrated normalized data to {full_file_name} is done!")
        export_integrated_normalized_preview(
            output_folder=full_output_folder,
            integrated_normalized_image=integrated_normalized_data[_sample_run_number],
            normalized_stack=normalized_data.get(_sample_run_number) if normalized_data is not None else None,
            roi=sample_roi,
            bin_metadata=bin_metadata,
            spectrum_normalized_data=_spectrum_normalized_data,
        )

    if export_corrected_stack_of_normalized_data:
        output_stack_folder = os.path.join(full_output_folder, "stack")
        logging.info(f"\tmaking folder {output_stack_folder}")
        os.makedirs(output_stack_folder, exist_ok=True)

        for _index, _data in enumerate(normalized_data[_sample_run_number]):
            file_index = _index if bin_metadata is None else int(bin_metadata["active_bin_index_array"][_index])
            _output_file = os.path.join(output_stack_folder, f"image{file_index:04d}.tif")
            make_tiff(data=_data, filename=_output_file)
        logging.info(f"\t -> Exporting normalized data to {output_stack_folder} is done!")
        print(f"Exported normalized tif images are in: {output_stack_folder}!")
        
        # spectra_file = sample_master_dict[_sample_run_number][MasterDictKeys.spectra_file_name]
        export_spectra_file(spectra_array=spectra_array,
                            spectra_file=spectra_file,
                            output_stack_folder=output_stack_folder,
                            normalized_data=normalized_data[_sample_run_number])


        # create x-axis file
        create_x_axis_file(
            tof_array=tof_array,
            lambda_array=lambda_array,
            energy_array=energy_array,
            bin_metadata=bin_metadata,
            output_folder=output_stack_folder,
        )


def manually_create_and_export_spectra_file(spectra_array=None, output_folder=None, normalized_data=None):
     # manually create the file for spectra
        spectra_file_name = os.path.join(output_folder, f"manually_created_{SPECTRA_FILE_PREFIX}")
        _full_counts_array = np.empty_like(spectra_array)
        for _index, _data in enumerate(normalized_data):
            _full_counts_array[_index] = np.nansum(_data)
        pd_spectra = pd.DataFrame({
            "shutter_time": spectra_array,
            "counts": _full_counts_array
        })
        pd_spectra.to_csv(spectra_file_name, index=False, sep=",")
        logging.info(f"\t -> Exporting manually created spectra file to {spectra_file_name} is done!")


def export_spectra_file(spectra_array=None,
                            spectra_file=None,
                            output_stack_folder=None,
                            normalized_data=None):

    if spectra_array is not None:
        manually_create_and_export_spectra_file(spectra_array=spectra_array,
                                                output_folder=output_stack_folder,
                                                normalized_data=normalized_data)

    else:

        if spectra_file and Path(spectra_file).exists():
            logging.info(f"Exported time spectra file  {spectra_file} to {output_stack_folder}!")
            shutil.copy(spectra_file, output_stack_folder)


def  export_corrected_normalized_data(sample_master_dict=None,
                                      ob_master_dict=None,
                                       combined_normalized_data=None,
                                       integrated_normalized_data=None,
                                       export_corrected_integrated_combined_normalized_data=False,
                                       export_corrected_stack_of_combined_normalized_data=False,
                                       lambda_array=None,
                                       energy_array=None,
                                       output_folder="./",
                                       spectra_array=None
):

    list_sample_runs = list(sample_master_dict.keys())
    _sample_str = ""
    for _run in list_sample_runs:
        _sample_str += f"{sample_master_dict[_run]['run_number']}_"

    _ob_str = ""
    list_ob_runs = list(ob_master_dict.keys())
    for _run in list_ob_runs:
        _ob_str += f"{ob_master_dict[_run]['run_number']}_"

    full_output_folder = os.path.join(
        output_folder, f"combined_normalized_samples_{_sample_str}_obs_{_ob_str}"
    )  # issue for WEI here !
    full_output_folder = os.path.abspath(full_output_folder)
    os.makedirs(full_output_folder, exist_ok=True)

    if export_corrected_integrated_combined_normalized_data:
        # making up the integrated sample data
        data_integrated = np.nanmean(combined_normalized_data, axis=0)
        full_file_name = os.path.join(full_output_folder, "integrated.tif")
        logging.info(f"\t -> Exporting integrated combined normalized data to {full_file_name} ...")
        make_tiff(data=data_integrated, filename=full_file_name)
        logging.info(f"\t -> Exporting integrated combined normalized data to {full_file_name} is done!")

    if export_corrected_stack_of_combined_normalized_data:
        output_stack_folder = os.path.join(full_output_folder, "stack")
        logging.info(f"\tmaking folder {output_stack_folder}")
        os.makedirs(output_stack_folder, exist_ok=True)

        for _index, _data in enumerate(combined_normalized_data):
            _output_file = os.path.join(output_stack_folder, f"image{_index:04d}.tif")
            make_tiff(data=_data, filename=_output_file)
        logging.info(f"\t -> Exporting combined normalized data to {output_stack_folder} is done!")
        print(f"Exported combined normalized tif images are in: {output_stack_folder}!")
        
        export_spectra_file(spectra_array=spectra_array,
                            spectra_file=spectra_file,
                            output_stack_folder=output_stack_folder,
                            normalized_data=combined_normalized_data)


        # copy one of the spectra file to the output folder, or the manually defined one
        spectra_file = sample_master_dict[list_sample_runs[0]][MasterDictKeys.spectra_file_name]
        export_spectra_file(spectra_array=spectra_array,
                            spectra_file=spectra_file,
                            output_stack_folder=output_stack_folder,
                            normalized_data=combined_normalized_data)

        # create x-axis file
        create_x_axis_file(
            lambda_array=lambda_array,
            energy_array=energy_array,
            output_folder=output_stack_folder,
        )


def read_container_roi_file(container_roi_file=None) -> tuple[int, int, int, int]:
        master_dict = load_json(container_roi_file)
        list_container_values = master_dict['list_container_values']
        return list_container_values
        
        
def save_container_roi_file(output_folder:str,
                            sample_run_number: str, 
                            container_roi: Roi, 
                            list_container_values: list, 
                            integrated_image: np.ndarray):
    # container_roi_file = os.path.join(output_folder, f"container_roi_of_run_{sample_run_number}.tiff")
    container_roi_file = os.path.join(output_folder, f"container_roi_of_run_{sample_run_number}.json")
    
    logging.info(f"Saving container roi file to {container_roi_file}")
    # scitiff_dict = {'container_roi': container_roi,
    #                 'list_container_values': list_container_values,
    #                 }
    
    integrated_image = integrated_image.astype(float)
    list_container_values = [float(_value) for _value in list_container_values]
    master_dict = {'integrated_image': integrated_image.tolist(),
                   'container_roi': {'left': float(container_roi.left),
                                     'top': float(container_roi.top),
                                     'width': float(container_roi.width),
                                     'height': float(container_roi.height)},
                   'list_container_values': list_container_values}
    
    save_json(container_roi_file, master_dict)
    return container_roi_file
    
    
def calculate_container_roi_value_array(
    sample_data: np.ndarray,
    container_roi: Roi,
    container_roi_file: str,
    output_folder: str,
    sample_run_number: str,
) -> tuple[np.ndarray, str]:
    """Return the per-frame mean intensity in the selected container ROI."""

    logging.info(f"in calculate_container_roi_value_array:")
    if container_roi_file is not None:
        logging.info(f"\t {container_roi_file = }")
        _container_value_array = read_container_roi_file(container_roi_file=container_roi_file)
        logging.info(f"\t{_container_value_array =}")

    else:
        logging.info(f"\t {container_roi = }")
        x0: int = container_roi.left
        y0: int = container_roi.top
        width: int = container_roi.width
        height: int = container_roi.height
        
        list_container_values = []
        for i, _sample in enumerate(sample_data):
            _container_value = np.mean(np.mean(_sample[y0:y0 + height, x0:x0 + width], axis=0), axis=0)            
            list_container_values.append(_container_value)
        
        # save the container roi file
        container_roi_file = save_container_roi_file(output_folder=output_folder, 
                                                    sample_run_number=sample_run_number, 
                                                    container_roi=container_roi,
                                                    list_container_values=list_container_values,
                                                    integrated_image=np.sum(sample_data, axis=0))    

        _container_value_array = list_container_values

    _container_value_array = np.asarray(_container_value_array, dtype=np.float64)
    if _container_value_array.shape[0] != np.asarray(sample_data).shape[0]:
        raise ValueError(
            "Container ROI profile length does not match the sample stack length: "
            f"{_container_value_array.shape[0]} container values for {np.asarray(sample_data).shape[0]} frames."
        )
    return _container_value_array, container_roi_file


def normalize_by_container_value_array(
    sample_data: np.ndarray,
    container_value_array: np.ndarray,
) -> np.ndarray:
    array_data = np.asarray(sample_data, dtype=np.float64)
    container_values = np.asarray(container_value_array, dtype=np.float64)
    if container_values.shape[0] != array_data.shape[0]:
        raise ValueError(
            "Container ROI profile length does not match the sample stack length: "
            f"{container_values.shape[0]} container values for {array_data.shape[0]} frames."
        )
    denominator_shape = (container_values.shape[0],) + (1,) * (array_data.ndim - 1)
    denominator = container_values.reshape(denominator_shape)
    return np.divide(
        array_data,
        denominator,
        out=np.zeros_like(array_data, dtype=np.float64),
        where=denominator != 0,
    )


def normalize_by_container_roi(sample_data: np.ndarray,
                               container_roi: Roi,
                               container_roi_file: str,
                               output_folder: str,
                               sample_run_number: str) -> np.ndarray:
    """Normalize sample data by the mean intensity in a container-only ROI."""

    logging.info(f"in normalize_by_container_roi:")
    _container_value_array, container_roi_file = calculate_container_roi_value_array(
        sample_data=sample_data,
        container_roi=container_roi,
        container_roi_file=container_roi_file,
        output_folder=output_folder,
        sample_run_number=sample_run_number,
    )
    _normalized_sample = normalize_by_container_value_array(
        sample_data=sample_data,
        container_value_array=_container_value_array,
    )
    return _normalized_sample, container_roi_file


def logging_statistics_of_data(data=None, data_type=DataType.sample):
        data_shape = data.shape
        nbr_pixels = data_shape[1] * data_shape[2]
        logging.info(f" **** Statistics of {data_type} data *****")
        number_of_zeros = np.sum(data == 0)
        logging.info(f"\t {data_type} data shape: {data_shape}")
        logging.info(f"\t data type of _sample_data: {data.dtype}")
        logging.info(f"\t Number of zeros in {data_type} data: {number_of_zeros}")
        logging.info(f"\t Number of nan in {data_type} data: {np.sum(np.isnan(data))}")
        logging.info(f"\t Percentage of zeros in {data_type} data: {number_of_zeros / (data_shape[0] * nbr_pixels) * 100:.2f}%")
        logging.info(f"\t Mean of {data_type} data: {np.mean(data)}")
        logging.info(f"\t maximum of {data_type} data: {np.max(data)}")
        logging.info(f"\t minimum of {data_type} data: {np.min(data)}")
        logging.info("**********************************")
