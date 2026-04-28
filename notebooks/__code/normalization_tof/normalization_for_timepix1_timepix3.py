import argparse
import glob
import logging
import multiprocessing as mp
import os
from random import sample
import shutil
from pathlib import Path
from typing import Tuple

from annotated_types import Not
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

from __code.normalization_tof import RebinCustomBasis, RebinCustomScale, RebinMode
from __code.normalization_tof.utilities import *

# from enum import Enum
# from scipy.constants import h, c, electron_volt, m_n
# from timepix_geometry_correction.correct import TimepixGeometryCorrection

MARKERSIZE = 6

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

LOG_PATH = "/SNS/VENUS/shared/log/"
LOAD_DTYPE = np.uint16

PROTON_CHARGE_TOLERANCE = 0.1

def initialize_logging():
    """initialize logging"""
    file_name, ext = os.path.splitext(os.path.basename(__file__))
    user_name = os.getlogin()  # add user name to the log file name
    log_file_name = os.path.join(LOG_PATH, f"{user_name}_{file_name}.log")
    logging.basicConfig(filename=log_file_name,
                        filemode='w',
                        format='[%(levelname)s] - %(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info(f"*** Starting a new script {file_name} ***")


def normalization_with_list_of_full_path(
    sample_dict: dict = None,
    combine_samples: bool = False,
    ob_dict: dict = None,
    dc_dict: dict = None,
    bragg_edge_cd_sample_background_dict: dict = None,
    bragg_edge_cd_ob_background_dict: dict = None,
    spectra_array: np.ndarray = None,
    output_folder: str = "./",
    verbose: bool = False,
    proton_charge_flag=True,
    # monitor_counts_flag=False,
    # shutter_counts_flag=True,
    replace_ob_zeros_by_nan_flag=False,
    replace_ob_zeros_by_local_median_flag=False,
    kernel_size_for_local_median: Tuple[int, int, int] = (3, 3, 3),
    max_iterations: int = 10,
    output_tif: bool = True,
    instrument: str = "VENUS",
    detector_delay_us: float = None,
    preview: bool = False,
    distance_source_detector_m: float = 25,
    correct_chips_alignment_flag: bool = True,
    correct_chips_alignment_config: dict = None,
    export_mode: dict = None,
    roi = None,
    container_roi = None,
    container_roi_file = None,
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
    experimental_uncertainties_flag: bool = False,
    black_filter_background_config: dict = None,
    bragg_edge_cd_background_config: dict = None,
    measured_background_correction_configs: list = None) -> NormalizedData:
     
    # """
    # normalize the sample data with ob data using proton charge and shutter counts
    
    # Args:
    #     sample_dict (dict): dictionary with sample run numbers and their data
    #         {base_name_run1: {'full_path': full_path, 'nexus': nexus_path},
    #          base_name_run2: {'full_path': full_path, 'nexus': nexus_path}, ...}

    #     ob_dict (dict): dictionary with ob run numbers and their data
    #         {base_name_run1: {'full_path': full_path, 'nexus': nexus_path},
    #          base_name_run2: {'full_path': full_path, 'nexus': nexus_path}, ...}

    #     dc_dict (dict): dictionary with dc run numbers and their data
    #         {base_name_run1: {'full_path': full_path, 'nexus': nexus_path},
    #          base_name_run2: {'full_path': full_path, 'nexus': nexus_path}, ...}

    #                  output_folder (str): folder to save the output data
    #     verbose (bool): if True, display additional information
    #     combine_samples (bool): if True, combine sample runs
    #     proton_charge_flag (bool): if True, normalize by proton charge
    #     monitor_counts_flag (bool): if True, normalize by monitor counts
    #     shutter_counts_flag (bool): if True, normalize by shutter counts
    #     replace_ob_zeros_by_nan_flag (bool): if True, replace OB zeros by NaN
    #     replace_ob_zeros_by_local_median_flag (bool): if True, replace OB zeros by local median
    #     kernel_size_for_local_median (Tuple[int, int, int]): kernel size for local median (y, x, tof)
    #     max_iterations (int): maximum number of iterations for local median
    #     output_tif (bool): if True, export the data as tif files
    #     instrument (str): instrument name
    #     detector_delay_us (float): detector delay in microseconds
    #     preview (bool): if True, display preview of the data
    #     distance_source_detector_m (float): distance from source to detector in meters
    #     correct_chips_alignment_flag (bool): if True, correct chips alignment
    #     correct_chips_alignment_config (dict): configuration for chips alignment correction
    #     export_mode (dict): dictionary with export options
    #     roi (Roi): region of interest for full spectrum normalization
    #     container_roi (Roi): region of interest for container only normalization 
    #     container_roi_file (str): file path to container ROI file (scitiff format) (will take precedence over container_roi if both are provided)

    # Returns:
    #     normalized_data | np.ndarray: normalized data
    
    # """

    initialize_logging()

    logging.info("=============== Starting normalization ===============")
    dict_to_return = NormalizedData()

    # list sample and ob run numbers
    logging.info(f"{sample_dict.keys() = }")
    if verbose:
        display(HTML(f"Sample run numbers: {list(sample_dict.keys())}"))

    logging.info(f"{ob_dict.keys() = }")
    if verbose:
        display(HTML(f"List of ob run numbers: {list(ob_dict.keys())}"))

    logging.info(f"{output_folder = }")

    export_corrected_stack_of_sample_data = export_mode.get("sample_stack", False)
    export_corrected_stack_of_ob_data = export_mode.get("ob_stack", False)
    export_corrected_stack_of_normalized_data = export_mode.get("normalized_stack", False)
    # export_corrected_stack_of_combined_normalized_data = export_mode.get("combined_normalized_stack", False)
    export_corrected_integrated_sample_data = export_mode.get("sample_integrated", False)
    export_corrected_integrated_ob_data = export_mode.get("ob_integrated", False)
    export_corrected_integrated_normalized_data = export_mode.get("normalized_integrated", False)
    # export_corrected_integrated_combined_normalized_data = export_mode.get("combined_normalized_integrated", False)

    export_x_axis = export_mode.get("x_axis", True)

    logging.info("Input parameters:")
    
    logging.info(f"\t{sample_dict = }")
    logging.info(f"\t{combine_samples =}")
    logging.info(f"\t{ob_dict = }")
    logging.info(f"\t{dc_dict = }")
    logging.info(f"{spectra_array = }")
    
    logging.info(f"")
    logging.info(f"- export mode:")
    logging.info(f"\t{export_corrected_stack_of_sample_data = }")
    logging.info(f"\t{export_corrected_stack_of_ob_data = }")
    logging.info(f"\t{export_corrected_stack_of_normalized_data = }")
    logging.info(f"\t{export_corrected_integrated_sample_data = }")
    logging.info(f"\t{export_corrected_integrated_ob_data = }")
    logging.info(f"\t{export_corrected_integrated_normalized_data = }")
    
    logging.info(f"")
    logging.info(f"{roi =}")
    logging.info(f"{export_x_axis = }")
    logging.info(f"{proton_charge_flag = }")
    logging.info(f"{replace_ob_zeros_by_nan_flag = }")
    logging.info(f"{replace_ob_zeros_by_local_median_flag = }")
    logging.info(f"{kernel_size_for_local_median = }")
    logging.info(f"{max_iterations = }")
    logging.info(f"{correct_chips_alignment_flag = }")
    logging.info(f"{distance_source_detector_m = }")
    logging.info(f"{detector_delay_us = }")
    logging.info(f"{rebin_mode = }")
    logging.info(f"{rebin_delta_tof_us = }")
    logging.info(f"{rebin_delta_lambda_a = }")
    logging.info(f"{rebin_delta_tof_over_tof = }")
    logging.info(f"{rebin_delta_lambda_over_lambda = }")
    logging.info(f"{rebin_delta_lambda_squared_a2 = }")
    logging.info(f"{rebin_custom_basis = }")
    logging.info(f"{rebin_custom_scale = }")
    logging.info(f"{rebin_custom_schedule = }")
    logging.info(f"{rebin_full_bins_only = }")
    logging.info(f"{rebin_snap_to_native_grid = }")
    logging.info(f"{experimental_uncertainties_flag = }")
    logging.info(f"{black_filter_background_config = }")
    logging.info(f"{bragg_edge_cd_background_config = }")
    logging.info(f"{measured_background_correction_configs = }")
    logging.info(f"")

    container_normalization_requested = (container_roi is not None) or (container_roi_file is not None)
    container_only_without_ob = (not ob_dict) and container_normalization_requested
    if not ob_dict and not container_only_without_ob:
        raise ValueError(
            "No open-beam runs were provided. This is only supported when container normalization "
            "is enabled, because the container ROI supplies the internal reference."
        )
    if container_only_without_ob:
        logging.info("No OB runs provided; using container-only normalization mode.")
        if verbose:
            display(
                HTML(
                    "<span style='color:blue'>No OB runs provided; using container-only "
                    "normalization mode.</span>"
                )
            )
        if dc_dict:
            raise ValueError("Container-only normalization without OB is not supported with dark-current data.")
        if black_filter_background_config and black_filter_background_config.get("enabled", False):
            raise ValueError("Black-filter background correction requires an open-beam run.")
        if export_corrected_stack_of_ob_data or export_corrected_integrated_ob_data:
            logging.warning("No OB runs were provided; disabling OB export options.")
            export_corrected_stack_of_ob_data = False
            export_corrected_integrated_ob_data = False

    bragg_edge_cd_sample_background_dict = bragg_edge_cd_sample_background_dict or {}
    bragg_edge_cd_ob_background_dict = bragg_edge_cd_ob_background_dict or {}
    measured_background_correction_configs = [
        dict(_config)
        for _config in (measured_background_correction_configs or [])
        if _config and _config.get("enabled", True)
    ]
    bragg_edge_cd_background_enabled = bool(
        bragg_edge_cd_background_config and bragg_edge_cd_background_config.get("enabled", False)
    )
    if bragg_edge_cd_background_enabled:
        legacy_background_config = dict(bragg_edge_cd_background_config)
        legacy_background_config.setdefault("mode", "Cd-filter background correction for Bragg edge mode")
        legacy_background_config.setdefault("column_label", "Cd-filter")
        legacy_background_config.setdefault("key_prefix", "bragg_edge_cd")
        legacy_background_config.setdefault("weight", 1.0)
        legacy_background_config["sample_background_dict"] = bragg_edge_cd_sample_background_dict
        legacy_background_config["ob_background_dict"] = bragg_edge_cd_ob_background_dict
        measured_background_correction_configs.append(legacy_background_config)

    normalized_background_correction_configs = []
    for background_index, background_config in enumerate(measured_background_correction_configs, start=1):
        background_mode = background_config.get(
            "mode",
            f"Measured background correction {background_index} for Bragg edge mode",
        )
        background_column_label = background_config.get(
            "column_label",
            f"measured background {background_index}",
        )
        background_key_prefix = background_config.get(
            "key_prefix",
            f"measured_background_{background_index}",
        )
        normalized_background_correction_configs.append(
            {
                **background_config,
                "enabled": True,
                "mode": background_mode,
                "column_label": background_column_label,
                "key_prefix": background_key_prefix,
                "weight": float(background_config.get("weight", 1.0)),
                "sample_background_dict": background_config.get("sample_background_dict") or {},
                "ob_background_dict": background_config.get("ob_background_dict") or {},
            }
        )
    measured_background_correction_configs = normalized_background_correction_configs
    measured_background_correction_enabled = bool(measured_background_correction_configs)
    if measured_background_correction_enabled:
        for background_config in measured_background_correction_configs:
            background_mode = background_config.get(
                "mode",
                "Measured background correction for Bragg edge mode",
            )
            if not background_config.get("sample_background_dict") or not background_config.get("ob_background_dict"):
                raise ValueError(
                    f"{background_mode} requires both sample background runs and OB background runs."
                )
        if dc_dict:
            raise ValueError(
                "Measured background corrections for Bragg edge mode are not supported together "
                "with the legacy single dark-current correction."
            )
        if black_filter_background_config and black_filter_background_config.get("enabled", False):
            raise ValueError(
                "Measured background corrections for Bragg edge mode cannot be combined with "
                "black-filter background correction in the same run."
            )
        if container_only_without_ob:
            raise ValueError("Measured background corrections require open-beam runs.")
    
    sample_master_dict, sample_status_metadata = create_master_dict(
        data_dictionary=sample_dict, 
        data_type=DataType.sample, 
        instrument=instrument,
        spectra_array=spectra_array,
    )
    ob_master_dict, ob_status_metadata = create_master_dict(
        data_dictionary=ob_dict, 
        data_type=DataType.ob, 
        instrument=instrument,
        spectra_array=spectra_array,
    )

    dc_master_dict, dc_status_metadata = create_master_dict(
        data_dictionary=dc_dict, 
        data_type=DataType.dc, 
        instrument=instrument,
        spectra_array=spectra_array,
    )

    measured_background_runtime_configs = []
    for background_config in measured_background_correction_configs:
        sample_background_master_dict, sample_background_status_metadata = create_master_dict(
            data_dictionary=background_config["sample_background_dict"],
            data_type=DataType.sample,
            instrument=instrument,
            spectra_array=spectra_array,
        )
        ob_background_master_dict, ob_background_status_metadata = create_master_dict(
            data_dictionary=background_config["ob_background_dict"],
            data_type=DataType.ob,
            instrument=instrument,
            spectra_array=spectra_array,
        )
        measured_background_runtime_configs.append(
            {
                **background_config,
                "sample_master_dict": sample_background_master_dict,
                "ob_master_dict": ob_background_master_dict,
                "sample_status_metadata": sample_background_status_metadata,
                "ob_status_metadata": ob_background_status_metadata,
                "sample_background_data_combined": None,
                "sample_background_variance": None,
                "ob_background_data_combined": None,
                "ob_background_variance": None,
            }
        )

    ob_data_combined_variance = None
    dc_data_combined_variance = None
    detector_model_uncertainty_available = any(
        _run_info.get(MasterDictKeys.shutter_counts) not in [None, []]
        for _master_dict in [
            sample_master_dict,
            ob_master_dict,
            dc_master_dict,
            *[
                _background_config["sample_master_dict"]
                for _background_config in measured_background_runtime_configs
            ],
            *[
                _background_config["ob_master_dict"]
                for _background_config in measured_background_runtime_configs
            ],
        ]
        for _run_info in _master_dict.values()
    )
    uncertainty_model_label = (
        "TPX1 detector-model uncertainty (iBeatles-style) where shutter counts are available; "
        "otherwise Poisson counting statistics. Proton charge propagated as an exact scale factor."
        if experimental_uncertainties_flag and detector_model_uncertainty_available
        else "Poisson counting statistics; proton charge treated as an exact scale factor"
    )
    if measured_background_correction_enabled:
        background_weight_summary = ", ".join(
            f"{_config['column_label']} weight={_config['weight']:g}"
            for _config in measured_background_runtime_configs
        )
        uncertainty_model_label += (
            "; measured Bragg-edge background correction propagates independent "
            "sample/OB background variances on the native frame grid before rebinning "
            f"with squared weights ({background_weight_summary})"
        )

    def prepare_rebinned_payload(
        current_sample_data,
        current_sample_variance,
        current_time_spectra,
        current_detector_delay_us,
        current_container_value_array=None,
    ):
        sample_data_for_rebin = current_sample_data
        sample_variance_for_rebin = current_sample_variance
        ob_data_for_rebin = ob_data_combined
        ob_variance_for_rebin = ob_data_combined_variance
        measured_background_diagnostic = None

        if measured_background_correction_enabled:
            total_sample_background_data = np.zeros_like(current_sample_data, dtype=np.float64)
            total_sample_background_variance = np.zeros_like(current_sample_variance, dtype=np.float64)
            total_ob_background_data = np.zeros_like(ob_data_combined, dtype=np.float64)
            total_ob_background_variance = np.zeros_like(ob_data_combined_variance, dtype=np.float64)
            background_terms = []

            for background_config in measured_background_runtime_configs:
                weight = float(background_config.get("weight", 1.0))
                sample_background_data = background_config["sample_background_data_combined"]
                ob_background_data = background_config["ob_background_data_combined"]
                sample_background_variance = background_config["sample_background_variance"]
                ob_background_variance = background_config["ob_background_variance"]
                if sample_background_data.shape != current_sample_data.shape:
                    raise ValueError(
                        "Measured sample background shape does not match the sample data before rebinning: "
                        f"{background_config['mode']} has {sample_background_data.shape}, "
                        f"sample has {current_sample_data.shape}."
                    )
                if ob_background_data.shape != ob_data_combined.shape:
                    raise ValueError(
                        "Measured OB background shape does not match the OB data before rebinning: "
                        f"{background_config['mode']} has {ob_background_data.shape}, "
                        f"OB has {ob_data_combined.shape}."
                    )
                total_sample_background_data += weight * sample_background_data
                total_ob_background_data += weight * ob_background_data
                total_sample_background_variance += (weight**2) * sample_background_variance
                total_ob_background_variance += (weight**2) * ob_background_variance
                sign = "-" if weight < 0 else ("+" if background_terms else "")
                background_terms.append(f"{sign}{abs(weight):g}*{background_config['column_label']}")

            combined_label = " ".join(background_terms)
            measured_background_diagnostic = {
                "mode": f"Measured background correction for Bragg edge mode ({combined_label})",
                "sample_background_data": total_sample_background_data,
                "sample_background_variance": total_sample_background_variance,
                "ob_background_data": total_ob_background_data,
                "ob_background_variance": total_ob_background_variance,
            }

            logging.info(
                "Subtracting measured Bragg-edge background on the native frame grid before rebinning: "
                f"{combined_label}"
            )
            sample_data_for_rebin = current_sample_data - total_sample_background_data
            sample_variance_for_rebin = current_sample_variance + total_sample_background_variance
            ob_data_for_rebin = ob_data_combined - total_ob_background_data
            ob_variance_for_rebin = ob_data_combined_variance + total_ob_background_variance

        rebinned_payload = maybe_rebin_data_and_axes(
            sample_data=sample_data_for_rebin,
            sample_variance=sample_variance_for_rebin,
            ob_data_combined=ob_data_for_rebin,
            ob_data_combined_variance=ob_variance_for_rebin,
            dc_data_combined=dc_data_combined,
            dc_data_combined_variance=dc_data_combined_variance,
            time_spectra=current_time_spectra,
            distance_source_detector_m=distance_source_detector_m,
            detector_delay_us=current_detector_delay_us,
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

        if container_only_without_ob and current_container_value_array is not None:
            active_frame_groups = rebinned_payload["active_frame_groups"]
            rebinned_container_value_array = rebin_array_from_bin_groups(
                current_container_value_array,
                active_frame_groups,
                reducer="sum",
            )
            logging.info(
                "Applying container-only normalization after rebinning so each output bin "
                "uses the summed container reference for the same native frames."
            )
            rebinned_payload["sample_data"] = normalize_by_container_value_array(
                sample_data=rebinned_payload["sample_data"],
                container_value_array=rebinned_container_value_array,
            )
            if rebinned_payload["sample_variance"] is not None:
                denominator_shape = (rebinned_container_value_array.shape[0],) + (
                    1,
                ) * (np.asarray(rebinned_payload["sample_variance"]).ndim - 1)
                denominator = rebinned_container_value_array.reshape(denominator_shape)
                rebinned_payload["sample_variance"] = np.divide(
                    rebinned_payload["sample_variance"],
                    denominator**2,
                    out=np.zeros_like(rebinned_payload["sample_variance"], dtype=np.float64),
                    where=denominator != 0,
                )
            rebinned_payload["container_roi_reference_value_array"] = rebinned_container_value_array

        rebinned_payload["bragg_edge_cd_background_profile"] = None
        rebinned_payload["measured_background_profiles"] = []
        if measured_background_diagnostic is not None:
            active_frame_groups = rebinned_payload["active_frame_groups"]
            rebinned_payload["measured_background_profiles"].append(
                calculate_bragg_edge_cd_background_profile(
                    roi=roi,
                    raw_sample_data=rebin_array_from_bin_groups(
                        current_sample_data,
                        active_frame_groups,
                        reducer="sum",
                    ),
                    raw_sample_variance=rebin_array_from_bin_groups(
                        current_sample_variance,
                        active_frame_groups,
                        reducer="sum",
                    ),
                    raw_ob_data=rebin_array_from_bin_groups(
                        ob_data_combined,
                        active_frame_groups,
                        reducer="sum",
                    ),
                    raw_ob_variance=rebin_array_from_bin_groups(
                        ob_data_combined_variance,
                        active_frame_groups,
                        reducer="sum",
                    ),
                    sample_background_data=rebin_array_from_bin_groups(
                        measured_background_diagnostic["sample_background_data"],
                        active_frame_groups,
                        reducer="sum",
                    ),
                    sample_background_variance=rebin_array_from_bin_groups(
                        measured_background_diagnostic["sample_background_variance"],
                        active_frame_groups,
                        reducer="sum",
                    ),
                    ob_background_data=rebin_array_from_bin_groups(
                        measured_background_diagnostic["ob_background_data"],
                        active_frame_groups,
                        reducer="sum",
                    ),
                    ob_background_variance=rebin_array_from_bin_groups(
                        measured_background_diagnostic["ob_background_variance"],
                        active_frame_groups,
                        reducer="sum",
                    ),
                    mode=measured_background_diagnostic["mode"],
                    column_label="measured background combined",
                    key_prefix="measured_background_combined",
                )
            )

        rebinned_payload["ob_data_combined_for_spectrum"] = (
            None if container_only_without_ob else
            calculate_ob_data_combined_used_by_spectrum_normalization(
                roi=roi,
                ob_data_combined=rebinned_payload["ob_data_combined"],
                verbose=verbose,
            )
        )
        rebinned_payload["ob_data_combined_variance_for_spectrum"] = (
            None if container_only_without_ob else
            calculate_roi_profile(
                data=rebinned_payload["ob_data_combined_variance"],
                roi=roi,
            )
        )

        rebinned_payload["dc_data_combined_for_spectrum"] = calculate_roi_profile(
            data=rebinned_payload["dc_data_combined"],
            roi=roi,
        )
        rebinned_payload["dc_data_combined_variance_for_spectrum"] = calculate_roi_profile(
            data=rebinned_payload["dc_data_combined_variance"],
            roi=roi,
        )

        rebinned_payload["black_filter_background_profile"] = None
        if black_filter_background_config and black_filter_background_config.get("enabled", False):
            if roi is None:
                raise ValueError("Black-filter background correction is enabled but no ROI was provided.")
            if dc_data_combined is not None:
                raise ValueError("Black-filter background correction is not currently supported with dark-current data.")
            rebinned_payload["black_filter_background_profile"] = (
                calculate_black_filter_background_corrected_spectrum(
                    roi=roi,
                    sample_data=current_sample_data,
                    sample_variance=current_sample_variance,
                    ob_data_combined=ob_data_combined,
                    ob_data_combined_variance=ob_data_combined_variance,
                    energy_array=rebinned_payload["original_energy_array"],
                    active_frame_groups=rebinned_payload["active_frame_groups"],
                    background_shape_file=black_filter_background_config.get(
                        "background_shape_file",
                        DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
                    ),
                    anchor_energy_eV=black_filter_background_config.get(
                        "anchor_energy_eV",
                        5.1044,
                    ),
                )
            )

        if rebin_mode == RebinMode.none:
            rebinned_payload["export_spectra_array"] = spectra_array
        else:
            rebinned_payload["export_spectra_array"] = rebinned_payload["tof_array"]

        return rebinned_payload

    # load ob images ===============================
    if ob_master_dict:
        load_images(master_dict=ob_master_dict, data_type=DataType.ob, verbose=verbose)
    else:
        logging.info("Skipping OB image loading because no OB runs were provided.")
   
    if proton_charge_flag:
        ob_proton_charge_available = (
            True if container_only_without_ob else ob_status_metadata.all_proton_charge_found
        )
        normalized_by_proton_charge = (
            sample_status_metadata.all_proton_charge_found
            and ob_proton_charge_available
            and all(
                _background_config["sample_status_metadata"].all_proton_charge_found
                and _background_config["ob_status_metadata"].all_proton_charge_found
                for _background_config in measured_background_runtime_configs
            )
        )
    else:
        normalized_by_proton_charge = False
    logging.info(f"{normalized_by_proton_charge = }")

    # combine all ob images
    if ob_master_dict:
        ob_data_combined, ob_sum_proton_charge = combine_images(
                                        data_type=DataType.ob,
                                        master_dict=ob_master_dict,
                                        use_proton_charge=normalized_by_proton_charge,
                                        replace_zeros_by_nan=replace_ob_zeros_by_nan_flag,
                                        replace_zeros_by_local_median=replace_ob_zeros_by_local_median_flag,
                                        kernel_size_for_local_median=kernel_size_for_local_median,
                                        max_iterations=max_iterations,
                                    )
        ob_data_combined_variance = calculate_combined_data_variance(
            master_dict=ob_master_dict,
            use_proton_charge=normalized_by_proton_charge,
            use_experimental_uncertainties=experimental_uncertainties_flag,
        )
        logging.info(f"{ob_data_combined.shape = }")
        logging.info(f"{ob_sum_proton_charge = }")
        logging.info(f"number of NaN in ob_data_combined data: {np.sum(np.isnan(ob_data_combined))}")
        logging.info(f"number of inf in ob_data_combined data: {np.sum(np.isinf(ob_data_combined))}")
        logging.info(f"number of zeros in ob_data_combined data: {np.sum(ob_data_combined == 0)} ")

        if correct_chips_alignment_flag:
            ob_data_combined = correct_chips_alignment(ob_data_combined,
                                                       correct_chips_alignment_config,
                                                       verbose=verbose)
            ob_data_combined_variance = correct_chips_alignment(
                ob_data_combined_variance,
                correct_chips_alignment_config,
                verbose=verbose,
            )
    else:
        ob_data_combined = None
        ob_sum_proton_charge = None
        ob_data_combined_variance = None

    # load dc images ================================
    load_images(master_dict=dc_master_dict, data_type=DataType.dc, verbose=verbose)

    # combine all dc images
    dc_data_combined = combine_dc_images(dc_master_dict)
    dc_data_combined_variance = calculate_dc_combined_variance(dc_master_dict)
    
    if dc_data_combined is not None:
        
        if correct_chips_alignment_flag:
            dc_data_combined = correct_chips_alignment(dc_data_combined, 
                                                    correct_chips_alignment_config, 
                                                    verbose=verbose)
            dc_data_combined_variance = correct_chips_alignment(
                dc_data_combined_variance,
                correct_chips_alignment_config,
                verbose=verbose,
            )

    for background_config in measured_background_runtime_configs:
        logging.info(
            "Loading measured background inputs: "
            f"{background_config['mode']} with weight {background_config['weight']:g}"
        )
        load_images(
            master_dict=background_config["sample_master_dict"],
            data_type=DataType.sample,
            verbose=verbose,
        )
        load_images(
            master_dict=background_config["ob_master_dict"],
            data_type=DataType.ob,
            verbose=verbose,
        )

        if correct_chips_alignment_flag:
            correct_all_samples_chips_alignment(
                background_config["sample_master_dict"],
                correct_chips_alignment_config,
                verbose=verbose,
            )
            correct_all_samples_chips_alignment(
                background_config["ob_master_dict"],
                correct_chips_alignment_config,
                verbose=verbose,
            )

        background_config["sample_background_data_combined"], _ = combine_images(
            data_type=DataType.sample,
            master_dict=background_config["sample_master_dict"],
            use_proton_charge=normalized_by_proton_charge,
            replace_zeros_by_nan=False,
            replace_zeros_by_local_median=False,
            kernel_size_for_local_median=kernel_size_for_local_median,
            max_iterations=max_iterations,
        )
        background_config["sample_background_variance"] = calculate_combined_data_variance(
            master_dict=background_config["sample_master_dict"],
            use_proton_charge=normalized_by_proton_charge,
            use_experimental_uncertainties=experimental_uncertainties_flag,
        )
        background_config["ob_background_data_combined"], _ = combine_images(
            data_type=DataType.ob,
            master_dict=background_config["ob_master_dict"],
            use_proton_charge=normalized_by_proton_charge,
            replace_zeros_by_nan=False,
            replace_zeros_by_local_median=False,
            kernel_size_for_local_median=kernel_size_for_local_median,
            max_iterations=max_iterations,
        )
        background_config["ob_background_variance"] = calculate_combined_data_variance(
            master_dict=background_config["ob_master_dict"],
            use_proton_charge=normalized_by_proton_charge,
            use_experimental_uncertainties=experimental_uncertainties_flag,
        )

    # load sample images ===============================
    load_images(master_dict=sample_master_dict, data_type=DataType.sample, verbose=verbose)
    if correct_chips_alignment_flag:
        correct_all_samples_chips_alignment(sample_master_dict, 
                                            correct_chips_alignment_config, 
                                            verbose=verbose)

    normalized_data = {}
    integrated_normalized_data = {}
    spectrum_normalized_data = {}

    if combine_samples:
        
        # combine all sample images and then perform normalization 
        sample_data_combined, sample_sum_proton_charge = combine_images(
                                    data_type=DataType.sample,
                                    master_dict=sample_master_dict,
                                    use_proton_charge=normalized_by_proton_charge,
                                    # use_monitor_counts=normalized_by_monitor_counts,
                                    replace_zeros_by_nan=False,
                                    replace_zeros_by_local_median=replace_ob_zeros_by_local_median_flag,
                                    kernel_size_for_local_median=kernel_size_for_local_median,
                                    max_iterations=max_iterations,
                                )
        sample_data_combined_variance = calculate_combined_data_variance(
            master_dict=sample_master_dict,
            use_proton_charge=normalized_by_proton_charge,
            use_experimental_uncertainties=experimental_uncertainties_flag,
        )

        logging.info("**********************************")
        list_run_number = list(sample_master_dict.keys())
        str_list_run_number = '_'.join([str(r) for r in list_run_number])
        logging.info(f"normalization of combined sample runs {list_run_number}")
        if verbose:
            display(HTML(f"Normalization of combined sample runs {list_run_number}"))
            
        # get statistics of sample data
        logging_statistics_of_data(data=sample_data_combined, data_type=DataType.sample_combined)
        
        if normalized_by_proton_charge:
            logging.info("Combined sample data normalized by total proton charge during image combination")
            logging.info(f"\t{sample_sum_proton_charge = }")
            if verbose:
                display(HTML("Combined sample data normalized by total proton charge during image combination"))

        current_container_value_array = None
        combined_container_roi_file = container_roi_file
        if (container_roi is not None) or (container_roi_file is not None):
            logging.info(f"Applying container normalization:")
            logging.info(f"\t {container_roi = }")
            logging.info(f"\t {combined_container_roi_file = }")
            if verbose:
                display(HTML(f"Applying container normalization:"))

            if container_only_without_ob:
                current_container_value_array, combined_container_roi_file = calculate_container_roi_value_array(
                    sample_data=sample_data_combined,
                    container_roi=container_roi,
                    container_roi_file=combined_container_roi_file,
                    output_folder=output_folder,
                    sample_run_number=str_list_run_number,
                )
            else:
                sample_data_combined, combined_container_roi_file = normalize_by_container_roi(
                    sample_data=sample_data_combined,
                    container_roi=container_roi,
                    container_roi_file=combined_container_roi_file,
                    output_folder=output_folder,
                    sample_run_number=str_list_run_number,
                )
            if verbose and (combined_container_roi_file is not None):
                display(HTML(f"Container roi file created: {combined_container_roi_file}."))

        current_detector_delay_us = detector_delay_us
        if current_detector_delay_us is None:
            current_detector_delay_us = sample_master_dict[list_run_number[0]][MasterDictKeys.detector_delay_us]
            logging.info(
                f"detector_delay argument is None, using detector delay from first sample run: {current_detector_delay_us} us"
            )

        time_spectra = sample_master_dict[list_run_number[0]][MasterDictKeys.list_spectra]
        rebinned_payload = prepare_rebinned_payload(
            current_sample_data=sample_data_combined,
            current_sample_variance=sample_data_combined_variance,
            current_time_spectra=time_spectra,
            current_detector_delay_us=current_detector_delay_us,
            current_container_value_array=current_container_value_array,
        )

        sample_data_combined = rebinned_payload["sample_data"]
        sample_data_combined_variance = rebinned_payload["sample_variance"]
        ob_data_for_normalization = rebinned_payload["ob_data_combined"]
        dc_data_for_normalization = rebinned_payload["dc_data_combined"]
        ob_data_combined_for_spectrum = rebinned_payload["ob_data_combined_for_spectrum"]
        ob_data_combined_variance_for_spectrum = rebinned_payload["ob_data_combined_variance_for_spectrum"]
        dc_data_combined_for_spectrum = rebinned_payload["dc_data_combined_for_spectrum"]
        dc_data_combined_variance_for_spectrum = rebinned_payload["dc_data_combined_variance_for_spectrum"]
        time_spectra = rebinned_payload["tof_array"]
        lambda_array = rebinned_payload["lambda_array"]
        energy_array = rebinned_payload["energy_array"]
        output_suffix = rebinned_payload["output_suffix"]

        # export sample data after correction if requested
        if export_corrected_stack_of_sample_data or export_corrected_integrated_sample_data:
            export_sample_images(
                output_folder,
                export_corrected_stack_of_sample_data,
                export_corrected_integrated_sample_data,
                str_list_run_number,
                sample_data_combined,
                spectra_file_name=sample_master_dict[list_run_number[0]][MasterDictKeys.spectra_file_name],
                spectra_array=rebinned_payload["export_spectra_array"],
                output_suffix=output_suffix,
                bin_metadata=rebinned_payload["bin_metadata"],
            )

        if export_corrected_stack_of_ob_data or export_corrected_integrated_ob_data:
            if ob_master_dict:
                first_ob_run_number = list(ob_master_dict.keys())[0]
                export_ob_images(
                    ob_master_dict.keys(),
                    output_folder,
                    export_corrected_stack_of_ob_data,
                    export_corrected_integrated_ob_data,
                    ob_data_for_normalization,
                    spectra_file_name=ob_master_dict[first_ob_run_number][MasterDictKeys.spectra_file_name],
                    spectra_array=rebinned_payload["export_spectra_array"],
                    output_suffix=output_suffix,
                    bin_metadata=rebinned_payload["bin_metadata"],
                )
            else:
                logging.info("Skipping OB export because no OB runs were provided.")

        _normalized_dict = perform_normalization(sample_data_combined, ob_data_for_normalization, dc_data_for_normalization)
        _normalized_data = _normalized_dict['normalized_data']
        _integrated_normalized_data = _normalized_dict['integrated_normalized_data']       
        integrated_normalized_data[str_list_run_number] = _integrated_normalized_data
        normalized_data[str_list_run_number] = _normalized_data

        _spectrum_normalized_data = perform_spectrum_normalization(roi=roi, 
                                                                sample_data=sample_data_combined,
                                                                sample_variance=sample_data_combined_variance,
                                                                ob_data_combined_for_spectrum=ob_data_combined_for_spectrum,
                                                                ob_data_combined_variance_for_spectrum=ob_data_combined_variance_for_spectrum,
                                                                dc_data_combined=dc_data_for_normalization,
                                                                dc_data_combined_for_spectrum=dc_data_combined_for_spectrum,
                                                                dc_data_combined_variance=rebinned_payload["dc_data_combined_variance"],
                                                                dc_data_combined_variance_for_spectrum=dc_data_combined_variance_for_spectrum,
                                                                black_filter_background_profile=rebinned_payload["black_filter_background_profile"],
                                                                bragg_edge_cd_background_profile=rebinned_payload["bragg_edge_cd_background_profile"],
                                                                measured_background_profiles=rebinned_payload["measured_background_profiles"])
        spectrum_normalized_data[str_list_run_number] = _spectrum_normalized_data

        # normalized_data[_sample_run_number] = np.array(np.divide(_sample_data, ob_data_combined))
        logging.info(f"{normalized_data[str_list_run_number].shape = }")
        logging.info(f"{normalized_data[str_list_run_number].dtype = }")
        logging.info(f"number of NaN in normalized data: {np.sum(np.isnan(normalized_data[str_list_run_number]))}")
        logging.info(f"number of inf in normalized data: {np.sum(np.isinf(normalized_data[str_list_run_number]))}")

        dict_to_return.tof_array = time_spectra

        dict_to_return.lambda_array = lambda_array
        dict_to_return.energy_array = energy_array

        logging.info(f"Preview: {preview = }")
        if preview:
            preview_normalized_data(sample_data_combined, 
                                    ob_data_for_normalization, 
                                    dc_data_for_normalization, 
                                    normalized_data, 
                                    lambda_array,
                                    energy_array, 
                                    current_detector_delay_us, 
                                    str_list_run_number,
                                    combine_samples,
                                    _spectrum_normalized_data,
                                    roi,
                                    bin_metadata=rebinned_payload["bin_metadata"],
                                    )
            
        if export_corrected_integrated_normalized_data or export_corrected_stack_of_normalized_data:

            export_normalized_data(ob_master_dict=ob_master_dict, 
                sample_master_dict=sample_master_dict, 
                _sample_run_number=str_list_run_number,
                normalized_data=normalized_data, 
                integrated_normalized_data=integrated_normalized_data,
                _spectrum_normalized_data=_spectrum_normalized_data,
                tof_array=time_spectra,
                lambda_array=lambda_array, 
                energy_array=energy_array, 
                output_folder=output_folder, 
                export_corrected_stack_of_normalized_data=export_corrected_stack_of_normalized_data,
                export_corrected_integrated_normalized_data=export_corrected_integrated_normalized_data,
                roi=roi,
                spectra_array=rebinned_payload["export_spectra_array"],
                spectra_file=sample_master_dict[list_run_number[0]][MasterDictKeys.spectra_file_name],
                output_suffix=output_suffix,
                bin_metadata=rebinned_payload["bin_metadata"],
                uncertainty_model_label=uncertainty_model_label)

    else:
    
        # normalize the sample data
        for _sample_run_number in sample_master_dict.keys():
            
            logging.info("**********************************")
            logging.info(f"normalization of run {_sample_run_number}")
            if verbose:
                display(HTML(f"Normalization of run {_sample_run_number}"))

            _sample_data = sample_master_dict[_sample_run_number][MasterDictKeys.data]
            sample_variance_input = np.asarray(_sample_data, dtype=np.float64)

            # get statistics of sample data
            logging_statistics_of_data(data=_sample_data, data_type=DataType.sample)
      
            # if correct_chips_alignment_flag:
            #     _sample_data = correct_chips_alignment(_sample_data, 
            #                                            correct_chips_alignment_config, 
            #                                            verbose=verbose)
      
            if normalized_by_proton_charge:
                if verbose:
                    display(HTML(f"Normalizing by proton charge"))
                _sample_data = normalize_by_proton_charge(sample_master_dict, 
                                                          _sample_run_number, 
                                                          _sample_data)

            current_container_value_array = None
            container_roi_file_for_run = container_roi_file
            if (container_roi is not None) or (container_roi_file is not None):
                logging.info(f"Applying container normalization:")
                logging.info(f"\t {container_roi = }")
                logging.info(f"\t {container_roi_file_for_run = }")
                if verbose:
                    display(HTML(f"Applying container normalization:"))

                if container_only_without_ob:
                    current_container_value_array, container_roi_file_for_run = calculate_container_roi_value_array(
                        sample_data=_sample_data,
                        container_roi=container_roi,
                        container_roi_file=container_roi_file_for_run,
                        output_folder=output_folder,
                        sample_run_number=_sample_run_number,
                    )
                else:
                    _sample_data, container_roi_file_for_run = normalize_by_container_roi(
                        sample_data=_sample_data,
                        container_roi=container_roi,
                        container_roi_file=container_roi_file_for_run,
                        output_folder=output_folder,
                        sample_run_number=_sample_run_number,
                    )
                if verbose and (container_roi_file_for_run is not None):
                    display(HTML(f"Container roi file created: {container_roi_file_for_run}."))

            sample_proton_charge = None
            if normalized_by_proton_charge:
                sample_proton_charge = sample_master_dict[_sample_run_number][MasterDictKeys.proton_charge]
            _sample_variance = calculate_data_variance(
                data=sample_variance_input,
                shutter_counts=sample_master_dict[_sample_run_number].get(MasterDictKeys.shutter_counts),
                use_experimental_uncertainties=experimental_uncertainties_flag,
            )
            if sample_proton_charge is not None:
                _sample_variance = _sample_variance / (sample_proton_charge**2)

            logging.info(f"{_sample_data.shape = }")
            logging.info(f"{_sample_data.dtype = }")
            if ob_data_combined is not None:
                logging.info(f"{ob_data_combined.shape = }")
                logging.info(f"{ob_data_combined.dtype = }")
            else:
                logging.info("No OB data available; container-normalized sample will be used directly.")

            current_detector_delay_us = detector_delay_us
            if current_detector_delay_us is None:
                current_detector_delay_us = sample_master_dict[_sample_run_number][MasterDictKeys.detector_delay_us]
                logging.info(
                    f"detector_delay argument is None, using detector delay from sample run {_sample_run_number}: {current_detector_delay_us} us"
                )

            time_spectra = sample_master_dict[_sample_run_number][MasterDictKeys.list_spectra]
            rebinned_payload = prepare_rebinned_payload(
                current_sample_data=_sample_data,
                current_sample_variance=_sample_variance,
                current_time_spectra=time_spectra,
                current_detector_delay_us=current_detector_delay_us,
                current_container_value_array=current_container_value_array,
            )

            _sample_data = rebinned_payload["sample_data"]
            _sample_variance = rebinned_payload["sample_variance"]
            ob_data_for_normalization = rebinned_payload["ob_data_combined"]
            dc_data_for_normalization = rebinned_payload["dc_data_combined"]
            ob_data_combined_for_spectrum = rebinned_payload["ob_data_combined_for_spectrum"]
            ob_data_combined_variance_for_spectrum = rebinned_payload["ob_data_combined_variance_for_spectrum"]
            dc_data_combined_for_spectrum = rebinned_payload["dc_data_combined_for_spectrum"]
            dc_data_combined_variance_for_spectrum = rebinned_payload["dc_data_combined_variance_for_spectrum"]
            time_spectra = rebinned_payload["tof_array"]
            lambda_array = rebinned_payload["lambda_array"]
            energy_array = rebinned_payload["energy_array"]
            output_suffix = rebinned_payload["output_suffix"]

            # export sample data after correction if requested
            if export_corrected_stack_of_sample_data or export_corrected_integrated_sample_data:
                export_sample_images(
                    output_folder,
                    export_corrected_stack_of_sample_data,
                    export_corrected_integrated_sample_data,
                    _sample_run_number,
                    _sample_data,
                    spectra_file_name=sample_master_dict[_sample_run_number][MasterDictKeys.spectra_file_name],
                    spectra_array=rebinned_payload["export_spectra_array"],
                    output_suffix=output_suffix,
                    bin_metadata=rebinned_payload["bin_metadata"],
                )

            if export_corrected_stack_of_ob_data or export_corrected_integrated_ob_data:
                if ob_master_dict:
                    first_ob_run_number = list(ob_master_dict.keys())[0]
                    export_ob_images(
                        ob_master_dict.keys(),
                        output_folder,
                        export_corrected_stack_of_ob_data,
                        export_corrected_integrated_ob_data,
                        ob_data_for_normalization,
                        spectra_file_name=ob_master_dict[first_ob_run_number][MasterDictKeys.spectra_file_name],
                        spectra_array=rebinned_payload["export_spectra_array"],
                        output_suffix=output_suffix,
                        bin_metadata=rebinned_payload["bin_metadata"],
                    )
                else:
                    logging.info("Skipping OB export because no OB runs were provided.")

            _normalized_dict = perform_normalization(_sample_data, ob_data_for_normalization, dc_data_for_normalization)
            _normalized_data = _normalized_dict['normalized_data']
            _integrated_normalized_data = _normalized_dict['integrated_normalized_data']       
            integrated_normalized_data[_sample_run_number] = _integrated_normalized_data
            normalized_data[_sample_run_number] = _normalized_data

            _spectrum_normalized_data = perform_spectrum_normalization(roi=roi, 
                                                                    sample_data=_sample_data,
                                                                    sample_variance=_sample_variance,
                                                                    ob_data_combined_for_spectrum=ob_data_combined_for_spectrum,
                                                                    ob_data_combined_variance_for_spectrum=ob_data_combined_variance_for_spectrum,
                                                                    dc_data_combined=dc_data_for_normalization,
                                                                    dc_data_combined_for_spectrum=dc_data_combined_for_spectrum,
                                                                    dc_data_combined_variance=rebinned_payload["dc_data_combined_variance"],
                                                                    dc_data_combined_variance_for_spectrum=dc_data_combined_variance_for_spectrum,
                                                                    black_filter_background_profile=rebinned_payload["black_filter_background_profile"],
                                                                    bragg_edge_cd_background_profile=rebinned_payload["bragg_edge_cd_background_profile"],
                                                                    measured_background_profiles=rebinned_payload["measured_background_profiles"])
            spectrum_normalized_data[_sample_run_number] = _spectrum_normalized_data

            # normalized_data[_sample_run_number] = np.array(np.divide(_sample_data, ob_data_combined))
            logging.info(f"{normalized_data[_sample_run_number].shape = }")
            logging.info(f"{normalized_data[_sample_run_number].dtype = }")
            logging.info(f"number of NaN in normalized data: {np.sum(np.isnan(normalized_data[_sample_run_number]))}")
            logging.info(f"number of inf in normalized data: {np.sum(np.isinf(normalized_data[_sample_run_number]))}")

            dict_to_return.tof_array = time_spectra

            dict_to_return.lambda_array = lambda_array
            dict_to_return.energy_array = energy_array

            logging.info(f"Preview: {preview = }")
            if preview:
                preview_normalized_data(_sample_data, 
                                        ob_data_for_normalization, 
                                        dc_data_for_normalization, 
                                        normalized_data, 
                                        lambda_array,
                                        energy_array, 
                                        current_detector_delay_us, 
                                        _sample_run_number,
                                        combine_samples,
                                        _spectrum_normalized_data,
                                        roi,
                                        bin_metadata=rebinned_payload["bin_metadata"],
                                        )
                
            if export_corrected_integrated_normalized_data or export_corrected_stack_of_normalized_data:

                export_normalized_data(ob_master_dict=ob_master_dict, 
                    sample_master_dict=sample_master_dict, 
                    _sample_run_number=_sample_run_number,
                    normalized_data=normalized_data, 
                    integrated_normalized_data=integrated_normalized_data,
                    _spectrum_normalized_data=_spectrum_normalized_data,
                    tof_array=time_spectra,
                    lambda_array=lambda_array, 
                    energy_array=energy_array, 
                    output_folder=output_folder, 
                    export_corrected_stack_of_normalized_data=export_corrected_stack_of_normalized_data,
                    export_corrected_integrated_normalized_data=export_corrected_integrated_normalized_data,
                    roi=roi,
                    spectra_array=rebinned_payload["export_spectra_array"],
                    spectra_file=sample_master_dict[_sample_run_number][MasterDictKeys.spectra_file_name],
                    output_suffix=output_suffix,
                    bin_metadata=rebinned_payload["bin_metadata"],
                    uncertainty_model_label=uncertainty_model_label)
          
    dict_to_return.data = normalized_data

    logging.info("Normalization and export is done!")
    if verbose:
        display(HTML("Normalization and export is done!"))

    return dict_to_return
