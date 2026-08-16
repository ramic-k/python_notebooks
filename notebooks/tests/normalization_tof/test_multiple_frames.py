import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from __code.normalization_tof import (
    DetectorType,
    RebinCustomBasis,
    RebinCustomScale,
    RebinMode,
)
from __code.normalization_tof import normalization_for_timepix1_timepix3 as production_normalization
from __code.normalization_tof import utilities as normalization_utilities
from __code.normalization_tof.multiple_frames import (
    AutoRoiConfig,
    BlackFilterBackgroundConfig,
    FRAME_SCALING_HYBRID_NATIVE_FLUX,
    PAIR_SCALING_NATIVE_FLUX,
    PAIR_SCALING_TRANSMISSION,
    FrameConfig,
    MeasuredBackgroundConfig,
    MultiFramePreviewEngine,
    MultiFrameRecipe,
    NativeFrameProfile,
    NativeFluxOverlapDiagnostics,
    OverlapWindowConfig,
    PairScalingConfig,
    RebinnedFramePreview,
    RebinConfig,
    ResolvedRun,
    RoiConfig,
    RoiProfileCache,
    RunSpec,
    SlitGapMetadata,
    calculate_overlap_diagnostics,
    calculate_native_flux_overlap_diagnostics,
    convert_tof_schedule_to_energy_schedule,
    export_scaled_spectrum_profile,
    load_integrated_image_preview,
    load_native_roi_profile,
    native_flux_overlap_arrays,
    overlap_ratio_arrays,
    parse_run_numbers,
    propose_roi_from_nexus,
    propose_roi_from_slit_gaps,
    read_slit_gaps_from_nexus,
    export_selected_spectrum_profile,
)
from __code.normalization_tof.multiple_frames_ui import (
    FrameEditor,
    MultiFrameNormalizationTof,
    PROMPT_FLASH_ENERGIES_EV,
    RunInputEditor,
    _add_prompt_flash_lines,
    _empty_frame,
    _frame_energy_log_range,
    _roi_preview_intensity_limits,
)
from __code.normalization_tof.utilities import (
    MasterDictKeys,
    build_rebin_bin_groups,
    calculate_detector_corrected_variance,
    load_data_using_multithreading,
    load_rebinned_normalized_data_from_tiffs,
    load_rebinned_tof_data,
    perform_spectrum_normalization,
    rebin_array_from_bin_groups,
)


def _write_run(root: Path, run_number: str, frames: list[np.ndarray], charge_c: float = 1.0):
    data_path = root / f"Run_{run_number}"
    data_path.mkdir()
    for index, frame in enumerate(frames):
        Image.fromarray(np.asarray(frame, dtype=np.uint16)).save(data_path / f"image{index:04d}.tif")
    tof = (np.arange(len(frames), dtype=float) + 0.5) * 1e-6
    (data_path / f"Run_{run_number}_Spectra.txt").write_text(
        "shutter_time,counts\n"
        + "".join(f"{time:.12g},0\n" for time in tof),
        encoding="utf-8",
    )
    (data_path / f"Run_{run_number}_ShutterCount.txt").write_text("0\t1000000\n", encoding="utf-8")

    nexus_path = root / f"VENUS_{run_number}.nxs.h5"
    with h5py.File(nexus_path, "w") as nexus:
        entry = nexus.create_group("entry")
        entry.create_dataset("proton_charge", data=[charge_c * 1e12])
        daslogs = entry.create_group("DASlogs")
        delay = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
        delay.create_dataset("value", data=[0.0])
    return data_path, nexus_path


def _write_slit_logs(
    nexus_path: Path,
    horizontal_value: float,
    vertical_value: float,
    units: str = "mm",
) -> None:
    with h5py.File(nexus_path, "a") as nexus:
        daslogs = nexus["entry/DASlogs"]
        for axis, value in (("X", horizontal_value), ("Y", vertical_value)):
            group = daslogs.create_group(f"BL10:Mot:s1:{axis}:Gap.RBV")
            dataset = group.create_dataset("average_value", data=[value])
            dataset.attrs["units"] = units


def test_read_slit_gaps_from_nexus_converts_length_units(tmp_path):
    _data_path, nexus_path = _write_run(
        tmp_path,
        "97",
        [np.ones((4, 4), dtype=np.uint16)],
    )
    _write_slit_logs(nexus_path, horizontal_value=1.5, vertical_value=1.2, units="cm")

    metadata = read_slit_gaps_from_nexus(nexus_path)

    assert metadata.horizontal_mm == 15.0
    assert metadata.vertical_mm == 12.0
    assert metadata.horizontal_path.endswith("X:Gap.RBV/average_value")
    assert metadata.vertical_path.endswith("Y:Gap.RBV/average_value")


def test_slit_roi_projection_uses_default_quarter_mm_edge_inset():
    image = np.ones((512, 512), dtype=np.float64)
    proposal = propose_roi_from_slit_gaps(
        image,
        SlitGapMetadata(
            horizontal_mm=15.0,
            vertical_mm=10.0,
            horizontal_path="x",
            vertical_path="y",
        ),
        AutoRoiConfig(refine_center_from_image=False),
    )

    assert proposal.roi == RoiConfig(left=124, top=170, width=263, height=172)
    assert proposal.center_x_pixels == 255.5
    assert proposal.center_y_pixels == 255.5


def test_slit_roi_projection_refines_center_from_integrated_image(tmp_path):
    image = np.zeros((512, 512), dtype=np.float64)
    image[170:370, 190:390] = 100.0
    _data_path, nexus_path = _write_run(
        tmp_path,
        "96",
        [np.ones((4, 4), dtype=np.uint16)],
    )
    _write_slit_logs(nexus_path, horizontal_value=10.0, vertical_value=10.0)

    proposal = propose_roi_from_nexus(
        image,
        nexus_path,
        AutoRoiConfig(edge_inset_mm=1.25, max_center_shift_pixels=64.0),
    )

    assert proposal.roi.width == 136
    assert proposal.roi.height == 136
    assert proposal.center_x_pixels == 288.5
    assert proposal.center_y_pixels == 268.5
    assert proposal.roi.left == 221
    assert proposal.roi.top == 201


def test_old_frame_recipe_defaults_to_pending_automatic_roi():
    frame = FrameConfig.from_dict(
        {
            "name": "0.3 A",
            "detector_type": DetectorType.tpx1,
            "sample_runs": ["1"],
            "ob_runs": ["2"],
            "roi": {"left": 10, "top": 11, "width": 12, "height": 13},
        }
    )

    assert frame.auto_roi.enabled
    assert not frame.auto_roi.applied


def test_tiff_stack_loader_preserves_frame_order_orientation_and_dtype(tmp_path):
    frames = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        np.array([[100, 200, 300], [400, 500, 600]], dtype=np.uint16),
    ]
    data_path, _ = _write_run(tmp_path, "99", frames)
    files = sorted(str(path) for path in data_path.glob("*.tif"))

    stack = load_data_using_multithreading(files)
    integrated = load_data_using_multithreading(files, combine_tof=True)
    expected = np.stack(frames, axis=0).astype(np.float32).swapaxes(1, 2)

    assert stack.dtype == np.float32
    np.testing.assert_array_equal(stack, expected)
    np.testing.assert_array_equal(integrated, expected.sum(axis=0, dtype=np.float32))


def test_streaming_rebin_matches_full_stack_counts_and_experimental_variance(tmp_path):
    frames = [
        np.array([[4, 2], [3, 1]], dtype=np.uint16),
        np.array([[2, 1], [1, 2]], dtype=np.uint16),
        np.array([[3, 1], [2, 4]], dtype=np.uint16),
        np.array([[1, 3], [2, 2]], dtype=np.uint16),
    ]
    data_path, _ = _write_run(tmp_path, "98", frames)
    files = sorted(str(path) for path in data_path.glob("*.tif"))
    groups = [np.asarray([0, 1]), np.asarray([2, 3])]

    streamed_data, streamed_variance = load_rebinned_tof_data(
        files,
        active_frame_groups=groups,
        shutter_counts=[1_000_000.0],
        use_experimental_uncertainties=True,
    )
    full_stack = load_data_using_multithreading(files)
    full_variance = calculate_detector_corrected_variance(
        full_stack,
        shutter_counts=[1_000_000.0],
    )

    np.testing.assert_allclose(
        streamed_data,
        rebin_array_from_bin_groups(full_stack, groups, reducer="sum"),
    )
    np.testing.assert_allclose(
        streamed_variance,
        rebin_array_from_bin_groups(full_variance, groups, reducer="sum"),
        rtol=1e-13,
    )


def test_streaming_native_ratio_matches_production_repeated_run_scaling(tmp_path):
    shape = (2, 2)
    sample_1 = [np.full(shape, value, dtype=np.uint16) for value in (2, 6, 3, 9)]
    sample_2 = [np.full(shape, value, dtype=np.uint16) for value in (1, 3, 2, 6)]
    ob_1 = [np.full(shape, value, dtype=np.uint16) for value in (2, 4, 3, 6)]
    ob_2 = [np.full(shape, value, dtype=np.uint16) for value in (4, 8, 6, 12)]
    paths = {}
    for run_number, frames, charge in (
        ("91", sample_1, 2.0),
        ("92", sample_2, 1.0),
        ("93", ob_1, 1.0),
        ("94", ob_2, 3.0),
    ):
        data_path, _ = _write_run(tmp_path, run_number, frames, charge_c=charge)
        paths[run_number] = sorted(str(path) for path in data_path.glob("*.tif"))

    def info(run_number, charge):
        return {
            MasterDictKeys.list_tif: paths[run_number],
            MasterDictKeys.proton_charge: charge,
        }

    sample_master = {"91": info("91", 2.0), "92": info("92", 1.0)}
    ob_master = {"93": info("93", 1.0), "94": info("94", 3.0)}
    groups = [np.asarray([0, 1]), np.asarray([2, 3])]

    streamed = load_rebinned_normalized_data_from_tiffs(
        sample_master_dict=sample_master,
        ob_master_dict=ob_master,
        sample_run_numbers=["91", "92"],
        active_frame_groups=groups,
        combine_samples=True,
        use_proton_charge=True,
    )
    sample_native = (np.asarray([2, 6, 3, 9]) + np.asarray([1, 3, 2, 6])) / 3.0
    ob_native = (
        np.asarray([2, 4, 3, 6])
        + np.asarray([4, 8, 6, 12])
    ) / 4.0
    native_ratio = sample_native / ob_native
    expected = np.asarray(
        [
            np.mean(native_ratio[:2]),
            np.mean(native_ratio[2:]),
        ],
        dtype=np.float32,
    )[:, None, None]
    expected = np.broadcast_to(expected, streamed.shape)

    np.testing.assert_allclose(streamed, expected, rtol=1e-7)


def test_streaming_native_ratio_applies_weighted_measured_backgrounds(tmp_path):
    shape = (2, 2)
    frame_values = {
        "85": (10, 18, 12, 16),
        "86": (20, 30, 24, 32),
        "87": (2, 4, 2, 4),
        "88": (4, 6, 4, 8),
    }
    paths = {}
    for run_number, values in frame_values.items():
        frames = [np.full(shape, value, dtype=np.uint16) for value in values]
        data_path, _ = _write_run(tmp_path, run_number, frames)
        paths[run_number] = sorted(str(path) for path in data_path.glob("*.tif"))

    def info(run_number):
        return {
            MasterDictKeys.list_tif: paths[run_number],
            MasterDictKeys.proton_charge: 1.0,
        }

    groups = [np.asarray([0, 1]), np.asarray([2, 3])]
    streamed = load_rebinned_normalized_data_from_tiffs(
        sample_master_dict={"85": info("85")},
        ob_master_dict={"86": info("86")},
        sample_run_numbers=["85"],
        active_frame_groups=groups,
        combine_samples=True,
        use_proton_charge=True,
        measured_background_runtime_configs=[
            {
                "weight": 0.5,
                "sample_master_dict": {"87": info("87")},
                "ob_master_dict": {"88": info("88")},
            }
        ],
    )

    sample = np.asarray(frame_values["85"], dtype=np.float64)
    ob = np.asarray(frame_values["86"], dtype=np.float64)
    sample_background = np.asarray(frame_values["87"], dtype=np.float64)
    ob_background = np.asarray(frame_values["88"], dtype=np.float64)
    corrected_native_ratio = (
        (sample - 0.5 * sample_background)
        / (ob - 0.5 * ob_background)
    )
    expected = np.asarray(
        [
            corrected_native_ratio[:2].mean(),
            corrected_native_ratio[2:].mean(),
        ],
        dtype=np.float32,
    )[:, None, None]

    np.testing.assert_allclose(
        streamed,
        np.broadcast_to(expected, streamed.shape),
        rtol=1e-7,
    )


def test_production_normalization_uses_streaming_rebin_path(tmp_path, monkeypatch):
    shape = (2, 2)
    sample_frames = [np.full(shape, value, dtype=np.uint16) for value in (1, 9, 2, 6)]
    ob_frames = [np.full(shape, value, dtype=np.uint16) for value in (1, 3, 2, 2)]
    sample_path, sample_nexus = _write_run(tmp_path, "81", sample_frames)
    ob_path, ob_nexus = _write_run(tmp_path, "82", ob_frames)
    output = tmp_path / "output"
    output.mkdir()
    roi = RoiConfig(left=0, top=0, width=2, height=2).to_roi()

    monkeypatch.setattr(production_normalization, "initialize_logging", lambda: None)

    def fail_full_stack_loader(*_args, **_kwargs):
        raise AssertionError("Stage 1 full-stack loader was called")

    monkeypatch.setattr(
        normalization_utilities,
        "load_data_using_multithreading",
        fail_full_stack_loader,
    )
    result = production_normalization.normalization_with_list_of_full_path(
        sample_dict={
            sample_path.name: {
                "full_path": str(sample_path),
                "nexus": str(sample_nexus),
            }
        },
        ob_dict={
            ob_path.name: {
                "full_path": str(ob_path),
                "nexus": str(ob_nexus),
            }
        },
        dc_dict={},
        spectra_array=(np.arange(4, dtype=np.float64) + 0.5) * 1e-6,
        output_folder=str(output),
        verbose=False,
        proton_charge_flag=False,
        output_tif=False,
        preview=False,
        correct_chips_alignment_flag=False,
        export_mode={
            "sample_stack": False,
            "ob_stack": False,
            "normalized_stack": False,
            "sample_integrated": False,
            "ob_integrated": False,
            "normalized_integrated": False,
            "x_axis": False,
        },
        roi=roi,
        sample_roi=roi,
        ob_roi=roi,
        rebin_mode=RebinMode.linear_tof,
        rebin_delta_tof_us=2.0,
        experimental_uncertainties_flag=True,
    )

    normalized = next(iter(result.data.values()))
    expected_native_ratio = np.asarray([1.0, 3.0, 1.0, 3.0], dtype=np.float32)
    expected = np.asarray(
        [expected_native_ratio[:2].mean(), expected_native_ratio[2:].mean()],
        dtype=np.float32,
    )[:, None, None]
    np.testing.assert_allclose(normalized, np.broadcast_to(expected, normalized.shape))
    assert len(result.tof_array) == 2


def test_production_streaming_rebin_supports_measured_background(tmp_path, monkeypatch):
    shape = (2, 2)
    frame_values = {
        "75": (10, 18, 12, 16),
        "76": (20, 30, 24, 32),
        "77": (2, 4, 2, 4),
        "78": (4, 6, 4, 8),
    }
    runs = {}
    for run_number, values in frame_values.items():
        frames = [np.full(shape, value, dtype=np.uint16) for value in values]
        runs[run_number] = _write_run(tmp_path, run_number, frames)

    def run_dict(run_number):
        data_path, nexus_path = runs[run_number]
        return {
            data_path.name: {
                "full_path": str(data_path),
                "nexus": str(nexus_path),
            }
        }

    output = tmp_path / "background_output"
    output.mkdir()
    roi = RoiConfig(left=0, top=0, width=2, height=2).to_roi()
    monkeypatch.setattr(production_normalization, "initialize_logging", lambda: None)

    def fail_full_stack_loader(*_args, **_kwargs):
        raise AssertionError("Stage 1 full-stack loader was called")

    monkeypatch.setattr(
        normalization_utilities,
        "load_data_using_multithreading",
        fail_full_stack_loader,
    )
    result = production_normalization.normalization_with_list_of_full_path(
        sample_dict=run_dict("75"),
        ob_dict=run_dict("76"),
        dc_dict={},
        spectra_array=(np.arange(4, dtype=np.float64) + 0.5) * 1e-6,
        output_folder=str(output),
        verbose=False,
        proton_charge_flag=True,
        output_tif=False,
        preview=False,
        correct_chips_alignment_flag=False,
        export_mode={
            "sample_stack": False,
            "ob_stack": False,
            "normalized_stack": False,
            "sample_integrated": False,
            "ob_integrated": False,
            "normalized_integrated": False,
            "x_axis": False,
        },
        roi=roi,
        sample_roi=roi,
        ob_roi=roi,
        rebin_mode=RebinMode.linear_tof,
        rebin_delta_tof_us=2.0,
        experimental_uncertainties_flag=True,
        measured_background_correction_configs=[
            {
                "enabled": True,
                "weight": 0.5,
                "mode": "test measured background",
                "column_label": "test background",
                "key_prefix": "test_background",
                "sample_background_dict": run_dict("77"),
                "ob_background_dict": run_dict("78"),
            }
        ],
    )

    sample = np.asarray(frame_values["75"], dtype=np.float64)
    ob = np.asarray(frame_values["76"], dtype=np.float64)
    sample_background = np.asarray(frame_values["77"], dtype=np.float64)
    ob_background = np.asarray(frame_values["78"], dtype=np.float64)
    corrected_native_ratio = (
        (sample - 0.5 * sample_background)
        / (ob - 0.5 * ob_background)
    )
    expected = np.asarray(
        [
            corrected_native_ratio[:2].mean(),
            corrected_native_ratio[2:].mean(),
        ],
        dtype=np.float32,
    )[:, None, None]
    normalized = next(iter(result.data.values()))

    np.testing.assert_allclose(
        normalized,
        np.broadcast_to(expected, normalized.shape),
        rtol=1e-7,
    )
    assert len(result.tof_array) == 2


def _frame(
    name: str,
    sample: list[tuple[str, Path, Path]],
    ob: list[tuple[str, Path, Path]],
    roi: RoiConfig,
    rebin: RebinConfig | None = None,
    ob_roi: RoiConfig | None = None,
):
    def specs(values):
        return tuple(
            RunSpec(run_number=run, data_path=str(data_path), nexus_path=str(nexus))
            for run, data_path, nexus in values
        )

    return FrameConfig(
        name=name,
        detector_type=DetectorType.tpx1,
        sample_runs=specs(sample),
        ob_runs=specs(ob),
        roi=roi,
        auto_roi=AutoRoiConfig(enabled=False),
        ob_roi=ob_roi,
        rebin=rebin or RebinConfig(),
        use_experimental_uncertainties=False,
    )


def test_run_parser_and_recipe_round_trip(tmp_path):
    assert parse_run_numbers("19419, 19447-19448, Run_19599") == ["19419", "19447", "19448", "19599"]
    frame = FrameConfig(
        name="0.3 A",
        detector_type=DetectorType.tpx1,
        sample_runs=(RunSpec("19477"),),
        ob_runs=(RunSpec("19478"),),
        roi=RoiConfig(left=52, top=62, width=411, height=401),
        ob_roi=RoiConfig(left=53, top=63, width=410, height=400),
        black_filter_background=BlackFilterBackgroundConfig(
            enabled=False,
            shape_file="/tmp/background.csv",
            anchor_energy_eV=5.1,
        ),
        rebin=RebinConfig(
            mode=RebinMode.linear_tof,
            delta_tof_us=100,
            full_bins_only=True,
        ),
        output_energy_min_eV=0.011,
        output_energy_max_eV=0.2,
        output_excluded_energy_ranges_eV=((0.031, 0.034), (0.071, 0.072)),
    )
    recipe = MultiFrameRecipe(
        working_dir="/SNS/VENUS/IPTS-36914",
        frames=[frame],
        output_root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        same_rois_all_frames=True,
        export_scaled_spectra=True,
        frame_multipliers={"0.3 A": 0.987},
        frame_scaling_mode=FRAME_SCALING_HYBRID_NATIVE_FLUX,
        frame_sample_flux_multipliers={"0.3 A": 1.234},
        frame_ob_flux_multipliers={"0.3 A": 1.25},
        show_hybrid_flux_plots=False,
        overlap_windows=(
            OverlapWindowConfig("0.3 A", "resonance", 0.11, 0.19),
        ),
        pair_scaling_methods=(
            PairScalingConfig("0.3 A", "resonance", PAIR_SCALING_NATIVE_FLUX),
        ),
        show_prompt_flash_lines=True,
    )
    recipe_path = recipe.save(tmp_path / "recipe.json")
    loaded = MultiFrameRecipe.load(recipe_path)
    assert loaded == recipe
    assert loaded.frames[0].effective_ob_roi() == RoiConfig(left=53, top=63, width=410, height=400)
    assert loaded.frames[0].spectrum_only is True
    assert loaded.same_rois_all_frames is True
    assert loaded.export_scaled_spectra is True
    assert loaded.frame_multipliers == {"0.3 A": 0.987}
    assert loaded.frame_scaling_mode == FRAME_SCALING_HYBRID_NATIVE_FLUX
    assert loaded.frame_sample_flux_multipliers == {"0.3 A": 1.234}
    assert loaded.frame_ob_flux_multipliers == {"0.3 A": 1.25}
    assert loaded.show_hybrid_flux_plots is False
    assert loaded.overlap_windows == (
        OverlapWindowConfig("0.3 A", "resonance", 0.11, 0.19),
    )
    assert loaded.pair_scaling_methods == (
        PairScalingConfig("0.3 A", "resonance", PAIR_SCALING_NATIVE_FLUX),
    )
    assert loaded.frames[0].output_energy_min_eV == 0.011
    assert loaded.frames[0].output_energy_max_eV == 0.2
    assert loaded.frames[0].output_excluded_energy_ranges_eV == (
        (0.031, 0.034),
        (0.071, 0.072),
    )
    assert loaded.show_prompt_flash_lines is True
    assert json.loads(recipe_path.read_text())["recipe_version"] == 1


def test_scaled_spectrum_profile_scales_transmissions_but_not_counts(tmp_path):
    source = tmp_path / "spectrum_normalization_profile.txt"
    source.write_text(
        "# uncertainty model: test\n"
        "sample ROI counts,spectrum normalization,spectrum normalization uncertainty,"
        "closed-slits corrected spectrum normalization,"
        "closed-slits corrected spectrum normalization uncertainty\n"
        "10,0.5,0.02,0.4,0.03\n"
        "20,0.6,0.04,0.5,0.05\n",
        encoding="utf-8",
    )

    output = export_scaled_spectrum_profile(source, 1.25)
    original = pd.read_csv(source, comment="#")
    scaled = pd.read_csv(output, comment="#")

    np.testing.assert_allclose(scaled["sample ROI counts"], original["sample ROI counts"])
    np.testing.assert_allclose(scaled["spectrum normalization"], [0.625, 0.75])
    np.testing.assert_allclose(scaled["spectrum normalization uncertainty"], [0.025, 0.05])
    np.testing.assert_allclose(
        scaled["closed-slits corrected spectrum normalization"],
        [0.5, 0.625],
    )
    np.testing.assert_allclose(
        scaled["closed-slits corrected spectrum normalization uncertainty"],
        [0.0375, 0.0625],
    )
    assert "# frame multiplier: 1.25" in output.read_text(encoding="utf-8")


def test_scaled_spectrum_profile_applies_optional_energy_limits(tmp_path):
    source = tmp_path / "spectrum_normalization_profile.txt"
    source.write_text(
        "mean_energy (eV),sample ROI counts,spectrum normalization,"
        "spectrum normalization uncertainty\n"
        "0.01,10,0.5,0.02\n"
        "0.1,20,0.6,0.03\n"
        "1.0,30,0.7,0.04\n",
        encoding="utf-8",
    )

    output = export_scaled_spectrum_profile(
        source,
        2.0,
        energy_min_eV=0.05,
        energy_max_eV=0.5,
    )
    scaled = pd.read_csv(output, comment="#")

    np.testing.assert_allclose(scaled["mean_energy (eV)"], [0.1])
    np.testing.assert_allclose(scaled["sample ROI counts"], [20])
    np.testing.assert_allclose(scaled["spectrum normalization"], [1.2])


def test_selected_spectrum_profile_preserves_full_profile(tmp_path):
    source = tmp_path / "spectrum_normalization_profile.txt"
    source.write_text(
        "# uncertainty model: test\n"
        "mean_energy (eV),sample ROI counts,spectrum normalization\n"
        "0.01,10,0.5\n"
        "0.1,20,0.6\n"
        "1.0,30,0.7\n",
        encoding="utf-8",
    )
    original = source.read_bytes()

    output = export_selected_spectrum_profile(
        source,
        energy_min_eV=0.05,
        energy_max_eV=0.5,
    )

    assert output.name == "spectrum_normalization_profile_selected.txt"
    assert source.read_bytes() == original
    selected = pd.read_csv(output, comment="#")
    np.testing.assert_allclose(selected["mean_energy (eV)"], [0.1])
    assert "# uncertainty model: test" in output.read_text(encoding="utf-8")
    assert "# frame output energy range (eV): 0.05, 0.5" in output.read_text(
        encoding="utf-8"
    )

    second_output = export_selected_spectrum_profile(source, energy_min_eV=0.5)
    reselection = pd.read_csv(second_output, comment="#")
    np.testing.assert_allclose(reselection["mean_energy (eV)"], [1.0])
    assert source.read_bytes() == original


def test_selected_and_scaled_profiles_apply_inclusive_excluded_ranges(tmp_path):
    source = tmp_path / "spectrum_normalization_profile.txt"
    source.write_text(
        "mean_energy (eV),sample ROI counts,spectrum normalization,"
        "spectrum normalization uncertainty\n"
        "0.01,10,0.5,0.02\n"
        "0.1,20,0.6,0.03\n"
        "0.2,30,0.7,0.04\n"
        "0.3,40,0.8,0.05\n",
        encoding="utf-8",
    )

    selected_path = export_selected_spectrum_profile(
        source,
        excluded_energy_ranges_eV=((0.1, 0.2),),
    )
    scaled_path = export_scaled_spectrum_profile(
        source,
        2.0,
        output_name="scaled_selected.txt",
        excluded_energy_ranges_eV=((0.1, 0.2),),
    )

    selected = pd.read_csv(selected_path, comment="#")
    scaled = pd.read_csv(scaled_path, comment="#")
    np.testing.assert_allclose(selected["mean_energy (eV)"], [0.01, 0.3])
    np.testing.assert_allclose(scaled["mean_energy (eV)"], [0.01, 0.3])
    np.testing.assert_allclose(scaled["spectrum normalization"], [1.0, 1.6])
    assert "# excluded output energy ranges (eV, inclusive):" in (
        selected_path.read_text(encoding="utf-8")
    )


def test_prompt_flash_helper_adds_only_visible_fixed_markers():
    figure = go.Figure()
    _add_prompt_flash_lines(
        figure,
        True,
        energy_min_eV=0.002,
        energy_max_eV=0.012,
    )

    marker_positions = [float(shape.x0) for shape in figure.layout.shapes]
    np.testing.assert_allclose(marker_positions, PROMPT_FLASH_ENERGIES_EV[1:])

    hidden = go.Figure()
    _add_prompt_flash_lines(hidden, False)
    assert not hidden.layout.shapes


def test_prompt_flash_marker_does_not_expand_frame_energy_axis():
    frame_energy = np.asarray([0.0035, 0.0049, 0.0065, 0.0117628, 0.013])
    expected_range = _frame_energy_log_range(frame_energy)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame_energy, y=np.ones(frame_energy.size)))
    figure.update_xaxes(type="log", range=expected_range)

    _add_prompt_flash_lines(
        figure,
        True,
        energy_min_eV=float(np.min(frame_energy)),
        energy_max_eV=float(np.max(frame_energy)),
    )

    np.testing.assert_allclose(figure.layout.xaxis.range, expected_range)
    np.testing.assert_allclose(
        [float(shape.x0) for shape in figure.layout.shapes],
        [PROMPT_FLASH_ENERGIES_EV[-1]],
    )


def test_direct_data_folder_infers_nexus_from_its_own_ipts(tmp_path):
    source_ipts = tmp_path / "SNS" / "VENUS" / "IPTS-36914"
    data_path = (
        source_ipts
        / "shared"
        / "autoreduce"
        / "images"
        / "tpx1"
        / "raw"
        / "radiography"
        / "Run_19560"
    )
    data_path.mkdir(parents=True)
    nexus_path = source_ipts / "nexus" / "VENUS_19560.nxs.h5"
    nexus_path.parent.mkdir()
    with h5py.File(nexus_path, "w") as nexus:
        entry = nexus.create_group("entry")
        entry.create_dataset("proton_charge", data=[1.0e12])

    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path / "SNS" / "VENUS" / "IPTS-35167"),
        frames=[],
    )
    resolved = MultiFramePreviewEngine(recipe).resolve_run(
        RunSpec("19560", data_path=str(data_path)),
        DetectorType.tpx1,
    )

    assert resolved.data_path == data_path
    assert resolved.nexus_path == nexus_path


def test_new_frame_defaults_match_single_frame_notebook():
    frame = _empty_frame("frame", DetectorType.tpx1)
    recipe = MultiFrameRecipe(working_dir="/SNS/VENUS/IPTS-36914", frames=[frame])
    editor = FrameEditor(frame, recipe.working_dir)

    assert editor.roi_left.value == 156
    assert editor.roi_top.value == 156
    assert editor.roi_width.value == 200
    assert editor.roi_height.value == 200
    assert editor.auto_roi_enabled.value is True
    assert editor.auto_roi_projection_scale.value == 1.0
    assert editor.auto_roi_edge_inset_mm.value == 0.25
    assert editor.auto_roi_refine_center.value is True
    assert editor.ob_roi_linked.value is False
    assert editor.ob_roi_linked.disabled is True
    assert editor.ob_roi_left.value == 156
    assert editor.ob_roi_top.value == 156
    assert editor.ob_roi_width.value == 200
    assert editor.ob_roi_height.value == 200
    assert editor.ob_roi_left.disabled is False
    assert editor.rebin_mode.value == RebinMode.none
    assert editor.custom_basis.value == RebinCustomBasis.energy_tof
    assert editor.preview_bins_button.disabled is True
    assert editor.delta_tof.value == 30.0
    assert editor.delta_lambda.value == 0.01
    assert editor.delta_tof_relative.value == 0.01
    assert editor.delta_lambda_relative.value == 0.01
    assert editor.full_bins_only.value is True
    assert editor.snap_to_native.value is True
    assert editor.use_proton_charge.value is True
    assert editor.experimental_uncertainties.value is True
    assert editor.spectrum_only.value is True
    assert editor.combine_sample_runs.value is True
    assert editor.correct_chips_alignment.value is False
    assert editor.correct_chips_alignment.disabled is True
    assert editor.black_filter_enabled.value is False
    assert editor.cd_background_enabled.value is False
    assert editor.closed_background_enabled.value is False
    assert editor.replace_ob_zeros.value is False
    assert (editor.median_kernel_y.value, editor.median_kernel_x.value, editor.median_kernel_tof.value) == (3, 3, 1)
    assert editor.median_iterations.value == 2
    assert editor.container_enabled.value is False
    assert editor.sample_runs.style.description_width == "initial"
    assert editor.manual_tof.style.description_width == "initial"
    assert editor.container_roi_box.layout.display == "none"
    assert editor.roi_row.layout.display == "flex"
    assert editor.roi_preview_row.layout.display == "flex"
    assert editor.axis_row.layout.display == "flex"
    assert editor.rebin_flags_row.layout.display == "flex"
    assert editor.roi_preview_button.description == "Preview/select sample ROI"
    assert editor.ob_roi_preview_button.description == "Preview/select OB ROI"
    assert recipe.export_mode == {
        "sample_stack": False,
        "ob_stack": False,
        "normalized_stack": True,
        "sample_integrated": False,
        "ob_integrated": False,
        "normalized_integrated": False,
        "combined_normalized_integrated": False,
        "x_axis": True,
    }

    editor.auto_roi_enabled.value = False
    assert editor.ob_roi_linked.disabled is False
    editor.ob_roi_linked.value = True
    assert editor.ob_roi_linked.value is True
    editor.auto_roi_enabled.value = True
    assert editor.ob_roi_linked.value is False
    assert editor.ob_roi_linked.disabled is True

    editor.detector.value = DetectorType.tpx3
    assert editor.experimental_uncertainties.value is False
    assert editor.experimental_uncertainties.disabled is True
    editor.detector.value = DetectorType.tpx1
    assert editor.experimental_uncertainties.value is True
    assert editor.experimental_uncertainties.disabled is False


def test_energy_edge_tof_width_schedule_reverses_regions_onto_tof_axis():
    tof_us = np.arange(1, 22, dtype=np.float64) * 100.0
    tof_s = tof_us * 1e-6
    energy_eV = np.linspace(2.1, 0.1, len(tof_s))
    lambda_a = np.linspace(0.1, 2.1, len(tof_s))

    groups, edges = build_rebin_bin_groups(
        rebin_mode=RebinMode.custom_schedule,
        tof_array=tof_s,
        lambda_array=lambda_a,
        energy_array=energy_eV,
        rebin_custom_basis=RebinCustomBasis.energy_tof,
        rebin_custom_scale=RebinCustomScale.linear,
        rebin_custom_schedule=[
            {"end_value": 0.7, "step": 300.0},
            {"end_value": 1.3, "step": 100.0},
            {"end_value": None, "step": 200.0},
        ],
        rebin_full_bins_only=False,
        rebin_snap_to_native_grid=False,
    )

    np.testing.assert_allclose(
        edges * 1e6,
        [100, 300, 500, 700, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1800, 2100],
        atol=1e-9,
    )
    assert sum(len(group) for group in groups) == len(tof_s) - 1


def test_energy_edge_tof_width_schedule_rejects_out_of_frame_edge():
    with np.testing.assert_raises_regex(ValueError, "strictly inside this frame's energy range"):
        build_rebin_bin_groups(
            rebin_mode=RebinMode.custom_schedule,
            tof_array=np.asarray([1.0, 2.0, 3.0]) * 1e-3,
            lambda_array=np.asarray([1.0, 2.0, 3.0]),
            energy_array=np.asarray([3.0, 2.0, 1.0]),
            rebin_custom_basis=RebinCustomBasis.energy_tof,
            rebin_custom_scale=RebinCustomScale.linear,
            rebin_custom_schedule=[{"end_value": 4.0, "step": 100.0}],
        )


def test_tof_schedule_conversion_reverses_edges_and_preserves_physical_widths():
    tof_s = np.arange(0.0, 1.1e-3, 1.0e-4)
    energy_eV = np.arange(11.0, 0.0, -1.0)

    converted = convert_tof_schedule_to_energy_schedule(
        (
            (200.0, 20.0),
            (500.0, 30.0),
            (800.0, 40.0),
            (None, 50.0),
        ),
        tof_s,
        energy_eV,
    )

    assert converted == (
        (3.0, 50.0),
        (6.0, 40.0),
        (9.0, 30.0),
        (None, 20.0),
    )


def test_named_frame_starts_with_earlier_tof_recipe_for_automatic_conversion():
    editor = FrameEditor(_empty_frame("6.3 A", DetectorType.tpx1), "/SNS/VENUS/IPTS-36914")

    assert editor.custom_basis.value == RebinCustomBasis.tof
    assert editor.custom_schedule.value == "886, 70\n4015, 100\n, 150"


def test_frame_editor_bin_preview_converts_legacy_tof_schedule(tmp_path, monkeypatch):
    sample_frames = [np.full((2, 2), 20 + index, dtype=np.uint16) for index in range(8)]
    ob_frames = [np.full((2, 2), 40 + index, dtype=np.uint16) for index in range(8)]
    sample_path, sample_nexus = _write_run(tmp_path, "721", sample_frames)
    ob_path, ob_nexus = _write_run(tmp_path, "722", ob_frames)
    frame = _frame(
        "legacy preview",
        [("721", sample_path, sample_nexus)],
        [("722", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
        rebin=RebinConfig(
            mode=RebinMode.custom_schedule,
            custom_basis=RebinCustomBasis.tof,
            custom_scale=RebinCustomScale.linear,
            custom_schedule=((2.5, 1.0), (5.5, 2.0), (None, 1.0)),
            full_bins_only=False,
            snap_to_native_grid=False,
        ),
    )
    editor = FrameEditor(frame, str(tmp_path))
    editor.use_proton_charge.value = False
    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: captured.append(figure))

    editor._preview_bins()

    assert len(captured) == 3, editor.rebin_bin_summary.value
    assert editor.custom_basis.value == RebinCustomBasis.energy_tof
    converted = editor.to_config().rebin.custom_schedule
    assert [edge for edge, _ in converted[:-1]] == sorted(edge for edge, _ in converted[:-1])
    assert [width for _, width in converted] == [1.0, 2.0, 1.0]
    assert "converted 2 earlier TOF boundaries" in editor.rebin_bin_summary.value


def test_ob_roi_can_be_unlinked_and_saved_independently():
    frame = _empty_frame("frame", DetectorType.tpx1)
    editor = FrameEditor(frame, "/SNS/VENUS/IPTS-36914")

    editor.ob_roi_linked.value = False
    editor.ob_roi_left.value = 188
    editor.ob_roi_top.value = 198
    editor.ob_roi_width.value = 135
    editor.ob_roi_height.value = 131

    config = editor.to_config()
    assert config.roi == RoiConfig(left=156, top=156, width=200, height=200)
    assert config.ob_roi == RoiConfig(left=188, top=198, width=135, height=131)
    assert editor.ob_roi_left.disabled is False


def test_same_rois_all_frames_propagates_both_sample_and_ob_values():
    planner = MultiFrameNormalizationTof(
        working_dir="/SNS/VENUS/IPTS-36914",
        frames=[_empty_frame("first", DetectorType.tpx1), _empty_frame("second", DetectorType.tpx1)],
    )
    planner.same_rois_all_frames.value = True
    source, target = planner.frame_editors

    source.roi_left.value = 47
    source.roi_top.value = 36
    source.ob_roi_linked.value = False
    source.ob_roi_left.value = 52
    source.ob_roi_top.value = 62

    assert target.roi_left.value == 47
    assert target.roi_top.value == 36
    assert target.ob_roi_linked.value is False
    assert target.ob_roi_left.value == 52
    assert target.ob_roi_top.value == 62
    assert planner.recipe().same_rois_all_frames is True


def test_direct_folder_override_can_infer_run_number(tmp_path):
    run_path = tmp_path / "normalized_Run_19558_images"
    run_path.mkdir()
    editor = RunInputEditor("Sample", (), str(tmp_path))
    editor.folders.value = str(run_path)
    specs = editor.specs()
    assert specs == (RunSpec(run_number="19558", data_path=str(run_path)),)
    assert editor.runs.value == "19558"


def test_overlap_inspection_provides_manual_window_for_each_adjacent_pair():
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    names = ("6.3 A", "4.5 A", "2.5 A", "0.3 A", "resonance")
    assert ui.inspect_frames.value == names
    assert ui._selected_frame_names() == list(names)
    assert tuple(ui.frame_scale_widgets) == names
    assert all(widget.value == 1.0 for widget in ui.frame_scale_widgets.values())
    assert all(widget.disabled is False for widget in ui.frame_scale_widgets.values())
    assert len(ui.overlap_window_widgets) == len(names) - 1
    assert all(
        controls["auto"].value
        and controls["minimum"].disabled
        and controls["maximum"].disabled
        for controls in ui.overlap_window_widgets.values()
    )

    pair = ui.overlap_window_widgets[ui._overlap_pair_key("2.5 A", "0.3 A")]
    pair["auto"].value = False
    pair["minimum"].value = 0.0105
    pair["maximum"].value = 0.0115

    assert ui._overlap_window_for_pair("2.5 A", "0.3 A") == (0.0105, 0.0115)
    assert ui._overlap_window_for_pair("0.3 A", "2.5 A") == (0.0105, 0.0115)
    assert ui._overlap_window_for_pair("0.3 A", "resonance", None) is None
    assert ui._overlap_window_for_pair("resonance", "0.3 A", (0.1, 0.3)) == (0.1, 0.3)

    windows = ui.recipe().overlap_windows
    assert windows == (
        OverlapWindowConfig("2.5 A", "0.3 A", 0.0105, 0.0115),
    )


def test_highest_energy_overlap_cap_is_coverage_driven_not_name_driven():
    ui = MultiFrameNormalizationTof(
        "/SNS/VENUS/IPTS-36914",
        frames=[
            _empty_frame("thermal", DetectorType.tpx1),
            _empty_frame("fast arbitrary name", DetectorType.tpx3),
        ],
    )
    ui.engine = SimpleNamespace(
        previews={
            "thermal": _preview(
                "thermal",
                [0.05, 0.1, 0.2, 0.4],
                [0.5] * 4,
                [0.01] * 4,
            ),
            "fast arbitrary name": _preview(
                "fast arbitrary name",
                [0.08, 0.2, 1.0, 10.0],
                [0.5] * 4,
                [0.01] * 4,
            ),
        }
    )

    assert ui._overlap_window_for_pair(
        "thermal",
        "fast arbitrary name",
    ) == (0.0, 0.2)


def test_frame_move_controls_reorder_all_outputs_without_rebuilding_editors():
    frames = [
        _empty_frame("low", DetectorType.tpx1),
        _empty_frame("middle", DetectorType.tpx1),
        _empty_frame("high", DetectorType.tpx1),
    ]
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914", frames=frames)
    low, middle, high = ui.frame_editors
    middle.roi_left.value = 123
    middle.output_energy_min.value = 0.02
    ui.frame_scale_widgets["middle"].value = 1.25

    middle.move_up_button.click()

    assert ui.frame_editors == [middle, low, high]
    assert middle.roi_left.value == 123
    assert middle.output_energy_min.value == 0.02
    assert tuple(ui.inspect_frames.options) == ("middle", "low", "high")
    assert tuple(ui.frame_scale_widgets) == ("middle", "low", "high")
    assert ui.frame_scale_widgets["middle"].value == 1.25
    assert [frame.name for frame in ui.recipe().frames] == ["middle", "low", "high"]
    assert [
        (controls["frame_a"], controls["frame_b"])
        for controls in ui.overlap_window_widgets.values()
    ] == [("middle", "low"), ("low", "high")]
    assert middle.move_up_button.disabled is True
    assert middle.move_down_button.disabled is False
    assert high.move_down_button.disabled is True
    assert ui.frame_box.selected_index == 0

    middle.move_down_button.click()
    assert ui.frame_editors == [low, middle, high]
    assert ui.frame_box.selected_index == 1


def test_integrated_roi_preview_uses_production_orientation_and_subsampling(tmp_path):
    frames = [np.arange(20, dtype=np.uint16).reshape(4, 5) + offset for offset in (0, 20, 40)]
    data_path, _ = _write_run(tmp_path, "200", frames)
    integrated, selected_count, total_count = load_integrated_image_preview(data_path, max_images=2)
    np.testing.assert_array_equal(integrated, frames[0].astype(float).T + frames[2].astype(float).T)
    assert selected_count == 2
    assert total_count == 3


def test_roi_preview_intensity_limits_ignore_isolated_hot_pixel():
    image = np.full((100, 100), 200.0)
    image[0, 0] = 2_329_179.0

    low, high, data_max = _roi_preview_intensity_limits(image)

    assert (low, high) == (200, 201)
    assert data_max == 2_329_179


def test_roi_profile_matches_production_transposed_orientation(tmp_path):
    frames = [np.arange(20, dtype=np.uint16).reshape(4, 5) + offset for offset in (0, 20)]
    data_path, nexus_path = _write_run(tmp_path, "100", frames, charge_c=2.0)
    roi = RoiConfig(left=1, top=2, width=2, height=2)
    profile = load_native_roi_profile(
        ResolvedRun("100", data_path, nexus_path),
        roi=roi,
        use_experimental_uncertainties=False,
        manual_tof_bin_size_ns=None,
    )
    expected = [np.sum(frame.astype(float).T[2:4, 1:3]) for frame in frames]
    np.testing.assert_allclose(profile.counts, expected)
    np.testing.assert_allclose(profile.variance, expected)
    assert profile.proton_charge_c == 2.0


def test_roi_profile_falls_back_for_compressed_tiff(tmp_path):
    frames = [np.arange(20, dtype=np.uint16).reshape(4, 5)]
    data_path, nexus_path = _write_run(tmp_path, "105", frames)
    Image.fromarray(frames[0]).save(data_path / "image0000.tif", compression="tiff_lzw")
    roi = RoiConfig(left=1, top=2, width=2, height=2)

    profile = load_native_roi_profile(
        ResolvedRun("105", data_path, nexus_path),
        roi=roi,
        use_experimental_uncertainties=False,
        manual_tof_bin_size_ns=None,
    )

    expected = np.sum(frames[0].astype(float).T[2:4, 1:3])
    np.testing.assert_allclose(profile.counts, [expected])


def test_frame_preview_uses_distinct_sample_and_ob_rois(tmp_path):
    sample_frames = [
        np.asarray([[10, 20], [30, 40]], dtype=np.uint16),
        np.asarray([[20, 30], [40, 50]], dtype=np.uint16),
    ]
    ob_frames = [
        np.asarray([[1, 2], [5, 6]], dtype=np.uint16),
        np.asarray([[2, 3], [10, 12]], dtype=np.uint16),
    ]
    sample_path, sample_nexus = _write_run(tmp_path, "701", sample_frames)
    ob_path, ob_nexus = _write_run(tmp_path, "702", ob_frames)
    frame = _frame(
        "frame",
        [("701", sample_path, sample_nexus)],
        [("702", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=1, height=1),
        ob_roi=RoiConfig(left=1, top=0, width=1, height=1),
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        cache_dir=str(tmp_path / "cache"),
    )

    profile = MultiFramePreviewEngine(recipe).load_frame(frame)
    np.testing.assert_allclose(profile.sample_counts, [10.0, 20.0])
    np.testing.assert_allclose(profile.ob_counts, [5.0, 10.0])
    np.testing.assert_allclose(profile.transmission, [2.0, 2.0])


def test_frame_editor_bin_preview_reports_counts_and_shows_flux_plot(tmp_path, monkeypatch):
    sample_frames = [np.full((2, 2), 20 + index, dtype=np.uint16) for index in range(6)]
    ob_frames = [np.full((2, 2), 40 + index, dtype=np.uint16) for index in range(6)]
    sample_path, sample_nexus = _write_run(tmp_path, "711", sample_frames)
    ob_path, ob_nexus = _write_run(tmp_path, "712", ob_frames)
    frame = _frame(
        "preview frame",
        [("711", sample_path, sample_nexus)],
        [("712", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
        rebin=RebinConfig(
            mode=RebinMode.linear_tof,
            delta_tof_us=2.0,
            full_bins_only=False,
            snap_to_native_grid=False,
        ),
    )
    editor = FrameEditor(frame, str(tmp_path))
    editor.use_proton_charge.value = False
    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: captured.append(figure))

    editor._preview_bins()

    assert len(captured) == 3, editor.rebin_bin_summary.value
    assert "active output bins: 3" in editor.rebin_bin_summary.value
    assert captured[0].layout.xaxis.type == "log"
    assert [trace.name for trace in captured[1].data] == [
        "sample native ROI flux",
        "OB native ROI flux",
    ]
    assert captured[1].layout.yaxis.title.text == "ROI signal (counts / run)"
    expected_sample_flux = frame.roi.width * frame.roi.height * np.arange(25, 19, -1)
    np.testing.assert_allclose(captured[1].data[0].y, expected_sample_flux)
    assert captured[2].layout.yaxis2.title.text == "Actual TOF span (us)"
    assert editor.preview_bins_button.disabled is False


def test_spectrum_normalization_uses_distinct_dc_rois_and_covariance():
    sample_data = np.asarray([[[10.0, 0.0], [0.0, 0.0]]])
    sample_variance = np.asarray([[[10.0, 0.0], [0.0, 0.0]]])
    dc_data = np.asarray([[[1.0, 2.0], [0.0, 0.0]]])
    dc_variance = dc_data.copy()

    result = perform_spectrum_normalization(
        sample_roi=RoiConfig(left=0, top=0, width=1, height=1),
        ob_roi=RoiConfig(left=1, top=0, width=1, height=1),
        sample_data=sample_data,
        sample_variance=sample_variance,
        ob_data_combined_for_spectrum=np.asarray([5.0]),
        ob_data_combined_variance_for_spectrum=np.asarray([5.0]),
        dc_data_combined=dc_data,
        dc_data_combined_variance=dc_variance,
    )

    np.testing.assert_allclose(result["sample_dc_roi_counts"], [1.0])
    np.testing.assert_allclose(result["ob_dc_roi_counts"], [2.0])
    np.testing.assert_allclose(result["spectrum_normalization"], [3.0])
    np.testing.assert_allclose(result["spectrum_normalization_uncertainty"] ** 2, [74.0 / 9.0])


def test_spectrum_normalization_preserves_legacy_same_roi_dc_uncertainty():
    result = perform_spectrum_normalization(
        roi=RoiConfig(left=0, top=0, width=1, height=1),
        sample_data=np.asarray([[[10.0]]]),
        sample_variance=np.asarray([[[10.0]]]),
        ob_data_combined_for_spectrum=np.asarray([5.0]),
        ob_data_combined_variance_for_spectrum=np.asarray([5.0]),
        dc_data_combined=np.asarray([[[1.0]]]),
        dc_data_combined_variance=np.asarray([[[1.0]]]),
    )

    np.testing.assert_allclose(result["spectrum_normalization"], [2.25])
    np.testing.assert_allclose(
        result["spectrum_normalization_uncertainty"] ** 2,
        [2.3046875],
    )


def test_streamed_experimental_variance_matches_production_roi_result(tmp_path):
    frames = [
        np.asarray([[4, 2, 1], [3, 1, 2], [1, 2, 3]], dtype=np.uint16),
        np.asarray([[2, 1, 3], [1, 2, 1], [3, 1, 2]], dtype=np.uint16),
    ]
    data_path, nexus_path = _write_run(tmp_path, "1001", frames)
    roi = RoiConfig(left=0, top=0, width=2, height=2)
    profile = load_native_roi_profile(
        ResolvedRun("1001", data_path, nexus_path),
        roi=roi,
        use_experimental_uncertainties=True,
        manual_tof_bin_size_ns=None,
    )
    production_stack = np.asarray([frame.astype(float).T for frame in frames])
    expected_pixel_variance = calculate_detector_corrected_variance(
        production_stack,
        shutter_counts=[1_000_000.0],
    )
    expected_roi_variance = np.sum(expected_pixel_variance[:, :2, :2], axis=(1, 2))
    np.testing.assert_allclose(profile.variance, expected_roi_variance)


def test_repeated_runs_are_charge_weighted_and_rebinned_before_division(tmp_path):
    shape = (3, 3)
    s1 = [np.full(shape, value) for value in (1, 9, 2, 6)]
    s2 = [np.full(shape, value) for value in (2, 6, 4, 2)]
    ob = [np.full(shape, value) for value in (1, 3, 2, 2)]
    s1_path, s1_nexus = _write_run(tmp_path, "101", s1, charge_c=2.0)
    s2_path, s2_nexus = _write_run(tmp_path, "102", s2, charge_c=1.0)
    ob_path, ob_nexus = _write_run(tmp_path, "103", ob, charge_c=1.0)
    frame = _frame(
        "frame",
        [("101", s1_path, s1_nexus), ("102", s2_path, s2_nexus)],
        [("103", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=3, height=3),
        RebinConfig(mode=RebinMode.linear_tof, delta_tof_us=2.0, full_bins_only=False),
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        cache_dir=str(tmp_path / "cache"),
    )
    engine = MultiFramePreviewEngine(recipe)
    preview = engine.preview_all()["frame"]

    sample_native = (np.asarray([1, 9, 2, 6]) * 9 + np.asarray([2, 6, 4, 2]) * 9) / 3.0
    ob_native = np.asarray([1, 3, 2, 2]) * 9
    np.testing.assert_allclose(preview.native.sample_counts, sample_native)
    np.testing.assert_allclose(preview.native.ob_counts, ob_native)
    expected = np.asarray(
        [
            np.sum(sample_native[:2]) / np.sum(ob_native[:2]),
            np.sum(sample_native[2:]) / np.sum(ob_native[2:]),
        ]
    )
    np.testing.assert_allclose(preview.transmission, expected)
    sample_variance_native = (np.asarray([1, 9, 2, 6]) * 9 + np.asarray([2, 6, 4, 2]) * 9) / 9.0
    ob_variance_native = ob_native
    sample_variance = np.asarray([np.sum(sample_variance_native[:2]), np.sum(sample_variance_native[2:])])
    ob_variance = np.asarray([np.sum(ob_variance_native[:2]), np.sum(ob_variance_native[2:])])
    sample_rebinned = np.asarray([np.sum(sample_native[:2]), np.sum(sample_native[2:])])
    ob_rebinned = np.asarray([np.sum(ob_native[:2]), np.sum(ob_native[2:])])
    expected_uncertainty = np.sqrt(
        sample_variance / ob_rebinned**2
        + sample_rebinned**2 * ob_variance / ob_rebinned**4
    )
    np.testing.assert_allclose(preview.uncertainty, expected_uncertainty)
    assert not np.isclose(expected[0], np.mean(sample_native[:2] / ob_native[:2]))
    np.testing.assert_array_equal(preview.source_frame_count, [2, 2])
    assert not any(
        "full normalization will process them separately" in item
        for item in preview.native.warnings
    )


def test_measured_background_is_subtracted_before_preview_division(tmp_path):
    shape = (2, 2)
    sample_path, sample_nexus = _write_run(
        tmp_path,
        "301",
        [np.full(shape, value) for value in (10, 20)],
    )
    ob_path, ob_nexus = _write_run(
        tmp_path,
        "302",
        [np.full(shape, value) for value in (20, 40)],
    )
    sample_bg_path, sample_bg_nexus = _write_run(
        tmp_path,
        "303",
        [np.full(shape, value) for value in (2, 4)],
    )
    ob_bg_path, ob_bg_nexus = _write_run(
        tmp_path,
        "304",
        [np.full(shape, value) for value in (4, 8)],
    )
    frame = _frame(
        "frame",
        [("301", sample_path, sample_nexus)],
        [("302", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
    )
    frame = replace(
        frame,
        measured_backgrounds=(
            MeasuredBackgroundConfig(
                key="bragg_edge_cd",
                enabled=True,
                sample_runs=(RunSpec("303", str(sample_bg_path), str(sample_bg_nexus)),),
                ob_runs=(RunSpec("304", str(ob_bg_path), str(ob_bg_nexus)),),
            ),
        ),
    )
    engine = MultiFramePreviewEngine(
        MultiFrameRecipe(working_dir=str(tmp_path), frames=[frame], cache_dir=str(tmp_path / "cache"))
    )
    preview = engine.preview_all()["frame"]
    np.testing.assert_allclose(preview.transmission, [0.5, 0.5])
    assert preview.native.corrections_applied == ["Cd-filter measured background (weight 1)"]


def test_persistent_cache_round_trip(tmp_path):
    frames = [np.ones((3, 3), dtype=np.uint16)]
    data_path, nexus_path = _write_run(tmp_path, "104", frames)
    run = ResolvedRun("104", data_path, nexus_path)
    roi = RoiConfig(left=0, top=0, width=2, height=2)
    cache = RoiProfileCache(tmp_path / "cache")
    key = cache.key(run, roi, False, None)
    profile = load_native_roi_profile(run, roi, False, None)
    cache.put(key, profile)

    second_cache = RoiProfileCache(tmp_path / "cache")
    restored = second_cache.get(key)
    assert restored is not None
    assert restored.cache_hit
    np.testing.assert_array_equal(restored.counts, profile.counts)
    np.testing.assert_array_equal(restored.variance, profile.variance)


def _preview(name, energy, transmission, uncertainty):
    energy = np.asarray(energy, dtype=float)
    transmission = np.asarray(transmission, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    native = NativeFrameProfile(
        name=name,
        tof_s=np.arange(len(energy), dtype=float),
        lambda_a=np.ones(len(energy)),
        energy_eV=energy,
        sample_counts=np.ones(len(energy)),
        sample_variance=np.ones(len(energy)),
        ob_counts=np.ones(len(energy)),
        ob_variance=np.ones(len(energy)),
        transmission=transmission,
        uncertainty=uncertainty,
        sample_total_proton_charge_c=1.0,
        ob_total_proton_charge_c=1.0,
    )
    return RebinnedFramePreview(
        name=name,
        tof_s=native.tof_s,
        lambda_a=native.lambda_a,
        energy_eV=energy,
        sample_counts=native.sample_counts,
        sample_variance=native.sample_variance,
        ob_counts=native.ob_counts,
        ob_variance=native.ob_variance,
        transmission=transmission,
        uncertainty=uncertainty,
        source_frame_count=np.ones(len(energy), dtype=int),
        native=native,
    )


def _flux_preview(name, energy, sample_counts, ob_counts, variance=1.0):
    energy = np.asarray(energy, dtype=float)
    sample_counts = np.asarray(sample_counts, dtype=float)
    ob_counts = np.asarray(ob_counts, dtype=float)
    sample_variance = np.full(len(energy), float(variance))
    ob_variance = np.full(len(energy), float(variance))
    transmission = sample_counts / ob_counts
    uncertainty = np.sqrt(
        sample_variance / ob_counts**2
        + sample_counts**2 * ob_variance / ob_counts**4
    )
    native = NativeFrameProfile(
        name=name,
        tof_s=np.arange(len(energy), dtype=float),
        lambda_a=np.ones(len(energy)),
        energy_eV=energy,
        sample_counts=sample_counts,
        sample_variance=sample_variance,
        ob_counts=ob_counts,
        ob_variance=ob_variance,
        transmission=transmission,
        uncertainty=uncertainty,
        sample_total_proton_charge_c=1.0,
        ob_total_proton_charge_c=1.0,
    )
    return RebinnedFramePreview(
        name=name,
        tof_s=native.tof_s,
        lambda_a=native.lambda_a,
        energy_eV=energy,
        sample_counts=sample_counts,
        sample_variance=sample_variance,
        ob_counts=ob_counts,
        ob_variance=ob_variance,
        transmission=transmission,
        uncertainty=uncertainty,
        source_frame_count=np.ones(len(energy), dtype=int),
        native=native,
    )


def test_draw_plot_splits_transmission_and_each_adjacent_overlap(monkeypatch):
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    ui.show_native.value = False
    names = list(ui.inspect_frames.value)
    ui.frame_scale_widgets[names[1]].value = 2.0
    first_pair = ui.overlap_window_widgets[
        ui._overlap_pair_key(names[0], names[1])
    ]
    first_pair["auto"].value = False
    first_pair["minimum"].value = 0.015
    first_pair["maximum"].value = 0.035
    previews = {
        name: _preview(
            name,
            [0.01, 0.02, 0.03, 0.04],
            np.asarray([0.5, 0.6, 0.7, 0.8]) * (1.0 + 0.01 * index),
            [0.01] * 4,
        )
        for index, name in enumerate(names)
    }
    ui.engine = SimpleNamespace(previews=previews)
    ui.show_prompt_flash_lines.value = True
    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: captured.append(figure))
    ui._draw_plot()

    assert len(captured) == len(names)
    transmission_figure, *overlap_figures = captured
    assert len(transmission_figure.data) == len(names)
    assert transmission_figure.layout.yaxis.title.text == "Transmission"
    np.testing.assert_allclose(
        [
            float(shape.x0)
            for shape in transmission_figure.layout.shapes
            if shape.line.color == "#b2182b"
        ],
        PROMPT_FLASH_ENERGIES_EV,
    )
    np.testing.assert_allclose(
        transmission_figure.layout.xaxis.range,
        [np.log10(0.001), np.log10(30.0)],
    )
    np.testing.assert_allclose(
        transmission_figure.data[1].y,
        previews[names[1]].transmission * 2.0,
    )
    np.testing.assert_allclose(
        transmission_figure.data[1].error_y.array,
        previews[names[1]].uncertainty * 2.0,
    )
    assert len(overlap_figures) == len(names) - 1
    assert all(len(figure.data) == 1 for figure in overlap_figures)
    assert all(figure.layout.yaxis.title.text == "Ratio" for figure in overlap_figures)
    _, unscaled_ratio, _ = overlap_ratio_arrays(previews[names[0]], previews[names[1]])
    np.testing.assert_allclose(overlap_figures[0].data[0].x, [0.02, 0.03])
    np.testing.assert_allclose(overlap_figures[0].data[0].y, unscaled_ratio[1:3] * 2.0)
    assert "multipliers: 4.5 A x2" in overlap_figures[0].layout.title.text


def test_overlap_preview_hides_native_profiles_by_default():
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")

    assert ui.show_native.value is False


def test_output_spectra_preview_applies_frame_ranges_and_export_scales(monkeypatch):
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    names = list(ui.inspect_frames.value)
    previews = {
        name: _preview(
            name,
            [0.01, 0.02, 0.03, 0.04],
            [0.5, 0.6, 0.7, 0.8],
            [0.01] * 4,
        )
        for name in names
    }
    ui.engine = SimpleNamespace(previews=previews)
    ui.export_scaled_spectra.value = True
    ui.frame_scale_widgets[names[0]].value = 2.0
    ui.frame_editors[0].output_energy_min.value = 0.02
    ui.frame_editors[0].output_energy_max.value = 0.04
    ui.frame_editors[0]._add_output_exclusion_range(values=(0.029, 0.031))
    ui.show_prompt_flash_lines.value = True
    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: captured.append(figure))

    ui._preview_output_spectra()

    assert len(captured) == 1
    figure = captured[0]
    assert len(figure.data) == len(names)
    np.testing.assert_allclose(figure.data[0].x, [0.02, 0.04])
    np.testing.assert_allclose(figure.data[0].y, [1.2, 1.6])
    np.testing.assert_allclose(figure.data[0].error_y.array, [0.02, 0.02])
    np.testing.assert_allclose(
        [float(shape.x0) for shape in figure.layout.shapes],
        PROMPT_FLASH_ENERGIES_EV,
    )


def test_auto_scale_chains_from_highest_energy_frame(monkeypatch):
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    names = list(ui.inspect_frames.value)
    amplitudes = {
        "6.3 A": 0.05,
        "4.5 A": 0.10,
        "2.5 A": 0.20,
        "0.3 A": 0.40,
        "resonance": 0.80,
    }
    previews = {
        name: _preview(
            name,
            [0.01, 0.02, 0.03, 0.04],
            [amplitudes[name]] * 4,
            [0.01] * 4,
        )
        for name in names
    }
    ui.engine = SimpleNamespace(previews=previews)
    ui.frame_scale_widgets["resonance"].value = 0.75
    for controls in ui.overlap_window_widgets.values():
        controls["auto"].value = False
        controls["minimum"].value = 0.015
        controls["maximum"].value = 0.035
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)

    ui._auto_scale_from_highest_energy()

    expected = {
        "6.3 A": 12.0,
        "4.5 A": 6.0,
        "2.5 A": 3.0,
        "0.3 A": 1.5,
        "resonance": 0.75,
    }
    for name, scale in expected.items():
        np.testing.assert_allclose(ui.frame_scale_widgets[name].value, scale)
        assert ui.frame_scale_widgets[name].disabled is False
    assert "resonance x0.75 (editable highest-energy anchor" in ui.frame_scale_status.value
    assert "2 overlap points" in ui.frame_scale_status.value


def test_auto_scale_highest_energy_anchor_does_not_depend_on_frame_name(monkeypatch):
    frames = [
        _empty_frame("thermal", DetectorType.tpx1),
        _empty_frame("fast", DetectorType.tpx3),
    ]
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914", frames=frames)
    previews = {
        "thermal": _preview("thermal", [0.01, 0.02, 0.03, 0.04], [0.4] * 4, [0.01] * 4),
        "fast": _preview("fast", [0.02, 0.04, 0.2, 2.0], [0.8] * 4, [0.01] * 4),
    }
    ui.engine = SimpleNamespace(previews=previews)
    ui.frame_scale_widgets["fast"].value = 0.9
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)

    ui._auto_scale_from_highest_energy()

    np.testing.assert_allclose(ui.frame_scale_widgets["thermal"].value, 1.8)
    np.testing.assert_allclose(ui.frame_scale_widgets["fast"].value, 0.9)
    assert ui.frame_scale_widgets["fast"].disabled is False
    assert ui.frame_scale_widgets["thermal"].disabled is False
    assert "fast x0.9 (editable highest-energy anchor" in ui.frame_scale_status.value


def test_hybrid_auto_scale_uses_transmission_then_native_flux_and_draws_flux_plots(
    monkeypatch,
):
    names = ["low", "middle", "next", "high"]
    frames = [_empty_frame(name, DetectorType.tpx1) for name in names]
    ui = MultiFrameNormalizationTof(
        "/SNS/VENUS/IPTS-36914",
        frames=frames,
    )
    energy = [0.01, 0.02, 0.03, 0.04]
    previews = {
        "high": _flux_preview("high", energy, [80.0] * 4, [100.0] * 4),
        "next": _flux_preview("next", energy, [40.0] * 4, [100.0] * 4),
        "middle": _flux_preview("middle", energy, [20.0] * 4, [50.0] * 4),
        "low": _flux_preview("low", energy, [20.0] * 4, [25.0] * 4),
    }
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)
    pair_controls = list(ui.overlap_window_widgets.values())
    for controls in pair_controls[:-1]:
        controls["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    ui.engine = SimpleNamespace(previews=previews)
    for controls in ui.overlap_window_widgets.values():
        controls["auto"].value = False
        controls["minimum"].value = 0.015
        controls["maximum"].value = 0.035
    ui._auto_scale_from_highest_energy()

    expected_transmission = {"high": 1.0, "next": 2.0, "middle": 2.0, "low": 1.0}
    for name, multiplier in expected_transmission.items():
        np.testing.assert_allclose(ui.frame_scale_widgets[name].value, multiplier)
    expected_sample = {
        "high": 1.0,
        "next": 2.0,
        "middle": 4.0,
        "low": 4.0,
    }
    expected_ob = {
        "high": 1.0,
        "next": 1.0,
        "middle": 2.0,
        "low": 4.0,
    }
    for name, multiplier in expected_sample.items():
        np.testing.assert_allclose(
            ui.frame_sample_flux_multipliers[name], multiplier
        )
    for name, multiplier in expected_ob.items():
        np.testing.assert_allclose(ui.frame_ob_flux_multipliers[name], multiplier)
    assert "next x2 from transmission" in ui.frame_scale_status.value
    assert "middle x2 from native fluxes" in ui.frame_scale_status.value
    assert "direct-transmission comparison" in ui.frame_scale_status.value

    captured = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: captured.append(figure))
    ui._draw_plot()

    assert len(captured) == 5
    flux_figure = captured[1]
    assert "Selected frame native sample/open-beam flux" in flux_figure.layout.title.text
    assert len(flux_figure.data) == 8
    assert len(flux_figure.layout.shapes) == 4
    assert flux_figure.layout.yaxis.type == "log"
    assert flux_figure.layout.yaxis2.type == "log"


def test_native_flux_auto_scale_uses_native_window_when_rebinned_window_is_empty(
    monkeypatch,
):
    names = ["low", "high"]
    ui = MultiFrameNormalizationTof(
        "/SNS/VENUS/IPTS-36914",
        frames=[_empty_frame(name, DetectorType.tpx1) for name in names],
    )
    native_energy = [0.02, 0.024, 0.028, 0.032]
    high = _flux_preview("high", native_energy, [80.0] * 4, [100.0] * 4)
    low = _flux_preview("low", native_energy, [40.0] * 4, [100.0] * 4)

    # The manual fit window contains several native bins but no output-bin
    # centers. Native-flux scaling must not depend on rebinned coverage.
    high = replace(
        high,
        tof_s=np.asarray([0.0, 1.0]),
        lambda_a=np.ones(2),
        energy_eV=np.asarray([0.015, 1.0]),
        sample_counts=np.asarray([80.0, 80.0]),
        sample_variance=np.ones(2),
        ob_counts=np.asarray([100.0, 100.0]),
        ob_variance=np.ones(2),
        transmission=np.asarray([0.8, 0.8]),
        uncertainty=np.asarray([0.01, 0.01]),
        source_frame_count=np.ones(2, dtype=int),
    )
    low = replace(
        low,
        tof_s=np.asarray([0.0, 1.0]),
        lambda_a=np.ones(2),
        energy_eV=np.asarray([0.01, 0.5]),
        sample_counts=np.asarray([40.0, 40.0]),
        sample_variance=np.ones(2),
        ob_counts=np.asarray([100.0, 100.0]),
        ob_variance=np.ones(2),
        transmission=np.asarray([0.4, 0.4]),
        uncertainty=np.asarray([0.01, 0.01]),
        source_frame_count=np.ones(2, dtype=int),
    )
    ui.engine = SimpleNamespace(previews={"low": low, "high": high})
    controls = ui.overlap_window_widgets[
        ui._overlap_pair_key("low", "high")
    ]
    controls["auto"].value = False
    controls["minimum"].value = 0.021
    controls["maximum"].value = 0.03
    controls["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)

    ui._auto_scale_from_highest_energy()

    np.testing.assert_allclose(ui.frame_scale_widgets["high"].value, 1.0)
    np.testing.assert_allclose(ui.frame_scale_widgets["low"].value, 2.0)
    assert "low x2 from native fluxes" in ui.frame_scale_status.value
    assert "direct-transmission comparison unavailable on the rebinned grid" in (
        ui.frame_scale_status.value
    )


def test_hybrid_auto_scale_requires_proton_charge_normalization(monkeypatch):
    frames = [
        replace(_empty_frame("low", DetectorType.tpx1), use_proton_charge=False),
        _empty_frame("middle", DetectorType.tpx1),
        _empty_frame("high", DetectorType.tpx1),
    ]
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914", frames=frames)
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)
    ui.overlap_window_widgets[
        ui._overlap_pair_key("low", "middle")
    ]["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    ui.engine = SimpleNamespace(
        previews={
            "low": _flux_preview("low", [1, 2, 3], [1, 1, 1], [2, 2, 2]),
            "middle": _flux_preview(
                "middle", [1, 2, 3], [1.5, 1.5, 1.5], [2, 2, 2]
            ),
            "high": _flux_preview("high", [1, 2, 4], [2, 2, 2], [2, 2, 2]),
        }
    )
    ui._auto_scale_from_highest_energy()

    assert "requires proton-charge normalization" in ui.frame_scale_status.value


def test_pair_scaling_selectors_support_mixed_methods_without_named_resonance(
    monkeypatch,
):
    names = ["low", "middle", "next", "high"]
    ui = MultiFrameNormalizationTof(
        "/SNS/VENUS/IPTS-36914",
        frames=[_empty_frame(name, DetectorType.tpx1) for name in names],
    )
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)
    previews = {
        "high": _flux_preview("high", [0.01, 0.02, 0.03, 0.04], [80] * 4, [100] * 4),
        "next": _flux_preview("next", [0.01, 0.02, 0.03, 0.04], [40] * 4, [100] * 4),
        "middle": _flux_preview("middle", [0.01, 0.02, 0.03, 0.04], [20] * 4, [50] * 4),
        "low": _flux_preview("low", [0.01, 0.02, 0.03, 0.04], [20] * 4, [25] * 4),
    }
    ui.engine = SimpleNamespace(previews=previews)
    for controls in ui.overlap_window_widgets.values():
        controls["auto"].value = False
        controls["minimum"].value = 0.015
        controls["maximum"].value = 0.035
        controls["scaling_method"].value = PAIR_SCALING_TRANSMISSION
    ui.overlap_window_widgets[
        ui._overlap_pair_key("next", "high")
    ]["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    ui.overlap_window_widgets[
        ui._overlap_pair_key("low", "middle")
    ]["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX

    ui._auto_scale_from_highest_energy()

    expected_transmission = {"high": 1.0, "next": 2.0, "middle": 2.0, "low": 1.0}
    for name, multiplier in expected_transmission.items():
        np.testing.assert_allclose(ui.frame_scale_widgets[name].value, multiplier)
    for name, multiplier in {
        "high": 1.0,
        "next": 2.0,
        "middle": 2.0,
        "low": 2.0,
    }.items():
        np.testing.assert_allclose(
            ui.frame_sample_flux_multipliers[name], multiplier
        )
    for name, multiplier in {
        "high": 1.0,
        "next": 1.0,
        "middle": 1.0,
        "low": 2.0,
    }.items():
        np.testing.assert_allclose(ui.frame_ob_flux_multipliers[name], multiplier)
    assert "next x2 from native fluxes" in ui.frame_scale_status.value
    assert "middle x2 from transmission" in ui.frame_scale_status.value
    assert "low x1 from native fluxes" in ui.frame_scale_status.value
    figure = ui._combined_native_flux_scaling_figure(
        names,
        {name: "#123456" for name in names},
    )
    assert figure is not None
    assert len(figure.data) == 8
    assert len(figure.layout.shapes) == 4


def test_hybrid_two_frame_case_remains_transmission_only(monkeypatch):
    frames = [
        replace(_empty_frame("low", DetectorType.tpx1), use_proton_charge=False),
        replace(_empty_frame("high", DetectorType.tpx3), use_proton_charge=False),
    ]
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914", frames=frames)
    monkeypatch.setattr(go.Figure, "show", lambda _figure: None)
    ui.engine = SimpleNamespace(
        previews={
            "low": _flux_preview("low", [1, 2, 3], [1, 1, 1], [2, 2, 2]),
            "high": _flux_preview("high", [1, 2, 4], [2, 2, 2], [2, 2, 2]),
        }
    )

    ui._auto_scale_from_highest_energy()

    np.testing.assert_allclose(ui.frame_scale_widgets["high"].value, 1.0)
    np.testing.assert_allclose(ui.frame_scale_widgets["low"].value, 2.0)
    assert "low x2 from transmission" in ui.frame_scale_status.value
    assert "requires proton-charge normalization" not in ui.frame_scale_status.value
    assert ui._combined_native_flux_scaling_figure(["low", "high"], {}) is None


def test_ui_recipe_captures_scaled_spectrum_export_settings():
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    ui.export_scaled_spectra.value = True
    ui.show_hybrid_flux_plots.value = False
    ui.frame_editors[0].output_energy_min.value = 0.0012
    ui.frame_editors[0].output_energy_max.value = 0.0021
    ui.frame_editors[0]._add_output_exclusion_range(values=(0.0014, 0.0015))
    ui.show_prompt_flash_lines.value = True
    ui.frame_scale_widgets["4.5 A"].value = 1.011
    ui.frame_scale_widgets["2.5 A"].value = 0.966
    ui.frame_scale_widgets["resonance"].value = 0.973
    pair = ui.overlap_window_widgets[
        ui._overlap_pair_key("4.5 A", "2.5 A")
    ]
    pair["auto"].value = False
    pair["minimum"].value = 0.0033
    pair["maximum"].value = 0.0038
    pair["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    ui.frame_sample_flux_multipliers = {"0.3 A": 1.02, "resonance": 0.98}
    ui.frame_ob_flux_multipliers = {"0.3 A": 1.01, "resonance": 1.0}

    recipe = ui.recipe()

    assert recipe.export_scaled_spectra is True
    assert recipe.frame_scaling_mode == FRAME_SCALING_HYBRID_NATIVE_FLUX
    assert recipe.frame_sample_flux_multipliers == {
        "0.3 A": 1.02,
        "resonance": 0.98,
    }
    assert recipe.frame_ob_flux_multipliers == {
        "0.3 A": 1.01,
        "resonance": 1.0,
    }
    assert recipe.show_hybrid_flux_plots is False
    assert recipe.frame_multipliers["4.5 A"] == 1.011
    assert recipe.frame_multipliers["2.5 A"] == 0.966
    assert recipe.frame_multipliers["resonance"] == 0.973
    assert recipe.frames[0].output_energy_min_eV == 0.0012
    assert recipe.frames[0].output_energy_max_eV == 0.0021
    assert recipe.frames[0].output_excluded_energy_ranges_eV == ((0.0014, 0.0015),)
    assert recipe.show_prompt_flash_lines is True
    assert recipe.overlap_windows == (
        OverlapWindowConfig("4.5 A", "2.5 A", 0.0033, 0.0038),
    )
    assert PairScalingConfig(
        "4.5 A",
        "2.5 A",
        PAIR_SCALING_NATIVE_FLUX,
    ) in recipe.pair_scaling_methods


def test_ui_load_recipe_restores_frame_multipliers(tmp_path):
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    ui.export_scaled_spectra.value = True
    ui.show_hybrid_flux_plots.value = False
    ui.frame_scale_widgets["6.3 A"].value = 1.06
    ui.frame_scale_widgets["4.5 A"].value = 1.01
    ui.frame_scale_widgets["2.5 A"].value = 0.966
    ui.frame_scale_widgets["resonance"].value = 0.973
    ui.frame_editors[0].output_energy_min.value = 0.0011
    ui.frame_editors[0].output_energy_max.value = 0.0022
    ui.frame_editors[0]._add_output_exclusion_range(values=(0.0014, 0.0015))
    ui.show_prompt_flash_lines.value = True
    pair = ui.overlap_window_widgets[
        ui._overlap_pair_key("0.3 A", "resonance")
    ]
    pair["auto"].value = False
    pair["minimum"].value = 0.11
    pair["maximum"].value = 0.19
    pair["scaling_method"].value = PAIR_SCALING_NATIVE_FLUX
    ui.frame_sample_flux_multipliers = {"0.3 A": 1.02, "resonance": 0.98}
    ui.frame_ob_flux_multipliers = {"0.3 A": 1.01, "resonance": 1.0}
    recipe_path = ui.recipe().save(tmp_path / "scaled_recipe.json")

    restored = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    restored.recipe_file.value = str(recipe_path)
    restored._load_recipe(None)

    assert restored.export_scaled_spectra.value is True
    assert restored._legacy_frame_scaling_mode() == FRAME_SCALING_HYBRID_NATIVE_FLUX
    assert restored.frame_sample_flux_multipliers == {
        "0.3 A": 1.02,
        "resonance": 0.98,
    }
    assert restored.frame_ob_flux_multipliers == {
        "0.3 A": 1.01,
        "resonance": 1.0,
    }
    assert restored.show_hybrid_flux_plots.value is False
    assert restored.frame_scale_widgets["6.3 A"].value == 1.06
    assert restored.frame_scale_widgets["4.5 A"].value == 1.01
    assert restored.frame_scale_widgets["2.5 A"].value == 0.966
    assert restored.frame_scale_widgets["resonance"].value == 0.973
    assert restored.frame_editors[0].output_energy_min.value == 0.0011
    assert restored.frame_editors[0].output_energy_max.value == 0.0022
    assert restored.frame_editors[0].output_excluded_energy_ranges() == (
        (0.0014, 0.0015),
    )
    assert restored.show_prompt_flash_lines.value is True
    restored_pair = restored.overlap_window_widgets[
        restored._overlap_pair_key("0.3 A", "resonance")
    ]
    assert restored_pair["auto"].value is False
    assert restored_pair["minimum"].value == 0.11
    assert restored_pair["maximum"].value == 0.19
    assert restored_pair["scaling_method"].value == PAIR_SCALING_NATIVE_FLUX


def test_ui_load_migrates_legacy_hybrid_preset_to_pair_methods(tmp_path):
    recipe = MultiFrameRecipe(
        working_dir="/SNS/VENUS/IPTS-36914",
        frames=[
            _empty_frame("low", DetectorType.tpx1),
            _empty_frame("middle", DetectorType.tpx1),
            _empty_frame("high", DetectorType.tpx3),
        ],
        frame_scaling_mode=FRAME_SCALING_HYBRID_NATIVE_FLUX,
        pair_scaling_methods=(),
    )
    recipe_path = recipe.save(tmp_path / "legacy_hybrid_recipe.json")

    restored = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    restored.recipe_file.value = str(recipe_path)
    restored._load_recipe(None)

    assert not hasattr(restored, "frame_scaling_mode")
    assert restored._pair_scaling_method("low", "middle") == PAIR_SCALING_NATIVE_FLUX
    assert restored._pair_scaling_method("middle", "high") == PAIR_SCALING_TRANSMISSION


def test_ui_header_path_browsers_select_directories_and_recipe(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    selectors = []

    class FakeFileSelectorPanel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            selectors.append(self)

        def show(self):
            return None

    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames_ui.FileSelectorPanel",
        FakeFileSelectorPanel,
    )
    ui = MultiFrameNormalizationTof(str(tmp_path))

    ui._browse_output_root(None)
    output_selector = selectors[-1].kwargs
    assert output_selector["type"] == "directory"
    assert output_selector["multiple"] is False
    assert output_selector["newdir_toolbar_button"] is True
    output_selector["next"](str(output))
    assert ui.output_root.value == str(output)

    ui._browse_cache_dir(None)
    cache_selector = selectors[-1].kwargs
    assert cache_selector["type"] == "directory"
    assert cache_selector["newdir_toolbar_button"] is True
    cache_selector["next"](str(cache))
    assert ui.cache_dir.value == str(cache)

    ui._browse_recipe_file(None)
    recipe_selector = selectors[-1].kwargs
    assert recipe_selector["type"] == "file"
    assert recipe_selector["newdir_toolbar_button"] is False
    assert recipe_selector["filters"] == {"JSON recipes": "*.json"}
    assert recipe_selector["default_filter"] == "JSON recipes"
    recipe_selector["next"](str(recipe))
    assert ui.recipe_file.value == str(recipe)


def test_ui_save_recipe_writes_new_file_without_confirmation(tmp_path):
    target = tmp_path / "new_recipe.json"
    ui = MultiFrameNormalizationTof(str(tmp_path))
    ui.recipe_file.value = str(target)

    ui._save_recipe(None)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["working_dir"] == str(tmp_path)
    assert ui._pending_recipe_overwrite is None
    assert ui.recipe_overwrite_box.layout.display == "none"


def test_ui_save_recipe_requires_confirmation_before_overwrite(tmp_path):
    target = tmp_path / "existing_recipe.json"
    target.write_text("do not replace yet", encoding="utf-8")
    ui = MultiFrameNormalizationTof(str(tmp_path))
    ui.recipe_file.value = str(target)

    ui._save_recipe(None)

    assert target.read_text(encoding="utf-8") == "do not replace yet"
    assert ui._pending_recipe_overwrite == target.absolute()
    assert ui.recipe_overwrite_box.layout.display == "flex"
    assert str(target.absolute()) in ui.recipe_overwrite_message.value

    ui._confirm_recipe_overwrite(None)

    assert json.loads(target.read_text(encoding="utf-8"))["working_dir"] == str(tmp_path)
    assert ui._pending_recipe_overwrite is None
    assert ui.recipe_overwrite_box.layout.display == "none"


def test_ui_save_recipe_cancel_preserves_existing_file(tmp_path):
    target = tmp_path / "existing_recipe.json"
    target.write_text("keep this recipe", encoding="utf-8")
    ui = MultiFrameNormalizationTof(str(tmp_path))
    ui.recipe_file.value = str(target)

    ui._save_recipe(None)
    ui._cancel_recipe_overwrite(None)

    assert target.read_text(encoding="utf-8") == "keep this recipe"
    assert ui._pending_recipe_overwrite is None
    assert ui.recipe_overwrite_box.layout.display == "none"


def test_ui_save_recipe_rejects_confirmation_after_path_changes(tmp_path):
    original = tmp_path / "original_recipe.json"
    replacement = tmp_path / "replacement_recipe.json"
    original.write_text("keep original", encoding="utf-8")
    ui = MultiFrameNormalizationTof(str(tmp_path))
    ui.recipe_file.value = str(original)

    ui._save_recipe(None)
    ui.recipe_file.value = str(replacement)
    ui._confirm_recipe_overwrite(None)

    assert original.read_text(encoding="utf-8") == "keep original"
    assert not replacement.exists()
    assert ui._pending_recipe_overwrite is None
    assert ui.recipe_overwrite_box.layout.display == "none"


def test_ui_load_recipe_does_not_require_overwrite_confirmation(tmp_path):
    source = MultiFrameNormalizationTof(str(tmp_path))
    recipe_path = source.recipe().save(tmp_path / "recipe.json")
    ui = MultiFrameNormalizationTof(str(tmp_path))
    ui.recipe_file.value = str(recipe_path)
    ui._pending_recipe_overwrite = recipe_path.absolute()
    ui.recipe_overwrite_box.layout.display = "flex"

    ui._load_recipe(None)

    assert ui._pending_recipe_overwrite is None
    assert ui.recipe_overwrite_box.layout.display == "none"


def test_overlap_diagnostic_reports_scale_without_applying_it():
    reference = _preview("reference", [1, 2, 3, 4], [0.5, 0.6, 0.7, 0.8], [0.01] * 4)
    comparison = _preview("comparison", [1.5, 2.5, 3.5], [0.605, 0.715, 0.825], [0.01] * 3)
    result = calculate_overlap_diagnostics(reference, comparison)
    np.testing.assert_allclose(result.comparison_over_reference, 1.1, rtol=1e-12)
    np.testing.assert_allclose(result.scale_comparison_to_reference, 1 / 1.1, rtol=1e-12)
    assert result.point_count == 3


def test_native_flux_overlap_fits_sample_and_ob_before_taking_scale_ratio():
    reference = _flux_preview(
        "reference",
        [1.0, 2.0, 3.0, 4.0],
        [100.0, 200.0, 300.0, 400.0],
        [200.0, 400.0, 600.0, 800.0],
    )
    comparison = _flux_preview(
        "comparison",
        [1.5, 2.5, 3.5],
        [75.0, 125.0, 175.0],
        [75.0, 125.0, 175.0],
    )
    # Deliberately make the rebinned views unrelated. The native-flux
    # estimator must use preview.native, not these output-bin arrays.
    reference = replace(
        reference,
        energy_eV=np.asarray([100.0, 200.0]),
        transmission=np.asarray([9.0, 9.0]),
        uncertainty=np.asarray([1.0, 1.0]),
    )
    comparison = replace(
        comparison,
        energy_eV=np.asarray([150.0, 250.0]),
        transmission=np.asarray([3.0, 3.0]),
        uncertainty=np.asarray([1.0, 1.0]),
    )

    result = calculate_native_flux_overlap_diagnostics(reference, comparison)

    assert isinstance(result, NativeFluxOverlapDiagnostics)
    np.testing.assert_allclose(
        result.sample.scale_comparison_to_reference,
        2.0,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.ob.scale_comparison_to_reference,
        4.0,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.transmission_scale_comparison_to_reference,
        0.5,
        rtol=1e-12,
    )
    assert result.sample.point_count == 3
    assert result.ob.point_count == 3

    arrays = native_flux_overlap_arrays(
        reference,
        comparison,
        "sample",
        reference_scale=1.0,
        comparison_scale=2.0,
    )
    energy, reference_flux, _, raw_comparison, _, scaled_comparison, _ = arrays
    np.testing.assert_allclose(energy, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(reference_flux, [150.0, 250.0, 350.0])
    np.testing.assert_allclose(raw_comparison, [75.0, 125.0, 175.0])
    np.testing.assert_allclose(scaled_comparison, reference_flux)


def test_native_flux_overlap_uses_robust_equal_native_bin_center():
    energy = [1.0, 2.0, 3.0, 4.0, 5.0]
    reference = _flux_preview(
        "reference",
        energy,
        [100.0] * 5,
        [100.0] * 5,
        variance=1.0,
    )
    comparison = _flux_preview(
        "comparison",
        energy,
        [200.0, 200.0, 200.0, 200.0, 2000.0],
        [100.0] * 5,
        variance=1.0,
    )

    result = calculate_native_flux_overlap_diagnostics(reference, comparison)

    # The high-count final point is a frame-shape outlier. A Poisson-weighted
    # fit would let it dominate; the equal-native-bin median recovers the central
    # multiplicative mismatch across the overlap.
    np.testing.assert_allclose(
        result.sample.scale_comparison_to_reference,
        0.5,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.ob.scale_comparison_to_reference,
        1.0,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.transmission_scale_comparison_to_reference,
        0.5,
        rtol=1e-12,
    )
    assert result.sample.reduced_chi_square > 1.0


def test_full_normalization_uses_separate_campaign_folders_and_snapshot(tmp_path, monkeypatch):
    frame_data = [np.ones((2, 2), dtype=np.uint16)]
    sample_path, sample_nexus = _write_run(tmp_path, "105", frame_data)
    ob_path, ob_nexus = _write_run(tmp_path, "106", frame_data)
    frame = replace(_frame(
        "6.3 A",
        [("105", sample_path, sample_nexus)],
        [("106", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
        rebin=RebinConfig(mode=RebinMode.none),
        ob_roi=RoiConfig(left=1, top=0, width=1, height=2),
    ), spectrum_only=False)
    output_root = tmp_path / "output"
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(output_root),
        cache_dir=str(tmp_path / "cache"),
    )
    calls = []

    def fake_normalize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames.normalization_with_list_of_full_path",
        fake_normalize,
    )
    campaign = MultiFramePreviewEngine(recipe).run_full_normalization("test campaign")
    assert campaign.parent == output_root
    assert (campaign / "normalization_tof_multiple_frames_recipe.json").exists()
    assert (campaign / "6.3_A").is_dir()
    assert len(calls) == 1
    assert calls[0]["output_folder"] == str(campaign / "6.3_A")
    assert calls[0]["roi"].width == 2
    assert calls[0]["sample_roi"].width == 2
    assert calls[0]["ob_roi"].left == 1
    assert calls[0]["ob_roi"].width == 1
    assert calls[0]["combine_samples"] is True
    assert calls[0]["correct_chips_alignment_flag"] is False
    assert calls[0]["replace_ob_zeros_by_local_median_flag"] is False
    assert calls[0]["kernel_size_for_local_median"] == (3, 3, 1)
    assert calls[0]["max_iterations"] == 2
    assert calls[0]["export_mode"]["normalized_integrated"] is False


def test_full_normalization_forwards_per_frame_corrections(tmp_path, monkeypatch):
    frame_data = [np.ones((2, 2), dtype=np.uint16)]
    paths = {}
    for run_number in ("401", "402", "403", "404", "405"):
        paths[run_number] = _write_run(tmp_path, run_number, frame_data)

    def spec(run_number):
        data_path, nexus_path = paths[run_number]
        return RunSpec(run_number, str(data_path), str(nexus_path))

    frame = FrameConfig(
        name="frame",
        detector_type=DetectorType.tpx1,
        sample_runs=(spec("401"),),
        ob_runs=(spec("402"),),
        dc_runs=(spec("403"),),
        roi=RoiConfig(left=0, top=0, width=2, height=2),
        container_roi=RoiConfig(left=0, top=0, width=1, height=1),
        combine_sample_runs=False,
        correct_chips_alignment=False,
        replace_ob_zeros_by_local_median=True,
        local_median_kernel=(5, 5, 1),
        local_median_max_iterations=4,
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "output"),
    )
    calls = []
    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames.normalization_with_list_of_full_path",
        lambda **kwargs: calls.append(kwargs),
    )
    MultiFramePreviewEngine(recipe).run_full_normalization("corrections")
    assert len(calls) == 1
    call = calls[0]
    assert list(call["dc_dict"]) == [paths["403"][0].name]
    assert call["container_roi"].width == 1
    assert call["container_roi_file"] is None
    assert call["replace_ob_zeros_by_local_median_flag"] is True
    assert call["kernel_size_for_local_median"] == (5, 5, 1)
    assert call["max_iterations"] == 4


def test_full_normalization_forwards_measured_background_runs(tmp_path, monkeypatch):
    frame_data = [np.ones((2, 2), dtype=np.uint16)]
    paths = {
        run_number: _write_run(tmp_path, run_number, frame_data)
        for run_number in ("501", "502", "503", "504")
    }

    def spec(run_number):
        data_path, nexus_path = paths[run_number]
        return RunSpec(run_number, str(data_path), str(nexus_path))

    frame = FrameConfig(
        name="frame",
        detector_type=DetectorType.tpx1,
        sample_runs=(spec("501"),),
        ob_runs=(spec("502"),),
        roi=RoiConfig(left=0, top=0, width=2, height=2),
        measured_backgrounds=(
            MeasuredBackgroundConfig(
                key="closed_slits",
                enabled=True,
                sample_runs=(spec("503"),),
                ob_runs=(spec("504"),),
                weight=0.75,
            ),
        ),
        spectrum_only=False,
    )
    calls = []
    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames.normalization_with_list_of_full_path",
        lambda **kwargs: calls.append(kwargs),
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "output"),
    )
    MultiFramePreviewEngine(recipe).run_full_normalization("measured background")
    configs = calls[0]["measured_background_correction_configs"]
    assert len(configs) == 1
    assert configs[0]["key_prefix"] == "closed_slits"
    assert configs[0]["weight"] == 0.75
    assert list(configs[0]["sample_background_dict"]) == [paths["503"][0].name]
    assert list(configs[0]["ob_background_dict"]) == [paths["504"][0].name]


def test_full_image_production_retains_full_and_selected_profiles(tmp_path, monkeypatch):
    sample_path, sample_nexus = _write_run(
        tmp_path,
        "601",
        [np.ones((2, 2), dtype=np.uint16)],
    )
    ob_path, ob_nexus = _write_run(
        tmp_path,
        "602",
        [np.ones((2, 2), dtype=np.uint16)],
    )
    frame = _frame(
        "full image",
        [("601", sample_path, sample_nexus)],
        [("602", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
    )
    frame = replace(
        frame,
        spectrum_only=False,
        output_energy_min_eV=0.005,
        output_energy_max_eV=1.1,
        output_excluded_energy_ranges_eV=((0.05, 0.5),),
    )

    def write_profile(**kwargs):
        output = Path(kwargs["output_folder"]) / "spectrum_normalization_profile.txt"
        output.write_text(
            "mean_energy (eV),sample ROI counts,spectrum normalization,"
            "spectrum normalization uncertainty\n"
            "0.01,10,0.5,0.02\n"
            "0.1,20,0.6,0.03\n"
            "1.0,30,0.7,0.04\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames.normalization_with_list_of_full_path",
        write_profile,
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "output"),
        export_scaled_spectra=True,
        frame_multipliers={"full image": 1.5},
    )

    campaign = MultiFramePreviewEngine(recipe).run_full_normalization(
        "full image selected",
        preview=False,
    )
    profile_path = next(campaign.rglob("spectrum_normalization_profile.txt"))
    scaled = pd.read_csv(
        profile_path.with_name("spectrum_normalization_profile_scaled.txt"),
        comment="#",
    )
    selected = pd.read_csv(
        profile_path.with_name("spectrum_normalization_profile_selected.txt"),
        comment="#",
    )
    scaled_selected = pd.read_csv(
        profile_path.with_name("spectrum_normalization_profile_scaled_selected.txt"),
        comment="#",
    )

    np.testing.assert_allclose(
        pd.read_csv(profile_path, comment="#")["mean_energy (eV)"],
        [0.01, 0.1, 1.0],
    )
    np.testing.assert_allclose(scaled["spectrum normalization"], [0.75, 0.9, 1.05])
    np.testing.assert_allclose(selected["mean_energy (eV)"], [0.01, 1.0])
    np.testing.assert_allclose(scaled_selected["spectrum normalization"], [0.75, 1.05])


def test_stage3_spectrum_only_exports_rebinned_and_native_roi_profiles(tmp_path, monkeypatch):
    sample_values = np.asarray([10, 20, 30, 40], dtype=np.float64)
    ob_values = np.asarray([20, 40, 60, 80], dtype=np.float64)
    sample_path, sample_nexus = _write_run(
        tmp_path,
        "701",
        [np.asarray([[value]], dtype=np.uint16) for value in sample_values],
    )
    ob_path, ob_nexus = _write_run(
        tmp_path,
        "702",
        [np.asarray([[value]], dtype=np.uint16) for value in ob_values],
    )
    frame = _frame(
        "resonance",
        [("701", sample_path, sample_nexus)],
        [("702", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=1, height=1),
        rebin=RebinConfig(
            mode=RebinMode.linear_tof,
            delta_tof_us=2.0,
            full_bins_only=False,
        ),
    )
    frame = replace(frame, output_energy_max_eV=1.0e6)
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        export_scaled_spectra=True,
        frame_multipliers={"resonance": 0.8},
    )

    def fail_full_image_engine(**_kwargs):
        raise AssertionError("The full-image production engine was called")

    monkeypatch.setattr(
        "__code.normalization_tof.multiple_frames.normalization_with_list_of_full_path",
        fail_full_image_engine,
    )
    campaign = MultiFramePreviewEngine(recipe).run_full_normalization(
        "stage3",
        preview=False,
    )
    profile_path = next(campaign.rglob("spectrum_normalization_profile.txt"))
    scaled_path = profile_path.with_name("spectrum_normalization_profile_scaled.txt")
    selected_path = profile_path.with_name("spectrum_normalization_profile_selected.txt")
    scaled_selected_path = profile_path.with_name(
        "spectrum_normalization_profile_scaled_selected.txt"
    )
    native_path = profile_path.with_name("native_spectrum_normalization_inputs.txt")
    profile = pd.read_csv(profile_path, comment="#")
    scaled = pd.read_csv(scaled_path, comment="#")
    selected = pd.read_csv(selected_path, comment="#")
    scaled_selected = pd.read_csv(scaled_selected_path, comment="#")
    native = pd.read_csv(native_path, comment="#")

    np.testing.assert_allclose(profile["sample ROI counts"], [30.0, 70.0])
    np.testing.assert_allclose(profile["ob ROI counts"], [60.0, 140.0])
    np.testing.assert_allclose(profile["spectrum normalization"], [0.5, 0.5])
    np.testing.assert_allclose(scaled["spectrum normalization"], [0.4, 0.4])
    np.testing.assert_allclose(scaled["sample ROI counts"], [30.0, 70.0])
    np.testing.assert_allclose(selected["sample ROI counts"], [70.0])
    np.testing.assert_allclose(selected["spectrum normalization"], [0.5])
    np.testing.assert_allclose(scaled_selected["sample ROI counts"], [70.0])
    np.testing.assert_allclose(scaled_selected["spectrum normalization"], [0.4])
    np.testing.assert_allclose(
        scaled["spectrum normalization uncertainty"],
        profile["spectrum normalization uncertainty"] * 0.8,
    )
    np.testing.assert_allclose(native["native sample ROI counts"], sample_values)
    np.testing.assert_allclose(native["native OB ROI counts"], ob_values)
    np.testing.assert_allclose(
        native["native sample ROI uncertainty"],
        np.sqrt(sample_values),
    )
    np.testing.assert_allclose(
        native["native OB ROI uncertainty"],
        np.sqrt(ob_values),
    )
    assert "# native ROI input profile: native_spectrum_normalization_inputs.txt" in (
        profile_path.read_text(encoding="utf-8")
    )
    assert not any(campaign.rglob("stack"))


def test_stage3_measured_background_matches_count_domain_rebin(tmp_path):
    frame_values = {
        "711": (10, 18, 12, 16),
        "712": (20, 30, 24, 32),
        "713": (2, 4, 2, 4),
        "714": (4, 6, 4, 8),
    }
    runs = {}
    for run_number, values in frame_values.items():
        runs[run_number] = _write_run(
            tmp_path,
            run_number,
            [np.asarray([[value]], dtype=np.uint16) for value in values],
        )

    def spec(run_number):
        data_path, nexus_path = runs[run_number]
        return RunSpec(run_number, str(data_path), str(nexus_path))

    frame = FrameConfig(
        name="resonance",
        detector_type=DetectorType.tpx1,
        sample_runs=(spec("711"),),
        ob_runs=(spec("712"),),
        roi=RoiConfig(left=0, top=0, width=1, height=1),
        measured_backgrounds=(
            MeasuredBackgroundConfig(
                key="closed_slits",
                enabled=True,
                sample_runs=(spec("713"),),
                ob_runs=(spec("714"),),
                weight=0.5,
            ),
        ),
        rebin=RebinConfig(
            mode=RebinMode.linear_tof,
            delta_tof_us=2.0,
            full_bins_only=False,
        ),
        use_proton_charge=False,
        use_experimental_uncertainties=False,
    )
    recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "background_output"),
    )
    campaign = MultiFramePreviewEngine(recipe).run_full_normalization(
        "stage3 background",
        preview=False,
    )
    profile_path = next(campaign.rglob("spectrum_normalization_profile.txt"))
    profile = pd.read_csv(profile_path, comment="#")
    native = pd.read_csv(
        profile_path.with_name("native_spectrum_normalization_inputs.txt"),
        comment="#",
    )

    corrected_sample = np.asarray(frame_values["711"], dtype=float) - 0.5 * np.asarray(
        frame_values["713"], dtype=float
    )
    corrected_ob = np.asarray(frame_values["712"], dtype=float) - 0.5 * np.asarray(
        frame_values["714"], dtype=float
    )
    expected_sample = np.asarray([corrected_sample[:2].sum(), corrected_sample[2:].sum()])
    expected_ob = np.asarray([corrected_ob[:2].sum(), corrected_ob[2:].sum()])
    np.testing.assert_allclose(profile["sample ROI counts"], expected_sample)
    np.testing.assert_allclose(profile["ob ROI counts"], expected_ob)
    np.testing.assert_allclose(
        profile["spectrum normalization"],
        expected_sample / expected_ob,
    )
    np.testing.assert_allclose(native["native sample ROI counts"], corrected_sample)
    np.testing.assert_allclose(native["native OB ROI counts"], corrected_ob)
    assert "measured background combined corrected spectrum normalization" in profile.columns


def test_stage3_profile_matches_full_image_production_profile(tmp_path, monkeypatch):
    sample_path, sample_nexus = _write_run(
        tmp_path,
        "721",
        [
            np.asarray([[2, 4], [6, 8]], dtype=np.uint16),
            np.asarray([[3, 5], [7, 9]], dtype=np.uint16),
            np.asarray([[4, 6], [8, 10]], dtype=np.uint16),
            np.asarray([[5, 7], [9, 11]], dtype=np.uint16),
        ],
    )
    ob_path, ob_nexus = _write_run(
        tmp_path,
        "722",
        [
            np.asarray([[4, 8], [12, 16]], dtype=np.uint16),
            np.asarray([[6, 10], [14, 18]], dtype=np.uint16),
            np.asarray([[8, 12], [16, 20]], dtype=np.uint16),
            np.asarray([[10, 14], [18, 22]], dtype=np.uint16),
        ],
    )
    frame = _frame(
        "2.5 A",
        [("721", sample_path, sample_nexus)],
        [("722", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
        rebin=RebinConfig(
            mode=RebinMode.linear_tof,
            delta_tof_us=2.0,
            full_bins_only=False,
        ),
    )
    monkeypatch.setattr(production_normalization, "initialize_logging", lambda: None)

    stage3_recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[frame],
        output_root=str(tmp_path / "stage3_output"),
    )
    full_image_recipe = MultiFrameRecipe(
        working_dir=str(tmp_path),
        frames=[replace(frame, spectrum_only=False)],
        output_root=str(tmp_path / "full_image_output"),
    )
    stage3_campaign = MultiFramePreviewEngine(stage3_recipe).run_full_normalization(
        "stage3 parity",
        preview=False,
    )
    full_image_campaign = MultiFramePreviewEngine(full_image_recipe).run_full_normalization(
        "full image parity",
        preview=False,
    )
    stage3_profile = pd.read_csv(
        next(stage3_campaign.rglob("spectrum_normalization_profile.txt")),
        comment="#",
    )
    full_image_profile = pd.read_csv(
        next(full_image_campaign.rglob("spectrum_normalization_profile.txt")),
        comment="#",
    )

    for column in (
        "sample ROI counts",
        "sample ROI uncertainty",
        "ob ROI counts",
        "ob ROI uncertainty",
        "spectrum normalization",
        "spectrum normalization uncertainty",
    ):
        np.testing.assert_allclose(
            stage3_profile[column],
            full_image_profile[column],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_spectrum_only_controls_apply_to_every_frame():
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    assert all(editor.spectrum_only.value for editor in ui.frame_editors)

    ui._set_all_spectrum_only(False)
    assert all(not editor.spectrum_only.value for editor in ui.frame_editors)

    ui._set_all_spectrum_only(True)
    assert all(editor.spectrum_only.value for editor in ui.frame_editors)


def test_each_frame_delete_button_removes_that_frame_but_keeps_one():
    ui = MultiFrameNormalizationTof("/SNS/VENUS/IPTS-36914")
    initial_names = [editor.name.value for editor in ui.frame_editors]
    target = ui.frame_editors[2]
    target_name = target.name.value

    target.delete_button.click()

    remaining_names = [editor.name.value for editor in ui.frame_editors]
    assert len(remaining_names) == len(initial_names) - 1
    assert target_name not in remaining_names

    while len(ui.frame_editors) > 1:
        ui.frame_editors[0].delete_button.click()

    assert len(ui.frame_editors) == 1
    assert ui.frame_editors[0].delete_button.disabled is True
