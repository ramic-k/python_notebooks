import json
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from __code.normalization_tof import DetectorType, RebinMode
from __code.normalization_tof.multiple_frames import (
    BlackFilterBackgroundConfig,
    FrameConfig,
    MeasuredBackgroundConfig,
    MultiFramePreviewEngine,
    MultiFrameRecipe,
    NativeFrameProfile,
    RebinnedFramePreview,
    RebinConfig,
    ResolvedRun,
    RoiConfig,
    RoiProfileCache,
    RunSpec,
    calculate_overlap_diagnostics,
    load_integrated_image_preview,
    load_native_roi_profile,
    parse_run_numbers,
)
from __code.normalization_tof.multiple_frames_ui import FrameEditor, RunInputEditor, _empty_frame
from __code.normalization_tof.utilities import calculate_detector_corrected_variance


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


def _frame(
    name: str,
    sample: list[tuple[str, Path, Path]],
    ob: list[tuple[str, Path, Path]],
    roi: RoiConfig,
    rebin: RebinConfig | None = None,
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
    )
    recipe = MultiFrameRecipe(
        working_dir="/SNS/VENUS/IPTS-36914",
        frames=[frame],
        output_root=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
    )
    recipe_path = recipe.save(tmp_path / "recipe.json")
    loaded = MultiFrameRecipe.load(recipe_path)
    assert loaded == recipe
    assert json.loads(recipe_path.read_text())["recipe_version"] == 1


def test_new_frame_defaults_match_single_frame_notebook():
    frame = _empty_frame("frame", DetectorType.tpx1)
    recipe = MultiFrameRecipe(working_dir="/SNS/VENUS/IPTS-36914", frames=[frame])
    editor = FrameEditor(frame, recipe.working_dir)

    assert editor.roi_left.value == 156
    assert editor.roi_top.value == 156
    assert editor.roi_width.value == 200
    assert editor.roi_height.value == 200
    assert editor.rebin_mode.value == RebinMode.none
    assert editor.delta_tof.value == 30.0
    assert editor.delta_lambda.value == 0.01
    assert editor.delta_tof_relative.value == 0.01
    assert editor.delta_lambda_relative.value == 0.01
    assert editor.full_bins_only.value is True
    assert editor.snap_to_native.value is True
    assert editor.use_proton_charge.value is True
    assert editor.experimental_uncertainties.value is True
    assert editor.combine_sample_runs.value is False
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

    editor.detector.value = DetectorType.tpx3
    assert editor.experimental_uncertainties.value is False
    assert editor.experimental_uncertainties.disabled is True
    editor.detector.value = DetectorType.tpx1
    assert editor.experimental_uncertainties.value is True
    assert editor.experimental_uncertainties.disabled is False


def test_direct_folder_override_can_infer_run_number(tmp_path):
    run_path = tmp_path / "normalized_Run_19558_images"
    run_path.mkdir()
    editor = RunInputEditor("Sample", (), str(tmp_path))
    editor.folders.value = str(run_path)
    specs = editor.specs()
    assert specs == (RunSpec(run_number="19558", data_path=str(run_path)),)
    assert editor.runs.value == "19558"


def test_integrated_roi_preview_uses_production_orientation_and_subsampling(tmp_path):
    frames = [np.arange(20, dtype=np.uint16).reshape(4, 5) + offset for offset in (0, 20, 40)]
    data_path, _ = _write_run(tmp_path, "200", frames)
    integrated, selected_count, total_count = load_integrated_image_preview(data_path, max_images=2)
    np.testing.assert_array_equal(integrated, frames[0].astype(float).T + frames[2].astype(float).T)
    assert selected_count == 2
    assert total_count == 3


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
    assert any("full normalization will process them separately" in item for item in preview.native.warnings)


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


def test_overlap_diagnostic_reports_scale_without_applying_it():
    reference = _preview("reference", [1, 2, 3, 4], [0.5, 0.6, 0.7, 0.8], [0.01] * 4)
    comparison = _preview("comparison", [1.5, 2.5, 3.5], [0.605, 0.715, 0.825], [0.01] * 3)
    result = calculate_overlap_diagnostics(reference, comparison)
    np.testing.assert_allclose(result.comparison_over_reference, 1.1, rtol=1e-12)
    np.testing.assert_allclose(result.scale_comparison_to_reference, 1 / 1.1, rtol=1e-12)
    assert result.point_count == 3


def test_full_normalization_uses_separate_campaign_folders_and_snapshot(tmp_path, monkeypatch):
    frame_data = [np.ones((2, 2), dtype=np.uint16)]
    sample_path, sample_nexus = _write_run(tmp_path, "105", frame_data)
    ob_path, ob_nexus = _write_run(tmp_path, "106", frame_data)
    frame = _frame(
        "6.3 A",
        [("105", sample_path, sample_nexus)],
        [("106", ob_path, ob_nexus)],
        RoiConfig(left=0, top=0, width=2, height=2),
        RebinConfig(mode=RebinMode.none),
    )
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
    assert calls[0]["combine_samples"] is False
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
