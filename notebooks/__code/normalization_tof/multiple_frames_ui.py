"""Notebook widgets for planning and previewing several normalization frames."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display
from plotly.subplots import make_subplots

from __code.ipywe.fileselector import FileSelectorPanel
from __code.normalization_tof import (
    DetectorType,
    RebinCustomBasis,
    RebinCustomScale,
    RebinMode,
)
from __code.normalization_tof.multiple_frames import (
    AutoRoiConfig,
    BlackFilterBackgroundConfig,
    FRAME_SCALING_HYBRID_NATIVE_FLUX,
    FRAME_SCALING_TRANSMISSION,
    PAIR_SCALING_NATIVE_FLUX,
    PAIR_SCALING_TRANSMISSION,
    FrameConfig,
    MeasuredBackgroundConfig,
    MultiFramePreviewEngine,
    MultiFrameRecipe,
    OverlapWindowConfig,
    PairScalingConfig,
    RebinnedFramePreview,
    RebinConfig,
    RoiConfig,
    RunSpec,
    calculate_overlap_diagnostics,
    calculate_native_flux_overlap_diagnostics,
    convert_tof_schedule_to_energy_schedule,
    load_integrated_image_preview,
    native_flux_overlap_arrays,
    overlap_ratio_arrays,
    parse_run_numbers,
    propose_roi_from_nexus,
)
from __code.normalization_tof.utilities import (
    build_rebin_bin_groups,
    build_rebin_bin_metadata,
)


_REBIN_MODES = [
    RebinMode.none,
    RebinMode.linear_tof,
    RebinMode.linear_lambda,
    RebinMode.log_tof,
    RebinMode.log_lambda,
    RebinMode.inverse_log_lambda,
    RebinMode.custom_schedule,
]

PROMPT_FLASH_ENERGIES_EV = (0.001307, 0.0029404, 0.0117628)


def _frame_energy_log_range(
    *energy_arrays: np.ndarray,
    padding_fraction: float = 0.02,
) -> list[float] | None:
    """Return a padded Plotly log-axis range based only on frame energies."""
    finite_positive = []
    for values in energy_arrays:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite_positive.append(array[np.isfinite(array) & (array > 0)])
    finite_positive = [values for values in finite_positive if values.size]
    if not finite_positive:
        return None

    log_values = np.log10(np.concatenate(finite_positive))
    lower = float(np.min(log_values))
    upper = float(np.max(log_values))
    span = upper - lower
    padding = max(span * float(padding_fraction), 0.005)
    return [lower - padding, upper + padding]


def _add_prompt_flash_lines(
    figure: go.Figure,
    enabled: bool,
    energy_min_eV: float | None = None,
    energy_max_eV: float | None = None,
) -> None:
    """Add the fixed VENUS prompt-flash markers without changing plot coverage."""
    if not enabled:
        return
    for energy_eV in PROMPT_FLASH_ENERGIES_EV:
        if energy_min_eV is not None and energy_eV < energy_min_eV:
            continue
        if energy_max_eV is not None and energy_eV > energy_max_eV:
            continue
        figure.add_vline(
            x=energy_eV,
            line_color="#b2182b",
            line_dash="dash",
            line_width=1.4,
            annotation_text=f"prompt flash {energy_eV:.7g} eV",
            annotation_position="top left",
            annotation_font=dict(color="#8b0a1a", size=10),
        )


_EARLIER_FRAME_TOF_SCHEDULES_US: dict[str, tuple[tuple[float | None, float], ...]] = {
    "6.3 a": ((886.0, 70.0), (4015.0, 100.0), (None, 150.0)),
    "6.5 a": ((886.0, 70.0), (4015.0, 100.0), (None, 150.0)),
    "4.5 a": (
        (1081.0, 100.0),
        (3500.0, 80.0),
        (12174.0, 120.0),
        (None, 100.0),
    ),
    "2.5 a": (
        (313.0, 70.0),
        (866.0, 25.0),
        (13681.0, 120.0),
        (None, 80.0),
    ),
    "0.3 a": (
        (2130.0, 60.0),
        (3380.0, 30.0),
        (14261.0, 60.0),
        (14768.0, 25.0),
        (None, 100.0),
    ),
    "resonance": (
        (560.0, 60.0),
        (2700.0, 60.0),
        (4000.0, 40.0),
        (5500.0, 30.0),
        (None, 100.0),
    ),
}


def _layout(width: str = "220px") -> widgets.Layout:
    return widgets.Layout(width=width)


def _wrapping_row_layout() -> widgets.Layout:
    return widgets.Layout(display="flex", flex_flow="row wrap", align_items="center")


def _show_full_descriptions(controls: list[widgets.Widget]) -> None:
    for control in controls:
        style = getattr(control, "style", None)
        if style is not None and hasattr(style, "description_width"):
            style.description_width = "initial"


def _roi_preview_intensity_limits(values: np.ndarray) -> tuple[int, int, int]:
    """Return a robust default range and the absolute maximum for an ROI preview."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("ROI preview contains no finite intensity values.")
    data_max = max(1, int(np.ceil(np.max(finite))))
    robust_low = max(0, int(np.floor(np.percentile(finite, 1.0))))
    robust_high = min(data_max, max(robust_low + 1, int(np.ceil(np.percentile(finite, 99.9)))))
    return robust_low, robust_high, data_max


class RunInputEditor:
    """Run-number input with an optional direct image-folder override."""

    def __init__(
        self,
        label: str,
        specs: tuple[RunSpec, ...],
        working_dir: str,
        on_change=None,
    ):
        self.label = label
        self.working_dir = working_dir
        self.runs = widgets.Text(
            value=", ".join(run.run_number for run in specs),
            description=f"{label} runs",
            placeholder="19558 or 19419, 19447-19448",
            layout=_layout("620px"),
        )
        direct_paths = [run.data_path for run in specs if run.data_path]
        self.folders = widgets.Textarea(
            value="\n".join(direct_paths),
            description=f"{label} folders",
            placeholder="Optional direct image folder; one folder per run",
            layout=widgets.Layout(width="760px", height="70px"),
        )
        self.browse = widgets.Button(description=f"Browse {label} folders", icon="folder-open")
        self.browser_output = widgets.Output()
        self.browse.on_click(self._show_browser)
        if on_change is not None:
            self.runs.observe(on_change, names="value")
            self.folders.observe(on_change, names="value")
        _show_full_descriptions([self.runs, self.folders])
        self.widget = widgets.VBox(
            [
                self.runs,
                widgets.HBox([self.folders, self.browse]),
                self.browser_output,
            ]
        )

    @property
    def value_widgets(self) -> list[widgets.Widget]:
        return [self.runs, self.folders]

    def specs(self) -> tuple[RunSpec, ...]:
        run_numbers = parse_run_numbers(self.runs.value)
        folders = [line.strip() for line in self.folders.value.splitlines() if line.strip()]
        if not folders:
            return tuple(RunSpec(run_number) for run_number in run_numbers)
        if run_numbers and len(run_numbers) != len(folders):
            raise ValueError(
                f"{self.label}: provide one direct folder for each run number, or clear the run-number field."
            )
        if not run_numbers:
            run_numbers = [_run_number_from_folder(folder) for folder in folders]
            self.runs.value = ", ".join(run_numbers)
        return tuple(
            RunSpec(run_number=run_number, data_path=folder)
            for run_number, folder in zip(run_numbers, folders, strict=True)
        )

    def _show_browser(self, _button) -> None:
        with self.browser_output:
            clear_output(wait=True)
            folders = [line.strip() for line in self.folders.value.splitlines() if line.strip()]
            start_dir = str(Path(folders[0]).parent) if folders else str(Path(self.working_dir) / "shared")
            if not Path(start_dir).is_dir():
                start_dir = self.working_dir
            selector = FileSelectorPanel(
                instruction=f"Select {self.label} image folder(s)",
                start_dir=start_dir,
                type="directory",
                multiple=True,
                next=self._folders_selected,
            )
            selector.show()
            self._selector = selector

    def _folders_selected(self, selected) -> None:
        paths = selected if isinstance(selected, (list, tuple)) else [selected]
        self.folders.value = "\n".join(str(Path(path)) for path in paths)
        with self.browser_output:
            clear_output(wait=True)


class FrameEditor:
    def __init__(
        self,
        config: FrameConfig,
        working_dir: str,
        on_change=None,
        on_delete=None,
        on_move_up=None,
        on_move_down=None,
        show_prompt_flash_lines=None,
    ):
        self.on_change = on_change
        self.working_dir = working_dir
        self.show_prompt_flash_lines = show_prompt_flash_lines
        self.enabled = widgets.Checkbox(value=config.enabled, description="Use frame", indent=False)
        self.name = widgets.Text(value=config.name, description="Name", layout=_layout("360px"))
        self.move_up_button = widgets.Button(
            icon="arrow-up",
            tooltip="Move frame earlier (toward lower energy)",
            layout=_layout("42px"),
        )
        self.move_down_button = widgets.Button(
            icon="arrow-down",
            tooltip="Move frame later (toward higher energy)",
            layout=_layout("42px"),
        )
        self.delete_button = widgets.Button(
            description="Delete frame",
            icon="trash",
            button_style="danger",
            layout=_layout("150px"),
            tooltip="Delete this frame",
        )
        if on_delete is not None:
            self.delete_button.on_click(lambda _button: on_delete(self))
        if on_move_up is not None:
            self.move_up_button.on_click(lambda _button: on_move_up(self))
        if on_move_down is not None:
            self.move_down_button.on_click(lambda _button: on_move_down(self))
        self.detector = widgets.Dropdown(
            options=[DetectorType.tpx1_legacy, DetectorType.tpx1, DetectorType.tpx3],
            value=config.detector_type,
            description="Detector",
            layout=_layout("520px"),
        )
        self.sample_input = RunInputEditor("Sample", config.sample_runs, working_dir)
        self.ob_input = RunInputEditor("OB", config.ob_runs, working_dir)
        self.sample_runs = self.sample_input.runs
        self.ob_runs = self.ob_input.runs
        self.roi_left = widgets.BoundedIntText(
            value=config.roi.left, min=0, max=10000, description="left", layout=_layout()
        )
        self.roi_top = widgets.BoundedIntText(
            value=config.roi.top, min=0, max=10000, description="top", layout=_layout()
        )
        self.roi_width = widgets.BoundedIntText(
            value=config.roi.width, min=1, max=10000, description="width", layout=_layout()
        )
        self.roi_height = widgets.BoundedIntText(
            value=config.roi.height, min=1, max=10000, description="height", layout=_layout()
        )
        self._auto_roi_applied_sources = (
            {"sample", "ob"} if config.auto_roi.applied else set()
        )
        self.auto_roi_enabled = widgets.Checkbox(
            value=config.auto_roi.enabled,
            description="Auto ROI from NeXus slit gaps",
            indent=False,
            layout=_layout("280px"),
        )
        self.auto_roi_projection_scale = widgets.BoundedFloatText(
            value=config.auto_roi.projection_scale,
            min=0.01,
            max=100.0,
            step=0.01,
            description="Projection scale",
            layout=_layout("230px"),
        )
        self.auto_roi_edge_inset_mm = widgets.BoundedFloatText(
            value=config.auto_roi.edge_inset_mm,
            min=0.0,
            max=100.0,
            step=0.05,
            description="Edge inset (mm)",
            layout=_layout("230px"),
        )
        self.auto_roi_refine_center = widgets.Checkbox(
            value=config.auto_roi.refine_center_from_image,
            description="Refine center from image",
            indent=False,
            layout=_layout("240px"),
        )
        self.auto_roi_max_center_shift = widgets.BoundedFloatText(
            value=config.auto_roi.max_center_shift_pixels,
            min=0.0,
            max=512.0,
            step=1.0,
            description="Max center shift (px)",
            layout=_layout("260px"),
        )
        self.auto_roi_help = widgets.HTML(
            "<span style='font-size:12px'>Uses the 512 x 512 detector at 0.055 mm/pixel. "
            "The projected slit opening is inset on every edge; opening an ROI preview "
            "applies the proposal before showing the editable rectangle.</span>"
        )
        ob_roi = config.effective_ob_roi()
        self.ob_roi_linked = widgets.Checkbox(
            value=config.ob_roi is None,
            description="Use OB ROI for sample",
            indent=False,
            layout=_layout("240px"),
        )
        self.ob_roi_left = widgets.BoundedIntText(
            value=ob_roi.left, min=0, max=10000, description="left", layout=_layout()
        )
        self.ob_roi_top = widgets.BoundedIntText(
            value=ob_roi.top, min=0, max=10000, description="top", layout=_layout()
        )
        self.ob_roi_width = widgets.BoundedIntText(
            value=ob_roi.width, min=1, max=10000, description="width", layout=_layout()
        )
        self.ob_roi_height = widgets.BoundedIntText(
            value=ob_roi.height, min=1, max=10000, description="height", layout=_layout()
        )
        self.roi_preview_images = widgets.BoundedIntText(
            value=200,
            min=1,
            max=5000,
            description="Preview images",
            layout=_layout("250px"),
        )
        self.roi_preview_button = widgets.Button(description="Preview/select sample ROI", icon="crop")
        self.roi_preview_button.on_click(self._show_sample_roi_selector)
        self.ob_roi_preview_button = widgets.Button(description="Preview/select OB ROI", icon="crop")
        self.ob_roi_preview_button.on_click(self._show_ob_roi_selector)
        self.roi_preview_output = widgets.Output()
        container_roi = config.container_roi or RoiConfig(left=150, top=150, width=40, height=40)
        self.container_enabled = widgets.Checkbox(
            value=config.container_roi is not None or bool(config.container_roi_file),
            description="Remove container signal",
            indent=False,
            layout=_layout("240px"),
        )
        self.container_mode = widgets.Dropdown(
            options=["Container ROI values", "Saved container ROI file"],
            value=("Saved container ROI file" if config.container_roi_file else "Container ROI values"),
            description="Source",
            layout=_layout("360px"),
        )
        self.container_left = widgets.BoundedIntText(
            value=container_roi.left, min=0, max=10000, description="left", layout=_layout()
        )
        self.container_top = widgets.BoundedIntText(
            value=container_roi.top, min=0, max=10000, description="top", layout=_layout()
        )
        self.container_width = widgets.BoundedIntText(
            value=container_roi.width, min=1, max=10000, description="width", layout=_layout()
        )
        self.container_height = widgets.BoundedIntText(
            value=container_roi.height, min=1, max=10000, description="height", layout=_layout()
        )
        self.container_file = widgets.Text(
            value=config.container_roi_file or "",
            description="ROI file",
            layout=_layout("760px"),
        )
        self.container_file_browse = widgets.Button(description="Browse ROI file", icon="folder-open")
        self.container_file_browse.on_click(self._show_container_file_browser)
        self.container_file_browser_output = widgets.Output()
        self.distance = widgets.BoundedFloatText(
            value=config.distance_source_detector_m,
            min=0.001,
            description="Flight path (m)",
            layout=_layout("260px"),
        )
        self.detector_delay = widgets.FloatText(
            value=0.0 if config.detector_delay_us is None else config.detector_delay_us,
            description="Delay (us)",
            layout=_layout("230px"),
        )
        self.auto_detector_delay = widgets.Checkbox(
            value=config.detector_delay_us is None,
            description="Read delay from NeXus",
            indent=False,
            layout=_layout("220px"),
        )
        self.manual_tof = widgets.FloatText(
            value=0.0 if config.manual_tof_bin_size_ns is None else config.manual_tof_bin_size_ns,
            description="Manual TOF bin (ns)",
            layout=_layout("280px"),
        )
        self.output_energy_min = widgets.FloatText(
            value=0.0 if config.output_energy_min_eV is None else config.output_energy_min_eV,
            description="min energy (eV)",
            layout=_layout("250px"),
        )
        self.output_energy_max = widgets.FloatText(
            value=0.0 if config.output_energy_max_eV is None else config.output_energy_max_eV,
            description="max energy (eV)",
            layout=_layout("250px"),
        )
        self.output_exclusion_rows: list[dict[str, widgets.Widget]] = []
        self.output_exclusion_box = widgets.VBox()
        self.add_output_exclusion_button = widgets.Button(
            description="Add excluded range",
            icon="plus",
            tooltip="Exclude another inclusive energy interval from this frame's selected output",
            layout=_layout("210px"),
        )
        self.add_output_exclusion_button.on_click(self._add_output_exclusion_range)
        for excluded_range in config.output_excluded_energy_ranges_eV:
            self._add_output_exclusion_range(values=excluded_range)
        self.use_proton_charge = widgets.Checkbox(
            value=config.use_proton_charge,
            description="Normalize by proton charge",
            indent=False,
            layout=_layout("250px"),
        )
        self.experimental_uncertainties = widgets.Checkbox(
            value=(
                config.use_experimental_uncertainties
                if config.detector_type != DetectorType.tpx3
                else False
            ),
            description="Experimental uncertainty model",
            indent=False,
            layout=_layout("280px"),
        )
        self.spectrum_only = widgets.Checkbox(
            value=config.spectrum_only,
            description="Spectrum-only production",
            indent=False,
            layout=_layout("240px"),
        )
        self.combine_sample_runs = widgets.Checkbox(
            value=config.combine_sample_runs,
            description="Combine sample runs before normalization",
            indent=False,
            layout=_layout("340px"),
        )
        self.correct_chips_alignment = widgets.Checkbox(
            value=False if config.correct_chips_alignment is None else config.correct_chips_alignment,
            description="Correct chip alignment",
            disabled=True,
            indent=False,
            layout=_layout("220px"),
        )
        self.dc_input = RunInputEditor("Dark current", config.dc_runs, working_dir)
        self.replace_ob_zeros = widgets.Checkbox(
            value=bool(config.replace_ob_zeros_by_local_median),
            description="Replace OB zeros by local median",
            indent=False,
            layout=_layout("290px"),
        )
        median_kernel = config.local_median_kernel or (3, 3, 1)
        self.median_kernel_y = widgets.BoundedIntText(
            value=median_kernel[0], min=1, description="kernel y", layout=_layout()
        )
        self.median_kernel_x = widgets.BoundedIntText(
            value=median_kernel[1], min=1, description="kernel x", layout=_layout()
        )
        self.median_kernel_tof = widgets.BoundedIntText(
            value=median_kernel[2], min=1, description="kernel TOF", layout=_layout()
        )
        self.median_iterations = widgets.BoundedIntText(
            value=config.local_median_max_iterations or 2,
            min=1,
            description="max iterations",
            layout=_layout("250px"),
        )

        black_filter = config.black_filter_background
        self.black_filter_enabled = widgets.Checkbox(
            value=black_filter.enabled,
            description="Enable black-filter background correction",
            indent=False,
            layout=_layout("360px"),
        )
        self.black_filter_shape_file = widgets.Text(
            value=black_filter.shape_file,
            description="Shape CSV",
            layout=_layout("820px"),
        )
        self.black_filter_anchor = widgets.BoundedFloatText(
            value=black_filter.anchor_energy_eV,
            min=0.0,
            description="Anchor (eV)",
            layout=_layout("250px"),
        )

        measured_by_key = {item.key: item for item in config.measured_backgrounds}
        cd_background = measured_by_key.get(
            "bragg_edge_cd",
            MeasuredBackgroundConfig(key="bragg_edge_cd"),
        )
        self.cd_background_enabled = widgets.Checkbox(
            value=cd_background.enabled,
            description="Enable Cd-filter measured background",
            indent=False,
            layout=_layout("330px"),
        )
        self.cd_background_weight = widgets.FloatText(
            value=cd_background.weight,
            description="Weight",
            layout=_layout("200px"),
        )
        self.cd_sample_input = RunInputEditor(
            "Cd sample background",
            cd_background.sample_runs,
            working_dir,
        )
        self.cd_ob_input = RunInputEditor(
            "Cd OB background",
            cd_background.ob_runs,
            working_dir,
        )

        closed_background = measured_by_key.get(
            "closed_slits",
            MeasuredBackgroundConfig(key="closed_slits"),
        )
        self.closed_background_enabled = widgets.Checkbox(
            value=closed_background.enabled,
            description="Enable closed-slits measured background",
            indent=False,
            layout=_layout("350px"),
        )
        self.closed_background_weight = widgets.FloatText(
            value=closed_background.weight,
            description="Weight",
            layout=_layout("200px"),
        )
        self.closed_sample_input = RunInputEditor(
            "Closed-slits sample background",
            closed_background.sample_runs,
            working_dir,
        )
        self.closed_ob_input = RunInputEditor(
            "Closed-slits OB background",
            closed_background.ob_runs,
            working_dir,
        )

        rebin = config.rebin
        self.rebin_mode = widgets.Dropdown(
            options=_REBIN_MODES,
            value=rebin.mode,
            description="Rebin mode",
            layout=_layout("430px"),
        )
        self.delta_tof = widgets.FloatText(value=rebin.delta_tof_us or 30.0, description="delta TOF (us)")
        self.delta_lambda = widgets.FloatText(value=rebin.delta_lambda_a or 0.01, description="delta lambda (A)")
        self.delta_tof_relative = widgets.FloatText(
            value=rebin.delta_tof_over_tof or 0.01, description="delta TOF / TOF"
        )
        self.delta_lambda_relative = widgets.FloatText(
            value=rebin.delta_lambda_over_lambda or 0.01, description="delta lambda / lambda"
        )
        self.delta_lambda_squared = widgets.FloatText(
            value=rebin.delta_lambda_squared_a2 or 0.01, description="delta lambda^2 (A^2)"
        )
        earlier_tof_schedule = _EARLIER_FRAME_TOF_SCHEDULES_US.get(
            config.name.strip().lower()
        )
        if rebin.custom_schedule:
            default_custom_basis = rebin.custom_basis or RebinCustomBasis.energy_tof
            initial_custom_schedule = rebin.custom_schedule
        elif earlier_tof_schedule:
            default_custom_basis = RebinCustomBasis.tof
            initial_custom_schedule = earlier_tof_schedule
        else:
            default_custom_basis = rebin.custom_basis or RebinCustomBasis.energy_tof
            initial_custom_schedule = ()
        self.custom_basis = widgets.Dropdown(
            options=[
                ("Energy edges / TOF widths", RebinCustomBasis.energy_tof),
                ("TOF edges / TOF widths (legacy)", RebinCustomBasis.tof),
                ("Lambda edges / lambda widths (legacy)", RebinCustomBasis.lambda_),
                ("Lambda^2 edges / lambda^2 widths (legacy)", RebinCustomBasis.lambda_squared),
            ],
            value=default_custom_basis,
            description="Custom basis",
            layout=_layout("430px"),
        )
        self.custom_scale = widgets.Dropdown(
            options=[RebinCustomScale.linear, RebinCustomScale.log, RebinCustomScale.reverse_log],
            value=rebin.custom_scale or RebinCustomScale.linear,
            description="Custom scale",
        )
        self.custom_schedule = widgets.Textarea(
            value=_format_schedule(initial_custom_schedule),
            placeholder="0.108, 100\n0.199, 30\n, 60",
            description="Schedule",
            layout=widgets.Layout(width="520px", height="120px"),
        )
        self.custom_schedule_help = widgets.HTML()
        self.full_bins_only = widgets.Checkbox(
            value=rebin.full_bins_only, description="Full bins only", indent=False
        )
        self.snap_to_native = widgets.Checkbox(
            value=rebin.snap_to_native_grid, description="Snap widths to native grid", indent=False
        )
        self.preview_bins_button = widgets.Button(
            description="Preview bins",
            icon="bar-chart",
            button_style="info",
            layout=_layout("180px"),
        )
        self.preview_bins_button.on_click(self._preview_bins)
        self.rebin_bin_summary = widgets.HTML(
            "<span style='font-size:12px'>Preview bins to calculate the active-bin count.</span>"
        )
        self.rebin_preview_output = widgets.Output()

        self.rebin_parameters = widgets.VBox()
        self.rebin_mode.observe(self._update_rebin_parameters, names="value")
        self.custom_basis.observe(self._update_rebin_parameters, names="value")
        self.custom_scale.observe(self._update_rebin_parameters, names="value")
        for rebin_widget in (
            self.rebin_mode,
            self.delta_tof,
            self.delta_lambda,
            self.delta_tof_relative,
            self.delta_lambda_relative,
            self.delta_lambda_squared,
            self.custom_basis,
            self.custom_scale,
            self.custom_schedule,
            self.full_bins_only,
            self.snap_to_native,
        ):
            rebin_widget.observe(self._invalidate_bin_preview, names="value")
        self.auto_detector_delay.observe(self._update_delay_state, names="value")
        self.detector.observe(self._update_detector_state, names="value")
        self.container_enabled.observe(self._update_container_state, names="value")
        self.container_mode.observe(self._update_container_state, names="value")
        self.replace_ob_zeros.observe(self._update_median_state, names="value")
        self.black_filter_enabled.observe(self._update_background_state, names="value")
        self.cd_background_enabled.observe(self._update_background_state, names="value")
        self.closed_background_enabled.observe(self._update_background_state, names="value")
        self.ob_roi_linked.observe(self._update_ob_roi_state, names="value")
        self.auto_roi_enabled.observe(self._update_auto_roi_state, names="value")
        for auto_roi_widget in (
            self.auto_roi_projection_scale,
            self.auto_roi_edge_inset_mm,
            self.auto_roi_refine_center,
            self.auto_roi_max_center_shift,
            self.roi_preview_images,
            self.detector,
            self.ob_roi_linked,
            *self.sample_input.value_widgets,
            *self.ob_input.value_widgets,
        ):
            auto_roi_widget.observe(self._invalidate_auto_roi, names="value")
        for ob_roi_widget in (
            self.ob_roi_left,
            self.ob_roi_top,
            self.ob_roi_width,
            self.ob_roi_height,
        ):
            ob_roi_widget.observe(self._sync_sample_roi_from_ob, names="value")
        self._update_rebin_parameters()
        self._update_delay_state()
        self._update_detector_state()
        self._update_median_state()
        self._update_background_state()
        self._update_ob_roi_state()
        self._update_auto_roi_state()

        all_widgets = self._all_widgets()
        _show_full_descriptions(all_widgets)
        if on_change is not None:
            for widget in all_widgets:
                widget.observe(on_change, names="value")

        self.roi_row = widgets.HBox(
            [self.roi_left, self.roi_top, self.roi_width, self.roi_height],
            layout=_wrapping_row_layout(),
        )
        self.auto_roi_controls = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.auto_roi_enabled,
                        self.auto_roi_projection_scale,
                        self.auto_roi_edge_inset_mm,
                        self.auto_roi_refine_center,
                        self.auto_roi_max_center_shift,
                    ],
                    layout=_wrapping_row_layout(),
                ),
                self.auto_roi_help,
            ]
        )
        self.roi_preview_row = widgets.HBox(
            [self.roi_preview_button],
            layout=_wrapping_row_layout(),
        )
        self.ob_roi_row = widgets.HBox(
            [self.ob_roi_left, self.ob_roi_top, self.ob_roi_width, self.ob_roi_height],
            layout=_wrapping_row_layout(),
        )
        self.ob_roi_preview_row = widgets.HBox(
            [self.roi_preview_images, self.ob_roi_preview_button],
            layout=_wrapping_row_layout(),
        )
        self.axis_row = widgets.HBox(
            [self.distance, self.detector_delay, self.auto_detector_delay, self.manual_tof],
            layout=_wrapping_row_layout(),
        )
        flags_row = widgets.HBox(
            [
                self.use_proton_charge,
                self.experimental_uncertainties,
                self.spectrum_only,
                self.combine_sample_runs,
                self.correct_chips_alignment,
            ],
            layout=_wrapping_row_layout(),
        )
        self.rebin_flags_row = widgets.HBox(
            [self.full_bins_only, self.snap_to_native],
            layout=_wrapping_row_layout(),
        )
        median_box = widgets.VBox(
            [
                self.replace_ob_zeros,
                widgets.HBox(
                    [
                        self.median_kernel_y,
                        self.median_kernel_x,
                        self.median_kernel_tof,
                        self.median_iterations,
                    ],
                    layout=_wrapping_row_layout(),
                ),
            ]
        )
        self.container_roi_box = widgets.HBox(
            [
                self.container_left,
                self.container_top,
                self.container_width,
                self.container_height,
            ],
            layout=_wrapping_row_layout(),
        )
        self.container_file_box = widgets.VBox(
            [
                widgets.HBox(
                    [self.container_file, self.container_file_browse],
                    layout=_wrapping_row_layout(),
                ),
                self.container_file_browser_output,
            ]
        )
        self._update_container_state()
        container_box = widgets.VBox(
            [
                self.container_enabled,
                self.container_mode,
                self.container_roi_box,
                self.container_file_box,
            ]
        )
        black_filter_box = widgets.VBox(
            [
                self.black_filter_enabled,
                self.black_filter_shape_file,
                self.black_filter_anchor,
            ]
        )
        self.cd_background_box = widgets.VBox(
            [
                widgets.HBox(
                    [self.cd_background_enabled, self.cd_background_weight],
                    layout=_wrapping_row_layout(),
                ),
                self.cd_sample_input.widget,
                self.cd_ob_input.widget,
            ]
        )
        self.closed_background_box = widgets.VBox(
            [
                widgets.HBox(
                    [self.closed_background_enabled, self.closed_background_weight],
                    layout=_wrapping_row_layout(),
                ),
                self.closed_sample_input.widget,
                self.closed_ob_input.widget,
            ]
        )
        corrections = widgets.Accordion(
            children=[
                widgets.VBox([flags_row, self.dc_input.widget]),
                widgets.VBox([black_filter_box, self.cd_background_box, self.closed_background_box]),
                median_box,
                container_box,
            ],
            selected_index=None,
        )
        corrections.set_title(0, "Normalization and dark current")
        corrections.set_title(1, "Background corrections")
        corrections.set_title(2, "OB zero handling")
        corrections.set_title(3, "Container correction")
        self.widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.enabled,
                        self.name,
                        self.move_up_button,
                        self.move_down_button,
                        self.delete_button,
                    ],
                    layout=_wrapping_row_layout(),
                ),
                self.detector,
                self.sample_input.widget,
                self.ob_input.widget,
                self.auto_roi_controls,
                widgets.HTML("<b>Open-beam ROI</b>"),
                self.ob_roi_row,
                self.ob_roi_preview_row,
                widgets.HTML("<b>Sample ROI</b>"),
                self.ob_roi_linked,
                self.roi_row,
                self.roi_preview_row,
                self.roi_preview_output,
                self.axis_row,
                widgets.HTML("<b>Proposed rebinning</b>"),
                self.rebin_mode,
                self.rebin_parameters,
                self.rebin_flags_row,
                widgets.HBox([self.preview_bins_button], layout=_wrapping_row_layout()),
                self.rebin_bin_summary,
                self.rebin_preview_output,
                widgets.HTML("<b>Corrections and normalization options</b>"),
                corrections,
            ],
            layout=widgets.Layout(border="1px solid #bbb", padding="8px", margin="0 0 8px 0"),
        )

    @property
    def roi_value_widgets(self) -> tuple[widgets.Widget, ...]:
        return (
            self.roi_left,
            self.roi_top,
            self.roi_width,
            self.roi_height,
            self.ob_roi_linked,
            self.ob_roi_left,
            self.ob_roi_top,
            self.ob_roi_width,
            self.ob_roi_height,
        )

    def _all_widgets(self) -> list[widgets.Widget]:
        return [
            self.enabled,
            self.name,
            self.detector,
            *self.sample_input.value_widgets,
            *self.ob_input.value_widgets,
            self.roi_left,
            self.roi_top,
            self.roi_width,
            self.roi_height,
            self.auto_roi_enabled,
            self.auto_roi_projection_scale,
            self.auto_roi_edge_inset_mm,
            self.auto_roi_refine_center,
            self.auto_roi_max_center_shift,
            self.ob_roi_linked,
            self.ob_roi_left,
            self.ob_roi_top,
            self.ob_roi_width,
            self.ob_roi_height,
            self.container_enabled,
            self.container_mode,
            self.container_left,
            self.container_top,
            self.container_width,
            self.container_height,
            self.container_file,
            self.distance,
            self.detector_delay,
            self.auto_detector_delay,
            self.manual_tof,
            self.use_proton_charge,
            self.experimental_uncertainties,
            self.spectrum_only,
            self.combine_sample_runs,
            self.correct_chips_alignment,
            *self.dc_input.value_widgets,
            self.replace_ob_zeros,
            self.median_kernel_y,
            self.median_kernel_x,
            self.median_kernel_tof,
            self.median_iterations,
            self.black_filter_enabled,
            self.black_filter_shape_file,
            self.black_filter_anchor,
            self.cd_background_enabled,
            self.cd_background_weight,
            *self.cd_sample_input.value_widgets,
            *self.cd_ob_input.value_widgets,
            self.closed_background_enabled,
            self.closed_background_weight,
            *self.closed_sample_input.value_widgets,
            *self.closed_ob_input.value_widgets,
            self.rebin_mode,
            self.delta_tof,
            self.delta_lambda,
            self.delta_tof_relative,
            self.delta_lambda_relative,
            self.delta_lambda_squared,
            self.custom_basis,
            self.custom_scale,
            self.custom_schedule,
            self.full_bins_only,
            self.snap_to_native,
        ]

    def _add_output_exclusion_range(
        self,
        _button=None,
        values: tuple[float, float] | None = None,
    ) -> None:
        minimum = widgets.FloatText(
            value=0.0 if values is None else float(values[0]),
            description="min (eV)",
            layout=_layout("220px"),
        )
        maximum = widgets.FloatText(
            value=0.0 if values is None else float(values[1]),
            description="max (eV)",
            layout=_layout("220px"),
        )
        remove = widgets.Button(
            icon="trash",
            tooltip="Remove this excluded range",
            layout=_layout("42px"),
        )
        controls: dict[str, widgets.Widget] = {
            "minimum": minimum,
            "maximum": maximum,
            "remove": remove,
        }
        remove.on_click(lambda _clicked, row=controls: self._remove_output_exclusion_range(row))
        self.output_exclusion_rows.append(controls)
        self._refresh_output_exclusion_rows()

    def _remove_output_exclusion_range(self, controls: dict[str, widgets.Widget]) -> None:
        if controls in self.output_exclusion_rows:
            self.output_exclusion_rows.remove(controls)
        self._refresh_output_exclusion_rows()

    def _refresh_output_exclusion_rows(self) -> None:
        rows = []
        for index, controls in enumerate(self.output_exclusion_rows, start=1):
            rows.append(
                widgets.HBox(
                    [
                        widgets.HTML(
                            f"exclude {index}",
                            layout=_layout("180px"),
                        ),
                        controls["minimum"],
                        controls["maximum"],
                        controls["remove"],
                    ],
                    layout=_wrapping_row_layout(),
                )
            )
        self.output_exclusion_box.children = tuple(rows)

    def output_excluded_energy_ranges(self) -> tuple[tuple[float, float], ...]:
        ranges = []
        frame_name = self.name.value.strip() or "frame"
        for index, controls in enumerate(self.output_exclusion_rows, start=1):
            minimum = float(controls["minimum"].value)
            maximum = float(controls["maximum"].value)
            if minimum == 0.0 and maximum == 0.0:
                continue
            if minimum <= 0 or maximum <= 0:
                raise ValueError(
                    f"{frame_name}: excluded range {index} needs both positive endpoints."
                )
            if maximum <= minimum:
                raise ValueError(
                    f"{frame_name}: excluded range {index} maximum must be greater than minimum."
                )
            ranges.append((minimum, maximum))
        return tuple(ranges)

    def _update_delay_state(self, _change=None) -> None:
        self.detector_delay.disabled = self.auto_detector_delay.value

    def _update_detector_state(self, change=None) -> None:
        is_tpx3 = self.detector.value == DetectorType.tpx3
        self.experimental_uncertainties.disabled = is_tpx3
        if is_tpx3:
            self.experimental_uncertainties.value = False
        elif change is not None:
            self.experimental_uncertainties.value = True

    def _update_container_state(self, _change=None) -> None:
        enabled = self.container_enabled.value
        self.container_mode.disabled = not enabled
        use_file = enabled and self.container_mode.value == "Saved container ROI file"
        self.container_roi_box.layout.display = "none" if not enabled or use_file else None
        self.container_file_box.layout.display = None if use_file else "none"

    def _update_median_state(self, _change=None) -> None:
        disabled = not self.replace_ob_zeros.value
        for widget in (
            self.median_kernel_y,
            self.median_kernel_x,
            self.median_kernel_tof,
            self.median_iterations,
        ):
            widget.disabled = disabled

    def _update_background_state(self, _change=None) -> None:
        self.black_filter_shape_file.disabled = not self.black_filter_enabled.value
        self.black_filter_anchor.disabled = not self.black_filter_enabled.value
        self.cd_background_weight.disabled = not self.cd_background_enabled.value
        self.closed_background_weight.disabled = not self.closed_background_enabled.value
        self.cd_sample_input.widget.layout.display = None if self.cd_background_enabled.value else "none"
        self.cd_ob_input.widget.layout.display = None if self.cd_background_enabled.value else "none"
        self.closed_sample_input.widget.layout.display = (
            None if self.closed_background_enabled.value else "none"
        )
        self.closed_ob_input.widget.layout.display = (
            None if self.closed_background_enabled.value else "none"
        )

    def _sync_sample_roi_from_ob(self, _change=None) -> None:
        if not self.ob_roi_linked.value:
            return
        self.roi_left.value = self.ob_roi_left.value
        self.roi_top.value = self.ob_roi_top.value
        self.roi_width.value = self.ob_roi_width.value
        self.roi_height.value = self.ob_roi_height.value

    def _update_ob_roi_state(self, _change=None) -> None:
        if self.auto_roi_enabled.value and self.ob_roi_linked.value:
            self.ob_roi_linked.value = False
        linked = self.ob_roi_linked.value
        if linked:
            self._sync_sample_roi_from_ob()
        self.ob_roi_linked.disabled = self.auto_roi_enabled.value
        for widget in (
            self.roi_left,
            self.roi_top,
            self.roi_width,
            self.roi_height,
        ):
            widget.disabled = linked
        self.roi_preview_button.disabled = linked

    def _invalidate_auto_roi(self, _change=None) -> None:
        self._auto_roi_applied_sources.clear()

    def _update_auto_roi_state(self, _change=None) -> None:
        enabled = self.auto_roi_enabled.value
        if enabled:
            self.ob_roi_linked.value = False
        self.ob_roi_linked.disabled = enabled
        for widget in (
            self.auto_roi_projection_scale,
            self.auto_roi_edge_inset_mm,
            self.auto_roi_refine_center,
            self.auto_roi_max_center_shift,
        ):
            widget.disabled = not enabled
        if not enabled:
            self._auto_roi_applied_sources.clear()

    def _auto_roi_config(self) -> AutoRoiConfig:
        required_sources = {"ob"}
        if not self.ob_roi_linked.value:
            required_sources.add("sample")
        return AutoRoiConfig(
            enabled=self.auto_roi_enabled.value,
            pixel_size_mm=0.055,
            projection_scale=self.auto_roi_projection_scale.value,
            edge_inset_mm=self.auto_roi_edge_inset_mm.value,
            refine_center_from_image=self.auto_roi_refine_center.value,
            max_center_shift_pixels=self.auto_roi_max_center_shift.value,
            applied=required_sources.issubset(self._auto_roi_applied_sources),
        )

    def _apply_auto_roi(self, source: str) -> tuple[Any, Any, np.ndarray, int, int]:
        is_sample = source == "sample"
        run_input = self.sample_input if is_sample else self.ob_input
        source_label = "sample" if is_sample else "OB"
        run_specs = run_input.specs()
        if not run_specs:
            raise ValueError(f"Enter or select at least one {source_label} run first.")
        resolver = MultiFramePreviewEngine(
            MultiFrameRecipe(working_dir=self.working_dir, frames=[])
        )
        run = resolver.resolve_run(run_specs[0], self.detector.value)
        if run.nexus_path is None:
            raise ValueError(
                f"Run {run.run_number} has no resolved NeXus file. Disable automatic ROI "
                "for this frame or provide the matching NeXus path."
            )
        integrated, selected_count, total_count = load_integrated_image_preview(
            run.data_path,
            max_images=self.roi_preview_images.value,
        )
        proposal = propose_roi_from_nexus(
            integrated,
            run.nexus_path,
            config=self._auto_roi_config(),
        )
        if is_sample:
            roi_widgets = (self.roi_left, self.roi_top, self.roi_width, self.roi_height)
            self._auto_roi_applied_sources.add("sample")
        else:
            roi_widgets = (
                self.ob_roi_left,
                self.ob_roi_top,
                self.ob_roi_width,
                self.ob_roi_height,
            )
            self._auto_roi_applied_sources.add("ob")
            if self.ob_roi_linked.value:
                self._auto_roi_applied_sources.add("sample")
        for widget, value in zip(
            roi_widgets,
            (
                proposal.roi.left,
                proposal.roi.top,
                proposal.roi.width,
                proposal.roi.height,
            ),
        ):
            widget.value = value
        return run, proposal, integrated, selected_count, total_count

    def ensure_auto_roi(self) -> list[str]:
        """Resolve any enabled automatic ROI proposals not already accepted in this editor."""
        if not self.auto_roi_enabled.value:
            return []
        sources = ["ob"]
        if not self.ob_roi_linked.value:
            sources.append("sample")
        notes = []
        for source in sources:
            if source in self._auto_roi_applied_sources:
                continue
            run, proposal, _integrated, selected_count, total_count = self._apply_auto_roi(source)
            notes.append(
                f"{self.name.value or 'frame'} {source}: ROI "
                f"({proposal.roi.left}, {proposal.roi.top}, {proposal.roi.width}, "
                f"{proposal.roi.height}) from {proposal.slit_gaps.horizontal_mm:.4g} x "
                f"{proposal.slit_gaps.vertical_mm:.4g} mm slits in run {run.run_number}; "
                f"center ({proposal.center_x_pixels:.1f}, {proposal.center_y_pixels:.1f}); "
                f"integrated {selected_count}/{total_count} TIFFs."
            )
        return notes

    def _update_rebin_parameters(self, _change=None) -> None:
        controls: list[widgets.Widget]
        mode = self.rebin_mode.value
        if mode == RebinMode.linear_tof:
            controls = [self.delta_tof]
        elif mode == RebinMode.linear_lambda:
            controls = [self.delta_lambda]
        elif mode == RebinMode.log_tof:
            controls = [self.delta_tof_relative]
        elif mode == RebinMode.log_lambda:
            controls = [self.delta_lambda_relative]
        elif mode == RebinMode.inverse_log_lambda:
            controls = [self.delta_lambda_squared]
        elif mode == RebinMode.custom_schedule:
            energy_tof = self.custom_basis.value == RebinCustomBasis.energy_tof
            if energy_tof and self.custom_scale.value != RebinCustomScale.linear:
                self.custom_scale.value = RebinCustomScale.linear
            self.custom_scale.disabled = energy_tof
            if energy_tof:
                self.custom_schedule_help.value = (
                    "<span style='font-size:12px'>One row per energy region: "
                    "<code>upper energy edge (eV), TOF width (us)</code>. "
                    "List energy edges in increasing order and leave the final edge blank "
                    "to cover the remaining high-energy data.</span>"
                )
            else:
                self.custom_schedule_help.value = (
                    "<span style='font-size:12px'>Legacy schedule: "
                    "<code>upper axis edge, step on that same axis</code>. "
                    "For a linear TOF schedule, <b>Preview bins</b> automatically converts "
                    "these earlier TOF boundaries to this frame's energy boundaries.</span>"
                )
            controls = [
                self.custom_basis,
                self.custom_scale,
                self.custom_schedule,
                self.custom_schedule_help,
            ]
        else:
            controls = [widgets.HTML("Native TOF bins will be used.")]
            self.custom_scale.disabled = False
        self.rebin_parameters.children = tuple(controls)
        rebin_enabled = mode != RebinMode.none
        self.full_bins_only.disabled = not rebin_enabled
        self.preview_bins_button.disabled = not rebin_enabled
        fixed_width_mode = mode in (RebinMode.linear_tof, RebinMode.linear_lambda)
        custom_fixed_width = (
            mode == RebinMode.custom_schedule and self.custom_scale.value == RebinCustomScale.linear
        )
        self.snap_to_native.disabled = not (fixed_width_mode or custom_fixed_width)

    def _invalidate_bin_preview(self, _change=None) -> None:
        if self.rebin_mode.value == RebinMode.none:
            message = "Native TOF bins will be used; no rebin preview is needed."
        else:
            message = "Settings changed. Preview bins to update the active-bin count and plots."
        self.rebin_bin_summary.value = f"<span style='font-size:12px'>{message}</span>"

    def prepare_rebin_for_native(self, frame: FrameConfig, native) -> tuple[FrameConfig, str | None]:
        """Convert a legacy linear TOF recipe after the frame axis is known."""
        if not (
            frame.rebin.mode == RebinMode.custom_schedule
            and frame.rebin.custom_basis == RebinCustomBasis.tof
            and frame.rebin.custom_scale == RebinCustomScale.linear
        ):
            return frame, None

        converted_schedule = convert_tof_schedule_to_energy_schedule(
            frame.rebin.custom_schedule,
            native.tof_s,
            native.energy_eV,
        )
        finite_edge_count = sum(end_value is not None for end_value, _ in converted_schedule)
        self.custom_scale.value = RebinCustomScale.linear
        self.custom_schedule.value = _format_schedule(converted_schedule)
        self.custom_basis.value = RebinCustomBasis.energy_tof
        converted_frame = self.to_config()
        note = (
            f"{frame.name}: converted {finite_edge_count} earlier TOF boundaries to "
            "frame-specific energy boundaries using the loaded TOF/energy axis."
        )
        return converted_frame, note

    def _preview_bins(self, _button=None) -> None:
        self.preview_bins_button.disabled = True
        with self.rebin_preview_output:
            clear_output(wait=True)
            display(HTML("Loading the sample and OB ROI profiles for this frame..."))
        try:
            auto_roi_notes = self.ensure_auto_roi()
            frame = self.to_config()
            if frame.rebin.mode == RebinMode.none:
                raise ValueError("Select a rebin mode before previewing bins.")

            engine = MultiFramePreviewEngine(
                MultiFrameRecipe(working_dir=self.working_dir, frames=[frame])
            )
            native = engine.load_frame(frame)
            frame, conversion_note = self.prepare_rebin_for_native(frame, native)
            preview = engine.rebin_frame(frame, native=native)
            groups, bin_edges = build_rebin_bin_groups(
                tof_array=native.tof_s,
                lambda_array=native.lambda_a,
                energy_array=native.energy_eV,
                **frame.rebin.engine_kwargs(),
            )
            metadata = build_rebin_bin_metadata(
                tof_array=native.tof_s,
                lambda_array=native.lambda_a,
                energy_array=native.energy_eV,
                bin_groups=groups,
            )
            if len(preview.tof_s) == 0:
                raise ValueError("The current settings produced no active output bins.")

            source_counts = np.asarray(metadata["source_frame_count_array"], dtype=int)
            native_tof_diffs = np.diff(np.asarray(native.tof_s, dtype=np.float64))
            positive_native_tof_diffs = native_tof_diffs[
                np.isfinite(native_tof_diffs) & (native_tof_diffs > 0)
            ]
            native_tof_step_s = (
                float(np.median(positive_native_tof_diffs))
                if len(positive_native_tof_diffs)
                else 0.0
            )
            tof_widths_us = (
                np.asarray(metadata["ending_tof_array"], dtype=np.float64)
                - np.asarray(metadata["starting_tof_array"], dtype=np.float64)
                + native_tof_step_s
            ) * 1e6
            unique_counts, count_frequency = np.unique(source_counts, return_counts=True)
            frame_count_summary = ", ".join(
                f"{int(count)} native frames: {int(frequency)} bins"
                for count, frequency in zip(unique_counts, count_frequency)
            )
            width_summary = (
                f"TOF span min/median/max: {np.min(tof_widths_us):.4g} / "
                f"{np.median(tof_widths_us):.4g} / {np.max(tof_widths_us):.4g} us"
            )
            self.rebin_bin_summary.value = (
                "<span style='font-size:12px; color:#176b36'>"
                + (f"{' '.join(_escape(note) for note in auto_roi_notes)} " if auto_roi_notes else "")
                + (f"{_escape(conversion_note)} " if conversion_note else "")
                + f"Native bins: {len(native.tof_s)}; active output bins: {len(preview.tof_s)}; "
                f"{frame_count_summary}. {width_summary}.</span>"
            )

            transmission_figure = go.Figure()
            native_order = np.argsort(native.energy_eV)
            transmission_figure.add_trace(
                go.Scattergl(
                    x=native.energy_eV[native_order],
                    y=native.transmission[native_order],
                    mode="lines",
                    line=dict(color="#777", width=1),
                    opacity=0.55,
                    name="native transmission",
                )
            )
            preview_order = np.argsort(preview.energy_eV)
            low_energy_edges = np.minimum(
                metadata["starting_energy_array"],
                metadata["ending_energy_array"],
            )
            high_energy_edges = np.maximum(
                metadata["starting_energy_array"],
                metadata["ending_energy_array"],
            )
            if frame.rebin.custom_basis == RebinCustomBasis.energy_tof:
                all_edge_energies = np.interp(
                    np.asarray(bin_edges, dtype=np.float64),
                    native.tof_s,
                    native.energy_eV,
                )
                active_bin_indices = np.asarray(metadata["active_bin_index_array"], dtype=int)
                first_edge_energy = all_edge_energies[active_bin_indices]
                second_edge_energy = all_edge_energies[active_bin_indices + 1]
                low_energy_edges = np.minimum(first_edge_energy, second_edge_energy)
                high_energy_edges = np.maximum(first_edge_energy, second_edge_energy)

            preview_customdata = np.column_stack(
                [
                    low_energy_edges[preview_order],
                    high_energy_edges[preview_order],
                    tof_widths_us[preview_order],
                    source_counts[preview_order],
                ]
            )
            transmission_figure.add_trace(
                go.Scatter(
                    x=preview.energy_eV[preview_order],
                    y=preview.transmission[preview_order],
                    error_y=dict(
                        type="data",
                        array=preview.uncertainty[preview_order],
                        visible=True,
                        thickness=0.8,
                        width=0,
                    ),
                    mode="lines+markers",
                    line=dict(color="#1f77b4", width=1.5),
                    marker=dict(size=5),
                    name="rebinned transmission",
                    customdata=preview_customdata,
                    hovertemplate=(
                        "Mean energy: %{x:.6g} eV<br>"
                        "Transmission: %{y:.6g}<br>"
                        "Energy bin: %{customdata[0]:.6g} to %{customdata[1]:.6g} eV<br>"
                        "TOF span: %{customdata[2]:.6g} us<br>"
                        "Native frames: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
            if len(preview.energy_eV) <= 80:
                for index, (low_edge, high_edge) in enumerate(
                    zip(low_energy_edges, high_energy_edges)
                ):
                    if index % 2 == 0 and low_edge > 0:
                        transmission_figure.add_vrect(
                            x0=float(low_edge),
                            x1=float(high_edge),
                            fillcolor="#4c78a8",
                            opacity=0.07,
                            line_width=0,
                            layer="below",
                        )
            transmission_figure.update_layout(
                title=f"{frame.name}: proposed bins over ROI transmission",
                template="plotly_white",
                height=520,
                margin=dict(l=70, r=30, t=75, b=60),
                hovermode="closest",
            )
            frame_energy_log_range = _frame_energy_log_range(
                native.energy_eV,
                preview.energy_eV,
            )
            transmission_figure.update_xaxes(
                type="log",
                title_text="Incident neutron energy (eV)",
                range=frame_energy_log_range,
            )
            transmission_figure.update_yaxes(title_text="Transmission")
            finite_plot_energy = np.concatenate(
                [
                    np.asarray(native.energy_eV, dtype=np.float64),
                    np.asarray(preview.energy_eV, dtype=np.float64),
                ]
            )
            finite_plot_energy = finite_plot_energy[
                np.isfinite(finite_plot_energy) & (finite_plot_energy > 0)
            ]
            _add_prompt_flash_lines(
                transmission_figure,
                bool(
                    self.show_prompt_flash_lines is not None
                    and self.show_prompt_flash_lines.value
                ),
                energy_min_eV=(float(np.min(finite_plot_energy)) if finite_plot_energy.size else None),
                energy_max_eV=(float(np.max(finite_plot_energy)) if finite_plot_energy.size else None),
            )

            flux_figure = go.Figure()
            flux_ylabel = (
                "ROI signal (counts / C)"
                if frame.use_proton_charge
                else "ROI signal (counts / run)"
            )
            for label, counts, variance, color in (
                (
                    "sample native ROI flux",
                    native.sample_counts,
                    native.sample_variance,
                    "#1f77b4",
                ),
                (
                    "OB native ROI flux",
                    native.ob_counts,
                    native.ob_variance,
                    "#e45756",
                ),
            ):
                uncertainty = np.sqrt(np.maximum(np.asarray(variance), 0.0))
                flux_figure.add_trace(
                    go.Scattergl(
                        x=native.energy_eV[native_order],
                        y=np.asarray(counts)[native_order],
                        mode="lines",
                        line=dict(color=color, width=1.2),
                        name=label,
                        customdata=uncertainty[native_order],
                        hovertemplate=(
                            "Energy: %{x:.6g} eV<br>"
                            "ROI signal: %{y:.6g}<br>"
                            "Uncertainty: %{customdata:.3g}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
            flux_figure.update_layout(
                title=f"{frame.name}: native sample and OB ROI flux before division",
                template="plotly_white",
                height=440,
                margin=dict(l=70, r=30, t=75, b=60),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            flux_figure.update_xaxes(
                type="log",
                title_text="Incident neutron energy (eV)",
                range=frame_energy_log_range,
            )
            flux_figure.update_yaxes(title_text=flux_ylabel)

            construction_figure = go.Figure()
            construction_figure.add_trace(
                go.Bar(
                    x=preview.energy_eV[preview_order],
                    y=source_counts[preview_order],
                    name="native frames per bin",
                    marker_color="#4c78a8",
                    opacity=0.65,
                )
            )
            construction_figure.add_trace(
                go.Scatter(
                    x=preview.energy_eV[preview_order],
                    y=tof_widths_us[preview_order],
                    mode="lines+markers",
                    name="actual TOF span",
                    line=dict(color="#e45756", width=1.5),
                    marker=dict(size=4),
                    yaxis="y2",
                )
            )
            construction_figure.update_layout(
                title=f"{frame.name}: output-bin construction",
                template="plotly_white",
                height=380,
                margin=dict(l=70, r=70, t=70, b=60),
                xaxis=dict(
                    type="log",
                    title="Incident neutron energy (eV)",
                    range=frame_energy_log_range,
                ),
                yaxis=dict(title="Native frames per output bin"),
                yaxis2=dict(title="Actual TOF span (us)", overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                hovermode="closest",
            )

            with self.rebin_preview_output:
                clear_output(wait=True)
                transmission_figure.show()
                flux_figure.show()
                construction_figure.show()
        except Exception as error:
            self.rebin_bin_summary.value = (
                "<span style='font-size:12px; color:#b00020'>"
                f"Bin preview unavailable: {_escape(error)}</span>"
            )
            with self.rebin_preview_output:
                clear_output(wait=True)
        finally:
            self.preview_bins_button.disabled = self.rebin_mode.value == RebinMode.none

    def _show_sample_roi_selector(self, _button) -> None:
        self._show_roi_selector("sample")

    def _show_ob_roi_selector(self, _button) -> None:
        self._show_roi_selector("ob")

    def _show_roi_selector(self, source: str) -> None:
        is_sample = source == "sample"
        preview_button = self.roi_preview_button if is_sample else self.ob_roi_preview_button
        run_input = self.sample_input if is_sample else self.ob_input
        source_label = "sample" if is_sample else "OB"
        if is_sample:
            roi_widgets = (self.roi_left, self.roi_top, self.roi_width, self.roi_height)
        else:
            roi_widgets = (
                self.ob_roi_left,
                self.ob_roi_top,
                self.ob_roi_width,
                self.ob_roi_height,
            )

        preview_button.disabled = True
        with self.roi_preview_output:
            clear_output(wait=True)
            display(HTML(f"Resolving the first {source_label} run and integrating the ROI preview..."))
        try:
            auto_roi_note = ""
            if self.auto_roi_enabled.value:
                run, proposal, integrated, selected_count, total_count = self._apply_auto_roi(
                    source
                )
                auto_roi_note = (
                    "<br><span style='color:#176b36'><b>Automatic ROI proposal applied:</b> "
                    f"slits {proposal.slit_gaps.horizontal_mm:.4g} x "
                    f"{proposal.slit_gaps.vertical_mm:.4g} mm; detector ROI "
                    f"({proposal.roi.left}, {proposal.roi.top}, {proposal.roi.width}, "
                    f"{proposal.roi.height}); center "
                    f"({proposal.center_x_pixels:.1f}, {proposal.center_y_pixels:.1f}) px. "
                    "Adjust the rectangle below if needed.</span>"
                )
            else:
                run_specs = run_input.specs()
                if not run_specs:
                    raise ValueError(f"Enter or select at least one {source_label} run first.")
                resolver = MultiFramePreviewEngine(
                    MultiFrameRecipe(working_dir=self.working_dir, frames=[])
                )
                run = resolver.resolve_run(run_specs[0], self.detector.value)
                integrated, selected_count, total_count = load_integrated_image_preview(
                    run.data_path,
                    max_images=self.roi_preview_images.value,
                )
            if integrated is None or integrated.ndim != 2:
                raise ValueError(f"The integrated {source_label} preview is not a two-dimensional image.")
            if not np.any(np.isfinite(integrated)):
                raise ValueError(f"The integrated {source_label} preview contains no finite values.")

            image_height, image_width = integrated.shape
            roi_left, roi_top, roi_width, roi_height = roi_widgets
            left = min(max(int(roi_left.value), 0), image_width - 1)
            top = min(max(int(roi_top.value), 0), image_height - 1)
            right = min(max(left + int(roi_width.value), left + 1), image_width)
            bottom = min(max(top + int(roi_height.value), top + 1), image_height)
            intensity_low, intensity_high, intensity_data_max = _roi_preview_intensity_limits(
                integrated
            )

            intensity = widgets.IntRangeSlider(
                value=(intensity_low, intensity_high),
                min=0,
                max=intensity_high,
                step=1,
                description="Intensity",
                continuous_update=False,
                layout=widgets.Layout(width="780px"),
            )
            include_extreme_pixels = widgets.Checkbox(
                value=False,
                description="Include extreme pixels",
                indent=False,
                disabled=intensity_data_max <= intensity_high,
            )
            left_right = widgets.IntRangeSlider(
                value=(left, right),
                min=0,
                max=image_width,
                step=1,
                description="left/right",
                continuous_update=False,
                layout=widgets.Layout(width="780px"),
            )
            top_bottom = widgets.IntRangeSlider(
                value=(top, bottom),
                min=0,
                max=image_height,
                step=1,
                description="top/bottom",
                continuous_update=False,
                layout=widgets.Layout(width="780px"),
            )
            plot_output = widgets.Output()

            def update_roi_plot(_change=None) -> None:
                selected_left, selected_right = (int(value) for value in left_right.value)
                selected_top, selected_bottom = (int(value) for value in top_bottom.value)
                if selected_right <= selected_left or selected_bottom <= selected_top:
                    return
                roi_left.value = selected_left
                roi_top.value = selected_top
                roi_width.value = selected_right - selected_left
                roi_height.value = selected_bottom - selected_top
                figure = go.Figure(
                    go.Heatmap(
                        z=integrated,
                        colorscale="Viridis",
                        zmin=intensity.value[0],
                        zmax=intensity.value[1],
                        colorbar=dict(title="Integrated counts"),
                    )
                )
                figure.add_shape(
                    type="rect",
                    x0=selected_left,
                    y0=selected_top,
                    x1=selected_right,
                    y1=selected_bottom,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )
                figure.update_layout(
                    template="plotly_white",
                    title=f"{self.name.value or 'frame'}: select {source_label} ROI",
                    width=820,
                    height=760,
                    yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
                    margin=dict(l=55, r=40, t=65, b=50),
                )
                with plot_output:
                    clear_output(wait=True)
                    figure.show()

            def update_intensity_extent(change) -> None:
                if change["new"]:
                    intensity.max = intensity_data_max
                    intensity.value = (0, intensity_data_max)
                else:
                    intensity.value = (intensity_low, intensity_high)
                    intensity.max = intensity_high

            for slider in (intensity, left_right, top_bottom):
                slider.observe(update_roi_plot, names="value")
            include_extreme_pixels.observe(update_intensity_extent, names="value")

            with self.roi_preview_output:
                clear_output(wait=True)
                display(
                    HTML(
                        f"Preview source: <code>{_escape(run.data_path)}</code><br>"
                        f"Integrated {selected_count} evenly sampled TIFFs out of {total_count}. "
                        f"The {source_label} ROI fields above update when the sliders are released."
                        + (
                            " The sample ROI is linked to the OB ROI, so this also updates "
                            "the sample ROI."
                            if not is_sample and self.ob_roi_linked.value
                            else ""
                        )
                        + auto_roi_note
                    )
                )
                display(
                    widgets.VBox(
                        [intensity, include_extreme_pixels, left_right, top_bottom, plot_output]
                    )
                )
            update_roi_plot()
        except Exception as error:
            with self.roi_preview_output:
                clear_output(wait=True)
                display(HTML(f"<span style='color:#b00020'><b>ROI preview failed:</b> {_escape(error)}</span>"))
        finally:
            preview_button.disabled = is_sample and self.ob_roi_linked.value

    def _show_container_file_browser(self, _button) -> None:
        with self.container_file_browser_output:
            clear_output(wait=True)
            current = self.container_file.value.strip()
            start_dir = str(Path(current).parent) if current else str(Path(self.working_dir) / "shared")
            if not Path(start_dir).is_dir():
                start_dir = self.working_dir
            selector = FileSelectorPanel(
                instruction="Select a saved container ROI file",
                start_dir=start_dir,
                type="file",
                multiple=False,
                next=self._container_file_selected,
            )
            selector.show()
            self._container_selector = selector

    def _container_file_selected(self, selected) -> None:
        self.container_file.value = str(Path(selected))
        with self.container_file_browser_output:
            clear_output(wait=True)

    def to_config(self) -> FrameConfig:
        sample = self.sample_input.specs()
        ob = self.ob_input.specs()
        dc = self.dc_input.specs()
        mode = self.rebin_mode.value
        rebin = RebinConfig(
            mode=mode,
            delta_tof_us=self.delta_tof.value if mode == RebinMode.linear_tof else None,
            delta_lambda_a=self.delta_lambda.value if mode == RebinMode.linear_lambda else None,
            delta_tof_over_tof=self.delta_tof_relative.value if mode == RebinMode.log_tof else None,
            delta_lambda_over_lambda=(self.delta_lambda_relative.value if mode == RebinMode.log_lambda else None),
            delta_lambda_squared_a2=(
                self.delta_lambda_squared.value if mode == RebinMode.inverse_log_lambda else None
            ),
            custom_basis=self.custom_basis.value if mode == RebinMode.custom_schedule else None,
            custom_scale=self.custom_scale.value if mode == RebinMode.custom_schedule else None,
            custom_schedule=(
                tuple(_parse_schedule(self.custom_schedule.value)) if mode == RebinMode.custom_schedule else ()
            ),
            full_bins_only=self.full_bins_only.value,
            snap_to_native_grid=self.snap_to_native.value,
        )
        output_energy_min_eV = (
            self.output_energy_min.value if self.output_energy_min.value > 0 else None
        )
        output_energy_max_eV = (
            self.output_energy_max.value if self.output_energy_max.value > 0 else None
        )
        if (
            output_energy_min_eV is not None
            and output_energy_max_eV is not None
            and output_energy_max_eV <= output_energy_min_eV
        ):
            raise ValueError(
                f"{self.name.value or 'frame'}: output maximum energy must be greater "
                "than minimum."
            )
        return FrameConfig(
            name=self.name.value.strip(),
            detector_type=self.detector.value,
            sample_runs=sample,
            ob_runs=ob,
            dc_runs=dc,
            roi=RoiConfig(
                left=self.roi_left.value,
                top=self.roi_top.value,
                width=self.roi_width.value,
                height=self.roi_height.value,
            ),
            auto_roi=self._auto_roi_config(),
            ob_roi=(
                None
                if self.ob_roi_linked.value
                else RoiConfig(
                    left=self.ob_roi_left.value,
                    top=self.ob_roi_top.value,
                    width=self.ob_roi_width.value,
                    height=self.ob_roi_height.value,
                )
            ),
            container_roi=(
                RoiConfig(
                    left=self.container_left.value,
                    top=self.container_top.value,
                    width=self.container_width.value,
                    height=self.container_height.value,
                )
                if self.container_enabled.value and self.container_mode.value == "Container ROI values"
                else None
            ),
            container_roi_file=(
                self.container_file.value.strip()
                if self.container_enabled.value and self.container_mode.value == "Saved container ROI file"
                else None
            ),
            rebin=rebin,
            distance_source_detector_m=self.distance.value,
            detector_delay_us=None if self.auto_detector_delay.value else self.detector_delay.value,
            manual_tof_bin_size_ns=self.manual_tof.value if self.manual_tof.value > 0 else None,
            output_energy_min_eV=output_energy_min_eV,
            output_energy_max_eV=output_energy_max_eV,
            output_excluded_energy_ranges_eV=self.output_excluded_energy_ranges(),
            use_proton_charge=self.use_proton_charge.value,
            use_experimental_uncertainties=self.experimental_uncertainties.value,
            spectrum_only=self.spectrum_only.value,
            combine_sample_runs=self.combine_sample_runs.value,
            correct_chips_alignment=self.correct_chips_alignment.value,
            replace_ob_zeros_by_local_median=self.replace_ob_zeros.value,
            local_median_kernel=(
                self.median_kernel_y.value,
                self.median_kernel_x.value,
                self.median_kernel_tof.value,
            ),
            local_median_max_iterations=self.median_iterations.value,
            black_filter_background=BlackFilterBackgroundConfig(
                enabled=self.black_filter_enabled.value,
                shape_file=self.black_filter_shape_file.value.strip(),
                anchor_energy_eV=self.black_filter_anchor.value,
            ),
            measured_backgrounds=(
                MeasuredBackgroundConfig(
                    key="bragg_edge_cd",
                    enabled=self.cd_background_enabled.value,
                    sample_runs=self.cd_sample_input.specs(),
                    ob_runs=self.cd_ob_input.specs(),
                    weight=self.cd_background_weight.value,
                ),
                MeasuredBackgroundConfig(
                    key="closed_slits",
                    enabled=self.closed_background_enabled.value,
                    sample_runs=self.closed_sample_input.specs(),
                    ob_runs=self.closed_ob_input.specs(),
                    weight=self.closed_background_weight.value,
                ),
            ),
            enabled=self.enabled.value,
        )


class MultiFrameNormalizationTof:
    """Interactive pre-normalization planner for multiple wavelength frames."""

    def __init__(
        self,
        working_dir: str,
        frames: list[FrameConfig] | None = None,
        output_root: str | None = None,
        cache_dir: str | None = None,
    ):
        self.working_dir = str(Path(working_dir).expanduser())
        self.output_root = widgets.Text(
            value=output_root or str(Path(self.working_dir) / "shared"),
            description="Output root",
            layout=_layout("760px"),
        )
        self.cache_dir = widgets.Text(
            value=cache_dir or str(Path(self.working_dir) / "shared" / ".normalization_tof_multiple_frames_cache"),
            description="Preview cache",
            layout=_layout("760px"),
        )
        self.recipe_file = widgets.Text(
            value=str(Path(self.working_dir) / "shared" / "normalization_tof_multiple_frames_recipe.json"),
            description="Recipe JSON",
            layout=_layout("760px"),
        )
        self.output_root_browse = widgets.Button(
            description="Browse...",
            icon="folder-open",
            tooltip="Select or create the full-normalization output directory",
            layout=_layout("120px"),
        )
        self.cache_dir_browse = widgets.Button(
            description="Browse...",
            icon="folder-open",
            tooltip="Select or create the preview-cache directory",
            layout=_layout("120px"),
        )
        self.recipe_file_browse = widgets.Button(
            description="Browse...",
            icon="folder-open",
            tooltip="Select an existing multi-frame recipe JSON file",
            layout=_layout("120px"),
        )
        self.path_browser_output = widgets.Output()
        self.show_native = widgets.Checkbox(value=False, description="Show native profiles", indent=False)
        self.show_errors = widgets.Checkbox(value=True, description="Show error bars", indent=False)
        self.show_prompt_flash_lines = widgets.Checkbox(
            value=False,
            description="Show prompt-flash lines",
            indent=False,
            layout=_layout("240px"),
        )
        self.force_reload = widgets.Checkbox(value=False, description="Ignore cached profiles", indent=False)
        self.inspect_frames = widgets.SelectMultiple(
            options=[],
            value=(),
            description="Frames to inspect",
            rows=5,
            layout=widgets.Layout(width="560px", height="130px"),
        )
        self.overlap_window_widgets: dict[tuple[str, str], dict[str, Any]] = {}
        self._overlap_window_state_cache: dict[
            tuple[str, str], tuple[bool, float, float, str]
        ] = {}
        self.overlap_window_box = widgets.VBox()
        self.frame_scaling_mode = widgets.Dropdown(
            options=(
                ("Set all pairs to transmission", FRAME_SCALING_TRANSMISSION),
                (
                    "Preset: highest pair transmission, lower pairs native flux",
                    FRAME_SCALING_HYBRID_NATIVE_FLUX,
                ),
            ),
            value=FRAME_SCALING_TRANSMISSION,
            description="Pair-method preset",
            layout=_layout("680px"),
        )
        self.show_hybrid_flux_plots = widgets.Checkbox(
            value=True,
            description="Show native sample/OB flux scaling plots",
            indent=False,
            disabled=True,
            layout=_layout("380px"),
        )
        self.frame_scale_widgets: dict[str, widgets.BoundedFloatText] = {}
        self.frame_sample_flux_multipliers: dict[str, float] = {}
        self.frame_ob_flux_multipliers: dict[str, float] = {}
        self.frame_scale_box = widgets.HBox(layout=_wrapping_row_layout())
        self.reset_frame_scales_button = widgets.Button(
            description="Reset preview scales",
            icon="undo",
        )
        self.auto_frame_scales_button = widgets.Button(
            description="Auto-scale from highest energy",
            icon="magic",
            button_style="info",
            layout=_layout("340px"),
        )
        self.frame_scale_status = widgets.HTML()
        self.frame_output_range_box = widgets.VBox()
        self.preview_output_spectra_button = widgets.Button(
            description="Preview output spectra",
            icon="line-chart",
            button_style="info",
            layout=_layout("240px"),
        )
        self.output_spectra_plot_output = widgets.Output()
        self.export_scaled_spectra = widgets.Checkbox(
            value=False,
            description="Export scaled spectra after normalization",
            indent=False,
            layout=_layout("360px"),
        )
        self.add_frame_button = widgets.Button(description="Add frame", icon="plus")
        self.remove_frame_button = widgets.Button(description="Remove last", icon="minus")
        self.same_rois_all_frames = widgets.Checkbox(
            value=False,
            description="Use same ROIs for all frames",
            indent=False,
            layout=_layout("280px"),
        )
        self.enable_spectrum_only_all_button = widgets.Button(
            description="Spectrum only for all",
            icon="file-text-o",
            button_style="info",
        )
        self.disable_spectrum_only_all_button = widgets.Button(
            description="Full images for all",
            icon="picture-o",
        )
        self.preview_button = widgets.Button(description="Load ROI profiles and preview", icon="line-chart")
        self.replot_button = widgets.Button(description="Rebin cached profiles", icon="refresh")
        self.save_button = widgets.Button(description="Save recipe", icon="save")
        self.load_button = widgets.Button(description="Load recipe", icon="folder-open")
        self.run_button = widgets.Button(description="Run full normalization", icon="play", button_style="warning")
        self.arm_full_run = widgets.Checkbox(value=False, description="Enable full run", indent=False)
        self.status_output = widgets.Output()
        self.plot_output = widgets.Output()
        self.frame_box = widgets.Accordion()
        self.engine: MultiFramePreviewEngine | None = None
        self.loaded_frame_configs: dict[str, FrameConfig] = {}
        self.frame_editors: list[FrameEditor] = []
        self._syncing_frame_rois = False
        self._updating_frame_scales = False
        self._updating_overlap_windows = False

        _show_full_descriptions(
            [
                self.output_root,
                self.cache_dir,
                self.recipe_file,
                self.inspect_frames,
            ]
        )

        initial_frames = frames if frames is not None else _default_frames()
        self._set_frames(initial_frames)
        self._wire_events()
        self._update_run_button()

    def display(self) -> None:
        header = widgets.VBox(
            [
                widgets.HBox(
                    [self.add_frame_button, self.remove_frame_button],
                    layout=_wrapping_row_layout(),
                ),
                widgets.HBox(
                    [self.output_root, self.output_root_browse],
                    layout=_wrapping_row_layout(),
                ),
                widgets.HBox(
                    [self.cache_dir, self.cache_dir_browse],
                    layout=_wrapping_row_layout(),
                ),
                widgets.HBox(
                    [self.recipe_file, self.recipe_file_browse],
                    layout=_wrapping_row_layout(),
                ),
                self.path_browser_output,
                widgets.HBox([self.save_button, self.load_button]),
            ]
        )
        overlap_controls = widgets.VBox(
            [
                self.inspect_frames,
                widgets.HTML("<b>Overlap windows</b>"),
                widgets.HTML(
                    "<span style='font-size:12px'>Each row applies to one adjacent frame pair. "
                    "Leave <b>Auto</b> checked to use their full common energy coverage, or "
                    "uncheck it and enter the fit limits in eV. Choose the scaling estimator "
                    "independently for every pair.</span>"
                ),
                self.overlap_window_box,
                widgets.HBox(
                    [
                        self.show_native,
                        self.show_errors,
                        self.show_prompt_flash_lines,
                        self.force_reload,
                    ],
                    layout=_wrapping_row_layout(),
                ),
                widgets.HTML("<b>Frame scaling</b>"),
                self.frame_scaling_mode,
                self.show_hybrid_flux_plots,
                widgets.HTML(
                    "<span style='font-size:12px'>The preset only initializes the per-pair "
                    "selectors above. Native-flux pairs fit corrected, proton-charge-normalized "
                    "sample and OB fluxes separately before rebinning; their scale ratio sets "
                    "the transmission multiplier. The anchor is always the frame with the "
                    "highest measured energy coverage, regardless of its name.</span>"
                ),
                widgets.HTML("<b>Frame multipliers</b>"),
                self.frame_scale_box,
                widgets.HBox(
                    [self.auto_frame_scales_button, self.reset_frame_scales_button],
                    layout=_wrapping_row_layout(),
                ),
                self.export_scaled_spectra,
                widgets.HTML(
                    "<span style='font-size:12px'>The original normalized profile is retained. "
                    "When enabled, each frame also exports the full-range "
                    "<code>spectrum_normalization_profile_scaled.txt</code>.</span>"
                ),
                self.frame_scale_status,
                widgets.HBox(
                    [self.preview_button, self.replot_button],
                    layout=_wrapping_row_layout(),
                ),
            ]
        )
        full_run_controls = widgets.HBox(
            [self.arm_full_run, self.run_button],
            layout=_wrapping_row_layout(),
        )
        display(
            widgets.VBox(
                [
                    header,
                    widgets.HTML("<h3>Frame settings</h3>"),
                    widgets.HBox(
                        [
                            self.same_rois_all_frames,
                            self.enable_spectrum_only_all_button,
                            self.disable_spectrum_only_all_button,
                        ],
                        layout=_wrapping_row_layout(),
                    ),
                    self.frame_box,
                    widgets.HTML("<h3>Overlap inspection</h3>"),
                    overlap_controls,
                    self.status_output,
                    self.plot_output,
                    widgets.HTML("<h3>Frame output spectra</h3>"),
                    widgets.HTML(
                        "<span style='font-size:12px'>Set optional inclusive output "
                        "limits for each enabled frame. Use 0 for the native minimum or "
                        "maximum. Full-range normalized and scaled profiles are retained; "
                        "active limits add <code>..._selected.txt</code> companions.</span>"
                    ),
                    self.frame_output_range_box,
                    self.preview_output_spectra_button,
                    self.output_spectra_plot_output,
                    widgets.HTML("<h3>Full normalization</h3>"),
                    full_run_controls,
                ]
            )
        )

    def recipe(self) -> MultiFrameRecipe:
        frames = [editor.to_config() for editor in self.frame_editors]
        names = [frame.name for frame in frames]
        if len(names) != len(set(names)):
            raise ValueError("Frame names must be unique.")
        return MultiFrameRecipe(
            working_dir=self.working_dir,
            frames=frames,
            output_root=self.output_root.value.strip() or None,
            cache_dir=self.cache_dir.value.strip() or None,
            same_rois_all_frames=self.same_rois_all_frames.value,
            export_scaled_spectra=self.export_scaled_spectra.value,
            frame_multipliers={
                frame.name: self._frame_scale(frame.name)
                for frame in frames
                if frame.enabled
            },
            frame_scaling_mode=self.frame_scaling_mode.value,
            frame_sample_flux_multipliers={
                name: float(multiplier)
                for name, multiplier in self.frame_sample_flux_multipliers.items()
                if name in names
            },
            frame_ob_flux_multipliers={
                name: float(multiplier)
                for name, multiplier in self.frame_ob_flux_multipliers.items()
                if name in names
            },
            show_hybrid_flux_plots=self.show_hybrid_flux_plots.value,
            overlap_windows=self._manual_overlap_window_configs(),
            pair_scaling_methods=self._pair_scaling_configs(),
            show_prompt_flash_lines=self.show_prompt_flash_lines.value,
        )

    def _wire_events(self) -> None:
        self.add_frame_button.on_click(self._add_frame)
        self.remove_frame_button.on_click(self._remove_frame)
        self.preview_button.on_click(self._preview)
        self.replot_button.on_click(self._replot)
        self.save_button.on_click(self._save_recipe)
        self.load_button.on_click(self._load_recipe)
        self.output_root_browse.on_click(self._browse_output_root)
        self.cache_dir_browse.on_click(self._browse_cache_dir)
        self.recipe_file_browse.on_click(self._browse_recipe_file)
        self.run_button.on_click(self._run_full_normalization)
        self.preview_output_spectra_button.on_click(self._preview_output_spectra)
        self.reset_frame_scales_button.on_click(self._reset_frame_scales)
        self.auto_frame_scales_button.on_click(self._auto_scale_from_highest_energy)
        self.arm_full_run.observe(self._update_run_button, names="value")
        self.inspect_frames.observe(self._overlap_selection_changed, names="value")
        self.show_native.observe(self._plot_setting_changed, names="value")
        self.show_errors.observe(self._plot_setting_changed, names="value")
        self.show_prompt_flash_lines.observe(self._plot_setting_changed, names="value")
        self.frame_scaling_mode.observe(self._frame_scaling_mode_changed, names="value")
        self.show_hybrid_flux_plots.observe(self._plot_setting_changed, names="value")
        self.same_rois_all_frames.observe(self._same_rois_all_frames_changed, names="value")
        self.enable_spectrum_only_all_button.on_click(
            lambda _button: self._set_all_spectrum_only(True)
        )
        self.disable_spectrum_only_all_button.on_click(
            lambda _button: self._set_all_spectrum_only(False)
        )

    def _browse_output_root(self, _button) -> None:
        self._show_path_browser(
            target=self.output_root,
            instruction="Select or create the full-normalization output directory",
            selection_type="directory",
        )

    def _browse_cache_dir(self, _button) -> None:
        self._show_path_browser(
            target=self.cache_dir,
            instruction="Select or create the preview-cache directory",
            selection_type="directory",
        )

    def _browse_recipe_file(self, _button) -> None:
        self._show_path_browser(
            target=self.recipe_file,
            instruction="Select an existing multi-frame recipe JSON file",
            selection_type="file",
            filters={"JSON recipes": "*.json"},
            default_filter="JSON recipes",
        )

    def _show_path_browser(
        self,
        target: widgets.Text,
        instruction: str,
        selection_type: str,
        filters: dict[str, str] | None = None,
        default_filter: str | None = None,
    ) -> None:
        current = Path(target.value.strip()).expanduser() if target.value.strip() else None
        if current is not None and selection_type == "directory" and current.is_dir():
            start_dir = current
        elif current is not None:
            start_dir = current.parent
        else:
            start_dir = Path(self.working_dir) / "shared"
        if not start_dir.is_dir():
            shared_dir = Path(self.working_dir) / "shared"
            start_dir = shared_dir if shared_dir.is_dir() else Path(self.working_dir)

        def selected(path) -> None:
            paths = path if isinstance(path, (list, tuple)) else [path]
            if paths:
                target.value = str(Path(paths[0]))
            with self.path_browser_output:
                clear_output(wait=True)

        with self.path_browser_output:
            clear_output(wait=True)
            selector = FileSelectorPanel(
                instruction=instruction,
                start_dir=str(start_dir),
                type=selection_type,
                multiple=False,
                newdir_toolbar_button=selection_type == "directory",
                filters=filters or {},
                default_filter=default_filter,
                next=selected,
            )
            selector.show()
            self._path_selector = selector

    def _set_all_spectrum_only(self, enabled: bool) -> None:
        for editor in self.frame_editors:
            editor.spectrum_only.value = bool(enabled)

    def _set_frames(self, frames: list[FrameConfig]) -> None:
        self.frame_editors = [
            FrameEditor(
                frame,
                self.working_dir,
                on_change=self._frame_changed,
                on_delete=self._delete_frame,
                on_move_up=self._move_frame_up,
                on_move_down=self._move_frame_down,
                show_prompt_flash_lines=self.show_prompt_flash_lines,
            )
            for frame in frames
        ]
        only_one_frame = len(self.frame_editors) <= 1
        for editor in self.frame_editors:
            editor.delete_button.disabled = only_one_frame
        self._refresh_frame_order_widgets()
        self.frame_box.selected_index = 0 if self.frame_editors else None
        self._refresh_overlap_options()
        if self.same_rois_all_frames.value and self.frame_editors:
            self._sync_frame_rois(self.frame_editors[0])

    def _refresh_overlap_options(self) -> None:
        names = [editor.name.value for editor in self.frame_editors if editor.enabled.value and editor.name.value]
        old_selection = tuple(name for name in self.inspect_frames.value if name in names)
        self.inspect_frames.options = names
        self.inspect_frames.value = old_selection or tuple(names)
        self._refresh_frame_scale_controls(names)
        self._refresh_overlap_window_controls(names)
        self._refresh_frame_output_range_controls(names)

    def _refresh_frame_output_range_controls(self, names: list[str]) -> None:
        editors_by_name = {
            editor.name.value: editor
            for editor in self.frame_editors
            if editor.enabled.value and editor.name.value
        }
        rows = []
        for name in names:
            editor = editors_by_name[name]
            editor.output_energy_min.description = "min (eV)"
            editor.output_energy_max.description = "max (eV)"
            editor.output_energy_min.layout = _layout("240px")
            editor.output_energy_max.layout = _layout("240px")
            rows.append(
                widgets.VBox(
                    [
                        widgets.HBox(
                            [
                                widgets.HTML(
                                    f"<b>{_escape(name)}</b>",
                                    layout=_layout("180px"),
                                ),
                                editor.output_energy_min,
                                editor.output_energy_max,
                            ],
                            layout=_wrapping_row_layout(),
                        ),
                        widgets.HBox(
                            [
                                widgets.HTML("", layout=_layout("180px")),
                                editor.add_output_exclusion_button,
                            ],
                            layout=_wrapping_row_layout(),
                        ),
                        editor.output_exclusion_box,
                    ]
                )
            )
        self.frame_output_range_box.children = tuple(rows)

    @staticmethod
    def _overlap_pair_key(frame_a: str, frame_b: str) -> tuple[str, str]:
        return tuple(sorted((str(frame_a), str(frame_b))))

    def _cache_overlap_window_states(self) -> None:
        for key, controls in self.overlap_window_widgets.items():
            self._overlap_window_state_cache[key] = (
                bool(controls["auto"].value),
                float(controls["minimum"].value),
                float(controls["maximum"].value),
                str(controls["scaling_method"].value),
            )

    def _preset_pair_scaling_method(
        self,
        pair_index: int,
        pair_count: int,
    ) -> str:
        if self.frame_scaling_mode.value == FRAME_SCALING_TRANSMISSION:
            return PAIR_SCALING_TRANSMISSION
        return (
            PAIR_SCALING_TRANSMISSION
            if pair_index == pair_count - 1
            else PAIR_SCALING_NATIVE_FLUX
        )

    def _refresh_overlap_window_controls(self, names: list[str]) -> None:
        self._cache_overlap_window_states()
        controls_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        rows = []
        self._updating_overlap_windows = True
        try:
            pairs = list(zip(names[:-1], names[1:]))
            for pair_index, (frame_a, frame_b) in enumerate(pairs):
                key = self._overlap_pair_key(frame_a, frame_b)
                automatic, minimum_value, maximum_value, scaling_method = (
                    self._overlap_window_state_cache.get(
                        key,
                        (
                            True,
                            0.0,
                            0.0,
                            self._preset_pair_scaling_method(pair_index, len(pairs)),
                        ),
                    )
                )
                automatic_control = widgets.Checkbox(
                    value=automatic,
                    description="Auto",
                    indent=False,
                    layout=_layout("90px"),
                )
                minimum_control = widgets.FloatText(
                    value=minimum_value,
                    description="min (eV)",
                    disabled=automatic,
                    layout=_layout("230px"),
                )
                maximum_control = widgets.FloatText(
                    value=maximum_value,
                    description="max (eV)",
                    disabled=automatic,
                    layout=_layout("230px"),
                )
                scaling_method_control = widgets.Dropdown(
                    options=(
                        ("Transmission", PAIR_SCALING_TRANSMISSION),
                        ("Native sample/OB flux", PAIR_SCALING_NATIVE_FLUX),
                    ),
                    value=scaling_method,
                    description="scale with",
                    layout=_layout("300px"),
                )
                automatic_control.observe(
                    self._overlap_window_setting_changed,
                    names="value",
                )
                minimum_control.observe(self._overlap_window_setting_changed, names="value")
                maximum_control.observe(self._overlap_window_setting_changed, names="value")
                scaling_method_control.observe(
                    self._pair_scaling_setting_changed,
                    names="value",
                )
                row = widgets.HBox(
                    [
                        widgets.HTML(
                            f"<b>{_escape(frame_a)} / {_escape(frame_b)}</b>",
                            layout=_layout("300px"),
                        ),
                        automatic_control,
                        minimum_control,
                        maximum_control,
                        scaling_method_control,
                    ],
                    layout=_wrapping_row_layout(),
                )
                controls_by_pair[key] = {
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "auto": automatic_control,
                    "minimum": minimum_control,
                    "maximum": maximum_control,
                    "scaling_method": scaling_method_control,
                }
                rows.append(row)
        finally:
            self._updating_overlap_windows = False
        self.overlap_window_widgets = controls_by_pair
        self.overlap_window_box.children = tuple(rows)
        self._update_native_flux_plot_control()

    def _overlap_window_setting_changed(self, _change=None) -> None:
        for controls in self.overlap_window_widgets.values():
            automatic = bool(controls["auto"].value)
            controls["minimum"].disabled = automatic
            controls["maximum"].disabled = automatic
        self._cache_overlap_window_states()
        if self._updating_overlap_windows:
            return
        try:
            self._manual_overlap_window_configs()
        except ValueError:
            return
        self._plot_setting_changed()

    def _pair_scaling_setting_changed(self, _change=None) -> None:
        self._cache_overlap_window_states()
        if self._updating_overlap_windows:
            return
        self.frame_sample_flux_multipliers.clear()
        self.frame_ob_flux_multipliers.clear()
        self._update_native_flux_plot_control()
        self.frame_scale_status.value = (
            "<span style='font-size:12px; color:#7a5200'>"
            "A pair scaling method changed; run automatic scaling to refresh the "
            "frame multipliers and native-flux diagnostics.</span>"
        )
        self._plot_setting_changed()

    def _manual_overlap_window_configs(self) -> tuple[OverlapWindowConfig, ...]:
        windows = []
        for controls in self.overlap_window_widgets.values():
            if controls["auto"].value:
                continue
            window = OverlapWindowConfig(
                frame_a=str(controls["frame_a"]),
                frame_b=str(controls["frame_b"]),
                energy_min_eV=float(controls["minimum"].value),
                energy_max_eV=float(controls["maximum"].value),
            )
            window.validate()
            windows.append(window)
        return tuple(windows)

    def _pair_scaling_configs(self) -> tuple[PairScalingConfig, ...]:
        configs = []
        for controls in self.overlap_window_widgets.values():
            config = PairScalingConfig(
                frame_a=str(controls["frame_a"]),
                frame_b=str(controls["frame_b"]),
                method=str(controls["scaling_method"].value),
            )
            config.validate()
            configs.append(config)
        return tuple(configs)

    def _restore_overlap_windows(
        self,
        windows: tuple[OverlapWindowConfig, ...],
    ) -> None:
        self._updating_overlap_windows = True
        try:
            for window in windows:
                window.validate()
                key = self._overlap_pair_key(window.frame_a, window.frame_b)
                self._overlap_window_state_cache[key] = (
                    False,
                    float(window.energy_min_eV),
                    float(window.energy_max_eV),
                    str(
                        self._overlap_window_state_cache.get(
                            key,
                            (True, 0.0, 0.0, PAIR_SCALING_TRANSMISSION),
                        )[3]
                    ),
                )
                controls = self.overlap_window_widgets.get(key)
                if controls is None:
                    continue
                controls["minimum"].value = float(window.energy_min_eV)
                controls["maximum"].value = float(window.energy_max_eV)
                controls["auto"].value = False
                controls["minimum"].disabled = False
                controls["maximum"].disabled = False
        finally:
            self._updating_overlap_windows = False
        self._cache_overlap_window_states()

    def _restore_pair_scaling_methods(
        self,
        configs: tuple[PairScalingConfig, ...],
    ) -> None:
        self._updating_overlap_windows = True
        try:
            for config in configs:
                config.validate()
                key = self._overlap_pair_key(config.frame_a, config.frame_b)
                automatic, minimum, maximum, _method = (
                    self._overlap_window_state_cache.get(
                        key,
                        (True, 0.0, 0.0, PAIR_SCALING_TRANSMISSION),
                    )
                )
                self._overlap_window_state_cache[key] = (
                    automatic,
                    minimum,
                    maximum,
                    config.method,
                )
                controls = self.overlap_window_widgets.get(key)
                if controls is not None:
                    controls["scaling_method"].value = config.method
        finally:
            self._updating_overlap_windows = False
        self._cache_overlap_window_states()
        self._update_native_flux_plot_control()

    def _refresh_frame_scale_controls(self, names: list[str]) -> None:
        active_names = set(names)
        self.frame_sample_flux_multipliers = {
            name: multiplier
            for name, multiplier in self.frame_sample_flux_multipliers.items()
            if name in active_names
        }
        self.frame_ob_flux_multipliers = {
            name: multiplier
            for name, multiplier in self.frame_ob_flux_multipliers.items()
            if name in active_names
        }
        if tuple(self.frame_scale_widgets) == tuple(names):
            return
        old_values = {
            name: float(widget.value)
            for name, widget in self.frame_scale_widgets.items()
        }
        controls: dict[str, widgets.BoundedFloatText] = {}
        for name in names:
            control = widgets.BoundedFloatText(
                value=old_values.get(name, 1.0),
                min=1e-6,
                max=1e6,
                step=0.001,
                description=f"{name} x",
                disabled=False,
                layout=_layout("230px"),
            )
            control.style.description_width = "initial"
            control.observe(self._frame_scale_changed, names="value")
            controls[name] = control
        self.frame_scale_widgets = controls
        self.frame_scale_box.children = tuple(controls.values())

    def _pair_scaling_method(self, frame_a: str, frame_b: str) -> str:
        controls = self.overlap_window_widgets.get(
            self._overlap_pair_key(frame_a, frame_b)
        )
        if controls is None:
            return PAIR_SCALING_TRANSMISSION
        return str(controls["scaling_method"].value)

    def _has_native_flux_pairs(self) -> bool:
        return any(
            controls["scaling_method"].value == PAIR_SCALING_NATIVE_FLUX
            for controls in self.overlap_window_widgets.values()
        )

    def _update_native_flux_plot_control(self) -> None:
        self.show_hybrid_flux_plots.disabled = not self._has_native_flux_pairs()

    def _reset_frame_scales(self, _button=None) -> None:
        self._updating_frame_scales = True
        try:
            for control in self.frame_scale_widgets.values():
                control.value = 1.0
                control.disabled = False
        finally:
            self._updating_frame_scales = False
        self.frame_sample_flux_multipliers.clear()
        self.frame_ob_flux_multipliers.clear()
        self.frame_scale_status.value = ""
        if self.engine is not None and self.engine.previews:
            self._draw_plot()

    def _frame_scaling_mode_changed(self, _change=None) -> None:
        controls = list(self.overlap_window_widgets.values())
        self._updating_overlap_windows = True
        try:
            for pair_index, pair_controls in enumerate(controls):
                pair_controls["scaling_method"].value = (
                    self._preset_pair_scaling_method(pair_index, len(controls))
                )
        finally:
            self._updating_overlap_windows = False
        self._cache_overlap_window_states()
        self._update_native_flux_plot_control()
        self.auto_frame_scales_button.description = "Auto-scale selected pair methods"
        self.frame_sample_flux_multipliers.clear()
        self.frame_ob_flux_multipliers.clear()
        self.frame_scale_status.value = (
            "<span style='font-size:12px; color:#7a5200'>"
            "Pair-method preset applied; adjust any pair selector if needed, then "
            "run automatic scaling to refresh the fitted "
            "frame multipliers.</span>"
        )
        self._plot_setting_changed()

    def _frame_scale_changed(self, _change=None) -> None:
        if self._updating_frame_scales:
            return
        if self._has_native_flux_pairs():
            self.frame_sample_flux_multipliers.clear()
            self.frame_ob_flux_multipliers.clear()
            self.frame_scale_status.value = (
                "<span style='font-size:12px; color:#7a5200'>"
                "A frame multiplier changed; rerun automatic scaling to refresh "
                "the native sample/OB flux factors and plots.</span>"
            )
        self._plot_setting_changed()

    @staticmethod
    def _maximum_usable_energy(preview: RebinnedFramePreview) -> float:
        energy = np.asarray(preview.energy_eV, dtype=np.float64)
        transmission = np.asarray(preview.transmission, dtype=np.float64)
        valid = np.isfinite(energy) & (energy > 0) & np.isfinite(transmission)
        if not np.any(valid):
            raise ValueError(f"{preview.name}: no finite positive-energy transmission points.")
        return float(np.max(energy[valid]))

    def _auto_scale_from_highest_energy(self, _button=None) -> None:
        try:
            if self.engine is None or not self.engine.previews:
                raise ValueError("Load the ROI profiles before auto-scaling frames.")
            selected_names = [
                name for name in self._selected_frame_names() if name in self.engine.previews
            ]
            if len(selected_names) < 2:
                raise ValueError("Select at least two loaded frames for automatic scaling.")
            input_order = {name: index for index, name in enumerate(selected_names)}
            maximum_energy = {
                name: self._maximum_usable_energy(self.engine.previews[name])
                for name in selected_names
            }
            names = sorted(
                selected_names,
                key=lambda name: (maximum_energy[name], input_order[name]),
            )
            anchor_name = names[-1]
            anchor_control = self.frame_scale_widgets.get(anchor_name)
            anchor_scale = (
                1.0 if anchor_control is None else float(anchor_control.value)
            )
            if not np.isfinite(anchor_scale) or anchor_scale <= 0:
                raise ValueError(
                    f"The highest-energy anchor scale for {anchor_name} must be "
                    "positive and finite."
                )

            native_pair_names = {
                name
                for frame_a, frame_b in zip(names[:-1], names[1:])
                if self._pair_scaling_method(frame_a, frame_b)
                == PAIR_SCALING_NATIVE_FLUX
                for name in (frame_a, frame_b)
            }
            if native_pair_names:
                configs_by_name = {
                    editor.name.value: editor.to_config()
                    for editor in self.frame_editors
                    if editor.enabled.value and editor.name.value
                }
                for name in sorted(native_pair_names):
                    config = self.loaded_frame_configs.get(name, configs_by_name.get(name))
                    if config is None:
                        raise ValueError(f"Could not resolve the loaded settings for {name}.")
                    if not config.use_proton_charge:
                        raise ValueError(
                            f"{name}: native-flux pair scaling requires proton-charge "
                            "normalization."
                        )
                    if config.container_roi is not None or config.container_roi_file:
                        raise ValueError(
                            f"{name}: container correction is not present in the fast native "
                            "ROI preview, so native-flux pair scaling is unavailable."
                        )

            fitted_scales = {anchor_name: anchor_scale}
            sample_flux_scales = {anchor_name: anchor_scale}
            ob_flux_scales = {anchor_name: 1.0}
            result_lines = [
                f"{anchor_name} x{anchor_scale:.7g} (editable highest-energy anchor; "
                f"Emax={maximum_energy[anchor_name]:.6g} eV)"
            ]
            reference_name = anchor_name
            reference = self._scaled_preview_for_plot(
                self.engine.previews[anchor_name],
                anchor_scale,
            )
            for comparison_name in reversed(names[:-1]):
                comparison = self.engine.previews[comparison_name]
                fit_window = self._overlap_window_for_pair(
                    reference_name,
                    comparison_name,
                    manual_window=None,
                )
                transmission_diagnostics = calculate_overlap_diagnostics(
                    reference,
                    comparison,
                    fit_window,
                )
                pair_method = self._pair_scaling_method(
                    reference_name,
                    comparison_name,
                )
                if pair_method == PAIR_SCALING_NATIVE_FLUX:
                    flux_diagnostics = calculate_native_flux_overlap_diagnostics(
                        self.engine.previews[reference_name],
                        comparison,
                        fit_window,
                        reference_sample_scale=sample_flux_scales[reference_name],
                        reference_ob_scale=ob_flux_scales[reference_name],
                    )
                    sample_flux_scales[comparison_name] = float(
                        flux_diagnostics.sample.scale_comparison_to_reference
                    )
                    ob_flux_scales[comparison_name] = float(
                        flux_diagnostics.ob.scale_comparison_to_reference
                    )
                    scale = float(
                        flux_diagnostics.transmission_scale_comparison_to_reference
                    )
                    result_lines.append(
                        f"{comparison_name} x{scale:.7g} from native fluxes: "
                        f"sample x{sample_flux_scales[comparison_name]:.7g} "
                        f"({flux_diagnostics.sample.point_count} overlap points, chi2r="
                        f"{flux_diagnostics.sample.reduced_chi_square:.3g}); "
                        f"OB x{ob_flux_scales[comparison_name]:.7g} "
                        f"({flux_diagnostics.ob.point_count} overlap points, chi2r="
                        f"{flux_diagnostics.ob.reduced_chi_square:.3g}); "
                        f"direct-transmission comparison x"
                        f"{transmission_diagnostics.scale_comparison_to_reference:.7g} "
                        f"(chi2r={transmission_diagnostics.reduced_chi_square:.3g})"
                    )
                else:
                    scale = float(
                        transmission_diagnostics.scale_comparison_to_reference
                    )
                    # A transmission multiplier does not uniquely determine two
                    # channel factors. Preserve the reference OB gauge and assign
                    # the sample factor needed to reproduce the fitted transmission
                    # multiplier. This keeps a following native-flux fit chainable.
                    ob_flux_scales[comparison_name] = ob_flux_scales[reference_name]
                    sample_flux_scales[comparison_name] = (
                        scale * ob_flux_scales[comparison_name]
                    )
                    result_lines.append(
                        f"{comparison_name} x{scale:.7g} from transmission "
                        f"({transmission_diagnostics.point_count} overlap points, chi2r="
                        f"{transmission_diagnostics.reduced_chi_square:.3g})"
                    )
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        f"Could not determine a positive finite scale for {comparison_name}."
                    )
                fitted_scales[comparison_name] = scale
                reference_name = comparison_name
                reference = self._scaled_preview_for_plot(comparison, scale)

            self._updating_frame_scales = True
            try:
                for name, control in self.frame_scale_widgets.items():
                    control.disabled = False
                    if name in fitted_scales:
                        control.value = fitted_scales[name]
            finally:
                self._updating_frame_scales = False
            if native_pair_names:
                self.frame_sample_flux_multipliers = sample_flux_scales
                self.frame_ob_flux_multipliers = ob_flux_scales
            else:
                self.frame_sample_flux_multipliers.clear()
                self.frame_ob_flux_multipliers.clear()
            self.frame_scale_status.value = (
                "<span style='font-size:12px; color:#176b36'>"
                + "<br>".join(_escape(line) for line in result_lines)
                + "</span>"
            )
            self._draw_plot()
        except Exception as error:
            self.frame_scale_status.value = (
                "<span style='font-size:12px; color:#b00020'>"
                f"Auto-scale unavailable: {_escape(error)}</span>"
            )

    def _frame_scale(self, name: str) -> float:
        control = self.frame_scale_widgets.get(name)
        return 1.0 if control is None else float(control.value)

    def _scaled_preview_for_plot(
        self,
        preview: RebinnedFramePreview,
        scale: float,
    ) -> RebinnedFramePreview:
        scaled_native = replace(
            preview.native,
            transmission=np.asarray(preview.native.transmission) * scale,
            uncertainty=np.asarray(preview.native.uncertainty) * abs(scale),
        )
        return replace(
            preview,
            transmission=np.asarray(preview.transmission) * scale,
            uncertainty=np.asarray(preview.uncertainty) * abs(scale),
            native=scaled_native,
        )

    def _hybrid_flux_scaling_figures(
        self,
        selected_names: list[str],
        color_by_name: dict[str, str],
    ) -> list[go.Figure]:
        if (
            not self.show_hybrid_flux_plots.value
            or len(selected_names) < 2
        ):
            return []
        if not self.frame_sample_flux_multipliers or not self.frame_ob_flux_multipliers:
            return []

        input_order = {name: index for index, name in enumerate(selected_names)}
        maximum_energy = {
            name: self._maximum_usable_energy(self.engine.previews[name])
            for name in selected_names
        }
        names = sorted(
            selected_names,
            key=lambda name: (maximum_energy[name], input_order[name]),
        )
        figures = []
        for comparison_name, reference_name in zip(names[:-1], names[1:]):
            if (
                self._pair_scaling_method(reference_name, comparison_name)
                != PAIR_SCALING_NATIVE_FLUX
            ):
                continue
            if (
                comparison_name not in self.frame_sample_flux_multipliers
                or reference_name not in self.frame_sample_flux_multipliers
                or comparison_name not in self.frame_ob_flux_multipliers
                or reference_name not in self.frame_ob_flux_multipliers
            ):
                continue
            fit_window = self._overlap_window_for_pair(
                reference_name,
                comparison_name,
                manual_window=None,
            )
            figure = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.11,
                subplot_titles=("Sample native ROI flux", "Open-beam native ROI flux"),
            )
            channel_settings = (
                (
                    "sample",
                    self.frame_sample_flux_multipliers[reference_name],
                    self.frame_sample_flux_multipliers[comparison_name],
                    1,
                ),
                (
                    "ob",
                    self.frame_ob_flux_multipliers[reference_name],
                    self.frame_ob_flux_multipliers[comparison_name],
                    2,
                ),
            )
            overlap_min = None
            overlap_max = None
            for channel, reference_scale, comparison_scale, row in channel_settings:
                (
                    energy,
                    reference_flux,
                    reference_uncertainty,
                    comparison_flux,
                    comparison_uncertainty,
                    scaled_comparison_flux,
                    scaled_comparison_uncertainty,
                ) = native_flux_overlap_arrays(
                    self.engine.previews[reference_name],
                    self.engine.previews[comparison_name],
                    channel,
                    fit_window,
                    reference_scale=reference_scale,
                    comparison_scale=comparison_scale,
                )
                if len(energy) < 2:
                    continue
                overlap_min = float(np.min(energy)) if overlap_min is None else min(
                    overlap_min, float(np.min(energy))
                )
                overlap_max = float(np.max(energy)) if overlap_max is None else max(
                    overlap_max, float(np.max(energy))
                )
                reference_error = (
                    dict(
                        type="data",
                        array=reference_uncertainty,
                        visible=True,
                        thickness=0.8,
                        width=0,
                    )
                    if self.show_errors.value
                    else None
                )
                raw_error = (
                    dict(
                        type="data",
                        array=comparison_uncertainty,
                        visible=True,
                        thickness=0.6,
                        width=0,
                    )
                    if self.show_errors.value
                    else None
                )
                scaled_error = (
                    dict(
                        type="data",
                        array=scaled_comparison_uncertainty,
                        visible=True,
                        thickness=0.8,
                        width=0,
                    )
                    if self.show_errors.value
                    else None
                )
                figure.add_trace(
                    go.Scattergl(
                        x=energy,
                        y=reference_flux,
                        mode="lines",
                        line=dict(color=color_by_name[reference_name], width=2),
                        error_y=reference_error,
                        name=(
                            f"{reference_name} {channel} x{reference_scale:.7g}"
                        ),
                        legendgroup=f"{channel}-reference",
                    ),
                    row=row,
                    col=1,
                )
                figure.add_trace(
                    go.Scattergl(
                        x=energy,
                        y=comparison_flux,
                        mode="lines",
                        line=dict(color="#777", width=1, dash="dot"),
                        opacity=0.55,
                        error_y=raw_error,
                        name=f"{comparison_name} {channel} unscaled",
                        legendgroup=f"{channel}-comparison-raw",
                    ),
                    row=row,
                    col=1,
                )
                figure.add_trace(
                    go.Scattergl(
                        x=energy,
                        y=scaled_comparison_flux,
                        mode="lines",
                        line=dict(color=color_by_name[comparison_name], width=2),
                        error_y=scaled_error,
                        name=(
                            f"{comparison_name} {channel} x{comparison_scale:.7g}"
                        ),
                        legendgroup=f"{channel}-comparison-scaled",
                    ),
                    row=row,
                    col=1,
                )
            figure.update_xaxes(type="log", title_text="Incident neutron energy (eV)", row=2, col=1)
            figure.update_yaxes(type="log", title_text="counts / C", row=1, col=1)
            figure.update_yaxes(type="log", title_text="counts / C", row=2, col=1)
            figure.update_layout(
                title=(
                    f"Native-flux pair scaling: {comparison_name} onto {reference_name}"
                    + (
                        f"<br><sup>fit window={overlap_min * 1e3:.5g}-"
                        f"{overlap_max * 1e3:.5g} meV</sup>"
                        if overlap_min is not None and overlap_max is not None
                        else ""
                    )
                ),
                template="plotly_white",
                height=720,
                hovermode="closest",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(l=80, r=30, t=120, b=65),
            )
            figures.append(figure)
        return figures

    def _frame_changed(self, change=None) -> None:
        if self.same_rois_all_frames.value and not self._syncing_frame_rois and change is not None:
            source = next(
                (
                    editor
                    for editor in self.frame_editors
                    if change.get("owner") in editor.roi_value_widgets
                ),
                None,
            )
            if source is not None:
                self._sync_frame_rois(source)
        for index, editor in enumerate(self.frame_editors):
            self.frame_box.set_title(index, editor.name.value or f"frame {index + 1}")
        self._refresh_overlap_options()

    def _refresh_frame_order_widgets(self) -> None:
        self.frame_box.children = tuple(editor.widget for editor in self.frame_editors)
        last_index = len(self.frame_editors) - 1
        for index, editor in enumerate(self.frame_editors):
            self.frame_box.set_title(index, editor.name.value or f"frame {index + 1}")
            editor.move_up_button.disabled = index == 0
            editor.move_down_button.disabled = index == last_index

    def _move_frame_up(self, target: FrameEditor) -> None:
        self._move_frame(target, -1)

    def _move_frame_down(self, target: FrameEditor) -> None:
        self._move_frame(target, 1)

    def _move_frame(self, target: FrameEditor, offset: int) -> None:
        if target not in self.frame_editors:
            return
        source_index = self.frame_editors.index(target)
        destination_index = source_index + int(offset)
        if destination_index < 0 or destination_index >= len(self.frame_editors):
            return
        self.frame_editors[source_index], self.frame_editors[destination_index] = (
            self.frame_editors[destination_index],
            self.frame_editors[source_index],
        )
        self._refresh_frame_order_widgets()
        self.frame_box.selected_index = destination_index
        self._refresh_overlap_options()
        if self.engine is not None and self.engine.previews:
            self._draw_plot()

    def _same_rois_all_frames_changed(self, change=None) -> None:
        if not self.same_rois_all_frames.value or not self.frame_editors:
            return
        selected_index = self.frame_box.selected_index
        source_index = 0 if selected_index is None else selected_index
        self._sync_frame_rois(self.frame_editors[source_index])

    def _sync_frame_rois(self, source: FrameEditor) -> None:
        if self._syncing_frame_rois:
            return
        self._syncing_frame_rois = True
        try:
            for target in self.frame_editors:
                if target is source:
                    continue
                target.ob_roi_left.value = source.ob_roi_left.value
                target.ob_roi_top.value = source.ob_roi_top.value
                target.ob_roi_width.value = source.ob_roi_width.value
                target.ob_roi_height.value = source.ob_roi_height.value
                target.roi_left.value = source.roi_left.value
                target.roi_top.value = source.roi_top.value
                target.roi_width.value = source.roi_width.value
                target.roi_height.value = source.roi_height.value
                target.ob_roi_linked.value = source.ob_roi_linked.value
                target._auto_roi_applied_sources = set(source._auto_roi_applied_sources)
                target._update_ob_roi_state()
        finally:
            self._syncing_frame_rois = False

    def _ensure_auto_rois(self) -> list[str]:
        enabled_editors = [editor for editor in self.frame_editors if editor.enabled.value]
        if not enabled_editors:
            return []
        if self.same_rois_all_frames.value:
            source = enabled_editors[0]
            notes = source.ensure_auto_roi()
            self._sync_frame_rois(source)
            return notes
        notes = []
        for editor in enabled_editors:
            notes.extend(editor.ensure_auto_roi())
        return notes

    def _add_frame(self, _button) -> None:
        index = len(self.frame_editors) + 1
        frames = [editor.to_config() for editor in self.frame_editors]
        frames.append(_empty_frame(f"frame {index}", DetectorType.tpx1))
        self._set_frames(frames)

    def _remove_frame(self, _button) -> None:
        if len(self.frame_editors) <= 1:
            return
        self._set_frames([editor.to_config() for editor in self.frame_editors[:-1]])

    def _delete_frame(self, target: FrameEditor) -> None:
        if len(self.frame_editors) <= 1 or target not in self.frame_editors:
            return
        frames = [
            editor.to_config()
            for editor in self.frame_editors
            if editor is not target
        ]
        self._set_frames(frames)

    def _preview(self, _button) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                auto_roi_notes = self._ensure_auto_rois()
                recipe = self.recipe()
                self.engine = MultiFramePreviewEngine(recipe)
                display(
                    HTML(
                        ("<br>".join(_escape(note) for note in auto_roi_notes) + "<br>"
                         if auto_roi_notes else "")
                        + "Loading ROI profiles..."
                    )
                )
                conversion_notes = []
                for editor, frame in zip(self.frame_editors, recipe.frames):
                    if not frame.enabled:
                        continue
                    native = self.engine.load_frame(
                        frame,
                        force_reload=self.force_reload.value,
                    )
                    frame, note = editor.prepare_rebin_for_native(frame, native)
                    if note:
                        conversion_notes.append(note)
                    self.engine.rebin_frame(frame, native=native)
                recipe = self.recipe()
                self.engine.recipe = recipe
                self.loaded_frame_configs = {
                    frame.name: frame for frame in recipe.frames if frame.enabled
                }
                clear_output(wait=True)
                if conversion_notes:
                    display(
                        HTML(
                            "<div style='color:#176b36; margin-bottom:8px'>"
                            + "<br>".join(_escape(note) for note in conversion_notes)
                            + "</div>"
                        )
                    )
                self._display_profile_status()
                self._draw_plot()
            except Exception as error:
                clear_output(wait=True)
                display(HTML(f"<span style='color:#b00020'><b>Preview failed:</b> {_escape(error)}</span>"))

    def _replot(self, _button=None) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                if self.engine is None or not self.engine.native_profiles:
                    raise ValueError("Load ROI profiles first.")
                recipe = self.recipe()
                old_native = self.engine.native_profiles
                self.engine = MultiFramePreviewEngine(recipe)
                self.engine.native_profiles.update(old_native)
                conversion_notes = []
                for editor, frame in zip(self.frame_editors, recipe.frames):
                    if not frame.enabled:
                        continue
                    old_frame = self.loaded_frame_configs.get(frame.name)
                    source_unchanged = (
                        old_frame is not None
                        and replace(old_frame, rebin=RebinConfig(), enabled=True)
                        == replace(frame, rebin=RebinConfig(), enabled=True)
                    )
                    native = self.engine.native_profiles.get(frame.name) if source_unchanged else None
                    if native is None:
                        native = self.engine.load_frame(frame)
                    frame, note = editor.prepare_rebin_for_native(frame, native)
                    if note:
                        conversion_notes.append(note)
                    self.engine.rebin_frame(frame, native=native)
                recipe = self.recipe()
                self.engine.recipe = recipe
                self.loaded_frame_configs = {
                    frame.name: frame for frame in recipe.frames if frame.enabled
                }
                if conversion_notes:
                    display(
                        HTML(
                            "<div style='color:#176b36; margin-bottom:8px'>"
                            + "<br>".join(_escape(note) for note in conversion_notes)
                            + "</div>"
                        )
                    )
                self._display_profile_status()
                self._draw_plot()
            except Exception as error:
                display(HTML(f"<span style='color:#b00020'><b>Rebin failed:</b> {_escape(error)}</span>"))

    def _display_profile_status(self) -> None:
        assert self.engine is not None
        rows = []
        for frame in self.recipe().frames:
            if not frame.enabled or frame.name not in self.engine.previews:
                continue
            preview = self.engine.previews[frame.name]
            native = preview.native
            warning = "<br>".join(_escape(item) for item in native.warnings) or "none"
            corrections = "<br>".join(_escape(item) for item in native.corrections_applied) or "none"
            rows.append(
                "<tr>"
                f"<td>{_escape(frame.name)}</td>"
                f"<td>{len(native.tof_s)}</td>"
                f"<td>{len(preview.tof_s)}</td>"
                f"<td>{native.sample_total_proton_charge_c or 'not used'}</td>"
                f"<td>{native.ob_total_proton_charge_c or 'not used'}</td>"
                f"<td>{corrections}</td>"
                f"<td>{warning}</td>"
                "</tr>"
            )
        display(
            HTML(
                "<table style='border-collapse:collapse' border='1' cellpadding='5'>"
                "<tr><th>Frame</th><th>Native bins</th><th>Preview bins</th>"
                "<th>Sample charge (C)</th><th>OB charge (C)</th>"
                "<th>Corrections in preview</th><th>Warnings</th></tr>"
                + "".join(rows)
                + "</table>"
            )
        )

    def _draw_plot(self) -> None:
        if self.engine is None or not self.engine.previews:
            return
        selected_names = self._selected_frame_names()
        overlap_pairs = list(zip(selected_names[:-1], selected_names[1:]))
        scales = {name: self._frame_scale(name) for name in selected_names}
        scaled_previews = {
            name: self._scaled_preview_for_plot(self.engine.previews[name], scales[name])
            for name in selected_names
        }
        transmission_figure = go.Figure()
        overlap_figures: list[go.Figure] = []
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        color_by_name = {
            name: palette[index % len(palette)]
            for index, name in enumerate(self.engine.previews)
        }
        hybrid_flux_figures: list[go.Figure] = []
        hybrid_flux_error: Exception | None = None
        for name in selected_names:
            preview = scaled_previews[name]
            scale = scales[name]
            color = color_by_name[name]
            if self.show_native.value:
                native_order = np.argsort(preview.native.energy_eV)
                transmission_figure.add_trace(
                    go.Scattergl(
                        x=preview.native.energy_eV[native_order],
                        y=preview.native.transmission[native_order],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.25,
                        name=f"{name} native (x{scale:.7g})",
                        legendgroup=name,
                    )
                )
            order = np.argsort(preview.energy_eV)
            error = None
            if self.show_errors.value:
                error = dict(
                    type="data",
                    array=preview.uncertainty[order],
                    visible=True,
                    thickness=1.2,
                    width=2,
                )
            transmission_figure.add_trace(
                go.Scatter(
                    x=preview.energy_eV[order],
                    y=preview.transmission[order],
                    mode="markers+lines",
                    marker=dict(color=color, size=5),
                    line=dict(color=color, width=1),
                    error_y=error,
                    name=f"{name} proposed bins (x{scale:.7g})",
                    legendgroup=name,
                    customdata=preview.source_frame_count[order],
                    hovertemplate=(
                        "Energy=%{x:.6g} eV<br>Transmission=%{y:.6g}"
                        "<br>native frames/bin=%{customdata}<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        for reference_name, comparison_name in overlap_pairs:
            reference = scaled_previews.get(reference_name)
            comparison = scaled_previews.get(comparison_name)
            ratio_figure = go.Figure()
            if reference is not None and comparison is not None:
                try:
                    pair_window = self._overlap_window_for_pair(
                        reference_name,
                        comparison_name,
                        manual_window=None,
                    )
                    energy, ratio, uncertainty = overlap_ratio_arrays(
                        reference,
                        comparison,
                        pair_window,
                    )
                    diagnostics = calculate_overlap_diagnostics(
                        reference,
                        comparison,
                        pair_window,
                    )
                    ratio_figure.add_trace(
                        go.Scatter(
                            x=energy,
                            y=ratio,
                            mode="markers",
                            marker=dict(color=color_by_name[comparison_name], size=5),
                            error_y=(
                                dict(type="data", array=uncertainty, visible=True, thickness=0.8, width=0)
                                if self.show_errors.value
                                else None
                            ),
                            name=f"{comparison_name} / {reference_name}",
                            showlegend=False,
                        )
                    )
                    ratio_figure.add_hline(y=1.0, line_color="#666", line_width=1)
                    transmission_figure.add_vrect(
                        x0=diagnostics.energy_min_eV,
                        x1=diagnostics.energy_max_eV,
                        fillcolor="#999",
                        opacity=0.08,
                        line_width=0,
                    )
                    ratio_figure.update_layout(
                        title=(
                            f"{comparison_name} / {reference_name} overlap"
                            "<br><sup>"
                            f"multipliers: {comparison_name} x{scales[comparison_name]:.7g}, "
                            f"{reference_name} x{scales[reference_name]:.7g}; "
                            f"overlap={diagnostics.energy_min_eV * 1e3:.4g}-"
                            f"{diagnostics.energy_max_eV * 1e3:.4g} meV; "
                            f"ratio={diagnostics.comparison_over_reference:.5g}; "
                            f"residual scale={diagnostics.scale_comparison_to_reference:.5g} +/- "
                            f"{diagnostics.scale_uncertainty:.2g}; "
                            f"reduced chi2={diagnostics.reduced_chi_square:.3g}</sup>"
                        ),
                        template="plotly_white",
                        height=380,
                        margin=dict(l=70, r=30, t=85, b=60),
                        hovermode="closest",
                    )
                    ratio_figure.update_xaxes(type="log", title_text="Incident neutron energy (eV)")
                    ratio_figure.update_yaxes(title_text="Ratio")
                except Exception as error:
                    ratio_figure.add_annotation(
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        text=f"Overlap unavailable: {_escape(error)}",
                        showarrow=False,
                    )
                    ratio_figure.update_layout(
                        title=f"{comparison_name} / {reference_name} overlap",
                        template="plotly_white",
                        height=300,
                    )
            overlap_figures.append(ratio_figure)

        transmission_figure.update_xaxes(
            type="log",
            range=[np.log10(0.001), np.log10(30.0)],
            title_text="Incident neutron energy (eV)",
        )
        transmission_figure.update_yaxes(title_text="Transmission")
        _add_prompt_flash_lines(
            transmission_figure,
            self.show_prompt_flash_lines.value,
            energy_min_eV=0.001,
            energy_max_eV=30.0,
        )
        transmission_figure.update_layout(
            title="Selected frame transmission previews with frame multipliers",
            template="plotly_white",
            height=650,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=70, r=30, t=100, b=60),
        )
        try:
            hybrid_flux_figures = self._hybrid_flux_scaling_figures(
                selected_names,
                color_by_name,
            )
        except Exception as error:
            hybrid_flux_error = error
        with self.plot_output:
            clear_output(wait=True)
            transmission_figure.show()
            for ratio_figure in overlap_figures:
                ratio_figure.show()
            for flux_figure in hybrid_flux_figures:
                flux_figure.show()
            if hybrid_flux_error is not None:
                display(
                    HTML(
                        "<span style='color:#b00020'>Hybrid native-flux plot "
                        f"unavailable: {_escape(hybrid_flux_error)}</span>"
                    )
                )

    def _preview_output_spectra(self, _button=None) -> None:
        with self.output_spectra_plot_output:
            clear_output(wait=True)
            try:
                if self.engine is None or not self.engine.previews:
                    raise ValueError("Load the ROI profiles before previewing output spectra.")

                palette = [
                    "#1f77b4",
                    "#d62728",
                    "#2ca02c",
                    "#9467bd",
                    "#ff7f0e",
                    "#17becf",
                ]
                figure = go.Figure()
                trace_count = 0
                for editor in self.frame_editors:
                    name = editor.name.value
                    if not editor.enabled.value or name not in self.engine.previews:
                        continue
                    minimum = (
                        float(editor.output_energy_min.value)
                        if editor.output_energy_min.value > 0
                        else None
                    )
                    maximum = (
                        float(editor.output_energy_max.value)
                        if editor.output_energy_max.value > 0
                        else None
                    )
                    if minimum is not None and maximum is not None and maximum <= minimum:
                        raise ValueError(
                            f"{name}: output maximum energy must be greater than minimum."
                        )

                    preview = self.engine.previews[name]
                    energy = np.asarray(preview.energy_eV, dtype=np.float64)
                    transmission = np.asarray(preview.transmission, dtype=np.float64)
                    uncertainty = np.asarray(preview.uncertainty, dtype=np.float64)
                    selected = (
                        np.isfinite(energy)
                        & (energy > 0)
                        & np.isfinite(transmission)
                    )
                    if minimum is not None:
                        selected &= energy >= minimum
                    if maximum is not None:
                        selected &= energy <= maximum
                    for excluded_minimum, excluded_maximum in (
                        editor.output_excluded_energy_ranges()
                    ):
                        selected &= ~(
                            (energy >= excluded_minimum)
                            & (energy <= excluded_maximum)
                        )
                    if not np.any(selected):
                        raise ValueError(
                            f"{name}: the selected output energy range contains no preview bins."
                        )

                    scale = self._frame_scale(name) if self.export_scaled_spectra.value else 1.0
                    order = np.argsort(energy[selected])
                    selected_energy = energy[selected][order]
                    selected_transmission = transmission[selected][order] * scale
                    selected_uncertainty = uncertainty[selected][order] * abs(scale)
                    color = palette[trace_count % len(palette)]
                    figure.add_trace(
                        go.Scatter(
                            x=selected_energy,
                            y=selected_transmission,
                            mode="markers+lines",
                            marker=dict(color=color, size=5),
                            line=dict(color=color, width=1),
                            error_y=(
                                dict(
                                    type="data",
                                    array=selected_uncertainty,
                                    visible=True,
                                    thickness=1.2,
                                    width=2,
                                )
                                if self.show_errors.value
                                else None
                            ),
                            name=(
                                f"{name} selected x{scale:.7g}"
                                if self.export_scaled_spectra.value
                                else f"{name} selected"
                            ),
                        )
                    )
                    trace_count += 1

                if trace_count == 0:
                    raise ValueError("No enabled frame has a loaded preview profile.")
                figure.update_xaxes(
                    type="log",
                    range=[np.log10(0.001), np.log10(30.0)],
                    title_text="Incident neutron energy (eV)",
                )
                figure.update_yaxes(title_text="Transmission")
                _add_prompt_flash_lines(
                    figure,
                    self.show_prompt_flash_lines.value,
                    energy_min_eV=0.001,
                    energy_max_eV=30.0,
                )
                figure.update_layout(
                    title="Selected frame output spectra preview",
                    template="plotly_white",
                    height=650,
                    hovermode="closest",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                    ),
                    margin=dict(l=70, r=30, t=100, b=60),
                )
                figure.show()
            except Exception as error:
                display(
                    HTML(
                        "<span style='color:#b00020'>Output-spectrum preview failed: "
                        f"{_escape(error)}</span>"
                    )
                )

    def _overlap_window_for_pair(
        self,
        reference_name: str,
        comparison_name: str,
        manual_window: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        if manual_window is None:
            controls = self.overlap_window_widgets.get(
                self._overlap_pair_key(reference_name, comparison_name)
            )
            if controls is not None and not controls["auto"].value:
                window = OverlapWindowConfig(
                    frame_a=reference_name,
                    frame_b=comparison_name,
                    energy_min_eV=float(controls["minimum"].value),
                    energy_max_eV=float(controls["maximum"].value),
                )
                window.validate()
                manual_window = (window.energy_min_eV, window.energy_max_eV)
        if manual_window is not None:
            window = OverlapWindowConfig(
                frame_a=reference_name,
                frame_b=comparison_name,
                energy_min_eV=float(manual_window[0]),
                energy_max_eV=float(manual_window[1]),
            )
            window.validate()
            return window.energy_min_eV, window.energy_max_eV

        if self.engine is not None and self.engine.previews:
            loaded_names = [
                name
                for name in self._selected_frame_names()
                if name in self.engine.previews
            ]
            if loaded_names:
                highest_energy_name = max(
                    loaded_names,
                    key=lambda name: self._maximum_usable_energy(
                        self.engine.previews[name]
                    ),
                )
                if highest_energy_name in {reference_name, comparison_name}:
                    other_name = (
                        comparison_name
                        if reference_name == highest_energy_name
                        else reference_name
                    )
                    highest_energy = np.asarray(
                        self.engine.previews[highest_energy_name].energy_eV,
                        dtype=np.float64,
                    )
                    other_energy = np.asarray(
                        self.engine.previews[other_name].energy_eV,
                        dtype=np.float64,
                    )
                    highest_valid = highest_energy[
                        np.isfinite(highest_energy) & (highest_energy > 0)
                    ]
                    other_valid = other_energy[
                        np.isfinite(other_energy) & (other_energy > 0)
                    ]
                    if (
                        highest_valid.size
                        and other_valid.size
                        and np.min(highest_valid) <= 0.2 <= np.max(highest_valid)
                        and np.min(other_valid) <= 0.2 <= np.max(other_valid)
                    ):
                        return (0.0, 0.2)
        return None

    def _selected_frame_names(self) -> list[str]:
        selected = set(self.inspect_frames.value)
        ordered = [str(name) for name in self.inspect_frames.options]
        return [name for name in ordered if name in selected]

    def _overlap_selection_changed(self, _change=None) -> None:
        if self.engine is not None and self.engine.previews:
            self._draw_plot()

    def _plot_setting_changed(self, _change=None) -> None:
        if self._updating_frame_scales:
            return
        if self.engine is not None and self.engine.previews:
            self._draw_plot()

    def _save_recipe(self, _button) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                output = self.recipe().save(self.recipe_file.value)
                display(HTML(f"Saved recipe: <code>{_escape(output)}</code>"))
            except Exception as error:
                display(HTML(f"<span style='color:#b00020'>Save failed: {_escape(error)}</span>"))

    def _load_recipe(self, _button) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                recipe = MultiFrameRecipe.load(self.recipe_file.value)
                self.working_dir = recipe.working_dir
                self.output_root.value = recipe.output_root or str(Path(recipe.working_dir) / "shared")
                self.cache_dir.value = recipe.cache_dir or str(
                    Path(recipe.working_dir) / "shared" / ".normalization_tof_multiple_frames_cache"
                )
                self.same_rois_all_frames.value = recipe.same_rois_all_frames
                self.export_scaled_spectra.value = recipe.export_scaled_spectra
                self.show_prompt_flash_lines.value = recipe.show_prompt_flash_lines
                self.frame_scaling_mode.value = recipe.frame_scaling_mode
                self.show_hybrid_flux_plots.value = recipe.show_hybrid_flux_plots
                self._overlap_window_state_cache.clear()
                self.overlap_window_widgets = {}
                self._set_frames(recipe.frames)
                self._restore_overlap_windows(recipe.overlap_windows)
                self._restore_pair_scaling_methods(recipe.pair_scaling_methods)
                self._updating_frame_scales = True
                try:
                    for name, multiplier in recipe.frame_multipliers.items():
                        control = self.frame_scale_widgets.get(name)
                        if control is not None and not control.disabled:
                            control.value = float(multiplier)
                finally:
                    self._updating_frame_scales = False
                self.frame_sample_flux_multipliers = dict(
                    recipe.frame_sample_flux_multipliers
                )
                self.frame_ob_flux_multipliers = dict(
                    recipe.frame_ob_flux_multipliers
                )
                self.engine = None
                self.loaded_frame_configs = {}
                with self.plot_output:
                    clear_output()
                display(HTML(f"Loaded recipe: <code>{_escape(self.recipe_file.value)}</code>"))
            except Exception as error:
                display(HTML(f"<span style='color:#b00020'>Load failed: {_escape(error)}</span>"))

    def _update_run_button(self, _change=None) -> None:
        self.run_button.disabled = not self.arm_full_run.value

    def _run_full_normalization(self, _button) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                if not self.arm_full_run.value:
                    raise ValueError("Enable the full run first.")
                auto_roi_notes = self._ensure_auto_rois()
                recipe = self.recipe()
                recipe.save(self.recipe_file.value)
                engine = MultiFramePreviewEngine(recipe)
                display(
                    HTML(
                        ("<br>".join(_escape(note) for note in auto_roi_notes) + "<br>"
                         if auto_roi_notes else "")
                        + "Running each enabled frame in a separate output folder..."
                    )
                )
                campaign = engine.run_full_normalization(preview=True)
                clear_output(wait=True)
                display(HTML(f"Normalization campaign completed: <code>{_escape(campaign)}</code>"))
                self.arm_full_run.value = False
            except Exception as error:
                clear_output(wait=True)
                display(HTML(f"<span style='color:#b00020'><b>Full run failed:</b> {_escape(error)}</span>"))


def _format_schedule(schedule: tuple[tuple[float | None, float], ...]) -> str:
    lines = []
    for end_value, step in schedule:
        end_text = "" if end_value is None else f"{end_value:.10g}"
        lines.append(f"{end_text}, {step:g}")
    return "\n".join(lines)


def _parse_schedule(text: str) -> list[tuple[float | None, float]]:
    schedule = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Custom schedule line {line_number} must be 'end, step' or ', step'.")
        end_value = None if not parts[0] else float(parts[0])
        schedule.append((end_value, float(parts[1])))
    if not schedule:
        raise ValueError("Custom schedule needs at least one segment.")
    return schedule


def _run_number_from_folder(folder: str) -> str:
    match = re.search(r"run[_-]?(\d+)", str(folder), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Could not infer a run number from direct folder {folder!r}. "
            "Enter the matching run number in the run-number field."
        )
    return match.group(1)


def _empty_frame(name: str, detector_type: str) -> FrameConfig:
    return FrameConfig(
        name=name,
        detector_type=detector_type,
        sample_runs=(),
        ob_runs=(),
        roi=RoiConfig(left=156, top=156, width=200, height=200),
        rebin=RebinConfig(mode=RebinMode.none),
    )


def _default_frames() -> list[FrameConfig]:
    return [
        _empty_frame("6.3 A", DetectorType.tpx1),
        _empty_frame("4.5 A", DetectorType.tpx1),
        _empty_frame("2.5 A", DetectorType.tpx1),
        _empty_frame("0.3 A", DetectorType.tpx1),
        _empty_frame("resonance", DetectorType.tpx1),
    ]


def _escape(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
