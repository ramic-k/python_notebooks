"""Notebook widgets for planning and previewing several normalization frames."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display
from plotly.subplots import make_subplots

from __code.normalization_tof import (
    DetectorType,
    RebinCustomBasis,
    RebinCustomScale,
    RebinMode,
)
from __code.normalization_tof.multiple_frames import (
    FrameConfig,
    MultiFramePreviewEngine,
    MultiFrameRecipe,
    RebinConfig,
    RoiConfig,
    RunSpec,
    calculate_overlap_diagnostics,
    overlap_ratio_arrays,
    parse_run_numbers,
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


def _layout(width: str = "220px") -> widgets.Layout:
    return widgets.Layout(width=width)


class FrameEditor:
    def __init__(self, config: FrameConfig, on_change=None):
        self.on_change = on_change
        self.enabled = widgets.Checkbox(value=config.enabled, description="Use frame", indent=False)
        self.name = widgets.Text(value=config.name, description="Name", layout=_layout("360px"))
        self.detector = widgets.Dropdown(
            options=[DetectorType.tpx1_legacy, DetectorType.tpx1, DetectorType.tpx3],
            value=config.detector_type,
            description="Detector",
            layout=_layout("520px"),
        )
        self.sample_runs = widgets.Text(
            value=", ".join(run.run_number for run in config.sample_runs),
            description="Sample runs",
            placeholder="19558 or 19419, 19447-19448",
            layout=_layout("520px"),
        )
        self.ob_runs = widgets.Text(
            value=", ".join(run.run_number for run in config.ob_runs),
            description="OB runs",
            placeholder="19559 or 19420, 19449",
            layout=_layout("520px"),
        )
        self.roi_left = widgets.BoundedIntText(value=config.roi.left, min=0, description="left", layout=_layout())
        self.roi_top = widgets.BoundedIntText(value=config.roi.top, min=0, description="top", layout=_layout())
        self.roi_width = widgets.BoundedIntText(value=config.roi.width, min=1, description="width", layout=_layout())
        self.roi_height = widgets.BoundedIntText(value=config.roi.height, min=1, description="height", layout=_layout())
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
        self.use_proton_charge = widgets.Checkbox(
            value=config.use_proton_charge,
            description="Normalize by proton charge",
            indent=False,
            layout=_layout("250px"),
        )
        self.experimental_uncertainties = widgets.Checkbox(
            value=config.use_experimental_uncertainties,
            description="Experimental uncertainty model",
            indent=False,
            layout=_layout("280px"),
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
            value=rebin.delta_tof_over_tof or 0.005, description="delta TOF / TOF"
        )
        self.delta_lambda_relative = widgets.FloatText(
            value=rebin.delta_lambda_over_lambda or 0.005, description="delta lambda / lambda"
        )
        self.delta_lambda_squared = widgets.FloatText(
            value=rebin.delta_lambda_squared_a2 or 0.01, description="delta lambda^2 (A^2)"
        )
        self.custom_basis = widgets.Dropdown(
            options=[RebinCustomBasis.tof, RebinCustomBasis.lambda_, RebinCustomBasis.lambda_squared],
            value=rebin.custom_basis or RebinCustomBasis.tof,
            description="Custom basis",
        )
        self.custom_scale = widgets.Dropdown(
            options=[RebinCustomScale.linear, RebinCustomScale.log, RebinCustomScale.reverse_log],
            value=rebin.custom_scale or RebinCustomScale.linear,
            description="Custom scale",
        )
        self.custom_schedule = widgets.Textarea(
            value=_format_schedule(rebin.custom_schedule),
            placeholder="560, 60\n2700, 60\n, 100",
            description="Schedule",
            layout=widgets.Layout(width="520px", height="120px"),
        )
        self.full_bins_only = widgets.Checkbox(
            value=rebin.full_bins_only, description="Full bins only", indent=False
        )
        self.snap_to_native = widgets.Checkbox(
            value=rebin.snap_to_native_grid, description="Snap widths to native grid", indent=False
        )

        self.rebin_parameters = widgets.VBox()
        self.rebin_mode.observe(self._update_rebin_parameters, names="value")
        self.auto_detector_delay.observe(self._update_delay_state, names="value")
        self._update_rebin_parameters()
        self._update_delay_state()

        all_widgets = self._all_widgets()
        if on_change is not None:
            for widget in all_widgets:
                widget.observe(on_change, names="value")

        roi_row = widgets.HBox([self.roi_left, self.roi_top, self.roi_width, self.roi_height])
        axis_row = widgets.HBox([self.distance, self.detector_delay, self.auto_detector_delay, self.manual_tof])
        flags_row = widgets.HBox([self.use_proton_charge, self.experimental_uncertainties])
        rebin_flags = widgets.HBox([self.full_bins_only, self.snap_to_native])
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.enabled, self.name]),
                self.detector,
                self.sample_runs,
                self.ob_runs,
                widgets.HTML("<b>ROI</b>"),
                roi_row,
                axis_row,
                flags_row,
                widgets.HTML("<b>Proposed rebinning</b>"),
                self.rebin_mode,
                self.rebin_parameters,
                rebin_flags,
            ],
            layout=widgets.Layout(border="1px solid #bbb", padding="8px", margin="0 0 8px 0"),
        )

    def _all_widgets(self) -> list[widgets.Widget]:
        return [
            self.enabled,
            self.name,
            self.detector,
            self.sample_runs,
            self.ob_runs,
            self.roi_left,
            self.roi_top,
            self.roi_width,
            self.roi_height,
            self.distance,
            self.detector_delay,
            self.auto_detector_delay,
            self.manual_tof,
            self.use_proton_charge,
            self.experimental_uncertainties,
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

    def _update_delay_state(self, _change=None) -> None:
        self.detector_delay.disabled = self.auto_detector_delay.value

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
            controls = [self.custom_basis, self.custom_scale, self.custom_schedule]
        else:
            controls = [widgets.HTML("Native TOF bins will be used.")]
        self.rebin_parameters.children = tuple(controls)

    def to_config(self) -> FrameConfig:
        sample = tuple(RunSpec(run) for run in parse_run_numbers(self.sample_runs.value))
        ob = tuple(RunSpec(run) for run in parse_run_numbers(self.ob_runs.value))
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
        return FrameConfig(
            name=self.name.value.strip(),
            detector_type=self.detector.value,
            sample_runs=sample,
            ob_runs=ob,
            roi=RoiConfig(
                left=self.roi_left.value,
                top=self.roi_top.value,
                width=self.roi_width.value,
                height=self.roi_height.value,
            ),
            rebin=rebin,
            distance_source_detector_m=self.distance.value,
            detector_delay_us=None if self.auto_detector_delay.value else self.detector_delay.value,
            manual_tof_bin_size_ns=self.manual_tof.value if self.manual_tof.value > 0 else None,
            use_proton_charge=self.use_proton_charge.value,
            use_experimental_uncertainties=self.experimental_uncertainties.value,
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
        self.show_native = widgets.Checkbox(value=True, description="Show native profiles", indent=False)
        self.show_errors = widgets.Checkbox(value=True, description="Show error bars", indent=False)
        self.force_reload = widgets.Checkbox(value=False, description="Ignore cached profiles", indent=False)
        self.overlap_min = widgets.FloatText(value=0.0, description="Overlap min (eV)")
        self.overlap_max = widgets.FloatText(value=0.0, description="Overlap max (eV)")
        self.reference = widgets.Dropdown(options=[], description="Reference")
        self.comparison = widgets.Dropdown(options=[], description="Comparison")
        self.add_frame_button = widgets.Button(description="Add frame", icon="plus")
        self.remove_frame_button = widgets.Button(description="Remove last", icon="minus")
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

        initial_frames = frames if frames is not None else _default_frames()
        self._set_frames(initial_frames)
        self._wire_events()
        self._update_run_button()

    def display(self) -> None:
        header = widgets.VBox(
            [
                widgets.HBox([self.add_frame_button, self.remove_frame_button]),
                self.output_root,
                self.cache_dir,
                self.recipe_file,
                widgets.HBox([self.save_button, self.load_button]),
            ]
        )
        overlap_controls = widgets.VBox(
            [
                widgets.HBox([self.reference, self.comparison, self.overlap_min, self.overlap_max]),
                widgets.HBox([self.show_native, self.show_errors, self.force_reload]),
                widgets.HBox([self.preview_button, self.replot_button]),
            ]
        )
        full_run_controls = widgets.HBox([self.arm_full_run, self.run_button])
        display(
            widgets.VBox(
                [
                    header,
                    widgets.HTML("<h3>Independent frame settings</h3>"),
                    self.frame_box,
                    widgets.HTML("<h3>Overlap inspection</h3>"),
                    overlap_controls,
                    self.status_output,
                    self.plot_output,
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
        )

    def _wire_events(self) -> None:
        self.add_frame_button.on_click(self._add_frame)
        self.remove_frame_button.on_click(self._remove_frame)
        self.preview_button.on_click(self._preview)
        self.replot_button.on_click(self._replot)
        self.save_button.on_click(self._save_recipe)
        self.load_button.on_click(self._load_recipe)
        self.run_button.on_click(self._run_full_normalization)
        self.arm_full_run.observe(self._update_run_button, names="value")
        self.reference.observe(self._pair_changed, names="value")
        self.comparison.observe(self._pair_changed, names="value")
        self.show_native.observe(self._plot_setting_changed, names="value")
        self.show_errors.observe(self._plot_setting_changed, names="value")

    def _set_frames(self, frames: list[FrameConfig]) -> None:
        self.frame_editors = [FrameEditor(frame, on_change=self._frame_changed) for frame in frames]
        self.frame_box.children = tuple(editor.widget for editor in self.frame_editors)
        for index, editor in enumerate(self.frame_editors):
            self.frame_box.set_title(index, editor.name.value or f"frame {index + 1}")
        self.frame_box.selected_index = 0 if self.frame_editors else None
        self._refresh_pair_options()

    def _refresh_pair_options(self) -> None:
        names = [editor.name.value for editor in self.frame_editors if editor.enabled.value and editor.name.value]
        old_reference, old_comparison = self.reference.value, self.comparison.value
        self.reference.options = names
        self.comparison.options = names
        if old_reference in names:
            self.reference.value = old_reference
        elif names:
            self.reference.value = names[0]
        if old_comparison in names and old_comparison != self.reference.value:
            self.comparison.value = old_comparison
        elif len(names) > 1:
            self.comparison.value = names[1]

    def _frame_changed(self, _change=None) -> None:
        for index, editor in enumerate(self.frame_editors):
            self.frame_box.set_title(index, editor.name.value or f"frame {index + 1}")
        self._refresh_pair_options()

    def _add_frame(self, _button) -> None:
        index = len(self.frame_editors) + 1
        frames = [editor.to_config() for editor in self.frame_editors]
        frames.append(_empty_frame(f"frame {index}", DetectorType.tpx1))
        self._set_frames(frames)

    def _remove_frame(self, _button) -> None:
        if len(self.frame_editors) <= 1:
            return
        self._set_frames([editor.to_config() for editor in self.frame_editors[:-1]])

    def _preview(self, _button) -> None:
        with self.status_output:
            clear_output(wait=True)
            try:
                recipe = self.recipe()
                self.engine = MultiFramePreviewEngine(recipe)
                display(HTML("Loading ROI profiles..."))
                self.engine.preview_all(force_reload=self.force_reload.value)
                self.loaded_frame_configs = {frame.name: frame for frame in recipe.frames if frame.enabled}
                clear_output(wait=True)
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
                for frame in recipe.frames:
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
                    self.engine.rebin_frame(frame, native=native)
                self.loaded_frame_configs = {frame.name: frame for frame in recipe.frames if frame.enabled}
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
            rows.append(
                "<tr>"
                f"<td>{_escape(frame.name)}</td>"
                f"<td>{len(native.tof_s)}</td>"
                f"<td>{len(preview.tof_s)}</td>"
                f"<td>{native.sample_total_proton_charge_c or 'not used'}</td>"
                f"<td>{native.ob_total_proton_charge_c or 'not used'}</td>"
                f"<td>{warning}</td>"
                "</tr>"
            )
        display(
            HTML(
                "<table style='border-collapse:collapse' border='1' cellpadding='5'>"
                "<tr><th>Frame</th><th>Native bins</th><th>Preview bins</th>"
                "<th>Sample charge (C)</th><th>OB charge (C)</th><th>Warnings</th></tr>"
                + "".join(rows)
                + "</table>"
            )
        )

    def _draw_plot(self) -> None:
        if self.engine is None or not self.engine.previews:
            return
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Frame transmission previews", "Selected overlap ratio"),
        )
        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        for index, (name, preview) in enumerate(self.engine.previews.items()):
            color = palette[index % len(palette)]
            if self.show_native.value:
                native_order = np.argsort(preview.native.energy_eV)
                figure.add_trace(
                    go.Scattergl(
                        x=preview.native.energy_eV[native_order],
                        y=preview.native.transmission[native_order],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.25,
                        name=f"{name} native",
                        legendgroup=name,
                    ),
                    row=1,
                    col=1,
                )
            order = np.argsort(preview.energy_eV)
            error = None
            if self.show_errors.value:
                error = dict(type="data", array=preview.uncertainty[order], visible=True, thickness=0.8, width=0)
            figure.add_trace(
                go.Scatter(
                    x=preview.energy_eV[order],
                    y=preview.transmission[order],
                    mode="markers+lines",
                    marker=dict(color=color, size=5),
                    line=dict(color=color, width=1),
                    error_y=error,
                    name=f"{name} proposed bins",
                    legendgroup=name,
                    customdata=preview.source_frame_count[order],
                    hovertemplate=(
                        "Energy=%{x:.6g} eV<br>Transmission=%{y:.6g}"
                        "<br>native frames/bin=%{customdata}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=1,
            )

        reference_name, comparison_name = self.reference.value, self.comparison.value
        if reference_name and comparison_name and reference_name != comparison_name:
            window = self._energy_window()
            reference = self.engine.previews.get(reference_name)
            comparison = self.engine.previews.get(comparison_name)
            if reference is not None and comparison is not None:
                try:
                    energy, ratio, uncertainty = overlap_ratio_arrays(reference, comparison, window)
                    diagnostics = calculate_overlap_diagnostics(reference, comparison, window)
                    figure.add_trace(
                        go.Scatter(
                            x=energy,
                            y=ratio,
                            mode="markers",
                            marker=dict(color="#111", size=5),
                            error_y=(
                                dict(type="data", array=uncertainty, visible=True, thickness=0.8, width=0)
                                if self.show_errors.value
                                else None
                            ),
                            name=f"{comparison_name} / {reference_name}",
                        ),
                        row=2,
                        col=1,
                    )
                    figure.add_hline(y=1.0, line_color="#666", line_width=1, row=2, col=1)
                    figure.add_vrect(
                        x0=diagnostics.energy_min_eV,
                        x1=diagnostics.energy_max_eV,
                        fillcolor="#999",
                        opacity=0.08,
                        line_width=0,
                        row="all",
                        col=1,
                    )
                    figure.add_annotation(
                        x=0.01,
                        y=0.98,
                        xref="x2 domain",
                        yref="y2 domain",
                        xanchor="left",
                        yanchor="top",
                        showarrow=False,
                        text=(
                            f"comparison/reference={diagnostics.comparison_over_reference:.5g}; "
                            f"diagnostic scale={diagnostics.scale_comparison_to_reference:.5g} +/- "
                            f"{diagnostics.scale_uncertainty:.2g}; reduced chi2={diagnostics.reduced_chi_square:.3g}"
                        ),
                    )
                except Exception as error:
                    figure.add_annotation(
                        x=0.5,
                        y=0.5,
                        xref="x2 domain",
                        yref="y2 domain",
                        text=f"Overlap unavailable: {_escape(error)}",
                        showarrow=False,
                    )

        figure.update_xaxes(type="log", title_text="Incident neutron energy (eV)", row=2, col=1)
        figure.update_yaxes(title_text="Transmission", row=1, col=1)
        figure.update_yaxes(title_text="Ratio", row=2, col=1)
        figure.update_layout(
            template="plotly_white",
            height=820,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=70, r=30, t=100, b=60),
        )
        with self.plot_output:
            clear_output(wait=True)
            figure.show()

    def _energy_window(self) -> tuple[float, float] | None:
        if self.overlap_min.value > 0 and self.overlap_max.value > 0:
            if self.overlap_max.value <= self.overlap_min.value:
                raise ValueError("Overlap maximum must be greater than minimum.")
            return self.overlap_min.value, self.overlap_max.value
        return None

    def _pair_changed(self, _change=None) -> None:
        if self.engine is not None and self.engine.previews:
            self._draw_plot()

    def _plot_setting_changed(self, _change=None) -> None:
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
                self._set_frames(recipe.frames)
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
                recipe = self.recipe()
                recipe.save(self.recipe_file.value)
                engine = MultiFramePreviewEngine(recipe)
                display(HTML("Running each enabled frame in a separate output folder..."))
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
        end_text = "" if end_value is None else f"{end_value:g}"
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
