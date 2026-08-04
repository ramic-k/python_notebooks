"""Calculation and recipe support for multi-frame normalization previews.

The preview intentionally works on ROI count profiles before the expensive image
normalization. Counts and variances are summed inside each proposed TOF bin before
sample/OB division, matching the order used by the production normalization code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from skimage.io import imread
from tifffile import memmap as tiff_memmap

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
from __code.normalization_tof.config import timepix1_config, timepix3_config
from __code.normalization_tof.normalization_for_timepix1_timepix3 import (
    normalization_with_list_of_full_path,
)
from __code.normalization_tof.utilities import (
    DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE,
    _evaluate_black_filter_background_shape,
    _load_black_filter_background_shape,
    build_rebin_bin_groups,
    build_rebin_bin_metadata,
    calculate_ratio_and_uncertainty,
    calculate_time_lambda_energy_arrays,
    get_detector_offset_from_nexus,
    retrieve_list_of_tif,
)


RECIPE_VERSION = 1


def _clean_run_number(value: str | int) -> str:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Invalid run number: {value!r}")
    return match.group(1)


def parse_run_numbers(value: str | Iterable[str | int]) -> list[str]:
    """Parse comma-separated runs and inclusive ranges such as ``19419-19421``."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in value if str(token).strip()]

    runs: list[str] = []
    for token in tokens:
        range_match = re.fullmatch(r"(?:Run_)?(\d+)\s*-\s*(?:Run_)?(\d+)", token)
        if range_match:
            start, end = (int(item) for item in range_match.groups())
            if end < start:
                raise ValueError(f"Run range must increase: {token!r}")
            runs.extend(str(run) for run in range(start, end + 1))
        else:
            runs.append(_clean_run_number(token))
    return runs


@dataclass(frozen=True)
class RoiConfig:
    left: int = 0
    top: int = 0
    width: int = 1
    height: int = 1

    def validate(self) -> None:
        if self.left < 0 or self.top < 0:
            raise ValueError("ROI left and top must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive.")

    def to_roi(self) -> Roi:
        self.validate()
        return Roi(left=self.left, top=self.top, width=self.width, height=self.height)


@dataclass(frozen=True)
class RunSpec:
    run_number: str
    data_path: str | None = None
    nexus_path: str | None = None

    @classmethod
    def from_value(cls, value: str | int | dict[str, Any]) -> "RunSpec":
        if isinstance(value, dict):
            return cls(
                run_number=_clean_run_number(value["run_number"]),
                data_path=value.get("data_path"),
                nexus_path=value.get("nexus_path"),
            )
        return cls(run_number=_clean_run_number(value))


@dataclass(frozen=True)
class BlackFilterBackgroundConfig:
    enabled: bool = False
    shape_file: str = DEFAULT_BLACK_FILTER_BACKGROUND_SHAPE_FILE
    anchor_energy_eV: float = 5.1044

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "BlackFilterBackgroundConfig":
        return cls(**(values or {}))


@dataclass(frozen=True)
class MeasuredBackgroundConfig:
    key: str
    enabled: bool = False
    sample_runs: tuple[RunSpec, ...] = ()
    ob_runs: tuple[RunSpec, ...] = ()
    weight: float = 1.0
    mode: str | None = None
    column_label: str | None = None
    key_prefix: str | None = None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MeasuredBackgroundConfig":
        values = dict(values)
        values["sample_runs"] = tuple(RunSpec.from_value(item) for item in values.get("sample_runs", ()))
        values["ob_runs"] = tuple(RunSpec.from_value(item) for item in values.get("ob_runs", ()))
        return cls(**values)

    def normalized_labels(self) -> tuple[str, str, str]:
        defaults = {
            "bragg_edge_cd": (
                "Cd-filter background correction for Bragg edge mode",
                "Cd-filter",
                "bragg_edge_cd",
            ),
            "closed_slits": (
                "Closed-slits background correction for Bragg edge mode",
                "closed-slits",
                "closed_slits",
            ),
        }
        default_mode, default_label, default_prefix = defaults.get(
            self.key,
            (f"Measured background correction: {self.key}", self.key, self.key),
        )
        return (
            self.mode or default_mode,
            self.column_label or default_label,
            self.key_prefix or default_prefix,
        )


@dataclass(frozen=True)
class RebinConfig:
    mode: str = RebinMode.none
    delta_tof_us: float | None = None
    delta_lambda_a: float | None = None
    delta_tof_over_tof: float | None = None
    delta_lambda_over_lambda: float | None = None
    delta_lambda_squared_a2: float | None = None
    custom_basis: str | None = None
    custom_scale: str | None = None
    custom_schedule: tuple[tuple[float | None, float], ...] = ()
    full_bins_only: bool = True
    snap_to_native_grid: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "RebinConfig":
        values = dict(values or {})
        schedule = values.pop("custom_schedule", ()) or ()
        normalized_schedule = []
        for segment in schedule:
            if isinstance(segment, dict):
                normalized_schedule.append((segment.get("end_value"), float(segment["step"])))
            else:
                normalized_schedule.append((segment[0], float(segment[1])))
        return cls(custom_schedule=tuple(normalized_schedule), **values)

    def schedule_for_engine(self) -> list[dict[str, float | None]] | None:
        if not self.custom_schedule:
            return None
        return [
            {"end_value": end_value, "step": step}
            for end_value, step in self.custom_schedule
        ]

    def engine_kwargs(self) -> dict[str, Any]:
        return {
            "rebin_mode": self.mode,
            "rebin_delta_tof_us": self.delta_tof_us,
            "rebin_delta_lambda_a": self.delta_lambda_a,
            "rebin_delta_tof_over_tof": self.delta_tof_over_tof,
            "rebin_delta_lambda_over_lambda": self.delta_lambda_over_lambda,
            "rebin_delta_lambda_squared_a2": self.delta_lambda_squared_a2,
            "rebin_custom_basis": self.custom_basis,
            "rebin_custom_scale": self.custom_scale,
            "rebin_custom_schedule": self.schedule_for_engine(),
            "rebin_full_bins_only": self.full_bins_only,
            "rebin_snap_to_native_grid": self.snap_to_native_grid,
        }


def convert_tof_schedule_to_energy_schedule(
    tof_schedule: Iterable[tuple[float | None, float]],
    tof_s: np.ndarray,
    energy_eV: np.ndarray,
) -> tuple[tuple[float | None, float], ...]:
    """Translate a linear TOF-region recipe to energy edges with TOF widths.

    Legacy TOF schedules are ordered from short to long TOF, while energy-edge
    schedules are ordered from low to high energy. The region widths therefore
    have to be reversed along with the converted boundaries.
    """
    schedule = tuple(tof_schedule)
    if not schedule:
        raise ValueError("The earlier TOF schedule needs at least one segment.")

    tof_values_us = np.asarray(tof_s, dtype=np.float64) * 1e6
    energy_values = np.asarray(energy_eV, dtype=np.float64)
    if tof_values_us.ndim != 1 or energy_values.ndim != 1:
        raise ValueError("TOF-schedule conversion requires one-dimensional TOF and energy axes.")
    if tof_values_us.shape != energy_values.shape or len(tof_values_us) < 2:
        raise ValueError("TOF-schedule conversion requires matching non-empty TOF and energy axes.")
    if np.any(~np.isfinite(tof_values_us)) or np.any(np.diff(tof_values_us) <= 0):
        raise ValueError("TOF-schedule conversion requires a strictly increasing finite TOF axis.")
    if np.any(~np.isfinite(energy_values)) or np.any(energy_values <= 0):
        raise ValueError("TOF-schedule conversion requires finite positive energies.")
    if np.any(np.diff(energy_values) > 0):
        raise ValueError("TOF-schedule conversion requires energy to decrease as TOF increases.")

    none_indices = [index for index, (end_value, _) in enumerate(schedule) if end_value is None]
    if len(none_indices) > 1 or (none_indices and none_indices[0] != len(schedule) - 1):
        raise ValueError("The open-ended TOF segment must appear at most once, on the final line.")

    finite_segments = [(float(end_value), float(step)) for end_value, step in schedule if end_value is not None]
    all_steps = [float(step) for _, step in schedule]
    if any(not np.isfinite(step) or step <= 0 for step in all_steps):
        raise ValueError("Every TOF schedule width must be finite and strictly positive.")

    requested_tof_edges_us = np.asarray(
        [end_value for end_value, _ in finite_segments],
        dtype=np.float64,
    )
    if len(requested_tof_edges_us):
        if np.any(np.diff(requested_tof_edges_us) <= 0):
            raise ValueError("Earlier TOF schedule boundaries must be strictly increasing.")
        tof_min_us = float(tof_values_us[0])
        tof_max_us = float(tof_values_us[-1])
        if np.any(requested_tof_edges_us <= tof_min_us) or np.any(
            requested_tof_edges_us >= tof_max_us
        ):
            raise ValueError(
                "Earlier TOF schedule edges must lie strictly inside this frame's TOF range "
                f"({tof_min_us:.6g}, {tof_max_us:.6g}) us."
            )

    finite_widths_us = [step for _, step in finite_segments]
    low_energy_width_us = (
        float(schedule[-1][1])
        if schedule[-1][0] is None
        else finite_widths_us[-1]
    )
    tof_region_widths_us = [*finite_widths_us, low_energy_width_us]
    converted_energy_edges = np.interp(
        requested_tof_edges_us,
        tof_values_us,
        energy_values,
    )[::-1]
    energy_region_widths_us = tof_region_widths_us[::-1]

    converted = [
        (float(edge), float(width))
        for edge, width in zip(converted_energy_edges, energy_region_widths_us[:-1])
    ]
    converted.append((None, float(energy_region_widths_us[-1])))
    return tuple(converted)


@dataclass(frozen=True)
class FrameConfig:
    name: str
    detector_type: str
    sample_runs: tuple[RunSpec, ...]
    ob_runs: tuple[RunSpec, ...]
    roi: RoiConfig
    ob_roi: RoiConfig | None = None
    container_roi: RoiConfig | None = None
    container_roi_file: str | None = None
    dc_runs: tuple[RunSpec, ...] = ()
    black_filter_background: BlackFilterBackgroundConfig = field(default_factory=BlackFilterBackgroundConfig)
    measured_backgrounds: tuple[MeasuredBackgroundConfig, ...] = ()
    rebin: RebinConfig = field(default_factory=RebinConfig)
    distance_source_detector_m: float = 25.0
    detector_delay_us: float | None = None
    manual_tof_bin_size_ns: float | None = None
    use_proton_charge: bool = True
    use_experimental_uncertainties: bool = True
    combine_sample_runs: bool = False
    correct_chips_alignment: bool | None = None
    replace_ob_zeros_by_local_median: bool | None = None
    local_median_kernel: tuple[int, int, int] | None = (3, 3, 1)
    local_median_max_iterations: int | None = 2
    enabled: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FrameConfig":
        values = dict(values)
        values["sample_runs"] = tuple(RunSpec.from_value(item) for item in values.get("sample_runs", ()))
        values["ob_runs"] = tuple(RunSpec.from_value(item) for item in values.get("ob_runs", ()))
        values["dc_runs"] = tuple(RunSpec.from_value(item) for item in values.get("dc_runs", ()))
        values["roi"] = RoiConfig(**values.get("roi", {}))
        if values.get("ob_roi") is not None:
            values["ob_roi"] = RoiConfig(**values["ob_roi"])
        if values.get("container_roi") is not None:
            values["container_roi"] = RoiConfig(**values["container_roi"])
        values["black_filter_background"] = BlackFilterBackgroundConfig.from_dict(
            values.get("black_filter_background")
        )
        values["measured_backgrounds"] = tuple(
            MeasuredBackgroundConfig.from_dict(item)
            for item in values.get("measured_backgrounds", ())
        )
        values["rebin"] = RebinConfig.from_dict(values.get("rebin"))
        if values.get("local_median_kernel") is not None:
            values["local_median_kernel"] = tuple(values["local_median_kernel"])
        return cls(**values)

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Every frame needs a non-empty name.")
        if not self.sample_runs:
            raise ValueError(f"{self.name}: at least one sample run is required.")
        if not self.ob_runs:
            raise ValueError(f"{self.name}: at least one OB run is required.")
        if self.distance_source_detector_m <= 0:
            raise ValueError(f"{self.name}: source-detector distance must be positive.")
        if self.manual_tof_bin_size_ns is not None and self.manual_tof_bin_size_ns <= 0:
            raise ValueError(f"{self.name}: manual TOF bin size must be positive.")
        enabled_backgrounds = [item for item in self.measured_backgrounds if item.enabled]
        for background in enabled_backgrounds:
            if not background.sample_runs or not background.ob_runs:
                raise ValueError(
                    f"{self.name}: enabled {background.key} correction requires sample and OB background runs."
                )
        if self.black_filter_background.enabled and enabled_backgrounds:
            raise ValueError(
                f"{self.name}: black-filter and measured-background corrections cannot be enabled together."
            )
        if self.dc_runs and (self.black_filter_background.enabled or enabled_backgrounds):
            raise ValueError(
                f"{self.name}: dark-current correction cannot be combined with the selected background correction."
            )
        if self.black_filter_background.enabled and not self.black_filter_background.shape_file.strip():
            raise ValueError(f"{self.name}: black-filter correction requires a background-shape CSV file.")
        if self.container_roi_file is not None and not self.container_roi_file.strip():
            raise ValueError(f"{self.name}: saved container ROI file path cannot be empty.")
        if self.local_median_kernel is not None:
            if len(self.local_median_kernel) != 3 or any(value <= 0 for value in self.local_median_kernel):
                raise ValueError(f"{self.name}: local median kernel must contain three positive integers.")
        if self.local_median_max_iterations is not None and self.local_median_max_iterations <= 0:
            raise ValueError(f"{self.name}: local median maximum iterations must be positive.")
        self.roi.validate()
        if self.ob_roi is not None:
            self.ob_roi.validate()
        if self.container_roi is not None:
            self.container_roi.validate()

    def effective_ob_roi(self) -> RoiConfig:
        return self.roi if self.ob_roi is None else self.ob_roi


@dataclass
class MultiFrameRecipe:
    working_dir: str
    frames: list[FrameConfig]
    output_root: str | None = None
    cache_dir: str | None = None
    instrument: str = "VENUS"
    export_mode: dict[str, bool] = field(
        default_factory=lambda: {
            "sample_stack": False,
            "ob_stack": False,
            "normalized_stack": True,
            "sample_integrated": False,
            "ob_integrated": False,
            "normalized_integrated": False,
            "combined_normalized_integrated": False,
            "x_axis": True,
        }
    )
    correct_chips_alignment: bool = False
    replace_ob_zeros_by_local_median: bool = False
    local_median_kernel: tuple[int, int, int] = (3, 3, 1)
    local_median_max_iterations: int = 2
    same_rois_all_frames: bool = False
    recipe_version: int = RECIPE_VERSION

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MultiFrameRecipe":
        values = dict(values)
        version = int(values.get("recipe_version", 1))
        if version != RECIPE_VERSION:
            raise ValueError(f"Unsupported recipe version {version}; expected {RECIPE_VERSION}.")
        values["frames"] = [FrameConfig.from_dict(frame) for frame in values.get("frames", [])]
        if "local_median_kernel" in values:
            values["local_median_kernel"] = tuple(values["local_median_kernel"])
        return cls(**values)

    @classmethod
    def load(cls, file_name: str | os.PathLike[str]) -> "MultiFrameRecipe":
        with Path(file_name).expanduser().open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, file_name: str | os.PathLike[str]) -> Path:
        output = Path(file_name).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
            stream.write("\n")
        return output


@dataclass(frozen=True)
class ResolvedRun:
    run_number: str
    data_path: Path
    nexus_path: Path | None


@dataclass
class NativeRunProfile:
    run_number: str
    tof_s: np.ndarray
    counts: np.ndarray
    variance: np.ndarray
    proton_charge_c: float | None
    detector_delay_us: float | None
    data_path: str
    nexus_path: str | None
    warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False


@dataclass
class NativeFrameProfile:
    name: str
    tof_s: np.ndarray
    lambda_a: np.ndarray
    energy_eV: np.ndarray
    sample_counts: np.ndarray
    sample_variance: np.ndarray
    ob_counts: np.ndarray
    ob_variance: np.ndarray
    transmission: np.ndarray
    uncertainty: np.ndarray
    sample_total_proton_charge_c: float | None
    ob_total_proton_charge_c: float | None
    warnings: list[str] = field(default_factory=list)
    corrections_applied: list[str] = field(default_factory=list)


@dataclass
class RebinnedFramePreview:
    name: str
    tof_s: np.ndarray
    lambda_a: np.ndarray
    energy_eV: np.ndarray
    sample_counts: np.ndarray
    sample_variance: np.ndarray
    ob_counts: np.ndarray
    ob_variance: np.ndarray
    transmission: np.ndarray
    uncertainty: np.ndarray
    source_frame_count: np.ndarray
    native: NativeFrameProfile


@dataclass(frozen=True)
class OverlapDiagnostics:
    reference_name: str
    comparison_name: str
    energy_min_eV: float
    energy_max_eV: float
    point_count: int
    comparison_over_reference: float
    scale_comparison_to_reference: float
    scale_uncertainty: float
    reduced_chi_square: float


def _path_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class RoiProfileCache:
    """Small persistent NPZ cache keyed by source files, ROI, and variance mode."""

    def __init__(self, cache_dir: str | os.PathLike[str] | None):
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.memory: dict[str, NativeRunProfile] = {}

    def key(
        self,
        run: ResolvedRun,
        roi: RoiConfig,
        use_experimental_uncertainties: bool,
        manual_tof_bin_size_ns: float | None,
    ) -> str:
        tiffs = retrieve_list_of_tif(str(run.data_path))
        spectra_files = sorted(run.data_path.glob("*_Spectra.txt"))
        shutter_files = sorted(run.data_path.glob("*_ShutterCount.txt"))
        payload = {
            "cache_schema": 1,
            "run_number": run.run_number,
            "roi": asdict(roi),
            "experimental_uncertainties": use_experimental_uncertainties,
            "manual_tof_bin_size_ns": manual_tof_bin_size_ns,
            "tiffs": [_path_signature(Path(path)) for path in tiffs],
            "spectra": [_path_signature(path) for path in spectra_files],
            "shutter": [_path_signature(path) for path in shutter_files],
            "nexus": _path_signature(run.nexus_path),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, key: str) -> NativeRunProfile | None:
        if key in self.memory:
            profile = self.memory[key]
            profile.cache_hit = True
            return profile
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{key}.npz"
        if not cache_file.exists():
            return None
        with np.load(cache_file, allow_pickle=False) as payload:
            profile = NativeRunProfile(
                run_number=str(payload["run_number"].item()),
                tof_s=payload["tof_s"],
                counts=payload["counts"],
                variance=payload["variance"],
                proton_charge_c=_optional_float(payload["proton_charge_c"].item()),
                detector_delay_us=_optional_float(payload["detector_delay_us"].item()),
                data_path=str(payload["data_path"].item()),
                nexus_path=_optional_string(payload["nexus_path"].item()),
                warnings=list(json.loads(str(payload["warnings_json"].item()))),
                cache_hit=True,
            )
        self.memory[key] = profile
        return profile

    def put(self, key: str, profile: NativeRunProfile) -> None:
        self.memory[key] = profile
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        output = self.cache_dir / f"{key}.npz"
        temporary = output.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                run_number=profile.run_number,
                tof_s=profile.tof_s,
                counts=profile.counts,
                variance=profile.variance,
                proton_charge_c=np.nan if profile.proton_charge_c is None else profile.proton_charge_c,
                detector_delay_us=np.nan if profile.detector_delay_us is None else profile.detector_delay_us,
                data_path=profile.data_path,
                nexus_path="" if profile.nexus_path is None else profile.nexus_path,
                warnings_json=json.dumps(profile.warnings),
            )
        temporary.replace(output)


def _optional_float(value: Any) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else value


def _optional_string(value: Any) -> str | None:
    value = str(value)
    return value or None


class MultiFramePreviewEngine:
    def __init__(self, recipe: MultiFrameRecipe):
        self.recipe = recipe
        default_cache = Path(recipe.working_dir) / "shared" / ".normalization_tof_multiple_frames_cache"
        self.cache = RoiProfileCache(recipe.cache_dir or default_cache)
        self.native_profiles: dict[str, NativeFrameProfile] = {}
        self.previews: dict[str, RebinnedFramePreview] = {}

    @property
    def working_dir(self) -> Path:
        return Path(self.recipe.working_dir).expanduser()

    def resolve_run(self, spec: RunSpec, detector_type: str) -> ResolvedRun:
        run_number = _clean_run_number(spec.run_number)
        nexus_path = Path(spec.nexus_path).expanduser() if spec.nexus_path else (
            self.working_dir / "nexus" / f"{self.recipe.instrument.upper()}_{run_number}.nxs.h5"
        )
        if not nexus_path.exists():
            nexus_path = None

        if spec.data_path:
            data_path = Path(spec.data_path).expanduser()
        elif detector_type == DetectorType.tpx1_legacy:
            base = (
                Path(autoreduce_dir[self.recipe.instrument][detector_type][0])
                / self.working_dir.name
                / autoreduce_dir[self.recipe.instrument][detector_type][1]
            )
            data_path = base / f"Run_{run_number}"
        else:
            if nexus_path is None:
                raise FileNotFoundError(
                    f"Run {run_number}: NeXus file is required to locate {detector_type} image data."
                )
            relative_data_path = extract_file_path_from_nexus(nexus_path)
            if detector_type == DetectorType.tpx1:
                autoreduce_base = (
                    Path(autoreduce_dir[self.recipe.instrument][detector_type][0])
                    / self.working_dir.name
                    / autoreduce_dir[self.recipe.instrument][detector_type][1]
                )
                data_path = autoreduce_base.parent.parent / relative_data_path
            elif detector_type == DetectorType.tpx3:
                raw_base = Path(raw_dir[self.recipe.instrument][detector_type][0]) / self.working_dir.name
                data_path = raw_base / relative_data_path
            else:
                raise ValueError(f"Unsupported detector type: {detector_type}")

        if not data_path.is_dir():
            raise FileNotFoundError(f"Run {run_number}: image directory not found: {data_path}")
        return ResolvedRun(run_number=run_number, data_path=data_path, nexus_path=nexus_path)

    def load_frame(self, frame: FrameConfig, force_reload: bool = False) -> NativeFrameProfile:
        frame.validate()
        sample_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.sample_runs]
        ob_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.ob_runs]
        sample_profiles = [self._load_run(run, frame, frame.roi, force_reload) for run in sample_runs]
        ob_profiles = [
            self._load_run(run, frame, frame.effective_ob_roi(), force_reload) for run in ob_runs
        ]

        sample_tof, sample_counts, sample_variance, sample_charge, sample_warnings = self._combine_runs(
            sample_profiles, frame.use_proton_charge, role="sample"
        )
        if len(sample_profiles) > 1 and not frame.combine_sample_runs:
            sample_warnings.append(
                "The fast ROI preview combines the selected sample runs; full normalization will "
                "process them separately unless 'Combine sample runs' is enabled."
            )
        ob_tof, ob_counts, ob_variance, ob_charge, ob_warnings = self._combine_runs(
            ob_profiles, frame.use_proton_charge, role="OB"
        )
        common_length = min(len(sample_tof), len(ob_tof))
        if common_length == 0:
            raise ValueError(f"{frame.name}: sample or OB profile is empty.")
        sample_tof = sample_tof[:common_length]
        ob_tof = ob_tof[:common_length]
        if not np.allclose(sample_tof, ob_tof, rtol=1e-7, atol=1e-12):
            raise ValueError(f"{frame.name}: sample and OB TOF axes do not match.")

        detector_delays = [
            profile.detector_delay_us
            for profile in sample_profiles + ob_profiles
            if profile.detector_delay_us is not None
        ]
        detector_delay_us = frame.detector_delay_us
        if detector_delay_us is None:
            detector_delay_us = detector_delays[0] if detector_delays else 0.0
        if detector_delays and not np.allclose(detector_delays, detector_delay_us, rtol=0, atol=1e-6):
            sample_warnings.append(
                f"Detector delays differ across runs; preview uses {detector_delay_us:g} us."
            )

        tof_s, lambda_a, energy_eV = calculate_time_lambda_energy_arrays(
            time_spectra=sample_tof,
            distance_source_detector_m=frame.distance_source_detector_m,
            detector_delay_us=detector_delay_us,
        )
        sample_counts = np.asarray(sample_counts[:common_length], dtype=np.float64).copy()
        sample_variance = np.asarray(sample_variance[:common_length], dtype=np.float64).copy()
        ob_counts = np.asarray(ob_counts[:common_length], dtype=np.float64).copy()
        ob_variance = np.asarray(ob_variance[:common_length], dtype=np.float64).copy()
        corrections_applied: list[str] = []

        if frame.dc_runs:
            dc_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.dc_runs]
            sample_dc_profiles = [
                self._load_run(run, frame, frame.roi, force_reload) for run in dc_runs
            ]
            ob_dc_profiles = [
                self._load_run(run, frame, frame.effective_ob_roi(), force_reload)
                for run in dc_runs
            ]
            dc_tof, sample_dc_counts, sample_dc_variance, _, dc_warnings = self._combine_runs(
                sample_dc_profiles,
                use_proton_charge=False,
                role="sample dark current",
            )
            ob_dc_tof, ob_dc_counts, ob_dc_variance, _, ob_dc_warnings = self._combine_runs(
                ob_dc_profiles,
                use_proton_charge=False,
                role="OB dark current",
            )
            _require_matching_axis(sample_tof, dc_tof, f"{frame.name} dark current")
            _require_matching_axis(sample_tof, ob_dc_tof, f"{frame.name} OB dark current")
            sample_counts -= sample_dc_counts
            ob_counts -= ob_dc_counts
            sample_variance += sample_dc_variance
            ob_variance += ob_dc_variance
            sample_warnings.extend(dc_warnings + ob_dc_warnings)
            corrections_applied.append("dark current")

        for background in frame.measured_backgrounds:
            if not background.enabled:
                continue
            sample_background_runs = [
                self.resolve_run(spec, frame.detector_type) for spec in background.sample_runs
            ]
            ob_background_runs = [
                self.resolve_run(spec, frame.detector_type) for spec in background.ob_runs
            ]
            sample_background_profiles = [
                self._load_run(run, frame, frame.roi, force_reload)
                for run in sample_background_runs
            ]
            ob_background_profiles = [
                self._load_run(run, frame, frame.effective_ob_roi(), force_reload)
                for run in ob_background_runs
            ]
            (
                sample_background_tof,
                sample_background_counts,
                sample_background_variance,
                _,
                sample_background_warnings,
            ) = self._combine_runs(
                sample_background_profiles,
                frame.use_proton_charge,
                role=f"{background.key} sample background",
            )
            (
                ob_background_tof,
                ob_background_counts,
                ob_background_variance,
                _,
                ob_background_warnings,
            ) = self._combine_runs(
                ob_background_profiles,
                frame.use_proton_charge,
                role=f"{background.key} OB background",
            )
            _require_matching_axis(sample_tof, sample_background_tof, f"{frame.name} sample background")
            _require_matching_axis(sample_tof, ob_background_tof, f"{frame.name} OB background")
            weight = float(background.weight)
            sample_counts -= weight * sample_background_counts
            ob_counts -= weight * ob_background_counts
            sample_variance += weight**2 * sample_background_variance
            ob_variance += weight**2 * ob_background_variance
            sample_warnings.extend(sample_background_warnings + ob_background_warnings)
            _, label, _ = background.normalized_labels()
            corrections_applied.append(f"{label} measured background (weight {weight:g})")

        if frame.black_filter_background.enabled:
            sample_counts, ob_counts, black_filter_metadata = _apply_black_filter_background_to_profiles(
                sample_counts=sample_counts,
                ob_counts=ob_counts,
                energy_eV=energy_eV,
                config=frame.black_filter_background,
            )
            corrections_applied.append(
                "black-filter shape "
                f"(anchor {black_filter_metadata['anchor_energy_eV']:.6g} eV)"
            )

        if frame.container_roi is not None or frame.container_roi_file:
            sample_warnings.append(
                "Container correction is not applied to the fast ROI preview; "
                "it will be applied during full normalization."
            )

        transmission, uncertainty = _ratio_with_nan(
            sample_counts,
            ob_counts,
            sample_variance,
            ob_variance,
        )
        profile = NativeFrameProfile(
            name=frame.name,
            tof_s=tof_s,
            lambda_a=lambda_a,
            energy_eV=energy_eV,
            sample_counts=sample_counts,
            sample_variance=sample_variance,
            ob_counts=ob_counts,
            ob_variance=ob_variance,
            transmission=transmission,
            uncertainty=uncertainty,
            sample_total_proton_charge_c=sample_charge,
            ob_total_proton_charge_c=ob_charge,
            warnings=sample_warnings + ob_warnings,
            corrections_applied=corrections_applied,
        )
        self.native_profiles[frame.name] = profile
        return profile

    def rebin_frame(self, frame: FrameConfig, native: NativeFrameProfile | None = None) -> RebinnedFramePreview:
        native = native or self.native_profiles.get(frame.name) or self.load_frame(frame)
        groups, _ = build_rebin_bin_groups(
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
        active_groups = metadata["list_file_index_array"]
        sample_counts = _sum_groups(native.sample_counts, active_groups)
        sample_variance = _sum_groups(native.sample_variance, active_groups)
        ob_counts = _sum_groups(native.ob_counts, active_groups)
        ob_variance = _sum_groups(native.ob_variance, active_groups)
        transmission, uncertainty = _ratio_with_nan(
            sample_counts, ob_counts, sample_variance, ob_variance
        )
        preview = RebinnedFramePreview(
            name=frame.name,
            tof_s=metadata["mean_tof_array"],
            lambda_a=metadata["mean_lambda_array"],
            energy_eV=metadata["mean_energy_array"],
            sample_counts=sample_counts,
            sample_variance=sample_variance,
            ob_counts=ob_counts,
            ob_variance=ob_variance,
            transmission=transmission,
            uncertainty=uncertainty,
            source_frame_count=metadata["source_frame_count_array"],
            native=native,
        )
        self.previews[frame.name] = preview
        return preview

    def preview_all(self, force_reload: bool = False) -> dict[str, RebinnedFramePreview]:
        output = {}
        for frame in self.recipe.frames:
            if not frame.enabled:
                continue
            native = self.load_frame(frame, force_reload=force_reload)
            output[frame.name] = self.rebin_frame(frame, native=native)
        return output

    def overlap_diagnostics(
        self,
        reference_name: str,
        comparison_name: str,
        energy_window_eV: tuple[float, float] | None = None,
    ) -> OverlapDiagnostics:
        reference = self.previews[reference_name]
        comparison = self.previews[comparison_name]
        return calculate_overlap_diagnostics(reference, comparison, energy_window_eV)

    def run_full_normalization(self, campaign_label: str | None = None, preview: bool = True) -> Path:
        output_root = Path(self.recipe.output_root or (self.working_dir / "shared")).expanduser()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = _safe_name(campaign_label or "normalization_tof_multiple_frames")
        campaign_dir = output_root / f"{label}_{timestamp}"
        campaign_dir.mkdir(parents=True, exist_ok=False)
        self.recipe.save(campaign_dir / "normalization_tof_multiple_frames_recipe.json")

        for frame in self.recipe.frames:
            if not frame.enabled:
                continue
            frame.validate()
            frame_output = campaign_dir / _safe_name(frame.name)
            frame_output.mkdir(parents=True, exist_ok=False)
            sample_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.sample_runs]
            ob_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.ob_runs]
            dc_runs = [self.resolve_run(spec, frame.detector_type) for spec in frame.dc_runs]
            sample_dict = _normalization_input_dict(sample_runs)
            ob_dict = _normalization_input_dict(ob_runs)
            dc_dict = _normalization_input_dict(dc_runs)
            measured_background_configs = []
            all_background_runs: list[ResolvedRun] = []
            for background in frame.measured_backgrounds:
                if not background.enabled:
                    continue
                sample_background_runs = [
                    self.resolve_run(spec, frame.detector_type) for spec in background.sample_runs
                ]
                ob_background_runs = [
                    self.resolve_run(spec, frame.detector_type) for spec in background.ob_runs
                ]
                all_background_runs.extend(sample_background_runs + ob_background_runs)
                mode, column_label, key_prefix = background.normalized_labels()
                measured_background_configs.append(
                    {
                        "enabled": True,
                        "mode": mode,
                        "column_label": column_label,
                        "key_prefix": key_prefix,
                        "weight": float(background.weight),
                        "sample_background_dict": _normalization_input_dict(sample_background_runs),
                        "ob_background_dict": _normalization_input_dict(ob_background_runs),
                    }
                )
            spectra_array = self._manual_axis_for_frame(
                frame,
                sample_runs + ob_runs + dc_runs + all_background_runs,
            )
            detector_delay_us = frame.detector_delay_us
            correct_chips_alignment = (
                self.recipe.correct_chips_alignment
                if frame.correct_chips_alignment is None
                else frame.correct_chips_alignment
            )
            correct_config = None
            if correct_chips_alignment:
                correct_config = timepix3_config if frame.detector_type == DetectorType.tpx3 else timepix1_config
            replace_ob_zeros = (
                self.recipe.replace_ob_zeros_by_local_median
                if frame.replace_ob_zeros_by_local_median is None
                else frame.replace_ob_zeros_by_local_median
            )
            local_median_kernel = frame.local_median_kernel or self.recipe.local_median_kernel
            local_median_max_iterations = (
                frame.local_median_max_iterations or self.recipe.local_median_max_iterations
            )
            black_filter_config = None
            if frame.black_filter_background.enabled:
                black_filter_config = {
                    "enabled": True,
                    "background_shape_file": frame.black_filter_background.shape_file,
                    "anchor_energy_eV": frame.black_filter_background.anchor_energy_eV,
                }

            normalization_with_list_of_full_path(
                sample_dict=sample_dict,
                combine_samples=frame.combine_sample_runs,
                ob_dict=ob_dict,
                dc_dict=dc_dict,
                spectra_array=spectra_array,
                output_folder=str(frame_output),
                verbose=True,
                proton_charge_flag=frame.use_proton_charge,
                replace_ob_zeros_by_local_median_flag=replace_ob_zeros,
                kernel_size_for_local_median=local_median_kernel,
                max_iterations=local_median_max_iterations,
                output_tif=True,
                instrument=self.recipe.instrument,
                detector_delay_us=detector_delay_us,
                preview=preview,
                distance_source_detector_m=frame.distance_source_detector_m,
                correct_chips_alignment_flag=correct_chips_alignment,
                correct_chips_alignment_config=correct_config,
                export_mode=dict(self.recipe.export_mode),
                roi=frame.roi.to_roi(),
                sample_roi=frame.roi.to_roi(),
                ob_roi=frame.effective_ob_roi().to_roi(),
                container_roi=(None if frame.container_roi is None else frame.container_roi.to_roi()),
                container_roi_file=frame.container_roi_file,
                experimental_uncertainties_flag=frame.use_experimental_uncertainties,
                black_filter_background_config=black_filter_config,
                measured_background_correction_configs=measured_background_configs,
                **frame.rebin.engine_kwargs(),
            )
        return campaign_dir

    def _manual_axis_for_frame(self, frame: FrameConfig, runs: list[ResolvedRun]) -> np.ndarray | None:
        if frame.manual_tof_bin_size_ns is None:
            return None
        lengths = [len(retrieve_list_of_tif(str(run.data_path))) for run in runs]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{frame.name}: manual TOF axis requires equal TIFF counts in all sample, OB, "
                "dark-current, and measured-background runs."
            )
        return (np.arange(lengths[0], dtype=np.float64) + 0.5) * frame.manual_tof_bin_size_ns * 1e-9

    def _load_run(
        self,
        run: ResolvedRun,
        frame: FrameConfig,
        roi: RoiConfig,
        force_reload: bool,
    ) -> NativeRunProfile:
        cache_key = self.cache.key(
            run,
            roi,
            frame.use_experimental_uncertainties,
            frame.manual_tof_bin_size_ns,
        )
        if not force_reload:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        profile = load_native_roi_profile(
            run=run,
            roi=roi,
            use_experimental_uncertainties=frame.use_experimental_uncertainties,
            manual_tof_bin_size_ns=frame.manual_tof_bin_size_ns,
        )
        self.cache.put(cache_key, profile)
        return profile

    @staticmethod
    def _combine_runs(
        profiles: list[NativeRunProfile], use_proton_charge: bool, role: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, list[str]]:
        if not profiles:
            raise ValueError(f"No {role} profiles were supplied.")
        common_length = min(len(profile.tof_s) for profile in profiles)
        axis = profiles[0].tof_s[:common_length]
        warnings: list[str] = []
        for profile in profiles:
            if not np.allclose(axis, profile.tof_s[:common_length], rtol=1e-7, atol=1e-12):
                raise ValueError(f"{role} run {profile.run_number} has an incompatible TOF axis.")
            if len(profile.tof_s) != common_length:
                warnings.append(f"{role} runs have unequal profile lengths; preview uses {common_length} bins.")
            warnings.extend(profile.warnings)

        counts_sum = np.sum([profile.counts[:common_length] for profile in profiles], axis=0)
        variance_sum = np.sum([profile.variance[:common_length] for profile in profiles], axis=0)
        if use_proton_charge:
            charges = [profile.proton_charge_c for profile in profiles]
            if any(charge is None or charge <= 0 for charge in charges):
                missing = [profile.run_number for profile in profiles if not profile.proton_charge_c]
                raise ValueError(f"{role} proton charge is missing or invalid for runs: {', '.join(missing)}")
            total_charge = float(np.sum(charges))
            denominator = total_charge
        else:
            total_charge = None
            denominator = float(len(profiles))
        return axis, counts_sum / denominator, variance_sum / denominator**2, total_charge, warnings


def _normalization_input_dict(runs: list[ResolvedRun]) -> dict[str, dict[str, str | None]]:
    return {
        run.data_path.name: {
            "full_path": str(run.data_path),
            "nexus": None if run.nexus_path is None else str(run.nexus_path),
        }
        for run in runs
    }


def _require_matching_axis(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if len(reference) != len(candidate) or not np.allclose(reference, candidate, rtol=1e-7, atol=1e-12):
        raise ValueError(f"{label} TOF axis does not match the sample/OB frame axis.")


def _apply_black_filter_background_to_profiles(
    sample_counts: np.ndarray,
    ob_counts: np.ndarray,
    energy_eV: np.ndarray,
    config: BlackFilterBackgroundConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    shape = _load_black_filter_background_shape(config.shape_file)
    sample_shape, sample_in_range = _evaluate_black_filter_background_shape(
        energy_eV,
        shape["energy"],
        shape["sample"],
    )
    ob_shape, ob_in_range = _evaluate_black_filter_background_shape(
        energy_eV,
        shape["energy"],
        shape["ob"],
    )
    anchor_index = int(np.nanargmin(np.abs(np.asarray(energy_eV) - config.anchor_energy_eV)))
    if not (sample_in_range[anchor_index] and ob_in_range[anchor_index]):
        raise ValueError(
            "Black-filter background anchor is outside the fitted shape range for this frame: "
            f"requested {config.anchor_energy_eV:g} eV, nearest measured bin "
            f"{energy_eV[anchor_index]:g} eV."
        )
    sample_scale = float(sample_counts[anchor_index] / sample_shape[anchor_index])
    ob_scale = float(ob_counts[anchor_index] / ob_shape[anchor_index])
    corrected_sample = np.asarray(sample_counts, dtype=np.float64) - sample_scale * sample_shape
    corrected_ob = np.asarray(ob_counts, dtype=np.float64) - ob_scale * ob_shape
    return corrected_sample, corrected_ob, {
        "anchor_energy_eV": float(energy_eV[anchor_index]),
        "sample_scale": sample_scale,
        "ob_scale": ob_scale,
    }


def load_integrated_image_preview(
    data_path: str | os.PathLike[str],
    max_images: int = 200,
) -> tuple[np.ndarray, int, int]:
    """Integrate an evenly sampled subset of a run for interactive ROI selection."""
    tiffs = [Path(path) for path in retrieve_list_of_tif(str(data_path))]
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files found in {data_path}")
    max_images = max(1, int(max_images))
    if len(tiffs) > max_images:
        indices = np.unique(np.linspace(0, len(tiffs) - 1, max_images, dtype=int))
        selected = [tiffs[index] for index in indices]
    else:
        selected = tiffs

    def _load(path: Path) -> np.ndarray:
        return np.asarray(imread(path), dtype=np.float64).swapaxes(0, 1)

    integrated = None
    with ThreadPoolExecutor(max_workers=min(len(selected), os.cpu_count() or 1, 8)) as executor:
        for image in executor.map(_load, selected):
            if integrated is None:
                integrated = image
            else:
                integrated += image
    return integrated, len(selected), len(tiffs)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return cleaned or "frame"


def _sum_groups(values: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray([np.sum(array[group], dtype=np.float64) for group in groups], dtype=np.float64)


def _ratio_with_nan(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_variance: np.ndarray,
    denominator_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ratio, uncertainty = calculate_ratio_and_uncertainty(
        numerator,
        denominator,
        numerator_variance,
        denominator_variance,
    )
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & np.isfinite(numerator_variance)
        & np.isfinite(denominator_variance)
        & (numerator >= 0)
        & (denominator > 0)
    )
    ratio = np.where(valid, ratio, np.nan)
    uncertainty = np.where(valid, uncertainty, np.nan)
    return ratio, uncertainty


def _read_spectra_axis(data_path: Path, frame_count: int, manual_tof_bin_size_ns: float | None) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    spectra_files = sorted(data_path.glob("*_Spectra.txt"))
    if spectra_files:
        spectra = pd.read_csv(spectra_files[0], sep=",", header=0)
        column = "shutter_time" if "shutter_time" in spectra.columns else spectra.columns[0]
        tof = np.asarray(spectra[column], dtype=np.float64)
    elif manual_tof_bin_size_ns is not None:
        tof = (np.arange(frame_count, dtype=np.float64) + 0.5) * manual_tof_bin_size_ns * 1e-9
        warnings.append(f"No Spectra.txt file; generated {manual_tof_bin_size_ns:g} ns TOF-bin centers.")
    else:
        raise FileNotFoundError(
            f"No *_Spectra.txt file in {data_path}; set manual_tof_bin_size_ns for this frame."
        )

    if len(tof) != frame_count:
        common_length = min(len(tof), frame_count)
        warnings.append(
            f"Spectra axis has {len(tof)} entries but TIFF stack has {frame_count}; preview uses {common_length}."
        )
        tof = tof[:common_length]
    return tof, warnings


def _read_primary_shutter_count(data_path: Path) -> float | None:
    shutter_files = sorted(data_path.glob("*_ShutterCount.txt"))
    if not shutter_files:
        return None
    values: list[float] = []
    with shutter_files[0].open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            value = float(parts[-1])
            if value == 0:
                break
            if value > 0:
                values.append(value)
    return values[0] if values else None


def _read_proton_charge(nexus_path: Path | None) -> float | None:
    if nexus_path is None:
        return None
    try:
        with h5py.File(nexus_path, "r") as nexus:
            return float(nexus["entry"]["proton_charge"][0] / 1e12)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _read_detector_delay(nexus_path: Path | None) -> float | None:
    if nexus_path is None:
        return None
    try:
        value = get_detector_offset_from_nexus(str(nexus_path))
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return None if value is None else float(value)


def load_native_roi_profile(
    run: ResolvedRun,
    roi: RoiConfig,
    use_experimental_uncertainties: bool,
    manual_tof_bin_size_ns: float | None,
) -> NativeRunProfile:
    """Stream one run's ROI and calculate a count/variance profile per native frame."""
    roi.validate()
    tiffs = [Path(path) for path in retrieve_list_of_tif(str(run.data_path))]
    if not tiffs:
        raise FileNotFoundError(f"Run {run.run_number}: no TIFF files found in {run.data_path}")

    tof_s, warnings = _read_spectra_axis(run.data_path, len(tiffs), manual_tof_bin_size_ns)
    tiffs = tiffs[: len(tof_s)]
    shutter_count = _read_primary_shutter_count(run.data_path)
    if use_experimental_uncertainties and shutter_count is None:
        warnings.append("Shutter count not found; preview uncertainty falls back to Poisson counts.")

    counts = np.empty(len(tiffs), dtype=np.float64)
    variance = np.empty(len(tiffs), dtype=np.float64)
    cumulative_raw: np.ndarray | None = None
    y0, y1 = roi.top, roi.top + roi.height
    x0, x1 = roi.left, roi.left + roi.width
    for index, tiff in enumerate(tiffs):
        roi_image, image_shape = _read_tiff_roi(tiff, x0=x0, x1=x1, y0=y0, y1=y1)
        if y1 > image_shape[0] or x1 > image_shape[1]:
            raise ValueError(
                f"Run {run.run_number}: ROI {roi} is outside image shape {image_shape}."
            )
        counts[index] = np.sum(roi_image, dtype=np.float64)
        if use_experimental_uncertainties and shutter_count is not None:
            if cumulative_raw is None:
                cumulative_raw = np.zeros_like(roi_image, dtype=np.float64)
            numerator = roi_image * (shutter_count - cumulative_raw)
            denominator = shutter_count + roi_image
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_frame = np.where((denominator > 0) & (numerator >= 0), numerator / denominator, 0.0)
            cumulative_raw += raw_frame
            occupancy = cumulative_raw / shutter_count
            with np.errstate(divide="ignore", invalid="ignore"):
                pixel_variance = np.where(1.0 - occupancy > 0, roi_image / (1.0 - occupancy), 0.0)
            variance[index] = np.sum(np.maximum(pixel_variance, 0.0), dtype=np.float64)
        else:
            variance[index] = max(counts[index], 0.0)

    return NativeRunProfile(
        run_number=run.run_number,
        tof_s=tof_s,
        counts=counts,
        variance=variance,
        proton_charge_c=_read_proton_charge(run.nexus_path),
        detector_delay_us=_read_detector_delay(run.nexus_path),
        data_path=str(run.data_path),
        nexus_path=None if run.nexus_path is None else str(run.nexus_path),
        warnings=warnings,
    )


def _read_tiff_roi(
    path: Path,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Read only the requested ROI when the TIFF can be memory mapped."""
    try:
        mapped = tiff_memmap(path, mode="r")
        if mapped.ndim != 2:
            raise ValueError("Only 2D TIFFs use the memory-mapped ROI path.")
        image_shape = (int(mapped.shape[1]), int(mapped.shape[0]))
        # Production swaps the detector axes before applying the ROI. Selecting
        # the transposed coordinates directly avoids reading the full image.
        roi_image = np.array(mapped[x0:x1, y0:y1].T, dtype=np.float64, copy=True)
        del mapped
        return roi_image, image_shape
    except (OSError, TypeError, ValueError):
        image = np.asarray(imread(path), dtype=np.float64).swapaxes(0, 1)
        return image[y0:y1, x0:x1], tuple(int(value) for value in image.shape)


def calculate_overlap_diagnostics(
    reference: RebinnedFramePreview,
    comparison: RebinnedFramePreview,
    energy_window_eV: tuple[float, float] | None = None,
) -> OverlapDiagnostics:
    ref_e, ref_t, ref_u = _finite_sorted(reference.energy_eV, reference.transmission, reference.uncertainty)
    cmp_e, cmp_t, cmp_u = _finite_sorted(comparison.energy_eV, comparison.transmission, comparison.uncertainty)
    overlap_min = max(float(np.min(ref_e)), float(np.min(cmp_e)))
    overlap_max = min(float(np.max(ref_e)), float(np.max(cmp_e)))
    if energy_window_eV is not None:
        overlap_min = max(overlap_min, float(min(energy_window_eV)))
        overlap_max = min(overlap_max, float(max(energy_window_eV)))
    if overlap_min >= overlap_max:
        raise ValueError(f"{reference.name} and {comparison.name} have no common energy coverage.")

    mask = (cmp_e >= overlap_min) & (cmp_e <= overlap_max)
    cmp_e, cmp_t, cmp_u = cmp_e[mask], cmp_t[mask], cmp_u[mask]
    interp_ref = np.interp(cmp_e, ref_e, ref_t)
    interp_ref_u = np.interp(cmp_e, ref_e, ref_u)
    valid = (
        np.isfinite(cmp_t)
        & np.isfinite(cmp_u)
        & np.isfinite(interp_ref)
        & np.isfinite(interp_ref_u)
        & (cmp_t > 0)
        & (interp_ref > 0)
    )
    cmp_t, cmp_u, interp_ref, interp_ref_u = (
        values[valid] for values in (cmp_t, cmp_u, interp_ref, interp_ref_u)
    )
    if len(cmp_t) < 2:
        raise ValueError("Overlap needs at least two finite comparison points.")

    ratio = cmp_t / interp_ref
    ratio_variance = (cmp_u / interp_ref) ** 2 + ((cmp_t * interp_ref_u) / interp_ref**2) ** 2
    positive_variance = np.isfinite(ratio_variance) & (ratio_variance > 0)
    if np.any(positive_variance):
        weights = 1.0 / ratio_variance[positive_variance]
        mean_ratio = float(np.sum(weights * ratio[positive_variance]) / np.sum(weights))
        ratio_uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
        chi_square = float(np.sum(weights * (ratio[positive_variance] - mean_ratio) ** 2))
        reduced_chi_square = chi_square / max(int(np.sum(positive_variance)) - 1, 1)
    else:
        mean_ratio = float(np.mean(ratio))
        ratio_uncertainty = float(np.std(ratio, ddof=1) / np.sqrt(len(ratio)))
        reduced_chi_square = float("nan")

    scale = 1.0 / mean_ratio
    scale_uncertainty = ratio_uncertainty / mean_ratio**2
    return OverlapDiagnostics(
        reference_name=reference.name,
        comparison_name=comparison.name,
        energy_min_eV=overlap_min,
        energy_max_eV=overlap_max,
        point_count=len(cmp_t),
        comparison_over_reference=mean_ratio,
        scale_comparison_to_reference=scale,
        scale_uncertainty=scale_uncertainty,
        reduced_chi_square=reduced_chi_square,
    )


def overlap_ratio_arrays(
    reference: RebinnedFramePreview,
    comparison: RebinnedFramePreview,
    energy_window_eV: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_e, ref_t, ref_u = _finite_sorted(reference.energy_eV, reference.transmission, reference.uncertainty)
    cmp_e, cmp_t, cmp_u = _finite_sorted(comparison.energy_eV, comparison.transmission, comparison.uncertainty)
    lower = max(float(np.min(ref_e)), float(np.min(cmp_e)))
    upper = min(float(np.max(ref_e)), float(np.max(cmp_e)))
    if energy_window_eV is not None:
        lower = max(lower, float(min(energy_window_eV)))
        upper = min(upper, float(max(energy_window_eV)))
    mask = (cmp_e >= lower) & (cmp_e <= upper)
    energy = cmp_e[mask]
    cmp_t, cmp_u = cmp_t[mask], cmp_u[mask]
    ref_t_i = np.interp(energy, ref_e, ref_t)
    ref_u_i = np.interp(energy, ref_e, ref_u)
    ratio = cmp_t / ref_t_i
    uncertainty = np.sqrt((cmp_u / ref_t_i) ** 2 + ((cmp_t * ref_u_i) / ref_t_i**2) ** 2)
    return energy, ratio, uncertainty


def _finite_sorted(
    energy: np.ndarray, values: np.ndarray, uncertainty: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.asarray(energy, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    mask = np.isfinite(energy) & np.isfinite(values) & np.isfinite(uncertainty) & (energy > 0)
    order = np.argsort(energy[mask])
    return energy[mask][order], values[mask][order], uncertainty[mask][order]
