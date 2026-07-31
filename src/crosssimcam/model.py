"""Transient analog CAM array model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from .lut import CAMLookupTables, InterpolationMethod

MODEL_DEFAULTS: dict[str, dict[str, float | int | None]] = {
    "6T2M": {"max_time": 7e-9, "max_steps": 2, "cell_capacitance": 10e-15, "tau_transistor": 1e-9},
    "10T2M": {"max_time": 4e-9, "max_steps": 2, "cell_capacitance": 10e-15, "tau_transistor": 1e-9},
    "SALMC": {"max_time": 11e-9, "max_steps": 2, "cell_capacitance": 3e-15, "tau_transistor": 1e-9},
}


@dataclass(frozen=True)
class SimulationResult:
    """Outputs of one batched transient simulation."""

    time: NDArray[np.float64]
    vml_over_time: NDArray[np.float64]
    vml: NDArray[np.float64]
    matches: NDArray[np.bool_]
    iml: NDArray[np.float64]
    vx_lb: NDArray[np.float64]
    vx_hb: NDArray[np.float64]


class ACAMModel:
    """Fast transient model of a memristive analog CAM array.

    Array shapes follow ``(batch, features)`` for input data and
    ``(features, words)`` for stored lower/upper conductances.
    """

    supported_models = tuple(MODEL_DEFAULTS)

    def __init__(
        self,
        model: str = "6T2M",
        *,
        max_time: float | None = None,
        max_steps: int | None = None,
        cell_capacitance: float | None = None,
        vml_initial: float = 1.0,
        vml_min: float = 1e-12,
        sense_threshold: float = 0.5,
        tau_transistor: float | None = None,
        interpolation: InterpolationMethod = "monotone",
        data_dir: str | Path | None = None,
    ) -> None:
        model = model.upper()
        if model not in MODEL_DEFAULTS:
            choices = ", ".join(MODEL_DEFAULTS)
            raise ValueError(f"unknown model {model!r}; choose one of {choices}")
        defaults = MODEL_DEFAULTS[model]
        self.model = model
        self.max_time = float(max_time if max_time is not None else defaults["max_time"])
        self.max_steps = int(max_steps if max_steps is not None else defaults["max_steps"])
        self.cell_capacitance = float(
            cell_capacitance if cell_capacitance is not None else defaults["cell_capacitance"]
        )
        self.vml_initial = float(vml_initial)
        self.vml_min = float(vml_min)
        self.sense_threshold = float(sense_threshold)
        self.tau_transistor = (
            tau_transistor if tau_transistor is not None else defaults["tau_transistor"]
        )
        self.interpolation = interpolation
        if self.max_time <= 0 or self.max_steps <= 0 or self.cell_capacitance <= 0:
            raise ValueError("max_time, max_steps, and cell_capacitance must be positive")
        if self.tau_transistor is not None and self.tau_transistor <= 0:
            raise ValueError("tau_transistor must be positive or None")

        root = Path(data_dir) if data_dir is not None else Path(str(files("crosssimcam") / "data"))
        self.lut = CAMLookupTables(
            root / f"{model}_VXLBandHB.npz", root / f"{model}_IML.npz"
        )
        self._bound_curves: tuple[NDArray[np.float64], ...] | None = None

    def simulate(
        self,
        dl_voltage: ArrayLike,
        gmem_lb: ArrayLike,
        gmem_hb: ArrayLike,
        *,
        ignore_lb: bool = False,
        ignore_hb: bool = False,
        capacitance_length: int | None = None,
        dl_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> SimulationResult:
        """Simulate a batch of input vectors against stored CAM words."""
        dl = np.asarray(dl_voltage, dtype=float)
        lb = np.asarray(gmem_lb, dtype=float)
        hb = np.asarray(gmem_hb, dtype=float)
        if dl.ndim != 2 or lb.ndim != 2 or hb.ndim != 2:
            raise ValueError("dl_voltage, gmem_lb, and gmem_hb must all be 2-D arrays")
        if lb.shape != hb.shape:
            raise ValueError("gmem_lb and gmem_hb must have the same shape")
        batch, features = dl.shape
        if lb.shape[0] != features:
            raise ValueError("conductance rows must equal the number of input features")
        if capacitance_length is not None and capacitance_length <= 0:
            raise ValueError("capacitance_length must be positive")
        if dl_noise_std < 0:
            raise ValueError("dl_noise_std cannot be negative")
        if dl_noise_std:
            generator = rng if rng is not None else np.random.default_rng()
            dl = dl + generator.normal(0.0, dl_noise_std, dl.shape)
        dl = np.clip(dl, *self.lut.ranges["DL"])

        words = lb.shape[1]
        shape = (batch, features, words)
        dl_grid = np.broadcast_to(dl[:, :, None], shape)
        lb_grid = np.broadcast_to(lb[None, :, :], shape)
        hb_grid = np.broadcast_to(hb[None, :, :], shape)
        vx_lb = np.zeros(shape) if ignore_lb else self.lut.vx(
            dl_grid, lb_grid, "LB", method=self.interpolation, oob="clip"
        )
        vx_hb = np.zeros(shape) if ignore_hb else self.lut.vx(
            dl_grid, hb_grid, "HB", method=self.interpolation, oob="clip"
        )
        vx_lb = np.clip(vx_lb, *self.lut.ranges["VX"])
        vx_hb = np.clip(vx_hb, *self.lut.ranges["VX"])

        dt = self.max_time / self.max_steps
        time = np.linspace(0.0, self.max_time, self.max_steps + 1)
        vml = np.full((batch, words), self.vml_initial)
        history = np.empty((batch, self.max_steps + 1, words))
        history[:, 0, :] = vml
        transient_lb = np.zeros_like(vx_lb) if self.tau_transistor else vx_lb
        transient_hb = np.zeros_like(vx_hb) if self.tau_transistor else vx_hb
        length = features if capacitance_length is None else capacitance_length
        capacitance = self.cell_capacitance * length
        iml = np.zeros((batch, words))

        for step in range(self.max_steps):
            if self.tau_transistor:
                alpha = 1.0 - np.exp(-dt / self.tau_transistor)
                transient_lb += (vx_lb - transient_lb) * alpha
                transient_hb += (vx_hb - transient_hb) * alpha
            vml_grid = np.broadcast_to(vml[:, None, :], shape)
            iml = (
                self.lut.matchline_current(vml_grid, transient_lb)
                + self.lut.matchline_current(vml_grid, transient_hb)
            ).sum(axis=1)
            vml = np.clip(vml - (iml / capacitance) * dt, 0.0, self.vml_initial)
            history[:, step + 1, :] = vml

        return SimulationResult(
            time=time,
            vml_over_time=history,
            vml=vml,
            matches=self.lut.sense(vml, self.sense_threshold),
            iml=iml,
            vx_lb=vx_lb,
            vx_hb=vx_hb,
        )

    def calibrate_normalized_bounds(
        self, *, resolution: int = 256, capacitance_length: int | None = None
    ) -> None:
        """Calibrate conversion from normalized interval bounds to conductance."""
        if resolution < 16:
            raise ValueError("resolution must be at least 16")
        dl_axis = np.linspace(*self.lut.ranges["DL"], resolution)
        gmem_axis = np.linspace(*self.lut.ranges["GMEM"], resolution)
        dl = dl_axis[:, None]
        gmem = gmem_axis[None, :]
        gmin, gmax = self.lut.ranges["GMEM"]
        lb_result = self.simulate(
            dl, gmem, np.full_like(gmem, gmax), ignore_hb=True,
            capacitance_length=capacitance_length,
        ).matches
        hb_result = self.simulate(
            dl, np.full_like(gmem, gmin), gmem, ignore_lb=True,
            capacitance_length=capacitance_length,
        ).matches
        lb_voltage = self._first_crossing(dl_axis, lb_result, True)
        hb_voltage = self._first_crossing(dl_axis, hb_result, False)
        self._bound_curves = (lb_voltage, gmem_axis, hb_voltage, gmem_axis.copy())

    @staticmethod
    def _first_crossing(
        dl_axis: NDArray[np.float64], states: NDArray[np.bool_], target: bool
    ) -> NDArray[np.float64]:
        mask = states == target
        if np.any(~mask.any(axis=0)):
            raise ValueError(
                "bound calibration failed: the model does not cross the sense threshold; "
                "adjust the transient parameters"
            )
        return dl_axis[np.argmax(mask, axis=0)]

    def encode_normalized(
        self,
        data: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        smooth: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Convert normalized inputs and interval bounds in ``[0, 1]`` to physical values."""
        if self._bound_curves is None:
            self.calibrate_normalized_bounds()
        assert self._bound_curves is not None
        lb_voltage, lb_gmem, hb_voltage, hb_gmem = self._bound_curves
        lb_curve = self._prepare_curve(lb_voltage, lb_gmem, smooth)
        hb_curve = self._prepare_curve(hb_voltage, hb_gmem, smooth)
        voltage_min = float(hb_curve[0][0])
        voltage_max = float(lb_curve[0][-1])
        if voltage_min >= voltage_max:
            voltage_min = min(float(lb_curve[0][0]), float(hb_curve[0][0]))
            voltage_max = max(float(lb_curve[0][-1]), float(hb_curve[0][-1]))
        data_array = np.clip(np.asarray(data, dtype=float), 0.0, 1.0)
        lower_array = np.clip(np.asarray(lower, dtype=float), 0.0, 1.0)
        upper_array = np.clip(np.asarray(upper, dtype=float), 0.0, 1.0)
        lower_array, upper_array = np.broadcast_arrays(lower_array, upper_array)
        dl = voltage_min + data_array * (voltage_max - voltage_min)
        lb_query = voltage_min + lower_array * (voltage_max - voltage_min)
        hb_query = voltage_min + upper_array * (voltage_max - voltage_min)
        lb_interp = PchipInterpolator(*lb_curve, extrapolate=True)
        hb_interp = PchipInterpolator(*hb_curve, extrapolate=True)
        gmin, gmax = self.lut.ranges["GMEM"]
        return dl, np.clip(lb_interp(lb_query), gmin, gmax), np.clip(hb_interp(hb_query), gmin, gmax)

    @staticmethod
    def _prepare_curve(
        voltage: NDArray[np.float64], gmem: NDArray[np.float64], smooth: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        order = np.argsort(voltage)
        voltage, gmem = voltage[order], gmem[order]
        unique, inverse = np.unique(voltage, return_inverse=True)
        averaged = np.zeros_like(unique)
        counts = np.zeros_like(unique)
        np.add.at(averaged, inverse, gmem)
        np.add.at(counts, inverse, 1)
        averaged /= counts
        if smooth and averaged.size >= 7:
            window = min(21, averaged.size if averaged.size % 2 else averaged.size - 1)
            averaged = savgol_filter(averaged, window, min(2, window - 1), mode="interp")
        if unique.size < 2:
            raise ValueError("calibration produced too few distinct voltage points")
        return unique, averaged
