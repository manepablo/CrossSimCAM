"""Interpolation of SPICE-derived aCAM lookup tables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator, RectBivariateSpline, RegularGridInterpolator

InterpolationMethod = Literal["linear", "spline", "monotone"]
OutOfBoundsPolicy = Literal["clip", "error", "nan", "extrapolate"]


class CAMLookupTables:
    """Load and interpolate the ``VX(DL, Gmem)`` and ``IML(VML, VX)`` tables."""

    def __init__(self, vx_file: str | Path, iml_file: str | Path) -> None:
        with np.load(vx_file) as vx_data:
            dl = np.asarray(vx_data["DL"], dtype=float)
            gmem = np.asarray(vx_data["GMEM"], dtype=float)
            vx_lb = np.asarray(vx_data["VX_LB"], dtype=float)
            vx_hb = np.asarray(vx_data["VX_HB"], dtype=float)

        with np.load(iml_file) as iml_data:
            vml = np.asarray(iml_data["VD"], dtype=float)
            vx = np.asarray(iml_data["VG"], dtype=float)
            iml = np.asarray(iml_data["IML"], dtype=float)

        self.dl_axis, self.gmem_axis, self.vx_lb = self._sort_table(dl, gmem, vx_lb)
        _, _, self.vx_hb = self._sort_table(dl, gmem, vx_hb)
        self.vml_axis, self.vx_axis, self.iml = self._sort_table(vml, vx, iml)

        self._vx_linear = {
            "LB": RegularGridInterpolator(
                (self.dl_axis, self.gmem_axis), self.vx_lb,
                bounds_error=False, fill_value=np.nan,
            ),
            "HB": RegularGridInterpolator(
                (self.dl_axis, self.gmem_axis), self.vx_hb,
                bounds_error=False, fill_value=np.nan,
            ),
        }
        self._iml_linear = RegularGridInterpolator(
            (self.vml_axis, self.vx_axis), self.iml,
            bounds_error=False, fill_value=np.nan,
        )
        self._vx_splines: dict[str, RectBivariateSpline] = {}
        self._iml_spline: RectBivariateSpline | None = None
        self._vx_pchip = {
            bound: [
                PchipInterpolator(self.dl_axis, table[:, column], extrapolate=True)
                for column in range(self.gmem_axis.size)
            ]
            for bound, table in (("LB", self.vx_lb), ("HB", self.vx_hb))
        }

    @staticmethod
    def _sort_table(
        x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        ix, iy = np.argsort(x), np.argsort(y)
        x_sorted, y_sorted = x[ix], y[iy]
        if np.any(np.diff(x_sorted) <= 0) or np.any(np.diff(y_sorted) <= 0):
            raise ValueError("LUT axes must be strictly increasing and contain no duplicates")
        expected = (x.size, y.size)
        if z.shape != expected:
            raise ValueError(f"LUT table has shape {z.shape}; expected {expected}")
        return x_sorted, y_sorted, z[np.ix_(ix, iy)]

    @property
    def ranges(self) -> dict[str, tuple[float, float]]:
        """Valid input ranges for each physical quantity."""
        return {
            "DL": (float(self.dl_axis[0]), float(self.dl_axis[-1])),
            "GMEM": (float(self.gmem_axis[0]), float(self.gmem_axis[-1])),
            "VML": (float(self.vml_axis[0]), float(self.vml_axis[-1])),
            "VX": (float(self.vx_axis[0]), float(self.vx_axis[-1])),
        }

    @staticmethod
    def _points(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], tuple[int, ...]]:
        a_array, b_array = np.broadcast_arrays(
            np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        )
        return np.column_stack((a_array.ravel(), b_array.ravel())), a_array.shape

    @staticmethod
    def _apply_oob(
        points: NDArray[np.float64],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        policy: OutOfBoundsPolicy,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        outside = (
            (points[:, 0] < x_range[0]) | (points[:, 0] > x_range[1])
            | (points[:, 1] < y_range[0]) | (points[:, 1] > y_range[1])
        )
        if policy == "error" and np.any(outside):
            raise ValueError("LUT query lies outside the valid range")
        if policy == "clip":
            points = points.copy()
            points[:, 0] = np.clip(points[:, 0], *x_range)
            points[:, 1] = np.clip(points[:, 1], *y_range)
        return points, outside

    def vx(
        self,
        dl: ArrayLike,
        gmem: ArrayLike,
        bound: Literal["LB", "HB"] = "HB",
        *,
        method: InterpolationMethod = "monotone",
        oob: OutOfBoundsPolicy = "clip",
    ) -> NDArray[np.float64]:
        """Evaluate the internal node voltage for a lower or upper-bound branch."""
        bound = bound.upper()  # type: ignore[assignment]
        if bound not in {"LB", "HB"}:
            raise ValueError("bound must be 'LB' or 'HB'")
        if method not in {"linear", "spline", "monotone"}:
            raise ValueError("method must be 'linear', 'spline', or 'monotone'")
        if oob not in {"clip", "error", "nan", "extrapolate"}:
            raise ValueError("unsupported out-of-bounds policy")

        points, shape = self._points(dl, gmem)
        original = points.copy()
        points, outside = self._apply_oob(
            points, self.ranges["DL"], self.ranges["GMEM"], oob
        )

        if method == "linear":
            if oob == "extrapolate":
                table = self.vx_lb if bound == "LB" else self.vx_hb
                interpolator = RegularGridInterpolator(
                    (self.dl_axis, self.gmem_axis), table,
                    bounds_error=False, fill_value=None,
                )
                values = interpolator(points)
            else:
                values = self._vx_linear[bound](points)
        elif method == "spline":
            if bound not in self._vx_splines:
                table = self.vx_lb if bound == "LB" else self.vx_hb
                self._vx_splines[bound] = RectBivariateSpline(
                    self.dl_axis, self.gmem_axis, table, kx=3, ky=3
                )
            if oob == "extrapolate":
                warnings.warn("spline extrapolation can overshoot the LUT", RuntimeWarning)
            values = self._vx_splines[bound].ev(points[:, 0], points[:, 1])
        else:
            dl_query, gmem_query = points[:, 0], points[:, 1]
            column = np.searchsorted(self.gmem_axis, gmem_query, side="right") - 1
            column = np.clip(column, 0, self.gmem_axis.size - 2)
            g0, g1 = self.gmem_axis[column], self.gmem_axis[column + 1]
            weight = (gmem_query - g0) / (g1 - g0)
            low = np.empty_like(dl_query)
            high = np.empty_like(dl_query)
            interpolators = self._vx_pchip[bound]
            for index in np.unique(column):
                mask = column == index
                low[mask] = interpolators[index](dl_query[mask])
                high[mask] = interpolators[index + 1](dl_query[mask])
            values = (1.0 - weight) * low + weight * high

        if oob == "nan":
            values[outside] = np.nan
        elif oob == "extrapolate" and method == "monotone":
            # PCHIP handles DL extrapolation; linear interpolation handles Gmem.
            _ = original
        return values.reshape(shape)

    def matchline_current(
        self,
        vml: ArrayLike,
        vx: ArrayLike,
        *,
        method: Literal["linear", "spline"] = "linear",
        oob: OutOfBoundsPolicy = "clip",
    ) -> NDArray[np.float64]:
        """Evaluate the match-line discharge current ``IML(VML, VX)``."""
        points, shape = self._points(vml, vx)
        points, outside = self._apply_oob(
            points, self.ranges["VML"], self.ranges["VX"], oob
        )
        if method == "linear":
            if oob == "extrapolate":
                interpolator = RegularGridInterpolator(
                    (self.vml_axis, self.vx_axis), self.iml,
                    bounds_error=False, fill_value=None,
                )
                values = interpolator(points)
            else:
                values = self._iml_linear(points)
        elif method == "spline":
            if self._iml_spline is None:
                self._iml_spline = RectBivariateSpline(
                    self.vml_axis, self.vx_axis, self.iml, kx=3, ky=3
                )
            values = self._iml_spline.ev(points[:, 0], points[:, 1])
        else:
            raise ValueError("method must be 'linear' or 'spline'")
        if oob == "nan":
            values[outside] = np.nan
        return values.reshape(shape)

    @staticmethod
    def sense(vml: ArrayLike, threshold: float = 0.5) -> NDArray[np.bool_]:
        """Digitize match-line voltages using a sense-amplifier threshold."""
        return np.asarray(vml, dtype=float) > threshold

