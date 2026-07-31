import numpy as np
import pytest

from crosssimcam import ACAMModel


def test_lut_preserves_broadcast_shape() -> None:
    lut = ACAMModel("6T2M").lut
    values = lut.vx(np.array([[0.5], [0.7]]), np.array([2e-6, 5e-6, 8e-6]))
    assert values.shape == (2, 3)
    assert np.isfinite(values).all()


def test_lut_out_of_bounds_policies() -> None:
    lut = ACAMModel("6T2M").lut
    assert np.isnan(lut.vx(0.0, 5e-6, oob="nan"))
    assert np.isfinite(lut.vx(0.0, 5e-6, oob="clip"))
    with pytest.raises(ValueError, match="outside"):
        lut.vx(0.0, 5e-6, oob="error")

