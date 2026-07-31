import numpy as np
import pytest

from crosssimcam import ACAMModel


@pytest.mark.parametrize("model_name", ACAMModel.supported_models)
def test_bundled_model_data_loads(model_name: str) -> None:
    model = ACAMModel(model_name)
    assert model.lut.ranges["DL"][0] < model.lut.ranges["DL"][1]


def test_simulation_shapes_and_ranges() -> None:
    model = ACAMModel("6T2M", max_steps=5)
    result = model.simulate(
        np.full((3, 2), 0.65),
        np.full((2, 4), 2e-6),
        np.full((2, 4), 8e-6),
    )
    assert result.vml_over_time.shape == (3, 6, 4)
    assert result.matches.shape == (3, 4)
    assert np.all((result.vml >= 0) & (result.vml <= 1))
    assert np.all(np.diff(result.time) > 0)


def test_input_shape_validation() -> None:
    model = ACAMModel("6T2M")
    with pytest.raises(ValueError, match="same shape"):
        model.simulate(np.ones((1, 2)), np.ones((2, 1)), np.ones((2, 2)))


def test_normalized_encoding_is_finite_and_in_range() -> None:
    model = ACAMModel("6T2M")
    model.calibrate_normalized_bounds(resolution=64)
    dl, lb, hb = model.encode_normalized(
        np.array([[0.25, 0.75]]), np.full((2, 1), 0.2), np.full((2, 1), 0.8)
    )
    assert dl.shape == (1, 2)
    assert lb.shape == hb.shape == (2, 1)
    assert np.isfinite(dl).all() and np.isfinite(lb).all() and np.isfinite(hb).all()
    assert np.all((dl >= model.lut.ranges["DL"][0]) & (dl <= model.lut.ranges["DL"][1]))
