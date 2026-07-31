"""Visualize the joint response of a two-feature match line."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crosssimcam import ACAMModel


def main() -> None:
    resolution = 40
    axis = np.linspace(0.0, 1.0, resolution)
    x0, x1 = np.meshgrid(axis, axis, indexing="xy")
    inputs = np.column_stack((x0.ravel(), x1.ravel()))
    lower = np.full((2, 1), 0.20)
    upper = np.full((2, 1), 0.80)

    model = ACAMModel("6T2M")
    dl, gmem_lb, gmem_hb = model.encode_normalized(inputs, lower, upper)
    result = model.simulate(dl, gmem_lb, gmem_hb)
    voltage = result.vml[:, 0].reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.pcolormesh(x0, x1, voltage, shading="auto", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Final match-line voltage [V]")
    for bound in (0.20, 0.80):
        ax.axhline(bound, color="white", linestyle="--")
        ax.axvline(bound, color="white", linestyle="--")
    ax.set(xlabel="Normalized input 0", ylabel="Normalized input 1", title="Two-feature response")
    fig.tight_layout()
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "crosstalk_map.png", dpi=160)


if __name__ == "__main__":
    main()

