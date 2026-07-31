"""Simulate two inputs against two stored aCAM words."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crosssimcam import ACAMModel


def main() -> None:
    # Word 0 contains input 0; word 1 contains input 1. The off-diagonal
    # comparisons are mismatches, which makes the four traces easy to compare.
    inputs = np.array([[0.30, 0.65], [0.75, 0.25]])
    lower = np.array([[0.20, 0.65], [0.55, 0.10]])
    upper = np.array([[0.55, 0.90], [0.80, 0.40]])

    calibration_model = ACAMModel("6T2M")
    dl_voltage, gmem_lb, gmem_hb = calibration_model.encode_normalized(
        inputs, lower, upper
    )
    model = ACAMModel("6T2M", max_steps=50)
    result = model.simulate(dl_voltage, gmem_lb, gmem_hb)

    print("Final match-line voltages [V]:")
    print(np.round(result.vml, 4))
    print("Digital matches:")
    print(result.matches.astype(int))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for batch in range(result.vml_over_time.shape[0]):
        for word in range(result.vml_over_time.shape[2]):
            ax.plot(
                result.time * 1e9,
                result.vml_over_time[batch, :, word],
                label=f"input {batch}, word {word}",
            )
    ax.axhline(model.sense_threshold, color="black", linestyle="--", label="sense threshold")
    ax.set(xlabel="Time [ns]", ylabel="Match-line voltage [V]", ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "basic_transient.png", dpi=160)


if __name__ == "__main__":
    main()
