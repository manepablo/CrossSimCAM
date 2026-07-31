"""Search normalized input vectors against normalized interval words."""

import numpy as np

from crosssimcam import ACAMModel


def main() -> None:
    model = ACAMModel("6T2M")
    inputs = np.array([[0.30, 0.65], [0.75, 0.25], [0.50, 0.50]])
    lower = np.array([[0.20, 0.65], [0.55, 0.10]])
    upper = np.array([[0.55, 0.90], [0.80, 0.40]])
    dl, gmem_lb, gmem_hb = model.encode_normalized(inputs, lower, upper)
    result = model.simulate(dl, gmem_lb, gmem_hb)

    print("Rows are inputs; columns are stored words (1 = match):")
    print(result.matches.astype(int))
    print("Final match-line voltages [V]:")
    print(np.round(result.vml, 3))


if __name__ == "__main__":
    main()

