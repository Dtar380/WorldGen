from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Renderer:

    @staticmethod
    def save_map(
        array: npt.NDArray[np.float64],
        path: str,
        title: str = "",
        cmap: str = "gray",
    ) -> None:
        plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.imshow(array, cmap=cmap, origin="upper")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
