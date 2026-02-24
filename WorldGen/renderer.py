from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import numpy as np
import numpy.typing as npt

from .biome import Biome


class Renderer:

    @staticmethod
    def save_map(
        array: npt.NDArray[np.float64 | np.int16],
        path: str,
        title: str = "",
        cmap: str | Colormap = "gray",
        norm: Normalize | None = None
    ) -> None:
        plt.figure(figsize=(8, 8))
        plt.title(title)
        plt.imshow(array, cmap=cmap, norm=norm, origin="upper")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def biome_cmap(biomes: list[Biome]) -> tuple[Colormap, Normalize]:

        colors = [biome.color for biome in biomes]
        cmap = ListedColormap(colors)
        bounds = np.arange(-0.5, len(biomes), 1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm
