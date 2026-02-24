from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter  # type: ignore

from .biome import Biome


class Renderer:

    def __init__(self, biomes: list[Biome]) -> None:
        self._biomes = biomes

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

    def _compute_normals(
        self, elevation: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        padded = np.pad(elevation, 1, mode="edge")

        dx = padded[1:-1, 2:] - padded[1:-1, :-2]
        dy = padded[2:, 1:-1] - padded[:-2, 1:-1]

        length = np.sqrt(dx**2 + dy**2 + 1)
        nx = -dx / length
        ny = -dy / length
        nz = np.ones_like(elevation) / length

        return np.stack([nx, ny, nz], axis=-1)

    def _hex_to_rgb(self, hex_color: str) -> tuple[float, float, float]:
        return (
            int(hex_color[1:3], 16) / 255,
            int(hex_color[3:5], 16) / 255,
            int(hex_color[5:7], 16) / 255,
        )

    @staticmethod
    def hour_to_azimuth(
        hour: float,
        sun_hours: tuple[float, float],
        altitud: float,
    ) -> float:
        dawn, dusk = sun_hours
        azimuth = 90 + ((hour - dawn) / (dusk - dawn)) * 180
        return azimuth if altitud <= 0 else (azimuth + 180) % 360

    def _sun(
        self,
        altitud: float = -45,
        azimuth: float = 12,
    ) -> npt.NDArray[np.float64]:
        alt = np.radians(90 - altitud)
        azi = np.radians(azimuth)
        x = np.cos(alt) * np.sin(azi)
        y = np.cos(alt) * np.cos(azi)
        z = np.sin(alt)
        return np.array([x, y, z])

    def _biome_blend(
        self,
        colors: npt.NDArray[np.float64],
        biome_map: npt.NDArray[np.int16],
        elevation: npt.NDArray[np.float64],
        sea_level: float = 0.5,
        radius: int = 10,
    ) -> npt.NDArray[np.float64]:
        diff_x = biome_map[:, 1:] != biome_map[:, :-1]
        diff_y = biome_map[1:, :] != biome_map[:-1, :]

        marine = elevation <= sea_level

        same_type_x = (marine[:, 1:] == marine[:, :-1])
        same_type_y = (marine[1:, :] == marine[:-1, :])

        border_x = diff_x & same_type_x
        border_y = diff_y & same_type_y

        borders = np.zeros(biome_map.shape, dtype=bool)
        borders[:, 1:] |= border_x
        borders[:, :-1] |= border_x
        borders[1:, :] |= border_y
        borders[:-1, :] |= border_y

        dilated = binary_dilation(borders, iterations=radius)
        blurred = gaussian_filter(colors, sigma=(radius/2, radius/2, 0))
        colors[dilated] = blurred[dilated]

        return colors

    def render(
        self,
        elevation: npt.NDArray[np.float64],
        biome_map: npt.NDArray[np.int16],
        sea_level: float = 0.5,
        biome_blend: int = 10,
        relief: float = 100.0,
        ambient: float = 0.4,
        sun_altitud: float = -45,
        sun_azimuth: float = 180,
    ) -> Image.Image:
        flat_elevation = elevation.copy()
        flat_elevation[elevation <= sea_level] = sea_level
        normals = self._compute_normals(flat_elevation * relief)
        sun = self._sun(sun_altitud, sun_azimuth)

        h, w = elevation.shape
        colors = np.zeros((h, w, 3))
        for i, biome in enumerate(self._biomes):
            mask = biome_map == i
            colors[mask] = self._hex_to_rgb(biome.color)
        colors = self._biome_blend(
            colors, biome_map, elevation, sea_level, biome_blend
        )

        intensity = ambient + (1 - ambient) * np.maximum(0, np.dot(normals, sun))
        colors *= intensity[:, :, np.newaxis]

        return Image.fromarray((colors * 255).astype(np.uint8))
