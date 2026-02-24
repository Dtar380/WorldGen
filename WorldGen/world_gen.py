from __future__ import annotations

import hashlib
from random import randint
from typing import Any

import numpy as np
import numpy.typing as npt

from .noise import Noise


ARRAY_64 = npt.NDArray[np.float64]


class WorldGen:

    def __init__(
        self,
        seed: Any | None = None,
        sea_level: float = 0.5,
        simplex: bool = False,
    ) -> None:
        self._seed = seed if seed is not None else randint(-2**32, 2**32)
        self._sea_level = sea_level
        self._simplex = simplex

        self._noise_elevation = Noise(self._hash_seed(f"{self._seed}_elevation"))
        self._noise_humidity = Noise(self._hash_seed(f"{self._seed}_humidity"))
        self._noise_temperature = Noise(self._hash_seed(f"{self._seed}_temperature"))

    def _hash_seed(self, seed: Any) -> int:
        return int(hashlib.md5(str(seed).encode()).hexdigest(), 16) & (2**32 - 1)

    def generate_elevation_map(
        self,
        origin: tuple[float, float],
        size: int,
        scale: float,
        octaves: int = 8,
        k: float = 0.05,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
    ) -> ARRAY_64:
        ox, oy = origin

        xs = np.linspace(ox - scale, ox + scale, size)
        ys = np.linspace(oy - scale, oy + scale, size)
        xx, yy = np.meshgrid(xs, ys)

        chunk = self._noise_elevation.fbm_gradient(
            xx,
            yy,
            octaves=octaves,
            k=k,
            lacunarity=lacunarity,
            persistance=persistance,
            simplex=self._simplex,
        )
        normalised_chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())

        return normalised_chunk

    def generate_humidity_map(
        self,
        elevation: ARRAY_64,
        origin: tuple[float, float],
        size: int,
        scale: float,
        octaves: int = 4,
        k: float = 1,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
    ) -> ARRAY_64:
        ox, oy = origin

        xs = np.linspace(ox - scale, ox + scale, size)
        ys = np.linspace(oy - scale, oy + scale, size)
        xx, yy = np.meshgrid(xs, ys)

        chunk = self._noise_humidity.fbm(
            xx,
            yy,
            octaves=octaves,
            lacunarity=lacunarity,
            persistance=persistance,
            simplex=self._simplex,
        )
        f = np.minimum(1, (1 - elevation) / (1 - self._sea_level))
        print(f ** 2)
        chunk += chunk * k * f ** 2
        normalised_chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())

        return normalised_chunk

    def generate_temperature_map(
        self,
        elevation: ARRAY_64,
        origin: tuple[float, float],
        size: int,
        scale: float,
        octaves: int = 4,
        k: float = 1,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
    ) -> ARRAY_64:
        ox, oy = origin

        xs = np.linspace(ox - scale, ox + scale, size)
        ys = np.linspace(oy - scale, oy + scale, size)
        xx, yy = np.meshgrid(xs, ys)

        chunk = self._noise_temperature.fbm(
            xx,
            yy,
            octaves=octaves,
            lacunarity=lacunarity,
            persistance=persistance,
            simplex=self._simplex,
        )
        f = np.maximum(0, (elevation - self._sea_level) / (1 - self._sea_level))
        chunk -= chunk * k * f ** 2
        normalised_chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())

        return normalised_chunk
