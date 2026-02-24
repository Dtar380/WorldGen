from __future__ import annotations

import hashlib
from random import randint
from typing import Any

import numpy as np
import numpy.typing as npt

from .noise import Noise
from .biome import Biome


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
        self._noise_waves = Noise(self._hash_seed(f"{self._seed}_waves"))

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

    def assign_biomes(
        self,
        elevation: ARRAY_64,
        humidity: ARRAY_64,
        temperature: ARRAY_64,
        biomes: list[Biome],
    ) -> npt.NDArray[np.int16]:

        marine_mask = elevation <= self._sea_level

        result = np.full(elevation.shape, -1, dtype=np.int16)
        min_dist = np.full(elevation.shape, np.inf)

        for i, biome in enumerate(biomes):
            valid = (
                (elevation >= biome.elevation_range[0]) & (elevation <= biome.elevation_range[1])
                & (humidity >= biome.humidity_range[0]) & (humidity <= biome.humidity_range[1])
                & (temperature >= biome.temperature_range[0]) & (temperature <= biome.temperature_range[1])
                & (marine_mask == biome.marine)
            )
            dist = biome.distance(elevation, humidity, temperature)
            update = valid & (dist < min_dist)
            result[update] = i
            min_dist[update] = dist[update]

        return result

    def generate_waves(
        self,
        origin: tuple[float, float],
        size: int,
        scale: float,
        wave_frequency: float = 8.0,
        octaves: int = 4,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
    ) -> ARRAY_64:
        ox, oy = origin
        wave_scale = scale * wave_frequency

        xs = np.linspace(ox - wave_scale, ox + wave_scale, size)
        ys = np.linspace(oy - wave_scale, oy + wave_scale, size)
        xx, yy = np.meshgrid(xs, ys)

        chunk = self._noise_waves.fbm(
            xx,
            yy,
            octaves=octaves,
            lacunarity=lacunarity,
            persistance=persistance,
            simplex=self._simplex,
        )
        normalised_chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())

        return normalised_chunk
