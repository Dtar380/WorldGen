from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


ARRAY_64 = npt.NDArray[np.float64]


@dataclass()
class Biome:

    elevation_range: tuple[float, float]
    humidity_range: tuple[float, float]
    temperature_range: tuple[float, float]
    marine: bool

    name: str
    color: str

    @property
    def ideal_elevation(self) -> float:
        return sum(self.elevation_range) / 2

    @property
    def ideal_humidity(self) -> float:
        return sum(self.humidity_range) / 2

    @property
    def ideal_temperature(self) -> float:
        return sum(self.temperature_range) / 2

    def distance(
        self, elevation: ARRAY_64, humidity: ARRAY_64, temperature: ARRAY_64
    ) -> ARRAY_64:
        return np.sqrt(
            (elevation - self.ideal_elevation) ** 2
            + (humidity - self.ideal_humidity) ** 2
            + (temperature - self.ideal_temperature) ** 2
        )
