from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class River:

    path: npt.NDArray[np.float16]
    initial_flow: float = 1
    width_growth: float = 2
    depth_ratio: float = 0.001

    def flow_at(self, t: float) -> float:
        return self.initial_flow * (1 + t)

    def width_at(self, index: float, base_width: float) -> float:
        return base_width * self.flow_at(index) ** self.width_growth

    def depth_at(self, index: float, base_width: float) -> float:
        return self.width_at(index, base_width) * self.depth_ratio

    def max_width(self, base_width: float) -> float:
        return self.width_at(len(self.path) - 1, base_width)
