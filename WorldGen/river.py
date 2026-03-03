from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class River:

    path: npt.NDArray[np.float16]
    initial_flow: float = 1
    width_growth: float = 0.5
    depth_ratio: float = 0.1

    def flow_at(self, index: int) -> float:
        return self.initial_flow * (1 + index / len(self.path))

    def width_at(self, index: int) -> float:
        return self.flow_at(index) ** self.width_growth

    def depth_at(self, index: int) -> float:
        return self.width_at(index) * self.depth_ratio
