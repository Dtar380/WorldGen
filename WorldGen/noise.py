from __future__ import annotations

import numpy as np
import numpy.typing as npt

ARRAY_64 = npt.NDArray[np.float64]


class Noise:

    def __init__(self, seed: int) -> None:
        self._seed = seed
        perm = np.random.default_rng(self._seed).permutation(256)
        self._perm = np.concatenate([perm, perm])

    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _fade_deriv(self, t: float) -> float:
        tp = t - 1
        return 30 * t * t * tp * tp

    def _grad(self, hash_: ARRAY_64, x: npt.ArrayLike, y: npt.ArrayLike) -> ARRAY_64:
        h = hash_.astype(int) & 7
        u = np.where(h < 4, x, y)
        v = np.where(h < 4, y, x)
        u = np.where((h & 1) == 0, u, -u)
        v = np.where((h & 2) == 0, v, np.where(h < 6, 0.0, -v))
        return u + v

    def _grad_deriv(
        self, hash_: ARRAY_64, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[ARRAY_64, ARRAY_64, ARRAY_64]:
        value = self._grad(hash_, x, y)
        dx = self._grad(hash_, np.ones_like(x), np.zeros_like(x))
        dy = self._grad(hash_, np.zeros_like(x), np.ones_like(x))
        return value, dx, dy

    def _lerp(self, a: ARRAY_64, b: ARRAY_64, t: float) -> ARRAY_64:
        return a + t * (b - a)

    def perlin(self, x: ARRAY_64, y: ARRAY_64) -> ARRAY_64:
        xi = x.astype(int) & 255
        yi = y.astype(int) & 255
        xf = x - x.astype(int)
        yf = y - y.astype(int)

        u = self._fade(xf)
        v = self._fade(yf)

        n00 = self._perm[self._perm[xi] + yi]
        n01 = self._perm[self._perm[xi] + yi + 1]
        n10 = self._perm[self._perm[xi + 1] + yi]
        n11 = self._perm[self._perm[xi + 1] + yi + 1]

        x1 = self._lerp(
            self._grad(n00, xf, yf), self._grad(n10, xf - 1, yf), u
        )
        x2 = self._lerp(
            self._grad(n01, xf, yf - 1), self._grad(n11, xf - 1, yf - 1), u
        )
        return self._lerp(x1, x2, v)

    def perlin_deriv(
        self, x: ARRAY_64, y: ARRAY_64
    ) -> tuple[ARRAY_64, ARRAY_64, ARRAY_64]:
        xi = x.astype(int) & 255
        yi = y.astype(int) & 255
        xf = x - x.astype(int)
        yf = y - y.astype(int)

        u = self._fade(xf)
        du = self._fade_deriv(xf)
        v = self._fade(yf)
        dv = self._fade_deriv(yf)

        n00 = self._perm[self._perm[xi] + yi]
        n01 = self._perm[self._perm[xi] + yi + 1]
        n10 = self._perm[self._perm[xi + 1] + yi]
        n11 = self._perm[self._perm[xi + 1] + yi + 1]

        g00, dg00_dx, dg00_dy = self._grad_deriv(n00, xf, yf)
        g10, dg10_dx, dg10_dy = self._grad_deriv(n10, xf - 1, yf)
        g01, dg01_dx, dg01_dy = self._grad_deriv(n01, xf, yf - 1)
        g11, dg11_dx, dg11_dy = self._grad_deriv(n11, xf - 1, yf - 1)

        x1 = self._lerp(g00, g10, u)
        dx1 = self._lerp(dg00_dx, dg10_dx, u) + (g10 - g00) * du
        dy1 = self._lerp(dg00_dy, dg10_dy, u)

        x2 = self._lerp(g01, g11, u)
        dx2 = self._lerp(dg01_dx, dg11_dx, u) + (g11 - g01) * du
        dy2 = self._lerp(dg01_dy, dg11_dy, u)

        value = self._lerp(x1, x2, v)
        dx = self._lerp(dx1, dx2, v)
        dy = self._lerp(dy1, dy2, v) + (x2 - x1) * dv
        return value, dx, dy

    def simplex(self, x: ARRAY_64, y: ARRAY_64) -> ARRAY_64:
        F = (np.sqrt(3) - 1) / 2
        G = (3 - np.sqrt(3)) / 6

        s = (x + y) * F
        xi = np.floor(x + s).astype(int)
        yi = np.floor(y + s).astype(int)

        t = (xi + yi) * G
        x0 = x - (xi - t)
        y0 = y - (yi - t)

        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)

        x1 = x0 - i1 + G
        y1 = y0 - j1 + G
        x2 = x0 - 1 + 2 * G
        y2 = y0 - 1 + 2 * G

        ii = xi & 255
        jj = yi & 255

        n0 = self._perm[self._perm[ii] + jj]
        n1 = self._perm[self._perm[ii + i1] + jj + j1]
        n2 = self._perm[self._perm[ii + 1] + jj + 1]

        t0 = 0.5 - x0**2 - y0**2
        t1 = 0.5 - x1**2 - y1**2
        t2 = 0.5 - x2**2 - y2**2

        c0 = np.where(t0 < 0, 0.0, t0**4 * self._grad(n0, x0, y0))
        c1 = np.where(t1 < 0, 0.0, t1**4 * self._grad(n1, x1, y1))
        c2 = np.where(t2 < 0, 0.0, t2**4 * self._grad(n2, x2, y2))

        return 70.0 * (c0 + c1 + c2)

    def simplex_deriv(
        self, x: ARRAY_64, y: ARRAY_64
    ) -> tuple[ARRAY_64, ARRAY_64, ARRAY_64]:
        F = (np.sqrt(3) - 1) / 2
        G = (3 - np.sqrt(3)) / 6

        s = (x + y) * F
        xi = np.floor(x + s).astype(int)
        yi = np.floor(y + s).astype(int)

        t = (xi + yi) * G
        x0 = x - (xi - t)
        y0 = y - (yi - t)

        i1 = np.where(x0 > y0, 1, 0)
        j1 = np.where(x0 > y0, 0, 1)

        x1 = x0 - i1 + G
        y1 = y0 - j1 + G
        x2 = x0 - 1 + 2 * G
        y2 = y0 - 1 + 2 * G

        ii = xi & 255
        jj = yi & 255

        n0 = self._perm[self._perm[ii] + jj]
        n1 = self._perm[self._perm[ii + i1] + jj + j1]
        n2 = self._perm[self._perm[ii + 1] + jj + 1]

        t0 = 0.5 - x0**2 - y0**2
        t1 = 0.5 - x1**2 - y1**2
        t2 = 0.5 - x2**2 - y2**2

        g0, dg0_dx, dg0_dy = self._grad_deriv(n0, x0, y0)
        g1, dg1_dx, dg1_dy = self._grad_deriv(n1, x1, y1)
        g2, dg2_dx, dg2_dy = self._grad_deriv(n2, x2, y2)

        t03 = t0**3
        t13 = t1**3
        t23 = t2**3

        c0 = np.where(t0 < 0, 0.0, t0 * t03 * g0)
        c1 = np.where(t1 < 0, 0.0, t1 * t13 * g1)
        c2 = np.where(t2 < 0, 0.0, t2 * t23 * g2)

        dc0_dx = np.where(t0 < 0, 0.0, -8 * t03 * x0 * g0 + t0 * t03 * dg0_dx)
        dc0_dy = np.where(t0 < 0, 0.0, -8 * t03 * y0 * g0 + t0 * t03 * dg0_dy)

        dc1_dx = np.where(t1 < 0, 0.0, -8 * t13 * x1 * g1 + t1 * t13 * dg1_dx)
        dc1_dy = np.where(t1 < 0, 0.0, -8 * t13 * y1 * g1 + t1 * t13 * dg1_dy)

        dc2_dx = np.where(t2 < 0, 0.0, -8 * t23 * x2 * g2 + t2 * t23 * dg2_dx)
        dc2_dy = np.where(t2 < 0, 0.0, -8 * t23 * y2 * g2 + t2 * t23 * dg2_dy)

        value = 70.0 * (c0 + c1 + c2)
        dx = 70.0 * (dc0_dx + dc1_dx + dc2_dx)
        dy = 70.0 * (dc0_dy + dc1_dy + dc2_dy)
        return value, dx, dy

    def fbm(
        self,
        x: ARRAY_64,
        y: ARRAY_64,
        octaves: int = 6,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
        simplex: bool = False,
    ) -> ARRAY_64:
        noise_func = self.simplex if simplex else self.perlin

        result = np.zeros_like(x)

        amplitud = 1.0
        frequency = 1.0
        max_amplitud = 0.0

        for _ in range(octaves):
            result += amplitud * noise_func(x * frequency, y * frequency)
            max_amplitud += amplitud
            frequency *= lacunarity
            amplitud *= persistance

        return result / max_amplitud

    def fbm_gradient(
        self,
        x: ARRAY_64,
        y: ARRAY_64,
        octaves: int = 6,
        k: float = 0.05,
        lacunarity: float = 2.0,
        persistance: float = 0.5,
        simplex: bool = False,
    ) -> ARRAY_64:
        noise_func = self.simplex_deriv if simplex else self.perlin_deriv

        result = np.zeros_like(x)
        dx = np.zeros_like(x)
        dy = np.zeros_like(y)

        amplitud = np.ones_like(x)
        frequency = 1.0
        max_amplitud = np.zeros_like(x)

        for _ in range(octaves):
            value, temp_dx, temp_dy = noise_func(x * frequency, y * frequency)
            dx += temp_dx
            dy += temp_dy
            result += amplitud * value
            max_amplitud += amplitud
            frequency *= lacunarity
            amplitud *= persistance / (1 + k * np.sqrt(dx**2 + dy**2))

        return result / max_amplitud
