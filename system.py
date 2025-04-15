from logging import warn
import numpy as np


def delay(y, tau, deltaT):
    N = int(np.ceil(tau / deltaT))
    if N == 0:
        return y
    else:
        padding = np.tile(y[:, 0:1], (1, N))
        return np.hstack((padding, y[:, :-N]))


def moving_average(y, tau, deltaT):
    N = int(np.ceil(tau / deltaT))
    if N == 0:
        return y
    else:
        v = [deltaT] * N
        ys = []
        for i in range(y.shape[0]):
            yi = np.convolve(y[i, :], v)[0 : y.shape[1]]
            ys.append(yi.reshape((1, -1)))
        return np.hstack(ys) / tau


class AffineSystem:

    def __init__(self, f, g, h, n, std_noise=0, seed=1234) -> None:
        self.f = f
        self.g = g
        self.h = h
        self.n = n
        self.std_noise = std_noise
        self.default_rng = np.random.default_rng(seed)

    def simulate(self, x0, T, deltaT, u=lambda t: np.array([[0]])):
        ts = np.arange(0, T, deltaT)
        self.t = ts.reshape((1, -1))
        self.deltaT = deltaT
        self.xs = [x0]
        self.ys = [self.h(x0)]
        us = u(self.t)
        for i in range(len(ts) - 1):
            x_plus = self.xs[-1] + deltaT * (
                self.f(self.xs[-1]) + self.g(self.xs[-1]) @ us[:, i : (i + 1)]
            )
            self.xs.append(x_plus)
            self.ys.append(self.h(x_plus))
        self.xs = np.hstack(self.xs)
        self.ys = np.hstack(self.ys)
        self.ys += self.std_noise * self.default_rng.normal(size=self.ys.shape)
        return self.xs

    def y(self):
        return self.ys
