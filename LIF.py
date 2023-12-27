"""
simple LIF neuron implementation
"""

from math import exp
from simple_model import simple_free
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(
        self,
        u_rest: float,
        tau: float,
        R: float,
        C: float,
        dt: float,
        t_0: float = 0,
    ):
        self.R = R
        # self.q = q
        self.C = C
        self.u_rest = u_rest
        self.tau = tau
        self.t_0 = t_0
        self.dt = dt
        self.end_time: None | float = None
        self.voltages: list[float] = []
        self.free = simple_free(u_rest, tau, 0, dt, t_0)

    def spike(self, q: float, t: float):
        # Add a spike at time t with charge q
        self.free.add_spike(q, t)

    def u(self, t: float) -> float:
        return self.free.u(t)

    def simulate(self, end_time: float, spikes: list[tuple[float, float]]):
        self.end_time = end_time
        spike_idx = 0  # index to track the current spike

        for t in np.arange(self.t_0, end_time, self.dt):
            if spike_idx < len(spikes) and t >= spikes[spike_idx][0]:
                self.spike(spikes[spike_idx][1], t)

                spike_idx += 1

            self.voltages.append(self.u(t))

    def plot(self):
        x = np.arange(self.t_0, self.end_time, self.dt)
        plt.plot(x, self.voltages)
        plt.show()


neuron = Neuron(0, 1, 1, 1, 0.1)
# neuron.simulate(20, [(0, 5), (2.3, 4), (15, 10)])
neuron.simulate(20, [(0, 5), (1, 6), (2, 3), (5, 2), (5.5, 3)])
neuron.plot()
