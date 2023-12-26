"""
'free solution' for neuron potential from https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
"""

from math import exp

# import numpy
import matplotlib.pyplot as plt
import numpy as np

# u(t) = dU exp(-(t-t_0)/tau) + u_rest


class simple_free:
    def __init__(self, u_rest: float, tau: float, dU: float, dt: float, t_0: float = 0):
        self.u_rest = u_rest
        self.tau = tau
        self.t_0 = t_0
        self.dU = dU
        self.dt = dt
        self.voltages = []

    def u(self, t: float) -> float:
        return self.dU * exp(-(t - self.t_0) / self.tau) + self.u_rest

    def simulate(self, end_time: float):
        # simulate from t_0 to end_time seconds
        self.end_time = end_time
        for t in np.arange(self.t_0, end_time, self.dt):
            self.voltages.append(self.u(t))

    def plot(self):
        if not self.end_time or not self.voltages:
            raise ValueError("Must run `simulate()` first.")
        x = np.arange(self.t_0, self.end_time, self.dt)
        plt.plot(x, self.voltages)
        plt.show()


if __name__ == "__main__":
    model = simple_free(1, 5, 1, 0.01)
    model.simulate(10)
    model.plot()
