"""
constant input current pulse for neuron potential from https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
"""

from math import exp

# import numpy
import matplotlib.pyplot as plt
import numpy as np

# u(t) = R*I_0 * exp(-(t-t_0)/tau) + u_rest


class simple_constant:
    def __init__(
        self, u_rest: float, tau: float, R: float, I_0: float, dt: float, t_0: float = 0
    ):
        self.u_rest = u_rest
        self.tau = tau
        self.t_0 = t_0
        self.R = R
        self.I_0 = I_0
        self.dt = dt
        self.voltages = []

    def u(self, t: float) -> float:
        RI = self.R * self.I_0
        expon = 1 - exp(-(t - self.t_0) / self.tau)
        return self.u_rest + RI * expon

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
    model = simple_constant(10, 5, 1, 1, 0.01)
    model.simulate(100)
    model.plot()
