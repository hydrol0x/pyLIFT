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
        self.end_time = None

    def add_spike(self, q: float, t: float):
        # Update the accumulated effect of spikes
        self.dU = self.dU * exp(-(t - self.t_0) / self.tau) + q
        # Reset t_0 to the current time
        self.t_0 = t

    def u(self, t: float) -> float:
        return self.dU * exp(-(t - self.t_0) / self.tau) + self.u_rest
