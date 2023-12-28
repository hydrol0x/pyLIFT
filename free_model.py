"""
'free solution' for neuron potential from https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
"""

from math import exp

# import numpy
import matplotlib.pyplot as plt
import numpy as np

# u(t) = dU exp(-(t-t_0)/tau) + u_rest


class simple_free:
    def __init__(self, u_rest: float, tau: float, dU: float):
        self.u_rest = u_rest
        self.tau = tau
        self.dU = dU

    def add_input(self, q: float, t: float):
        # Update the accumulated effect of inputs
        print(q)
        self.dU = self.dU * exp(-(t) / self.tau) + q
        # Reset t_0 to the current time

    def u(self, t: float) -> float:
        return self.dU * exp(-(t) / self.tau) + self.u_rest
