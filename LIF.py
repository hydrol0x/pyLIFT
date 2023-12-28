"""
simple LIF neuron implementation
"""

from math import exp
from free_model import simple_free
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(
        self,
        u_rest: float,
        tau: float,
        R: float,
        C: float,
        # dt: float,
        t_refrac: float,
        spike_threshold: float,
        # t_0: float = 0,
    ):
        # Neuron parameters
        self.R = R
        self.C = C
        self.tau = tau
        self.u_rest = u_rest
        self.t_refrac = t_refrac
        self.spike_threshold = spike_threshold

        # Simulation
        self.voltage: float = self.u_rest
        self.free = simple_free(u_rest, tau, 0)
        self.last_spike_time: float | None = None
        self.refractory = False

        # self.t_0 = t_0
        # self.dt = dt
        # self.end_time: None | float = None

        # self.voltages: list[float] = []

    def input(self, q: float, t: float):
        # Add a spike at time t with charge q
        self.free.add_input(q, t)

    @staticmethod
    def is_in_range(time: float, t_0: float, t: float):
        after_t_0 = time >= t_0
        before_t = time <= t
        return after_t_0 and before_t

    def u(self, t: float) -> float:
        if self.voltage >= self.spike_threshold and not self.refractory:
            self.last_spike_time = t
            self.refractory = True
            print("spike")
        if (
            self.last_spike_time
            and self.is_in_range(
                t, self.last_spike_time, self.last_spike_time + self.t_refrac
            )
            and self.refractory
        ):
            print("refractory")
            return self.u_rest
        self.voltage = self.free.u(t)
        return self.voltage


class Simulator:
    def __init__(
        self,
        neuron: Neuron,
        inputs: list[tuple[float, float]],
        start_time: float,
        end_time: float,
        dt: float,
    ):
        self.neuron = neuron
        self.start_time = start_time
        self.end_time = end_time
        self.inputs = inputs
        self.dt = dt
        self.voltages = []

    def simulate(self):
        input_idx = 0  # index to track the current spike

        input_time = 0
        for t in np.arange(self.start_time, self.end_time, self.dt):
            if input_idx < len(self.inputs) and t >= self.inputs[input_idx][0]:
                self.neuron.input(self.inputs[input_idx][1], t - input_time)
                input_time = self.inputs[input_idx][0]

                input_idx += 1
            self.voltages.append(self.neuron.u(t - input_time))

    def plot(self):
        x = np.arange(self.start_time, self.end_time, self.dt)
        plt.plot(x, self.voltages)
        plt.show()


if __name__ == "__main__":
    neuron = Neuron(0, 1, 1, 1, 1, 15)
    # inputs: list[tuple[float, float]] = [(0, 5), (1, 6), (2, 3), (5, 2), (5.5, 3)]
    inputs = [(1, 3), (2, 3), (3, 3), (4, 9)]
    simulator = Simulator(neuron, inputs, 0, 10, 0.01)
    simulator.simulate()
    simulator.plot()


# TODO: refactor so neuron has no memory and just outputs the voltage its at now
