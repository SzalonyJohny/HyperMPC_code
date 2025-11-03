import matplotlib.pyplot as plt
from utils.bspline import BSpline
import torch
import numpy as np


class ControlVectorSampler():

    def __init__(self, framerate, T_end, control_points_count, max_value) -> None:
        self.bspline = BSpline(control_points_count, 4,
                               framerate * T_end, "control_sample")
        self.Ns = self.bspline.N
        self.max_value = max_value
        self.control_points_count = control_points_count
        self.rng = torch.Generator().manual_seed(42)

    def sample(self):
        control_points = torch.rand(
            (self.control_points_count), generator=self.rng) * 2 * self.max_value - self.max_value

        control_points = control_points.numpy()
        control_points[-1] = 0.0
        control_points[0] = 0.0
        spline = np.matmul(self.Ns, control_points)
        return spline[0, :]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    frame_rate = 100
    time = 20
    points_count = frame_rate * time

    sampled_points_count = 20
    max_value = 1.0

    sampler = ControlVectorSampler(frame_rate,
                                   time,
                                   sampled_points_count,
                                   max_value)

    for i in range(5):
        t2 = np.linspace(0, 20, points_count)

        spline = sampler.sample()
        plt.plot(t2, spline, label='spline')

    plt.grid()
    plt.show()
