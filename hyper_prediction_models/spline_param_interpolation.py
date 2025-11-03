import torch
from utils.bspline import BSpline


class SplineParamInterpolation(torch.nn.Module):
    def __init__(self,
                 param_ctrl_point: int,
                 prediction_horizon: int,
                 last_ctrl_point_on_const: bool,
                 integration_method: str,
                 compile_model: bool,
                 device: str):

        super(SplineParamInterpolation, self).__init__()

        assert integration_method == "rk4" or integration_method == "euler"

        self.input_time_size = param_ctrl_point
        
        # FIXME removed due to causal model compatibility
        # if last_ctrl_point_on_const:
        #     self.input_time_size += 1

        self.output_time_size = prediction_horizon

        if integration_method == "rk4":
            self.output_time_size *= 2
            print("Using RK4 integration method param interpolation")

        assert self.input_time_size >= 5  # minimum number of control points

        self.device = torch.device(device)

        self.bspline = BSpline(self.input_time_size, 4,
                               self.output_time_size, "param_interpolation")
        self.Ns, self.dNs, self.ddNs, self.dddNs = self.bspline.calculate_N()

        self.Ns = torch.tensor(self.Ns,
                               dtype=torch.float32).contiguous().to(self.device)

        if compile_model is True:
            self.forward = torch.compile(self.forward, fullgraph=True)
            print("Compiling SplineParamInterpolation")

    def forward(self, p):
        """
            p: torch.Tensor of shape (batch_size, input_time_size, param_size)
        """
        assert len(p.shape) == 3
        assert p.shape[1] == self.input_time_size

        return torch.matmul(self.Ns, p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    frame_rate = 100
    time = 20
    points_count = frame_rate * time

    sampled_points_count = 5
    max_value = 0.25

    p = torch.randn((10, sampled_points_count, 12))

    # make args namespace
    class Args:
        def __init__(self):
            self.prediction_horizon = 20
            self.param_time_size = 5

    args = Args()
    p_interp = SplineParamInterpolation(args)

    print(p.shape)
    p_out = p_interp(p)
    print(p_out.shape)

    batch_see = 0

    for i in range(p_out.shape[-1]):
        t2 = torch.linspace(0, 20, 20)
        spline = p_out[batch_see, :, i]
        plt.scatter(t2, spline, label='spline')
        plt.plot(t2, spline, label='spline')
        plt.scatter(torch.linspace(0, 20, args.param_time_size),
                    p[batch_see, :, i], label='control points')

    plt.grid()
    plt.show()
