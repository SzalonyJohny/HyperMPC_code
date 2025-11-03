import torch
from utils.bspline import BSpline


class ConstParamInterpolation(torch.nn.Module):
    def __init__(self,
                 param_ctrl_point,
                 prediction_horizon,
                 integration_method: str, 
                 device: str):
        super(ConstParamInterpolation, self).__init__()
        self.param_ctrl_point = param_ctrl_point
        assert self.param_ctrl_point == 1
        self.prediction_horizon = prediction_horizon

        assert integration_method == "rk4" or integration_method == "euler"
        if integration_method == 'rk4':
            self.prediction_horizon *= 2

    def forward(self, p):
        """
            p: torch.Tensor of shape (batch_size, param_ctrl_point, param_size)
        """
        assert len(p.shape) == 3
        assert p.shape[1] == self.param_ctrl_point

        return p.repeat(1, self.prediction_horizon, 1)
