import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from copy import deepcopy


class StateLessLinearLayer(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super(StateLessLinearLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x, p):
        p_this_layer = p[..., :self.in_size * self.out_size]
        p_this_layer = p_this_layer.reshape(self.in_size, self.out_size)

        res = torch.matmul(x, p_this_layer)
        p_out = p[..., self.in_size * self.out_size:]
        
        return res, p_out


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # remove after
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.nn(x)

    def set_params(self, params_delta: torch.Tensor):
        """
        params tensor of shape (param size)
        """

        assert params_delta.shape[0] == sum(
            [t.numel() for t in self.state_dict().values()])

        i = 0
        for param_name in deepcopy(self.state_dict()):

            t = self.state_dict()[param_name]
            numel = t.numel()

            self.state_dict()[
                param_name].data *= params_delta[i: i + numel].reshape(t.shape)


if __name__ == '__main__':
    model = BaseModel()

    x = torch.tensor([1.0, 2.0])

    y1 = model(x)

    print(model.state_dict())

    new_p = torch.zeros(41)

    model.set_params(new_p)

    y2 = model(x)

    print(model.state_dict())

    print(y1, y2)
