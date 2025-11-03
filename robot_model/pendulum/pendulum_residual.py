import torch
from robot_model.pendulum.pendulum_model import PendulumDynamisc
from robot_model.acrobot.acrobot_params import ActobotParams
from robot_model.mlp.mlp_external_p import MlpExternalParams
from typing import List


class ResidualPendulumDynamisc(torch.nn.Module):
    def __init__(self,
                 preprocessor: torch.nn.Module,
                 layer_sizes: List, # [input_dim, hidden1_dim, ..., output_dim]
                 activation: torch.nn.Module,
                 compile_model: bool,
                 device: str,
                 init_params=None,
                 skip_gr=False) -> None:
                
        super(ResidualPendulumDynamisc, self).__init__()
        
        self.pendulum = PendulumDynamisc(init_params, skip_gr=skip_gr).to(device)
        
        self.nn = MlpExternalParams(
            preprocessor=preprocessor,
            layer_sizes=layer_sizes,
            activation=activation
        ).to(device)
        
        if init_params is None:
            init_params = self.pendulum.get_default_params().to(device)
        self.register_buffer("params", init_params)
        
        if compile_model:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True)

    def get_default_params(self, batch_size=1):
        return self.nn.get_default_params()

    @staticmethod
    def positive_params():
        return False
    
    def get_params_names(self):
        return self.nn.get_params_names()
    
    @staticmethod
    def get_state_names():
        return ["q", "dq"]
    
    @staticmethod
    def get_control_names():
        return ["u"]

    @staticmethod
    def save_param_traj():
        return False

    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # q
            1.0,  # dq
        ])
        
    def forward(self, t, x, u, p):
        dq = self.nn(t, x, u, p)
        x_dot_nn = torch.cat([torch.zeros_like(dq), dq], dim=-1)
        return self.pendulum(t, x, u, self.params) + x_dot_nn

