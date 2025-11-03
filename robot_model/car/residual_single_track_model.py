import robot_model.car.single_track
import robot_model.mlp.mlp_external_p
from robot_model.car.single_track import PacejkaTiresSingleTrack
from robot_model.mlp.mlp_external_p import MlpExternalParams
from typing import List
import torch



class ResidualSingleTrack(torch.nn.Module):
    def __init__(self, 
                 preprocessor: torch.nn.Module,
                 layer_sizes: List, # [input_dim, hidden1_dim, ..., output_dim]
                 activation: torch.nn.Module,
                 compile_model: bool,
                 device: str,
                 init_params=None):
        super(ResidualSingleTrack, self).__init__()
        
        self.single_track = PacejkaTiresSingleTrack(init_params).to(device)
        
        single_track_params = self.single_track.get_default_params().to(device)
        self.register_buffer('single_track_params', single_track_params)
        
        self.nn = MlpExternalParams(
            preprocessor=preprocessor,
            layer_sizes=layer_sizes,
            activation=activation
        ).to(device)
                
        if compile_model:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True)
        
    def forward(self, t, x, u, p):
        return self.single_track(t, x, u, self.single_track_params) + self.nn(t, x, u, p)

    def get_default_params(self, batch_size=1):
        return self.nn.get_default_params()

    def positive_params(self):
        return self.nn.positive_params()
    
    def state_weight(self):
        return self.nn.state_weight()

    def state_weights(self):
        return self.single_track.state_weights()
        
    def get_state_names(self):
        return self.single_track.get_state_names()
    
    def get_control_names(self):
        return self.single_track.get_control_names()

    @staticmethod
    def positive_params():
        # FIXME implement as index tensor for each parameter
        return False    
    
    @staticmethod
    def save_param_traj():
        False
        
    def get_params_names(self):
        return self.nn.get_params_names()