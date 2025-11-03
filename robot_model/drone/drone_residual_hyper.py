import torch
from robot_model.drone.drone_model import DroneDynamisc
from robot_model.drone.drone_params_selected import DroneParamsWrapper
from robot_model.drone.drone_params_only_forces import DroneParamsWrapperForce
from robot_model.drone.drone_state_preprocesor import DroneStateCtrlPreprocessor
from robot_model.mlp.mlp_external_p import MlpExternalParams
import numpy as np

class DroneHyperResidualDynamisc(torch.nn.Module):
    def __init__(self,
                 init_params=None,
                 only_forces_regres_params = False,
                 layers_size = [10, 32, 6] ,
                 activation = torch.nn.Tanh(),
                 device = 'cpu',
                 model_nr = 1172,
                 *args, **kwargs) -> None:
        
        super(DroneHyperResidualDynamisc, self).__init__(*args, **kwargs)

        self.register_buffer('init_params', init_params)
        
        self.drone_model = DroneDynamisc(
            init_params=init_params,
            enable_dist_forces=True,
            enable_dist_torques=True,
            only_forces=True,
            only_forces_regres_params=only_forces_regres_params
        ).to(device)
        
        self.nn = MlpExternalParams(
            preprocessor=DroneStateCtrlPreprocessor(),
            layer_sizes=layers_size,
            activation=activation
        ).to(device)
        
        p_mlp = np.load(f"p_mlp_{model_nr}.npz")
        self.p_mlp = torch.tensor(p_mlp['p_mlp'], dtype=torch.float32).unsqueeze(0)
        self.drone_model.load_state_dict(torch.load(f"dyn_model_sd_{model_nr}.pt"))
            
    def get_default_params(self, batch_size=1):
        if self.init_params is not None:
            return self.init_params
        return self.drone_model.param_wrapper.get_default_params(batch_size)
    
    def count_params(self):
        return self.drone_model.param_wrapper.param_count
    
    def forward(self, t, x, u, p):
        """            
            t = [batch]
            x = [batch, state dim -> 13]
            u = [batch, propeler_speed_Omega -> 4]
            p = [batch, param_count]
        """ 
        p_mlp = self.p_mlp.expand(x.shape[0], -1)
        
        p_f_res = self.nn(t, x, u, p_mlp)
        
        p_f_res[..., :3] += p
        
        x_dot = self.drone_model(t, x, u, p_f_res)
        
        return x_dot
    
    def positive_params(self):
        return torch.tensor([], dtype=torch.int)        
    
    def free_params(self):
        return None
    
    @staticmethod
    def state_weights():
        return torch.tensor([1.0, 1.0, 10.0, # pos
                             0.0, 0.0, 0.0, 0.0, # quat
                             0.1, 0.1, 1.0, # vel
                             0.0, 0.0, 0.0])
    
    def get_params_names(self):
        return self.drone_model.get_params_names()
    
    @staticmethod
    def get_state_names():
        return ["x", "y", "z", # pos
                "qw", "qx", "qy", "qz", # quat
                "vx", "vy", "vz", # vel
                "wx", "wy", "wz"] # ang_vel
    
    @staticmethod
    def get_control_names():
        return ["m1", "m2", "m3", "m4"]
    
    @staticmethod
    def save_param_traj():
        return False


if __name__ == "__main__":
    
    model = DroneHyperResidualDynamisc()
    
    p = model.get_default_params()
    print(f"Default params: {p.shape}")
    
    t = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0, 0.0, # pos 
                       1.0, 0.0, 0.0, 0.0, # quat
                       0.0, 0.0, 0.0, # vel
                       0.0, 0.0, 0.0]], # ang_vel
                     dtype=torch.float32)
    u = torch.tensor([[20.0, 20.0, 20.0, 20.0]], dtype=torch.float32)
    
    x_dot = model(t, x, u, p)
    
    # test on batch size 64
    x = x.expand(64, -1)
    u = u.expand(64, -1)
    p = p.expand(64, -1)
    
    x_dot = model(t, x, u, p)
    print(f"x_dot shape: {x_dot.shape}")
    
        
        
        