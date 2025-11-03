import torch
from robot_model.drone.drone_model import DroneDynamisc
from robot_model.drone.drone_params_selected import DroneParamsWrapper
from robot_model.drone.drone_params_only_forces import DroneParamsWrapperForce
from robot_model.drone.drone_state_preprocesor import DroneStateCtrlPreprocessor
from robot_model.mlp.mlp_external_p import MlpExternalParams

class DroneResidualDynamisc(torch.nn.Module):
    def __init__(self,
                 init_params=None,
                 only_forces_regres_params = False,
                 layers_size = [10, 32, 6] ,
                 activation = torch.nn.Tanh(),
                 device = 'cpu',
                 *args, **kwargs) -> None:
        
        super(DroneResidualDynamisc, self).__init__(*args, **kwargs)

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
        
            
    def get_default_params(self, batch_size=1):
        if self.init_params is not None:
            return self.init_params
        return self.nn.get_default_params()
    
    def count_params(self):
        return self.nn.param_count

    def forward(self, t, x, u, p):
        """            
            t = [batch]
            x = [batch, state dim -> 13]
            u = [batch, propeler_speed_Omega -> 4]
            p = [batch, param_count]
        """ 
        p_f_res = self.nn(t, x, u, p)
        
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
        return self.nn.get_params_names()
    
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
        return True


if __name__ == "__main__":
    
    model = DroneResidualDynamisc()
    
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
    
        
        
        