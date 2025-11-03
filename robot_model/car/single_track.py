import torch
import collections
from robot_model.car.state_wrapper import StateWrapper
from robot_model.car.pacejka_params import PacejkaParameters
from robot_model.car.pacejka_tire_model import PacejkaTireModel
from robot_model.car.single_track_params import VehicleParameters


class PacejkaTiresSingleTrack(torch.nn.Module):

    def __init__(self, init_params=None) -> None:
        super(PacejkaTiresSingleTrack, self).__init__()
        # Parameters wrappers
        self.vehicle_parameters = VehicleParameters()
        self.tire_model_parameters = PacejkaParameters()
        
        # Tire model
        self.tire_model = PacejkaTireModel()
        
        # Load initial parameters
        if init_params is not None:
            self.init_params = init_params
            self.get_default_params = self.get_default_params_pretrain
        else:
            self.get_default_params = self.get_default_params_static
            

    def forward(self, t, x, u, p):      
        
        x_and_u = torch.cat([x, u], dim=-1)

        wx = StateWrapper(x_and_u)

        # assert torch.all(torch.sqrt(wx.v_x**2 + wx.v_y**2) >= 0.05), "Car is not moving"

        wp, p = self.vehicle_parameters(p)
        wp_tire_f, p = self.tire_model_parameters(p)
        wp_tire_r, p = self.tire_model_parameters(p)
        
        tire_forces = self.tire_model(x_and_u, wp, wp_tire_f, wp_tire_r) * wx.friction.unsqueeze(-1)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)
        
        F_drag = wp.Cd0 * torch.sign(wx.v_x) +\
            wp.Cd1 * wx.v_x +\
            wp.Cd2 * wx.v_x * wx.v_x

        v_x_dot = 1.0 / wp.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + wp.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / wp.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - wp.m * wx.v_x * wx.r)  #

        r_dot = 1.0 / wp.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * wp.lf - Fy_r * wp.lr)

        # omega_wheels_dot = wp.R / wp.I_e * (wp.K_fi * wx.Iq - wp.R * Fx_f - wp.R * Fx_r
        #                                   - wx.omega_wheels * wp.b1 - torch.sign(wx.omega_wheels) * wp.b0)

        return torch.stack([v_x_dot, v_y_dot, r_dot,
                            # Dynamics that model is not posible,
                            # e.g. control inputs and friction
                            torch.zeros_like(wx.friction),
                            ], dim=-1)
 

    @staticmethod
    def get_default_params_static(batch_size=1):
        return torch.cat([
            VehicleParameters.default_params_tensor(batch_size),
            PacejkaParameters.default_params_tensor(batch_size),
            PacejkaParameters.default_params_tensor(batch_size)
        ], dim=-1)
    
    def get_default_params_pretrain(self, batch_size=1):
        return self.init_params      
    
    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # v_x
            1.0,  # v_y
            1.0,  # r
            0.0,  # friction
        ])
        
    @staticmethod
    def get_state_names():
        return ["v_x", "v_y", "r", "friction"]
    
    @staticmethod
    def get_control_names():
        return ["omega_wheels", "delta"]

    def positive_params(self):
        # FIXME implement as index tensor for each parameter
        l = self.get_default_params_static().shape[-1]
        return torch.arange(l)
    
    def get_params_names(self):
        front_tire = self.tire_model_parameters.get_params_names().copy()
        rear_tire = self.tire_model_parameters.get_params_names().copy()
        vehicle = self.vehicle_parameters.get_params_names().copy()
        
        for i in range(len(front_tire)):
            front_tire[i] = "front_tire_" + front_tire[i]
            
        for i in range(len(rear_tire)):
            rear_tire[i] = "rear_tire_" + rear_tire[i]
            
        return vehicle + front_tire + rear_tire
    
    @staticmethod
    def save_param_traj():
        return True