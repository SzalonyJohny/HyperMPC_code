import torch
from collections import namedtuple

# Define the new parameters list
DroneParamsList = [
    "mq",        # total mass
    "Db",        # drag coefficient body
    "Cd",        # drag coefficient motor
    "Ct",        # thrust coefficient
    "g0",        # gravitational acceleration
    "Jx",        # inertia x
    "Jy",        # inertia y
    "Jz",        # inertia z
    "l",         # distance (half between motors' center and rotation axis)
    "f_dist_x",  # disturbance force x
    "f_dist_y",  # disturbance force y
    "f_dist_z",  # disturbance force z
    "tau_dist_x",# disturbance torque x
    "tau_dist_y",# disturbance torque y
    "tau_dist_z" # disturbance torque z
]

DroneParams = namedtuple("DroneParams", DroneParamsList, defaults=[None] * len(DroneParamsList))

class DroneParamsWrapperForce(torch.nn.Module):
    def __init__(self,
                 ball_mass = 0.5,
                 enable_forces = False,
                 enable_torques = False,
                 regres_params =  False,
                 *args, **kwargs) -> None:
        super(DroneParamsWrapperForce, self).__init__(*args, **kwargs)
        self.param_count = self.get_default_params().shape[-1]
        
        self.register_buffer("mq", torch.tensor([1.325 + ball_mass], dtype=torch.float32))
        self.register_buffer("g0", torch.tensor([9.80665], dtype=torch.float32))
        self.register_buffer("l", torch.tensor([0.228035], dtype=torch.float32))
        
        f_e = float(enable_forces)
        self.register_buffer("f_dist_scaler", torch.tensor([f_e], dtype=torch.float32))
        
        def set_const(name, value):
            self.register_buffer(name, torch.tensor([value], dtype=torch.float32).log())
        
        def set_learnable(name, value):
            self.register_parameter(name, torch.nn.Parameter(torch.tensor([value], dtype=torch.float32).log()))
        
        register_func = set_learnable if regres_params else set_const
        register_func("Db", 0.11951028)
        register_func("Cd", 0.29075044) 
        register_func("Ct", 0.99068546)
        register_func("Jx", 0.04977089)
        register_func("Jy", 0.05110507)
        register_func("Jz", 0.0954449)
        
    def forward(self, p):
        p_this_layer = p[..., :self.param_count]

        named_tuple = DroneParams(
            mq=self.mq,
            Db=self.Db.exp(),
            Cd=self.Cd.exp(),
            Ct=self.Ct.exp(),   
            g0=self.g0,
            Jx=self.Jx.exp(),
            Jy=self.Jy.exp(),
            Jz=self.Jz.exp(),
            l=self.l,
            f_dist_x=p_this_layer[..., 0] * self.f_dist_scaler,
            f_dist_y=p_this_layer[..., 1] * self.f_dist_scaler,
            f_dist_z=p_this_layer[..., 2] * self.f_dist_scaler,
            tau_dist_x=torch.zeros_like(p_this_layer[..., 0]),
            tau_dist_y=torch.zeros_like(p_this_layer[..., 0]),
            tau_dist_z=torch.zeros_like(p_this_layer[..., 0]),
        )
        
        return named_tuple, p[..., self.param_count:]

    @staticmethod
    def get_default_params(batch_size=1):
        default_params = [
            0.0,               # f_dist_x
            0.0,               # f_dist_y
            0.0,               # f_dist_z
        ]
        tensor = torch.tensor(default_params, dtype=torch.get_default_dtype())
        return tensor.unsqueeze(0).repeat(batch_size, 1)

    @staticmethod
    def get_params_names():
        """
        return: list of strings
        """
        return ["f_dist_x", "f_dist_y", "f_dist_z"]

    # @staticmethod
    # def positive_params():
    #     return True
    
    def positive_params(self):
        """
        return: tensor of bools
        """
        params = torch.tensor([
            False, # f_dist_x
            False, # f_dist_y
            False, # f_dist_z
        ], dtype=torch.bool)
        
        indexes = torch.arange(0, self.param_count, dtype=torch.long)
        indexes = indexes[params]
        return indexes

    def free_params(self):
        """
        return: tensor of bools
        """
        params = torch.tensor([
            True, # f_dist_x
            True, # f_dist_y
            True, # f_dist_z
        ], dtype=torch.bool)
        
        indexes = torch.arange(0, self.param_count, dtype=torch.long)
        indexes = indexes[params]
        return indexes  


    def params_specification(self):
        """
        return: option dictionary
        
        Option type
            0: params not regressed
            1: params regresed const
            2: params with trajectory as scaler of const
            3: params with trajectory from zero to max
            
        Positive type
            0: params not positive
            1: params allways positive
        """
        return NotImplementedError
        