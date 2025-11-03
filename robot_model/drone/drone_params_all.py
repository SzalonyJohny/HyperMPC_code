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

class DroneParamsWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DroneParamsWrapper, self).__init__(*args, **kwargs)
        self.param_count = len(DroneParamsList)
        self.dist_scaler = torch.tensor([0.0], dtype=torch.float32)
        

    def forward(self, p):
        p_this_layer = p[..., :self.param_count]

        named_tuple = DroneParams(
            mq=p_this_layer[..., 0],
            Db=p_this_layer[..., 1],
            Cd=p_this_layer[..., 2],
            Ct=p_this_layer[..., 3],
            g0=p_this_layer[..., 4],
            Jx=p_this_layer[..., 5],
            Jy=p_this_layer[..., 6],
            Jz=p_this_layer[..., 7],
            l=p_this_layer[..., 8],
            f_dist_x=p_this_layer[..., 9] * self.dist_scaler,
            f_dist_y=p_this_layer[..., 10] * self.dist_scaler,
            f_dist_z=p_this_layer[..., 11] * self.dist_scaler,
            tau_dist_x=p_this_layer[..., 12] * self.dist_scaler,
            tau_dist_y=p_this_layer[..., 13] * self.dist_scaler,
            tau_dist_z=p_this_layer[..., 14] * self.dist_scaler
        )
        
        return named_tuple, p[..., self.param_count:]

    @staticmethod
    def get_default_params(batch_size=1):
        # CrazyFlie 2.1 parameters
        default_params = [
            1.525,             # mq
            0.0,               # Db
            7.9379e-06,        # Cd
            3.25e-4,           # Ct
            9.80665,           # g0
            1.395e-3,          # Jx
            1.395e-3,          # Jy
            2.173e-3,          # Jz
            92e-3 / 2,         # l
            0.0,               # f_dist_x
            0.0,               # f_dist_y
            0.0,               # f_dist_z
            0.0,               # tau_dist_x
            0.0,               # tau_dist_y
            0.0                # tau_dist_z
        ]
        tensor = torch.tensor(default_params, dtype=torch.get_default_dtype())
        return tensor.unsqueeze(0).repeat(batch_size, 1)

    @staticmethod
    def get_params_names():
        """
        return: list of strings
        """
        return DroneParamsList

    @staticmethod
    def positive_params():
        return True
    

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
        