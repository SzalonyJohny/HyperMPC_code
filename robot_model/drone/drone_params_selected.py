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
    def __init__(self,
                 ball_mass = 0.5,
                 enable_forces = False,
                 enable_torques = False, *args, **kwargs) -> None:
        super(DroneParamsWrapper, self).__init__(*args, **kwargs)
        self.param_count = self.get_default_params().shape[-1]
        
        self.register_buffer("mq", torch.tensor([1.325 + ball_mass], dtype=torch.float32))
        self.register_buffer("g0", torch.tensor([9.80665], dtype=torch.float32))
        self.register_buffer("l", torch.tensor([0.228035], dtype=torch.float32))
        
        f_e, tau_e = float(enable_forces), float(enable_torques)
        self.register_buffer("f_dist_scaler", torch.tensor([f_e], dtype=torch.float32))
        self.register_buffer("tau_dist_scaler", torch.tensor([tau_e], dtype=torch.float32))
        
        
    def forward(self, p):
        p_this_layer = p[..., :self.param_count]

        named_tuple = DroneParams(
            mq=self.mq,
            Db=p_this_layer[..., 0],
            Cd=p_this_layer[..., 1],
            Ct=p_this_layer[..., 2],
            g0=self.g0,
            Jx=p_this_layer[..., 3],
            Jy=p_this_layer[..., 4],
            Jz=p_this_layer[..., 5],
            l=self.l,
            f_dist_x=p_this_layer[..., 6] * self.f_dist_scaler,
            f_dist_y=p_this_layer[..., 7] * self.f_dist_scaler,
            f_dist_z=p_this_layer[..., 8] * self.f_dist_scaler,
            tau_dist_x=p_this_layer[..., 9] * self.tau_dist_scaler,
            tau_dist_y=p_this_layer[..., 10] * self.tau_dist_scaler,
            tau_dist_z=p_this_layer[..., 11] * self.tau_dist_scaler
        )
        
        return named_tuple, p[..., self.param_count:]

    @staticmethod
    def get_default_params(batch_size=1):
        default_params = [
            6.5330e-02,        # Db
            3.0288e-05,        # Cd
            8.2688e-01,        # Ct
            5.2578e-02,        # Jx
            4.3686e-02,        # Jy
            5.9212e-03,        # Jz 
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
        return ["Db" , "Cd", "Ct", "Jx", "Jy", "Jz", "f_dist_x", "f_dist_y", "f_dist_z", "tau_dist_x", "tau_dist_y", "tau_dist_z"]

    # @staticmethod
    # def positive_params():
    #     return True
    
    def positive_params(self):
        """
        return: tensor of bools
        """
        params = torch.tensor([
            True, # Db
            True, # Cd
            True, # Ct
            True, # Jx
            True, # Jy
            True, # Jz
            False, # f_dist_x
            False, # f_dist_y
            False, # f_dist_z
            False, # tau_dist_x
            False, # tau_dist_y
            False  # tau_dist_z
        ], dtype=torch.bool)
        
        indexes = torch.arange(0, self.param_count, dtype=torch.long)
        indexes = indexes[params]
        return indexes

    def free_params(self):
        """
        return: tensor of bools
        """
        params = torch.tensor([
            False, # Db
            False, # Cd
            False, # Ct
            False, # Jx
            False, # Jy
            False, # Jz
            True, # f_dist_x
            True, # f_dist_y
            True, # f_dist_z
            True, # tau_dist_x
            True, # tau_dist_y
            True  # tau_dist_z
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
        