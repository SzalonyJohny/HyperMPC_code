import torch
import collections
from robot_model.car.state_wrapper import StateWrapper

SingleTrackParamsList = ['m', 'g', 'I_z', 'L', 'lr', 'lf', 'Cd0', 'Cd2', 'Cd1',
                         'mu_static', 'I_e', 'K_fi', 'b1', 'b0', 'R', 'eps']
SingleTrackParams = collections.namedtuple('SingleTrackParams', SingleTrackParamsList,
                                           defaults=(None,) * len(SingleTrackParamsList))

class VehicleParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(VehicleParameters, self).__init__()
        self.param_count = VehicleParameters.default_params_tensor().shape[-1]
        
        # Define constant tensors for detached parameters
        self.m_ = torch.tensor([5.1], requires_grad=False)
        self.g_ = torch.tensor([9.81], requires_grad=False)
        self.L_ = torch.tensor([0.33], requires_grad=False)
        self.mu_static_ = torch.tensor([0.8], requires_grad=False)
        self.eps_ = torch.tensor([1e-6], requires_grad=False)
        
        self.register_buffer('m', self.m_)
        self.register_buffer('g', self.g_)
        self.register_buffer('L', self.L_)
        self.register_buffer('mu_static', self.mu_static_)
        self.register_buffer('eps', self.eps_)        
        

    def forward(self, p: torch.Tensor) -> SingleTrackParams:
        p_this_layer = p[..., :self.param_count]
        
        named_tuple = SingleTrackParams(
            m=self.m,
            g=self.g,
            I_z=p_this_layer[..., 0],
            L=self.L,
            lr=p_this_layer[..., 1],
            lf=self.L - p_this_layer[..., 1],  # lf = L - lr
            Cd0=p_this_layer[..., 2],
            Cd2=p_this_layer[..., 3],
            Cd1=p_this_layer[..., 4],
            mu_static=self.mu_static,
            I_e=p_this_layer[..., 5],
            K_fi=p_this_layer[..., 6],
            b1=p_this_layer[..., 7],
            b0=p_this_layer[..., 8],
            R=p_this_layer[..., 9],
            eps=self.eps
        )
        return named_tuple, p[..., self.param_count:]
    

    @staticmethod
    def default_params_tensor(batch_size=1):
        return torch.tensor([
            0.46,  # I_z
            0.115,  # lr
            0.01,  # Cd0
            0.01,  # Cd2
            0.01,  # Cd1
            0.2,  # I_e
            0.90064745,  # K_fi
            0.304115174,  # b1
            0.50421894,  # b0
            0.05,  # R
        ]).unsqueeze(0).repeat(batch_size, 1)
        
    @staticmethod
    def get_params_names():
        return [
            'I_z',
            'lr',
            'Cd0',
            'Cd2',
            'Cd1',
            'I_e',
            'K_fi',
            'b1',
            'b0',
            'R'
        ]
        
    
    