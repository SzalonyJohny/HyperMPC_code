import torch


class ConstParamModel(torch.nn.Module):
    def __init__(self,
                 time_series_encoder: torch.nn.Module, # not for compatibility
                 default_params: torch.Tensor,
                 always_positive: bool, 
                 free_params: torch.Tensor, # params that are not scalers
                 device) -> None:
        
        super(ConstParamModel, self).__init__()
        
        self.device = torch.device(device)
        self.always_positive = always_positive
        
        default_params = default_params.to(self.device).clone()

        assert torch.is_tensor(always_positive)

        default_params[:, always_positive] = torch.log(default_params[:, always_positive])
            
        self.p = torch.nn.Parameter(default_params.clone().unsqueeze(0))
        assert len(self.p.shape) == 3
        # dim: [batch, time, param_size]

    def forward(self, M, P=None, enable_prediction_head=True):
        batch_size = M.shape[0]
        
        p = self.p.expand(batch_size, -1, -1)
        p_exp = p.clone()
        p_exp[..., self.always_positive] = p[..., self.always_positive].exp().clone()
        return p_exp, torch.ones_like(p)
        
    def get_inactive_parameter_count(self):
        return 0