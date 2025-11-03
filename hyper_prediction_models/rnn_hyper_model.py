import torch


class HyperModel(torch.nn.Module):
    def __init__(self,
                 time_series_encoder: torch.nn.Module,
                 default_params: torch.Tensor,
                 activation: torch.nn.Module,
                 middle_layer_size: int,
                 param_ctrl_point: int,
                 max_param_scaler: float,
                 last_ctrl_point_on_const: bool,
                 always_positive: bool,
                 compile_model: bool,
                 device: str):

        super(HyperModel, self).__init__()

        self.device = torch.device(device)

        self.last_ctrl_point_on_const = last_ctrl_point_on_const
        self.param_ctrl_point = param_ctrl_point
        self.max_param_scaler = max_param_scaler
        self.always_positive = always_positive

        if always_positive:
            default_params = torch.log(default_params)

        self.default_params = default_params.to(self.device)
        
        self.const_params = torch.nn.Parameter(default_params)

        self.param_size = default_params.shape[-1]

        self.time_series_encoder2 = time_series_encoder
        
        
        self.fc = None
        if middle_layer_size == 0:
            self.fc = torch.nn.Linear(time_series_encoder.time_series_latent_size,
                                        param_ctrl_point * self.param_size).to(self.device)
            
            torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.001)
        else:    
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(time_series_encoder.time_series_latent_size,
                                middle_layer_size),
                activation,
                torch.nn.Linear(middle_layer_size,
                                param_ctrl_point * self.param_size)
            ).to(self.device)            

            torch.nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.001)
            torch.nn.init.normal_(self.fc[2].weight, mean=0.0, std=0.001)

        if compile_model is True:
            self.forward = torch.compile(self.forward)
            print("Compiling HyperModel")

    def forward(self, M, P=None, enable_prediction_head=True):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: p, delta_p

            p_delta for L1 loss on increments   
        """

        hn = self.time_series_encoder2(M)

        fc_out = self.fc(hn)

        fc_out = 1.0 + torch.nn.Tanh()(fc_out) * self.max_param_scaler

        p_delta = fc_out.view(-1, self.param_ctrl_point, self.param_size)

        if self.last_ctrl_point_on_const:
            p_last = torch.ones(hn.shape[0], self.param_size).to(self.device)
            # FIXME p_delta = torch.cat((p_delta, p_last), dim=1)
            p_delta[:, -1, :] = p_last
            

        if self.always_positive:
            p_const = self.const_params.exp()
        else:
            p_const = self.const_params

        p_ret = p_delta * p_const

        return p_ret, p_delta

    def get_inactive_parameter_count(self):
        return 0