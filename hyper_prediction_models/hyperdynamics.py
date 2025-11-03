import torch


class HyperDynamicsModel(torch.nn.Module):
    def __init__(self,
                 time_series_encoder: torch.nn.Module,
                 default_params: torch.Tensor,
                 always_positive: bool,
                 compile_model: bool,
                 device: str,
                 middle_layer_size: int = 128, 
                 free_params = None):

        super(HyperDynamicsModel, self).__init__()

        self.device = torch.device(device)

        # self.const_params = torch.nn.Parameter(default_params)
        self.register_buffer("const_params", default_params)

        self.param_size = default_params.shape[-1]

        self.time_series_encoder = time_series_encoder

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(time_series_encoder.time_series_latent_size,
                            middle_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_layer_size,
                            self.param_size)
        ).to(self.device)
        
        for i in range(len(self.fc)):
            if hasattr(self.fc[i], 'weight') and self.fc[i].weight is not None:
                torch.nn.init.normal_(self.fc[i].weight, mean=0.0, std=0.001)
                torch.nn.init.zeros_(self.fc[i].bias)

        if compile_model is True:
            self.forward = torch.compile(self.forward)
            print("Compiling HyperModel")

    def forward(self, M, P=None, enable_prediction_head=True):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: p, delta_p

            p_delta for L1 loss on increments DISABLED for this model  
        """

        hn = self.time_series_encoder(M)

        p_ret = self.fc(hn).view(-1, 1, self.param_size) + self.const_params

        return p_ret, torch.ones_like(p_ret)

    def get_inactive_parameter_count(self):
        return 0
