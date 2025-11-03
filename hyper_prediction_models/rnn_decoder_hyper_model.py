import torch

class GruDecoderHyperModel(torch.nn.Module):
    def __init__(self,
                 time_series_encoder: torch.nn.Module,
                 future_prediction_encoder: torch.nn.Module,
                 pred_noise_sigma: float,
                 default_params: torch.Tensor,
                 activation: torch.nn.Module,
                 hp_size: int, # 32
                 predictions_botteckleneck_size: int, # 2
                 param_ctrl_point: int, # n
                 max_param_scaler: float,
                 last_ctrl_point_on_const: bool,
                 always_positive: bool,
                 compile_model: bool,
                 device: str,
                 disable_pred_channel_mixer: bool = True
                 ):

        super(GruDecoderHyperModel, self).__init__()
        
        self.device = torch.device(device)

        # sizes
        self.last_ctrl_point_on_const = last_ctrl_point_on_const
        self.param_ctrl_point = param_ctrl_point
        self.max_param_scaler = max_param_scaler
        self.always_positive = always_positive
        self.predictions_botteckleneck_size = predictions_botteckleneck_size
        
        self.param_size = default_params.shape[-1]
        self.future_predictions_channel_count = future_prediction_encoder.get_channel_count()
        self.pred_noise_sigma = pred_noise_sigma
        
         # FIXME 
        # hl1_size = time_series_encoder.time_series_latent_size
        
        self.activation = activation

        if self.always_positive:
            assert torch.all(default_params > 0.0)
            default_params = torch.log(default_params)

        self.default_params = default_params.to(self.device)
        
        # Not learnable modules
        self.future_predictions_encoder = future_prediction_encoder.to(self.device)
        
        # Learnable modules
        self.const_params = torch.nn.Parameter(default_params)
        
        self.time_series_encoder = time_series_encoder
        
        self.hidden_state_upscale = torch.nn.Linear(time_series_encoder.time_series_latent_size,
                                                    hp_size).to(self.device)

        if disable_pred_channel_mixer:
            self.predictions_channels_mixer = lambda x: x    
        else:
            self.predictions_channels_mixer = torch.nn.Linear(self.future_predictions_channel_count, 
                                                             predictions_botteckleneck_size, bias=True).to(self.device)
                        
        self.param_h_evo = torch.nn.GRU(hidden_size=hp_size,
                                      input_size=predictions_botteckleneck_size,
                                      num_layers=1,
                                      batch_first=True).to(self.device)
        
        self.param_h_to_out = torch.nn.Sequential(
            torch.nn.Linear(hp_size, hp_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hp_size, self.param_size)
        ).to(self.device)
        
        torch.nn.init.normal_(self.param_h_to_out[0].weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.param_h_to_out[0].bias)
        
        torch.nn.init.normal_(self.param_h_to_out[2].weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.param_h_to_out[2].bias)
    
        if compile_model is True:
            self.forward_without_tse = torch.compile(self.forward_without_tse)
            print("Compiling HyperModel")

    
    def forward(self, M, P, enable_prediction_head=True):
        hn = self.time_series_encoder(M)
        return self.forward_without_tse(hn, P, enable_prediction_head)

    def forward_without_tse(self, hn, P, enable_prediction_head):
        """
            M : observations (batch_size, observation window, observarions_channels)
            P : predictions  (batch_size, prediction horizon, prediction_channels)

            return: p, delta_p

            p_delta for L1 loss on increments   
        """
        # Past head
        hl1 = self.hidden_state_upscale(hn)
        
        # Future head
        spline_ctrl_points = self.future_predictions_encoder(P, self.pred_noise_sigma) # [batch, n, pred_chn]     
        l = self.predictions_channels_mixer(spline_ctrl_points) # [batch, n, pred_chn]
         
        if not enable_prediction_head:
            l = torch.zeros_like(l)
        
        hidden_evo, _ = self.param_h_evo(l, hl1.unsqueeze(0)) # [batch, n, hl1_size]
        
        fc_out = self.param_h_to_out(hidden_evo) # [batch, n, param_size]
        
        fc_out = fc_out.view(-1, self.param_ctrl_point * self.param_size)
  
        fc_out = 1.0 + torch.nn.Tanh()(fc_out) * self.max_param_scaler

        p_delta = fc_out.view(-1, self.param_ctrl_point, self.param_size)

        if self.last_ctrl_point_on_const:
            p_last = torch.ones(hn.shape[0], self.param_size, device=self.device)
            # FIXME p_delta = torch.cat((p_delta, p_last), dim=1)
            p_delta[:, -1, :] = p_last

        if self.always_positive:
            p_const = self.const_params.exp()
        else:
            p_const = self.const_params

        p_ret = p_delta * p_const

        return p_ret, p_delta
    
    def get_past_encoder_params(self):
        param_list = []
        for module in self.list_of_modules_to_encode_past:
            param_list += list(module.parameters())
        return param_list
    
    def get_inactive_parameter_count(self):
        return  0