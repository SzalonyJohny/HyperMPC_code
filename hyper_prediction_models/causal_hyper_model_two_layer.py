import torch
from hyper_prediction_models.causal_masked_mlp import CausalMaskedLinear

class CausalHyperModel2(torch.nn.Module):
    def __init__(self,
                 time_series_encoder: torch.nn.Module,
                 future_prediction_encoder: torch.nn.Module,
                 pred_noise_sigma: float,
                 default_params: torch.Tensor,
                 activation: torch.nn.Module,
                 hl2_size: int, # 32
                 m1_feature_size: int, # param_size / 2
                 param_ctrl_point: int, # n
                 max_param_scaler: float,
                 last_ctrl_point_on_const: bool,
                 always_positive: bool,
                 compile_model: bool,
                 device: str,
                 triangular_block_size: int = -1, # -1 for disabled
                 disable_pred_channel_mixer: bool = False, # not used
                 ):

        super(CausalHyperModel2, self).__init__()

        self.device = torch.device(device)

        # sizes
        self.last_ctrl_point_on_const = last_ctrl_point_on_const
        self.param_ctrl_point = param_ctrl_point
        self.max_param_scaler = max_param_scaler
        self.always_positive = always_positive
        self.m1_feature_size = m1_feature_size
        self.param_size = default_params.shape[-1]
        self.future_predictions_channel_count = future_prediction_encoder.get_channel_count()
        self.pred_noise_sigma = pred_noise_sigma
        
         # FIXME 
        # hl1_size = time_series_encoder.time_series_latent_size
        hl1_size = 32
        
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
                                                    hl1_size).to(self.device)

    
        self.hl1_to_hl2 = torch.nn.Linear(hl1_size, hl2_size).to(self.device)
        
        # [hl1, l] -> m1
        self.causal_mlp_1 = CausalMaskedLinear(self.param_ctrl_point,
                                               self.future_predictions_channel_count,
                                               self.m1_feature_size,
                                               triangular_block_size=triangular_block_size,
                                               bias=False).to(self.device)
        
        self.hl1_to_m1 = torch.nn.Linear(hl1_size,
                                         self.m1_feature_size * self.param_ctrl_point,
                                         bias=True).to(self.device)

        # [hl2, m1] -> params
        self.causal_mlp_2 = CausalMaskedLinear(self.param_ctrl_point,
                                               self.m1_feature_size,
                                               self.param_size,
                                               triangular_block_size=triangular_block_size, 
                                               bias=False).to(self.device)
        
        self.hl2_to_params = torch.nn.Linear(hl2_size,
                                             self.param_size * self.param_ctrl_point, 
                                             bias=True).to(self.device)
        
        # Initialize weights of last layer to be close to 0, 
        # so that the initial predictions are close to the default parameters         
        
        # from prediction head
        torch.nn.init.normal_(self.causal_mlp_2.linear.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.causal_mlp_1.linear.weight, mean=0.0, std=0.001)
        
        # from past encoder
        torch.nn.init.normal_(self.hl1_to_hl2.weight, mean=0.0, std=0.001)
        torch.nn.init.normal_(self.hl2_to_params.weight, mean=0.0, std=0.001)
        self.hl1_to_hl2.bias.data.fill_(0)
        self.hl2_to_params.bias.data.fill_(0)
        
    
        if compile_model is True:
            self.forward_without_tse = torch.compile(self.forward_without_tse, fullgraph=True)
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
        hl1_act = self.activation(hl1)
        
        # Future head
        spline_ctrl_points = self.future_predictions_encoder(P, self.pred_noise_sigma) # [batch, n, pred_chn]     
        l = spline_ctrl_points.view(-1, self.param_ctrl_point * self.future_predictions_channel_count)
        
        if not enable_prediction_head:
            l = torch.zeros_like(l)
        
        # Causal first layer
        m1 = self.causal_mlp_1(l) + self.hl1_to_m1(hl1_act)
        m1_act = self.activation(m1)

        # latent past state evolution
        hl2 = self.hl1_to_hl2(hl1_act)
        hl2_act = self.activation(hl2)

        # Causal second layer
        fc_out = self.causal_mlp_2(m1_act) + self.hl2_to_params(hl2_act)

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
        lc1 = self.causal_mlp_1.inactive_parameters_count()
        lc2 = self.causal_mlp_2.inactive_parameters_count()
        return lc2 + lc1