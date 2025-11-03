import torch



class TCNNTimeSeriesEncoder(torch.nn.Module):
    def __init__(self,
                 observations_preprocessor: torch.nn.Module,
                 observations_channels: int,
                 filter_count: int,
                 kernel_len: int,
                 observation_length: int,
                 time_series_latent_size: int,
                 layer_norm: bool,
                 compile: bool,
                 device: str):

        super(TCNNTimeSeriesEncoder, self).__init__()

        self.device = torch.device(device)
        self.observation_length = observation_length
        self.observations_channels = observations_channels
        self.time_series_latent_size = time_series_latent_size
        self.observations_preprocessor = observations_preprocessor.to(self.device)
        

        self.conv = torch.nn.Conv2d(1, filter_count, (kernel_len, observations_channels))
        
        linear_input_size = filter_count * (observation_length - kernel_len + 1)
        self.linear = torch.nn.Linear(linear_input_size,
                                      time_series_latent_size)        

        self.activation = torch.nn.LeakyReLU()

        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(linear_input_size)
        else:
            self.layer_norm = lambda x: x

        if compile:
            print("Compiling TCNNTimeSeriesEncoder")
            self.forward = torch.compile(self.forward, fullgraph=True)

    def forward(self, M):
        """
            M : observations (batch_size, time_steps, M_size)

            return: time series latent representation (batch_size, time_series_latent_size)
        """
        M_prep = self.observations_preprocessor(M)
        
        M_cut = M_prep[:, -self.observation_length:, :]
        
        M_cut = M_cut.unsqueeze(1)
        
        conv = self.conv(M_cut)
        
        conv_flat = conv.view(conv.shape[0], -1)
        
        ln_conv = self.layer_norm(conv_flat)
        
        ln_conv_act = self.activation(ln_conv)
        
        hn = self.linear(ln_conv_act)
                
        return hn