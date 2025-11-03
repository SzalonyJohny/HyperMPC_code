import torch


class MlpTimeSeriesEncoder(torch.nn.Module):
    def __init__(self,
                 observations_preprocessor: torch.nn.Module,
                 observations_channels: int,
                 middle_layer_size: int,
                 observation_length: int,
                 time_series_latent_size: int,
                 layer_norm_1: bool,
                 layer_norm_4: bool,
                 compile: bool,
                 device: str):

        super(MlpTimeSeriesEncoder, self).__init__()

        self.device = torch.device(device)
        self.observation_length = observation_length
        self.observations_channels = observations_channels
        self.time_series_latent_size = time_series_latent_size
        self.observations_preprocessor = observations_preprocessor.to(self.device)

        mlp_layers = [
            torch.nn.Linear(observations_channels * observation_length, middle_layer_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(middle_layer_size, time_series_latent_size),
            torch.nn.LeakyReLU()
        ]

        if layer_norm_1:
            mlp_layers.insert(1, torch.nn.LayerNorm(middle_layer_size))
        
        if layer_norm_4:
            mlp_layers.insert(4, torch.nn.LayerNorm(time_series_latent_size))
        
        self.tse_mlp = torch.nn.Sequential(*mlp_layers).to(self.device)
        
        if compile:
            self.forward = torch.compile(self.forward)

    def forward(self, M):
        """
            M : observations (batch_size, time_steps, M_size)

            return: hidden_representation (batch_size, rnn_hidden_size)
        """
        M_prep = self.observations_preprocessor(M)
        M_cut = M_prep[:, -self.observation_length:, :]
        hn = self.tse_mlp(M_cut.view(-1, self.observations_channels * self.observation_length))
        return hn
