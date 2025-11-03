import torch


class RnnTimeSeriesEncoder(torch.nn.Module):
    def __init__(self,
                 observations_preprocessor: torch.nn.Module,
                 observarions_channels: int,
                 time_series_latent_size: int,
                 rnn_layers: int,
                 rnn_cell: str, # gru, lstm
                 compile_model: bool,
                 device: str):

        super(RnnTimeSeriesEncoder, self).__init__()

        
        self.device = torch.device(device)
        self.time_series_latent_size = time_series_latent_size
        self.rnn = None 
        self.observations_preprocessor = observations_preprocessor.to(self.device)

        if rnn_cell == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=observarions_channels,
                hidden_size=time_series_latent_size,
                num_layers=rnn_layers,
                batch_first=True,
                bias=True,
                dropout=0,
                bidirectional=False)
        elif rnn_cell == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=observarions_channels,
                hidden_size=time_series_latent_size,
                num_layers=rnn_layers,
                batch_first=True,
                bias=True,
                dropout=0,
                bidirectional=False)
        else:
            raise ValueError(f"Invalid rnn_cell: {rnn_cell}")

        self.rnn = self.rnn.to(self.device)

        if compile_model:
            print("compiling Rnn time series encoder")
            self.forward = torch.compile(self.forward)

    def forward(self, M):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: time series latent representation (batch_size, time_series_latent_size)
        """
        M_prep = self.observations_preprocessor(M)
        rnn_out, _ = self.rnn(M_prep)
        hn = rnn_out[:, -1, :]
        return hn
