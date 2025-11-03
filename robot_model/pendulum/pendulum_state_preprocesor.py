import torch


class PendulumStatePreprocessor(torch.nn.Module):
    def __init__(self,
                 max_dq: float = 1.0,
                 max_u: float = 1.0,
                 compile: bool = False):
        super(PendulumStatePreprocessor, self).__init__()

        self.max_dq = max_dq
        self.max_u = max_u
        
        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True)

    def forward(self, xu):
        """
            x : States (batch_size, time_steps, observarions_channels)

            return: States with sin cos encodning of angle (batch_size, time_series_latent_size)
        """
        q, dq, u = torch.unbind(xu, dim=-1)
        sq = torch.sin(q)
        cq = torch.cos(q)
        
        dq = dq / self.max_dq
        u = u / self.max_u

        return torch.stack([sq, cq, dq, u], dim=-1)
