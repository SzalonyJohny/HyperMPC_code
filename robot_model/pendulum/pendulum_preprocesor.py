import torch


class PendulumObservationPreprocessor(torch.nn.Module):
    def __init__(self,
                 max_dq: float = 1.0,
                 max_backlash: float = 1.0,
                 max_d_backlash: float = 1.0,
                 max_u: float = 1.0,
                 backlash_only: bool = False,
                 state_only: bool = False,
                 compile: bool = False):
        super(PendulumObservationPreprocessor, self).__init__()

        self.max_dq = max_dq
        self.max_backlash = max_backlash
        self.max_d_backlash = max_d_backlash
        self.max_u = max_u
        self.backlash_only = backlash_only
        self.state_only = state_only
        
        assert not (backlash_only and state_only),\
            "backlash_only and state_only cannot be True at the same time"

        if compile:
            self.forward = torch.compile(self.forward, fullgraph=True)

    def forward(self, M):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: observations with sin cos encodning of angle (batch_size, time_series_latent_size)
        """
        q, dq, backlash, d_backlash, u = torch.unbind(M, dim=-1)
        sq = torch.sin(q)
        cq = torch.cos(q)
        
        dq = dq / self.max_dq
        backlash = backlash / self.max_backlash
        d_backlash = d_backlash / self.max_d_backlash
        u = u / self.max_u

        if self.backlash_only:
            return torch.stack([backlash, d_backlash, u], dim=-1)

        if self.state_only:
            return torch.stack([sq, cq, dq, u], dim=-1)
        
        return torch.stack([sq, cq, dq, backlash, d_backlash, u], dim=-1)
