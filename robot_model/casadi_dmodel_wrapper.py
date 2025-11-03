import torch


class CasadiDynamicsModelWraper(torch.nn.Module):
    def __init__(self, model, state_dim, control_dim):
        super(CasadiDynamicsModelWraper, self).__init__()
        self.model = model
        self.state_dim = state_dim
        self.control_dim = control_dim

    def forward(self, txup):
        t = txup[:, 0]
        x = txup[:, 1:1+self.state_dim]
        u = txup[:, 1+self.state_dim:1 + self.state_dim + self.control_dim]
        p = txup[:, self.state_dim+self.control_dim+1:]
        return self.model(t, x, u, p)
