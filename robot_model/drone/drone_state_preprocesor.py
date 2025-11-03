import torch


class DroneStateCtrlPreprocessor(torch.nn.Module):
    def __init__(self, max_bxy=4.0, max_bz=4.0, max_v=20.9, max_u=13.0):
        super(DroneStateCtrlPreprocessor, self).__init__()
        self.register_buffer('max_bxy', torch.tensor([max_bxy], dtype=torch.float32))
        self.register_buffer('max_bz', torch.tensor([max_bz], dtype=torch.float32))
        self.register_buffer('max_v', torch.tensor([max_v], dtype=torch.float32))
        self.register_buffer('max_u', torch.tensor([max_u], dtype=torch.float32))
        
    def forward(self, x_u):
        """
            M : States (batch_size, time_steps, observarions_channels)

            return: States obs prep (batch_size, time_series_latent_size)
        """
        x, y, z, qw, qx, qy, qz, vx, vy, vz, bx, by, bz, u1, u2, u3, u4 = torch.unbind(x_u, dim=-1)

        return torch.stack([
            qw, qx, qy, qz,
            vx / self.max_v,
            vy / self.max_v,
            vz / self.max_v,
            bx / self.max_bxy,
            by / self.max_bxy,
            bz / self.max_bz,
            # u1 / self.max_u,
            # u2 / self.max_u,
            # u3 / self.max_u,
            # u4 / self.max_u
        ], dim=-1)        
        
        
