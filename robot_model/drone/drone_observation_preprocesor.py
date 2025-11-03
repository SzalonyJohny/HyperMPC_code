import torch


class DroneObservationPreprocessor(torch.nn.Module):
    def __init__(self, max_bxy=4.0, max_bz=4.0, max_v=20.0):
        super(DroneObservationPreprocessor, self).__init__()
        self.register_buffer('max_bxy', torch.tensor([max_bxy], dtype=torch.float32))
        self.register_buffer('max_bz', torch.tensor([max_bz], dtype=torch.float32))
        self.register_buffer('max_v', torch.tensor([max_v], dtype=torch.float32))
        
    def forward(self, M):
        """
            M : observations (batch_size, time_steps, observarions_channels)

            return: observations with sin cos encodning of angle (batch_size, time_series_latent_size)
        """
        qw, qx, qy, qz, vx, vy, vz, bx, by, bz = torch.unbind(M, dim=-1)

        return torch.stack([
            qw, qx, qy, qz,
            vx / self.max_v,
            vy / self.max_v,
            vz / self.max_v,
            bx / self.max_bxy,
            by / self.max_bxy,
            bz / self.max_bz
        ], dim=-1)        
        
        
