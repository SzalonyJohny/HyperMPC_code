import torch


class CarStateAndCtrlPreprocessor(torch.nn.Module):
    def __init__(self, max_velocity=1.0,
                       max_yaw_rate=1.0,
                       max_omega_wheels=1.0,
                       max_delta=1.0):
        super(CarStateAndCtrlPreprocessor, self).__init__()
        
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.max_omega_wheels = max_omega_wheels
        self.max_delta = max_delta

    def forward(self, M):
        """
            M : StateAndCtrl
        s (batch_size, time_steps, observarions_channels)

            return: StateAndCtrl
        s with sin cos encodning of angle (batch_size, time_series_latent_size)
        """
        v_x, v_y, r, firction, omega_wheels, delta = torch.unbind(M, dim=-1)
        v_x = v_x / self.max_velocity
        v_y = v_y / self.max_velocity
        r = r / self.max_yaw_rate
        omega_wheels = omega_wheels / self.max_omega_wheels
        delta = delta / self.max_delta
        return torch.stack([v_x, v_y, r, firction, omega_wheels, delta], dim=-1)
