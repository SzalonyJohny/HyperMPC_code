import torch
from robot_model.acrobot.acrobot_params import ActobotParams


class PendulumDynamisc(torch.nn.Module):
    def __init__(self,
                 init_params=None,
                 skip_gr=False,
                 *args, **kwargs) -> None:
        super(PendulumDynamisc, self).__init__(*args, **kwargs)

        self.skip_gr = skip_gr
        self.init_params = init_params
    
    def get_default_params(self, batch_size=1):
        if self.init_params is not None:
            return self.init_params
        return torch.tensor([1.0, 1.0, 0.2, 0.001, 1.0]).unsqueeze(0).repeat(batch_size, 1)
    
    def positive_params(self):
        l = self.get_default_params().shape[-1]
        return torch.arange(l)
    
    def free_params(self):
        return None
            
    def get_params_names(self):
        return ['m', 'l', 'b', 'f', 'gr']
    
    @staticmethod
    def get_state_names():
        return ["q", "dq"]
    
    @staticmethod
    def get_control_names():
        return ["u"]

    @staticmethod
    def save_param_traj():
        return True

    @staticmethod
    def state_weights():
        return torch.tensor([
            1.0,  # q
            1.0,  # dq
        ])
        

    def forward(self, t, x, u, p):
        """
            t = [batch]
            x = [batch, state dim -> 2]
            u = [batch, torque -> 1]
            p = [batch, param_count]
        """
        assert x.shape[-1] == 2
        assert u.shape[-1] == 1
        
        q, dq = torch.unbind(x, dim=-1)
        
        q = q - torch.pi # offset to have 0 at the bottom
        
        m, l, b, f, gr = torch.unbind(p, dim=-1)
        
        g = 9.81
        sign_tanh_aprox = 30.0
        
        if self.skip_gr:
            gr = 1.0
            
        # Compute dynamics
        u_with_loss = gr * u[..., 0] - b * dq - f * torch.tanh(sign_tanh_aprox * dq)
        ddq = 3 * g / (2 * l) * torch.sin(q) + 3.0 / (m * l**2) * u_with_loss
        
        return torch.stack([dq, ddq], dim=-1)
    

if __name__ == "__main__":
    
    model = PendulumDynamisc()
    
    p = model.get_default_params(1)
    
    def rk4_step(x, u, p, dt):
        k1 = model(0, x, u, p)
        k2 = model(0, x + k1 * dt / 2, u, p)
        k3 = model(0, x + k2 * dt / 2, u, p)
        k4 = model(0, x + k3 * dt, u, p)
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

    x0i = torch.tensor([[0.0, 0.0]])  # [ q, dq]
    rollout_len = 1000
    dt = 0.01
    x_save = torch.ones(1, rollout_len, 2)
    
    u = torch.ones(1, rollout_len, 1)
    p = model.get_default_params(1)
    
    with torch.inference_mode():
        for i in range(rollout_len):
            x_save[:, i, :] = x0i
            x0i = rk4_step(x0i, u[:, i, :], p, dt)
                           
    # plot the results
    import matplotlib.pyplot as plt
    plt.plot(x_save[0, :, 0].numpy(), label='q')
    plt.plot(x_save[0, :, 1].numpy(), label='dq')
    plt.legend()
    plt.show()
    