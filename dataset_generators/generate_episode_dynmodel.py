import torch
from hyper_prediction_models.rollout_model import RolloutModel
from robot_model.cartpole.cartpole_model import CartPole
import dataset_generators.sample_control as sc
import pandas as pd


class CartPoleSimTorch:
    
    def __init__(self, config: dict) -> None:
        # start with default settings, update with any provided configuration
        self.config = config
        
        Ts = self.config["Ts"]
        f = int(1 / Ts)
        episode_len_s = self.config["episode_len_s"]
        n_frames = int(episode_len_s * f)
        u_max = self.config["u_max"]
        u_max_Bspline = u_max * self.config["u_max_Bspline_factor"]
        
        self.n_frames = n_frames
        self.Ts = Ts
        self.u_max = u_max
        
        self.v1_init_range = self.config["v1_init_range"]
        self.dtheta_init_range = self.config["dtheta_init_range"]
        self.batch_size = self.config["batch_size"]
        
        print("n_frames: ", n_frames)
        
        self.control_sampler = sc.ControlVectorSampler(
            framerate=f,
            T_end=episode_len_s,
            control_points_count=self.config["control_points_count"],
            max_value=u_max_Bspline
        )
        
        self.dyn_model = CartPole()
        
        self.rollout_model = RolloutModel(
            dyn_model=self.dyn_model,
            intergration_method=self.config["integration_method"],
            compile=self.config["compile"],
            chunk_mode=self.config["chunk_mode"],
            chunk_size=self.config["chunk_size"],
            Tp=Ts
        )
        
        self.p = self.dyn_model.get_default_params().unsqueeze(0)
        self.p = self.p.repeat(self.batch_size, self.n_frames * 2, 1)
        print(f"p shape: {self.p.shape}")
        
        self.collumns = (
            ["t"] +
            self.dyn_model.get_state_names() +
            self.dyn_model.get_control_names()
        )
        
    @staticmethod
    def sample_uniform(low, high, size):
        return torch.rand(size) * (high - low) + low

    def sample_X0(self):
        x0 = torch.zeros(self.batch_size, 4)
        x0[:, 1] = self.sample_uniform(-torch.pi, torch.pi, self.batch_size)
        x0[:, 2] = self.sample_uniform(-self.v1_init_range, self.v1_init_range, self.batch_size) * 0.0
        x0[:, 3] = self.sample_uniform(-self.dtheta_init_range, self.dtheta_init_range, self.batch_size)
        return x0

    def sample_control(self):
        u = torch.zeros(self.batch_size, self.n_frames)
        for i in range(self.batch_size):
            u_temp = self.control_sampler.sample()
            u_temp = torch.tensor(u_temp, dtype=torch.float32).unsqueeze(0)
            u_temp = torch.clamp(u_temp, -self.u_max, self.u_max)
            u[i] = u_temp.clone()
        return u
        
    def generate_episode(self):
        x0 = self.sample_X0().unsqueeze(1)
        u = self.sample_control()
        x_traj = self.rollout_model(x0, u, self.p, self.n_frames)
        
        # take episode with minimum v1 as high u - quickly accelerates the cart
        v1_vals = x_traj[:, :, 2].abs().max(dim=1).values
        _, min_ixd = torch.min(v1_vals, dim=0)
        
        x_traj = x_traj[min_ixd]
        u_traj = u[min_ixd].unsqueeze(-1)
        t = torch.arange(0, self.n_frames).unsqueeze(-1).float() * self.Ts

        xut_traj = torch.cat([t, x_traj, u_traj], dim=-1)
        
        df = pd.DataFrame(xut_traj, columns=self.collumns)
        return df        
        
if __name__ == "__main__":
    
    config = {
        "Ts": 0.01,
        "episode_len_s": 5,
        "control_points_count": 20,
        "u_max": 20.0,
        "u_max_Bspline_factor": 2.0,
        "v1_init_range": 10.0 / 5.0,
        "dtheta_init_range": 5.0 / 2.0,
        "batch_size": 16,
        "integration_method": "rk4",
        "compile": True,
        "chunk_mode": False,
        "chunk_size": 10, 
        "number_of_runs": 180,
    }
    sim = CartPoleSimTorch(config)
    df = None
    for i in range(10):
        df = sim.generate_episode()
    print(df)
    print("done")
    
    from robot_model.cartpole.visualize_df import visualize_cartpole
    visualize_cartpole(df, show=True)