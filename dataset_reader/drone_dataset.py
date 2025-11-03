import torch
import numpy as np
import pandas as pd
from pathlib import Path


class DroneHyperDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 observation_window: int, # in samples
                 prediction_horizon: int, # in samples
                 filter_pwm: bool,
                 moving_window_median_filter: int, # in samples
                 stride: int, # in samples
                 device: str):
        self.df = pd.read_csv(dataset_path)

        # Observation columns
        M_cols = ['qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz']
        
        # Control columns
        U_cols = ['pwm_m1', 'pwm_m2', 'pwm_m3', 'pwm_m4']
        
        # State vector columns
        X_cols = ['x', 'y', 'z',
                  'qw', 'qx', 'qy', 'qz',
                  'vx', 'vy', 'vz',
                  'brx', 'bry', 'brz']
        
        # Future prediction columns
        P_cols = ['pwm_m1', 'pwm_m2', 'pwm_m3', 'pwm_m4']
        
        for el in U_cols:
            if filter_pwm:
                self.df[el] = self._pwm_to_omega(self.df[el].values)
                self.df[el] = self._moving_median(self.df[el].values, moving_window_median_filter)
        
        self.M = torch.from_numpy(self.df[M_cols].to_numpy()).float().contiguous()
        self.U = torch.from_numpy(self.df[U_cols].to_numpy()).float().contiguous()
        self.X = torch.from_numpy(self.df[X_cols].to_numpy()).float().contiguous()
        self.P = torch.from_numpy(self.df[P_cols].to_numpy()).float().contiguous()

        index_list = []
        droped_data = 0
        
        for i in range(observation_window, len(self.df) - prediction_horizon, stride):
    
            # i - current time sample      
            t_minus_to = i - observation_window # start of observation window
            t_plus_tp = i + prediction_horizon # end of prediction horizon
            
            if self.df["run_id"].iloc[t_minus_to] != self.df["run_id"].iloc[i + prediction_horizon]:
                droped_data += 1
                continue
                        
            index_list.append((t_minus_to, i, t_plus_tp))

        print(f"Droped data: {droped_data / (len(index_list)+droped_data) * 100}%")
                
        self.index_tensor = torch.tensor(index_list, dtype=torch.long)
        self.length = (self.index_tensor).shape[0]
        
        # index_tensor and data to device
        self.index_tensor = self.index_tensor.to(device).contiguous()
        self.M = self.M.to(device)
        self.U = self.U.to(device)
        self.X = self.X.to(device)
        self.P = self.P.to(device)
        print(f"Dataset length: {self.length}")
        print(f"Observation window: {self.M.shape}")
        
        # df max abs values
        self.max_values = self.df.abs().max().to_dict() 
        print(f"Max values: {self.max_values}")
        
        # dt
        dt = np.median(np.diff(self.df['t'].values))
        dt_std = np.diff(self.df['t'].values).std()
        print(f"1/dt: {1/dt},  dt: {dt} +/- {dt_std}")
        
        
    @staticmethod   
    def _pwm_to_omega(pwm):
        return (pwm * 0.2685 - -4070.3) / 1000
    
    @staticmethod   
    def _moving_median(a, window_size):
        pad = window_size // 2
        a_padded = np.pad(a, (pad, pad), mode='edge')
        return np.array([np.median(a_padded[i:i+window_size]) for i in range(len(a))])

    
    def plot_dataset(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(self.df['x'], self.df['y'], self.df['z'], label='Position', color='b')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("3D Pose")
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.set_title("Motor krpm as f(PWM)")
        ax2.plot(self.df['t'].values, self.df['pwm_m1'], label='m1')
        ax2.plot(self.df['t'].values, self.df['pwm_m2'], label='m2')
        ax2.plot(self.df['t'].values, self.df['pwm_m3'], label='m3')
        ax2.plot(self.df['t'].values, self.df['pwm_m4'], label='m4')

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('PWM')
        ax2.set_title('Motor PWM')
        ax2.legend()
        ax2.grid()
            
        plt.tight_layout()
        plt.show()

    
    #####################################################################################
    #                                   Dataset interface                               #
    #####################################################################################


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index 

        Returns:
            torch.Tensor: M      [time_dim, measurement_dim] -> time_dim = observation window
            torch.Tensor: U      [time_dim, control_dim]
            torch.Tensor: X0     [1, state_dim]

            torch.Tensor: X      [time, state_dim] -> time_dim = prediciton horizon
        """
        t_minus_to, t, t_plus_tp = self.index_tensor[idx, :].unbind(-1)
        
        return self.M[t_minus_to:t, :], \
               self.P[t:t_plus_tp, :], \
               self.U[t:t_plus_tp, :], \
               self.X[t, :].unsqueeze(0), \
               self.X[t:t_plus_tp, :]    


if __name__ == "__main__":
    
    dataset = DroneHyperDataset(dataset_path="test_data/jana03.csv",
                                observation_window=50,
                                prediction_horizon=100,
                                filter_pwm=True,
                                moving_window_median_filter=9,
                                stride=10,
                                device='cpu')
    dataset.plot_dataset()