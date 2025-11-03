import torch
import numpy as np
import pandas as pd
from pathlib import Path


class DroneSimHyperDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 dataset_file: str,
                 observation_window: int, # in samples
                 prediction_horizon: int, # in samples
                 stride: int, # in samples
                 device: str):
        
        dataset_path = Path(dataset_path) / dataset_file
        self.df = pd.read_csv(dataset_path)

        # Observation columns
        M_cols = ['qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'bx', 'by', 'bz']
        
        # Control columns
        U_cols = ['t1', 't2', 't3', 't4']
        
        # State vector columns
        X_cols = ['x', 'y', 'z',
                  'qw', 'qx', 'qy', 'qz',
                  'vx', 'vy', 'vz',
                  'bx', 'by', 'bz']
        
        # Future prediction columns
        P_cols = ['t1', 't1', 't3', 't4']
        
        self.M = torch.from_numpy(self.df[M_cols].to_numpy()).float().contiguous()
        self.U = torch.from_numpy(self.df[U_cols].to_numpy()).float().contiguous()
        self.X = torch.from_numpy(self.df[X_cols].to_numpy()).float().contiguous()
        self.P = torch.from_numpy(self.df[P_cols].to_numpy()).float().contiguous()

        index_list = []
        droped_data = 0
        id_to_skip = []
        
        for i in range(observation_window, len(self.df) - prediction_horizon, stride):
    
            # i - current time sample      
            t_minus_to = i - observation_window # start of observation window
            t_plus_tp = i + prediction_horizon # end of prediction horizon
            
            if self.df["run_id"].iloc[t_minus_to] != self.df["run_id"].iloc[i + prediction_horizon]:
                droped_data += 1
                continue
            
            if np.any(self.df["z"].iloc[t_minus_to:t_plus_tp] > 10.0):
                droped_data += 1
                id_to_skip.append(self.df["run_id"].iloc[t_minus_to])
                continue
            
            if self.df["run_id"].iloc[t_minus_to] in id_to_skip:
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
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['t'], self.df['vx'], label='X', color='r')
        plt.plot(self.df['t'], self.df['vy'], label='Y', color='g')
        plt.plot(self.df['t'], self.df['vz'], label='Z', color='b')
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.title("Velocity")
        plt.legend()
        
        # plot body rates
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['t'], self.df['bx'], label='X', color='r')
        plt.plot(self.df['t'], self.df['by'], label='Y', color='g')
        plt.plot(self.df['t'], self.df['bz'], label='Z', color='b')
        plt.xlabel("Time")
        plt.ylabel("Body rates")
        plt.title("Body rates")
        
            
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
    
    dataset = DroneSimHyperDataset(dataset_path="/hyper_prediction_models/artifacts/drone_dataset:v11",
                                   dataset_file="test.csv",
                                observation_window=50,
                                prediction_horizon=100,
                                stride=10,
                                device='cpu')
    import matplotlib.pyplot as plt
    
    dataset.plot_dataset()
    
    df = dataset.df
    
    id_count = df['run_id'].max()
    
    for i in range(10):
        df_i = df[df['run_id'] == i]
        
        x = df['x'].values.to_numpy()
        y = df['y'].values.to_numpy()
        z = df['z'].values.to_numpy()
        t = df['t'].values.to_numpy()   
    
        pose = np.array([t, x, y, z])
        np.savez('test_traj.npz', pose=pose)
    
    # load 
    data = np.load('pose.npz')
    pose = data['pose']
    t, x, y, z = pose

    plt.figure(figsize=(12, 6))
    plt.plot(t, x, label='X', color='r')
    plt.plot(t, y, label='Y', color='g')
    plt.plot(t, z, label='Z', color='b')
    
    
    
    # plt.show()
    
    