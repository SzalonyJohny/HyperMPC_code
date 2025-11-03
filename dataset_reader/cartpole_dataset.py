import torch
import pandas as pd
from pathlib import Path
from robot_model.car.state_wrapper import StateWrapper


class CartPoleHyperDataset2(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 dataset_file: str,
                 observation_window: int, # in samples
                 prediction_horizon: int, # in samples
                 stride: int, # in samples
                 device: str):

        # Observation columns
        M_cols = ["x1", "theta", "v1", "dtheta", "F"]
        
        # Control columns
        U_cols = ["F"]
        
        # State vector columns
        X_cols = ["x1", "theta", "v1", "dtheta"]
        
        # Future prediction columns
        P_cols = [ "F"]
        
        dataset_path = Path(dataset_path) / dataset_file
        self.df = pd.read_csv(dataset_path)
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
                
        self.index_tensor = torch.tensor(index_list, dtype=torch.long).contiguous()
        self.length = (self.index_tensor).shape[0]
        
        # index_tensor and data to device
        self.index_tensor = self.index_tensor.to(device)        
        self.M = self.M.to(device)
        self.U = self.U.to(device)
        self.X = self.X.to(device)
        self.P = self.P.to(device)
        print(f"Dataset length: {self.length}")
        print(f"Observation window: {self.M.shape}")
        
        # df max abs values
        self.max_values = self.df.abs().max().to_dict() 
        print(f"Max values: {self.max_values}")
        
    
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
