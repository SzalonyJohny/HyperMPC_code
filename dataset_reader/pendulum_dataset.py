import torch
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from dataset_reader.dataset_files_spliter import split_dataset_files


class PendulumHyperDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 mode: str, # "train", "val", "test"
                 train_val_test_split: list, # [0.7, 0.2, 0.1]
                 observation_window: int,
                 prediction_horizon: int,
                 stride: int):

        dataset_path = Path(dataset_path)
        
        train_files, val_files, test_files = split_dataset_files(
            path=dataset_path,
            train_ratio=train_val_test_split[0],
            val_ratio=train_val_test_split[1],
        )

        if mode == "train":
            files = train_files
        elif mode == "val":
            files = val_files
        elif mode == "test":
            files = test_files
        else:
            raise ValueError("mode should be one of ['train', 'val', 'test']")

        t_max = np.inf

        segment_length = observation_window + prediction_horizon

        list_files = list(files)

        def extract_episode_number(file_path):
            file_name = Path(file_path).stem
            return int(file_name.split("_")[-1])

        list_files.sort(key=extract_episode_number)

        M_cols = ["q", "dq", "backlash", "d_backlash", "u"]
        U_cols = ["u"]
        X_cols = ["q", "dq"]
        P_cols = ["u"]

        list_of_tensors_M = []
        list_of_tensors_U = []
        list_of_tensors_X0 = []
        list_of_tensors_X = []
        list_of_tensors_P = []

        for file in list_files:

            df = pd.read_csv(file)
            df = df[df["t"] < t_max]  # cut off the end of the episode

            episode_length = len(df)

            for i in range(observation_window, episode_length - prediction_horizon, stride):
                M = df[M_cols].iloc[i - observation_window: i].to_numpy()

                U = df[U_cols].iloc[i: i + prediction_horizon].to_numpy()

                X = df[X_cols].iloc[i: i + prediction_horizon].to_numpy()
                P = df[P_cols].iloc[i: i + prediction_horizon].to_numpy()

                M = torch.from_numpy(M).float().unsqueeze(0)
                U = torch.from_numpy(U).float().unsqueeze(0)
                X = torch.from_numpy(X).float().unsqueeze(0)
                X0 = X[:, 0, :].unsqueeze(0)
                P = torch.from_numpy(P).float().unsqueeze(0)

                list_of_tensors_M.append(M)
                list_of_tensors_U.append(U)
                list_of_tensors_X0.append(X0)
                list_of_tensors_X.append(X)
                list_of_tensors_P.append(P)

        self.M = torch.cat(list_of_tensors_M, dim=0).contiguous()
        self.U = torch.cat(list_of_tensors_U, dim=0).contiguous()
        self.X0 = torch.cat(list_of_tensors_X0, dim=0).contiguous()
        self.X = torch.cat(list_of_tensors_X, dim=0).contiguous()
        self.P = torch.cat(list_of_tensors_P, dim=0).contiguous()

        # make dtype float32
        self.M = self.M.float()
        self.U = self.U.float()
        self.X0 = self.X0.float()
        self.X = self.X.float()
        self.P = self.P.float()

        assert self.M.shape[0] == self.U.shape[0] == self.X0.shape[0] == self.X.shape[0]
        self.length = self.M.shape[0]
        
        # calac max channel values  q, dq, backlash, d_backlash
        self.max_q = self.M[:, :, 0].abs().max()
        self.max_dq = self.M[:, :, 1].abs().max()
        self.max_backlash = self.M[:, :, 2].abs().max()
        self.max_d_backlash = self.M[:, :, 3].abs().max()
        
        print(f"Mode : {mode}")
        print("max_q", self.max_q)
        print("max_dq", self.max_dq)
        print("max_backlash", self.max_backlash)
        print("max_d_backlash", self.max_d_backlash)
        

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
        return self.M[idx],self.P[idx], self.U[idx], self.X0[idx], self.X[idx]
