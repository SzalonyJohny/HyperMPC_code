import torch


class RolloutModel(torch.nn.Module):
    def __init__(self,
                 dyn_model: torch.nn.Module,
                 intergration_method: str,
                 compile: bool,
                 chunk_mode: bool,
                 chunk_size: int,
                 Tp: float):

        super(RolloutModel, self).__init__()
        self.dyn_model = dyn_model
        self.Tp = Tp
        self.chunk_size = chunk_size

        assert intergration_method == "rk4" or intergration_method == "euler"
        self.intergration_method = intergration_method

        if intergration_method == "rk4":
            self._step = self._rk4_step
            print("Using RK4 integration method for rollout")
        else:
            self._step = self._euler_step

        if not chunk_mode:
            self.forward = self.forward_old

        if compile:
            if chunk_mode:
                self._forward_n_step = torch.compile(self._forward_n_step, fullgraph=True, mode='max-autotune-no-cudagraphs')
            else:
                self._step = torch.compile(self._step, fullgraph=True, mode='default')

    def _euler_step(self, t, X_sim, U, p0):
        return X_sim + self.dyn_model(t,
                                      X_sim.clone(),
                                      U,
                                      p0) * self.Tp

    def _rk4_step(self, t, X_sim, U, p0, p05, p1):
        k1 = self.dyn_model(t, X_sim.clone(),
                            U, p0)
        k2 = self.dyn_model(t, X_sim.clone() + k1 * self.Tp / 2,
                            U, p05)
        k3 = self.dyn_model(t, X_sim.clone() + k2 * self.Tp / 2,
                            U, p05)
        k4 = self.dyn_model(t, X_sim.clone() + k3 * self.Tp,
                            U, p1)
        return X_sim + (k1 + 2*k2 + 2*k3 + k4) * self.Tp / 6

    def forward_old(self, X0, U, p_interpolated, prediction_horizon: int):
        """
            X0: (batch_size, 1, state_dim)
            U: (batch_size, prediction_horizon, control_dim)
            p: (batch_size, prediction_horizon [*2 if rk4], param_dim)

            return: X_sim (batch_size, prediction_horizon, state_dim)
        """
        # TODO add support for different prediction horizon for train and test
        # if p_interpolated.shape[1] < prediction_horizon:
        #     len_diff = prediction_horizon - p_interpolated.shape[1]
        #     if self.intergration_method == "rk4":
        #         len_diff = len_diff * 2
        #     p_expand = p_interpolated[:, -1].unsqueeze(1).repeat(1, len_diff, 1)
        #     p_interpolated = torch.cat([p_interpolated,
        #                                 p_expand], dim=1)
        
        X_sim = X0.repeat(1, prediction_horizon, 1)
        for i in range(1, prediction_horizon):

            t_now = torch.tensor([i * self.Tp], device=X0.device)

            if self.intergration_method == "rk4":
                # assert p_interpolated.shape[1] == U.shape[1] * 2
                p0_rk4 = p_interpolated[:, i*2, :]
                p05_rk4 = p_interpolated[:, i*2 + 1, :]
                p1_rk4 = p_interpolated[:, min(i*2 + 2, p_interpolated.shape[1] - 1), :]
                
                X_sim[:, i] = self._step(t_now,
                                        X_sim[:, i - 1],
                                        U[:, i-1],
                                        p0_rk4,
                                        p05_rk4,
                                        p1_rk4)
            else:
                p0 = p_interpolated[:, i, :]
                X_sim[:, i] = self._step(t_now,
                                        X_sim[:, i - 1],
                                        U[:, i-1],
                                        p0)       
            
        return X_sim
    
    
    def _forward_n_step(self, X0, U, p_interpolated, prediction_horizon: int):
        """
            X0: (batch_size, 1, state_dim)
            U: (batch_size, prediction_horizon, control_dim)
            p: (batch_size, prediction_horizon [*2 if rk4], param_dim)

            return: X_sim (batch_size, prediction_horizon, state_dim)
        """
        # TODO add support for different prediction horizon for train and test
        # if p_interpolated.shape[1] < prediction_horizon:
        #     len_diff = prediction_horizon - p_interpolated.shape[1]
        #     if self.intergration_method == "rk4":
        #         len_diff = len_diff * 2
        #     p_expand = p_interpolated[:, -1].unsqueeze(1).repeat(1, len_diff, 1)
        #     p_interpolated = torch.cat([p_interpolated,
        #                                 p_expand], dim=1)
        
        X_evo = X0.clone().squeeze(1)
        X_list = [X_evo]
        
        for i in range(1, prediction_horizon):

            t_now = torch.tensor([i * self.Tp], device=X0.device)

            if self.intergration_method == "rk4":
                # assert p_interpolated.shape[1] == U.shape[1] * 2
                p0_rk4 = p_interpolated[:, i*2, :]
                p05_rk4 = p_interpolated[:, i*2 + 1, :]
                p1_rk4 = p_interpolated[:, min(i*2 + 2, p_interpolated.shape[1] - 1), :]
                
                X_evo = self._step(t_now,
                                        X_evo,
                                        U[:, i-1],
                                        p0_rk4,
                                        p05_rk4,
                                        p1_rk4)
            else:
                p0 = p_interpolated[:, i, :]
                X_evo = self._step(t_now,
                                   X_evo,
                                   U[:, i-1],
                                   p0)       
            X_list.append(X_evo)
                
        return torch.stack(X_list, dim=1)
    
        
    def forward(self, X0, U, p_interpolated, prediction_horizon: int):
        """
        Forward method that uses _forward_n_step in overlapping chunks.

        Args:
            X0: (batch_size, 1, state_dim)
            U: (batch_size, prediction_horizon, control_dim)
            p_interpolated: (batch_size, prediction_horizon [*2 if rk4], param_dim)
            prediction_horizon: int

        Returns:
            X_sim: (batch_size, prediction_horizon, state_dim)
        """
        n = self.chunk_size  # Number of steps in each chunk
        overlap = n - 1  # Overlap between segments

        # Check that the prediction horizon is compatible
        assert (prediction_horizon - 1) % overlap == 0, \
            f"prediction_horizon - 1 must be divisible by {overlap}"

        num_chunks = ((prediction_horizon - 1) // overlap) + 1

        # Initialize X_sim with zeros and set the first state
        X_sim = torch.zeros(X0.shape[0], prediction_horizon, X0.shape[2], device=X0.device)
        X_sim[:, 0] = X0.squeeze(1)

        for i in range(num_chunks):
            # Calculate start and end indices for the current chunk
            start_idx = i * overlap
            end_idx = min(start_idx + n, prediction_horizon)  # Ensure we don't exceed the horizon
            current_n = end_idx - start_idx  # Adjust n for the last chunk if necessary

            # Handle the indexing for U and p_interpolated
            U_chunk = U[:, start_idx:end_idx]
            
            if self.intergration_method == "rk4":
                p_chunk = p_interpolated[:, start_idx*2:end_idx*2 + 2]
            else:
                p_chunk = p_interpolated[:, start_idx:end_idx]

            # Initial state for the chunk
            if i == 0:
                X0_chunk = X0
            else:
                X0_chunk = X_sim[:, start_idx].unsqueeze(1)

            # Compute the state evolution for the chunk
            X_chunk = self._forward_n_step(X0_chunk, U_chunk, p_chunk, current_n)

            # Assign the computed states to X_sim
            X_sim[:, start_idx:end_idx] = X_chunk

        return X_sim

