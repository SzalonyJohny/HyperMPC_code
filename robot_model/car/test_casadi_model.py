import torch
import numpy as np
from robot_model.car.casadi_car_model_naive import pacejca_single_track_casadi
from robot_model.car.single_track import PacejkaTiresSingleTrack

def test_casadi_vs_pytorch():
    # Initialize Casadi model
    casadi_dyn_fun, tire_forc_model = pacejca_single_track_casadi()

    # Initialize PyTorch model
    pytorch_model = PacejkaTiresSingleTrack()
    pytorch_model.eval()

    # Parameters (update with actual values if available)
    params = pytorch_model.get_default_params_static().squeeze(0).numpy()

    # Sample multiple initial states
    num_samples = 200
    state_dim = 4  # Adjust based on your state size
    control_dim = 2

    for _ in range(num_samples):
        # Sample initial state from a reasonable range
        state = np.random.uniform(low=-5.0, high=5.0, size=state_dim)
        # Sample control inputs for the sequence
        control = np.random.uniform(low=-1.0, high=1.0, size=(control_dim))     
        
        param_per = np.random.uniform(low=0.9, high=1.1, size=(30))
        params = params * param_per   

        # Simulate over the control sequence
        # Casadi prediction
        casadi_state_dot = casadi_dyn_fun(
            state=state,
            control=control,
            params=params
        )
        casadi_state_dot = casadi_state_dot['state_dot'].full().flatten()
        print(casadi_state_dot)      
        
        # PyTorch prediction
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        control_tensor = torch.tensor(control).unsqueeze(0)
        params_tensor = torch.tensor(params).unsqueeze(0)

        with torch.no_grad():
            pytorch_state_dot = pytorch_model(0.0, state_tensor, control_tensor, params_tensor).squeeze(0).numpy()
            
        print(pytorch_state_dot)
        assert np.allclose(casadi_state_dot, pytorch_state_dot, atol=1e-6)
        

if __name__ == "__main__":
    test_casadi_vs_pytorch()