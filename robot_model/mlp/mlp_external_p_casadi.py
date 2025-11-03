import numpy as np
import torch
import casadi as ca
from robot_model.mlp.mlp_external_p import MlpExternalParams


def external_params_mlp(x, u, params, layer_sizes, activation=ca.tanh, skip_u = False):
    """MLP with external parameters using CasADi.
    Args:
        x (ca.MX): State vector (1, n_x)
        u (ca.MX): Control input (1, n_u) 
        params (ca.MX): Flattened weights and biases
        layer_sizes (list): Neurons per layer [n_x + n_u, ..., n_out]
        activation (function): Activation function, default ca.tanh
    Returns:
        ca.MX: Network output
    """
        
    out = ca.vertcat(x, u)
    if skip_u:
        out = x
    
    idx = 0
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        w_size = in_dim * out_dim

        w_flat = params[idx : idx + w_size]
        idx += w_size
        W = w_flat.reshape((in_dim, out_dim)).T
        b_flat = params[idx : idx + out_dim]
        idx += out_dim 
        b = ca.reshape(b_flat, out_dim, 1)

        out = ca.mtimes(W, out) + b

        # Apply activation, except on the last layer
        if i < len(layer_sizes) - 2:
            out = activation(out)

    return out


def external_params_mlp_casadifun(layer_sizes, activation=ca.tanh):
    """Create CasADi function for MLP with external parameters.
    Args:
        layer_sizes (list): Neurons per layer [n_x + n_u, ..., n_out]
        activation (function): Activation function, default ca.tanh
    Returns:
        casadi.Function: CasADi function
    """
    
    x = ca.MX.sym("x", 4)
    u = ca.MX.sym("u", 2)
    
    param_size = MlpExternalParams(
        preprocessor=torch.nn.Identity(),
        layer_sizes=layer_sizes,
        activation=torch.nn.Tanh(),
    ).parameter_count()
    
    p = ca.MX.sym("params", param_size)
    x_dot = external_params_mlp(x, u, p, layer_sizes, activation)
    return ca.Function("external_params_mlp", [x, u, p], [x_dot], ["x", "u", "params"], ["x_dot"]), param_size

def test():
    
    layer_sizes = [6, 5, 5, 5, 5, 1]
    model = MlpExternalParams(
    preprocessor=torch.nn.Identity(),
    layer_sizes=layer_sizes,
    activation=torch.nn.Tanh(),
    )
    
    model_casadi, _ = external_params_mlp_casadifun(layer_sizes, ca.tanh)

    diff_list = []

    for i in range(1000):
        # Get default parameters from PyTorch model
        p = model.get_default_params() * 10
        p *= torch.randn_like(p)

        # Create random input
        batch_size = 1
        x = torch.randn(batch_size, 4)  # State
        u = torch.randn(batch_size, 2)  # Control input
        t = torch.zeros(batch_size)  # Time

        # Get PyTorch output
        torch_out = model(t, x, u, p)
        x_np = x.detach().numpy()
        u_np = u.detach().numpy()
        p_np = p.detach().numpy()
        casadi_out = model_casadi(x_np[0, :], u_np[0, :], p_np[0, :])
        

        diff = np.max(np.abs(torch_out.detach().numpy() - casadi_out.T))
        diff_list.append(diff)
        
        if diff > 1e-5:
            print("Max difference:", diff)
            break

    print("Max difference:", np.max(diff_list))
    
            

if __name__ == "__main__":
    test()