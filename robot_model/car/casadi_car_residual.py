import casadi as cs
import numpy as np
from robot_model.car.casadi_car_model_naive import pacejca_single_track_casadi
from robot_model.mlp.mlp_external_p_casadi import external_params_mlp_casadifun
from robot_model.car.residual_single_track_model import ResidualSingleTrack
import torch



def residual_single_track_casadi(layers, activation):

    car_model, _ = pacejca_single_track_casadi()
    
    residual_nn, param_size = external_params_mlp_casadifun(layers, activation)
    
    x = cs.MX.sym("x", 4)
    u = cs.MX.sym("u", 2)
    
    p_car = cs.MX.sym('p', 30)
    p_nn = cs.MX.sym('p', param_size)
    p_all = cs.vertcat(p_car, p_nn)
    
    x_dot = car_model(x, u, p_car) + residual_nn(x, u, p_nn)
    
    dyn_fun = cs.Function("residual_single_track", [x, u, p_all], [x_dot], ["x", "u", "p_all"], ["x_dot"])
    
    return dyn_fun, param_size


def test():
    
    torch_model = ResidualSingleTrack(
        preprocessor=torch.nn.Identity(),
        layer_sizes=[6, 32, 32, 4],
        activation=torch.nn.ReLU(),
        compile_model=False,
        device='cpu'
    )
    
    relu_in = cs.MX.sym("relu_in", 1)
    casadi_relu = cs.Function("casadi_relu", [relu_in], [cs.fmax(0, relu_in)], ["in"], ["out"])
    
    model_casadi, param_size = residual_single_track_casadi([6, 32, 32, 4], casadi_relu)
    print (model_casadi)
    
    p_car = torch_model.single_track.get_default_params()
    p_nn = torch_model.get_default_params()

    l_diff = []
    
    for _ in range(100):
    
        p_nn += 0.1 * torch.randn_like(p_nn)
    
        batch_size = 1
        x = torch.randn(batch_size, 4)
        u = torch.randn(batch_size, 2)
        t = torch.zeros(batch_size)
        
        # Get PyTorch output
        torch_out = torch_model(t, x, u, p_nn)
        
        x_np = x[0, :].detach().numpy()
        u_np = u[0, :].detach().numpy()
        p_np_car = p_car[0, :].detach().numpy()
        p_np_nn = p_nn[0, :].detach().numpy()
        p_all = np.concatenate([p_np_car, p_np_nn])
        
        casadi_out = model_casadi(x_np, u_np, p_all)
        
        diff = torch_out.detach().numpy() - casadi_out.full().flatten()
        max_diff = np.max(np.abs(diff))        
        
        l_diff.append(max_diff)
        assert max_diff < 5e-5, f"Max diff: {max_diff}"
        
    print (f"Max diff: {np.max(l_diff)}")


if __name__ == "__main__":
    test()