import casadi as cs
import numpy as np
from robot_model.drone.drone_model_casadi import drone_dynamics
from robot_model.mlp.mlp_external_p_casadi import external_params_mlp
import torch
from robot_model.mlp.mlp_external_p import MlpExternalParams

def residual_drone_dynamics(const_params=None, layer_sizes=[10, 32, 6], activation=cs.tanh):
    
    def relu(x):
        return cs.fmax(x, 0.0)
    
    model = drone_dynamics(const_params)
    
    px, py, pz, qw, qx, qy, qz, vx, vy, vz, bx, by, bz = cs.vertsplit(model.x)
    
    max_bxy = 4.0
    max_bz = 4.0
    max_v = 20.9
    
    mlp_in = cs.vertcat(qw, qx, qy, qz,
                        vx / max_v,
                        vy / max_v,
                        vz / max_v,
                        bx / max_bxy,
                        by / max_bxy,
                        bz / max_bz)
    
    mlp_param_size = MlpExternalParams(
        preprocessor=torch.nn.Identity(),
        layer_sizes=layer_sizes,
        activation=torch.nn.Tanh(),
    ).parameter_count()
    
    p_mlp = cs.MX.sym("p_mlp", mlp_param_size)
    
    forces_and_torques = external_params_mlp(mlp_in, cs.vertcat(), p_mlp,
                                             layer_sizes, activation, skip_u=True)
    
    f_x, f_y, f_z, tau_x, tau_y, tau_z = cs.vertsplit(forces_and_torques)
    
    f_dist = cs.vertcat(f_x, f_y, f_z)
    tau_dist = cs.vertcat(tau_x, tau_y, tau_z) * 0.0
    
    p_6 = cs.MX.sym("p_nominal", 6)
    p_nominal = cs.vertcat(p_6, f_dist, tau_dist)
    p_dist = cs.MX.sym("p_dist", 6)
    
    f_expl = model.dyn_fun(model.x, model.u, p_nominal)
    
    all_params = cs.vertcat(p_6, p_mlp)
    
    default_nominal_params = np.array([0.0000e+00, 3.0375e-05,\
                                       1.0295e+00, 4.7043e-02,\
                                       8.4545e-02, 2.0622e-02])
    default_mlp_params = np.zeros(mlp_param_size)
    default_params_val = np.concatenate((default_nominal_params, default_mlp_params))
    
    
    res_model = cs.types.SimpleNamespace()
    res_model.x = model.x
    res_model.xdot = model.xdot
    res_model.u = model.u
    res_model.z = cs.vertcat([])
    res_model.p = all_params
    res_model.parameter_values = default_params_val
    res_model.f_expl = f_expl
    res_model.x_start = np.array([
        0.5, 0.0, 0.0,  # pose [p]
        1.0, 0.0, 0.0, 0.0,  # orientation [q]
        0.0, 0.0, 0.0,  # velocity [v]
        0.0, 0.0, 0.0   # angular velocity [r]
    ])
    res_model.constraints = cs.vertcat([])
    res_model.name = "res_drone_residual_casadi"
    res_model.dyn_fun = cs.Function(
        "res_drone_residual",
        [model.x, model.u, all_params],
        [f_expl],
        ["x", "u", "params"],
        ["xdot"],
    )
    return res_model


if __name__ == "__main__":
    import torch
    import numpy as np
    from robot_model.drone.drone_residual import DroneResidualDynamisc
    from robot_model.mlp.mlp_external_p import MlpExternalParams

    # same layer sizes & activation as in the CasADi model
    layer_sizes = [10, 32, 6]

    # instantiate the CasADi residual model
    const_params = {
        "mq": 1.325 + 0.5,
        "g0": 9.80665,
        "l": 0.228035,
    }
    
    res_cas = residual_drone_dynamics(
        const_params=const_params,
        layer_sizes=layer_sizes,
        activation=cs.tanh,
    )
    cas_fun = res_cas.dyn_fun

    torch_dyn = DroneResidualDynamisc()

    # parameter counts
    param_count = MlpExternalParams(
        preprocessor=torch.nn.Identity(),
        layer_sizes=layer_sizes,
        activation=torch.nn.Tanh(),
    ).parameter_count()
    
    p_dict = {} 
    for k, v in torch_dyn.state_dict().items():
        if "." in k:
            k = k.split(".")[-1]
        p_dict[k] = v.exp().item()
    
    nominal_params = torch.tensor([p_dict['Db'], p_dict['Cd'], p_dict['Ct'], 
                                   p_dict['Jx'], p_dict['Jy'], p_dict['Jz']]).numpy()    

    for _ in range(100):
        # random state + control
        x = np.random.rand(13)
        x[3:7] /= np.linalg.norm(x[3:7])
        u = np.random.rand(4)

        # sample a random residual-MLP parameter vector
        p_mlp = np.random.randn(param_count)
        p_all = np.concatenate([nominal_params, p_mlp])

        # CasADi rollout
        xdot_c = np.array(cas_fun(x, u, p_all)).squeeze()
        print(f"xdot_c: {xdot_c}")

        # Torch rollout
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        p_t = torch.tensor(p_mlp, dtype=torch.float32).unsqueeze(0)
        xdot_t = torch_dyn(0.0, x_t, u_t, p_t).detach().numpy().squeeze()
        print(f"xdot_t: {xdot_t}")

        diff = np.linalg.norm(xdot_c - xdot_t)
        print(f"||xdot_c - xdot_t|| = {diff:.3e}")
        assert diff < 1e-5, f"Difference too large: {diff}"