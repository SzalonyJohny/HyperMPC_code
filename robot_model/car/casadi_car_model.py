import torch
import casadi

from robot_model.casadi_dmodel_wrapper import CasadiDynamicsModelWraper
import l4casadi as l4c
import casadi as cs
import numpy as np
from mpc.tracks.map_reader import getTrackCustom
from robot_model.car.casadi_car_model_naive import pacejca_single_track_casadi
from robot_model.car.casadi_car_residual import residual_single_track_casadi

# fmt: off

def car_dynamics(p_init, track_path : str, cfg, cfg_model=None, errors=False, sens_error=False):

    # load track parameters
    [s0, _, _, _, kapparef] = getTrackCustom(track_path)

    ds = np.median(np.diff(s0))
    track_length = s0[-1] + ds * 0.99
    
    # add periodicity
    s0 = np.append(s0, [s0[-1] + s0[1:]])
    kapparef = np.append(kapparef, kapparef[1:])    

    # interpolate track parameters
    kapparef_s = cs.interpolant("kapparef_s", "bspline", [s0], kapparef)

    # time
    t = cs.MX.sym('t')
    # pose state variables
    s = cs.MX.sym('s')
    n = cs.MX.sym('n')
    mu = cs.MX.sym('mu')
    # vel state variables extended by ctrl [delta, wheel_speed]
    vx = cs.MX.sym('vx')
    vy = cs.MX.sym('vy')
    r = cs.MX.sym('r')
    friction = cs.MX.sym('friction')
    delta = cs.MX.sym('delta')
    wheel_speed = cs.MX.sym('wheel_speed')
    # stack state variables
    x_dyn_model = cs.vertcat(vx, vy, r, friction)
    u_dyn_model = cs.vertcat(wheel_speed, delta)
    x_exp = cs.vertcat(s, n, mu,
                       vx, vy, r, friction,
                       wheel_speed, delta)

    # control variables u_ref for MPC
    delta_ref = cs.MX.sym('delta_ref')
    wheel_speed_ref = cs.MX.sym('wheel_speed_ref')
    u_ref = cs.vertcat(wheel_speed_ref, delta_ref)

    # parameters
    p_dyn_model = cs.MX.sym('p', p_init.shape[-1]) 

    # x_dot symbolic variables
    s_dot = cs.MX.sym('s_dot')
    n_dot = cs.MX.sym('n_dot')
    mu_dot = cs.MX.sym('mu_dot')
    vx_dot = cs.MX.sym('vx_dot')
    vy_dot = cs.MX.sym('vy_dot')
    r_dot = cs.MX.sym('r_dot')
    friction_dot = cs.MX.sym('friction_dot')
    wheel_speed_dot = cs.MX.sym('wheel_speed_dot')
    delta_dot = cs.MX.sym('delta_dot')
    x_dot = cs.vertcat(s_dot, n_dot, mu_dot, # pose
                       vx_dot, vy_dot, r_dot, friction_dot, # vel
                       wheel_speed_dot, delta_dot) # u
    
    
    # model setup
    if cfg_model.dmodel.model._target_ == "robot_model.car.residual_single_track_model.ResidualSingleTrack":
        # Relu
        relu_in = cs.MX.sym("relu_in", 1)
        casadi_relu = cs.Function("casadi_relu", [relu_in], [cs.fmax(0.0, relu_in)], ["in"], ["out"])        
        activation_dict = {"torch.nn.ReLU": casadi_relu,
                           "torch.nn.Tanh": cs.tanh}
        activation_fun = activation_dict[cfg_model.dmodel.model.activation._target_]
        l4c_model, nn_param_size = residual_single_track_casadi(cfg_model.dmodel.model.layer_sizes,
                                                                activation_fun)
        p_nn = cs.MX.sym('p', nn_param_size)
        p_dyn_model = cs.vertcat(p_dyn_model, p_nn)
        p_init = np.concatenate([p_init, np.zeros(nn_param_size)])
    elif cfg_model.dmodel.model._target_ == "robot_model.car.single_track.PacejkaTiresSingleTrack":
        l4c_model, _ = pacejca_single_track_casadi()
    else:
        raise ValueError(f"Unknown model: {cfg_model.dmodel.model._target_}")

    e_x_dot_dyn_model = l4c_model(x_dyn_model, u_dyn_model, p_dyn_model)

    # Model: x_dot = f(x, u)
    e_s_dot = (vx * cs.cos(mu) - vy * cs.sin(mu)) / (1 - kapparef_s(s) * n)
    e_n_dot = vx * cs.sin(mu) + vy * cs.cos(mu)
    e_u_dot = r - kapparef_s(s) * e_s_dot
    e_pose_dot = cs.vertcat(e_s_dot, e_n_dot, e_u_dot)

    # u model
    e_delta_dot = (delta_ref - delta) / (cfg.steering_time_constant)
    e_wheel_speed_dot = (wheel_speed_ref - wheel_speed) / (cfg.wheel_speed_time_constant)

    e_u_dot = cs.vertcat(e_wheel_speed_dot, e_delta_dot)
    f_expl = cs.vertcat(e_pose_dot, e_x_dot_dyn_model, e_u_dot)

    # track constraints as soft constraints
    h_left = cfg.track_width - n
    h_right = cfg.track_width + n

    # control barrier function
    def soft_constraint(h, lambda_ = cfg.soft_constraint_lambda):
        return cs.log(1 + cs.exp( - lambda_ * h))

    track_soft_constraints_cost = soft_constraint(h_left) + soft_constraint(h_right)

    # slip angle regularize     
    beta_kin = cs.arctan(delta * 0.15 / (0.33)) # # FIXME hyperparameter
    beta_dyn = cs.arctan(vy / vx)


    # slip ratio cost 
    wheel_speed_diff = (wheel_speed_ref - wheel_speed)**2
    wheel_slip_pow2 = (wheel_speed - vx)**2

    # cost function
    cost_expr_for_all = (- s) \
                + cfg.q_beta * (beta_kin - beta_dyn)**2 \
                + cfg.q_n * n**2 \
                + cfg.q_mu * mu**2 \
                + cfg.q_track_constraint * track_soft_constraints_cost \
                + cfg.q_slip * wheel_slip_pow2

    cost_expr = cost_expr_for_all + cfg.q_speed_diff * wheel_speed_diff + cfg.q_delta * (delta_ref - delta) ** 2 
    cost_expr_e = cost_expr_for_all + cfg.q_mu_e * mu**2 # terminal cost

    forward_model_func = cs.Function('forward_model', [t, x_exp, u_ref, p_dyn_model], [f_expl])
    cost_func = cs.Function('cost_func', [t, x_exp, u_ref, p_dyn_model], [cost_expr])
    
    cost_func_jac = cs.Function('cost_func_jac', [t, x_exp, u_ref, p_dyn_model],
                                [cs.jacobian(cost_expr, x_exp)])


    # discrete dynamics
    k1 = forward_model_func(0.0, x_exp, u_ref, p_dyn_model)
    k2 = forward_model_func(0.0, x_exp + cfg.dt / 2 * k1, u_ref, p_dyn_model)
    k3 = forward_model_func(0.0, x_exp + cfg.dt / 2 * k2, u_ref, p_dyn_model)
    k4 = forward_model_func(0.0, x_exp + cfg.dt * k3, u_ref, p_dyn_model)
    x_next = x_exp + cfg.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    
    # errors for sensitivity analysis
    if errors:
        e_states = cs.MX.sym('e_states', 4)
        p_dyn_model = cs.vertcat(p_dyn_model, e_states)
        e_states_exp = cs.vertcat(0, 0, 0, e_states, 0, 0)
        p_init = np.concatenate([p_init, np.zeros(4)])
        x_next = x_next + e_states_exp
            
    # progress function for eval
    s_dot_func = cs.Function('s_dot_func', [x_exp], [e_s_dot])
    cost_for_eval = cs.Function('cost_for_eval', [x_exp, u_ref], [cost_expr + s])

    # store to struct
    model = cs.types.SimpleNamespace()
    model.x = x_exp
    model.xdot = x_dot
    model.u = u_ref
    model.z = cs.vertcat([])
    model.p = p_dyn_model # all parameters of the model
    model.parameter_values = p_init
    
    # model.p_global = p_global
    # model.p_global_values = np.zeros(p_global.shape)
    
    model.f_expl = f_expl
    model.x_start = np.array([
                              0.5, 0.0, 0.0, # pose [s, n, mu]
                              2.0, 0.0, 0.0, 0.80, # vel [vx, vy, r, friction]
                              2.1, 0.0 # u [wheel_speed, delta]
                              ])
    model.constraints = casadi.vertcat([])
    model.cost_expr_ext_cost = cost_expr
    model.cost_expr_ext_cost_e = cost_expr_e

    model.name = "single_track_casadi"

    model.disc_dyn_expr = x_next
    model.test_forward_model_func = forward_model_func
    model.test_cost_func = cost_func
    model.test_cost_jac_func = cost_func_jac

    model.track_length = track_length
    model.track_path = track_path
    model.s_dot_func = s_dot_func
    model.cost_for_eval = cost_for_eval


    # model.external_shared_lib_dir = l4c_model.shared_lib_dir
    # model.external_shared_lib_name = l4c_model.name    
    return model


def main():

    from robot_model.car.single_track import PacejkaTiresSingleTrack

    single_track = PacejkaTiresSingleTrack()

    model = car_dynamics(single_track, "/hyper_prediction_models/mpc/tracks/tro_out_icra_v1.csv")

    forward_model = model.test_forward_model_func
    cost_func = model.test_cost_func

    t = np.array([0.0])
    pose = np.array([0.0, 0.0, 0.0])
    v_state = np.array([2.0, 0.0, 0.0, 0.80])
    u = np.array([1.0, 0.0])
    x = np.concatenate([pose, v_state, u])
    u_ref = np.array([2.0, 0.0])
    p = model.parameter_values

    print(forward_model(t, x, u_ref, p))
    print(cost_func(t, x, u_ref, p))

    import time
    start = time.time()
    n = 1000
    u_ref = np.random.rand(2)

    for _ in range(n):
        x_dot = forward_model(t, x, u_ref, p)
        # cost = cost_func(t, x, u_ref, p)

    print(f"Forward model time: {(time.time() - start)/n}")


if __name__ == "__main__":
    main()