from joblib.externals.loky.backend.context import get_context
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import hydra
from hydra.utils import instantiate
import os
import time
from time import perf_counter
import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import pandas as pd
import casadi as cs
import hyper_prediction_models.const_param_model as const_param_model
from omegaconf import OmegaConf
import wandb

# car
from utils.set_up_hypermodel import setup_hypermodel
from mpc.mpc_formulation_car import MPC
from robot_model.car.casadi_car_model import car_dynamics
from utils.prepare_P_for_hypermodel import HyperModelPreprocesor


log = logging.getLogger(__name__)
torch.set_grad_enabled(False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

max_torch_num_threads = 8
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)


@hydra.main(version_base=None, config_path="conf_mpc", config_name="config_car")
def main(cfg: DictConfig) -> float:

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    log.info(f"Process ID {os.getpid()}")

    wandb_mode = 'online' if cfg.mpc.wandb.enable else 'disabled'

    run = wandb.init(project='hpm_car_mpc', group=cfg.mpc.wandb.group,
                     config=conf_dict, mode=wandb_mode)

    hyper_param_model, param_interp, cfgm, dyn_model = setup_hypermodel(cfg, run)
    print(f"hyper_param_model: {hyper_param_model}")
    
    # Base model parameters for initialization
    p_const = np.array([1.4583106e-01, 1.7316505e-01, 8.4769918e-04, 2.7608933e-04, 5.9258885e-04,
                        1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                        1.2260245e-01, 8.1564398e+00, 5.7084054e-02, 5.8887583e-01, 8.5324222e-01,
                        2.3114243e-01, 2.8331285e+00, 6.8991023e-01, 8.0986267e-01, 6.5774233e-03,
                        1.3231562e-01, 7.5576849e+00, 6.9186933e-02, 5.9321409e-01, 8.6027056e-01,
                        4.2846470e+00, 4.6745505e+00, 1.5791984e+00, 8.5530567e-01, 7.2961149e+00])

    dt = cfg.mpc.dt
        
    track_path = Path(__file__).parent / "mpc" / "tracks" / ("prep_" + cfg.mpc.track_name + ".csv")
    
    
    model = car_dynamics(p_init=p_const,
                         track_path=track_path,
                         cfg=cfg.mpc,
                         cfg_model=cfgm, 
                         errors=False)

    mpc = MPC(model=model, dt=dt, cfg_mpc=cfg.mpc)
    solver = mpc.solver
    sim = mpc.simulator
    

    N = cfg.mpc.N
    T_horizon = N * dt
    print(f"T_horizon: {T_horizon}, with N: {N}")
    print(f"model.track_length: {model.track_length}")

    SIM_LEN = cfg.mpc.SIM_LEN

    x_at_t = model.x_start.copy()
    
    if sample_sensitivity:
        x_at_t = np.array([2.07052501, 0.05041181, -0.45353745, 3.4721033, 0.77963439, -1.07383075,  0.8, 3.19628477, -0.49996018])
    
    for n in range(N+1):
        x_at_n = x_at_t.copy()
        x_at_n[0] = x_at_n[0] + x_at_n[3] * dt
        solver.set(n, "x", x_at_n)
    

    l_solve_time = []
    l_control = []
    l_state = []
    l_get_time = []
    l_inf_time = []
    l_p = []

    hypep_prep = HyperModelPreprocesor(N, dt,
                                       cfgm.dataset.prediction_horizon,
                                       cfgm.dataset.Tp,
                                       cfgm.dataset.observation_window,
                                       cfgm.dataset.observarions_channels)
    P_x = np.zeros((N+1, 9))
    
    
    
    # solver.load_iterate("mpc_iterate.json")

    for i in range(SIM_LEN):
        print(f"i: {i}, s: {x_at_t[0]}")
        
        now = perf_counter()

        M, P = hypep_prep.forward(x_at_t, P_x[:, -2:])
        
        with torch.inference_mode():
            M = torch.from_numpy(M).float().unsqueeze(0)
            P = torch.from_numpy(P).float().unsqueeze(0)
            p_preint, _ = hyper_param_model(M, P)
            p = param_interp(p_preint)
            p = p[0, ::2, :].cpu().detach().numpy()
            p0 = p[0, :]

        p_to_mpc = hypep_prep.params_interp(p)
        

        if cfg.mpc.residual_model:
            p_const_rep = np.tile(p_const.reshape(1, -1), (81, 1))
            p_to_mpc = np.concatenate([p_const_rep, p_to_mpc], axis=-1)
            
        if enable_errors:
            sample_errors = np.random.randn(N+1, 4) * 2.5e-3
            p_to_mpc = np.concatenate([p_to_mpc, sample_errors], axis=-1)
            
        if i > 200:
            l_inf_time.append(perf_counter() - now)
            solver.set_flat("p", p_to_mpc.flatten())
            p_now = p_to_mpc[5, :]
            l_p.append(p_now)

        # preparation phase
        # solver.options_set('rti_phase', 1)
        # res = solver.solve()        
    
        now = perf_counter()

        # feedback phase
        solver.set(0, "lbx", x_at_t)
        solver.set(0, "ubx", x_at_t)
        # solver.options_set('rti_phase', 2)
        res = solver.solve()      
        
        l_solve_time.append(perf_counter() - now)

        if res != 0:
            solver.print_statistics()           
            break

        t_get = perf_counter()

        P_x = (solver.get_flat("x").reshape((N+1, 9)))
        P_u = (solver.get_flat("u").reshape((N, 2)))

        l_get_time.append(perf_counter() - t_get)

        l_state.append(x_at_t)
        l_control.append(solver.get(0, "u"))

        u0 = solver.get(0, "u")
    
        p_sim = np.concatenate([p0, np.zeros(4)], axis=-1)
        x_at_t = sim.simulate(x=x_at_t, u=u0, p=p_sim)
        # x_at_t[:2] += np.random.normal(0, 0.005, 2)
            
        
        if (x_at_t[0] > model.track_length):
            print("Resetting s")
            P_x[:, 0] -= model.track_length
            P_x[:, 0] = np.clip(P_x[:, 0], 0, model.track_length)
            x_at_t[0] = x_at_t[0] - model.track_length
            print(f"Resetting x_at_t: {x_at_t}")
            print(f"Resetting P_x: {P_x[:, 0]}")
            solver.set_flat("x", P_x.flatten())    
            
    # calculate mean solve time
    mean_solve_time = np.mean(l_solve_time)
    print(f"mean_solve_time: {mean_solve_time}")

    mean_get_time = np.mean(l_get_time)
    print(f"mean_get_time: {mean_get_time}")

    mean_inf_time = np.mean(l_inf_time)
    print(f"mean_inf_time: {mean_inf_time}")


    l_state = np.array(l_state)
    s = l_state[:, 0]

    # plot control
    l_control = np.array(l_control)
    plt.plot(s, l_control[:, 0], label="wheel_speed_ref")
    plt.plot(s, l_control[:, 1], label="delta")

    # plt.plot(l_state[:, 0], label="s")
    # plt.plot(l_state[:, 1], label="n")

    plt.plot(s, l_state[:, 3], label="v_x")
    plt.plot(s, l_state[:, 4], label="v_y")
    plt.plot(s, l_state[:, 5], label="r")
    # plt.plot(l_state[:, 6], label="friction")

    plt.plot(s, l_state[:, 7], label="wheel_speed")
    plt.plot(s, l_state[:, 8], label="delta")

    plt.legend()
    
    # plt.show()

    # mean speed
    mean_speed = np.mean(l_state[:, 3])
    print(f"mean_speed: {mean_speed}")

    s_dot = np.zeros_like(l_state[:, 0])
    for i in range(1, len(l_state)):
        s_dot[i] = model.s_dot_func(l_state[i])

    plt.figure()
    plt.plot(s_dot, label="s_dot")
    # vx
    plt.plot(l_state[:, 3], label="v_x")
    # vy
    plt.plot(l_state[:, 4], label="v_y")
    v_norm = np.sqrt(l_state[:, 3]**2 + l_state[:, 4]**2)
    plt.plot(v_norm, label="v_norm")
    plt.legend()

    track_path_org = Path(__file__).parent / "mpc" / "tracks" / (cfg.mpc.track_name + ".csv")
    from mpc.tracks.track_preprocesor import TrackReader
    track = TrackReader(track_path_org)
    track.plot_track()
    track.plot_points(l_state[:, 0] % track.length(),
                      l_state[:, 1], l_state[:, 3])

    # Plot all parameters from l_p
    l_p = np.array(l_p)
    from robot_model.car.single_track import PacejkaTiresSingleTrack
    single_track = PacejkaTiresSingleTrack()
    param_names = single_track.get_params_names()

    # plt.figure(figsize=(15, 10))
    # for i, param_name in enumerate(param_names):
    #     plt.plot(l_p[:, i], label=param_name)

    plt.xlabel('Time step')
    plt.ylabel('Parameter value')
    plt.legend()
    plt.title('Evolution of Parameters over Time')

    plt.figure()
    
    solve_times = np.array(l_solve_time)
    plt.hist(solve_times, bins=50, alpha=0.5, label='solve_time') # range=(0, 0.5)

    plt.show()

if __name__ == "__main__":
    main()
