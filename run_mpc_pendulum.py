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
from tqdm import tqdm

# car
from utils.set_up_hypermodel import setup_hypermodel
from mpc.mpc_pendulum import MPC
from robot_model.pendulum.casadi_pendulum import pendulum_dynamics
from utils.prepare_P_for_hypermodel import HyperModelPreprocesor
from dataset_generators.pendulum_backlash_sim_mujoco import PendulumSimMujocoBacklash


log = logging.getLogger(__name__)
torch.set_grad_enabled(False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

max_torch_num_threads = 8
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)


@hydra.main(version_base=None, config_path="conf_mpc", config_name="config_pendulum")
def main(cfg: DictConfig) -> float:

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    log.info(f"Process ID {os.getpid()}")

    wandb_mode = 'online' if cfg.mpc.wandb.enable else 'disabled'

    run = wandb.init(project='hpm_pendulum_mpc', group=cfg.mpc.wandb.group,
                     config=conf_dict, mode=wandb_mode)

    hyper_param_model, param_interp, cfgm, dyn_model = setup_hypermodel(
        cfg, run)
    log.info(f"hyper_param_model: {hyper_param_model}")

    # FIXME make it shure that the p_const is the same as in the init model
    # if the residual model i used
    p_const = np.array([1.0, 1.0, 0.2, 0.001, 1.0])

    dt = cfg.mpc.dt

    model = pendulum_dynamics()

    solver = MPC(model=model, dt=dt, cfg_mpc=cfg.mpc).solver

    N = cfg.mpc.N
    log.info(f"T_horizon: {N * dt}, with N: {N}")

    l_of_traj = []
    l_of_costs = []
    l_solve_succ = []

    np.random.seed(42)

    for iii in range(20):
        q_start = np.random.uniform(-np.pi/2, np.pi/2)
        dq_start = np.random.uniform(-0.5, 0.5)
        x_at_t = np.array([q_start, dq_start, 0.0])
        # x_at_t = np.array([0.0, 0.0, 0.0])
        
        log.info(f"Starting state: {x_at_t}")
        

        l_solve_time = []
        l_control = []
        l_state = []
        l_get_time = []
        l_inf_time = []
        l_p = []
        P_x_save = None
        solver_fail_flag = False

        def M_from_x_fun(x):
            return np.array([x[0], x[1], 0.0, 0.0, x[2]])

        hypep_prep = HyperModelPreprocesor(N, dt,
                                           cfgm.dataset.prediction_horizon,
                                           cfgm.dataset.Tp,
                                           cfgm.dataset.observation_window,
                                           5,  # cfgm.dataset.observarions_channels
                                           M_from_x_fun=M_from_x_fun)
        P_x = np.zeros((N+1, 3))

        psim = {
            "sim_time_step": 10e-5,
            "slider_range":  0.2,
            "m": 1.0,
            "l": 0.5,
            "r": 0.025*5,  # Not used
            "f": 0.0,
            "b": 0.05,
            "backlash": cfg.mpc.sim_backlash,
        }

        dgen_settings = {
            "sim_implementation": "mujoco",
            # debug
            "render": False,
            "render_width": 480,
            "render_height": 480,
            # episode parametes
            "sample_per_second": 100,
            "episode_len_s": 10,
            # dataset size and path
            "number_of_runs": 360,  # 360 * 10s = 1 hour
            # to bias dataset with more data q1 = pi
            "q_range": (- np.pi, np.pi),
            "dq_range": (10.0, 10.0),
            # control signal
            "u_max": 1.0,
            "u_control_pt": 15,
        }

        sim = PendulumSimMujocoBacklash(psim, dgen_settings)
        sim.init_sim_with_render()
        sim.set_state(x_at_t)

        i_save_traj = 300

        t_sim = np.linspace(0, (cfg.mpc.SIM_LEN + N) * dt, cfg.mpc.SIM_LEN + N)

        if cfg.mpc.y_ref_traj:

            q_ref = np.sin(t_sim * np.pi * cfg.mpc.y_ref_freq) * \
                cfg.mpc.y_ref_A
            dq_ref = np.cos(t_sim * np.pi * cfg.mpc.y_ref_freq) * \
                cfg.mpc.y_ref_A * np.pi * cfg.mpc.y_ref_freq
            tau_ref = np.zeros_like(q_ref)
            dtau_ref = np.zeros_like(q_ref)
            y_ref = np.array([q_ref,
                              dq_ref,
                              tau_ref,
                              dtau_ref]).transpose()

        else:
            y_ref = np.array([cfg.mpc.y_ref])
            y_ref = np.repeat(y_ref, cfg.mpc.SIM_LEN + N, axis=0)

        solver.reset()

        for i in range(20):
            hypep_prep.obs_queue.forward(M_from_x_fun(x_at_t))

        for i in tqdm(range(cfg.mpc.SIM_LEN), desc="MPC Simulation..."):

            for n in range(N):
                solver.set(n, "yref", y_ref[i+n, :])
            solver.set(N, "yref", y_ref[i+N, :-1])

            now = perf_counter()

            M, P = hypep_prep.forward(x_at_t, P_x[:, -1].reshape(-1, 1))

            with torch.inference_mode():
                M = torch.from_numpy(M).float().unsqueeze(0)
                P = torch.from_numpy(P).float().unsqueeze(0)
                p_preint, _ = hyper_param_model(M, P)
                p = param_interp(p_preint)
                p = p[0, ::2, :].cpu().detach().numpy()

            p_to_mpc = hypep_prep.params_interp(p)
            
            if cfg.mpc.adaptation:
                contact = float(sim.in_contact())
                p_to_mpc[:, -1] *= contact

            if cfg.mpc.residual_model:
                p_const_rep = np.tile(p_const.reshape(1, -1), (N+1, 1))
                p_to_mpc = np.concatenate([p_const_rep, p_to_mpc], axis=-1)

            if i > 5:  # Enable hyper model
                l_inf_time.append(perf_counter() - now)
                solver.set_flat("p", p_to_mpc.flatten())
                p_now = p_to_mpc[5, :]
                l_p.append(p_now)

            solver.set(0, "lbx", x_at_t)
            solver.set(0, "ubx", x_at_t)

            now = perf_counter()

            res = solver.solve()

            l_solve_time.append(perf_counter() - now)

            t_get = perf_counter()

            P_x = (solver.get_flat("x").reshape((N+1, 3)))
            P_u = (solver.get_flat("u").reshape((N, 1)))

            if i == i_save_traj:
                P_x_save = P_x.copy()

            l_get_time.append(perf_counter() - t_get)

            l_state.append(x_at_t)
            l_control.append(solver.get(0, "u"))

            u = solver.get(1, "x")[-1]
            u = np.array([u])
            x_at_t = sim.step_sim_with_render(u, render=cfg.mpc.render)

            if res != 0:
                print(f"ERROR: {res}")
                solver_fail_flag = True
                break

        model_name = cfg.mpc.model.split('/')[-1]

        if cfg.mpc.render:
            sim.save_render_video(
                f"model_{model_name}_{iii}_wandb_{run.name}_{run.id}")

        l_state = np.array(l_state)
        t_state = np.linspace(0, l_state.shape[0] * dt, l_state.shape[0])
        q = l_state[:, 0]
        dq = l_state[:, 1]
        tau = l_state[:, 2]
        du = np.array(l_control)[:, 0]

        l = cfg.mpc.SIM_LEN

        if cfg.mpc.plot:

            if cfg.mpc.y_ref_traj:
                plt.plot(t_sim[:l], q_ref[:l],
                         label="q_ref_traj", linestyle='--')
                plt.plot(t_sim[:l], dq_ref[:l],
                         label="dq_ref_traj", linestyle='--')

            plt.plot(t_state, q, label="q")
            plt.plot(t_state, dq, label="dq")
            plt.plot(t_state, tau, label="tau")
            plt.legend()

            if P_x_save is not None and cfg.mpc.plot_save_pred:
                t_save = np.linspace(
                    i_save_traj * dt, i_save_traj*dt + P_x_save.shape[0] * dt, P_x_save.shape[0])
                plt.plot(t_save, P_x_save[:, 0], label="sq", marker='o')
                plt.plot(t_save, P_x_save[:, 1], label="sdq", marker='o')
                plt.plot(t_save, P_x_save[:, 2], label="stau", marker='o')
            plt.grid()
            plt.legend()
            # plt.savefig(f"documentation/{model_name}_{run.id}.pdf")

            p_name = ['m', 'l', 'b', 'f', 'gr']
            plt.figure()
            p = np.array(l_p)
            for i in range(5):
                plt.plot(t_state[2:], p[:, i], label=f"p_{p_name[i]}")
            plt.legend()
            plt.grid()
            plt.savefig(f"documentation/params_{model_name}_{run.id}.pdf")

            plt.show()

        if solver_fail_flag:
            l_solve_succ.append(0)
            episode_cost = 200
            log.info(f"Solver failed")
        else:
            l_solve_succ.append(1)
            q_ref = y_ref[:l, 0]
            dq_ref = y_ref[:l, 1]
            tau_ref = y_ref[:l, 2]
            step_cost = cfg.mpc.Q_q * (q - q_ref[:l])**2 \
                + cfg.mpc.Q_dq * (dq - dq_ref[:l])**2 \
                + cfg.mpc.Q_u * (tau - tau_ref[:l])**2
            episode_cost = np.sum(step_cost * dt)
            l_of_traj.append(l_state)
            l_of_costs.append(episode_cost)

            log.info(f"last cost : {step_cost[-1]}")

        log.info(f"episode_cost: {episode_cost}")
        log.info(f"final state: {x_at_t}")

        mean_solve_time = np.mean(l_solve_time)
        mean_inf_time = np.mean(l_inf_time)
        log.info(f"mean_solve_time: {mean_solve_time}")
        log.info(f"mean_inf_time: {mean_inf_time}")

        # return episode_cost

    l_of_traj = np.array(l_of_traj)
    l_of_costs = np.array(l_of_costs)
    log.info(f"fail solve ")
    log.info(f"l_of_costs: {l_of_costs.mean(), l_of_costs.std()}")
    np.save(f"pendulum_mpc_{cfg.mpc.model_nr}", l_of_traj)


if __name__ == "__main__":
    main()
