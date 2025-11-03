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
from robot_model.cartpole.visualize_df import visualize_cartpole
import gc

# drone
from utils.set_up_hypermodel import setup_hypermodel
from mpc.mpc_formulation_drone import MPC
from utils.prepare_P_for_hypermodel import HyperModelPreprocesor
from robot_model.mlp.mlp_external_p_casadi import external_params_mlp_casadifun, external_params_mlp
from robot_model.mlp.mlp_external_p import MlpExternalParams
from robot_model.drone.drone_model_casadi import drone_dynamics
from dataset_generators.drone_sim import DroneSim
from dataset_reader.drone_sim_dataset import DroneSimHyperDataset
from robot_model.drone.drone_residual import DroneResidualDynamisc
from robot_model.drone.drone_res_model_casadi import residual_drone_dynamics
from robot_model.drone.drone_hyper_res_model_casadi import hyper_residual_drone_dynamics
from robot_model.drone.drone_model import DroneDynamisc

os.environ["WANDB_API_KEY"] = "56dceee73d5b31715f9476dc86527a75377caf6c"

log = logging.getLogger(__name__)
torch.set_grad_enabled(False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

max_torch_num_threads = 8
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)

np.random.seed(0)

@hydra.main(version_base=None, config_path="conf_mpc", config_name="config_drone")
def main(cfg: DictConfig) -> float:

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    log.info(f"Process ID {os.getpid()}")

    wandb_mode = 'online' if cfg.mpc.wandb.enable else 'disabled'

    run = wandb.init(project='hpm_drone_mpc', group=cfg.mpc.wandb.group,
                     config=conf_dict, mode=wandb_mode)

    hyper_param_model, param_interp, cfgm, dyn_model = setup_hypermodel(cfg, run)

    params_dict = dyn_model.state_dict()
    p_dict = {}
    
    for k, v in params_dict.items():
        if "." in k:
            k = k.split(".")[-1]
        p_dict[k] = v.item()
    
    p = torch.tensor([np.exp(p_dict['Db']), np.exp(p_dict['Cd']), np.exp(p_dict['Ct']), 
                     np.exp(p_dict['Jx']), np.exp(p_dict['Jy']), np.exp(p_dict['Jz']),
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).numpy()
    print(f"p: {p}")
    
    p_init = p.copy()
    
    print(f"hyper_param_model: {dyn_model}")
    print(f"hyper_param_model: {hyper_param_model}")
    
    config = {
        'freq': int(1 / cfg.mpc.dt),
        'episode_len_s': 20,
        'u_control_pt': 25,
        'max_deflaction': 0.7,
        'mujoco_sub_steps': 100,
        'ball_mass': cfg.mpc.ball_mass,
        'rope_lenght' : cfg.mpc.rope_len,
        'render': cfg.mpc.render,
        'episode_count': 300
    }

    sim = DroneSim(config)
    sim.init_mpc_simulation()

    x_at_t = np.array([
        0.0, 0.0, 2.0,  # pose [p]
        1.0, 0.0, 0.0, 0.0,  # orientation [q]
        0.0, 0.0, 0.0,  # velocity [v]
        0.0, 0.0, 0.0   # angular velocity [r]
    ])
    # sim.set_state(x_at_t)
    Q_diag = np.array([
            100.0,      # x
            100.0,      # y
            100.0,      # z
            1.0e1,     # qw
            1.0e1,     # qx
            1.0e1,     # qy
            1.0e1,     # qz
            0.001,        # vbx
            0.001,        # vby
            0.001,        # vbz
            0.10,       # wx
            0.10,       # wy
            0.10,       # wz
            0.001,        # m1
            0.001,        # m2
            0.001,        # m3
            0.001         # m4
    ])
    # Q_diag[3:] = 0.0
    Q = np.diag(Q_diag)
    
    # FIXME me use, dyn_model.nn.preprocessor._buffers
    const_params = {
        'mq': p_dict['mq'],   # mass
        'g0': 9.80665, # gravitational acceleration
        'l': 0.228035  # distance (half between motors' center and rotation axis)
    }
    
    if cfg.mpc.adaptation:
        const_params['mq'] = const_params['mq'] - 0.5
    
    res_model = False
    res_hyper = False
    
    if isinstance(dyn_model, DroneResidualDynamisc):
        res_model = True
        model = residual_drone_dynamics(const_params, 
                                        layer_sizes=cfgm.dmodel.model.layers_size,
                                        activation=cs.tanh)
        p_mlp, _ = hyper_param_model(torch.zeros(1, 20), torch.zeros(1, 4))
        p_mlp = p_mlp.squeeze(0).squeeze(0).cpu().detach().numpy()
        # save state dict
        print(f"p_mlp: {cfgm.dmodel.model.layers_size}")
        # torch.save(dyn_model.drone_model.state_dict(), f"dyn_model_sd_{cfg.mpc.model_nr}.pt")
        # print(f"p_mlp: {p_mlp}")
        # np.savez(f"p_mlp_{cfg.mpc.model_nr}.npz", p_mlp=p_mlp)
    elif isinstance(dyn_model, DroneDynamisc):      
        model = drone_dynamics(const_params)
        
    else:
        res_hyper = True
        model = hyper_residual_drone_dynamics(
            const_params=const_params,
            layer_sizes=[10, 32, 6],
            activation=cs.tanh
        )
        base_model_nr = cfg.mpc.base_model
        mlp_params = np.load(f"p_mlp_{base_model_nr}.npz")['p_mlp']        
        

    dt_mpc = cfg.mpc.dt
    
    mpc = MPC(model=model, dt=dt_mpc, cfg_mpc=cfg.mpc)
    solver = mpc.solver
    
    N = cfg.mpc.N
    T_horizon = N * dt_mpc
    print(f"T_horizon: {T_horizon}, with N: {N}")

    SIM_LEN = cfg.mpc.SIM_LEN
    
    for n in range(N+1):
        x_at_n = x_at_t.copy()
        solver.set(n, "x", x_at_n)
        
    if res_model:
        for n in range(N+1):
            solver.set(n, "p", np.concatenate([p_init[:6], p_mlp]))
    
    l_solve_time = []
    l_states = []
    l_ref_states = []
    l_force = []
    
    # create the reference trajectory, figure 8
    TRAJ_LEN = SIM_LEN + 2 * N
    t = np.linspace(0, TRAJ_LEN * dt_mpc, TRAJ_LEN)
    
    dataset_path = "/mnt/storage_5/scratch/pl0467-01/janw/Predict_Prediction_Model/artifacts/drone_dataset:v8"
    # my_path = "./artifacts/drone_dataset:v8"
    # dataset_path = my_path  
    
    dataset = DroneSimHyperDataset(dataset_path=dataset_path,
                                   dataset_file="test.csv",
                                    observation_window=50,
                                    prediction_horizon=100,
                                    stride=10,
                                    device='cpu')
    
    id_count = dataset.df['run_id'].max()
    df_in = dataset.df
    
    list_of_costs = []    
    
    for i in range(id_count):
        
        gc.collect()
        
        sim = DroneSim(config)
        sim.init_mpc_simulation()
        solver.reset()
        
        df_i = df_in[df_in['run_id'] == i]
        pose_x = df_i['x'].values
        pose_y = df_i['y'].values
        pose_z = df_i['z'].values
        
        ref_quat = np.array([1.0, 0.0, 0.0, 0.0])
        ref_v = np.array([0.0, 0.0, 0.0])
        ref_r = np.array([0.0, 0.0, 0.0]) 
        hower_thrust = 3.0
        ref_u = hower_thrust * np.ones(4)
        
        def get_ref_pose(t):
            ref_pose = np.array([pose_x[t], pose_y[t], pose_z[t]])
            return np.concatenate([ref_pose, ref_quat, ref_v, ref_r, ref_u])
        
        if not res_model and not res_hyper:
            for i in range(N):
                solver.set(i, "p", p_init)
        
        if res_hyper:
            p_to_mpc = np.zeros((N+1, 559))
            p_to_mpc[:, :6] = p_init[:6]
            p_to_mpc[:, 6:-3] = mlp_params
            p_to_mpc[:, -3:] = 0.0
            solver.set_flat("p", p_to_mpc.flatten())
        
        for i in range(100):
            solver.solve()

        episode_cost = 0.0
        
        episode_costs = np.zeros(SIM_LEN)
        
        obs_preprocessor = instantiate(cfgm.dataset.observations_preprocessor)
        
        def M_from_x(x):
            with torch.inference_mode():
                x = x[3:]
                x = torch.from_numpy(x).float().unsqueeze(0)
                x = obs_preprocessor.forward(x).squeeze(0).cpu().detach().numpy()
            return x
        
        hypep_prep = HyperModelPreprocesor(N, dt_mpc,
                                        cfgm.dataset.prediction_horizon,
                                        cfgm.dataset.Tp,
                                        cfgm.dataset.observation_window,
                                        cfgm.dataset.observarions_channels, 
                                        M_from_x_fun=M_from_x)
        P_x = np.zeros((N+1, 9))
        P_u = np.zeros((N+1, 4))    
        
        for i in tqdm(range(SIM_LEN)):
            
            if not res_model:
                M, P = hypep_prep.forward(x_at_t, P_u)
                
                with torch.inference_mode():
                    M = torch.from_numpy(M).float().unsqueeze(0)
                    P = torch.from_numpy(P).float().unsqueeze(0)
                    p_preint, _ = hyper_param_model(M, P)
                    p = param_interp(p_preint)
                    p = p[0, ::2, :].cpu().detach().numpy() # rk4
            

                if res_hyper:
                    p_to_mpc = np.zeros((N+1, 559))
                    p_to_mpc[:, :6] = p_init[:6]
                    p_to_mpc[:, 6:-3] = mlp_params
                    p_to_mpc[:, -3:] = hypep_prep.params_interp(p)
                    # l_force.append(p_to_mpc[5, 6:9])
                else:   
                    p_to_mpc = np.zeros((N+1, 12))
                    p_to_mpc[:, :6] = p_init[:6]
                    p_to_mpc[:, 6:9] = hypep_prep.params_interp(p)
                    
                    if cfg.mpc.adaptation == True:
                        p_to_mpc[:, 6:9] = sim.get_forces()
                    
                    # l_force.append(p_to_mpc[5, 6:9])
                
                if i > 200:
                    solver.set_flat("p", p_to_mpc.flatten())
                                
            for n in range(N):
                solver.set(n, "yref", get_ref_pose(i + n))
            solver.set(N, "yref", get_ref_pose(i + N)[:13])    
            
            now = perf_counter()
            
            solver.set(0, "lbx", x_at_t)
            solver.set(0, "ubx", x_at_t)
            res = solver.solve()      
            
            l_solve_time.append(perf_counter() - now)

            if res != 0:
                solver.print_statistics()   
                break

            P_u = solver.get_flat("u").reshape((N, -1))
            P_u = np.concatenate([P_u, 3*np.ones((1, 4))], axis=0)        

            u0 = solver.get(0, "u")

            x_at_t = sim.step_sim(u0)
            ref_pose = get_ref_pose(i)
            
            xu_at_t = np.concatenate([x_at_t, u0])
            episode_costs[i] = (xu_at_t - ref_pose).T @ Q @ (xu_at_t - ref_pose)
            
            # l_states.append(xu_at_t)
            # l_ref_states.append(get_ref_pose(i))
            
            
        sim.close_render()   
        
        print(f"episode_cost: {episode_cost}")
        mean_solve_time = np.mean(l_solve_time)
        print(f"mean_solve_time: {mean_solve_time}")
        
        print(f"episode cost {np.sum(episode_cost)}")
        
        # cost after 20s
        start_idx = int(3.0 / dt_mpc)
        # print in red 
        print(f"\033[1;31mepisode cost: {np.sum(episode_costs[start_idx:])} model {cfg.mpc.model_nr}\033[0m")
        
        list_of_costs.append(np.sum(episode_costs[start_idx:]))
        
        # plot states
        # states = np.array(l_states)
        # ref_states = np.array(l_ref_states)
        # force = np.array(l_force)
        # states_and_ref_states = np.concatenate([states, ref_states, force], axis=1)
        
        # collumns = dyn_model.get_state_names() +\
        #         dyn_model.get_control_names() + \
        #         ['ref_' + s for s in dyn_model.get_state_names()] +\
        #         ['ref_' + s for s in dyn_model.get_control_names()] +\
        #             ['force_x', 'force_y', 'force_z']
        
        # df = pd.DataFrame(states_and_ref_states, columns=collumns)
        # df['time'] = np.arange(len(df)) * dt_mpc
        
        # df.to_csv(f'./data/mpc_drone{cfg.mpc.rope_len}_model_{cfg.mpc.model_nr}.csv', index=False)
        
        # df = df[df['time'] > 3.0]
        # plt.plot(df['time'], df['ref_x'] - df['x'], label='e_x', linestyle='dashed')
        # plt.plot(df['time'], df['ref_y'] - df['y'], label='e_y', linestyle='dashed')    
        # plt.plot(df['time'], df['ref_z'] - df['z'], label='e_z', linestyle='dashed')
        # plt.savefig(f'e_pos{cfg.mpc.rope_len}.png')
        # plt.show()
    
    wandb.log({"mean_solve_time": np.mean(l_solve_time), "episodes_costs": list_of_costs})
    print(f"list_of_costs: {list_of_costs}")
    

if __name__ == "__main__":
    main()
