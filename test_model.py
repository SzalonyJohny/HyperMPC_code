import torch
import wandb
from pathlib import Path
import dataset_reader.acrobot_dataset as acrobot_dataset
import dataset_reader.dataset_files_spliter as spliter
import robot_model.acrobot.acrobot_model
import robot_model.acrobot.observation_preprocesor
import robot_model.acrobot.acrobot_params
import hyper_prediction_models.spline_param_interpolation
import hyper_prediction_models.const_param_interpolation
import hyper_prediction_models.const_param_model
import hyper_prediction_models.rollout_model as rollout_model
import hyper_prediction_models.rnn_hyper_model
import hyper_prediction_models.rnn_time_series_encoder
import hyper_prediction_models.mlp_time_series_encoder
import hyper_prediction_models.tcnn_time_series_encoder
import robot_model.car.single_track
import robot_model.car.pacejka_params
import robot_model.car.pacejka_tire_model
import robot_model.car.state_wrapper
import time
from utils.df_save import create_df, create_df_p_traj
from pathlib import Path
import os
import math
import sys
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import hydra
from hydra.utils import instantiate
import os
import time
import logging
from typing import List
from copy import deepcopy
from utils.init_params_from_model import get_params_from_model

# car
from utils.set_up_hypermodel import setup_hypermodel


log = logging.getLogger(__name__)
torch.set_grad_enabled(False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

max_torch_num_threads = 8
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)


@hydra.main(version_base=None, config_path="conf", config_name="config_test")
def main(cfgt: DictConfig) -> float:

    conf_dict = OmegaConf.to_container(cfgt, resolve=True)

    log.info(f"Process ID {os.getpid()}")

    run = wandb.init(project='hpm_test_drone', group=cfgt.hpm.wandb.group,
                     config=conf_dict)
   
    hyper_param_model, param_interp, cfg, dyn_model = setup_hypermodel(cfgt, run)
    print(f"hyper_param_model: {hyper_param_model}")
    print(f" pred horizon: {cfg.dataset.val_dataset.prediction_horizon}")
  
    cfg.dataset.val_dataset.dataset_file = "test.csv"
    
    cfg.dataset.val_dataset.stride = 5
    # cfg.dataset.val_dataset.prediction_horizon = 253
    # cfg.dataset.val_dataset.mode = "test"
    M_win = deepcopy(cfg.dataset.val_dataset.observation_window)
    cfg.dataset.val_dataset.observation_window = 100
    loss_calc = 10000
    
    # cfg.dataset.val_dataset.dataset_path = "test_data/lab_rss1_test.csv"
    
    if cfg.dataset.wandb_path != "":
        artifact = run.use_artifact(cfg.dataset.wandb_path, type="dataset")
        artifact_dir = artifact.download()
        val_long_dataset = instantiate(
            cfg.dataset.val_dataset)(dataset_path=artifact_dir)
    else:
        val_long_dataset = instantiate(cfg.dataset.val_dataset)

    print(f"val_long_dataset: {val_long_dataset} len {len(val_long_dataset)}")
    

    val_long_dataset_loader = torch.utils.data.DataLoader(
        val_long_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=cfg.dataset.dataloader_workers,
        drop_last=False,
    )
    
    state_weights = dyn_model.state_weights().to(cfg.hpm.device)
    
    
    model_rollout = rollout_model.RolloutModel(dyn_model=dyn_model,
                                               intergration_method=cfg.hpm.integration_method,
                                               Tp=cfg.dataset.Tp,
                                               compile=cfg.hpm.compile,
                                               chunk_mode=False,
                                               chunk_size=1).to(cfg.hpm.device)
    
    model_rollout.eval()
    
    loss_fn = torch.nn.MSELoss(reduction="none")
    
    val_long_loss_list = []
    
    with torch.no_grad():

        for sample_i, data in enumerate(val_long_dataset_loader):
            M, P, U, X0, X = data
            M = M[:, :M_win]
            # print(f"m shape {M.shape}")
            
            p_preint, p_delta = hyper_param_model(M, P, True)
            p = param_interp(p_preint)
            
            # if p.shape[1] != cfg.dataset.val_dataset.prediction_horizon:
            #     p = torch.cat([p, p[:, -1:, :].repeat(1, 500, 1)], dim=1)
            
            X_sim = model_rollout(X0, U, p, cfg.dataset.val_dataset.prediction_horizon)

            loss_long = loss_fn(X_sim, X) * state_weights
            loss_long = loss_long[:, :loss_calc, :]

            val_long_loss_list.append(loss_long)

        
        val_long_loss = torch.cat(val_long_loss_list, dim=0)
        mean_epoch_val_long_loss = val_long_loss.mean().item()
        std_epoch_val_long_loss = val_long_loss.mean((-1, -2), keepdim=True).std().item()
        
        # log.info(f"all val_long_loss {val_long_loss_list}")
        log.info(f"all val_long_loss {val_long_loss.mean((-1, -2), keepdim=False).shape}")
        
        # np.savez(f"./data/drone_pred_res/model_{cfgt.hpm.model_nr}_val_long_loss", val_long_loss.cpu().numpy())
    
        print(f"\033[31m  model nr {cfgt.hpm.model_nr} val_long_loss: {mean_epoch_val_long_loss}, std_val_long_loss: {std_epoch_val_long_loss}\033[0m")

    
if __name__ == "__main__":
    main()
