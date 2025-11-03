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
import numpy as np

os.environ["WANDB_API_KEY"] = "56dceee73d5b31715f9476dc86527a75377caf6c"

log = logging.getLogger(__name__)

debug = False

max_torch_num_threads = 8
torch.backends.mkldnn.enabled = True
torch.set_num_threads(max_torch_num_threads)
torch.set_num_interop_threads(max_torch_num_threads)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('highest')


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


if debugger_is_active():
    debug = True
    # os.environ["WANDB_MODE"] = "disabled"
    torch.autograd.set_detect_anomaly(True)

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    args = cfg.hpm
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f"Process ID {os.getpid()}  seed {args.seed}")
    log.info(f"Output directory  : {cfg_dir}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    run = wandb.init(project=cfg.dataset.wandb_project,
                     group=args.wandb.group,
                     config=conf_dict)

    if cfg.dataset.wandb_path != "":
        artifact = run.use_artifact(cfg.dataset.wandb_path, type="dataset")
        artifact_dir = artifact.download()
        log.info(f"Artifact {artifact_dir}")
        train_dataset = instantiate(cfg.dataset.train_dataset)(
            dataset_path=artifact_dir)
        val_long_dataset = instantiate(
            cfg.dataset.val_dataset)(dataset_path=artifact_dir)
    else:
        train_dataset = instantiate(cfg.dataset.train_dataset)
        val_long_dataset = instantiate(cfg.dataset.val_dataset)

    # train_dataset.plot()
    # if debug:
    #     raise ValueError("Debug mode, select dataset artifact")

    # Dataset loaders
    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.dataloader_workers,
        pin_memory=False,
        # persistent_workers=True,
        # multiprocessing_context='fork',
        drop_last=True,
    )

    val_long_dataset_loader = torch.utils.data.DataLoader(
        val_long_dataset,
        batch_size=args.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.dataloader_workers,
        pin_memory=False,
        # persistent_workers=True,
        # multiprocessing_context='fork',
        drop_last=True,
    )
    
    # print batch per epoch
    log.info(f"Train dataset size in batch{len(train_dataset_loader)}")
    log.info(f"Val dataset size in batch{len(val_long_dataset_loader)}")

    if cfg.dmodel.pretrain_init_model_path != "":
        pretirain_init_params = get_params_from_model(
            run, cfg.dmodel.pretrain_init_model_path)
    else:
        pretirain_init_params = None
 
    # Model
    dyn_model = instantiate(cfg.dmodel.model)(
        init_params=pretirain_init_params)
    dyn_model = dyn_model.to(args.device)
    log.info(dyn_model)

    hyper_param_model = instantiate(cfg.hmodel.model)(default_params=dyn_model.get_default_params(),
                                                      always_positive=dyn_model.positive_params(),
                                                      free_params=dyn_model.free_params())

    hyper_param_model = hyper_param_model.to(args.device)
    log.info(hyper_param_model)

    param_interp = instantiate(cfg.hmodel.interpoler).to(args.device)

    chunk_mode = (args.rollout.val == args.rollout.train) and args.chunk_mode and (args.rollout.train != 2)

    model_rollout = rollout_model.RolloutModel(dyn_model=dyn_model,
                                               intergration_method=args.integration_method,
                                               Tp=cfg.dataset.Tp,
                                               compile=args.compile,
                                               chunk_mode=chunk_mode,
                                               chunk_size=args.chunk_size).to(args.device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_count = count_parameters(hyper_param_model)
    inactive_param_count = hyper_param_model.get_inactive_parameter_count()
    log.info(f"Hyper param_count {param_count}")
    log.info(f"Hyper inactive {inactive_param_count}")
    log.info(f"Model overal {param_count - inactive_param_count}")

    if not debug:
        wandb.watch(hyper_param_model, log_freq=1000)

    params_to_clip = list(hyper_param_model.parameters()) \
        + list(dyn_model.parameters())
        
    params_to_optimize = list(hyper_param_model.named_parameters()) \
        + list(dyn_model.named_parameters())

    # log.info(f"Total params to optimize {(params_to_optimize)}")

    param_dict = {pn: p for pn, p in params_to_optimize}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': cfg.hpm.optimizer.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Optimization
    optimizer = instantiate(cfg.hpm.optimizer)(optim_groups)

    log.info(optimizer)

    loss_fn = torch.nn.MSELoss(reduction="none")
    p_delta_loss_fn = torch.nn.L1Loss(reduction="none")

    best_val_loss = float("inf")
    best_hmodel = None
    best_dmodel = None
    best_model_df = None
    best_model_df_p_traj = None

    future_head_trining = False

    # first_sample = next(iter(train_dataset_loader))

    state_weights = dyn_model.state_weights().unsqueeze(0).unsqueeze(0).to(args.device)

    assert args.rollout.train <= args.rollout.val
    len_idx = args.rollout.train

    for epoch in range(args.train.epoch):

        epoch_start_time = time.time()

        future_head_trining = epoch > args.train.start_future_head_training

        grad_norm_sum = 0
        train_short_loss_list = []
        train_long_loss_list = []
        ratio_p_delta_L2_loss_list = []

        hyper_param_model.train()

        # Training loop
        for sample_i, data in enumerate(train_dataset_loader):
            M, P, U, X0, X = data
            # M, P, U, X0, X = first_sample

            M = M.to(args.device)
            U = U.to(args.device)
            X0 = X0.to(args.device)
            X = X.to(args.device)
            P = P.to(args.device)

            optimizer.zero_grad()

            # torch.compiler.cudagraph_mark_step_begin()
            p_preint, p_delta = hyper_param_model(M, P, future_head_trining)
            p_preint.retain_grad()
            p = param_interp(p_preint)
            X_sim = model_rollout(
                X0, U, p, cfg.dataset.train_dataset.prediction_horizon)

            loss_short_bts = loss_fn(
                X_sim[:, :len_idx, :], X[:, :len_idx, :]) * state_weights
            loss_long_bts = loss_fn(X_sim, X) * state_weights
  
            loss = torch.sum(loss_short_bts) / loss_short_bts.numel()

            p_delta_L2_loss_btp = p_delta_loss_fn(
                p_delta, torch.ones_like(p_delta))

            p_delta_loss_schedule = args.train.delta_p_L2_loss \
                + math.sin(0.5 * epoch) * args.train.delta_p_L2_loss_sin

            p_delta_L2_loss = (
                torch.sum(p_delta_L2_loss_btp) / p_delta_L2_loss_btp.numel()
            ) * p_delta_loss_schedule

            loss += p_delta_L2_loss
            loss.backward()

            with torch.no_grad():
                grad_norm_sum += (torch.norm(p_preint.grad) / (
                    (p_preint.grad.numel()) ** 0.5
                )).item()

            torch.nn.utils.clip_grad_norm_(
                params_to_clip, args.train.grad_clip
            )

            optimizer.step()

            # Logging
            ratio_p_delta_L2_loss_list.append(
                p_delta_L2_loss.detach() / loss.detach())
            train_short_loss_list.append(loss_short_bts.detach())
            train_long_loss_list.append(loss_long_bts.detach())

        # Validation
        val_short_loss_list = []
        val_long_loss_list = []

        df = None
        df_p_traj = None

        hyper_param_model.eval()

        with torch.no_grad():

            list_of_trajs = []
            p_const = None

            for sample_i, data in enumerate(val_long_dataset_loader):
                M, P, U, X0, X = data

                M = M.to(args.device)
                U = U.to(args.device)
                X0 = X0.to(args.device)
                X = X.to(args.device)
                P = P.to(args.device)

                p_preint, p_delta = hyper_param_model(
                    M, P, future_head_trining)
                p = param_interp(p_preint)
                
                X_sim = model_rollout(
                    X0, U, p, cfg.dataset.val_dataset.prediction_horizon)

                if p_const is None:
                    p_const = p_preint / p_delta  # same for all time steps
                    p_const = p_const[0, 0, :].detach().cpu()
                # every len / stride steps
                new_traj = X_sim.shape[1] // cfg.dataset.val_dataset.stride
                if args.rollout.val != 2:
                    list_of_trajs.append((X_sim[::new_traj, :, :].detach().cpu(),
                                        X[::new_traj, :, :].detach().cpu(),
                                        p[::new_traj, :, :].detach().cpu(),
                                        U[::new_traj, :, :].detach().cpu()))

                loss_short = loss_fn(
                    X_sim[:, :len_idx, :], X[:, :len_idx, :]) * state_weights
                loss_long = loss_fn(X_sim, X) * state_weights

                val_short_loss_list.append(loss_short)
                val_long_loss_list.append(loss_long)

            log.info(f"p_const {p_const}")
            time_df_start = time.time()
            if  args.rollout.val != 2:
                X_sim_traj = torch.cat([traj[0] for traj in list_of_trajs], dim=0)
                X_traj = torch.cat([traj[1] for traj in list_of_trajs], dim=0)
                p_traj = torch.cat([traj[2] for traj in list_of_trajs], dim=0)
                U_traj = torch.cat([traj[3] for traj in list_of_trajs], dim=0)

                df = create_df(
                    X=X_traj.numpy(),
                    X_sim=X_sim_traj.numpy(),
                    u=U_traj.numpy(),
                    wandb_table_len=args.wandb.table_len,
                    state_names=dyn_model.get_state_names(),
                    control_names=dyn_model.get_control_names()
                )

                df_p_traj = create_df_p_traj(
                    p=p_traj.numpy(),
                    p_const=p_const.numpy(),
                    wandb_table_len=args.wandb.table_len,
                    param_names=dyn_model.get_params_names(),
                )

            log.info(f"df time {time.time() - time_df_start}")

        # Logging
        mean_epoch_train_short_loss = (
            torch.cat(train_short_loss_list, dim=0).mean().item()
        )
        mean_epoch_train_long_loss = (
            torch.cat(train_long_loss_list, dim=0).mean().item()
        )
        mean_epoch_val_short_loss = torch.cat(
            val_short_loss_list, dim=0).mean().item()
        mean_epoch_val_long_loss = torch.cat(
            val_long_loss_list, dim=0).mean().item()
        mean_p_delta_L2_loss_ratio = (
            torch.tensor(ratio_p_delta_L2_loss_list).mean().item()
        )

        log_dict = {
            "train_short_loss": mean_epoch_train_short_loss,
            "train_long_loss": mean_epoch_train_long_loss,
            "val_short_loss": mean_epoch_val_short_loss,
            "val_long_loss": mean_epoch_val_long_loss,
            "best_long_val_loss": best_val_loss,
            "ratio_delta_p_L2_loss": mean_p_delta_L2_loss_ratio,
            "norm_grad": grad_norm_sum,
            "len_idx": len_idx,
            "epoch_time": time.time() - epoch_start_time,
        }
        
        # artifacts = torch.compiler.save_cache_artifacts()
        # assert artifacts is not None
        # artifact_bytes, cache_info = artifacts

        wandb.log(log_dict, step=epoch)

        print_str = f"epoch {epoch}, "
        for key, value in log_dict.items():
            print_str += f" {key} {value:.4f}, "
        log.info(print_str)

        # Save best model
        if mean_epoch_val_long_loss < best_val_loss:
            best_val_loss = mean_epoch_val_long_loss
            best_hmodel = deepcopy(hyper_param_model.state_dict())
            best_dmodel = deepcopy(dyn_model.state_dict())
            best_model_df = deepcopy(df)
            best_model_df_p_traj = deepcopy(df_p_traj)

        if epoch % args.wandb.send_interval == 0:
            wandb_run_name = run.name
            outpath = Path("./data/models") / wandb_run_name / f"epoch_{epoch}"
            outpath.mkdir(parents=True, exist_ok=True)
            torch.save(best_hmodel, outpath / "hyper_model.pt")
            torch.save(best_dmodel, outpath / "dyn_model.pt")
            artifact = wandb.Artifact(f"model", type="model")
            artifact.add_dir(outpath)
            artifact.add_dir(cfg_dir)
            artifact.save()

            if best_model_df is not None:
                wandb.log({"val_rollout": wandb.Table(dataframe=best_model_df)},
                        step=epoch)

            if dyn_model.save_param_traj() and best_model_df_p_traj is not None:
                wandb.log({"val_p_traj": wandb.Table(dataframe=best_model_df_p_traj)},
                          step=epoch)
        
        if torch.isnan(torch.tensor([mean_epoch_train_short_loss,
                                     mean_epoch_train_long_loss,
                                     mean_epoch_val_short_loss,
                                     mean_epoch_val_long_loss,
                                     mean_p_delta_L2_loss_ratio])).any():
            log.error("NAN detected")
            break

    # save model
    wandb_run_name = run.name
    outpath = Path("./data/models") / wandb_run_name
    outpath.mkdir(parents=True, exist_ok=True)
    torch.save(best_hmodel, outpath / "hyper_model.pt")
    torch.save(best_dmodel, outpath / "dyn_model.pt")
    artifact = wandb.Artifact("model", type="model")
    artifact.add_dir(outpath)
    artifact.add_dir(cfg_dir)
    artifact.save()

    run.finish()

    return best_val_loss


if __name__ == "__main__":
    main()
