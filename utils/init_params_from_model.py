import torch
import wandb

import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

def get_params_from_model(wandb_run, model_path: str):
    
    artifact = wandb_run.use_artifact(model_path, type='model')
    artifact_dir = artifact.download()
    
    model_path = os.path.join(artifact_dir, 'hyper_model.pt')   
    config_path = os.path.join(artifact_dir, '.hydra/config.yaml')
    model_config_path_rel = os.path.relpath(config_path, start=os.getcwd())

    device = torch.device("cpu")
    
    cfg2 = OmegaConf.load(model_config_path_rel)
    cfg2.hpm.device = 'cpu'

    init_params = None    
    dyn_model = instantiate(cfg2.dmodel.model)(init_params=init_params)
    dyn_model = dyn_model.to(device)
    
    # TODO add assert on dyn_model type

    hyper_param_model = instantiate(cfg2.hmodel.model)(default_params=dyn_model.get_default_params(),
                                                        always_positive=dyn_model.positive_params())

    hyper_param_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    hyper_param_model = hyper_param_model.to(device)
    
    p, _ = hyper_param_model(torch.zeros(1, 1))
    
    p = p.detach().squeeze(1) # squeeze time dimension
    
    return p