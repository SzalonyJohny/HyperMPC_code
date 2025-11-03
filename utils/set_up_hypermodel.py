from omegaconf import OmegaConf
from hydra.utils import instantiate
import os
import torch
from pathlib import Path



def setup_hypermodel(cfg, wandb_run):
    
    if not cfg.mpc.wandb.enable:
        artifact_dir = Path('/local_model_prediction/artifacts/') / cfg.mpc.model.split('/')[-1]
    else:
        artifact = wandb_run.use_artifact(cfg.mpc.model, type='model')
        artifact_dir = artifact.download()
    
    print(f"artifact_dir: {artifact_dir}")
    model_path = os.path.join(artifact_dir, 'hyper_model.pt')
    dyn_model_path = os.path.join(artifact_dir, 'dyn_model.pt')  
    config_path = os.path.join(artifact_dir, '.hydra/config.yaml')
    model_config_path_rel = os.path.relpath(config_path, start=os.getcwd())
    
    device = torch.device("cpu")
    cfg2 = OmegaConf.load(model_config_path_rel)
    cfg2.hpm.device = 'cpu'
    
    # FIXME download the init model eval and use as init params
    dyn_model = instantiate(cfg2.dmodel.model)(init_params=None)
    print(f"Model loaded {dyn_model.state_dict()}")
    
    # check if path exists
    if os.path.exists(dyn_model_path):
        print(f"Loading model from {dyn_model_path}")
        dyn_model.load_state_dict(torch.load(dyn_model_path, map_location=device, weights_only=False))
        print(f"Model loaded {dyn_model.state_dict()}")
    
    dyn_model.eval()
    dyn_model = dyn_model.to(device)
    print(f"dyn_model: {dyn_model}")
    
    
    try:
        hyper_param_model = instantiate(cfg2.hmodel.model)(default_params=dyn_model.get_default_params(),
                                                           always_positive=dyn_model.positive_params(), 
                                                           free_params=dyn_model.free_params())
    except Exception as e:
        hyper_param_model = instantiate(cfg2.hmodel.model)(default_params=dyn_model.get_default_params(),
                                                      always_positive=dyn_model.positive_params(),
                                                      free_params=None)
    
    
    hyper_param_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    hyper_param_model.eval()
    hyper_param_model = hyper_param_model.to(device)
    param_interp = instantiate(cfg2.hmodel.interpoler)
    param_interp = param_interp.to(device)
    param_interp.eval()
    
    return hyper_param_model, param_interp, cfg2, dyn_model