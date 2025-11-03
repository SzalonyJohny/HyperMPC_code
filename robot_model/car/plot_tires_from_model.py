import torch
import os
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate


run = wandb.init()
artifact = run.use_artifact('f1tenth/hpm_car/model:v23321', type='model')
artifact_dir = artifact.download()

print(f"artifact_dir: {artifact_dir}")

model_path = os.path.join(artifact_dir, 'hyper_model.pt')   
config_path = os.path.join(artifact_dir, '.hydra/config.yaml')
model_config_path_rel = os.path.relpath(config_path, start=os.getcwd())

cfg2 = OmegaConf.load(model_config_path_rel)

def replace_device(cfg, device):
    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            replace_device(cfg[key], device)
        elif key == 'device':
            cfg[key] = device
        

device = torch.device("cpu")
replace_device(cfg2, device)

dyn_model = instantiate(cfg2.dmodel.model)(init_params=None)
dyn_model.eval()
dyn_model = dyn_model.to(device)

hyper_param_model = instantiate(cfg2.hmodel.model)(default_params=dyn_model.get_default_params(),
                                                    always_positive=dyn_model.positive_params())

hyper_param_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
hyper_param_model.eval()
hyper_param_model = hyper_param_model.to(device)
param_interp = instantiate(cfg2.hmodel.interpoler)
param_interp = param_interp.to(device)
param_interp.eval()


print(hyper_param_model)

# const_params = hyper_param_model.const_params

# print(const_params.exp())
# p = const_params.exp()
M = torch.zeros(1, 1)
p, _ = hyper_param_model(M)

vehicle_parameters = dyn_model.vehicle_parameters
tire_model_parameters = dyn_model.tire_model_parameters

wp_car, p = vehicle_parameters(p)
wp_tire_f, p = tire_model_parameters(p)
wp_tire_r, p = tire_model_parameters(p)

wp = wp_tire_f  

import matplotlib.pyplot as plt
from robot_model.car.pacejka_tire_model import PacejkaTireModel
from robot_model.car.pacejka_params import PacejkaParameters


# Define slip angles and slip ratios for the plot
slip_angles = torch.linspace(0, 70, 100)  # Slip angles from 0 to 20 degrees
slip_ratios = [0, 0.03, 0.06, 0.09, 0.15, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8 ]  # Different slip ratios

# Create a plot for the tire model "Fy vs slip angle under different slip ratios"
plt.figure(figsize=(10, 6))

for Sx in slip_ratios:
    Fy_values = []
    for Alpha in slip_angles:
        Fy = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx, wp)[1]
        Fy_values.append(- Fy.item())
    
    plt.plot(slip_angles.numpy(), Fy_values, label=f'Slip ratio = {Sx * 100:.0f}%')

# Plot settings
plt.title('Lateral Force (Fy) vs Slip Angle under Different Slip Ratios')
plt.xlabel('Slip Angle (degrees)')
plt.ylabel('Lateral Force (N)')
plt.legend()
plt.grid(True)

# Define slip ratios and slip angles for the plot
slip_ratios = torch.linspace(0, 0.9, 100)  # Slip ratios from 0 to 0.6 (60%)
slip_angles = [0, 2, 4, 6, 8, 12, 20, 30, 40, 50, 60, 70]  # Different slip angles in degrees

# Create a plot for the tire model "Fx vs slip ratio under different slip angles"
plt.figure(figsize=(10, 6))

for Alpha in slip_angles:
    Fx_values = []
    for Sx in slip_ratios:
        Fx = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx.item(), wp)[0]
        Fx_values.append(Fx.item())
    
    plt.plot(slip_ratios.numpy(), Fx_values, label=f'Slip angle = {Alpha}°')

# Plot settings
plt.title('Longitudinal Force (Fx) vs Slip Ratio under Different Slip Angles')
plt.xlabel('Slip Ratio')
plt.ylabel('Longitudinal Force (N)')
plt.legend()
plt.grid(True)

# Generate a grid of slip angle and slip ratio pairs for testing
slip_ratios_grid = torch.linspace(0, 0.95, 200)  # Slip ratios from 0 to 0.95
slip_angles_grid = torch.linspace(0, 75, 200)  # Slip angles from 0 to 45 degrees

# Create meshgrid for the slip ratios and slip angles
Slip_ratios, Slip_angles = torch.meshgrid(slip_ratios_grid, slip_angles_grid, indexing='ij')

# Compute forces for each pair and store them
Fx_values = []
Fy_values = []

for i in range(Slip_ratios.shape[0]):
    for j in range(Slip_ratios.shape[1]):
        Sx = Slip_ratios[i, j]
        Alpha = Slip_angles[i, j]
        Fx, Fy = PacejkaTireModel.tire_forces_model(Alpha * torch.pi / 180.0, Sx.item(), wp)
        Fx_values.append(Fx.item())
        Fy_values.append(- Fy.item())

# Create the scatter plot (tire ellipse)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(Fx_values, Fy_values, c=Slip_angles, s=10, alpha=0.5, cmap='viridis', label='Tire Force Ellipse')
plt.colorbar(scatter, label='Slip Angle (degrees)')
plt.title('Tire Force Ellipse: Fx vs Fy')
plt.xlabel('Longitudinal Force (Fx) [N]')
plt.ylabel('Lateral Force (Fy) [N]')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.axis('equal')
plt.legend()

# Create the scatter plot (tire ellipse)
plt.figure(figsize=(8, 8))
scatter = plt.scatter(Fx_values, Fy_values, c=Slip_ratios, s=10, alpha=0.5, cmap='viridis', label='Tire Force Ellipse')
plt.colorbar(scatter, label='Slip Ratio (degrees)')
plt.title('Tire Force Ellipse: Fx vs Fy')
plt.xlabel('Longitudinal Force (Fx) [N]')
plt.ylabel('Lateral Force (Fy) [N]')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()