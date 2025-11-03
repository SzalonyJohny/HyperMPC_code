import torch
import numpy as np
from single_track import PacejkaTiresSingleTrack
from robot_model.car.state_wrapper import StateWrapper
from time import perf_counter
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# set default tensor type
# torch.set_default_tensor_type(torch.float64)

# Initialize the model
model = PacejkaTiresSingleTrack()

params = model.get_default_params()

model = torch.compile(model, fullgraph=True, mode="max-autotune")

print(f"Parameters shape: {params.shape}")

# Simulation parameters
dt = 0.01  # time step
T = 10  # total time
timesteps = int(T / dt)
time = torch.linspace(0, T, timesteps)

# Initial state [v_x, v_y, r, friction]
# u = [omega_wheels, delta]
x = torch.tensor([1.0, 0.0, 0.0, 0.90]).unsqueeze(0)
u = torch.tensor([0.0, 0.0]).unsqueeze(0)
print(x.shape)

# Storage for trajectory
trajectory = []
u_traj = []
inf_times = []

# Simulation loop
for i in range(timesteps):
    t = time[i]
    # Sinusoidal steering angle and constant current
    delta = 0.5 * torch.sin(2 * torch.pi * 0.1 * t)
    # Iq = torch.tensor(1.0)
    u[:, 0] =  torch.sin(2 * torch.pi * 20 * t + 0.5) + 2.0
    u[:, 1] = delta

    x = x.clone().detach()
    now = perf_counter()
    state_dot = model(t, x, u, params)
    
    # Update state using rk4
    k1 = state_dot * dt
    k2 = model(t + dt / 2, x + k1 / 2, u, params) * dt
    k3 = model(t + dt / 2, x + k2 / 2, u, params) * dt
    k4 = model(t + dt, x + k3, u, params) * dt
    x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    inference_time = perf_counter() - now
    
    if i > 100:
        inf_times.append(inference_time)    
    
    
    # Store the state
    trajectory.append(x.squeeze(0))
    u_traj.append(u.clone().squeeze(0))

# Convert trajectory to numpy array for plotting
trajectory = torch.stack(trajectory, dim=0).detach()
u = torch.stack(u_traj, dim=0).detach()

delta_traj = u[:, 1]
wheel_traj = u[:, 0]


# print(trajectory.shape)

print("Mean inference time: ", np.mean(inf_times))
# round output to 3 decimal places

# Extract the states
vx = trajectory[:, 0].numpy(force=True)
vy = trajectory[:, 1].numpy(force=True)
r = trajectory[:, 2].numpy(force=True)


# # Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(time, vx, label='v_x')
plt.plot(time, vy, label='v_y')
plt.plot(time, r, label='r')
plt.plot(time, delta_traj*10, label='delta')
plt.plot(time, wheel_traj, label='omega_wheels')

# # from robot_model.car.base_tire_model import BaseTireModel
# # from robot_model.car.single_track_params import VehicleParameters
# # tire = BaseTireModel()
# # param_wraper_st = VehicleParameters()

# # slip_angle = tire.slip_angle_front_func(StateWrapper(trajectory), param_wraper_st)

plt.xlabel('Time [s]')
plt.ylabel('State values')
plt.title('Trajectory of the Single Track Model')
plt.legend()
plt.grid()
plt.show()