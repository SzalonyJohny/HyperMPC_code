
import l4casadi as l4c
import casadi as cs
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from robot_model.car.single_track import PacejkaTiresSingleTrack

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class CasadiDynamicsModelWraper(torch.nn.Module):
    def __init__(self, model, state_dim, control_dim):
        super(CasadiDynamicsModelWraper, self).__init__()
        self.model = model
        self.state_dim = state_dim
        self.control_dim = control_dim

    def forward(self, txup):
        t = txup[:, 0]
        x = txup[:, 1:1+self.state_dim]
        u = txup[:, 1+self.state_dim:1 + self.state_dim + self.control_dim]
        p = txup[:, self.state_dim+self.control_dim+1:]
        return self.model(t, x, u, p)


# PacejkaTiresSingleTrack
dyn_model = PacejkaTiresSingleTrack()
p_init = dyn_model.get_default_params()
pyTorch_model = CasadiDynamicsModelWraper(dyn_model, 4, 2)

t = np.ones((1, 1))
x = np.array([2.0, 0.5, 1.0, 0.8]).reshape(1, -1)
u = np.zeros((1, 2))
p = p_init.numpy().reshape(1, -1)
txup = np.concatenate((t, x, u, p), axis=-1)
print(txup.shape)

# torch test
txup_torch = torch.tensor(txup, dtype=torch.float32)
print(txup_torch.shape)
x_dot_torch = pyTorch_model(txup_torch).squeeze(0)
print(f'x_dot: torch wraped: {x_dot_torch}')

model_name = pyTorch_model.__class__.__name__


l4c_model = l4c.L4CasADi(pyTorch_model, batched=True,
                         device='cpu', name=f'{model_name}',
                         scripting=True,
                         mutable=False)

# time
t = cs.MX.sym('t')

# state variables
vx = cs.MX.sym('vx')
vy = cs.MX.sym('vy')
r = cs.MX.sym('r')
friction = cs.MX.sym('friction')
x = cs.vertcat(vx, vy, r, friction)

# control variables
delta = cs.MX.sym('delta')
wheel_speed = cs.MX.sym('wheel_speed')
u = cs.vertcat(delta, wheel_speed)

# parameters
p = cs.MX.sym('p', p_init.shape[-1])

txup_sym = cs.vertcat(t, x, u, p).reshape((1, -1))

x_dot = l4c_model(txup_sym)

# rk4 integration
Tp = 0.01
t0 = np.array([0.0])
x0 = np.array([2.0, 0.5, 1.0, 0.8])
u0 = np.zeros(2)
p0 = p_init.numpy().reshape(-1)

txup0 = np.concatenate((t0, x0, u0, p0), axis=-1)
x_dot = l4c_model(cs.vertcat(t0, x0, u0, p0))

print(f'x_dot: l4casadi: {x_dot}')

x_torch = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
u_torch = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)
p = dyn_model.get_default_params()

x_dot = dyn_model(0.0, x_torch, u_torch, p).detach().numpy()
print(f'x_dot: torch: {x_dot}')

