import casadi as cs
import numpy as np


def drone_dynamics(const_params=None):
    """
    Computes the dynamics of a quadrotor drone using a new dynamic model formulation
    with parameters provided as a numpy array in the order:
      [ mq, Db, Cd, Ct, g0, Jx, Jy, Jz, l, 
        f_dist_x, f_dist_y, f_dist_z, tau_dist_x, tau_dist_y, tau_dist_z ]
    
    The state x is 13-dimensional:
         x = [ p (3), q (4), v (3), r (3) ]
    where:
         p: position in the inertial frame,
         q: attitude quaternion [w, x, y, z] (unit norm),
         v: linear velocity,
         r: angular velocity (body rates).

    The control input u is 4-dimensional, containing the motor speeds:
         u = [omega1, omega2, omega3, omega4]
    """

    def casadi_skew_symmetric(r):
        return cs.vertcat(
            cs.horzcat(0, -r[0], -r[1], -r[2]),
            cs.horzcat(r[0], 0, r[2], -r[1]),
            cs.horzcat(r[1], -r[2], 0, r[0]),
            cs.horzcat(r[2], r[1], -r[0], 0)
        )

    def casadi_quat_to_rot_matrix(q):
        w, x_, y_, z_ = q[0], q[1], q[2], q[3]
        return cs.vertcat(
            cs.horzcat(1 - 2*(y_**2 + z_**2), 2*(x_*y_ - w*z_),     2*(x_*z_ + w*y_)),
            cs.horzcat(2*(x_*y_ + w*z_),     1 - 2*(x_**2 + z_**2), 2*(y_*z_ - w*x_)),
            cs.horzcat(2*(x_*z_ - w*y_),     2*(y_*z_ + w*x_),     1 - 2*(x_**2 + y_**2))
        )
        
    # Declare model variables
    pos = cs.MX.sym('p', 3)  # position
    q = cs.MX.sym('a', 4)  # angle quaternion (wxyz)
    v = cs.MX.sym('v', 3)  # velocity
    r = cs.MX.sym('r', 3)  # angle rate

    # Full state vector (13-dimensional)
    x = cs.vertcat(pos, q, v, r)
    state_dim = 13
    
    x_dot = cs.MX.sym('x_dot', state_dim)  # state derivative
    
    # Control input vector
    u1 = cs.MX.sym('u1')
    u2 = cs.MX.sym('u2')
    u3 = cs.MX.sym('u3')
    u4 = cs.MX.sym('u4')
    u = cs.vertcat(u1, u2, u3, u4)
    
    # Parameters constants
    mq = const_params['mq']   # mass
    g0  = const_params['g0']  # lever arm
    l   = const_params['l']   # lever arm
    
    p = cs.MX.sym('params', 12)
    Db       = p[0]   # body drag coefficient
    Cd       = p[1]   # motor drag coefficient
    Ct       = p[2]   # thrust coefficient
    J_val    = cs.vertcat(p[3], p[4], p[5])  # inertias [Jx, Jy, Jz]
    f_dist   = cs.vertcat(p[6], p[7], p[8])   # disturbance forces
    tau_dist = cs.vertcat(p[9], p[10], p[11]) # disturbance torques

    # (1) Position derivative.
    p_dot = v

    # (2) Quaternion derivative.
    S = casadi_skew_symmetric(r)
    q_dot = 0.5 * (S @ q)

    # (3) Rotation matrix from quaternion.
    R_mat = casadi_quat_to_rot_matrix(q)

    # (4) Thrust force.
    f_total = Ct * (u1 + u2 + u3 + u4)
    F_body = cs.vertcat(0, 0, f_total)

    # (5) Thrust acceleration.
    a_thrust = (R_mat @ F_body) / mq

    # (6) Drag force.
    vx, vy, vz = v[0], v[1], v[2]
    drag = (Db / mq) * cs.vertcat(2*vx, 2*vy, 0.0)

    # (7) Gravity vector.
    g = cs.vertcat(0, 0, g0)

    # (8) Linear acceleration.
    v_dot = -g + a_thrust - drag + f_dist / mq

    # (9) Motor-induced moments.
    M_x = Ct * l * (u1 - u2 - u3 + u4)
    M_y = Ct * l * (- u1 - u2 + u3 + u4) 
    M_z = Cd * (- u1 + u2 - u3 + u4)
    M_motor = cs.vertcat(M_x, M_y, M_z) * -1.0
    M = M_motor + tau_dist

    # (10) Rotational dynamics.
    J = cs.diag(J_val)
    r_dot = (M - cs.cross(r, J @ r)) / J_val

    # (11) Concatenate state derivatives.
    f_expl =  cs.vertcat(p_dot, q_dot, v_dot, r_dot)
    
    func = cs.Function('drone_dynamics', [x, u, p], [f_expl], ['x', 'u', 'params'], ['x_dot'])    
    
    model = cs.types.SimpleNamespace()
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.z = cs.vertcat([])
    model.p = p
    model.parameter_values = np.array([0.0000e+00, 3.0375e-05,\
                                       1.0295e+00, 4.7043e-02,\
                                       8.4545e-02, 2.0622e-02,\
                                       0.0000e+00, 0.0000e+00,\
                                       0.0000e+00, 0.0000e+00,\
                                       0.0000e+00, 0.0000e+00])
    model.f_expl = f_expl
    model.x_start = np.array([
        0.5, 0.0, 0.0,  # pose [p]
        1.0, 0.0, 0.0, 0.0,  # orientation [q]
        0.0, 0.0, 0.0,  # velocity [v]
        0.0, 0.0, 0.0   # angular velocity [r]
    ])
    model.constraints = cs.vertcat([])
    model.name = "drone_casadi"
    model.dyn_fun = func
    return model
    

if __name__ == "__main__":

    import torch
    from robot_model.drone.drone_model import DroneDynamisc

    model = DroneDynamisc(enable_dist_forces=True)

    params = np.array([0.0000e+00, 3.0375e-05, 1.0295e+00, 4.7043e-02, 8.4545e-02, 2.0622e-02,\
                       0.00012e+00, 0.123000e+00, 0.0124000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])

    # FIXME  get const params from buffers of the model
    const_params = {
        'mq': 1.325 + 0.5,   # mass
        'g0': 9.80665, # gravitational acceleration
        'l': 0.228035  # distance (half between motors' center and rotation axis)
    }
    
    f = drone_dynamics(const_params).dyn_fun
    
    for i in range(100):
        x_casadi = np.random.rand(13)
        u_casadi = np.random.rand(4)
        x_casadi[3:7] /= np.linalg.norm(x_casadi[3:7])  # Normalize quaternio
        params[6:9] = np.random.rand(3)
    
        x_dot_casadi = f(x_casadi, u_casadi, params)  
        
        x_torch = torch.tensor(x_casadi, dtype=torch.float32).unsqueeze(0)
        u_torch = torch.tensor(u_casadi, dtype=torch.float32).unsqueeze(0)
        t_torch = torch.tensor([0.0], dtype=torch.float32)
        p_torch = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        # Compute state derivative via PyTorch.
        x_dot_torch = model(t_torch, x_torch, u_torch, p_torch).squeeze(0).detach().numpy()
        
        print(f"x_casadi: {x_casadi[3:]}")
        print(f"x_torch: {x_torch[0,3:]}")
        diff = np.linalg.norm(x_dot_casadi - x_dot_torch)
        print(diff)
        assert diff < 1e-5, f"Difference too large: {diff}"
    

    