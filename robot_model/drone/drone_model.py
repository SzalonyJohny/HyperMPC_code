import torch
from robot_model.drone.drone_params_selected import DroneParamsWrapper

from robot_model.drone.drone_params_only_forces import DroneParamsWrapperForce

class DroneDynamisc(torch.nn.Module):
    def __init__(self,
                 init_params=None,
                 enable_dist_forces=False,
                 enable_dist_torques=False,
                 only_forces=False,
                 only_forces_regres_params = False,
                 *args, **kwargs) -> None:
        super(DroneDynamisc, self).__init__(*args, **kwargs)

        self.register_buffer('init_params', init_params)
        
        if only_forces:
            self.param_wrapper = DroneParamsWrapperForce(
                enable_forces=enable_dist_forces,
                enable_torques=enable_dist_torques,
                regres_params=only_forces_regres_params
            )    
        else:
            self.param_wrapper = DroneParamsWrapper(
                enable_forces=enable_dist_forces,
                enable_torques=enable_dist_torques
            )
    
    def get_default_params(self, batch_size=1):
        if self.init_params is not None:
            return self.init_params
        return self.param_wrapper.get_default_params(batch_size)
    
    def count_params(self):
        return self.param_wrapper.param_count

    def forward(self, t, x, u, p):
        """            
            t = [batch]
            x = [batch, state dim -> 13]
            u = [batch, propeler_speed_Omega -> 4]
            p = [batch, param_count]
        """ 
        assert x.shape[-1] == 13    # x = [pos3D_3el, quat_SO3_4el,  <- global
                                    #      vel3D_3el, ang_vel3D_3el] <- body frame
        assert u.shape[-1] == 4     # torques on propelers
                
        pw, p = self.param_wrapper(p)

        pose = x[..., 0:3]      # position [x, y, z]
        q = x[..., 3:7]      # quaternion [w, x, y, z]
        v = x[..., 7:10]     # linear velocity [vx, vy, vz]
        r = x[..., 10:13]    # angular velocity [phi(roll), theta(pitch), psi(yaw)]
        
        q = self._renorm_quat(q)  # Ensure unit norm. 
        
        # (1) Position derivative.
        p_dot = v
        
        # (2) Quaternion derivative.
        S = self._skew_symmetric(r)
        q_dot = 0.5 * torch.matmul(S, q.unsqueeze(-1)).squeeze(-1)
        
        # (3) Compute thrust force.
        f_total = pw.Ct * torch.sum(u, dim=-1)  # shape: (batch,)
        F_body = torch.stack([torch.zeros_like(f_total),
                              torch.zeros_like(f_total),
                              f_total], dim=-1)  # (batch, 3)
        
        # (5) Thrust acceleration in inertial frame.
        R_mat = self._quat_to_rot_matrix(q)
        a_thrust = torch.matmul(R_mat, F_body.unsqueeze(-1)).squeeze(-1) / pw.mq
        
        # (6) Drag force in global frame !!.
        vx, vy, vz = torch.unbind(v, dim=-1)
        drag = (pw.Db / pw.mq).unsqueeze(-1) * torch.stack([2*vx, 2*vy, torch.zeros_like(vx)], dim=-1)
        
        # (7) Gravity vector.
        g = torch.tensor([0, 0, pw.g0], dtype=x.dtype, device=x.device)
        
        # (8) Linear acceleration.
        f_dist = torch.stack([pw.f_dist_x, pw.f_dist_y, pw.f_dist_z], dim=-1)
        v_dot = -g + a_thrust - drag + f_dist / pw.mq
        
        # (9) Motor-induced moments.
        u1, u2, u3, u4 = torch.unbind(u, dim=-1)
        M_x = pw.Ct * pw.l * (u1 - u2 - u3 + u4)
        M_y = pw.Ct * pw.l * (- u1 - u2 + u3 + u4)
        M_z = pw.Cd * (- u1 + u2 - u3 + u4)
        M_motor = torch.stack([M_x, M_y, M_z], dim=-1) * -1.0
        
        tau_dist = torch.stack([pw.tau_dist_x, pw.tau_dist_y, pw.tau_dist_z], dim=-1)
        M = M_motor + tau_dist
        
        # (10) Rotational dynamics.
        # Since J is diagonal, we can do elementwise division.
        J_val = torch.stack([pw.Jx, pw.Jy, pw.Jz], dim=-1)
        Jr = r * J_val  # elementwise multiplication (diagonal matrix action)
        r_dot = (M - torch.cross(r, Jr, dim=-1)) / J_val  # elementwise division
        
        # (11) Concatenate derivatives.
        return torch.cat([p_dot, q_dot, v_dot, r_dot], dim=-1)
    
    @staticmethod   
    def _renorm_quat(q):
        # Renormalizes a quaternion to unit norm.
        return q / torch.norm(q, dim=-1, keepdim=True)

    @staticmethod
    def _skew_symmetric(r):
        # Builds a 4x4 skew-symmetric matrix for quaternion kinematics.
        r0, r1, r2 = torch.unbind(r, dim=-1)
        zeros = torch.zeros_like(r0)
        return torch.stack([
            torch.stack([zeros, -r0, -r1, -r2], dim=-1),
            torch.stack([r0, zeros, r2, -r1], dim=-1),
            torch.stack([r1, -r2, zeros, r0], dim=-1),
            torch.stack([r2, r1, -r0, zeros], dim=-1)
        ], dim=-2)
    
    @staticmethod
    def _quat_to_rot_matrix(q):
        # Converts quaternion [w, x, y, z] to a 3x3 rotation matrix.
        w, x_, y_, z_ = torch.unbind(q, dim=-1)
        two_s = 2.0 / (q * q).sum(dim=-1)
        return torch.stack([
            torch.stack([1 - two_s*(y_*y_ + z_*z_), two_s*(x_*y_ - w*z_), two_s*(x_*z_ + w*y_)], dim=-1),
            torch.stack([two_s*(x_*y_ + w*z_), 1 - two_s*(x_*x_ + z_*z_), two_s*(y_*z_ - w*x_)], dim=-1),
            torch.stack([two_s*(x_*z_ - w*y_), two_s*(y_*z_ + w*x_), 1 - two_s*(x_*x_ + y_*y_)], dim=-1)
        ], dim=-2)
    

    def positive_params(self):
        return self.param_wrapper.positive_params()     
    
    def free_params(self):
        return self.param_wrapper.free_params()
    
    @staticmethod
    def state_weights():
        return torch.tensor([1.0, 1.0, 10.0, # pos
                             0.0, 0.0, 0.0, 0.0, # quat
                             0.1, 0.1, 1.0, # vel
                             0.0, 0.0, 0.0])
    
    def get_params_names(self):
        return self.param_wrapper.get_params_names()
    
    @staticmethod
    def get_state_names():
        return ["x", "y", "z", # pos
                "qw", "qx", "qy", "qz", # quat
                "vx", "vy", "vz", # vel
                "wx", "wy", "wz"] # ang_vel
    
    @staticmethod
    def get_control_names():
        return ["m1", "m2", "m3", "m4"]
    
    @staticmethod
    def save_param_traj():
        return True

if __name__ == "__main__":
    
    model = DroneDynamisc()
    psize = model.count_params()
    print(f"Model has {psize} parameters")
    
    t = torch.tensor([0.0], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0, 0.0, # pos 
                       1.0, 0.0, 0.0, 0.0, # quat
                       0.0, 0.0, 0.0, # vel
                       0.0, 0.0, 0.0]], # ang_vel
                     dtype=torch.float32)
    u = torch.tensor([[20.0, 20.0, 20.0, 20.0]], dtype=torch.float32)
    
    p = model.get_default_params()
    
    x = x.repeat(1, 1)
    u = u.repeat(1, 1)
    
    x_dot = model(t, x, u, p)

    print(x.shape)
    
    
    # rk4 step  
    h = 0.01
    k1 = model(t, x, u, p)
    k2 = model(t + h/2, x + h/2 * k1, u, p)
    k3 = model(t + h/2, x + h/2 * k2, u, p)
    k4 = model(t + h, x + h * k3, u, p)
    x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    print(x)
    print(x_next)   
    print(x_next.shape)
    

    
    print(x_dot.shape)
        
        
        