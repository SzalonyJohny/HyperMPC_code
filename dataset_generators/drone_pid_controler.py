import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PidData:
    kp: float
    ki: float
    kd: float
    i: float = 0.0
    last_error: float = 0.0

class DroneCascadedController:
    def __init__(self, mass=1.0, g=9.81):
                
        self.rng = np.random.default_rng(12345)
        self.mass = mass
        self.g = g
        self.sample_gains()

    def sample_gains(self):
        def sample_uniform(a, b):
            return self.rng.random() * (b - a) + a
        
        kp_at = sample_uniform(1.5, 6.0)
        kp_br = 0.7
        # print(f"kp_at: {kp_at}, kp_br: {kp_br}")
        
        self.att_pid = PidData(kp=kp_at, ki=0.0, kd=0.01)
        self.body_rate_pid = PidData(kp=kp_br, ki=0.0, kd=0.0)
                
        kp_z = sample_uniform(3.0, 15.0)
        kp_vz = sample_uniform(3.0, 15.0)
        
        # print(f"kp_z: {kp_z}, kp_vz: {kp_vz}")
        self.alt_pos_pid = PidData(kp=kp_z, ki=0.0, kd=0.01)
        self.alt_vel_pid = PidData(kp=kp_vz, ki=0.0, kd=0.01)

    def body_rate_control(self, pid_data: PidData, body_rate_error: np.ndarray) -> np.ndarray:
        # Use PID to compute moment commands for body rates (roll, pitch, yaw)
        return self.pid_control(pid_data, body_rate_error)

    def attitude_control(self, pid_data: PidData, att_error: np.ndarray) -> np.ndarray:
        # Use PID to convert attitude error to a desired body rate command
        return self.pid_control(pid_data, att_error)

    def vel_control(self, pid_data: PidData, vel_error: np.ndarray) -> np.ndarray:
        # Use PID to compute an "attitude" command from velocity error
        return self.pid_control(pid_data, vel_error)
    
    def pos_control(self, pid_data: PidData, pos_error: np.ndarray) -> np.ndarray:
        # Use PID to compute a desired velocity command from position error
        return self.pid_control(pid_data, pos_error)
    
    @staticmethod
    def pid_control(pid_data: PidData, error: np.ndarray) -> np.ndarray:
        # Simple PID update
        pid_data.i += error
        d_error = error - pid_data.last_error
        pid_data.last_error = error
        return pid_data.kp * error + pid_data.ki * pid_data.i + pid_data.kd * d_error
    
    def body_rate_to_thrusts(self, control_input: np.ndarray) -> np.ndarray:
        """
        Convert collective thrust and moment commands into individual rotor thrusts.
        control_input: [collective thrust, roll moment, pitch moment, yaw moment]
        A simple mixer for a quadrotor (assumed X-configuration) is used.
        """
        thrust_cmd = control_input[0]
        roll_cmd = control_input[1]
        pitch_cmd = control_input[2]
        yaw_cmd = control_input[3]
        L_arm = 0.1  # assumed arm length (meters)
        
        # Mixer equations (the signs and scaling factors are for illustration)
        F1 = thrust_cmd/4 + roll_cmd/(2*L_arm) - pitch_cmd/(2*L_arm) - yaw_cmd/4
        F2 = thrust_cmd/4 - roll_cmd/(2*L_arm) - pitch_cmd/(2*L_arm) + yaw_cmd/4
        F3 = thrust_cmd/4 - roll_cmd/(2*L_arm) + pitch_cmd/(2*L_arm) - yaw_cmd/4
        F4 = thrust_cmd/4 + roll_cmd/(2*L_arm) + pitch_cmd/(2*L_arm) + yaw_cmd/4
        
        return np.array([F1, F2, F3, F4])
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].
        """
        w, x, y, z = q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.pi/2 * np.sign(sinp)
        else:
            pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])
    
    def calc_control(self, desired_att: np.ndarray, des_z: float, current_pos: np.ndarray, current_quat: np.ndarray,
                     current_vel: np.ndarray, current_body_rates: np.ndarray, 
                    ) -> np.ndarray:
        """
        Compute the rotor thrust commands using cascaded control loops.
        Parameters:
          pos_cmd: Desired position [x, y, z]
          current_pos: Current position [x, y, z]
          current_quat: Current attitude quaternion [w, x, y, z]
          current_vel: Current velocity [vx, vy, vz]
          current_body_rates: Current body rates [p, q, r]
        Returns:
          Rotor thrusts as an array of 4 values.
        """
        
        current_euler = self.quaternion_to_euler(current_quat)
        
        # # ----- Attitude Control -----
        error_att = desired_att - current_euler
        desired_body_rates = self.attitude_control(self.att_pid, error_att)
        
        desired_body_rates = np.clip(desired_body_rates, -5.0, 5.0)  # Limit body rates
        
        # ----- Body Rate Control -----
        error_body_rates = desired_body_rates - current_body_rates
        moment_cmd = self.body_rate_control(self.body_rate_pid, error_body_rates) * -1.0
        
        # ----- Altitude Control (z-axis) -----
        error_alt = des_z - current_pos[2]
        desired_alt_vel = self.pos_control(self.alt_pos_pid, np.array([error_alt]))[0]
        desired_alt_vel = np.clip(desired_alt_vel, -10.0, 10.0)  # Limit vertical velocity
        error_alt_vel = desired_alt_vel - current_vel[2]
        alt_control_output = self.vel_control(self.alt_vel_pid, np.array([error_alt_vel]))[0]
        
        # Compute collective thrust: use mass*(g + vertical acceleration command)
        thrust_cmd = self.mass * (self.g + alt_control_output)
        
        # ----- Mixer -----
        control_input = np.array([thrust_cmd, moment_cmd[0], moment_cmd[1], moment_cmd[2]])
        rotor_thrusts = self.body_rate_to_thrusts(control_input)
        rotor_thrusts = np.clip(rotor_thrusts, 0.0, 13.0) 
        return rotor_thrusts, desired_alt_vel, desired_body_rates

# Example usage:
if __name__ == "__main__":
    # Create a controller instance with default mass and gravity.
    controller = DroneCascadedController(mass=1.0, g=9.81)
    
    # Define a desired position command and a sample current state.
    pos_cmd = np.array([1.0, 2.0, 3.0])         # Desired position (meters)
    current_pos = np.array([0.5, 1.5, 2.5])       # Current position (meters)
    current_quat = np.array([1.0, 0.0, 0.0, 0.0]) # Identity quaternion (no rotation)
    current_vel = np.array([0.0, 0.0, 0.0])       # Zero velocity
    current_body_rates = np.array([0.0, 0.0, 0.0])# Zero body rates

    rotor_thrusts = controller.calc_control(pos_cmd, current_pos, current_quat, current_vel, current_body_rates)
    print("Rotor thrust commands:", rotor_thrusts)
