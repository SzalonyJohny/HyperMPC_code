import mujoco
import numpy as np
from dataset_generators.drone_pid_controler import DroneCascadedController
import pandas as pd
import matplotlib.pyplot as plt
import dataset_generators.sample_control as sc


class DroneSim():
    def __init__(self, cfg) -> None:
        
        cfg = type('Config', (object,), cfg)()
        self.cfg = cfg
        
        self.control_sampler = sc.ControlVectorSampler(cfg.freq,
                                                       cfg.episode_len_s,
                                                       cfg.u_control_pt,
                                                       1.0)    
        self.model, self.data = self.create_mujoco_model()    
        
        # drone controller
        self.controller = DroneCascadedController(mass=0.321+cfg.ball_mass,
                                                  g=9.81)
        
        self.n_steps = int(self.cfg.episode_len_s * self.cfg.freq)
        
        self.collumns = ["x", "y", "z", 
            "qw", "qx", "qy", "qz",
            "vx", "vy", "vz",
            "bx", "by", "bz",
            "ref_z", "ref_vz", 
            "ref_roll", "ref_pitch", "ref_yaw",
            "ref_roll_rate", "ref_pitch_rate", "ref_yaw_rate",
            "t1", "t2", "t3", "t4"]
    
    def get_config(self):
        return {"mass": self.cfg.ball_mass + 0.321,
                "rope_lenght": self.cfg.rope_lenght,
                }
        

    def create_mujoco_model(self):
        sim_dt =  (1 / self.cfg.freq) / self.cfg.mujoco_sub_steps
        print(f"sim_dt: {sim_dt}")
        l = self.cfg.rope_lenght
        ball_init_l = l + 0.1
        scene_xml = f"""
        <mujoco model="Skydio X2 scene">
    
            <option timestep="{sim_dt}" density="1.225" viscosity="1.8e-5" integrator="RK4"/>
            <statistic center="0 0 0.1" extent="0.6" meansize=".05"/>
            <compiler autolimits="true" assetdir="dataset_generators/assets"/>

            <default>
                <default class="x2">
                <geom mass="0"/>
                <motor ctrlrange="0 13"/>
                <mesh scale="0.01 0.01 0.01"/>
                <default class="visual">
                    <geom group="2" type="mesh" contype="0" conaffinity="0"/>
                </default>
                <default class="collision">
                    <geom group="3" type="box"/>
                    <default class="rotor">
                    <geom type="ellipsoid" size=".13 .13 .01"/>
                    </default>
                </default>
                <site group="5"/>
                </default>
            </default>

            <asset>
                <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
                <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
                <material name="invisible" rgba="0 0 0 0"/>

                <mesh class="x2" file="X2_lowpoly.obj"/>
            </asset>

            <worldbody>
                <light name="spotlight" mode="targetbodycom" target="x2" pos="0 -1 2"/>
                <body name="x2" pos="0 0 0.1" childclass="x2">
                <freejoint/>
                <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
                <site name="imu" pos="0 0 0"/>
                <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"/>
                <geom class="collision" size=".06 .027 .02" pos=".04 0 .02"/>
                <geom class="collision" size=".06 .027 .02" pos=".04 0 .06"/>
                <geom class="collision" size=".05 .027 .02" pos="-.07 0 .065"/>
                <geom class="collision" size=".023 .017 .01" pos="-.137 .008 .065" quat="1 0 0 1"/>
                <geom name="rotor1" class="rotor" pos="-.14 -.18 .05" mass=".25"/>
                <geom name="rotor2" class="rotor" pos="-.14 .18 .05" mass=".25"/>
                <geom name="rotor3" class="rotor" pos=".14 .18 .08" mass=".25"/>
                <geom name="rotor4" class="rotor" pos=".14 -.18 .08" mass=".25"/>
                <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325" class="visual" material="invisible"/>
                <site name="thrust1" pos="-.14 -.18 .05"/>  
                <site name="thrust2" pos="-.14 .18 .05"/>
                <site name="thrust3" pos=".14 .18 .08"/>
                <site name="thrust4" pos=".14 -.18 .08"/>
                </body>

                <body name="ball" pos="0 0 {-ball_init_l}" childclass="x2">
                    <joint name="ball_x" type="slide" axis="1 0 0"/>
                    <joint name="ball_y" type="slide" axis="0 1 0"/>
                    <joint name="ball_z" type="slide" axis="0 0 1"/>
                    <geom name="ball" type="sphere" size=".025" contype="0" conaffinity="0" mass="{self.cfg.ball_mass}" rgba="1 0 0 1"/>
                    <site name="ball" size=".005"/>
                </body>

            </worldbody>

            <actuator>
                <motor class="x2" name="thrust1" site="thrust1" gear="0 0 1 0 0  0.201"/>
                <motor class="x2" name="thrust2" site="thrust2" gear="0 0 1 0 0 -0.201"/>
                <motor class="x2" name="thrust3" site="thrust3" gear="0 0 1 0 0  0.201"/>
                <motor class="x2" name="thrust4" site="thrust4" gear="0 0 1 0 0 -0.201"/>
            </actuator>

            <tendon>
                <spatial name="string" limited="true" range="0 {l}" width="0.006">
                <site site="imu"/>
                <site site="ball"/>
                </spatial>
            </tendon>

            <sensor>
                <gyro name="body_gyro" site="imu"/>
                <accelerometer name="body_linacc" site="imu"/>
                <framequat name="body_quat" objtype="site" objname="imu"/>
            </sensor>

            <keyframe>
                <key name="hover" qpos="0 0 .3 1 0 0 0 0 0 0" ctrl="3.2495625 3.2495625 3.2495625 3.2495625"/>
            </keyframe>


            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
                <rgba haze="0.15 0.25 0.35 1"/>
                <global azimuth="-20" elevation="-20" ellipsoidinertia="true"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                markrgb="0.8 0.8 0.8" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>

            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
                <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
            </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)  
        return model, data
    
    @staticmethod
    def calculate_forces(data, model, imu_site="imu", ball_site="ball"):
        """
        Calculate the forces in x, y, z directions based on positions of imu and ball sites.

        Parameters:
            data: Mujoco data object.
            model: Mujoco model object.
            imu_site (str): Name of the IMU site.
            ball_site (str): Name of the ball site.

        Returns:
            np.ndarray: Array of forces in [force_x, force_y, force_z].
        """
        force = data.efc_force

        if force.size == 0:
            force_magnitude = 0.0
        else:
            force_magnitude = force.item()

        imu_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site)]
        ball_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ball_site)]

        direction_vec = ball_pos - imu_pos
        norm = np.linalg.norm(direction_vec)

        if norm == 0:
            return np.zeros(3)
            # raise ValueError("IMU and Ball sites positions cannot be identical.")

        direction_vec /= norm

        force_xyz = force_magnitude * direction_vec

        return force_xyz

    def get_forces(self):
        """
        Get the forces in x, y, z directions based on positions of imu and ball sites.

        Returns:
            np.ndarray: Array of forces in [force_x, force_y, force_z].
        """
        return self.calculate_forces(self.data, self.model)

    def sample_z(self):
        return self.control_sampler.sample() * 2.0 + 4.5
    
    def sample_att(self):
        # roll pich yaw
        return np.stack([
            self.control_sampler.sample() * self.cfg.max_deflaction,
            self.control_sampler.sample() * self.cfg.max_deflaction,
            self.control_sampler.sample() * np.pi
        ])
    
    @staticmethod
    def _get_vel(data):
        return data.qvel[0:3]

    @staticmethod
    def _get_body_rates(data):
        return data.qvel[3:6]
    
    @staticmethod
    def _get_body_pos(data): 
        return data.qpos[0:3]
    
    @staticmethod
    def _get_body_quat(data):
        return data.qpos[3:7]
    
    def init_mpc_simulation(self):
        mujoco.mj_resetData(self.model, self.data)
        # self.data.qpos[0:3] = np.array([0, 0, 2.0])
        # self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        # self.data.qvel[0:3] = np.array([0, 0, 0])
        # self.data.qvel[3:6] = np.array([0, 0, 0])
        # self.data.ctrl[:4] = np.array([4, 4, 4, 4])
        
        if self.cfg.render:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
    def close_render(self):
        if self.cfg.render:
            assert self.viewer.is_alive
            self.viewer.close()
        self.viewer = None    
    
    def generate_episode(self):
        self.controller.sample_gains()
        ref_z = self.sample_z()
        ref_att = self.sample_att().transpose()
        
        print(f"shapes {ref_z.shape}, {ref_att.shape}")
        
        init_data = np.zeros((self.n_steps, len(self.collumns)))
        df = pd.DataFrame(init_data, columns=self.collumns)
        
        # init episode
        mujoco.mj_resetData(self.model, self.data)
        
        if self.cfg.render:
            import mujoco_viewer
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        for i in range(self.n_steps):

            for _ in range(self.cfg.mujoco_sub_steps):
                mujoco.mj_step(self.model, self.data)
                        
            # get state
            pos = self._get_body_pos(self.data)
            quat = self._get_body_quat(self.data)
            vel = self._get_vel(self.data)
            rates = self._get_body_rates(self.data)
            
            # get control
            if i > 100:
               u, des_vz, des_br = self.controller.calc_control(ref_att[i],
                                                                ref_z[i],
                                                                pos, quat, vel, rates)
            else:
                u, des_vz, des_br = self.controller.calc_control(np.zeros(3),
                                                                 np.array(2.0),
                                                                 pos, quat, vel, rates)
            self.data.ctrl[:4] = u     
            
            full_state = np.concatenate([pos,
                                        quat,
                                        vel,
                                        rates, 
                                        np.array([ref_z[i]]),
                                        np.array([des_vz]),
                                        np.array(ref_att[i]),
                                        np.array(des_br),
                                        u])
            df.loc[i] = full_state
    
            if self.cfg.render:
                assert viewer.is_alive
                viewer.render()
                
        df['t'] = np.linspace(0, self.cfg.episode_len_s, self.n_steps)
        if self.cfg.render:
            viewer.close()
        return df
    
    def set_state(self, x):
        assert x.shape == (13,)
        # x = [pos, quat, vel, rates]
        self.data.qpos[0:3] = x[0:3]
        self.data.qpos[3:7] = x[3:7]
        self.data.qvel[0:3] = x[7:10]
        self.data.qvel[3:6] = x[10:13]
        
    def step_sim(self, u):
        assert u.shape == (4,)

        self.data.ctrl[:4] = u

        for _ in range(self.cfg.mujoco_sub_steps):
            mujoco.mj_step(self.model, self.data)
        
        if self.cfg.render:
            assert self.viewer.is_alive
            self.viewer.render()            
        
        # get state
        pos = self._get_body_pos(self.data)
        quat = self._get_body_quat(self.data)
        vel = self._get_vel(self.data)
        rates = self._get_body_rates(self.data)
        
        return np.concatenate([pos, quat, vel, rates])
                       
    
    def validate_episode(df):
        return np.all(df['z'][200:] > 0.1)         
        
        
if __name__ == "__main__":
    
    cfg = {
        'freq': 100,
        'episode_len_s': 20,
        'u_control_pt': 25,
        'max_deflaction': 0.5,
        'mujoco_sub_steps': 10,
        'ball_mass': 0.5,
        'rope_lenght' : 1.0,
        'render': True,
        'episode_count': 300
    }
        
    sim = DroneSim(cfg)
    
    # df = sim.generate_episode()
    
    for i in range(10):
        df = sim.generate_episode()
        plt.figure(figsize=(10, 5))
        plt.plot(df['t'], df['bx'], label='x')
        plt.plot(df['t'], df['by'], label='y')
        plt.plot(df['t'], df['bz'], label='z')
        plt.plot(df['t'], df['ref_roll_rate'], label='roll_ref')
        plt.plot(df['t'], df['ref_pitch_rate'], label='pitch_ref')
        plt.plot(df['t'], df['ref_yaw_rate'], label='yaw_ref')
        plt.legend()
        
        plt.show()
        
        
    plt.figure(figsize=(10, 5))
    plt.plot(df['t'], df['z'], label='y')
    plt.plot(df['t'], df['ref_z'], label='y_ref')
    plt.plot(df['t'], df['ref_vz'], label='y_ref_vel')
    plt.plot(df['t'], df['vz'], label='roll_ref')
    plt.legend()
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['x'], df['y'], label='x-y')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.figure(figsize=(10, 5))
    # plot ref att
    plt.plot(df['t'], df['ref_roll'], label='roll')
    plt.plot(df['t'], df['ref_pitch'], label='pitch')
    plt.plot(df['t'], df['ref_yaw'], label='yaw')
    
    plt.show()
    
    
    pass
            