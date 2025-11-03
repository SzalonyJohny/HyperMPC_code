import mujoco
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import dataset_generators.sample_control as sc
import mediapy as media


class PendulumSimMujocoBacklash():

    def __init__(self, psim, dgen_settings) -> None:
        self.psim = psim
        self.dgen_settings = dgen_settings
        self.control_sampler = sc.ControlVectorSampler(dgen_settings['sample_per_second'],
                                                       dgen_settings['episode_len_s'],
                                                       dgen_settings['u_control_pt'],
                                                       dgen_settings['u_max'])
        self.model, self.data = self.create_mujoco_model()

    def create_mujoco_model(self):
        psim = self.psim
        r_vis = 0.2
        r_vis_size = 0.025
        cilinder_offset = 0.2
        backlash_rad = (psim['backlash'] ) * np.pi / 180 + cilinder_offset
        vis_angles = np.linspace(backlash_rad, 2*np.pi - backlash_rad, 20)
        offset_angle = - np.pi/2
        vis_x = np.cos(vis_angles + offset_angle) * r_vis
        vis_z = np.sin(vis_angles + offset_angle) * r_vis
        
        double_pendulum_hollow = f"""
        <mujoco>
        <option timestep="{psim['sim_time_step']}" integrator="RK4">
            <flag energy="enable" contact="enable"/>
        </option>

        <default>
            <joint type="hinge" axis="0 -1 0"/>
            <geom type="capsule" size=".02"/>
        </default>

        <worldbody>
            <light pos="0 -.4 1"/>
            <camera name="fixed" pos="0 -2 0.2" xyaxes="1 0 0 0 0 1"/>

            <body name="1" pos="0 0 0.2">   
            <joint name="second_link" damping="{psim['b']}" armature="0.0" frictionloss="{psim['f']}" />
            
            <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.04" rgba="0.5 0.5 0.5 1" />
            <geom type="sphere" pos="{vis_x[0]} 0 {vis_z[0]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[1]} 0 {vis_z[1]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[2]} 0 {vis_z[2]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[3]} 0 {vis_z[3]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[4]} 0 {vis_z[4]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[5]} 0 {vis_z[5]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[6]} 0 {vis_z[6]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[7]} 0 {vis_z[7]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[8]} 0 {vis_z[8]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[9]} 0 {vis_z[9]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[10]} 0 {vis_z[10]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[11]} 0 {vis_z[11]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[12]} 0 {vis_z[12]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[13]} 0 {vis_z[13]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[14]} 0 {vis_z[14]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[15]} 0 {vis_z[15]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[16]} 0 {vis_z[16]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[17]} 0 {vis_z[17]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[18]} 0 {vis_z[18]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            <geom type="sphere" pos="{vis_x[19]} 0 {vis_z[19]}" size="{r_vis_size}"  rgba="1 0 0 1" />
            
            
            <body name="backlash_body">
                <joint name="second_link_backlash" axis="0 -1 0" limited="true" range="-{psim['backlash']} {psim['backlash']}" armature="0.0"/>
                <geom type="cylinder" fromto="0 0 0 0 0 -{psim['l']}" mass="{psim['m']}" rgba="0 1 0 1"/>
            </body>
            
            </body>
        </worldbody>

        <actuator>
            <motor name="my_motor" joint="second_link" gear="1"/>
        </actuator>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(double_pendulum_hollow)
        data = mujoco.MjData(model)
        return model, data

    def in_contact(self):
        backlash = self.data.joint('second_link_backlash').qpos[0]
        
        if abs(backlash) > (self.psim['backlash'] ) * np.pi / 180 * 0.98:
            return True
        else:
            return False

    def render_mujoco_model(self):
        height, width = 480, 640
        with mujoco.Renderer(self.model, height, width) as renderer:
            mujoco.mj_forward(self.model, self.data)
            renderer.update_scene(self.data, camera="fixed")
            media.show_image(renderer.render())

    def init_episode(self):
        model, data = self.model, self.data
        settings = self.dgen_settings
        mujoco.mj_resetData(model, data)
        q2min, q2max = settings['q_range']
        data.joint('second_link').qpos = np.random.uniform(q2min, q2max)
        dq2min, dq2max = settings['dq_range']
        data.joint('second_link').qvel = np.random.uniform(dq2min, dq2max)
        data.ctrl[0] = 0.0

    def save_update_df_at_i(self, i, df, u):
        data = self.data
        psim = self.psim
        data.ctrl[0] = u[i]
        df.loc[i, 'u'] = u[i]
        backlash = data.joint('second_link_backlash').qpos[0]
        d_backlash = data.joint('second_link_backlash').qvel[0]
        df.loc[i, 'q'] = data.joint('second_link').qpos[0] + backlash
        df.loc[i, 'dq'] = data.joint('second_link').qvel[0] + d_backlash
        df.loc[i, 'backlash'] = backlash
        df.loc[i, 'd_backlash'] = d_backlash

    def step_sim(self, i, u=None):
        if u is not None:
            self.data.ctrl[0] = u

        while self.data.time * self.dgen_settings['sample_per_second'] < i:
            mujoco.mj_step(self.model, self.data)
    

    def generate_episode(self):
        dgen_settings = self.dgen_settings
        model = self.model
        data = self.data

        n_frames = int(dgen_settings['episode_len_s'] *
                       dgen_settings['sample_per_second'])

        t = np.linspace(0, dgen_settings['episode_len_s'], n_frames)

        u = self.control_sampler.sample() * 2.0
        u = np.clip(u, -dgen_settings['u_max'], dgen_settings['u_max'])

        df = pd.DataFrame(np.zeros((n_frames, 6)),
                          columns=['t', 'u', 'q', 'dq', 'backlash', 'd_backlash'])

        self.init_episode()

        if dgen_settings['render']:
            frames = []
            with mujoco.Renderer(model, dgen_settings['render_height'],
                                 dgen_settings['render_width']) as renderer:
                for i in range(n_frames):
                    self.step_sim(i)
                    self.save_update_df_at_i(i, df, u)
                    renderer.update_scene(data, "fixed")
                    frames.append(renderer.render())
                media.show_video(
                    frames, fps=dgen_settings['sample_per_second'])

        else:
            for i in range(n_frames):
                self.step_sim(i)
                self.save_update_df_at_i(i, df, u)

        df['t'] = t
        return df

    def init_sim_with_render(self):
        self.init_episode()
        self.i = 0
        self.frames = []

    def step_sim_with_render(self, u, render=False):
        dgen_settings = self.dgen_settings
        data = self.data

        if render:
            with mujoco.Renderer(self.model, dgen_settings['render_height'],
                                 dgen_settings['render_width']) as renderer:
                renderer.update_scene(data, "fixed")
                self.frames.append(renderer.render())

        self.step_sim(self.i, u)
        backlash = data.joint('second_link_backlash').qpos[0]
        d_backlash = data.joint('second_link_backlash').qvel[0]
        q = self.data.joint('second_link').qpos[0]
        dq = self.data.joint('second_link').qvel[0]
        x = np.array([q + backlash, dq + d_backlash, u[0]])
        self.i += 1
        return x

    def set_state(self, x):
        data = self.data
        data.joint('second_link').qpos[0] = x[0]
        data.joint('second_link').qvel[0] = x[1]
        data.joint('second_link_backlash').qpos[0] = 0.0
        data.joint('second_link_backlash').qvel[0] = 0.0
        data.ctrl[0] = 0.0
        

    def save_render_video(self, file_name=""):
        if len(self.frames) == 0:
            print("No frames to play")
            return

        import cv2
        height, width, layers = self.frames[0].shape
        frame_size = (width, height)

        out_path = f'pendulum_sim_{file_name}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(out_path, fourcc,
                                       self.dgen_settings['sample_per_second'],
                                       frame_size)

        # Write each frame to the video file
        for frame in self.frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        video_writer.release()

    