import mujoco
import mujoco_viewer
import numpy as np
from dataset_generators.drone_pid_controler import DroneCascadedController
import pandas as pd
import matplotlib.pyplot as plt


sim_dt =  0.001

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

    <body name="ball" pos="0 0 -0.3">
      <joint name="ball_x" type="slide" axis="1 0 0"/>
      <joint name="ball_y" type="slide" axis="0 1 0"/>
      <joint name="ball_z" type="slide" axis="0 0 1"/>
      <geom name="ball" type="sphere" size=".025" contype="0" conaffinity="0" mass="0.2" rgba="1 0 0 1"/>
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
    <spatial name="string" limited="true" range="0 0.3" width="0.006">
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
viewer = mujoco_viewer.MujocoViewer(model, data)

def get_body_vel(data):
  return data.qvel[0:3]

def get_body_rates(data):
  return data.qvel[3:6]

def get_body_pos(data): 
  return data.qpos[0:3]

def get_body_quat(data):
  return data.qpos[3:7]

ctrl = DroneCascadedController(mass=0.3+0.2, g=9.81)
  
list_of_state = []

des_att = np.zeros(3)
des_z = 1.0

def sample_des_att_and_z(scale=0.6):
  des_att = np.random.rand(3) * scale - scale/2
  des_z = np.random.rand() * 2.0 + 2.0
  return des_att, des_z

for t in range(10_00):
    if viewer.is_alive:
  
        for n in range(5):
          mujoco.mj_step(model, data)

        if t % 200 == 0:
          des_att, des_z = sample_des_att_and_z()
                
        t_set, des_vz, des_br = ctrl.calc_control(des_att,
                                  des_z,
                                  get_body_pos(data), 
                                  get_body_quat(data),
                                  get_body_vel(data),
                                  get_body_rates(data), 
                                  t)
        
        full_state = np.concatenate([get_body_pos(data),
                                     get_body_quat(data),
                                     get_body_vel(data),
                                     get_body_rates(data), 
                                     np.array([des_z]),
                                     np.array([des_vz]),
                                     des_att,
                                     des_br,
                                     t_set])
        
        list_of_state.append(full_state)
        
        data.ctrl[0:4] = np.array(t_set)
                
        viewer.render()
    else:
        break
    

# save the data
collumns = ["x", "y", "z", 
            "qw", "qx", "qy", "qz",
            "vx", "vy", "vz",
            "bx", "by", "bz",
            "ref_z", "ref_vz", 
            "ref_roll", "ref_pitch", "ref_yaw",
            "ref_roll_rate", "ref_pitch_rate", "ref_yaw_rate",
            "t1", "t2", "t3", "t4"]

df = pd.DataFrame(list_of_state, columns=collumns)
df.to_csv("drone_data_gains.csv", index=False)

plt.show()

viewer.close()