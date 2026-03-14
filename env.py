import gymnasium as gym
import mujoco
from gymnasium import spaces
import numpy as np
import mujoco.viewer

pendulum_xml = """
<mujoco>
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .9 .8 1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
      <geom name="cart_geom" type="box" size="0.2 0.1 0.05" rgba="0.7 0.3 0.3 1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.05" rgba="0 0.3 0.7 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" name="slide_motor" gear="10"/>
  </actuator>
</mujoco>
"""
class InvertedPendulumEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 25}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(pendulum_xml)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape =(1,), dtype=np.float32)
        high = np.array([2.0, np.pi, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,high= high, dtype=np.float32)

        self.viewer = None
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel().astype(np.float32)
    
    def reset(self, seed= None, options = None):
        super().reset(seed = seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[1]+= 0.075

        mujoco.mj_forward(self.model, self.data)
        
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), {}
    
    def step(self,action):
        action = np.clip(action, -1.0,1.0)
        self.data.ctrl[0]= action
        for _ in range(20): 
         mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        cart_pos = obs[0]
        pole_angle = obs[1]

        reward = 1.0
        reward += np.exp(-10.0*(pole_angle**2)) 
        reward += 0.5*np.exp(-1.0*(cart_pos**2)) 

        terminated = bool(cart_pos<-2.0 or cart_pos>2.0 or pole_angle<-0.2 or pole_angle>0.2)
        truncated = False

        if self.render_mode == 'human':
            self.render()
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

            import time
            time.sleep(0.01)
    def close(self):
        if self.viewer:
            self.viewer.close()
    
#print(InvertedPendulumEnv())
#print(getattr(InvertedPendulumEnv(), "_max_episode_steps", None))
#print(getattr(getattr(InvertedPendulumEnv(), "spec", None), "max_episode_steps", None))

