from collections import OrderedDict
import os
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

import random
import time

from numpy.core.arrayprint import format_float_scientific

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class TocabiEnv(gym.envs.mujoco.MujocoEnv):
    """Superclass for all MuJoCo environments.
    """
    def __init__(self, model_path, frame_skip):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Custum variable
        self.done_init = False
        self.epi_len = 0
        self.epi_reward = 0
        self.ground_collision_check_id = []
        self.self_collision_check_id = []
        self.ground_id = []
        self.deep_mimic_env = True
        self.time = 0.0
       
        self.init_q_desired = np.copy(self.data.qpos)
        self.init_q_desired[7:] = 0.0
        self.init_q_desired[9] = -0.24
        self.init_q_desired[10] = 0.6
        self.init_q_desired[11] = -0.36
        self.init_q_desired[15] = -0.24
        self.init_q_desired[16] = 0.6
        self.init_q_desired[17] = -0.36
        self.init_q_desired[22] = 0.3
        self.init_q_desired[23] = 0.3
        self.init_q_desired[24] = 1.5
        self.init_q_desired[25] = -1.27
        self.init_q_desired[26] = -1.0
        self.init_q_desired[28] = -1.0
        self.init_q_desired[32] = -0.3
        self.init_q_desired[33] = -0.3
        self.init_q_desired[34] = -1.5
        self.init_q_desired[35] = 1.27
        self.init_q_desired[36] = 1.0
        self.init_q_desired[38] = 1.0
        self.set_state(self.init_q_desired, self.init_qvel,)    

        self.target_position = np.copy(self.init_q_desired[0:3])

        self.nominal_body_mass = np.copy(self.model.body_mass)
        self.nominal_body_inertia = np.copy(self.model.body_inertia)
        self.nominal_body_ipos = np.copy(self.model.body_ipos)
        self.nominal_dof_damping = np.copy(self.model.dof_damping)
        self.nominal_dof_frictionloss = np.copy(self.model.dof_frictionloss)

        self.init_mocap_data_idx = 0
        self.mocap_data_idx = 0
        self.mocap_data_num = 340
        self.mocap_cycle_dt = 0.025

        self.qpos_noise = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_noise = np.zeros_like(self.sim.data.qvel[6:])
        self.qpos_pre = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_lpf = np.zeros_like(self.sim.data.qvel[6:])

        self.action_log = []
        self.data_log = []

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.custom_action_space = False
        self._set_action_space()

        action = self.action_space.sample()

        if self.deep_mimic_env:
            action = np.zeros_like(action)
        self.action_pre = np.copy(action)
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        if self.custom_action_space:
            bounds = np.array([[0.0, 0.10], # Phase
                                [-0.10, 0.10],[-0.10, 0.10],[-0.10, 0.10],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3], # Left Foot
                                [-0.10, 0.10],[-0.10, 0.10],[-0.10, 0.10],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3]]) # Right Foot
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            bounds = self.model.actuator_ctrlrange.copy()
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space