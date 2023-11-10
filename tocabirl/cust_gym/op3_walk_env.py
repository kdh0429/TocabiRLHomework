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


class Op3Env(gym.envs.mujoco.MujocoEnv):
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
        self.left_foot_id = []
        self.right_foot_id = []
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

        self.init_q_desired[20] = 1.5
        self.init_q_desired[25] = -1.5

        self.set_state(self.init_q_desired, self.init_qvel,)    

        self.target_position = np.copy(self.init_q_desired[0:3])

        self.nominal_body_mass = np.copy(self.model.body_mass)
        self.nominal_body_inertia = np.copy(self.model.body_inertia)
        self.nominal_body_ipos = np.copy(self.model.body_ipos)
        self.nominal_dof_damping = np.copy(self.model.dof_damping)
        self.nominal_dof_frictionloss = np.copy(self.model.dof_frictionloss)

        motor_constant_scale = np.random.uniform(0.95, 1.05, 6)
        self.motor_constant_scale = np.tile(motor_constant_scale, 2)

        # Deep Mimic
        self.init_mocap_data_idx = 0
        self.mocap_data_idx = 0
        self.mocap_data = np.genfromtxt('motions/processed_data_tocabi_walk.txt', encoding='ascii')
        self.mocap_data_num = len(self.mocap_data) - 1
        self.mocap_cycle_dt = 0.0005

        self.qpos_noise = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_noise = np.zeros_like(self.sim.data.qvel[6:])
        self.qvel_lpf = np.zeros_like(self.sim.data.qvel[6:])
        self.qpos_pre = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_pre = np.zeros_like(self.sim.data.qvel[6:])

        self.target_vel = np.array([np.random.uniform(0.0, 0.5), 0.0])        
        self.start_target_vel = np.array([np.random.uniform(0.0, 0.5), 0.0])
        self.final_target_vel = np.array([np.random.uniform(0.0, 0.5), 0.0])
        self.vel_change_duration = 0
        self.cur_vel_change_duration = 0
        
        self.ft_left_foot = np.zeros(3)
        self.ft_left_foot_pre = np.zeros(3)
        self.ft_right_foot = np.zeros(3)
        self.ft_right_foot_pre = np.zeros(3)
        self.torque_left_foot = np.zeros(3)
        self.torque_right_foot = np.zeros(3)
        for sensor_name, id in self.sim.model._sensor_name2id.items():
            if sensor_name == "LF_Force_sensor":
                    self.ft_left_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "LF_Torque_sensor":
                    self.torque_left_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "RF_Force_sensor":
                    self.ft_right_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "RF_Torque_sensor":
                    self.torque_right_foot_adr = self.sim.model.sensor_adr[id]
        
        self.action_log = []
        self.data_log = []
        self.action_delay = 5

        self.num_obs_hist = 5
        self.num_obs_skip = 2
        self.obs_buf = []
        self.action_buf = []

        self.q_bias = np.random.uniform(-0.034, 0.034, 12)
        self.quat_bias = np.random.uniform(-3.14/300.0, 3.14/300.0, 3)
        self.ft_bias = np.random.uniform(-100.0, 100.0, 2)
        self.ft_mx_bias = np.random.uniform(-10.0, 10.0, 2)
        self.ft_my_bias = np.random.uniform(-30.0, 30.0, 2)

        # Perturbation
        self.perturbation_on = False
        self.new_xfrc = np.zeros_like(self.sim.data.xfrc_applied)
        self.pert_duration = 0
        self.magnitude = 0
        self.cur_pert_duration = 0
        self.perturb_timing = 0

        self.obs_mean = np.genfromtxt('data/obs_mean_fixed.txt', encoding='ascii')
        self.obs_var = np.genfromtxt('data/obs_variance_fixed.txt', encoding='ascii')


        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.custom_action_space = True
        self._set_action_space()

        action = self.action_space.sample()
        self.action_last = self.action_space.sample()[0:-1]
        self.action_cur = np.copy(self.action_last)
        self.action_raw = self.action_space.sample()

        _, self.actuator_high = self.model.actuator_ctrlrange.copy().T
        self.action_high = np.concatenate([self.actuator_high[0:12], [self.dt]])

        if self.deep_mimic_env:
            action = np.zeros_like(action)
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        if self.custom_action_space:
            # bounds = np.array([[0.0, 0.10], # Phase
            #                     [-0.10, 0.10],[-0.10, 0.10],[-0.10, 0.10],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3], # Left Foot
            #                     [-0.10, 0.10],[-0.10, 0.10],[-0.10, 0.10],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3],[-3.14/3, 3.14/3]]) # Right Foot
            # bounds = np.concatenate([self.model.actuator_ctrlrange.copy(), [[-1.0, 1.0]]])
            bounds = self.model.actuator_ctrlrange.copy()[0:12]
            bounds[:] = [-1.0, 1.0]
            bounds = np.concatenate([bounds, [[0.0, 1.0]]])
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            bounds = self.model.actuator_ctrlrange.copy()
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space