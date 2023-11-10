import pickle
import warnings
from copy import deepcopy
from typing import Any, Dict, Union

import gym
import numpy as np

from stable_baselines3.common import utils
from .running_mean_clipped_std import RunningMeanClippedStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env import VecNormalize


class VecNormalizeClippedVar(VecNormalize):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: the vectorized environment to wrap
    :param training: Whether to update or not the moving average
    :param norm_obs: Whether to normalize observation or not (default: True)
    :param norm_reward: Whether to normalize rewards or not (default: True)
    :param clip_obs: Max absolute value for observation
    :param clip_reward: Max value absolute for discounted reward
    :param gamma: discount factor
    :param epsilon: To avoid division by zero
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        var_clip: float = 0.01,
    ):
        super(VecNormalizeClippedVar, self).__init__(venv,
            training=training,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            epsilon=epsilon,
        )

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_keys = set(self.observation_space.spaces.keys())
            self.obs_spaces = self.observation_space.spaces
            self.obs_rms = {key: RunningMeanClippedStd(shape=space.shape, var_clip=var_clip) for key, space in self.obs_spaces.items()}
        else:
            self.obs_keys, self.obs_spaces = None, None
            self.obs_rms = RunningMeanClippedStd(shape=self.observation_space.shape, var_clip=var_clip)
