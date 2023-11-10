from typing import Tuple

import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd

class RunningMeanClippedStd(RunningMeanStd):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), var_clip: float = 0.0027):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """    
        super(RunningMeanClippedStd, self).__init__(
            epsilon,
            shape,
        )

        self.var_clip = var_clip

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = np.clip(new_var, self.var_clip, 10000.0)
        self.count = new_count
