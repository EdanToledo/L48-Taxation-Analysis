import chex 
import numpy as np
import collections
import random

@chex.dataclass
class Transition:
    obs_tm1 : chex.Array
    a_tm1 : chex.Array
    r_t : chex.Numeric
    discount : chex.Numeric
    obs_t : chex.Array

class ReplayBuffer(object):
  """A simple Python replay buffer."""

  def __init__(self, capacity):
    self._prev = None
    self._action = None
    self._latest = None
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, env_timestep, action):
    self._prev = self._latest
    self._action = action
    self._latest = env_timestep

    if action is not None:
      self.buffer.append(
          (self._prev.observation, self._action, self._latest.reward,
           self._latest.discount, self._latest.observation))

  def sample(self, batch_size):
    obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
        *random.sample(self.buffer, batch_size))
    return (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
            np.asarray(discount_t), np.stack(obs_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)