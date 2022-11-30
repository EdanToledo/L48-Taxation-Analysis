import chex 
import numpy as np
import collections
import random
from typing import Any

import jax 
import jax.numpy as jnp 

def stack_trees(trees) -> Any:
    """_description_"""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)

@chex.dataclass
class Transition:
    obs_tm1 : Any 
    a_tm1 : Any
    r_t : Any
    discount : Any
    obs_t : Any

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
    
    return (stack_trees(obs_tm1), stack_trees(a_tm1), stack_trees(r_t),
            stack_trees(discount_t), stack_trees(obs_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)