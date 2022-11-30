from typing import Any, Dict, Optional
from ai_economist.foundation.env_wrapper import recursive_obs_dict_to_spaces_dict
import dm_env
from dm_env import specs
import gym
import tree
from gym import spaces
import numpy as np


class GymWrapper(dm_env.Environment):
  """Environment wrapper for OpenAI Gym environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self, environment: gym.Env):

    self._environment = environment
    self._reset_next_step = True
    self._last_info = None

    # Convert action and observation specs.
    # Add observation space to the env
    # --------------------------------
    # Note: when the collated agent "a" is present, add obs keys
    # for each individual agent to the env
    # and remove the collated agent "a" from the observation
    obs = self.obs_at_reset()
    obs_space = spaces.Dict(recursive_obs_dict_to_spaces_dict(obs))

    # Add action space to the env
    # ---------------------------
    act_space = {}
    for agent_id in range(len(self._environment.world.agents)):
        if self._environment.world.agents[agent_id].multi_action_mode:
            act_space[str([agent_id])] = spaces.MultiDiscrete(
                self._environment.get_agent(str(agent_id)).action_spaces
            )
        else:
            act_space[str(agent_id)] = spaces.Discrete(
                self._environment.get_agent(str(agent_id)).action_spaces
            )
        act_space[str(agent_id)].dtype = np.int32

    if self._environment.world.planner.multi_action_mode:
        act_space["p"] = spaces.MultiDiscrete(
            self._environment.get_agent("p").action_spaces
        )
    else:
        act_space["p"] = spaces.Discrete(self._environment.get_agent("p").action_spaces)
    act_space["p"].dtype = np.int32

    # Ensure the observation and action spaces share the same keys
    assert set(obs_space.keys()) == set(
        act_space.keys()
    )
    act_space = spaces.Dict(act_space)

    self._observation_spec = gym_convert_to_spec(obs_space, name='observation')
    self._action_spec = gym_convert_to_spec(act_space, name='action')

  def obs_at_reset(self):
        """
        Calls the (Python) env to reset and return the initial state
        """
        obs = self._environment.reset()
        obs = self._reformat_obs(obs)
        return obs

  def _reformat_obs(self, obs):
      if "a" in obs:
          # This means the env uses collated obs.
          # Set each individual agent as obs keys for processing with WarpDrive.
          for agent_id in range(self._environment.n_agents):
              obs[str(agent_id)] = {}
              for key in obs["a"].keys():
                  obs[str(agent_id)][key] = obs["a"][key][..., agent_id]
          del obs["a"]  # remove the key "a"
      return obs

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    observation = self._environment.reset()
    
    # Reset the diagnostic information.
    return dm_env.restart(observation)

  def step(self, action) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)
    self._reset_next_step = done
    self._last_info = info

    
    if done:
        return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def get_info(self) -> Optional[Dict[str, Any]]:
    """Returns the last info returned from env.step(action).
    Returns:
      info: dictionary of diagnostic information from the last environment step
    """
    return self._last_info

  @property
  def environment(self) -> gym.Env:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
          "attempted to get missing private attribute '{}'".format(name))
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def gym_convert_to_spec(space: gym.Space,
                     name: Optional[str] = None):
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.
  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).
  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(gym_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: gym_convert_to_spec(value, key)
        for key, value in space.spaces.items()
    }

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))

