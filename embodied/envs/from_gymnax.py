import functools
from pathlib import Path

import chex
import gym
import jax
import numpy as np

import embodied
from embodied.envs.rocksample import RockSample
from embodied.envs.pocman import PocMan
from embodied.envs.battleship import Battleship

from definitions import ROOT_DIR


def get_gymnax_env(env_name: str,
                   rand_key: chex.PRNGKey):
  if env_name.startswith('battleship'):
    rows = cols = 10
    ship_lengths = (5, 4, 3, 2)
    if env_name == 'battleship_5':
      rows = cols = 5
      ship_lengths = (3, 2)
    elif env_name == 'battleship_3':
      rows = cols = 3
      ship_lengths = (2, )

    env = Battleship(rows=rows, cols=cols, ship_lengths=ship_lengths)
    env_params = env.default_params

  elif env_name == 'pocman':
    env = PocMan()
    env_params = env.default_params

  elif 'rocksample' in env_name:  # [rocksample, rocksample_15_15]

    if len(env_name.split('_')) > 1:
      config_path = Path(ROOT_DIR, 'embodied', 'envs', 'env_configs', f'{env_name}_config.json')
      env = RockSample(rand_key, config_path=config_path)
    else:
      env = RockSample(rand_key)
    env_params = env.default_params
  else:
    raise NotImplementedError
  return env, env_params


class FromGymnax(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action',
               seed: int = 2024, **kwargs):

    rng = jax.random.PRNGKey(seed)
    self._rng, env_rng = jax.random.split(rng)

    if isinstance(env, str):
      self._env, self._env_params = get_gymnax_env(env, rand_key=env_rng)
    else:
      raise NotImplementedError
    self._state = None
    # TODO: spaces here?
    self._obs_dict = hasattr(self._env.observation_space(self._env_params), 'spaces')
    self._act_dict = hasattr(self._env.action_space(self._env_params), 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space(self._env_params).spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space(self._env_params)}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space(self._env_params).spaces)
    else:
      spaces = {self._act_key: self._env.action_space(self._env_params)}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    self._rng, env_rng = jax.random.split(self._rng)
    if action['reset'] or self._done:
      self._done = False
      obs, self._state = self._env.reset(env_rng, self._env_params)
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._info = self._env.step(env_rng, self._state, action)
    return self._obs(
      obs, reward,
      is_last=bool(self._done),
      is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
          self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
      reward=np.float32(reward),
      is_first=is_first,
      is_last=is_last,
      is_terminal=is_terminal)
    return obs
