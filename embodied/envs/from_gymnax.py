
import chex
import embodied


def get_gymnax_env(env_name: str,
                   rand_key: chex.PRNGKey):
    pass


class FromGymnax(embodied.Env):

    def __init__(self, env, obs_key='image', act_key='action', **kwargs):
        if isinstance(env, str):
            self._env = get_gymnax_env(env, **kwargs)
        else:
            assert not kwargs, kwargs
            self._env = env


