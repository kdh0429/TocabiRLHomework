import gym
from gym.envs.registration import registry, make, spec

def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return 
    else:
        return gym.envs.registration.register(id, *args, **kvargs)

# Registering environment requires a version ID and entry point
# The entry point is really a python module import statement, then the
# colon (:XXX) means import the XXX class name that we want

register(id='tocabi-stand-still-v0', 
        entry_point = 'tocabirl.cust_gym.tocabi_stand_still:DYROSTocabiEnv',
        max_episode_steps=2000)

register(id='tocabi-squat-v0', 
        entry_point = 'tocabirl.cust_gym.tocabi_squat:DYROSTocabiEnv',
        max_episode_steps=8000)

register(id='tocabi-walk-v0', 
        entry_point = 'tocabirl.cust_gym.tocabi_walk:DYROSTocabiEnv',
        max_episode_steps=8000)

register(id='tocabi-run-v0', 
        entry_point = 'tocabirl.cust_gym.tocabi_run:DYROSTocabiEnv',
        max_episode_steps=2000)

register(id='atlas-walk-v0', 
        entry_point = 'tocabirl.cust_gym.atlas_walk:AtlasEnv',
        max_episode_steps=2000)

register(id='atlas-run-v0', 
        entry_point = 'tocabirl.cust_gym.atlas_run:AtlasEnv',
        max_episode_steps=2000)

register(id='op3-walk-v0', 
        entry_point = 'tocabirl.cust_gym.op3_walk:Op3Env',
        max_episode_steps=8000)