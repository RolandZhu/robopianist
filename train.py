import os
import sys
sys.path.append('./train')

import numpy as np
from robopianist.suite.tasks import self_actuated_piano
from robopianist.suite.tasks import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env
from dmcgym import DMCGYM

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load("TwinkleTwinkleRousseau"),
            trim_silence=True,
            control_timestep=0.05,
            gravity_compensation=True,
            primitive_fingertip_collisions=False,
            reduced_action_space=False,
            n_steps_lookahead=10,
            disable_fingering_reward=False,
            disable_forearm_reward=False,
            disable_colorization=False,
            disable_hand_collisions=False,
            attachment_yaw=0.0,
        )

        env = composer_utils.Environment(
            task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
        )
        
        # env = PianoSoundVideoWrapper(
        #     env,
        #     record_every=1,
        #     camera_id=None, # "piano/back",
        #     record_dir="./videos",
        # )

        # env = CanonicalSpecWrapper(env)
    
        env = DMCGYM(env)
        
        return env
    
    set_random_seed(seed)
    
    return _init

if __name__ == '__main__':
    
    vec_env = SubprocVecEnv([make_env(i) for i in range(4)])
    
    obs = vec_env.reset()

    actions = np.load("examples/twinkle_twinkle_actions.npy")
    
    for action in actions:
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    