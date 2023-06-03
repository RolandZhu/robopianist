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
from stable_baselines3.common.evaluation import evaluate_policy

from sbx import DroQ

def make_env(song: str = 'TwinkleTwinkleRousseau', 
             seed: int = 0,
             sound: bool = False,):
    """
    Utility function for multiprocessed env.

    :param song: the name of the song
    :param seed: the inital seed for RNG
    """
    def _init():
        task = piano_with_shadow_hands.PianoWithShadowHands(
            change_color_on_activation=True,
            midi=music.load(song),
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
        
        if sound:
            env = PianoSoundVideoWrapper(
                env,
                record_every=1,
                camera_id=None, # "piano/back",
                record_dir="./videos",
            )
        env = CanonicalSpecWrapper(env)
    
        env = DMCGYM(env)
        
        return env
    
    set_random_seed(seed)
    
    return _init

if __name__ == '__main__':
    
    train_env = SubprocVecEnv([make_env('TwinkleTwinkleRousseau', 
                                      i,
                                      False) for i in range(2)])
    # test_env = make_env('TwinkleTwinkleRousseau', 100, True)()
    
    model = DroQ(
        "MultiInputPolicy",
        train_env,
        learning_starts=100,
        learning_rate=1e-3,
        tau=0.02,
        gamma=0.98,
        verbose=1,
        buffer_size=5000,
        gradient_steps=2,
        ent_coef="auto_1.0",
        seed=1,
        dropout_rate=0.001,
        layer_norm=True,
        # action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=1500)

    # # Check that something was learned
    # evaluate_policy(model, model.get_env(), reward_threshold=-800)
    # model.save(tmp_path / "test_save.zip")
    # env = model.get_env()
    # obs = env.observation_space.sample()
    # action_before = model.predict(obs, deterministic=True)[0]
    # # Check we have the same performance
    # model = DroQ.load(tmp_path / "test_save.zip")
    # evaluate_policy(model, env, reward_threshold=-800)
    # action_after = model.predict(obs, deterministic=True)[0]
    # assert np.allclose(action_before, action_after)
    # # Continue training
    # model.set_env(env, force_reset=False)
    # model.learn(100, reset_num_timesteps=False)