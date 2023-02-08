import numpy as np
import gym
import os
import sys
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(1, "./highway-env")
import highway_env

import pprint


TRAIN = False

if __name__ == '__main__':
    n_cpu = 4
    batch_size = 64
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log="racetrack_ppo/")
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e3))
        model.save("racetrack_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_ppo/model", env=env)

    env = gym.make("racetrack-v0")
    
    # additional configurations
    # Config environment
    env.configure({
        "collision_reward": -2,
        "lane_centering_cost": 3,
        "lane_centering_reward": 1,
        "controlled_vehicles": 3,
        "other_vehicles": 2,
        "screen_width": 800,
        "show_trajectories": True,
        "screen_heigth": 600
    })

    env.configure({
        "action": {
            "type": "MultiAgentAction",
                "action_config": {
                        "type": "ContinuousAction",
                        "longitudinal": False,
                        "lateral": True,
                        "target_speeds": [0, 5, 10]  
                        }
        },
    })

    #config multi-observation
    env.configure({
        "observation": {
            "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "OccupancyGrid",
                    "features": ['presence', 'on_road'],
                    "grid_size": [[-18, 18], [-18, 18]],
                    "grid_step": [3, 3],
                    "as_image": False,
                    "align_to_vehicle_axes": True
                    },
            }
    })

    # for manual control
    if MANUAL:
        env.config["action"]["longitudinal"] = True
        env.config["manual_control"] = False

    # allows one vehicule to be controlled
    env.config["manual_control"]= True

    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            ### MONO AGENT
            # # Predict
            # action, _states = model.predict(obs, deterministic=True)
            # # Get reward
            # obs, reward, done, truncated, info = env.step(action) # wbefore it was not 0.01 but action
            # # Render
            # env.render()

            ### MULTI AGENT
            print(obs)
            # Dispatch the observations to the model to get the tuple of actions
            action = tuple(model.predict(obs_i) for obs_i in obs)
            # Execute the actions
            next_obs, reward, done, truncated, info = env.step(action)
            # Update the model with the transitions observed by each agent
            for obs_i, action_i, next_obs_i in zip(obs, action, next_obs):
                model.update(obs_i, action_i, next_obs_i, reward, info, done, truncated)

    env.close()