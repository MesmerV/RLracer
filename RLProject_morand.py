import numpy as np
import os, sys
import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(1, "./highway-env")
import highway_env


import warnings
warnings.filterwarnings("ignore")




TRAIN = False
MANUAL = False

if __name__ == '__main__':

    n_cpu = os.cpu_count() - 1
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    

    #If train, create new model and train it
    if TRAIN:
        batch_size = 64
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
        model.learn(total_timesteps=int(1e5))
        model.save("racetrack_ppo/model_morand")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_ppo/model_morand", env=env)
    env = gym.make("racetrack-v0")

    
    # Config environment
    env.configure({"collision_reward": -2,
                    "lane_centering_cost": 3,
                    "lane_centering_reward": 1,
                    "controlled_vehicles": 1,
                    "other_vehicles": 1,
                    "screen_width": 600,
                    "show_trajectories": True,
                    "screen_heigth": 600})

    if MANUAL:
        env.config["action"]["longitudinal"] = True
        env.config["manual_control"] = True


    # for manual control

    #apply changes
    env.reset()
    

    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    
    print(env.config)
    
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
    env.close()