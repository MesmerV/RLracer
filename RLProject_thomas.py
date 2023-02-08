import numpy as np
import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env

import warnings
warnings.filterwarnings("ignore")

TRAIN = False

if __name__ == '__main__':
    n_cpu = 8
    batch_size = 64
    """env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log="racetrack_ppo/",
                device="cuda")
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("racetrack_ppo/model")
        del model"""
    

    # Run the algorithm
    
    env = gym.make("racetrack-v0")
    env.config["controlled_vehicles"] = 5
    env.config["manual_control"]= False
    
    env.config['other_vehicles']= 7
    env.config["action"] = {"type": "MultiAgentAction",
                "longitudinal":True,
                "lateral": True,
                "target_speeds": [0, 5, 10],
                "action_config": {
                "type": "ContinuousAction",
                }
    }
    env.config["observation"] = {
                "type": "MultiAgentObservation",
                "observation_config": {
                "type": "OccupancyGrid",
                },
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            }
    
    
    
    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    
    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        
        while not (done or truncated):
            # Predict
            #action, _states = model.predict(obs, deterministic=True)

            #print("action : "+str(action))
            # Get reward
            obs, reward, done, truncated, info = env.step(0.01)
            # Render
            env.render()
    env.close()