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
TRAINING_STEPS = 1e4
USE_PREVIOUS_MODEL = True
NUM_AGENTS = 3
PLAY = True
MANUAL = False

def CreateEnv():
    # dl racetrack as baseline
    env = gym.make("racetrack-v0")

    # General Config ( *configure in racetrack_env for training* )
    env.configure({ 
        "collision_reward": -1.5,
        "lane_centering_cost": 4,
        "lane_centering_reward": 3,
        "high_speed_reward": 1.5,
        "action_reward": -0.5,

        "other_vehicles": 8,
        "screen_width": 600,
        "show_trajectories": False,
        "screen_heigth": 600
    })
    
    env.configure({
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "target_speeds": [0, 5, 10]   
        },
        "policy_frequency": 10
    })
    

    # for manual control
    if MANUAL:
        env.config["manual_control"] = True

    #apply changes
    #env.reset()
    
    return env

def ConfigureMultiAgent(env,agent_num):
    
    #configure several agents
    env.configure({ "controlled_vehicles": agent_num })
    
    
    #get config for one agent
    action_config = env.config["action"]
    obs_config = env.config["observation"]
    
    
    
    #multi-action confige2
    env.configure({
        "action": {
                "type": "MultiAgentAction",
                    "action_config": action_config
            },
          })
    
    #config multi-observation
    env.configure({
        "observation": {
                "type": "MultiAgentObservation",
                    "observation_config": obs_config
                }
            })  

if __name__ == '__main__':

    n_cpu = os.cpu_count() - 1
    #env = CreateEnv()
    env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)


    # If TRAIN, create new model and train it
    if TRAIN:
        batch_size = 64
        
        
        if not USE_PREVIOUS_MODEL:
            model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=2e-4,
                gamma=0.9,
                verbose=3,
                tensorboard_log="racetrack_ppo/")
        else:
            # for further training of previous model
            model = PPO.load("racetrack_ppo/model_PPO", env=env)

        # Train the model
        model.learn(total_timesteps=int(TRAINING_STEPS))
        model.save("racetrack_ppo/model_PPO")
        del model

    if PLAY:    

        # Run the algorithm
        model = PPO.load("racetrack_ppo/model_PPO4", env=env)

        # dl racetrack as baseline
        env = CreateEnv()
        ConfigureMultiAgent(env, NUM_AGENTS)


        env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)

        print(env.config)

        done = truncated = False
        obs, info = env.reset()
        print("number of obs: ",len(obs))

        while not (done or truncated):
            # Predict

            # Dispatch the observations to the model to get the tuple of actions
            actions = tuple(model.predict(obs_i)[0] for obs_i in obs)
            #actions, _states = model.predict(obs, deterministic=True)

            # Execute the actions
            obs, reward, done, truncated, info = env.step(actions)

            # Render
            env.render()
    env.close()