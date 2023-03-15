import random
import gym
from car_model import Critic, Actor, get_env
import torch
import sys
sys.path.insert(1, "./highway-env")
import highway_env
import warnings
warnings.filterwarnings("ignore")
import numpy as np

actor_weights = "actor.pth"
critic_weights = "critic.pth"





env = get_env()
obs, info = env.reset()
try:
    n1, n2, n3 = obs.shape
    state_size = n1*n2*n3
    print(obs.shape)
    print(state_size)
except:
    n1, n2 = obs.shape
    state_size = n1*n2
state_size = n1*n2*n3
action_size = 2
actor = Actor(state_size, action_size)
actor.load_state_dict(torch.load(actor_weights))
done = False
state = obs[0:2]
state = torch.tensor(state[0:2], dtype=torch.float32)
while not done:
    state = state[0:2]
    state = torch.tensor(state, dtype=torch.float32)
    action = actor(state).detach().numpy()[0]
    action[0] = np.clip(action[0],0,20)
    state, rewards , terminated, truncated, _ = env.step(action)
    env.render()
