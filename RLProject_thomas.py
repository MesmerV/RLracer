import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
learning_rate = 5e-5
n_episode = 1000
# Définir le modèle de l'agent
from car_model import Actor, Critic
class A2C():
    def __init__(self, n_observations, n_actions):
        self.actor = Actor(n_observations, n_actions).to(device)
        self.critic = Critic(n_observations+n_actions, 1).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr= learning_rate)
        self.loss_fn = nn.MSELoss()
    def get_q_value(self,critic_input):
        return self.critic(critic_input )
    
    def get_action(self,state, episode):
        
        #std_v = 0.9 - 0.5 *np.min([episode/100,1])
        std_v = 0.9 - (0.03-0.9) *episode/n_episode
        actions = self.actor(state)
        temp = actions[0].clone()
        temp = temp.detach()
        accel, angle = temp.cpu().numpy()
        #accel, angle = self.actor(state)[0]
        accel =  np.random.normal(accel, std_v)
        accel = np.clip(accel, -5, 5)

        std_theta =  0.5 - (0.005-0.5)*episode/n_episode
        angle =  np.random.normal(angle, std_theta)
        angle = np.clip(angle, -0.78, 0.78)

        
        
        
        
        return actions, [accel, angle]
    
    def update_critics(self,q_value, reward):
        self.optimizer_critic.zero_grad()
        q_value_loss = self.loss_fn(q_value, reward)
        q_value_loss.backward(retain_graph = True)
        
        self.optimizer_critic.step()
        return q_value_loss

    def update_actor(self, action_torch, state):

        self.optimizer_actor.zero_grad()
        critic_input = state.reshape((1,16))
    
        critic_input = torch.cat([critic_input,action_torch],dim = 1 )
        q_value_actor = -self.get_q_value(critic_input)
        
        q_value_actor.backward()
        
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                print(f'{name} gradient: {param.grad}')
        self.optimizer_actor.step()
        
        return q_value_actor

# Initialiser l'environnement
def get_env():
    env = gym.make("racetrack-v0")
    env.config["controlled_vehicles"] = 1
    env.config["manual_control"]= False
    env.config["duration"] =12
    env.config["lane_centering_cost"] = 2
    env.config['other_vehicles']= 0
    env.config["action"] = {
                "type": "ContinuousAction",
                "longitudinal":True,
                "lateral": True,
                "target_speeds": [0, 5, 10,15,20]
            }
    env.config["observation"] = {
                        "type": "Kinematics",
                        "features": [ "x", "y", "vx", "vy", "cos_h", "sin_h","long_off", "lat_off"],
                        "features_range": {
                            "x": [-100, 100],
                            "y": [-100, 100],
                            "vx": [-20, 20],
                            "vy": [-20, 20]
                        },
                        "absolute": False,
                        "order": "sorted"

        }
    return env
device = "cuda"
env= get_env()
obs, info = env.reset()
try:
    n1, n2, n3 = obs.shape
    state_size = n1*n2*n3
except:
    n1, n2 = obs.shape
    state_size = n2*2
action_size = 2
actor = Actor(16,2)
critic = Critic(18,1)
actor = actor.to(device)
critic = critic.to(device)
optimizer_actor = optim.Adam(actor.parameters(), lr=0.001)
optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
def get_q_value(critic_input):
    return critic(critic_input )

def get_action(state, episode):
    
    #std_v = 0.9 - 0.5 *np.min([episode/100,1])
    #std_v = 0.9 - (0.03-0.9) *episode/n_episode
    std_v = 0.9
    actions = actor(state)
    temp = actions[0].clone()
    temp = temp.detach()
    accel, angle = temp.cpu().numpy()
    #accel, angle = self.actor(state)[0]
    accel =  np.random.normal(accel, std_v)
    accel = np.clip(accel, -5, 5)

    #std_theta =  0.15 - (0.005-0.15)*episode/n_episode
    std_theta = 0.15
    angle =  np.random.normal(angle, std_theta)
    angle = np.clip(angle, -0.78, 0.78)

    
    
    
    
    return actions, [accel, angle]

def update_critics(q_value, reward):
    optimizer_critic.zero_grad()
    q_value_loss = loss_fn(q_value, reward)
    q_value_loss.backward(retain_graph = True)
    """for name, param in critic.named_parameters():
        if param.grad is not None:
            print(f'{name} gradient: {param.grad}')"""
   
    optimizer_critic.step()
    return q_value_loss

def update_actor(action_torch, state):


    optimizer_actor.zero_grad()
    critic_input = state.reshape((1,16))
    
    critic_input = torch.cat([critic_input,action_torch],dim = 1 )
    q_value_actor = -get_q_value(critic_input)
    
    q_value_actor.backward()
    
    """for name, param in actor.named_parameters():
        if param.grad is not None:
            print(f'{name} gradient: {param.grad}')"""
    optimizer_actor.step()
    
    return 
    
torch.set_grad_enabled(True)
def train(seed):
    
    env = env= get_env()
    writer = SummaryWriter(f"runs/{seed}") 
    # Boucle d'entraînement de l'agent
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for i_episode in range(n_episode):
        """if (i_episode % 150 == 0  and i_episode !=0) :
            env.config["duration"] = 25
        if (i_episode % 300 == 0  and i_episode !=0) :
            env.config["duration"] = 40
        if (i_episode % 600 == 0  and i_episode !=0) :
            env.config["duration"] = 50"""
        state, info = env.reset()
        if (i_episode % 100 ==0  and i_episode != 0):
            torch.save(actor.state_dict(), "actor.pth")
            torch.save(critic.state_dict(), "critic.pth")
            print(i_episode)
        
        state = state[0:2]
        
        
        state = torch.tensor(state[0:2], dtype=torch.float32)
        done = False
        gains = []
        losses = []
        while not done:
            # Choisir une action
            long, lat = state[0][-2:]
            state = state[0:2]
            state = torch.tensor(state, dtype=torch.float32)
            state = state.to(device)
            action_torch, action = get_action(state, i_episode)
            # Efcfectuer l'action et observer les récompenses
            next_state, reward, terminated, truncated, _ = env.step([action[0],action[1]])
            
            
            
            
            done = terminated or truncated
            gains.append(reward)
            reward = torch.tensor([reward], dtype=torch.float)
            reward = reward.to(device)
            critic_input = state.reshape((1,state_size))
            
            critic_input = torch.cat([critic_input,action_torch],dim = 1 )
            
            
            

            q_value =  get_q_value(critic_input)
            q_value_loss = update_critics(q_value, reward)
            
            update_actor(action_torch, state)
            

            state = next_state
            env.render()
            losses.append(q_value_loss.item())
            if (np.sqrt(long**2 + lat**2) > 65):
                done = True
        writer.add_scalar("Loss/train", np.mean(losses), i_episode)
        writer.add_scalar("Rewards/train", np.mean(gains), i_episode)

    writer.close()
    torch.save(actor.state_dict(), "actor.pth")
    torch.save(critic.state_dict(), "critic.pth")
    

if __name__ == '__main__':
    processes = 3
    # initialiser une liste pour stocker les processus
    procs = []
    for i in range(processes):
        proc = mp.Process(target=train, args=(i,))
        print(i)
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        
    torch.save(actor.state_dict(), "actor.pth")
    torch.save(critic.state_dict(), "critic.pth")
