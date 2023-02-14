import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
# Définir le modèle de l'agent
class model(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(model, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 512)
        self.layer5 = nn.Linear(512, 512)
        self.layer6 = nn.Linear(512, 256)
        self.layer7 = nn.Linear(256,128)
        self.layer8 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = x.float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))

        return self.layer8(x)

# Initialiser l'environnement

device = "cuda"
env = gym.make("racetrack-v0")
env.config["controlled_vehicles"] = 1
env.config["manual_control"]= False
env.config["duration"] =10
env.config["lane_centering_cost"] = 10


env.config['other_vehicles']= 0
env.config["action"] = {
                "type": "ContinuousAction",
                "longitudinal":True,
                "lateral": True,
                "target_speeds": [0, 5, 10,15,20]
            }

"""env.config["observation"] = {
                "type": "OccupancyGrid",
                "features": ['presence','on_road'],
                "grid_size": [[-50, 50], [-50, 50 ]],
                "grid_step": [3,3],
                "as_image": False,
                "align_to_vehicle_axes": True

}"""
env.config["observation"] = {
                "type": "Kinematics",
                "features": ['presence', "x", "y", "vx", "vy", "long_off", "lat_off"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"

}
"""env.config["action"] = {"type": "MultiAgentAction",
            "longitudinal":True,
            "lateral": True,
            "target_speeds": [0, 5, 10],
            "action_config": {
            "type": "ContinuousAction",
            }
}"""
"""env.config["observation"] = {
            "type": "MultiAgentObservation",
            "observation_config": {
            "type": "OccupancyGrid",
            },
            "features": ['presence', 'on_road'],
            "grid_size": [[-18, 18], [-18, 18]],
            "grid_step": [3, 3],
            "as_image": False,
            "align_to_vehicle_axes": True
        }"""


obs, info = env.reset()
try:
    n1, n2, n3 = obs.shape
    state_size = n1*n2*n3
except:
    n1, n2 = obs.shape
    state_size = n1*n2
action_size = 2

actor = model(state_size, action_size).to(device)
critic = model(state_size + action_size, 1).to(device)


# Définir les hyperparamètres
learning_rate = 5e-4
memory_size = 1000000
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Initialiser la mémoire


# Définir l'optimiseur
optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate)
optimizer_critic = optim.Adam(critic.parameters(), lr= learning_rate)

def select_action(actor, state):
    accel, angle = actor(state)[0]
    
    accel =  torch.normal(accel, std=0.5)
    accel = torch.clamp(accel, -2, 2)

    angle =  torch.normal(angle, std=0.07)
    angle = torch.clamp(angle, -0.78, 0.78)

    actions = torch.tensor([accel.item(), angle.item()], device=device, dtype=torch.long)
    
    
    copy = [accel.item(), angle.item()]
    return actions, copy
    
torch.set_grad_enabled(True)
loss_fn = nn.MSELoss()
gains_totaux = []   
writer = SummaryWriter() 
# Boucle d'entraînement de l'agent
for i_episode in range(1000):
    if (i_episode == 120):
        env.config["duration"] = 30
    if (i_episode == 300):
        env.config["duration"] = 45
    if (i_episode == 400):
        env.config["duration"] = 60
    if (i_episode == 450):
        env.config["duration"] = 90
    state, info = env.reset()
    
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    gains = []
    losses = []
    while not done:
        # Choisir une action
        long, lat = state[0][-2:]
        state = torch.tensor(state, dtype=torch.float32)
        state = state.to(device)
        action_torch, action = select_action(actor,state)
        
        # Efcfectuer l'action et observer les récompenses
        next_state, reward, terminated, truncated, _ = env.step([action[0],action[1]])
        
        
        
        
        done = terminated or truncated
        gains.append(reward)
        reward = torch.tensor([reward])
        with torch.no_grad():
            critic_input = state.reshape((state_size,))
            critic_input = torch.cat([critic_input,action_torch])
            q_value =  critic(critic_input)
        q_value = q_value.to("cpu")
        
        optimizer_critic.zero_grad()
        q_value_loss = loss_fn(q_value[0], reward)
        losses.append(q_value_loss.item())
        q_value_loss.requires_grad = True
        
        q_value_loss.backward()
        optimizer_critic.step()

        optimizer_actor.zero_grad()
        policy_loss = -q_value.mean()
        policy_loss.requires_grad = True
        policy_loss.backward()
        optimizer_actor.step()

        state = next_state
        
        
        if (np.sqrt(long**2 + lat**2) > 70):
            done = True
    gains_totaux.append(np.mean(gains))
    print(f"Gains moyens sur le run {i_episode} est : {np.mean(gains)}")
    print(f"Pertes moyenness sur le run {i_episode} est : {np.mean(losses)}")
    print()
    writer.add_scalar("Loss/train", np.mean(losses), i_episode)
    writer.add_scalar("Rewards/train", np.mean(gains), i_episode)

writer.close()
    
        

