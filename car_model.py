
import torch.nn.functional as F
import torch.nn as nn
import gym
from torch.distributions import Normal
import torch
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 128)
        """self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512,1024)
        self.layer6 = nn.Linear(1024,2048)
        self.layer7 = nn.Linear(2048,2048)
        self.layer8 = nn.Linear(2048,2048)
        self.layer9 = nn.Linear(2048,2048)
        self.layer10 = nn.Linear(2048,1024)
        self.layer11 = nn.Linear(1024,512)
        self.layer12 = nn.Linear(512, 256)
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128,32)
        self.layer15 = nn.Linear(32,16)"""
        self.layer16 = nn.Linear(128, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        """x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))
        x = F.relu(self.layer12(x))
        x = F.relu(self.layer13(x))
        x = F.relu(self.layer14(x))
        x = F.relu(self.layer15(x))"""

        return self.layer16(x)

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        """self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(512,1024)
        self.layer6 = nn.Linear(1024,2048)
        self.layer7 = nn.Linear(2048,5096)
        self.layer8 = nn.Linear(5096,5096)
        self.layer9 = nn.Linear(5096,5096)
        self.layer10 = nn.Linear(5096,5096)
        self.layer11 = nn.Linear(5096,2048)
        self.layer12 = nn.Linear(2048,1024)
        self.layer13 = nn.Linear(1024,512)
        self.layer14 = nn.Linear(128, 128)
        self.layer15 = nn.Linear(128, 64)
        self.layer16 = nn.Linear(64,32)
        self.layer17 = nn.Linear(128,32)"""
        self.layer18 = nn.Linear(512, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        """x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))
        x = F.relu(self.layer12(x))
        x = F.relu(self.layer13(x))
        x = F.relu(self.layer14(x))
        x = F.relu(self.layer15(x))
        x = F.relu(self.layer16(x))
        x = F.relu(self.layer17(x))"""

        return self.layer18(x)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.relu(),
            nn.Linear(hidden_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.relu(),
            nn.Linear(hidden_size, num_actions),
        )
        self.num_actions = num_actions
    def forward(self, state):
        value = self.critic(state)
        policy_dist = Normal(self.actor(state), torch.ones(self.num_actions))
        return value, policy_dist















def get_env():
    env = gym.make("racetrack-v0")
    env.config["controlled_vehicles"] = 1
    env.config["manual_control"]= False
    env.config["duration"] =7
    env.config["lane_centering_cost"] = 5
    env.config["lane_centering_reward"] = 1
    env.config['other_vehicles']= 0
    env.config["action"] = {
                "type": "ContinuousAction",
                "longitudinal":True,
                "lateral": True,
                "target_speeds": [0, 5, 10]
        }
    """env.config["observation"] = {
                        "type": "Kinematics",
                        "features": [ "x", "y", "vx", "vy", "cos_h", "sin_h", "long_off", "lat_off"],
                        "features_range": {
                            "x": [-300, 300],
                            "y": [-300, 300],
                            "vx": [-20, 20],
                            "vy": [-20, 20]
                        },
                        "absolute": False,
                        "order": "sorted"

    }"""
    

    env.config["observation"] = {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            }
    return env