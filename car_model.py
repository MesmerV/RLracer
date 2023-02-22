
import torch.nn.functional as F
import torch.nn as nn
import gym
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        """self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512,1024)
        self.layer6 = nn.Linear(1024,2048)
        self.layer7 = nn.Linear(2048,2048)
        self.layer8 = nn.Linear(2048,2048)
        self.layer9 = nn.Linear(2048,2048)
        self.layer10 = nn.Linear(2048,1024)
        self.layer11 = nn.Linear(1024,512)
        self.layer12 = nn.Linear(512, 256)"""
        self.layer13 = nn.Linear(128, 128)
        self.layer14 = nn.Linear(128,32)
        self.layer15 = nn.Linear(32,16)
        self.layer16 = nn.Linear(16, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        """x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))
        x = F.relu(self.layer12(x))
        x = F.relu(self.layer13(x))"""
        x = F.relu(self.layer14(x))
        x = F.relu(self.layer15(x))

        return self.layer16(x)


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        """self.layer5 = nn.Linear(512,1024)
        self.layer6 = nn.Linear(1024,2048)
        self.layer7 = nn.Linear(2048,5096)
        self.layer8 = nn.Linear(5096,5096)
        self.layer9 = nn.Linear(5096,5096)
        self.layer10 = nn.Linear(5096,5096)
        self.layer11 = nn.Linear(5096,2048)
        self.layer12 = nn.Linear(2048,1024)
        self.layer13 = nn.Linear(1024,512)"""
        self.layer14 = nn.Linear(512, 256)
        self.layer15 = nn.Linear(256, 128)
        self.layer16 = nn.Linear(128,32)
        self.layer17 = nn.Linear(32,16)
        self.layer18 = nn.Linear(16, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        """x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))
        x = F.relu(self.layer12(x))
        x = F.relu(self.layer13(x))"""
        x = F.relu(self.layer14(x))
        x = F.relu(self.layer15(x))
        x = F.relu(self.layer16(x))
        x = F.relu(self.layer17(x))

        return self.layer18(x)

def get_env():
    env = gym.make("racetrack-v0")
    env.config["controlled_vehicles"] = 1
    env.config["manual_control"]= False
    env.config["duration"] =10
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