import numpy as np
import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO

from collections import namedtuple, deque
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import random
warnings.filterwarnings("ignore")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
TRAIN = False
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
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
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 100000
TAU = 0.005
LR = 1e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
if __name__ == '__main__':
    
    

    # Run the algorithm
    
    env = gym.make("racetrack-v0")
    env.config["controlled_vehicles"] = 1
    env.config["manual_control"]= False
    env.config["duration"] =25
    env.config["lane_centering_cost"] = 4
    
    
    env.config['other_vehicles']= 2
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
    n_actions = 2
    obs, info = env.reset()
    n1, n2, n3 = obs.shape
    print("n1 : "+str(n1))
    print("n2 : "+str(n2))
    print("n3 : "+str(n3))
    n_observations = n1*n2*n3
    
    
    
    env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    
    if (torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0
    def select_action(state):
        global steps_done
        sample = np.random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                
                
                act = policy_net(state)
                accel, angle = act[0][0], act[0][1]
                """accel = accel.to("cpu")
                angle = angle.to("cpu")"""
                
                return torch.tensor([accel.item(), angle.item()], device="cpu", dtype=torch.long)
        else:
            accel = np.random.uniform(-5,5)
            angle = np.random.uniform(-.0785, 0.785)
            return torch.tensor([accel,angle], device="cpu", dtype=torch.long)
    
    
    
    
    episode_durations = []
    

    
    
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
       
        
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        
        state_batch = state_batch.view((-1,12,12))
        
        #state_action_values = policy_net(state_batch)
        action_batch = action_batch.to(device)
        
        action_batch = action_batch.reshape((-1,2))
        
        state_action_values = policy_net(state_batch)
        #state_action_values = state_action_values.gather(1,action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            
            #next_state_values[non_final_mask] = target_net(non_final_next_states)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        if (i_episode == 120):
            env.config["duration"] = 30
        if (i_episode == 300):
            env.config["duration"] = 45
        if (i_episode == 400):
            env.config["duration"] = 60
        if (i_episode == 450):
            env.config["duration"] = 90
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state = torch.tensor(state[0])
        gain = 0
        for t in count():
            action = select_action(state)
            accel, angle = action[0], action[1]
            observation, reward, terminated, truncated, _ = env.step([accel, angle])
            gain += reward
            
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                next_state = next_state[0]
                
            
            #action = torch.Tensor(action)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


























