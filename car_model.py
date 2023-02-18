
import torch.nn.functional as F
import torch.nn as nn
class model(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(model, self).__init__()
        self.n_input = n_observations
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        self.layer5 = nn.Linear(512,512)
        self.layer6 = nn.Linear(512, 256)
        self.layer7 = nn.Linear(256, 128)
        self.layer8 = nn.Linear(128,32)
        self.layer9 = nn.Linear(32,16)
        self.layer10 = nn.Linear(16, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.view(-1,self.n_input)
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = F.tanh(self.layer4(x))
        x = F.tanh(self.layer5(x))
        x = F.tanh(self.layer6(x))
        x = F.tanh(self.layer7(x))
        x = F.tanh(self.layer8(x))
        x = F.tanh(self.layer9(x))
        return self.layer10(x)