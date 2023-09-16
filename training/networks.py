import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #cov2d layers NEW
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten= nn.Flatten()
        self.fc1 = nn.Linear(900, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)

        self.fc1_st2 = nn.Linear(2, int(hidden_size/4))
        self.bn1_st2 = nn.LayerNorm(int(hidden_size/4))

        self.fc2_st2 = nn.Linear(int(hidden_size/4), int(hidden_size/2))
        self.bn2_st2 = nn.LayerNorm(int(hidden_size/2))

        self.fc2 = nn.Linear(hidden_size +int(hidden_size/2), int(hidden_size/2))
        self.bn2 = nn.LayerNorm(int(hidden_size/2))

        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.bn3 = nn.LayerNorm(int(hidden_size/4))

        self.mu = nn.Linear(int(hidden_size/4), action_size)
        self.log_std_linear = nn.Linear(int(hidden_size/4), action_size)

        

    def forward(self, state1,state2):
        # print("State in act net:",state.shape)
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))

        x = F.relu(self.conv1(state1))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=self.flatten(x)
        # print('flatten shape :',x.shape)
        x=self.fc1(x)
        x=F.relu(self.bn1(x))

        x2 = F.relu(self.fc1_st2(state2))
        x2=F.relu(self.bn1_st2(x2))
        x2 = F.relu(self.fc2_st2(x2))
        x2=F.relu(self.bn2_st2(x2))

        y = torch.cat((x,x2), dim=-1)

        y=self.fc2(y)
        y=F.relu(self.bn2(y))

        y=self.fc3(y)
        y=F.relu(self.bn3(y))
        mu = self.mu(y)

        log_std = self.log_std_linear(y)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state1,state2, epsilon=1e-6):
        mu, log_std = self.forward(state1,state2)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state1.device) ##-------------------------------------
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state1,state2):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state1,state2)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state1.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state1,state2):
        mu, log_std = self.forward(state1,state2)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        #cov2d layers NEW
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten= nn.Flatten()
        self.fc1 = nn.Linear(900, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.LayerNorm(int(hidden_size/2))

        self.fc3 = nn.Linear(action_size,int(hidden_size/4))
        self.bn3 = nn.LayerNorm(int(hidden_size/4))

        self.fc4 = nn.Linear(int(hidden_size/2)+int(hidden_size/4)+int(hidden_size/2),int(hidden_size/4))
        self.bn4 = nn.LayerNorm(int(hidden_size/4))

        self.fc1_st2 = nn.Linear(2, int(hidden_size/4))
        self.bn1_st2 = nn.LayerNorm(int(hidden_size/4))

        self.fc2_st2 = nn.Linear(int(hidden_size/4), int(hidden_size/2))
        self.bn2_st2 = nn.LayerNorm(int(hidden_size/2))

        # self.fc3_st2 = nn.Linear(2, int(hidden_size))
        # self.bn3_st2 = nn.LayerNorm(int(hidden_size))
        
        self.fc_out = nn.Linear(int(hidden_size/4), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state1,state2, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        #state
        x = F.relu(self.conv1(state1))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=self.flatten(x)
        x=self.fc1(x)
        x=F.relu(self.bn1(x))
        x=self.fc2(x)
        x=F.relu(self.bn2(x))

        x2 = F.relu(self.fc1_st2(state2))
        x2=F.relu(self.bn1_st2(x2))

        x2 = F.relu(self.fc2_st2(x2))
        x2=F.relu(self.bn2_st2(x2))

        #action
        y=self.fc3(action)
        y=F.relu(self.bn3(y))

        z = torch.cat((x,x2,y), dim=-1)
        # print('concat shape :',z.shape)
        z=self.fc4(z)
        z=F.relu(self.bn4(z))

        return self.fc_out(z)