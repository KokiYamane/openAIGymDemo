import torch
from torch import nn
import random


class CNNQNetwork(nn.Module):
    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action

        # CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 32x20x20 -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 64x9x9 -> 64x7x7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.conv_layers = nn.Linear(4, 3136)

        # state value
        self.fc_state = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # advantage
        self.fc_advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_action),
        )

    def forward(self, observation):
        feature = self.conv_layers(observation)

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        action_values = state_values + advantage
        # print(advantage.shape)
        action_values -= torch.mean(advantage)
        return action_values

    # epsilon-greedy
    def act(self, observation, epsilon=0):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            with torch.no_grad():
                action = torch.argmax(self.forward(observation)).item()
        return action
