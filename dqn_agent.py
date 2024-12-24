import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN:

    def __init__(self, hparams):
        self.hparams = hparams
        self.num_actions = hparams['num_actions']
        self.num_states = hparams['num_states']
        self.epsilon = hparams['epsilon']
        self.epsilon_min = hparams['epsilon_min']
        self.rate_of_decay = hparams['rate_of_decay']
        self.alpha = hparams['learning_rate_alpha']
        self.gamma = hparams['gamma']
        self.num_episodes = hparams['num_episodes']
        self.num_neurons = hparams['num_of_neurons']

        self.epsilon_decay = self.epsilon / (self.num_episodes/self.rate_of_decay)

        self.num_columns = 12

        self.q_network = DQN_N(self.num_states, self.num_actions, self.num_neurons)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

    def get_action(self, state, env):

        if np.random.random() <= self.epsilon:
            return env.action_space.sample()
        else:
            state_tensor = nn.functional.one_hot(torch.tensor([state]), num_classes=self.num_states).float()
            #state_tensor = self.state_to_2d(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()


    def update_network(self, state, action, next_action, reward, new_state, done):
        state_tensor = F.one_hot(torch.tensor([state]), num_classes=self.num_states).float()
        #state_tensor = self.state_to_2d(state)
        next_state_tensor = F.one_hot(torch.tensor([new_state]), num_classes=self.num_states).float()
        #next_state_tensor = self.state_to_2d(new_state)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        q_values = self.q_network(state_tensor)[0]
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            max_next_q_value = torch.max(next_q_values)

        target_q_value = reward_tensor + (self.gamma * max_next_q_value * (1 - done))
        q_values[action] = target_q_value

        #if done:
        #   q_values[action] += self.alpha * (reward_tensor - q_values[action])
        #else:
        #    q_values[action] += self.alpha * (
        #                reward + self.gamma * max_next_q_value - q_values[action])

        loss = nn.MSELoss()(self.q_network(state_tensor)[0], q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)

    def state_to_2d(self, state):

        y = state // self.num_columns
        x = state % self.num_columns

        return torch.tensor([x,y], dtype=torch.float32)

class DQN_N(nn.Module):
    def __init__(self, state_size, action_size, num_neurons):
        super(DQN_N, self).__init__()
        #state_size = 2
        self.fc1 = nn.Linear(state_size, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.fc3 = nn.Linear(num_neurons, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)