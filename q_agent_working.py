import numpy as np


class Q:

    def __init__(self, hparams):
        self.hparams = hparams
        self.plt_title = hparams['plt_title']
        self.num_actions = hparams['num_actions']
        self.num_states = hparams['num_states']
        self.epsilon = hparams['epsilon']
        self.alpha = hparams['learning_rate_alpha']
        self.gamma = hparams['gamma']
        self.num_episodes = hparams['num_episodes']

        np.random.seed(hparams['Rand_Seed'])
        self.q_table = np.zeros(shape=(self.num_states, self.num_actions))

    def get_action(self, state, env):
        action = int(np.argmax(self.q_table[state]))
        if np.random.random() <= self.epsilon:
            action = env.action_space.sample()

        return action

    def update_network(self, state, action, reward, new_state, next_action):
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[new_state][next_action] - self.q_table[state][action])








