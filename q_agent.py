import numpy as np


class Q:

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

        np.random.seed(hparams['Rand_Seed'])
        self.q_table = np.zeros(shape=(self.num_states, self.num_actions))

        self.epsilon_decay = self.epsilon / (self.num_episodes/self.rate_of_decay)

    def get_action(self, state, env):

        if np.random.random() <= self.epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))

        return action

    def update_network(self, state, action, next_action, reward, new_state, done):
        if done:
            self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
        else:
            self.q_table[state][action] += self.alpha * (
                        reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_decay)








