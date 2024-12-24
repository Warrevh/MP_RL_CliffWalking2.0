import gymnasium as gym
import random

from gymnasium import wrappers
from collections import deque


class CliffWalkingEnv:

    def __init__(self, agent_type, par):
        self.STORE_PATH = par['STORE_PATH']
        self.RenderMode = par['RenderMode']

        self.replay_buffer_active = par['replay_buffer_active']
        self.replay_buffer = deque(maxlen=par['replay_buffer_size'])
        self.batch_size = par['batch_size']

        self.env = gym.make('CliffWalking-v0', render_mode = self.RenderMode)
        if par['monitor']:
            self.env = wrappers.RecordVideo(self.env,
                                            self.STORE_PATH + f"/vid/{agent_type}",
                                            episode_trigger=lambda episode_id: episode_id % 100 == 0,
                                            fps = 100
                                            )



    def one_rollout(self, agent):
        states, actions, next_action, rewards, new_states, dones = [], [], [], [], [], []
        state = self.env.reset()[0]
        terminated, truncated = False, False
        done = terminated or truncated

        while not done:
            action = agent.get_action(state, self.env)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            next_action = agent.get_action(new_state, self.env)

            done = terminated or truncated

            agent.update_network(state, action, next_action, reward, new_state, done)

            #agent.q_table[state][action] += agent.alpha * (reward + agent.gamma * agent.q_table[new_state][next_action] - agent.q_table[state][action])

            self.replay_buffer.append((state, action, next_action, reward, new_state, done))

            if len(self.replay_buffer) > self.batch_size and self.replay_buffer_active:

                samples = random.sample(self.replay_buffer,self.batch_size)

                for experience in samples:
                    s, a, n_a, r, n_s, d = experience
                    agent.update_network(s, a, n_a, r, n_s, d)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
            state = new_state

        #agent.decay_epsilon()
        return states, actions, rewards, new_states, dones







