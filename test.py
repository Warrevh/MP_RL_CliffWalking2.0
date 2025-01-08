import os
import datetime as dt
import numpy as np

from cliffwalking_training import run_training
from data_handling import PLTLIB,Txt_File
from q_agent import Q
from SARSA_agent import SARSA
from dqn_agent import DQN
from CliffWalking_env import CliffWalkingEnv

STORE_PATH = './tmp/cliffwalking_test1/Q_and_SARSA' + f"/test_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
data_path = STORE_PATH + f"/data"
agent_path = data_path + f"/agent"

os.makedirs(data_path, exist_ok=True)
os.makedirs(agent_path, exist_ok=True)


env_parameters_replay = {
    'RenderMode': 'rgb_array', #human, rgb_array
    'monitor': False, #If rendermode = human then monitor must be False
    'STORE_PATH': STORE_PATH,
    'mean_number_ep': 100,
    'threshold_q': -13,
    'threshold_SARSA': -17,
    'replay_buffer_active': False,
    'replay_buffer_size': 10000,
    'batch_size': 32
}
env_parameters_no_replay = {
    'RenderMode': 'rgb_array', #human, rgb_array
    'monitor': False, #If rendermode = human then monitor must be False
    'STORE_PATH': STORE_PATH,
    'mean_number_ep': 100,
    'threshold_q': -13,
    'threshold_SARSA': -17,
    'replay_buffer_active': False,
    'replay_buffer_size': 1000,
    'batch_size': 32
}

env_parameters_a = env_parameters_replay
env_parameters_b = env_parameters_no_replay

q_a_CW = CliffWalkingEnv("Q", env_parameters_a)
q_b_CW = CliffWalkingEnv("Q", env_parameters_b)

agent_parameters = {
    'num_actions': q_a_CW.env.action_space.n,
    'num_states': q_a_CW.env.observation_space.n,
    'Rand_Seed': 1,
    'epsilon': 0.1,
    'epsilon_min': 0.001,
    'rate_of_decay': 6,
    'learning_rate_alpha': 0.01,
    'gamma': 1,
    'num_episodes': 10000,
    'num_of_neurons': 100
}

q_a_policy = DQN(agent_parameters)
q_b_policy = DQN(agent_parameters)


q_a_tot_rewards = run_training("Q", q_a_CW, q_a_policy, agent_parameters, env_parameters_a)
q_b_tot_rewards = run_training("Q", q_b_CW, q_b_policy, agent_parameters, env_parameters_b)


q_a_CW.env.close()
q_b_CW.env.close()


MyPlot = PLTLIB(agent_parameters, data_path)
MyPlot.fig2(q_a_tot_rewards,q_b_tot_rewards)





