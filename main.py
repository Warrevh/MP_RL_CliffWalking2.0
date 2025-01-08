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


env_parameters = {
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

q_CW = CliffWalkingEnv("Q", env_parameters)
sarsa_CW = CliffWalkingEnv("SARSA", env_parameters)
dqn_CW = CliffWalkingEnv("DQN", env_parameters)

print(dqn_CW.env.observation_space.n)

agent_parameters = {
    'num_actions': dqn_CW.env.action_space.n,
    'num_states': dqn_CW.env.observation_space.n,
    'Rand_Seed': 1,
    'epsilon': 0.1,
    'epsilon_min': 0.001,
    'rate_of_decay': 6,
    'learning_rate_alpha': 0.01,
    'gamma': 0.99,
    'num_episodes': 10000,
    'num_of_neurons': 100
}

q_policy = Q(agent_parameters)
SARSA_policy = SARSA(agent_parameters)
dqn_policy = DQN(agent_parameters)

q_tot_rewards = run_training("Q", q_CW, q_policy, agent_parameters, env_parameters)
SARSA_tot_rewards = run_training("SARSA", sarsa_CW, SARSA_policy,agent_parameters, env_parameters)
dqn_tot_rewards = run_training("DQN", dqn_CW, dqn_policy, agent_parameters, env_parameters)

q_CW.env.close()
sarsa_CW.env.close()
dqn_CW.env.close()


MyParameters = Txt_File(data_path)
MyParameters.save_parameters(agent_parameters, env_parameters)

np.save(f'{agent_path}/q_learning_q_table.npy', q_policy.q_table)
np.save(f'{agent_path}/SARSA_learning_q_table.npy', SARSA_policy.q_table)

MyPlot = PLTLIB(agent_parameters, data_path)
MyPlot.plot(q_tot_rewards, "Q-LEARNING")
MyPlot.plot(SARSA_tot_rewards, "SARSA-LEARNING")
MyPlot.plot2(q_tot_rewards, SARSA_tot_rewards, "Q vs SARSA")
MyPlot.plot(dqn_tot_rewards,"DQN-learning")


def test():
    episode_length_q= []
    episode_length_sarsa = []
    for i in range(500):
        q_policy = Q(agent_parameters)
        q_tot_rewards = run_training("Q", q_CW, q_policy, agent_parameters, env_parameters)
        q_CW.env.close()
        episode_length_q.append(len(q_tot_rewards))

        SARSA_policy = SARSA(agent_parameters)
        SARSA_tot_rewards = run_training("SARSA", sarsa_CW, SARSA_policy, agent_parameters, env_parameters)
        sarsa_CW.env.close()
        episode_length_sarsa.append(len(SARSA_tot_rewards))

        print(i)

    env_parameters = {
        'RenderMode': 'rgb_array',  # human, rgb_array
        'monitor': False,  # If rendermode = human then monitor must be False
        'STORE_PATH': STORE_PATH,
        'mean_number_ep': 100,
        'threshold_q': -13,
        'threshold_SARSA': -17,
        'replay_buffer_active': True,
        'replay_buffer_size': 1000,
        'batch_size': 32
    }


    episode_length_q_r = []
    episode_length_sarsa_r = []
    for i in range(500):
        q_policy = Q(agent_parameters)
        q_tot_rewards = run_training("Q", q_CW, q_policy, agent_parameters, env_parameters)
        q_CW.env.close()
        episode_length_q_r.append(len(q_tot_rewards))

        SARSA_policy = SARSA(agent_parameters)
        SARSA_tot_rewards = run_training("SARSA", sarsa_CW, SARSA_policy, agent_parameters, env_parameters)
        sarsa_CW.env.close()
        episode_length_sarsa_r.append(len(SARSA_tot_rewards))

        print(i)


    avg_eps_len_q = np.mean(episode_length_q)
    avg_eps_len_sarsa = np.mean(episode_length_sarsa)
    avg_eps_len_q_r = np.mean(episode_length_q_r)
    avg_eps_len_sarsa_r = np.mean(episode_length_sarsa_r)



    print(f"Average episode length Q: {avg_eps_len_q}")
    print(f"Average episode length SARSA: {avg_eps_len_sarsa}")
    print(f"Average episode length Q_r: {avg_eps_len_q_r}")
    print(f"Average episode length SARSA_r: {avg_eps_len_sarsa_r}")





