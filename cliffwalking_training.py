import numpy as np

def run_training(agent_type, cw, policy, agent_par, env_par):

    if agent_type == "Q":
        threshold = env_par['threshold_q']

    elif agent_type == "SARSA":
        threshold = env_par['threshold_SARSA']

    else:
        threshold = -13


    tot_rewards = []
    mean_ep = -10000000
    for episode in range(agent_par['num_episodes']):
        states, actions, rewards, new_states, dones = cw.one_rollout(policy)

        tot_reward = sum(rewards)
        tot_rewards.append(tot_reward)

        if episode < env_par['mean_number_ep']:
            if episode == 0:
                mean_ep = tot_reward
            else:
                mean_ep = np.mean(tot_rewards[0:episode])
        else:
            mean_ep = np.mean(tot_rewards[-env_par['mean_number_ep'] - 1:-1])

        print(f"Episode {agent_type}: {episode}, Reward: {tot_reward}, Mean of 100 cons episodes: {mean_ep}")

        if mean_ep >= threshold:
            print(f"Problem is  solved.")

            break
    if mean_ep >= threshold:
        print(f"\n\nProblem is solved after {len(tot_rewards)} "
              f"Episode with the mean reward {mean_ep} over the last {env_par['mean_number_ep']} episodes")

    return tot_rewards



