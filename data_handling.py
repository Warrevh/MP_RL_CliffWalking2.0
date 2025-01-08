import numpy as np
import matplotlib.pyplot as plt


class PLTLIB:
    def __init__(self,agent_par,store_path):
        self.store_path = store_path
        self.agent = agent_par
        self.color = 'blue'
        self.transparency = 0.5
        self.optimal_reward = -13

        self.color_rma = 'red'

        self.size_avg = 100

    def plot(self, rewards, title):
        n_it = np.arange(len(rewards))

        rewards_rma = self.get_rma(rewards)

        plt.plot(n_it, rewards, color= self.color,alpha= self.transparency, label='Return')
        plt.plot(n_it,rewards_rma, color= self.color_rma, label= 'Return_'+str(self.size_avg)+'_rma')

        plt.axhline(y=self.optimal_reward, color='black', linestyle=':', linewidth=2, label='Optimal return')

        plt.title(title)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cumulative reward (Return)')
        plt.ylim(bottom=-500)
        plt.legend()

        plt.text(0.95, 0.05,
                 f'Alpha:{self.agent['learning_rate_alpha']}\nGamma:{self.agent['gamma']}\nMin epsilon:{self.agent['epsilon_min']}\nIterations:{len(rewards)}'
                 , fontsize=10, ha='right', va='bottom', transform=plt.gca().transAxes)

        plt.savefig(self.store_path + f"/{title}.pdf")
        plt.show()

    def plot2(self, rewards_1, rewards_2, title):

        n_it_1 = np.arange(len(rewards_1))
        n_it_2 = np.arange(len(rewards_2))

        rewards_1_rma = self.get_rma(rewards_1)
        rewards_2_rma = self.get_rma(rewards_2)

        plt.plot(n_it_1, rewards_1_rma, color= 'blue', label='Q-policy')
        plt.plot(n_it_2,rewards_2_rma, color= 'red', label= 'SARSA-policy')

        plt.axhline(y=self.optimal_reward, color='black', linestyle=':', linewidth=2, label='Optimal return')

        plt.title(title)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cumulative reward (Return)')
        plt.ylim(bottom=-500)
        plt.legend()

        plt.text(0.95, 0.05,
                 f'Alpha:{self.agent['learning_rate_alpha']}\nGamma:{self.agent['gamma']}\nMin epsilon:{self.agent['epsilon_min']}'
                 , fontsize=10, ha='right', va='bottom', transform=plt.gca().transAxes)

        plt.savefig(self.store_path + f"/{title}.pdf")
        plt.show()

    def get_rma(self, rewards):
        rewards_rma = []
        for i in range(len(rewards)):
            if i < self.size_avg:
                if i == 0:
                    rewards_rma.append(rewards[i])
                else:
                    mean = np.mean(rewards[0:i])
                    rewards_rma.append(mean)
            else:
                mean = np.mean(rewards[i - self.size_avg:i])
                rewards_rma.append(mean)
        return rewards_rma

    def sub_plot(self, ax, rewards, title):
        n_it = np.arange(len(rewards))

        rewards_rma = self.get_rma(rewards)

        ax.plot(n_it, rewards, color= self.color,alpha= self.transparency, label='Return')
        ax.plot(n_it,rewards_rma, color= self.color_rma, label= 'Return_'+str(self.size_avg)+'_rma')

        ax.axhline(y=self.optimal_reward, color='black', linestyle=':', linewidth=2, label='Optimal return')

        ax.set_title(title)
        ax.set_xlabel('Number of episodes')
        ax.set_ylabel('Cumulative reward (Return)')
        ax.set_ylim(bottom=-500)
        ax.legend()

    def fig2(self, rewards_1, rewards_2):

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        # Use the function to plot on each subplot
        self.sub_plot(axs[0], rewards_1, "Q-learning without epsilon decay")
        self.sub_plot(axs[1], rewards_2, "Q-learning with epsilon decay")

        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()


class Txt_File:
    def __init__(self,store_path):
        self.store_path = store_path

        self.title = 'PARAMETERS Q-LEARNING ON CLIFFWALKING'
        self.subtitle1 = 'Parameters environment'
        self.subtitle2 = 'Parameters agent'

    def save_parameters(self, agent_par,env_par):
        file_path = self.store_path + f"/parameters.txt"

        with open(file_path, 'w') as f:
            f.write(f"{self.title}\n\n\n")
            f.write(f"{self.subtitle1}\n\n")
            for key, value in env_par.items():
                f.write(f"{key}: {value}\n")
            f.write(f"{self.subtitle2}\n\n")
            for key, value in agent_par.items():
                f.write(f"{key}: {value}\n")





