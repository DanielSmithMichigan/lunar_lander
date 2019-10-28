import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ion()

import constants


class OverviewGraph:
    def __init__(
        self,
        env,
    ):
        self.env = env
        self.window = 100
        self.desired_reward = 200

        self.evaluative_rewards_fig = plt.figure()
        self.evaluative_rewards_over_time = []
        self.evaluative_rewards_moving_average = []
        self.evaluative_rewards_over_time_graph = self.evaluative_rewards_fig.add_subplot(1, 1, 1)

        self.training_rewards_fig = plt.figure()
        self.training_rewards_over_time = []
        self.training_rewards_over_time_graph = self.training_rewards_fig.add_subplot(1, 1, 1)

        self.actions_fig = plt.figure()
        self.actions_over_time = []
        self.epsilon_over_time = []
        self.actions_during_episode = None
        self.actions_over_time_graph = self.actions_fig.add_subplot(1, 1, 1)
    def init_episode(
        self,
    ):
        self.actions_during_episode = np.zeros(shape=(self.env.action_space.n,))
    def record_action(
        self,
        action,
    ):
        self.actions_during_episode[action] += 1
    def calc_mean_reward(
        self
    ):
        window = self.evaluative_rewards_over_time[-self.window:]
        mean_reward = np.mean(window)
        return mean_reward
    def end_episode(
        self,
        epsilon,
        reward,
    ):
        self.actions_over_time.append(
            [action_count / np.sum(self.actions_during_episode) for action_count in self.actions_during_episode]
        )
        self.epsilon_over_time.append(epsilon)

        self.evaluative_rewards_over_time.append(reward)
        self.evaluative_rewards_moving_average.append(self.calc_mean_reward())
    def record_training_reward(
        self,
        reward,
    ):
        self.training_rewards_over_time.append(reward)
    def update_and_display(
        self,
    ):
        self.evaluative_rewards_over_time_graph.cla()
        self.evaluative_rewards_over_time_graph.plot(self.evaluative_rewards_over_time, label="Reward")
        self.evaluative_rewards_over_time_graph.plot(self.evaluative_rewards_moving_average, label="Moving average")
        self.evaluative_rewards_over_time_graph.axhline(y=self.desired_reward, label="Goal reward")
        self.evaluative_rewards_over_time_graph.set_xlabel("Episode Number")
        self.evaluative_rewards_over_time_graph.set_ylabel("Cumulative Reward")
        self.evaluative_rewards_over_time_graph.legend(loc=2)

        self.training_rewards_over_time_graph.cla()
        self.training_rewards_over_time_graph.plot(self.training_rewards_over_time, label="Reward")
        self.training_rewards_over_time_graph.axhline(y=self.desired_reward, label="Goal reward")
        self.training_rewards_over_time_graph.set_xlabel("Episode Number")
        self.training_rewards_over_time_graph.set_ylabel("Cumulative Reward")
        self.training_rewards_over_time_graph.legend(loc=2)

        self.actions_over_time_graph.cla()
        for action_idx in range(self.env.action_space.n):
            self.actions_over_time_graph.plot([episode[action_idx] for episode in self.actions_over_time], label=constants.ACTION_NAMES[action_idx])
        self.actions_over_time_graph.plot(self.epsilon_over_time, label="Epsilon", linestyle=":")
        self.actions_over_time_graph.legend(loc=2)

        self.evaluative_rewards_fig.suptitle('Evaluation performance', fontsize=16)
        self.evaluative_rewards_fig.canvas.draw()
        self.training_rewards_fig.suptitle('Training episode performance', fontsize=16)
        self.training_rewards_fig.canvas.draw()
        self.actions_fig.canvas.draw()
        plt.pause(0.00001)
