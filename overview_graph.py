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
        self.overview = plt.figure()
        self.env = env
        self.window = 100
        self.desired_reward = 200

        self.rewards_over_time = []
        self.rewards_moving_average = []
        self.rewards_over_time_graph = self.overview.add_subplot(2, 1, 1)

        self.actions_over_time = []
        self.epsilon_over_time = []
        self.actions_during_episode = None
        self.actions_over_time_graph = self.overview.add_subplot(2, 1, 2)
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
        window = self.rewards_over_time[-self.window:]
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

        self.rewards_over_time.append(reward)
        self.rewards_moving_average.append(self.calc_mean_reward())
    def update_and_display(
        self,
    ):
        self.rewards_over_time_graph.cla()
        self.rewards_over_time_graph.plot(self.rewards_over_time, label="Reward")
        self.rewards_over_time_graph.plot(self.rewards_moving_average, label="Moving average")
        self.rewards_over_time_graph.axhline(y=self.desired_reward, label="Goal reward")
        self.rewards_over_time_graph.legend(loc=2)

        self.actions_over_time_graph.cla()
        for action_idx in range(self.env.action_space.n):
            self.actions_over_time_graph.plot([episode[action_idx] for episode in self.actions_over_time], label=constants.ACTION_NAMES[action_idx])
        self.actions_over_time_graph.plot(self.epsilon_over_time, label="Epsilon", linestyle=":")
        self.actions_over_time_graph.legend(loc=2)

        self.overview.canvas.draw()
        plt.pause(0.00001)