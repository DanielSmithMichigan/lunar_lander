import gym
import tensorflow as tf
import numpy as np
import random

from collections import deque

import constants
from q_network import QNetwork
from util import (
    generate_quantile_thresholds,
    current_epsilon,
    extract_data_for_placeholder,
)
from overview_graph import OverviewGraph

class LearningAgent:
    def __init__(
        self,
        hyperparameters,
        configuration,
        env_name,
    ):
        self.hyperparameters = hyperparameters
        self.configuration = configuration
        self.env = gym.make(env_name)
        self.environment_step = 0
        self.epsilon = 1.0
        self.sess = tf.Session()
        self.evaluations = []
        self.memory_buffer = deque([], maxlen=self.hyperparameters["max_memory_buffer_size"])
        self.build_graphs()
        self.build_placeholders()
        self.build_networks()
        self.build_training_operations()
        self.build_action_operations()
    def build_graphs(
        self
    ):
        self.overview_graph = OverviewGraph(
            env=self.env
        )
    def build_placeholders(
        self
    ):
        self.placeholders = {
            "state": tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0] + 1], "state_ph"),
            "next_state": tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0] + 1], "next_state_ph"),
            "actions": tf.placeholder(tf.int32, [None, 1], "actions_ph"),
            "rewards": tf.placeholder(tf.float32, [None, 1], "rewards_ph"),
            "terminals": tf.placeholder(tf.bool, [None, 1], "terminals_ph"),
            "quantile_thresholds": tf.placeholder(tf.float32, [None, self.hyperparameters["num_quantiles"]], "quantile_thresholds_ph"),
            "next_quantile_thresholds": tf.placeholder(tf.float32, [None, self.hyperparameters["num_quantiles"]], "next_quantile_thresholds_ph")
        }
    def build_networks(
        self
    ):
        self.learning_network = QNetwork(
            name=self.hyperparameters["agent_name"]+"_learning_q_network",
            env=self.env,
            hyperparameters=self.hyperparameters,
            placeholders=self.placeholders,
            sess=self.sess
        )
        self.target_network = QNetwork(
            name=self.hyperparameters["agent_name"]+"_target_q_network",
            env=self.env,
            hyperparameters=self.hyperparameters,
            placeholders=self.placeholders,
            sess=self.sess
        )
    def build_training_operations(
        self
    ):
        (
            _,
            learning_network_optimizer,
            learning_network_loss,
        ) = self.learning_network.build_training_operation()
        soft_copy_learning_network = self.target_network.weight_assignment_operation(
            self.learning_network,
            self.hyperparameters["tau"],
        )
        self.training_operations = [
            learning_network_optimizer,
            learning_network_loss,
            soft_copy_learning_network,
        ]
        self.hard_copy = self.target_network.weight_assignment_operation(
            self.learning_network,
            1.0
        )
    def build_action_operations(
        self
    ):
        (
            quantile_values,
            actions_chosen,
        ) = self.learning_network.build_network(
            self.placeholders["state"],
            self.placeholders["quantile_thresholds"],
        )
        self.action_operations = [
            quantile_values,
            actions_chosen,
        ]
    def get_action(
        self,
        state
    ):
        if len(self.memory_buffer) < self.hyperparameters["min_memory_buffer_size_for_training"]:
            return self.env.action_space.sample()
        if np.random.uniform() < self.epsilon:
            return self.env.action_space.sample()
        return self.get_best_action(state)
    def get_best_action(
        self,
        state
    ):
        (
            quantile_values,
            actions_chosen,
        ) = self.sess.run(
            self.action_operations,
            feed_dict={
                self.placeholders["state"]: [state],
                self.placeholders["quantile_thresholds"]: generate_quantile_thresholds(
                    hyperparameters=self.hyperparameters,
                    just_one=True
                )
            }
        )
        return actions_chosen[0]
    def episode(
        self,
        evaluative,
        disable_random_actions
    ):
        current_state = self.env.reset()
        total_reward = 0
        if self.configuration["render"] and evaluative:
            self.env.render()
        step_idx = 0
        current_state = np.append(current_state, [step_idx / self.hyperparameters["max_episode_length"]])
        while(True):
            step_idx += 1
            if not evaluative:
                self.environment_step += 1
                if self.environment_step % self.hyperparameters["environment_steps_per_training_step"] == 0:
                    if len(self.memory_buffer) > self.hyperparameters["min_memory_buffer_size_for_training"]:
                        self.training_step()
            action_chosen = self.get_action(current_state) if (not evaluative and not disable_random_actions) else self.get_best_action(current_state)
            # if evaluative:
            self.overview_graph.record_action(action_chosen)
            next_state, reward, is_terminal, info = self.env.step(action_chosen)
            next_state = np.append(next_state, [step_idx / self.hyperparameters["max_episode_length"]])
            if step_idx >= self.hyperparameters["max_episode_length"]:
                is_terminal = True
            total_reward += reward
            memory_entry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
            memory_entry[constants.STATE] = current_state
            memory_entry[constants.ACTION] = action_chosen
            memory_entry[constants.REWARD] = reward
            memory_entry[constants.NEXT_STATE] = next_state
            memory_entry[constants.IS_TERMINAL] = is_terminal
            self.memory_buffer.append(memory_entry)
            current_state = next_state
            if self.configuration["render"] and evaluative:
                self.env.render()
            if is_terminal:
                break
        return total_reward
    def feed_dict_from_training_batch(
        self,
        training_batch
    ):
        return {
            self.placeholders["state"]: extract_data_for_placeholder(training_batch,constants.STATE,self.placeholders["state"]),
            self.placeholders["next_state"]: extract_data_for_placeholder(training_batch,constants.NEXT_STATE,self.placeholders["next_state"]),
            self.placeholders["actions"]: extract_data_for_placeholder(training_batch,constants.ACTION,self.placeholders["actions"]),
            self.placeholders["rewards"]: extract_data_for_placeholder(training_batch,constants.REWARD,self.placeholders["rewards"]),
            self.placeholders["terminals"]: extract_data_for_placeholder(training_batch,constants.IS_TERMINAL,self.placeholders["terminals"]),
            self.placeholders["quantile_thresholds"]: generate_quantile_thresholds(
                    hyperparameters=self.hyperparameters,
            ),
            self.placeholders["next_quantile_thresholds"]: generate_quantile_thresholds(
                    hyperparameters=self.hyperparameters,
            ),
        }
    def training_step(
        self
    ):
        training_idx = np.random.randint(len(self.memory_buffer), size=self.hyperparameters["batch_size"])
        training_batch = [self.memory_buffer[idx] for idx in training_idx]
        (
            learning_network_optimizer_output,
            learning_network_loss_output,
            soft_copy_learning_network_output,
        ) = self.sess.run(self.training_operations, feed_dict=self.feed_dict_from_training_batch(training_batch))
    def execute(
        self
    ):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.hard_copy)
        for episode_idx in range(self.hyperparameters["max_episodes"]):
            if self.configuration["graph"]:
                self.overview_graph.init_episode()
            self.episode(
                evaluative=False,
                disable_random_actions=False,
            )
            reward = self.episode(
                evaluative=True,
                disable_random_actions=True,
            )
            self.evaluations.append(reward)
            print("ep: "+str(episode_idx) + " reward: "+str(self.overview_graph.calc_mean_reward()))
            if self.configuration["graph"]:
                self.overview_graph.end_episode(self.epsilon, reward)
                self.overview_graph.update_and_display()
            self.epsilon = self.epsilon * self.hyperparameters["epsilon_decay"]
        return self.evaluations