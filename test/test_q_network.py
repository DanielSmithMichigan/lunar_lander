import numpy as np
import tensorflow as tf
import unittest

from learning_agent import LearningAgent
from default_hyperparameters import hyperparameters
import constants

configuration = {
    "render": False,
    "graph": False
}

decimal_precision = 7

def build_memories(num_memories):
    return [build_memory() for i in range(num_memories)]

def build_memory():
    memory_entry = np.array(np.zeros(constants.NUM_MEMORY_ENTRIES), dtype=object)
    memory_entry[constants.STATE] = np.random.uniform(size=(9,))
    memory_entry[constants.ACTION] = np.random.randint(low=0, high=4)
    memory_entry[constants.REWARD] = np.random.uniform()
    memory_entry[constants.NEXT_STATE] = np.random.uniform(size=(9,))
    memory_entry[constants.IS_TERMINAL] = np.random.randint(low=0, high=2)
    return memory_entry

class TestNetwork(unittest.TestCase):
    def test_training_operation(self):
        la = LearningAgent(
            hyperparameters=hyperparameters,
            configuration=configuration,
            env_name="LunarLander-v2",
        )
        (
            [
                quantile_values_next_state,
                best_action_next_state,
                quantile_values_best_action_next_state,
                targets,
                quantile_values_current_state,
                quantile_values_chosen_action,
                predictions,
                td_error,
                l_sub_one,
                l_sub_two,
                total_error,
                below_quantile,
                quantile_thresholds,
                rho,
            ],
            optimizer,
            loss
        ) = la.learning_network.build_training_operation()
        la.sess.run(tf.global_variables_initializer())
        memories = build_memories(hyperparameters["batch_size"])
        (
            quantile_values_next_state_output,
            best_action_next_state_output,
            quantile_values_best_action_next_state_output,
            targets_output,
            quantile_values_current_state_output,
            quantile_values_chosen_action_output,
            predictions_output,
            td_error_output,
            l_sub_one_output,
            l_sub_two_output,
            total_error_output,
            below_quantile_output,
            quantile_thresholds_output,
            rho_output,
            loss_output,
        ) = la.sess.run([
            quantile_values_next_state,
            best_action_next_state,
            quantile_values_best_action_next_state,
            targets,
            quantile_values_current_state,
            quantile_values_chosen_action,
            predictions,
            td_error,
            l_sub_one,
            l_sub_two,
            total_error,
            below_quantile,
            quantile_thresholds,
            rho,
            loss,
        ], feed_dict=la.feed_dict_from_training_batch(memories))
        for batch_idx in range(hyperparameters["batch_size"]):
            for quantile_idx in range(hyperparameters["num_quantiles"]):
                np.testing.assert_equal(
                    quantile_values_best_action_next_state_output[batch_idx][quantile_idx],
                    quantile_values_next_state_output[batch_idx][quantile_idx][
                        best_action_next_state_output[batch_idx]
                    ]
                )
                np.testing.assert_equal(
                    quantile_values_chosen_action_output[batch_idx][quantile_idx],
                    quantile_values_current_state_output[batch_idx][quantile_idx][
                        memories[batch_idx][constants.ACTION]
                    ]
                )
                np.testing.assert_almost_equal(
                    targets_output[batch_idx][0][quantile_idx],
                    memories[batch_idx][constants.REWARD] + (1.0 - memories[batch_idx][constants.IS_TERMINAL]) * hyperparameters["gamma"] * quantile_values_best_action_next_state_output[batch_idx][quantile_idx],
                    decimal=decimal_precision
                )
                for quantile_idx_2 in range(hyperparameters["num_quantiles"]):
                    np.testing.assert_almost_equal(
                        td_error_output[batch_idx][quantile_idx][quantile_idx_2],
                        abs(targets_output[batch_idx][0][quantile_idx_2] - predictions_output[batch_idx][quantile_idx][0]),
                        decimal=decimal_precision
                    )
                    if(td_error_output[batch_idx][quantile_idx][quantile_idx_2] <= hyperparameters["kappa"]):
                        np.testing.assert_almost_equal(
                            .5 * td_error_output[batch_idx][quantile_idx][quantile_idx_2] ** 2,
                            l_sub_one_output[batch_idx][quantile_idx][quantile_idx_2],
                            decimal=decimal_precision
                        )
                        np.testing.assert_almost_equal(
                            0,
                            l_sub_two_output[batch_idx][quantile_idx][quantile_idx_2],
                            decimal=decimal_precision
                        )
                    else:
                        np.testing.assert_almost_equal(
                            0,
                            l_sub_one_output[batch_idx][quantile_idx][quantile_idx_2],
                            decimal=decimal_precision
                        )
                        np.testing.assert_almost_equal(
                            (abs(td_error_output[batch_idx][quantile_idx][quantile_idx_2]) - .5 * hyperparameters["kappa"]) * hyperparameters["kappa"],
                            l_sub_two_output[batch_idx][quantile_idx][quantile_idx_2],
                            decimal=decimal_precision
                        )
                    np.testing.assert_almost_equal(
                        total_error_output[batch_idx][quantile_idx][quantile_idx_2],
                        l_sub_one_output[batch_idx][quantile_idx][quantile_idx_2] + l_sub_two_output[batch_idx][quantile_idx][quantile_idx_2],
                        decimal=decimal_precision
                    )
                    error_scale_curr = quantile_thresholds_output[batch_idx][quantile_idx][0]
                    if predictions_output[batch_idx][quantile_idx][0] > targets_output[batch_idx][0][quantile_idx_2]:
                        error_scale_curr = 1.0 - error_scale_curr
                    np.testing.assert_almost_equal(
                        rho_output[batch_idx][quantile_idx][quantile_idx_2],
                        error_scale_curr * total_error_output[batch_idx][quantile_idx][quantile_idx_2] / hyperparameters["kappa"],
                        decimal=decimal_precision
                    )
        np.testing.assert_almost_equal(
            loss_output,
            np.mean(np.mean(np.mean(rho_output, axis=2), axis=1), axis=0),
            decimal=decimal_precision
        )

        
