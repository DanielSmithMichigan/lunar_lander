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
            embedding_step_size,
            unbounded_embedding,
            bounded_embedding,
            final_embedding,
        ) = la.learning_network.build_quantile_embedding(la.placeholders["quantile_thresholds"])
        la.sess.run(tf.global_variables_initializer())
        memories = build_memories(hyperparameters["batch_size"])
        (
            embedding_step_size,
            unbounded_embedding,
            bounded_embedding,
            final_embedding,
        ) = la.sess.run([
            embedding_step_size,
            unbounded_embedding,
            bounded_embedding,
            final_embedding,
        ], feed_dict=la.feed_dict_from_training_batch(memories))
        # np.testing.assert_almost_equal(
        #     loss_output,
        #     np.mean(np.mean(np.mean(rho_output, axis=2), axis=1), axis=0),
        #     decimal=decimal_precision
        # )