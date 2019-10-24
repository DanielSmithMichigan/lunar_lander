import math
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

class TestEmbedding(unittest.TestCase):
    # def test_sawtooth(self):
    #     la = LearningAgent(
    #         hyperparameters=hyperparameters,
    #         configuration=configuration,
    #         env_name="LunarLander-v2",
    #     )
    #     (
    #         quantile_thresholds_ph,
    #         inner_product,
    #         shaped_embedding,
    #         final_embedding,
    #     ) = la.learning_network.build_quantile_embedding(la.placeholders["quantile_thresholds"])
    #     la.sess.run(tf.global_variables_initializer())
    #     memories = build_memories(hyperparameters["batch_size"])
    #     (
    #         quantile_thresholds_ph_output,
    #         inner_product_output,
    #         shaped_embedding_output,
    #         final_embedding_output,
    #     ) = la.sess.run([
    #         quantile_thresholds_ph,
    #         inner_product,
    #         shaped_embedding,
    #         final_embedding,
    #     ], feed_dict=la.feed_dict_from_training_batch(memories))
    #     for batch_idx in range(hyperparameters["batch_size"]):
    #         for quantile_idx in range(hyperparameters["num_quantiles"]):
    #             for i_idx in range(hyperparameters["embedding_repeat"]):
    #                 np.testing.assert_equal(pow(2, i_idx) * quantile_thresholds_ph_output[batch_idx][quantile_idx][0], inner_product_output[batch_idx][quantile_idx][i_idx])
    #                 inner_product_output_i = inner_product_output[batch_idx][quantile_idx][i_idx]
    #                 shaped_embedding_output_i = inner_product_output_i - math.floor(inner_product_output_i)
    #                 shaped_embedding_output_i = 2.0 * (shaped_embedding_output_i - 0.5)
    #                 np.testing.assert_almost_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx], shaped_embedding_output_i)
    #                 np.testing.assert_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx] > -1.0, True)
    #                 np.testing.assert_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx] < 1.0, True)
    def test_cos(self):
        hp = hyperparameters.copy()
        hp["embedding_fn"] = "cos"
        la = LearningAgent(
            hyperparameters=hp,
            configuration=configuration,
            env_name="LunarLander-v2",
        )
        (
            quantile_thresholds_ph,
            inner_product,
            shaped_embedding,
            final_embedding,
        ) = la.learning_network.build_quantile_embedding(la.placeholders["quantile_thresholds"])
        la.sess.run(tf.global_variables_initializer())
        memories = build_memories(hyperparameters["batch_size"])
        (
            quantile_thresholds_ph_output,
            inner_product_output,
            shaped_embedding_output,
            final_embedding_output,
        ) = la.sess.run([
            quantile_thresholds_ph,
            inner_product,
            shaped_embedding,
            final_embedding,
        ], feed_dict=la.feed_dict_from_training_batch(memories))
        for batch_idx in range(hyperparameters["batch_size"]):
            for quantile_idx in range(hyperparameters["num_quantiles"]):
                for i_idx in range(hyperparameters["embedding_repeat"]):
                    np.testing.assert_almost_equal(i_idx * quantile_thresholds_ph_output[batch_idx][quantile_idx][0] * math.pi, inner_product_output[batch_idx][quantile_idx][i_idx], 5)
                    inner_product_output_i = inner_product_output[batch_idx][quantile_idx][i_idx]
                    np.testing.assert_almost_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx], math.cos(inner_product_output_i))
                    np.testing.assert_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx] >= -1, True)
                    np.testing.assert_equal(shaped_embedding_output[batch_idx][quantile_idx][i_idx] <= 1, True)