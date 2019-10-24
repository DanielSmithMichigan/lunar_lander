import tensorflow as tf
import numpy as np
average_episode_len = 80
hyperparameters = {
    "agent_name": "agent_"+str(np.random.randint(low=100000,high=999999)),
    "environment_embedding_network_layers": [{
        "size": 128
    },{
        "size": 128
    }],
    "output_layers": [{
        "size": 256
    }],
    "embedding_repeat": 4,
    "embedding_fn": "sawtooth",
    "layer_activation": tf.nn.leaky_relu,
    "gamma": 0.99,
    "tau": 0.001,
    "max_episodes": 1200,
    "max_episode_length": 1024,
    "batch_size": 64,
    "epsilon_shape": [[0, 1.0], [average_episode_len * 1300, 0.01]],
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "environment_steps_per_training_step": 4,
    "min_memory_buffer_size_for_training": 2048,
    "max_memory_buffer_size": int(1e6),
    "num_quantiles": 8,
    "kappa": 1.0,
    "learning_rate": 5e-4,
    "max_gradient_norm": 5.0,
}