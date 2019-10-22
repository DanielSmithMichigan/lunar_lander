import numpy as np

from learning_agent import LearningAgent
from default_hyperparameters import hyperparameters

configuration = {
    "render": False,
    "graph": True
}

la = LearningAgent(
    hyperparameters=hyperparameters,
    configuration=configuration,
    env_name="LunarLander-v2",
)

la.execute()