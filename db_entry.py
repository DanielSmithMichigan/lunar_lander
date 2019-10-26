import MySQLdb
import os
import numpy as np

from learning_agent import LearningAgent
from default_hyperparameters import hyperparameters

configuration = {
    "render": False,
    "graph": False
}

num_quantiles = 8
if "NUM_QUANTILES" in os.environ:
    num_quantiles = int(num_quantiles)

if num_quantiles != 8:
    hyperparameters["epsilon_decay"] = 0.995
    hyperparameters["epsilon_multiplier_pct"] = 0.0
    hyperparameters["num_quantiles"] = num_quantiles
    experiment_name = "lunar_lander_quantile_" + str(num_quantiles)
elif "QUANTILE_THRESHOLD_LOW" in os.environ:
    hyperparameters["quantile_threshold_low"] = float(os.environ["QUANTILE_THRESHOLD_LOW"])
    hyperparameters["quantile_threshold_high"] = float(os.environ["QUANTILE_THRESHOLD_HIGH"])
    hyperparameters["epsilon_decay"] = 0.995
    hyperparameters["epsilon_multiplier_pct"] = 0.0
    experiment_name = "lunar_lander_thresholds_"+str(hyperparameters["quantile_threshold_low"])+"_"+str(hyperparameters["quantile_threshold_high"])
else:
    experiment_name = "lunar_lander_new_exploration"

la = LearningAgent(
    hyperparameters=hyperparameters,
    configuration=configuration,
    env_name="LunarLander-v2",
)

evaluations = la.execute()

db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="dsmith682101", passwd=os.environ['MYSQL_PASS'], db="dqn_results")

for evaluation, evaluation_idx in zip(evaluations, range(len(evaluations))):
    cur = db.cursor()
    cur.execute("insert into experiments (label, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, y, checkpoint, trainingSteps, agent_name) values ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}', '{8}', '{9}', '{10}', '{11}', '{12}', '{13}', '{14}')".format(
            experiment_name,
            evaluation,
            evaluation_idx,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "checkpoint_"+str(evaluation_idx),
            0,
            "no_agent_name"
        )
    )
    db.commit()

cur.close()
db.close()
