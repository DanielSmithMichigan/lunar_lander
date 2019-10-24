import MySQLdb
import os
import numpy as np

from learning_agent import LearningAgent
from default_hyperparameters import hyperparameters

configuration = {
    "render": False,
    "graph": False
}

hyperparameters["embedding_fn"] = os.environ['EMBEDDING_FN']
hyperparameters["embedding_repeat"] = int(os.environ['EMBEDDING_REPEAT'])
experiment_name = "lunar_lander_" + os.environ['EMBEDDING_FN'] + "_" + os.environ['EMBEDDING_REPEAT']

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
