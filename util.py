import numpy as np

def generate_quantile_thresholds(hyperparameters, just_one=False):
    thresh = np.random.uniform(
        size=(
            1 if just_one else hyperparameters["batch_size"],
            hyperparameters["num_quantiles"]
        )
    )
    return thresh

def current_epsilon(
    hyperparameters,
    environment_step,
):
    y = 1
    x = 0
    slope = (hyperparameters["epsilon_shape"][1][y] - hyperparameters["epsilon_shape"][0][y]) / (hyperparameters["epsilon_shape"][1][x] - hyperparameters["epsilon_shape"][0][x])
    steps_in_descent = environment_step - hyperparameters["epsilon_shape"][0][x]
    return np.clip(steps_in_descent * slope + hyperparameters["epsilon_shape"][0][y], hyperparameters["epsilon_shape"][1][y], hyperparameters["epsilon_shape"][0][y])

def extract_data_for_placeholder(
    training_batch,
    col_idx,
    placeholder
):
    training_data = [row[col_idx] for row in training_batch]
    desired_shape = placeholder.get_shape().as_list()
    desired_shape[0] = -1
    return np.reshape(training_data, desired_shape)