import tensorflow as tf
import math

class QNetwork:
    def __init__(
        self,
        sess,
        name,
        env,
        hyperparameters,
        placeholders,
    ):
        self.sess = sess
        self.name = name
        self.env = env
        self.hyperparameters = hyperparameters
        self.placeholders = placeholders
    def build_network(
        self,
        state_ph,
        quantile_thresholds_ph,
    ):
        with tf.variable_scope(self.name):
            prev_layer = state_ph
            for layer_definition, layer_idx in zip(self.hyperparameters["environment_embedding_network_layers"], range(len(self.hyperparameters["environment_embedding_network_layers"]))):
                prev_layer = tf.layers.dense(
                    inputs=prev_layer,
                    units=layer_definition["size"],
                    activation=self.hyperparameters["layer_activation"],
                    name="q_network_environment_embedding_layer_" + str(layer_idx),
                    reuse=tf.AUTO_REUSE
                )
            # [batch_size, n]
            prev_layer = tf.reshape(prev_layer, [-1, 1, self.hyperparameters["environment_embedding_network_layers"][-1]["size"]])
            # [batch_size, n, 1]
            (
                _,
                _,
                _,
                final_embedding,
            ) = self.build_quantile_embedding(
                quantile_thresholds_ph
            )
            prev_layer = prev_layer * final_embedding
            for layer_definition, layer_idx in zip(self.hyperparameters["output_layers"], range(len(self.hyperparameters["output_layers"]))):
                prev_layer = tf.layers.dense(
                    inputs=prev_layer,
                    units=layer_definition["size"],
                    activation=self.hyperparameters["layer_activation"],
                    name="q_network_output_layer_" + str(layer_idx),
                    reuse=tf.AUTO_REUSE
                )
            # [batch_size, num_quantiles, n]
            quantile_values = tf.layers.dense(
                inputs=prev_layer,
                units=self.env.action_space.n,
                name="q_value",
                reuse=tf.AUTO_REUSE
            )
            # [batch_size, num_quantiles, action_space]
            q_values = tf.reduce_mean(
                quantile_values,
                axis=1
            )
            # [batch_size, action_space]
            best_action = tf.math.argmax(
                q_values,
                axis=1
            )
            best_action = tf.cast(best_action, tf.int32)
            # [batch_size, ]
            return (
                quantile_values,
                best_action
            )
    def build_quantile_embedding(
        self,
        quantile_thresholds_ph
    ):
        quantile_thresholds_ph = tf.reshape(quantile_thresholds_ph, [-1, self.hyperparameters["num_quantiles"], 1])
        # [batch_size, num_thresholds, 1]
        embedding_step_size = tf.cast(tf.range(self.hyperparameters["embedding_repeat"]), tf.float32)
        if self.hyperparameters["embedding_step_type"] == "exponential":
            embedding_step_size = tf.pow(2.0, embedding_step_size)
        elif self.hyperparameters["embedding_step_type"] == "multiplicative":
            embedding_step_size = 2.0 * embedding_step_size
        embedding_step_size = tf.reshape(embedding_step_size, [1, 1, self.hyperparameters["embedding_repeat"]])
        unbounded_embedding = quantile_thresholds_ph * embedding_step_size
        bounded_embedding = unbounded_embedding
        if self.hyperparameters["embedding_fn"] == "cos":
            bounded_embedding = tf.math.cos(bounded_embedding * math.pi)
        elif self.hyperparameters["embedding_fn"] == "sawtooth":
            bounded_embedding = bounded_embedding - tf.math.floor(bounded_embedding)
        final_embedding = tf.layers.dense(
            inputs=bounded_embedding,
            units=self.hyperparameters["environment_embedding_network_layers"][-1]["size"],
            activation=self.hyperparameters["layer_activation"],
            name="quantile_embedding",
            reuse=tf.AUTO_REUSE,
        )
        return (
            embedding_step_size,
            unbounded_embedding,
            bounded_embedding,
            final_embedding,
        )
    def build_training_operation(self):
        (
            quantile_values_next_state,
            best_action_next_state,
        ) = self.build_network(
            self.placeholders["next_state"],
            self.placeholders["next_quantile_thresholds"],
        )
        batch_size = tf.shape(quantile_values_next_state)[0]
        batch_idx = tf.cast(tf.range(batch_size), tf.int32)
        quantile_values_best_action_next_state = tf.gather_nd(
            tf.transpose(quantile_values_next_state, perm=[0, 2, 1]),
            tf.stack(
                [
                    batch_idx,
                    best_action_next_state
                ],
                axis=1
            )
        )
        # [batch_size, num_quantiles]
        targets = self.placeholders["rewards"] + self.hyperparameters["gamma"] * (1.0 - tf.cast(self.placeholders["terminals"], tf.float32)) * tf.stop_gradient(quantile_values_best_action_next_state)
        # [batch_size, num_quantiles]
        targets = tf.reshape(targets, [-1, 1, self.hyperparameters["num_quantiles"]])
        # [batch_size, 1, num_quantiles]
        (
            quantile_values_current_state,
            _,
        ) = self.build_network(
            self.placeholders["state"],
            self.placeholders["quantile_thresholds"],
        )
        quantile_values_chosen_action = tf.gather_nd(
            tf.transpose(quantile_values_current_state, perm=[0, 2, 1]),
            tf.stack(
                [
                    batch_idx,
                    tf.reshape(self.placeholders["actions"], [-1])
                ],
                axis=1
            )
        )
        # [batch_size, num_quantiles]
        predictions = tf.reshape(quantile_values_chosen_action, [-1, self.hyperparameters["num_quantiles"], 1])
        # [batch_size, num_quantiles, 1]
        td_error = tf.abs(targets - predictions)
        # [batch_size, num_quantiles, num_quantiles]
        l_sub_one = tf.stop_gradient(tf.cast(td_error <= self.hyperparameters["kappa"], tf.float32)) * 0.5 * td_error ** 2
        # [batch_size, num_quantiles, num_quantiles]
        l_sub_two = tf.stop_gradient(tf.cast(td_error > self.hyperparameters["kappa"], tf.float32)) * self.hyperparameters["kappa"] * (td_error - 0.5 * self.hyperparameters["kappa"])
        # [batch_size, num_quantiles, num_quantiles]
        total_error = l_sub_one + l_sub_two
        # [batch_size, num_quantiles, num_quantiles]
        below_quantile = tf.stop_gradient(tf.to_float(targets < predictions))
        quantile_thresholds = tf.reshape(self.placeholders["quantile_thresholds"], [-1, self.hyperparameters["num_quantiles"], 1])
        # [batch_size, num_quantiles, num_quantiles]
        error_scale = tf.abs(quantile_thresholds - below_quantile)
        # [batch_size, num_quantiles, num_quantiles]
        rho = error_scale * total_error / self.hyperparameters["kappa"]
        # [batch_size, num_quantiles, num_quantiles]
        per_quantile_loss = tf.reduce_mean(rho, axis=2)
        # [batch_size, num_quantiles]
        batch_loss = tf.reduce_mean(per_quantile_loss, axis=1)
        # [batch_size]
        loss = tf.reduce_mean(batch_loss)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.hyperparameters["learning_rate"])
            gradients, variables = zip(
                *optimizer.compute_gradients(
                    loss,
                    var_list=tf.trainable_variables(scope=self.name)
                )
            )
            (
                clipped_gradients,
                reg_term
            ) = tf.clip_by_global_norm(gradients, self.hyperparameters["max_gradient_norm"])
            optimizer = optimizer.apply_gradients(zip(clipped_gradients, variables))
        return (
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
        )
    def weight_assignment_operation(self, copy_weights_from, tau):
        return [tf.assign(target, (1 - tau) * target + tau * source) for target, source in zip(
            tf.trainable_variables(
                scope=self.name
            ),
            tf.trainable_variables(
                scope=copy_weights_from.name
            )
        )]


