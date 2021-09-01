import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
import scipy.signal




class PPO_agent:
    def __init__(
        self,
        size,
        n_jobs,
        n_machines,
        n_types,
        n_actions,
        a_lr = 3e-4,
        c_lr = 1e-3,
        gamma = 0.99,
        clip_ratio = 0.2,
        train_policy_iterations = 80,
        train_value_iterations = 80,
        target_kl = 0.01,
        lam = 0.97,
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_types = n_types
        self.n_actions = n_actions
        self.gamma, self.lam = gamma, lam
        self.clip_ratio = clip_ratio
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.target_kl = target_kl
        self.pointer, self.trajectory_start_index = 0, 0
        #buffer
        self.obs_m1_buffer = np.zeros((size, n_jobs, 6, 1), dtype=np.float32)
        self.obs_m2_buffer = np.zeros((size, n_types, n_types, 1), dtype=np.float32)
        self.obs_m3_buffer = np.zeros((size, n_machines, 4, 1), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        #network
        self.policy_optimizer = keras.optimizers.Adam(learning_rate = a_lr)
        self.value_optimizer = keras.optimizers.Adam(learning_rate = c_lr)
        
        self.initialization()

        
        
    def initialization(self):
        self.actor = self._build_net(self.n_actions)
        self.critic = self._build_net(1)

        #self.actor.summary()
        #self.critic.summary()

    def _build_net(self, ouput_dims):
        input_m1 = keras.Input(shape=(self.n_jobs, 6, 1), dtype=tf.float32)
        input_m2 = keras.Input(shape=(self.n_types, self.n_types, 1), dtype=tf.float32)
        input_m3 = keras.Input(shape=(self.n_machines, 4, 1), dtype=tf.float32)

        layer1 = layers.Conv2D(16, 1, strides=3, activation="relu")(input_m1)
        layer1 = layers.Conv2D(16, 1, strides=3, activation="relu")(layer1)
        layer1 = layers.Flatten()(layer1)

        layer2 = layers.Conv2D(16, 1, strides=3, activation="relu")(input_m2)
        layer2 = layers.Conv2D(16, 1, strides=3, activation="relu")(layer2)
        layer2 = layers.Flatten()(layer2)

        layer3 = layers.Conv2D(16, 1, strides=3, activation="relu")(input_m3)
        layer3 = layers.Conv2D(16, 1, strides=3, activation="relu")(layer3)
        layer3 = layers.Flatten()(layer3)

        combined = layers.concatenate([layer1, layer2, layer3])
        common = layers.Dense(units = 64, activation = 'relu')(combined)
        common = layers.Dense(units = 64, activation = 'relu')(common)
        outputs = layers.Dense(units = ouput_dims)(common)

        return keras.Model(inputs = [input_m1, input_m2, input_m3], outputs = outputs)



    def store(self, obs_m1, obs_m2, obs_m3, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.obs_m1_buffer[self.pointer] = obs_m1
        self.obs_m2_buffer[self.pointer] = obs_m2
        self.obs_m3_buffer[self.pointer] = obs_m3

        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.n_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    @tf.function
    def sample_action(self, obs_m1, obs_m2, obs_m3):
        logits = self.actor([obs_m1, obs_m2, obs_m3])
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action
    
    
    @tf.function
    def choose_action(self, obs_m1, obs_m2, obs_m3):
        logits = self.actor([obs_m1, obs_m2, obs_m3])
        logprobabilities_all = tf.nn.log_softmax(logits)
        prob = tf.squeeze(tf.exp(logprobabilities_all))
        return tf.math.argmax(prob)
    

    @tf.function()
    def train_actor(self, obs_m1_buffer, obs_m2_buffer, obs_m3_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:
            ratio = tf.exp(
                self.logprobabilities(self.actor([obs_m1_buffer, obs_m2_buffer, obs_m3_buffer]), action_buffer)
                - logprobability_buffer
            )

            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        kl_divergence = tf.reduce_mean(
            logprobability_buffer - self.logprobabilities(self.actor([obs_m1_buffer, obs_m2_buffer, obs_m3_buffer]), action_buffer)
        )
        kl = tf.reduce_sum(kl_divergence)
        return kl

    @tf.function()
    def train_critic(self, obs_m1_buffer, obs_m2_buffer, obs_m3_buffer, return_buffer):
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean((return_buffer - self.critic([obs_m1_buffer, obs_m2_buffer, obs_m3_buffer])) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def normalize(self, x):
        eps = np.finfo(np.float32).eps.item()
        mean, std = (np.mean(x), np.std(x))
        x = (x - mean)/(std + eps)

        return x

    def train(self):
        self.pointer, self.trajectory_start_index = 0, 0
        self.advantage_buffer = self.normalize(self.advantage_buffer)


        for _ in range(self.train_policy_iterations):
            kl = self.train_actor(
                self.obs_m1_buffer, self.obs_m2_buffer, self.obs_m3_buffer, self.action_buffer, self.logprobability_buffer, self.advantage_buffer
            )
            if kl > 1.5 * self.target_kl:
                break

        for _ in range(self.train_value_iterations):
            self.train_critic(
                self.obs_m1_buffer, self.obs_m2_buffer, self.obs_m3_buffer, self.return_buffer
            )