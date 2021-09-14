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

        layer1 = layers.Conv2D(16, 3, strides=1, activation="tanh")(input_m1)
        layer1 = layers.Conv2D(16, 3, strides=1, activation="tanh")(layer1)
        layer1 = layers.Flatten()(layer1)

        layer2 = layers.Conv2D(16, 3, strides=1, activation="tanh")(input_m2)
        layer2 = layers.Conv2D(16, 3, strides=1, activation="tanh")(layer2)
        layer2 = layers.Flatten()(layer2)

        layer3 = layers.Conv2D(16, 2, strides=1, activation="tanh")(input_m3)
        layer3 = layers.Conv2D(16, 2, strides=1, activation="tanh")(layer3)
        layer3 = layers.Flatten()(layer3)

        combined = layers.concatenate([layer1, layer2, layer3])
        common = layers.Dense(units = 64, activation = 'tanh')(combined)
        common = layers.Dense(units = 64, activation = 'tanh')(common)
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
            

class DQL_agent:
    def __init__(self, n_jobs, n_machines, n_actions, n_types):
        self.n_jobs = n_jobs
        self.n_actions = n_actions
        self.n_machines = n_machines
        self.n_types = n_types
        # q_network for choossing action
        # target_q_network for prediction of future rewards
        # build q_network and target_q_network
        self._build_q_network()
        self.optimizer = keras.optimizers.Adam(learning_rate = 1e-3, clipnorm=1.0)
        self.batch_size = 32
        self.action_buffer = []
        self.s_m1_buffer = []
        self.s_m2_buffer = []
        self.s_m3_buffer = []
        self.s_next_m1_buffer = []
        self.s_next_m2_buffer = []
        self.s_next_m3_buffer = []
        self.rewards_buffer = []
        self.done_buffer = []
        # timestep in an episode
        self.frame_count = 0
        # prob for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        # random steps
        self.epsilon_random_frames = 22000
        # for epsilon decay
        self.epsilon_greedy_frames = 50000.0
        # train model after actions
        self.loss_function = keras.losses.Huber()
        # discounted ratio
        self.gamma = 0.99
        
    def _build_q_network(self):
        # Network architecture
        input_m1 = keras.Input(shape=(self.n_jobs, 6, 1), dtype=tf.float32)
        input_m2 = keras.Input(shape=(self.n_types, self.n_types, 1), dtype=tf.float32)
        input_m3 = keras.Input(shape=(self.n_machines, 4, 1), dtype=tf.float32)

        layer1 = layers.Conv2D(16, 3, strides=1, activation="tanh")(input_m1)
        layer1 = layers.Conv2D(16, 3, strides=1, activation="tanh")(layer1)
        layer1 = layers.Flatten()(layer1)

        layer2 = layers.Conv2D(16, 3, strides=1, activation="tanh")(input_m2)
        layer2 = layers.Conv2D(16, 3, strides=1, activation="tanh")(layer2)
        layer2 = layers.Flatten()(layer2)

        layer3 = layers.Conv2D(16, 2, strides=1, activation="tanh")(input_m3)
        layer3 = layers.Conv2D(16, 2, strides=1, activation="tanh")(layer3)
        layer3 = layers.Flatten()(layer3)

        combined = layers.concatenate([layer1, layer2, layer3])
        common = layers.Dense(units = 64, activation = 'tanh')(combined)
        common = layers.Dense(units = 64, activation = 'tanh')(common)
        q_value = layers.Dense(units = self.n_actions)(common)

        self.q_network = keras.Model(inputs = [input_m1, input_m2, input_m3], outputs = q_value)
        self.t_q_network = keras.Model(inputs = [input_m1, input_m2, input_m3], outputs = q_value)
        
    def choose_action(self, state_m1, state_m2, state_m3):
        # exploration and exploitation
        if self.frame_count <= self.epsilon_random_frames or self.epsilon >= np.random.rand(1)[0]:
            action = np.random.choice(self.n_actions)
        else:
            action_probs = self.q_network([state_m1, state_m2, state_m3])
            action = np.argmax(action_probs)

        return action

    def decay_epsilon(self):
        # decay probability of taking random action
        self.epsilon -= (1.0 - self.epsilon_min)/self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def store(self, s_m1, s_m2, s_m3, action, s_next_m1, s_next_m2, s_next_m3, reward, done):
        # store training data
        self.action_buffer.append(action)
        
        self.s_m1_buffer.append(s_m1)
        self.s_m2_buffer.append(s_m2)
        self.s_m3_buffer.append(s_m3)
        
        self.s_next_m1_buffer.append(s_next_m1)
        self.s_next_m2_buffer.append(s_next_m2)
        self.s_next_m3_buffer.append(s_next_m3)
        
        self.rewards_buffer.append(reward)
        self.done_buffer.append(done)

    def train_q_network(self):
        # train per 4 actions
        indices = np.random.choice(range(len(self.done_buffer)), size = self.batch_size)

        # sample
        s_m1_sample = np.array([self.s_m1_buffer[i] for i in indices])
        s_m2_sample = np.array([self.s_m2_buffer[i] for i in indices])
        s_m3_sample = np.array([self.s_m3_buffer[i] for i in indices])
        
        s_next_m1_sample = np.array([self.s_next_m1_buffer[i] for i in indices])
        s_next_m2_sample = np.array([self.s_next_m2_buffer[i] for i in indices])
        s_next_m3_sample = np.array([self.s_next_m3_buffer[i] for i in indices])
        
        rewards_sample = [self.rewards_buffer[i] for i in indices]
        action_sample = [self.action_buffer[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(self.done_buffer[i]) for i in indices])

        # q(next_state)
        future_rewards = self.t_q_network.predict(
            [s_next_m1_sample, s_next_m2_sample, s_next_m3_sample]
        )
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                    future_rewards, axis=1
        )
        # set last q value to -1
        updated_q_values = updated_q_values*(1 - done_sample) - done_sample
        masks = tf.one_hot(action_sample, self.n_actions)

        with tf.GradientTape() as tape:
          # Train the model on the states and updated Q-values
          q_values = self.q_network([s_m1_sample, s_m2_sample, s_m3_sample])
          # only update q-value which is chosen
          q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
          # calculate loss between new Q-value and old Q-value
          loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def update_target_network(self):
        # update per update_target_network steps
        self.t_q_network.set_weights(self.q_network.get_weights())

    def remove_old_buffer(self):
        del self.s_m1_buffer[0]
        del self.s_m2_buffer[0]
        del self.s_m3_buffer[0]
        del self.s_next_m1_buffer[0]
        del self.s_next_m2_buffer[0]
        del self.s_next_m3_buffer[0]
        del self.action_buffer[0]
        del self.rewards_buffer[0]
        del self.done_buffer[0]
