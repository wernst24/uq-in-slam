
# TODO: Complete this draft and test Baseline CartPole 

import tensorflow as tf
from tensorflow.keras import layers

class A2CBaselineCartPole(tf.keras.Model):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = tf.keras.Sequential([
            layers.Input(shape=(obs_dim,)),
            layers.Dense(128, activation='relu'),
        ])
        self.policy_logits = layers.Dense(n_actions)
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.shared(inputs)
        return self.policy_logits(x), self.value(x)