"""Implementation of the actor critic model."""

from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten


class ActorCriticModel(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(self, state_size: tuple, action_size: int):
        """Initialize model."""
        super().__init__()

        self.common: Sequential = Sequential()
        self.add_common_layers(state_size=state_size)

        self.actor = layers.Dense(action_size)
        self.critic = layers.Dense(1)

    def add_common_layers(self, state_size: tuple):
        """Add convolutional and fully connected layers."""

        # Convolutional layers
        self.common.add(
            Conv2D(
                filters=16,
                kernel_size=8,
                strides=4,
                input_shape=state_size,
                padding="same",
                activation="relu",
                name="C1",
            )
        )
        self.common.add(
            Conv2D(
                filters=32,
                kernel_size=4,
                strides=2,
                padding="same",
                activation="relu",
                name="C2",
            )
        )
        # Fully-Connected layers
        self.common.add(Flatten())
        self.common.add(Dense(units=256, activation="relu", name="D1"))

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Return probability distribution for actions and value estimate for state."""
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def save_model(self, output_path: Path, episode: int):
        """Save model in keras format"""
        self.save_weights(output_path / f"{episode}_a3c.keras")


def initialize_model(
    observation: np.ndarray, action_size: int, weights_path: Optional[Path]
) -> ActorCriticModel:
    """Initialize model and load weights if specified."""
    state_size = observation.shape
    model = ActorCriticModel(state_size=state_size, action_size=action_size)
    # In order to load model weights model needs to build -> Do dummy inference.
    model(tf.expand_dims(observation, 0))
    if weights_path:
        model.load_weights(weights_path)
    return model
