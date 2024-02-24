"""Implementation for A3C Worker."""

import threading
from pathlib import Path
from collections import deque

import numpy as np
import tensorflow as tf
from keras.optimizers import Optimizer
from keras.losses import huber

from game.snake import Direction
from a3c.model.actor_critic_model import ActorCriticModel
from game.rl_game import RLGame
from game.snake import Snake
from a3c.model.actor_critic_model import initialize_model
from a3c.training.support import log_progress

eps = np.finfo(np.float32).eps.item()

GLOBAL_EPISODE = 0
GLOBAL_MOVING_AVERAGE = deque(maxlen=100)


class Memory:
    """Worker memory container."""

    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []

    def store(self, state: np.ndarray, action: int, reward: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


class Worker(threading.Thread):
    """Worker class used for asynchronous training."""

    save_lock = threading.Lock()

    def __init__(
        self,
        save_dir: Path,
        global_model: ActorCriticModel,
        optimizer: Optimizer,
        worker_index: int,
        grid_size: int,
        weights_path: Path,
        global_episode: int,
        gamma: float,
        update_freq: int,
        weights_save_freq: int,
        max_episodes: int,
    ):
        super(Worker, self).__init__()
        self.save_dir = save_dir
        self.global_model = global_model
        self.env = RLGame(grid_size=grid_size, snake=Snake())
        self.local_model = initialize_model(
            observation=self.env.get_observation(), action_size=4, weights_path=weights_path
        )

        self.optimizer = optimizer

        self.index = worker_index
        self.global_episode = global_episode
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.update_freq = update_freq
        self.weights_save_freq = weights_save_freq

    def run(self) -> None:
        """Start worker."""
        global GLOBAL_EPISODE
        global GLOBAL_MOVING_AVERAGE

        memory = Memory()
        while GLOBAL_EPISODE < self.max_episodes:
            self.env.reset_game()
            memory.clear()
            done = False
            state = self.env.get_observation()
            self.env.score = 0

            episode_reward = 0
            update_steps = 0
            episode_steps = 0

            while not done:
                # Play step
                action_logits, value = self.local_model(tf.expand_dims(state, 0))
                probs = tf.nn.softmax(action_logits)
                action = np.random.choice(4, p=probs.numpy()[0])
                new_direction = Direction.map_action_to_direction(action=action)
                self.env.snake.update_direction(new_direction=new_direction)
                reward, done = self.env.update_game()

                # Save game step to memory
                memory.store(state=state, action=action, reward=reward)

                new_state = self.env.get_observation()
                state = new_state

                episode_reward += reward
                if done or update_steps == self.update_freq:
                    update_steps = 0

                    with tf.GradientTape() as tape:
                        policy_loss, value_loss = self.compute_loss(
                            done=done, memory=memory, new_state=new_state
                        )
                        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))

                    gradients = tape.gradient(total_loss, self.local_model.trainable_weights)

                    # Update global model parameters
                    self.optimizer.apply_gradients(
                        zip(gradients, self.global_model.trainable_weights)
                    )

                    # Fetch latest global parameters
                    self.local_model.set_weights(self.global_model.get_weights())

                    if done:
                        GLOBAL_MOVING_AVERAGE.append(episode_reward)
                        log_progress(
                            worker_idx=self.index,
                            global_episode=GLOBAL_EPISODE,
                            reward=episode_reward,
                            reward_moving_average=float(np.mean(GLOBAL_MOVING_AVERAGE)),
                            steps=episode_steps,
                            policy_loss=float(policy_loss.numpy()),
                            value_loss=float(value_loss.numpy()),
                            filepath=self.save_dir / 'training_log.csv',
                        )

                    # Save weights
                    if GLOBAL_EPISODE % self.weights_save_freq == 0 and GLOBAL_EPISODE != 0:
                        filepath = self.save_dir / f"{GLOBAL_EPISODE}_a3c.keras"
                        with Worker.save_lock:
                            self.global_model.save_weights(str(filepath))

                    memory.clear()

                update_steps += 1
                episode_steps += 1
            GLOBAL_EPISODE += 1

    def compute_loss(
        self, done: bool, memory: Memory, new_state: np.ndarray
    ) -> (tf.Tensor, tf.Tensor):
        """Compute loss to guide agents training."""
        reward_sum = self.get_reward_sum(done=done, new_state=new_state)
        discounted_rewards = self.get_discounted_rewards(
            reward_sum=reward_sum, rewards=memory.rewards, gamma=self.gamma
        )
        action_logits, value_estimates, action_probs = self.get_probabilities_and_value_estimates(
            states=memory.states, actions=memory.actions
        )
        advantage = discounted_rewards - value_estimates

        value_loss = tf.reduce_sum(huber(discounted_rewards, value_estimates))
        # value_loss = tf.reduce_mean(advantage ** 2) #MSE

        log_probabilities = -tf.math.log(action_probs)
        policy_loss = tf.reduce_sum(log_probabilities * tf.stop_gradient(advantage))
        return policy_loss, value_loss

    def get_reward_sum(self, done: bool, new_state: np.ndarray) -> float:
        """If game didn't terminate estimate latest reward with Critic."""
        if done:
            return 0.0
        _, value_estimate = self.local_model(tf.expand_dims(new_state, axis=0))
        return float(value_estimate.numpy())

    @staticmethod
    def get_discounted_rewards(reward_sum: float, rewards: list[float], gamma: float) -> tf.Tensor:
        """Calculate discounted rewards."""
        discounted_rewards = []

        # Start discounting from the final reward
        for reward in reversed(rewards):
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append([reward_sum])  # Make format compatible with Tensors

        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)

        # Normalize
        discounted_rewards = (discounted_rewards - tf.math.reduce_mean(discounted_rewards)) / (
            tf.math.reduce_std(discounted_rewards) + eps
        )

        return discounted_rewards

    def get_probabilities_and_value_estimates(
        self, states: list[np.ndarray], actions: list[int]
    ) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """Do inference to get action probabilities and value estimates for GradientTape."""
        logits_selected_actions = []
        probabilities_selected_actions = []
        value_estimates = []
        for idx, state in enumerate(states):
            action_logits, value_estimate = self.local_model(tf.expand_dims(state, axis=0))
            logits_selected_actions.append(action_logits[0])
            value_estimates.append(value_estimate[0])

            action_probabilities = tf.nn.softmax(action_logits)
            selected_action = actions[idx]
            probabilities_selected_actions.append([action_probabilities[0][selected_action]])

        logits_selected_actions = tf.convert_to_tensor(logits_selected_actions)
        value_estimates = tf.convert_to_tensor(value_estimates)
        probabilities_selected_actions = tf.convert_to_tensor(probabilities_selected_actions)
        return logits_selected_actions, value_estimates, probabilities_selected_actions
