"""Implementation for Worker."""

import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.optimizers import Optimizer


from game.snake import Direction
from a3c.model.actor_critic_model import ActorCriticModel
from game.rl_game import RLGame
from game.snake import Snake
from a3c.model.actor_critic_model import initialize_model


class Memory:
    """Worker memory container."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    """Worker class used for asynchronous training."""
    def __init__(
            self,
            global_model: ActorCriticModel,
            optimizer: Optimizer,
            worker_index: int,
            grid_size: int,
            weights_path: Path,
            global_episode: int,
            # save_dir: Path,
            # master_agent: MasterAgent,
            episode_limit: int = 100000,
    ):
        super(Worker, self).__init__()
        snake = Snake()
        env = RLGame(grid_size=grid_size, snake=snake)
        self.global_model = global_model
        self.local_model = initialize_model(
            observation=env.get_observation(), action_size=4, weights_path=weights_path
        )
        self.env = RLGame(grid_size=grid_size, snake=Snake())
        self.optimizer = optimizer
        # self.save_dir = save_dir
        self.index = worker_index
        self.global_episode = global_episode
        self.episode_limit = episode_limit

    def run(self) -> None:
        """Start worker."""
        memory = Memory()
        while self.global_episode < self.episode_limit:
            done = False
            state = self.env.get_observation()
            episode_reward = 0
            episode_steps = 0

            while not done:
                action_logits, value = self.local_model(tf.expand_dims(state, 0))
                probs = tf.nn.softmax(action_logits)
                action = np.random.choice(4, p=probs.numpy()[0])
                new_direction = Direction.map_action_to_direction(action=action)
                self.env.snake.update_direction(new_direction=new_direction)
                reward, done = self.env.update_game()
                memory.store(state=state, action=action, reward=reward)
                state = self.env.get_observation()

                episode_reward += reward
                if done or episode_steps == 20:
                    print()
                # new_state = self.env.get_observation()
                # state = new_state