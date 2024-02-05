"""Master worker"""
from pathlib import Path
from typing import Optional

import tensorflow as tf
from keras.optimizers import Adam

from game.rl_game import RLGame
from game.snake import Snake
from a3c.model.actor_critic_model import initialize_model
from a3c.training.worker import Worker


class MasterAgent:
    """Master agent class controlling workers."""
    def __init__(self, save_dir: Path, grid_size: int, weights_path: Optional[Path] = None):
        self.save_dir = save_dir
        self.grid_size = grid_size
        self.weights_path = weights_path

        snake = Snake()
        env = RLGame(grid_size=self.grid_size, snake=snake)

        self.global_model = initialize_model(
            observation=env.get_observation(), action_size=4, weights_path=weights_path
        )
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer = Adam(learning_rate=0.001)

        # Load latest episode
        self.global_episode: int = 0
        self.episode_limit: int = 100000

    def train(self, number_of_workers: int = 1):
        """Do asynchronous training with workers."""
        workers = self._initialize_workers(number_of_workers=number_of_workers)
        for worker in workers:
            worker.start()

        print()
    # global_model: ActorCriticModel,
    # optimizer: Optimizer,
    # worker_index: int,
    # grid_size: int,
    # weights_path: Path,
    # # save_dir: Path,
    # master_agent: MasterAgent,

    def _initialize_workers(self, number_of_workers: int) -> list[Worker]:
        """Initialize workers."""
        workers = []
        for i in range(number_of_workers):
            worker = Worker(
                global_model=self.global_model,
                optimizer=self.optimizer,
                worker_index=i,
                grid_size=self.grid_size,
                weights_path=self.weights_path,
                global_episode=self.global_episode,
                episode_limit=self.episode_limit,
            )
            workers.append(worker)
        return workers

