"""Master worker for Snake A3C."""
from pathlib import Path
from typing import Optional

from keras.optimizers import Adam

from game.rl_game import RLGame
from game.snake import Snake
from a3c.model.actor_critic_model import initialize_model
from a3c.training.worker import Worker


class MasterAgent:
    """Master agent class controlling workers."""

    def __init__(
        self, save_dir: Path, grid_size: int, max_episodes: int, weights_path: Optional[Path] = None
    ):
        self.save_dir = save_dir
        self.grid_size = grid_size
        self.weights_path = weights_path
        self.max_episodes = max_episodes
        self.global_episode: int = 0

        snake = Snake()
        env = RLGame(grid_size=self.grid_size, snake=snake)

        self.global_model = initialize_model(
            observation=env.get_observation(), action_size=4, weights_path=weights_path
        )
        self.optimizer = Adam(learning_rate=0.0001)

    def train(
        self,
        save_dir: Path,
        gamma: float,
        max_episodes: int,
        update_freq: int,
        weights_save_freq: int,
        number_of_workers: int = 1,
    ):
        """Do asynchronous training with workers."""

        workers = self._initialize_workers(
            number_of_workers=number_of_workers,
            save_dir=save_dir,
            gamma=gamma,
            max_episodes=max_episodes,
            update_freq=update_freq,
            weights_save_freq=weights_save_freq,
        )
        print("Starting training!")
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

    def _initialize_workers(
        self,
        number_of_workers: int,
        save_dir: Path,
        gamma: float,
        max_episodes: int,
        update_freq: int,
        weights_save_freq: int,
    ) -> list[Worker]:
        """Initialize workers."""
        workers = []
        for i in range(number_of_workers):
            print(f"Initializing {i}. worker")
            worker = Worker(
                save_dir=save_dir,
                global_model=self.global_model,
                optimizer=self.optimizer,
                worker_index=i,
                grid_size=self.grid_size,
                weights_path=self.weights_path,
                global_episode=self.global_episode,
                gamma=gamma,
                update_freq=update_freq,
                max_episodes=max_episodes,
                weights_save_freq=weights_save_freq,
            )
            workers.append(worker)
        return workers
