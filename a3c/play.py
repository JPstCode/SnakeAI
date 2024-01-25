"""Play Snake game with model."""
from __future__ import annotations
import argparse
from pathlib import Path

from game.snake import Snake
from game.rl_game import RLGame
from a3c.actor_critic_model import initialize_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamiltonian loop example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    parser.add_argument(
        "--weights_path", type=Path, default=None, help="Path to model weights (.keras)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    snake = Snake()
    env = RLGame(grid_size=args.grid_size, snake=snake, show_game=True)
    model = initialize_model(observation=env.get_observation(), action_size=4, weights_path=args.weights_path)


if __name__ == '__main__':
    main()