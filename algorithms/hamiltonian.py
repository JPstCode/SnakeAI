"""Hamiltonian loop example"""
import os
import sys

sys.path.append(os.getcwd())

import argparse

import numpy.typing as npt
import numpy as np

from game.data_structures import Direction
from game.snake import Snake
from game.basic_game import BasicGame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamiltonian loop example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Specify how many repeats game is running."
    )
    return parser.parse_args()


def create_hamiltonian_path(grid_size: int) -> npt.NDArray[str]:
    """Create simple hamiltonian path"""
    path = np.zeros(shape=(grid_size, grid_size), dtype=object)

    # Top row move right
    path[0, :-1] = Direction.RIGHT

    # Right column move down
    path[:-1, -1] = Direction.DOWN

    # Bottom right corner move left
    path[-1, -1] = Direction.LEFT

    # Add zigzag from bottom right corner
    for col in reversed(range(1, grid_size - 1)):
        # Even columns
        if (col % 2) == 0:
            path[1, col] = Direction.LEFT
            path[-1, col] = Direction.UP
            path[2:-1, col] = Direction.UP
        else:
            path[2:-1, col] = Direction.DOWN
            path[1, col] = Direction.DOWN
            path[-1, col] = Direction.LEFT

    # Left column move up
    path[1:, 0] = Direction.UP
    return path


def main():
    args = parse_arguments()
    hamiltonian_path = create_hamiltonian_path(grid_size=args.grid_size)
    snake = Snake()
    env = BasicGame(grid_size=args.grid_size, snake=snake, show_game=True)
    env.reset_game()
    iterations = 0
    while iterations < args.repeats:
        new_direction = hamiltonian_path[snake.head_position.y, snake.head_position.x]
        env.snake.update_direction(new_direction=new_direction)
        env.update_game()
        if env.game_lost or env.game_won:
            env.reset_game()
            iterations += 1


if __name__ == '__main__':
    main()
