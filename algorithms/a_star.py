"""A* example."""

import argparse

from algorithms.a_star_utils import get_a_star_path
from game.snake import Snake
from game.basic_game import BasicGame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamiltonian loop example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    snake = Snake()
    env = BasicGame(grid_size=args.grid_size, snake=snake, show_game=True)
    env.reset_game()
    env.draw_elements()
    while True:
        path = get_a_star_path(
            goal=env.food,
            start=env.snake.head_position,
            obstacles=env.snake.body,
            grid_size=args.grid_size,
        )

        for i in range(0, len(path)):
            direction = path[i]
            env.snake.update_direction(new_direction=direction)
            env.update_game()

            if env.game_lost:
                env.reset_game()
                env.draw_elements()
                break


if __name__ == '__main__':
    main()
