"""Play Snake game with model."""
import sys
import os

sys.path.append(os.getcwd())

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from game.snake import Snake
from game.rl_game import RLGame
from game.data_structures import Direction
from a3c.model.actor_critic_model import initialize_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3C example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    parser.add_argument(
        "--weights_path", type=Path, default=None, help="Path to model weights (.keras)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Specify how many episodes game is running."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    snake = Snake()
    env = RLGame(grid_size=args.grid_size, snake=snake, show_game=True)
    model = initialize_model(
        observation=env.get_observation(), action_size=4, weights_path=args.weights_path
    )
    env.reset_game()

    episode_idx = 0
    while episode_idx < args.episodes:
        state = env.get_observation()
        action_logits, value = model(tf.expand_dims(state, 0))
        probs = tf.nn.softmax(action_logits)
        action = np.random.choice(4, p=probs.numpy()[0])
        new_direction = Direction.map_action_to_direction(action=action)
        env.snake.update_direction(new_direction=new_direction)

        # Take a step
        _, done = env.update_game()
        if done:
            env.reset_game()
            episode_idx += 1


if __name__ == '__main__':
    main()
