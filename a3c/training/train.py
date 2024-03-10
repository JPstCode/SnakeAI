"""Training script for A3C Snake agent.

Based on tensorflow tutorials:
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html
"""
import os
import sys

sys.path.append(os.getcwd())

import argparse
from pathlib import Path

from a3c.training.master_agent import MasterAgent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A3C Training loop.")
    parser.add_argument("save_dir", type=Path, help="Path where to save weights.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    parser.add_argument(
        "--weights_path", type=Path, default=None, help="Path to model weights (.keras)"
    )
    parser.add_argument(
        "--number_of_workers",
        type=int,
        default=8,
        help="Specify how many workers are utilized in training.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=15000, help="How many episodes training is running."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Coefficient to control effect of long/short term reward on learning.",
    )
    parser.add_argument(
        "--update_freq",
        type=float,
        default=50,
        help="Specify frequency for updating the global model.",
    )
    parser.add_argument(
        "--weights_save_freq",
        type=float,
        default=500,
        help="Specify frequency for saving model weights.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    master_agent = MasterAgent(
        save_dir=args.save_dir,
        grid_size=args.grid_size,
        weights_path=args.weights_path,
        max_episodes=args.max_episodes,
    )

    master_agent.train(
        save_dir=args.save_dir,
        gamma=args.gamma,
        update_freq=args.update_freq,
        weights_save_freq=args.weights_save_freq,
        max_episodes=args.max_episodes,
        number_of_workers=args.number_of_workers,
    )


if __name__ == '__main__':
    main()
