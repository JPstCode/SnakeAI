"""Training script for A3C Snake agent.

Based on tensorflow tutorial:
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
"""
import argparse
from pathlib import Path

from a3c.training.master_agent import MasterAgent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamiltonian loop example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    parser.add_argument(
        "--weights_path", type=Path, default=None, help="Path to model weights (.keras)"
    )
    parser.add_argument(
        "--save_dir", type=Path, required=True, help="Path where to save weights."
    )
    parser.add_argument(
        "--number_of_workers", type=int, default=1, help="Specify how many workers are utilized in training."
    )
    parser.add_argument(
        "--max_episodes", type=int, default=100000, help="How many episodes training is running."
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount coefficient."
    )
    parser.add_argument(
        "--update_freq", type=float, default=5, help="Specify frequency for updating the global model."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    master_agent = MasterAgent(
        save_dir=args.save_dir,
        grid_size=args.grid_size,
        weights_path=args.weights_path
    )
    master_agent.train(number_of_workers=args.number_of_workers)
    print()



if __name__ == '__main__':
    main()