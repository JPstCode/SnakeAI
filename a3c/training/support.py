"""Support module for logging."""

from pathlib import Path
import csv


def create_log_file(filepath: Path) -> None:
    """Create .csv file to record training progress"""
    with open(filepath, mode='w', newline='') as file:
        fieldnames = [
            'Episode',
            'Reward',
            'Steps',
            'Policy Loss',
            'Value Loss',
            'Moving Average',
            'Worker Idx',
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
    return


def log_progress(
    worker_idx: int,
    global_episode: int,
    reward: int,
    reward_moving_average: float,
    steps: int,
    policy_loss: float,
    value_loss: float,
    filepath: Path,
    print_out: bool = True,
) -> None:
    """Log training progress."""

    if not filepath.exists():
        create_log_file(filepath=filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                global_episode,
                reward,
                steps,
                policy_loss,
                value_loss,
                reward_moving_average,
                worker_idx,
            ]
        )

    if print_out:
        print(
            f"episode: {global_episode}, "
            f"reward: {reward}, "
            f"steps: {steps}, "
            f"moving_average: {reward_moving_average:.3f}"
        )
