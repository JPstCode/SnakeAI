from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy.typing as npt

from game.snake import Snake


@dataclass
class Direction:
    """Available directions for snake."""
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @staticmethod
    def get_x_step(direction: str) -> int:
        """Based on direction return x step size."""
        if direction == "UP":
            return 0
        elif direction == "RIGHT":
            return 1
        elif direction == "DOWN":
            return 0
        else:
            return -1

    @staticmethod
    def get_y_step(direction: str) -> int:
        """Based on direction return x step size"""
        if direction == "UP":
            return -1
        elif direction == "RIGHT":
            return 0
        elif direction == "DOWN":
            return 1
        else:
            return 0


class Position:
    """Store position of object"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update_position(self, new_x, new_y):
        self.x = new_x
        self.y = new_y


class GameBase(ABC):
    """Abstract implementation for game base."""
    def __init__(self):
        self.snake: Snake
        self.food: Position
        self.canvas: npt.NDArray

    @abstractmethod
    def reset_game(self):
        """
        Reset the game to its initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """
        Render the current state of the game.
        """
        raise NotImplementedError

    @abstractmethod
    def update_game(self):
        """
        Update the game state based on the current actions or events.
        """
        raise NotImplementedError

    @abstractmethod
    def position_food(self):
        """
        Update position of the food in the game.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        """
        Get the current observation of the game.
        """
        raise NotImplementedError
