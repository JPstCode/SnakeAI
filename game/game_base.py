from abc import ABC, abstractmethod

import numpy.typing as npt

from game.data_structures import Position
from game.snake import Snake


class GameBase(ABC):
    """Abstract implementation for game base."""

    def __init__(self):
        self.grid_size: int
        self.canvas: npt.NDArray
        self.snake: Snake
        self.food: Position

    @abstractmethod
    def reset_game(self):
        """
        Reset the game to its initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def draw_elements(self):
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
