from dataclasses import dataclass


@dataclass
class GameColors:
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 200, 255)


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

    @staticmethod
    def map_action_to_direction(action: int) -> str:
        if action == 0:
            return "RIGHT"
        elif action == 1:
            return "DOWN"
        elif action == 2:
            return "LEFT"
        else:
            return "UP"


class Position:
    """Store position of object"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update_position(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
