"""Snake game used with RL agent."""
from __future__ import annotations
from collections import deque

import numpy as np
import numpy.typing as npt

from game.basic_game import BasicGame
from game.snake import Snake


class RLGame(BasicGame):
    """Snake game compatible with reinforcement learning."""
    def __init__(self, grid_size: int, snake: Snake, show_game: bool = False):
        super().__init__(grid_size=grid_size, snake=snake, show_game=show_game)
        self.observation_que: deque = deque(maxlen=4)

    def get_observation(self) -> npt.NDArray:
        """Return 4 previous game states."""
        if len(self.observation_que) == 0:
            self.observation_que.append(self.game_canvas / 255)
            for i in range(self.observation_que.maxlen - 1):
                self.update_game()
                self.observation_que.append(self.game_canvas / 255)
        else:
            self.observation_que.append(self.game_canvas / 255)

        return np.asarray(list(self.observation_que))
