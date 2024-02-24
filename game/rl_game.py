"""Snake game used with RL agent."""
from __future__ import annotations
from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from game.basic_game import BasicGame
from game.snake import Snake
from game.data_structures import Direction


class RLGame(BasicGame):
    """Snake game compatible with reinforcement learning."""

    def __init__(self, grid_size: int, snake: Snake, show_game: bool = False):
        super().__init__(grid_size=grid_size, snake=snake, show_game=show_game)
        self.observation_que: deque = deque(maxlen=4)
        self.step: int = 0
        self.step_limit: Optional[int] = 5000

    def reset_game(self):
        """
        Reset the game to its initial state.
        """
        self.observation_que.clear()
        self.snake.reset_snake()
        self.draw_elements()
        self.position_food()
        self.game_lost = False
        self.game_won = False
        self.score = 0

    def update_game(self) -> (float, bool):
        """Play one step then return reward and done flag."""
        self.snake.move()
        self.snake.grow(food_position=self.food)

        reward = 0.0
        done = False
        if self.check_if_lost():
            reward = -1.0
            done = True

        if self.snake.eaten:
            self.snake.eaten = False
            self.position_food()
            reward = 1.0
            self.score += 1

        if self.step > self.step_limit:
            done = True

        if self.game_won:
            done = True
            reward = 1.0
            self.score += 1

        self.draw_elements()

        return reward, done

    def get_observation(self) -> npt.NDArray:
        """Return 4 previous game states."""
        if len(self.observation_que) == 0:
            self.observation_que.append(self.game_canvas / 255)
            self.snake.update_direction(new_direction=Direction.DOWN)
            for i in range(self.observation_que.maxlen - 1):
                _, _ = self.update_game()
                self.observation_que.append(self.game_canvas / 255)
        else:
            self.observation_que.append(self.game_canvas / 255)

        return np.asarray(list(self.observation_que))
