"""Basic implementation of the Snake game."""
import random
import time

import cv2
import numpy as np
import numpy.typing as npt

from game.game_base import GameBase
from game.snake import Snake
from game.data_structures import Position
from game.data_structures import GameColors


class BasicGame(GameBase):
    """Basic implementation for game."""

    def __init__(self, grid_size: int, snake: Snake, show_game: bool = False):
        super().__init__()
        self.block_size = 10  # For convenience
        self.grid_size = grid_size
        self.blank_canvas = np.zeros(
            shape=(grid_size * self.block_size, grid_size * self.block_size, 3), dtype=np.uint8
        )
        self.game_canvas = np.zeros(
            shape=(grid_size * self.block_size, grid_size * self.block_size, 3), dtype=np.uint8
        )
        self.snake = snake
        self.food = Position(x=0, y=0)
        self.show_game = show_game
        self.game_lost: bool = False
        self.game_won: bool = False
        self.score: int = 0

    def reset_game(self):
        """
        Reset the game to its initial state.
        """

        self.snake.reset_snake()
        self.position_food()
        self.game_lost = False
        self.game_won = False
        self.score = 0

    def position_food(self):
        """Find empty cell and update food position."""
        free_positions = []
        all_positions_x = np.arange(start=0, stop=self.grid_size).astype(int)
        all_positions_y = np.arange(start=0, stop=self.grid_size).astype(int)
        for x_pos in all_positions_x:
            for y_pos in all_positions_y:
                if x_pos == self.food.x and y_pos == self.food.y:
                    continue
                if self.snake.is_point_in_snake(x=x_pos, y=y_pos):
                    continue

                free_positions.append(Position(x=x_pos, y=y_pos))

        # Victory
        if len(free_positions) == 0:
            self.game_won = True
        else:
            food_position = random.choice(free_positions)
            self.food.update_position(new_x=food_position.x, new_y=food_position.y)

    def draw_elements(self):
        """
        Render the current state of the game.
        """
        canvas = self.blank_canvas.copy()
        for idx, position in enumerate(self.snake.body):
            if idx == 0:
                canvas = self._draw_block(canvas=canvas, position=position, color=GameColors.white)
            else:
                canvas = self._draw_block(canvas=canvas, position=position, color=GameColors.green)

        food_color = GameColors.red
        if self.snake.eaten:
            food_color = GameColors.orange
        canvas = self._draw_block(canvas=canvas, position=self.food, color=food_color)

        self.game_canvas = canvas
        if self.show_game:
            self.show_game_window(canvas=canvas)

    @staticmethod
    def show_game_window(canvas: npt.NDArray):
        """Draw game."""
        canvas = cv2.filter2D(
            cv2.pyrUp(canvas),
            -1,
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        )

        cv2.imshow('Game', cv2.pyrUp(canvas))
        cv2.waitKey(5)
        time.sleep(0.01)

    def _draw_block(self, canvas: npt.NDArray, position: Position, color: tuple) -> npt.NDArray:
        """Draw block to canvas"""
        left_corner_x = position.x * self.block_size
        left_corner_y = position.y * self.block_size

        return cv2.rectangle(
            canvas,
            (left_corner_x + 1, left_corner_y + 1),
            (left_corner_x + self.block_size - 1, left_corner_y + self.block_size - 1),
            color,
            -1,
        )

    def update_game(self):
        """
        Update the game state based on the current actions or events.
        """
        self.snake.move()
        self.snake.grow(food_position=self.food)

        if self.snake.eaten:
            self.snake.eaten = False
            self.position_food()

        if self.check_if_lost():
            self.game_lost = True

        self.draw_elements()

    def check_if_lost(self):
        """Check if snake collided to wall or to itself."""
        if self.snake.head_position.x < 0 or self.snake.head_position.x >= self.grid_size:
            return True

        if self.snake.head_position.y < 0 or self.snake.head_position.y >= self.grid_size:
            return True

        if self.snake.is_point_in_snake(
            x=self.snake.head_position.x, y=self.snake.head_position.y, include_head=False
        ):
            return True

        return False
