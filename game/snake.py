"""Implementation for snake class."""

from game.game_base import Position
from game.game_base import Direction


class Snake:
    """Snake class implementation."""

    def __init__(self):
        self.head_position: Position = Position(x=3, y=3)
        self.body = self._initialize_body()
        self.direction: str = Direction.RIGHT
        self.eaten: bool = False

    def _initialize_body(self) -> list[Position]:
        """Initialize and return body coordinates."""
        return [
            Position(x=self.head_position.x - 1, y=self.head_position.y),
            Position(x=self.head_position.x - 2, y=self.head_position.y),
            Position(x=self.head_position.x - 3, y=self.head_position.y),
        ]

    def move(self, new_direction: Direction):
        """Update head and body position."""
        self.update_direction(new_direction=new_direction)
        self.head_position.update_position(
            new_x=self.head_position.x + Direction.get_x_step(direction=self.direction),
            new_y=self.head_position.y + Direction.get_x_step(direction=self.direction),
        )

    def update_direction(self, new_direction: Direction) -> None:
        """Update snake direction."""
        if self.direction == new_direction:
            pass

        elif self.direction == Direction.DOWN and new_direction == Direction.UP:
            pass

        elif self.direction == Direction.UP and new_direction == Direction.DOWN:
            pass

        elif self.direction == Direction.LEFT and new_direction == Direction.RIGHT:
            pass

        elif self.direction == Direction.RIGHT and new_direction == Direction.LEFT:
            pass

        else:
            self.direction = new_direction

    def grow(self, food_position: Position):
        """Update body position and grow snake if eaten."""
        self.body.insert(0, self.head_position)
        if self.head_position.x == food_position.x and self.head_position.y == food_position.y:
            self.eaten = True
        else:
            self.body.pop()
