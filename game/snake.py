"""Implementation for snake class."""

from game.data_structures import Direction, Position


class Snake:
    """Snake class implementation."""

    def __init__(self):
        self.head_position: Position = Position(x=3, y=0)
        self.body: list[Position] = self._initialize_body()
        self.direction: str = Direction.RIGHT
        self.eaten: bool = False

    def reset_snake(self) -> None:
        """Reset snake to initial state."""
        self.head_position = Position(x=3, y=0)
        self.body = self._initialize_body()
        self.eaten = False

    @staticmethod
    def _initialize_body() -> list[Position]:
        """Initialize and return body coordinates."""
        return [
            Position(x=3, y=0),
            Position(x=2, y=0),
            Position(x=1, y=0),
        ]

    def move(self):
        """Update head and body position."""
        # self.update_direction(new_direction=new_direction)
        self.head_position.update_position(
            new_x=self.head_position.x + Direction.get_x_step(direction=self.direction),
            new_y=self.head_position.y + Direction.get_y_step(direction=self.direction),
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
        self.body.insert(0, Position(x=self.head_position.x, y=self.head_position.y))
        if self.head_position.x == food_position.x and self.head_position.y == food_position.y:
            self.eaten = True
        else:
            self.body.pop()

    def is_point_in_snake(self, x: int, y: int, include_head: bool = True) -> bool:
        """Check if point is inside snake."""
        if include_head:
            body_parts = self.body
        else:
            body_parts = self.body[1:]

        for body_position in body_parts:
            if x == body_position.x and y == body_position.y:
                return True
        return False
