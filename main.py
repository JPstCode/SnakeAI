from algorithms import a_star
from game_environment import Game
from snake_class import Snake

frame_x = 200
frame_y = 200
diff = 20

direction = "RIGHT"
new_direction = direction
snake = Snake([100, 50], direction, [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]])
game = Game(frame_x, frame_y, diff, snake)

# MODE = "A*"
MODE = "RL"


def a_star_loop(game, direction):

    route = []

    while not game.game_lost:
        game.snake.direction = direction
        game.update_game()

        if not route:
            route = a_star.a_star_path(
                game.food_position,
                game.snake.head_position,
                game.snake.body[1:],
                frame_x,
                frame_y,
            )

        if route:
            for direction in route:
                game.snake.direction = direction
                game.update_game()

                if game.snake.eaten:
                    route = []
                    game.snake.eaten = False
                    game.food_position = game.position_food()


if MODE == "A*":
    a_star_loop(game=game, direction=direction)

# if MODE == "RL":

