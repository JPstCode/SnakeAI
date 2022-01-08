from algorithms import a_star
from game_environment import Game
from snake_class import Snake

frame_x = 300
frame_y = 300
diff = 20

direction = 'RIGHT'
new_direction = direction
snake = Snake([100, 50], direction, [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]])
game = Game(frame_x, frame_y, diff, snake)

route = []

# Find route automatically
while not game.game_lost:

    game.snake.direction = direction
    game.update_game()

    if not route:
        route = a_star.a_star_path(game.food_position, game.snake.head_position, game.snake.body[1:], frame_x, frame_y)

    if route:
        for direction in route:
            game.snake.direction = direction
            game.update_game()

            if game.snake.eaten:
                route = []
                game.snake.eaten = False
                game.food_position = game.position_food()
