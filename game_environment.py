import random
import time

import numpy as np
import cv2
from snake_class import Snake, direction_map

# Colors (R, G, B)
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.3
fontColor = (255, 255, 255)
thickness = 1
lineType = 2


class Game:
    def __init__(self, frame_x_size, frame_y_size, difficulty, snake, show_score=False):

        self.frame_x_size = frame_x_size
        self.frame_y_size = frame_y_size
        self.snake = snake
        self.food_position = self.position_food()
        self.difficulty = difficulty
        self.game_lost = False
        self.canvas = np.zeros((self.frame_y_size, self.frame_x_size, 3))

        self.prev_total_distance = 0
        # self.total_distance_to_food = 0

        self.observation_feature_names = [
            "Head obstacles",
            "Distance to food X",
            "Distance to food Y",
            "Distance to frame X",
            "Distance to frame Y",
            "Direction to food",
        ]

        self.prev_dist_to_food = 0
        self.closer = False
        self.get_observation()

        self.epsilon = 0
        self.episode = -1
        self.score = 0
        self.show_score = show_score

    def draw_elements(self):
        canvas = self.canvas.copy()

        # Draw snake body
        # self.game_window.fill(black)
        for iter, pos in enumerate(self.snake.body):
            if iter == 0:
                cv2.rectangle(
                    canvas, (pos[0], pos[1]), (pos[0] + 10, pos[1] + 10), white, -1
                )
            else:
                cv2.rectangle(
                    canvas, (pos[0], pos[1]), (pos[0] + 10, pos[1] + 10), green, -1
                )

        # Draw food
        cv2.rectangle(
            canvas,
            (self.food_position[0], self.food_position[1]),
            (self.food_position[0] + 10, self.food_position[1] + 10),
            red,
            -1,
        )

        # show info
        if self.show_score:
            info_text = f"Score: {self.score}"
        else:
            info_text = f"Episode: {self.episode}, Epsilon: {self.epsilon:.3f}"

        cv2.putText(
            canvas,
            info_text,
            (int(self.frame_x_size / 10), self.frame_y_size - 20),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType
        )

        cv2.imshow("window", canvas)
        cv2.waitKey(5)
        time.sleep(0.01)
        return

    def update_game(self, done=False):

        self.snake.move(self.snake.direction)
        self.snake.grow(self.food_position)
        reward = 0

        self.game_lost = self.check_if_lost()
        if self.game_lost:
            done = True
            reward = -10

        self.draw_elements()

        if self.closer:
            reward += 1
            self.closer = False

        if self.snake.eaten:
            self.snake.eaten = False
            self.food_position = self.position_food()
            reward = 10
            print("MMMMMM")

        return reward, done

    def update_q_game(self, action):

        direction = direction_map[action]
        self.snake.update_direction(direction)

        self.snake.move(self.snake.direction)
        self.snake.grow(self.food_position)

        self.draw_elements()

        reward = 0
        if self.snake.eaten:
            self.snake.eaten = False
            self.food_position = self.position_food()
            reward = 10

        obs = self.get_observation()

        return obs, reward

    def position_food(self):

        while True:
            position = [
                random.randrange(1, (self.frame_x_size // 10)) * 10,
                random.randrange(1, (self.frame_y_size // 10)) * 10,
            ]

            if position not in self.snake.body:
                return position

    def check_if_lost(self):

        # Out of frame X
        if (
            self.snake.head_position[0] < 0
            or self.snake.head_position[0] >= self.frame_x_size
        ):
            print("Out of frame")
            return True

        # Out of frame Y
        if (
            self.snake.head_position[1] < 0
            or self.snake.head_position[1] >= self.frame_y_size
        ):
            print("Out of frame")
            return True

        # for block in snake.body[1:]:
        if self.snake.head_position in self.snake.body[1:]:
            print("Collision to body")
            return True

        return False

    def get_observation(self):

        # self.observation_feature = []
        observation = []

        # Direction to food [-1, 0] = up, [-1, 1] = northeast, [0, 1] = east...
        dir_to_food = self.get_direction_to_food()

        # Normalized distance
        y_food_dist, x_food_dist, total_dist = self.get_distance_to_food()

        if total_dist < self.prev_dist_to_food:
            self.closer = True
            self.prev_dist_to_food = total_dist

        # Get obstacles in all direction
        # obs_right, obs_bottom, obs_left, obs_top = self.get_blocking_status()

        # Distance to frame
        y_frame_dist, x_frame_dist = self.get_distance_to_frame()

        # Obstacles around head
        head_obstacles = self.get_obstacles_around_head()

        observation += head_obstacles
        observation += [x_food_dist, y_food_dist, x_frame_dist, y_frame_dist]
        observation += dir_to_food
        # self.observation_feature += head_obstacles
        # self.observation_feature += [x_food_dist, y_food_dist, x_frame_dist, y_frame_dist]
        # self.observation_feature += dir_to_food

        return np.asarray(observation)

    def get_direction_to_food(self):

        # Positions X, Y
        if self.snake.head_position[1] < self.food_position[1]:
            y_direction = 1
        elif self.snake.head_position[1] == self.food_position[1]:
            y_direction = 0
        else:
            y_direction = -1

        if self.snake.head_position[0] < self.food_position[0]:
            x_direction = 1
        elif self.snake.head_position[0] == self.food_position[0]:
            x_direction = 0
        else:
            x_direction = -1

        xs = [-1, 0, 1, 1, 1, 0, -1, -1]
        ys = [-1, -1, -1, 0, 1, 1, 1, 0]

        dir_to_food = []
        for x, y in zip(xs, ys):

            if [x, y] == [x_direction, y_direction]:
                dir_to_food.append(float(1))
            else:
                dir_to_food.append(float(0))

        return dir_to_food

    def get_distance_to_food(self):

        y_dist = (
            np.sqrt((self.snake.head_position[1] - self.food_position[1]) ** 2)
            / self.frame_y_size
        )
        x_dist = (
            np.sqrt((self.snake.head_position[0] - self.food_position[0]) ** 2)
            / self.frame_y_size
        )
        total = np.sqrt((x_dist**2 + y_dist**2)) / np.sqrt(
            (self.frame_y_size**2 + self.frame_x_size**2)
        )

        return float(y_dist), float(x_dist), float(total)

    def get_distance_to_frame(self):

        y_dist = self.snake.head_position[1] / self.frame_y_size
        x_dist = self.snake.head_position[0] / self.frame_x_size

        return float(y_dist), float(x_dist)

    def get_blocking_status(self):
        """See if there are obstacles between the head and frame"""

        obs_right = 0
        obs_bottom = 0
        obs_left = 0
        obs_top = 0  # bool

        for point in self.snake.body[1:]:

            # Block in horizontal line
            if point[1] == self.snake.head_position[1]:

                if point[0] > self.snake.head_position[0]:
                    obs_right = 1
                else:
                    obs_left = 1

            # Block in vertical line
            if point[0] == self.snake.head_position[0]:

                if point[1] > self.snake.head_position[1]:
                    obs_bottom = 1
                else:
                    obs_top = 1

        return float(obs_right), float(obs_bottom), float(obs_left), float(obs_top)

    def get_obstacles_around_head(self):

        positions_around_head = [
            [self.snake.head_position[0] - 10, self.snake.head_position[1] - 10],
            [self.snake.head_position[0], self.snake.head_position[1] - 10],
            [self.snake.head_position[0] + 10, self.snake.head_position[1] - 10],
            [self.snake.head_position[0] + 10, self.snake.head_position[1]],
            [self.snake.head_position[0] + 10, self.snake.head_position[1] + 10],
            [self.snake.head_position[0], self.snake.head_position[1] + 10],
            [self.snake.head_position[0] - 10, self.snake.head_position[1] + 10],
            [self.snake.head_position[0] - 10, self.snake.head_position[1]],
        ]

        position_status = []
        for position in positions_around_head:
            if position in self.snake.body[1:]:
                position_status.append(float(1))
                continue

            if -10 in position:
                position_status.append(float(1))
                continue

            if position[0] > self.frame_x_size:
                position_status.append(float(1))
                continue

            if position[1] > self.frame_y_size:
                position_status.append(float(1))
                continue

            position_status.append(float(0))

        return position_status

    def reset_game(self):
        direction = "RIGHT"
        self.snake = Snake(
            [100, 50], direction, [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        )
        self.food_position = self.position_food()
        return self.get_observation()
