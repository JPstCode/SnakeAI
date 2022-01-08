
direction_map = {0: "RIGHT",
                 1: "DOWN",
                 2: "LEFT",
                 3: "UP"}


class Snake:
    def __init__(self, heap_position, direction, body):
        self.head_position = heap_position
        self.direction = direction
        self.body = body
        self.direction_map = {'UP': [0, -10],
                              'DOWN': [0, 10],
                              'LEFT': [-10, 0],
                              'RIGHT': [10, 0],
                              'HOLD': [0, 0]}
        self.eaten = False

    def move(self, direction):

        self.update_direction(direction)
        self.head_position[0] += self.direction_map[self.direction][0]
        self.head_position[1] += self.direction_map[self.direction][1]

    def update_direction(self, direction):

        if self.direction == direction:
            return

        if self.direction == 'UP' and direction == 'DOWN':
            return

        if self.direction == 'DOWN' and direction == 'UP':
            return

        if self.direction == 'LEFT' and direction == 'RIGHT':
            return

        if self.direction == 'RIGHT' and direction == 'LEFT':
            return

        self.direction = direction

    def grow(self, food_position):
        self.body.insert(0, list(self.head_position))
        if self.head_position[0] == food_position[0] and self.head_position[1] == food_position[1]:
            self.eaten = True
        else:
            self.body.pop()

