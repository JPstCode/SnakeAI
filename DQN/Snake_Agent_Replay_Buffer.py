import os

import numpy as np

from game_environment import Game
from snake_class import Snake, direction_map
from DQN.Agent_Replay_Buffer import DQNAgent

frame_x = 200
frame_y = 200
diff = 20

direction = "RIGHT"
new_direction = direction
snake = Snake([100, 50], direction, [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]])
game = Game(frame_x, frame_y, diff, snake)

### AGENT PARAMETERS ###

state_size = game.get_observation().shape[0]
action_size = 4

# for SGD
batch_size = 64

# Number of games agent is allowed to play
n_episodes = 100001

output_dir = "Snake-Model-Weight"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

agent = DQNAgent(state_size, action_size)

# Interact with the environment
for e in range(n_episodes):

    game.reset_game()
    state = game.get_observation()
    steps = 0
    done = False
    reward = 0

    game.episode_n = e
    game.epsilon = agent.epsilon

    while not done:

        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        game.snake.direction = direction_map[action]
        head_obstacles = state[0][:9]

        # print('###########')
        # print(head_obstacles[0], head_obstacles[1], head_obstacles[2])
        # print(head_obstacles[7], "X.X", head_obstacles[3])
        # print(head_obstacles[6], head_obstacles[5], head_obstacles[4])

        reward, done = game.update_game()

        # Penalizing poor actions
        reward = reward if not done else -10

        # print('###########')
        # if done:
        # print(direction_map[action])
        # next_state = game.get_observation()
        # head_obstacles = next_state[:9]
        # print('------')
        # print(head_obstacles[0], head_obstacles[1], head_obstacles[2])
        # print(head_obstacles[7], "X.X", head_obstacles[3])
        # print(head_obstacles[6], head_obstacles[5], head_obstacles[4])
        # print('###########')

        next_state = game.get_observation()
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print(
                "Episode: {}/{}, steps: {}, e: {:.2f}".format(
                    e, n_episodes, steps, agent.epsilon
                )
            )
            break

        steps += 1
        game.episode = e

    if len(agent.memory) > batch_size:
        print("Training")
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(os.path.join(output_dir, "weights_" + "{:04d}".format(e) + ".hdf5"))
