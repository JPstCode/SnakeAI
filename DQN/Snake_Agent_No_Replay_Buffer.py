# Example: Deep Q Learning Networks https://www.youtube.com/watch?v=OYhFoMySoVs

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = " "

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from game_environment import Game
from snake_class import Snake, direction_map

# PARAMETERS

# Environment states

# 1. distance to apple
# 2. View around head

# Environment actions

# 1. Move UP, RIGHT, DOWN or LEFT

# state_size = 6
# action_size = 4


# AGENT


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # list to add and remove
        self.memory = deque(maxlen=2000)

        # Discount factor (How much future reward is appreciated vs. short term reward)
        self.gamma = 0.95

        # Exploration rate (Exploitation vs Exploration)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(36, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(14, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))

        # Surprisingly the best?
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # print('make prediction')
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )

            try:
                # Map maximized future reward to current reward - state ?
                target_f = self.model.predict(state)
                target_f[0][action] = target

                self.model.fit(state, target_f, epochs=1, verbose=0)

            except Exception as err:
                print(err)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    # cart_env = gym.make('CartPole-v0')

    snake = Snake([100, 50], "RIGHT", [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]])
    env = Game(200, 200, 5, snake)

    state_size = len(env.get_observation())
    action_size = 4

    # for SGD
    batch_size = 1

    # Number of games agent is allowed to play
    n_episodes = 1001

    output_dir = os.path.join(os.getcwd(), "Snake-Model-Weight")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    agent = DQNAgent(state_size, action_size)

    # agent.epsilon = 0.0
    # agent.load(r'C:\Users\juhop\Documents\Python\SnakeAI\DQN\Snake-Model-Weight\weights_0900.hdf5')

    # Interact with environment
    done = False
    for e in range(900, 1500):

        # Reset state
        state = env.reset_game()
        state = np.reshape(state, [1, state_size])

        for time in range(5000):

            action = agent.act(state)
            next_state, reward = env.update_q_game(action)
            next_state = np.reshape(next_state, [1, state_size])

            done = env.check_if_lost()
            if done:
                reward = -10
            agent.remember(state, action, reward, next_state, done)

            if done:
                print(
                    "Episode: {}/{}, score: {}, e: {:.2f}".format(
                        e, n_episodes, time, agent.epsilon
                    )
                )
                break

            state = next_state

        print("Training based on last run")
        for state, action, reward, next_state, done in agent.memory:
            target = reward
            if not done:
                target = reward + agent.gamma * np.amax(
                    agent.model.predict(next_state)[0]
                )

            try:
                target_f = agent.model.predict(state)
                target_f[0][action] = target
                agent.model.fit(state, target_f, epochs=1, verbose=0)

            except Exception as err:
                print(err)

        agent.memory.clear()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if e % 20 == 0:
            agent.save(
                os.path.join(output_dir, "weights_" + "{:04d}".format(e) + ".hdf5")
            )

        print(f"Agent epsilon: {agent.epsilon}")
