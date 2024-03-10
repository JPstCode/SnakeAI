# Snake AI

This repository contains Python 3.9 implementations of the classic Snake game and demonstrations of example algorithms, 
including Hamiltonian cycle, A*, and A3C.

## Installation

Original implementation was written with Python 3.9.13

Install required packages with pip:
```bash
pip install -r requirements.txt
```


## Algorithms

### Hamiltonian Cycle

![alt text](docs/Hamiltonian.gif)

Easiest way to win the game is to use hamiltonian cycle. Hamiltonian cycle is a closed loop that visits each cell 
exactly once, and returns to starting point [1].

Run Hamiltonian path example:
```bash
python algorithms\hamiltonian.py 
```

Specify grid size and number of repeats with keyword arguments. For example run 5 repeats on 10x10 grid:

```bash
python algorithms\hamiltonian.py --grid_size 10 --repeats 5 
```


### A*

![alt text](docs/A_star.gif)

A* star is a pathfinding algorithm that finds shortest path from source to goal [2]. 

Run A* path example:
```bash
python algorithms\a_star.py 
```

Specify grid size and number of repeats with keyword arguments
```bash
--grid_size --repeats 
```

### Asynchronous Advantage Actor Critic (A3C)

![alt text](docs/A3C.gif)

Actor-Critic method combines benefits from value-based and policy-based methods. The actor (policy function)
proposes a set of possible actions given a state. The critic (value function) evaluates actions taken by the actor
by determining expected return for an agent at given state [3].

During training Agent and Critic learn to perform their tasks, such that the recommended actions from the actor 
maximize the rewards [4].

A3C is a deep reinforcement learning algorithm, where multiple agents are executed asynchronously
on multiple instances of the environment [5]. 

### Training Agent

Start training by running train.py. Execution requires positional save_folder argument. 

For example start training and save log and weights to C:\tmp\a3c-training folder.

```bash
python a3c\training\train.py C:\tmp\a3c-training
```

Training will run until max_episodes limit is reached. Default value 15000. Change value with keyword 
argument

```bash
python a3c\training\train.py C:\tmp\a3c-training --max_episode 1000
```

See available all arguments for the training with --help  

```bash
python a3c\training\train.py --help
```

### References

[1] Hamiltonian Cycle [Wolfram MathWorld](https://mathworld.wolfram.com/HamiltonianCycle.html)

[2] A* [wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)

[3] Asynchronous Methods for Deep Reinforcement Learning [arXiv:1602.01783](https://arxiv.org/pdf/1602.01783.pdf)

[4] Playing CartPole with the Actor-Critic method [Tensorflow tutorials](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic)

[5] Actor Critic Method [Keras examples](https://keras.io/examples/rl/actor_critic_cartpole/)