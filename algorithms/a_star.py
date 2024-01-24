"""A* example."""

import argparse
import time
from typing import Optional
from random import shuffle

import numpy.typing as npt
import numpy as np
import cv2

from game.data_structures import Direction
from game.data_structures import Position
from game.snake import Snake
from game.basic_game import BasicGame


class NodeCost:
    def __init__(self, position: Position, path_cost: float, heuristic_cost: float, direction: Optional[str] = None):
        self.position = Position
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.total_cost = path_cost + heuristic_cost
        self.direction = direction


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hamiltonian loop example.")
    parser.add_argument(
        "--grid_size", type=int, default=6, help="Specify the grid size for the game."
    )
    return parser.parse_args()


def distance_to_point(start: Position, end: Position) -> float:
    """"""
    return np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)


def is_position_allowed(position: Position, obstacles: list[Position], grid_size: int) -> bool:
    """"""
    if 0 <= position.x < grid_size and 0 <= position.y < grid_size:
        for obstacle in obstacles:
            if position.x == obstacle.x and position.y == obstacle.y:
                return False
        return True
    return False


def is_node_in_list(node: Position, node_list: list[Position]) -> bool:
    """"""
    for closed_node in node_list:
        if node.x == closed_node.x and node.y == closed_node.y:
            return True
    return False


def calculate_cost(
        start: Position,
        current_position: Position,
        goal: Position,
        direction: Optional[str] = None
) -> NodeCost:
    """"""
    path_cost = distance_to_point(start=start, end=current_position)
    heuristic_cost = distance_to_point(start=current_position, end=goal)
    return NodeCost(position=current_position, path_cost=path_cost, heuristic_cost=heuristic_cost, direction=direction)


def get_node_idx_with_lowest_cost(node_costs: list[NodeCost]) -> int:
    """"""
    lowest_idx, _ = min(enumerate(node_costs), key=lambda x: x[1].total_cost)
    return lowest_idx


def is_current_goal(current: Position, goal: Position) -> bool:
    """"""
    if current.x == goal.x and current.y == goal.y:
        return True
    return False


def get_a_star_path(
        goal: Position,
        start: Position,
        obstacles: list[Position],
        direction: str,
        grid_size: int,
        canvas: npt.NDArray
):
    """"""
    node_cost = calculate_cost(start=start, current_position=start, goal=goal)
    open_nodes = [start]
    node_costs = [node_cost]
    closed_nodes = []
    current_node = open_nodes[0]

    possible_directions = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]
    shuffle(possible_directions)
    path_to_current = ","
    possible_paths = [path_to_current]

    path = []
    while True:
        # No route available
        if not open_nodes and closed_nodes:
            for direction in possible_directions:
                neighbour_node = Position(
                    x=current_node.x + Direction.get_x_step(direction),
                    y=current_node.y + Direction.get_y_step(direction)
                )
                if is_position_allowed(position=neighbour_node, obstacles=obstacles, grid_size=grid_size):
                    return [direction]

            return [
                np.random.choice(
                    [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                )
            ]

        lowest_idx = get_node_idx_with_lowest_cost(node_costs=node_costs)
        current_node = open_nodes[lowest_idx]
        closed_nodes.append(current_node)

        path_to_current = possible_paths[lowest_idx]
        possible_paths.pop(lowest_idx)

        if node_costs[lowest_idx].direction:
            path.append(node_costs[lowest_idx].direction)

        open_nodes.pop(lowest_idx)

        # cv2.rectangle(
        #     canvas,
        #     (int(current_node.x * 10) + 3, int(current_node.y * 10) + 3),
        #     (int(current_node.x * 10) + 8, int(current_node.y * 10) + 8),
        #     (255, 0, 255),
        #     -1
        # )
        #
        # for node in open_nodes:
        #     # if not node.x == start.x and node.y == start.y:
        #     cv2.rectangle(
        #         canvas,
        #         (int(node.x * 10) + 3, int(node.y * 10) + 3),
        #         (int(node.x * 10) + 8, int(node.y * 10) + 8),
        #         (255, 0, 0),
        #         -1
        #     )
        #
        # cv2.imshow('jaa', canvas)
        # cv2.waitKey(5)
        # time.sleep(0.05)

        node_costs.pop(lowest_idx)

        if is_current_goal(current=current_node, goal=goal):
            break

        # Check neighbouring nodes
        for direction in possible_directions:
            neighbour_node = Position(
                x=current_node.x + Direction.get_x_step(direction),
                y=current_node.y + Direction.get_y_step(direction)
            )
            if not is_position_allowed(position=neighbour_node, obstacles=obstacles, grid_size=grid_size):
                continue

            if is_node_in_list(node=neighbour_node, node_list=closed_nodes):
                continue

            if not is_node_in_list(node=neighbour_node, node_list=open_nodes):
                neighbour_node_cost = calculate_cost(
                    start=start,
                    current_position=neighbour_node,
                    goal=goal,
                    direction=direction
                )
                open_nodes.append(neighbour_node)
                node_costs.append(neighbour_node_cost)
                possible_paths.append(path_to_current + f"{direction}, ")

    final_path = []
    for step in path_to_current.split(',')[1:-1]:
        final_path.append(step.strip())

    return final_path


def main():
    args = parse_arguments()
    snake = Snake()
    env = BasicGame(grid_size=args.grid_size, snake=snake, show_game=True)
    env.reset_game()
    env.draw_elements()
    while True:
        path = get_a_star_path(
            goal=env.food,
            start=env.snake.head_position,
            obstacles=env.snake.body,
            direction=env.snake.direction,
            grid_size=args.grid_size,
            canvas=env.current_canvas
        )

        for i in range(0, len(path)):
            direction = path[i]
            env.snake.update_direction(new_direction=direction)
            env.update_game()

            if env.game_lost:
                env.reset_game()
                env.draw_elements()
                break

if __name__ == '__main__':
    main()
