from copy import deepcopy
from random import shuffle
from typing import Optional

import numpy as np

from game.data_structures import Position, Direction


class NodeCost:
    def __init__(self, path_cost: float, heuristic_cost: float, direction: Optional[str] = None):
        self.position = Position
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.total_cost = path_cost + heuristic_cost
        self.direction = direction


def distance_to_point(start: Position, end: Position) -> float:
    """Calculate Euclidean distance between points."""
    return np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)


def is_position_allowed(position: Position, obstacles: list[Position], grid_size: int) -> bool:
    """Check that position is not outside of grid and not inside of obstacles."""
    if 0 <= position.x < grid_size and 0 <= position.y < grid_size:
        for obstacle in obstacles:
            if position.x == obstacle.x and position.y == obstacle.y:
                return False
        return True
    return False


def is_node_in_list(node: Position, node_list: list[Position]) -> bool:
    """Check if node is in list."""
    for closed_node in node_list:
        if node.x == closed_node.x and node.y == closed_node.y:
            return True
    return False


def calculate_cost(
    start: Position, current_position: Position, goal: Position, direction: Optional[str] = None
) -> NodeCost:
    """Calculate cost for given node."""
    path_cost = distance_to_point(start=start, end=current_position)
    heuristic_cost = distance_to_point(start=current_position, end=goal)
    return NodeCost(path_cost=path_cost, heuristic_cost=heuristic_cost, direction=direction)


def get_node_idx_with_lowest_cost(node_costs: list[NodeCost]) -> int:
    """Return index of node with the lowest cost."""
    lowest_idx, _ = min(enumerate(node_costs), key=lambda x: x[1].total_cost)
    return lowest_idx


def is_current_goal(current: Position, goal: Position) -> bool:
    """Check if current position is equal to goal."""
    if current.x == goal.x and current.y == goal.y:
        return True
    return False


def get_random_direction(
    current_node: Position,
    obstacles: list[Position],
    possible_directions: list[str],
    grid_size: int,
) -> list[str]:
    """If no path routes are available, return random direction."""
    for direction in possible_directions:
        neighbour_node = Position(
            x=current_node.x + Direction.get_x_step(direction),
            y=current_node.y + Direction.get_y_step(direction),
        )
        if is_position_allowed(position=neighbour_node, obstacles=obstacles, grid_size=grid_size):
            return [direction]

    return [np.random.choice([Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP])]


def get_a_star_path(
    goal: Position,
    start: Position,
    obstacles: list[Position],
    grid_size: int,
) -> list[str]:
    """Use A* algorithm to find optimal path from goal to start."""
    node_cost = calculate_cost(start=start, current_position=start, goal=goal)
    open_nodes = [start]
    node_costs = [node_cost]
    closed_nodes = []

    possible_directions = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]
    shuffle(possible_directions)
    path_to_current = [None]
    possible_paths = [path_to_current]

    path = []
    while True:
        # No route available
        if not open_nodes and closed_nodes:
            return get_random_direction(
                current_node=start,
                obstacles=obstacles,
                possible_directions=possible_directions,
                grid_size=grid_size,
            )

        lowest_idx = get_node_idx_with_lowest_cost(node_costs=node_costs)
        current_node = open_nodes[lowest_idx]
        closed_nodes.append(current_node)

        path_to_current = possible_paths[lowest_idx]
        possible_paths.pop(lowest_idx)

        if node_costs[lowest_idx].direction:
            path.append(node_costs[lowest_idx].direction)

        open_nodes.pop(lowest_idx)
        node_costs.pop(lowest_idx)

        if is_current_goal(current=current_node, goal=goal):
            break

        # Check neighbouring nodes
        for direction in possible_directions:
            path_to_current_copy = deepcopy(path_to_current)
            neighbour_node = Position(
                x=current_node.x + Direction.get_x_step(direction),
                y=current_node.y + Direction.get_y_step(direction),
            )
            if not is_position_allowed(
                position=neighbour_node, obstacles=obstacles, grid_size=grid_size
            ):
                continue

            if is_node_in_list(node=neighbour_node, node_list=closed_nodes):
                continue

            if not is_node_in_list(node=neighbour_node, node_list=open_nodes):
                neighbour_node_cost = calculate_cost(
                    start=start, current_position=neighbour_node, goal=goal, direction=direction
                )
                open_nodes.append(neighbour_node)
                node_costs.append(neighbour_node_cost)
                path_to_current_copy.append(direction)
                possible_paths.append(path_to_current_copy)

    final_path = []
    for step in path_to_current:
        if step:
            final_path.append(step)

    return final_path
