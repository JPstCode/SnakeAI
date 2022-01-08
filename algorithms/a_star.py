import numpy as np

route_map = {'1': 'RIGHT',
             '2': 'DOWN',
             '3': 'LEFT',
             '4': 'UP'}


def get_node_idx_with_lowest_f_cost(open_nodes):

    return int(np.argmin(np.asarray(open_nodes)[:, -2]))


def distance_to_point(start, end):

    steps_x = abs(start[0] - end[0])
    steps_y = abs(start[1] - end[1])

    dst = steps_x + steps_y

    return dst


def get_f_cost(start, current, goal):

    # G cost = distance from starting point
    # H cost = Distance from end node
    # F cost = G cost + H cost

    g_cost = distance_to_point(start, current)
    h_cost = distance_to_point(current, goal)
    f_cost = g_cost + h_cost

    return g_cost, h_cost, f_cost


def is_pos_allowed(pos, obstacles, frame_x, frame_y):

    if pos not in obstacles:
        if 0 <= pos[0] < frame_x:
            if 0 <= pos[1] < frame_y:
                return True

    return False


def a_star_path(goal, starting_point, obstacles, frame_x, frame_y):

    goal_reached = False

    # Calculate g, h and f cost of starting node
    g_cost, h_cost, f_cost = get_f_cost(starting_point, starting_point, goal)

    # Add starting node to the open list
    # node = np.array([starting_point[0], starting_point[1], g_cost, h_cost, f_cost, '1'])
    node = [starting_point[0], starting_point[1]]
    open_nodes = [node]
    node_costs = [np.array([g_cost, h_cost, f_cost, ''])]

    closed_nodes = []

    possible_directions = [[10, 0, '1', 'RIGHT'],
                           [0, 10, '2', 'DOWN'],
                           [-10, 0, '3', 'LEFT'],
                           [0, -10, '4', 'UP']]

    path_to_current = ""
    while not goal_reached:

        # No Route available
        if not open_nodes and closed_nodes:
            return [np.random.choice(['RIGHT', 'DOWN', 'LEFT', 'UP'])]

        # Node with lowest F score
        lowest_idx = get_node_idx_with_lowest_f_cost(node_costs)

        # x, y, g, h, f
        current = open_nodes[lowest_idx]
        path_to_current = node_costs[lowest_idx][3]

        # Remove current from open list
        open_nodes.pop(lowest_idx)
        node_costs.pop(lowest_idx)

        # Add current to closed list
        closed_nodes.append(current)

        if current == goal:
            goal_reached = True
            break

        # Check neighbouring nodes
        for direction in possible_directions:
            neighbour = [current[0] + direction[0], current[1] + direction[1]]
            # print()
            # Check if neighbour is accessible, or already "visited"
            if not is_pos_allowed(neighbour, obstacles, frame_x, frame_y):
                continue

            if neighbour in closed_nodes:
                continue

            # If new path is shorter ? relates to diagonal movement?

            if neighbour not in open_nodes:
                n_g_cost, n_h_cost, n_f_cost = get_f_cost(starting_point, neighbour, goal)
                open_nodes.append(neighbour)
                node_costs.append(np.array([n_g_cost, n_h_cost, n_f_cost, path_to_current + direction[2]]))

    commands = []
    for step in path_to_current:
        commands.append(route_map[step])

    return commands