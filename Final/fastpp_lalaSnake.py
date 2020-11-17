# -*- coding: utf-8 -*
import numpy as np
import socket
from random import randint
import signal
from contextlib import contextmanager

default_action_table = ['s', 'd', 'w', 'a']
default_direction_table = [[1, 0], [0, 1], [-1, 0], [0, -1]]
port = 8080
LIMIT_TIME = 3
SCOPE_DELAY = 5
VIEW_CAPACITY = 19
SREACH_LENTH = 80
ESCAPE_LENTH = 50
PRIORITY_THRESHOLD = 36
SUICIDE_THRESHOLD = 24


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, val):
        self.queue.insert(0, val)

    def dequeue(self):
        if self.is_empty():
            return None
        else:
            return self.queue.pop()

    def append(self, val):
        self.queue.append(val)

    def dequeue_with_index(self, index):
        if self.is_empty():
            return None
        else:
            return self.queue.pop(index)

    def size(self):
        return len(self.queue)

    def is_empty(self):
        return self.size() == 0


class Node:
    def __init__(self, point):
        self.pos = point.pos
        self.previous_node = None
        self.action = None
        self.distance = 0
        self.manhattan = 0


class Point:
    def __init__(self, x, y):
        self.pos = [x, y]


def get_state(ts):
    # ts.send(b'g')
    message = str(ts.recv(1024), encoding="utf-8")
    if message == '':
        return None
    return eval(message)


def do_action(ts, action):
    ts.send(bytes('{}'.format(action_table[action]), encoding="utf8"))


def get_neighbors(pos):
    global priority
    bias = 1
    up = [pos[0] - bias, pos[1]]
    down = [pos[0] + bias, pos[1]]
    left = [pos[0], pos[1] - bias]
    right = [pos[0], pos[1] + bias]

    if pos[0] < 1:
        up = [0, 0]
    elif pos[0] > 97:
        down = [0, 0]

    if pos[1] < 1:
        left = [0, 0]
    elif pos[1] > 97:
        right = [0, 0]

    if priority == 0:
        return [down, right, up, left]
    elif priority == 1:
        return [right, up, left, down]
    elif priority == 2:
        return [up, left, down, right]
    else:
        return [left, down, right, up]


def recreate_path_for_node(node):
    path = []
    action = None
    while node.previous_node:
        path.append(node.pos)
        action = node.action
        node = node.previous_node
    path.reverse()
    return path, action


def is_scope(x, y, len_x, len_y):
    return x > 4 or y > 4 or len_x - x > 5 or len_y - y > 5


def not_all_deny(map, head_pos, deny_list):
    neighbors = get_neighbors(head_pos)
    for neighbor in neighbors:
        if map[neighbor[0], neighbor[1]] in deny_list:
            return True
    return False


def cal_manhattan(start, target):
    x = abs(start[0] - target[0])
    y = abs(start[1] - target[1])
    return x + y


def shortest_path(view, start, goal, deny_list):
    shortest_path = []
    next_action = None
    queue = Queue()
    start_node = Point(start[0], start[1])
    start_node = Node(start_node)
    queue.enqueue(start_node)
    visited_nodes = [start_node.pos]

    while not queue.is_empty():
        current_node = queue.dequeue()

        if view[current_node.pos[0], current_node.pos[1]] == goal:
            shortest_path, next_action = recreate_path_for_node(current_node)
            break

        for action, neighbor_pos in enumerate(get_neighbors(current_node.pos)):
            neighbor = view[neighbor_pos[0], neighbor_pos[1]]
            if neighbor not in deny_list:
                child_node_point = Point(neighbor_pos[0], neighbor_pos[1])
                child_node = Node(child_node_point)
                child_node.action = action
                child_node.previous_node = current_node
                if child_node.pos not in visited_nodes:
                    visited_nodes.append(child_node.pos)
                    queue.enqueue(child_node)

    return shortest_path, next_action


def A_star(view, start, target, deny_list, max_range):
    shortest_path = []
    next_action = None

    open_set = []
    open_dis_set = []
    close_set = []
    queue = Queue()
    
    start_node = Point(start[0], start[1])
    start_node = Node(start_node)
    start_node.manhattan = cal_manhattan(start, target)
    queue.append(start_node)
    open_set.append(start_node.pos)
    open_dis_set.append(start_node.distance + start_node.manhattan)

    while open_dis_set:
        min_index = open_dis_set.index(min(open_dis_set))
        _ = open_set.pop(min_index)
        _ = open_dis_set.pop(min_index)
        current_node = queue.dequeue_with_index(min_index)
        close_set.append(current_node.pos)

        if current_node.pos == target:
            shortest_path, next_action = recreate_path_for_node(current_node)
            break

        if len(open_set) > max_range:
            shortest_path, next_action = recreate_path_for_node(current_node)
            break

        for action, neighbor_pos in enumerate(get_neighbors(current_node.pos)):
            neighbor = view[neighbor_pos[0], neighbor_pos[1]]
            if neighbor in deny_list or neighbor_pos in close_set:
                continue

            if cal_manhattan(neighbor_pos, target) < current_node.manhattan or neighbor_pos not in open_set:
                child_node_point = Point(neighbor_pos[0], neighbor_pos[1])
                child_node = Node(child_node_point)
                child_node.action = action
                child_node.previous_node = current_node
                child_node.distance = current_node.distance + 1
                child_node.manhattan = cal_manhattan(neighbor_pos, target)

                if child_node.pos not in open_set:
                    queue.append(child_node)
                    open_set.append(child_node.pos)
                    open_dis_set.append(child_node.distance + child_node.manhattan)

    return shortest_path, next_action


def longest_path(view, start, goal, deny_list):
    neighbors = get_neighbors(start)

    up = neighbors[0]
    left = neighbors[1]
    down = neighbors[2]
    right = neighbors[3]

    target_x, target_y = np.where(view == goal)
    target = [target_x[0], target_y[0]]

    if view[up[0], up[1]] not in deny_list:
        up_path, _ = A_star(view, up, target, deny_list, ESCAPE_LENTH)
    else:
        up_path = [] # denied
    if view[left[0], left[1]] not in deny_list:
        left_path, _ = A_star(view, left, target, deny_list, ESCAPE_LENTH)
    else:
        left_path = [] # denied
    if view[down[0], down[1]] not in deny_list:
        down_path, _ = A_star(view, down, target, deny_list, ESCAPE_LENTH)
    else:
        down_path = [] # denied
    if view[right[0], right[1]] not in deny_list:
        right_path, _ = A_star(view, right, target, deny_list, ESCAPE_LENTH)
    else:
        right_path = [] # denied

    if not up_path and not left_path and not down_path and not right_path:
        return None, False

    length_list = [len(up_path), len(left_path), len(down_path), len(right_path)]

    return length_list.index(max(length_list)), True


def fill_other(map, view, head, snake):
    h_loc_x, h_loc_y = np.where(view == 12)  # head position in current view
    h_loc_x = h_loc_x[0]
    h_loc_y = h_loc_y[0]

    x0 = head[0] - h_loc_x
    y0 = head[1] - h_loc_y

    deny_list = [7, 8, 12]

    len_x, len_y = v.shape

    up_bound = view[0, :]
    left_bound = view[:, 0]
    down_bound = view[-1, :]
    right_bound = view[:, -1]

    snake_body = np.where(up_bound == snake)
    snake_body = snake_body[0]
    if snake_body.size > 1:
        min_pos = snake_body[0]
        max_pos = snake_body[-1]
        pos = np.linspace(min_pos, max_pos, max_pos - min_pos + 2)
        for index in pos:
            fill_pos = [int(x0 - 1), int(y0 + index)]
            if map[fill_pos[0], fill_pos[1]] not in deny_list:
                map[fill_pos[0], fill_pos[1]] = snake

    snake_body = np.where(left_bound == snake)
    snake_body = snake_body[0]
    if snake_body.size > 1:
        min_pos = snake_body[0]
        max_pos = snake_body[-1]
        pos = np.linspace(min_pos, max_pos, max_pos - min_pos + 2)
        for index in pos:
            fill_pos = [int(x0 + index), int(y0 - 1)]
            if map[fill_pos[0], fill_pos[1]] not in deny_list:
                map[fill_pos[0], fill_pos[1]] = snake

    snake_body = np.where(down_bound == snake)
    snake_body = snake_body[0]
    if snake_body.size > 1:
        min_pos = snake_body[0]
        max_pos = snake_body[-1]
        pos = np.linspace(min_pos, max_pos, max_pos - min_pos + 2)
        for index in pos:
            fill_pos = [int(x0 + len_x), int(y0 + index)]
            if map[fill_pos[0], fill_pos[1]] not in deny_list:
                map[fill_pos[0], fill_pos[1]] = snake

    snake_body = np.where(right_bound == snake)
    snake_body = snake_body[0]
    if snake_body.size > 1:
        min_pos = snake_body[0]
        max_pos = snake_body[-1]
        pos = np.linspace(min_pos, max_pos, max_pos - min_pos + 2)
        for index in pos:
            fill_pos = [int(x0 + index), int(y0 + len_y)]
            if map[fill_pos[0], fill_pos[1]] not in deny_list:
                map[fill_pos[0], fill_pos[1]] = snake

    return map


def avoid_others(map, view, head):
    h_loc_x, h_loc_y = np.where(view == 12)  # head position in current view
    h_loc_x = h_loc_x[0]
    h_loc_y = h_loc_y[0]

    x0 = head[0] - h_loc_x
    y0 = head[1] - h_loc_y

    deny_list = [7, 8, 12, 1, 2, 9, 10, 11, 13, 14, 15]

    other_snake_x, other_snake_y = np.where(view == 13)
    if other_snake_x.size:
        snake_loc = [int(x0 + other_snake_x), int(y0 + other_snake_y)]
        for neighbor in get_neighbors(snake_loc):
            if map[neighbor[0], neighbor[1]] not in deny_list:
                map[neighbor[0], neighbor[1]] = 1
            for avoid in get_neighbors(neighbor):
                if map[avoid[0], avoid[1]] not in deny_list:
                    map[avoid[0], avoid[1]] = 4

    other_snake_x, other_snake_y = np.where(view == 14)
    if other_snake_x.size:
        snake_loc = [int(x0 + other_snake_x), int(y0 + other_snake_y)]
        for neighbor in get_neighbors(snake_loc):
            if map[neighbor[0], neighbor[1]] not in deny_list:
                map[neighbor[0], neighbor[1]] = 1
            for avoid in get_neighbors(neighbor):
                if map[avoid[0], avoid[1]] not in deny_list:
                    map[avoid[0], avoid[1]] = 4

    other_snake_x, other_snake_y = np.where(view == 15)
    if other_snake_x.size:
        snake_loc = [int(x0 + other_snake_x), int(y0 + other_snake_y)]
        for neighbor in get_neighbors(snake_loc):
            if map[neighbor[0], neighbor[1]] not in deny_list:
                map[neighbor[0], neighbor[1]] = 1
            for avoid in get_neighbors(neighbor):
                if map[avoid[0], avoid[1]] not in deny_list:
                    map[avoid[0], avoid[1]] = 4

    return map


class Finder:
    def __init__(self, global_map, snake_list, tail_stay_counter):
        self.global_map = global_map
        self.snake_list = snake_list
        self.tail_stay_counter = tail_stay_counter

    def build_search_view(self, search_size):
        search_view = self.global_map[max(head_x - search_size + 1, 0): min(head_x + search_size, 99),
                      max(head_y - search_size + 1, 0): min(head_y + search_size, 99)].copy()

        search_view[0, :] = 1
        search_view[:, 0] = 1
        search_view[-1, :] = 1
        search_view[:, -1] = 1

        start_x, start_y = np.where(search_view == 12)
        start_x = start_x[0]
        start_y = start_y[0]
        self.start = [start_x, start_y]
        self.bias_x = head_x - start_x
        self.bias_y = head_y - start_y
        self.search_view = search_view

    def search(self, goal, step):
        goal_x, _ = np.where(self.search_view == goal)
        if goal == 3:
            worth_condition = 0
        elif goal == 6:
            worth_condition = 5
        else:
            worth_condition = 1

        if goal_x.size: # has goal around
            temp_reward = 0
            deny_list = [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            short_path, next_action = shortest_path(self.search_view, self.start, goal, deny_list)
            if not short_path: # not found, search again
                deny_list = [1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                short_path, next_action = shortest_path(self.search_view, self.start, goal, deny_list)
            if short_path:
                if step:
                    virtual_map = self.global_map.copy()
                    virtual_snake = self.snake_list.copy()
                    temp_tail_counter = self.tail_stay_counter
                    for pos in short_path:
                        virtual_snake.append([pos[0] + self.bias_x,
                                              pos[1] + self.bias_y])
                        if temp_tail_counter > 0:
                            temp_tail_counter -= 1
                        else:
                            _ = virtual_snake.pop(0)
                        got = virtual_map[pos[0] + self.bias_x,
                                          pos[1] + self.bias_y]
                        if got == 5:
                            temp_tail_counter += 1
                            temp_reward += 1
                        elif got == 6:
                            temp_tail_counter += 5
                            temp_reward += 5
                        elif got == 4:
                            temp_reward -= 1
                    if temp_reward >= worth_condition: # worth to walk
                        for pos in self.snake_list:
                            virtual_map[pos[0], pos[1]] = 0
                        for pos in virtual_snake:
                            virtual_map[pos[0], pos[1]] = 8
                        virtual_map[virtual_snake[0][0], virtual_snake[0][1]] = 7
                        virtual_map[virtual_snake[-1][0], virtual_snake[-1][1]] = 12
                        virtual_start = [virtual_snake[-1][0], virtual_snake[-1][1]]
                        target_x, target_y = np.where(virtual_map == 7)
                        target = [target_x[0], target_y[0]]
                        deny_list = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15]
                        virtual_path, _ = A_star(virtual_map, virtual_start, target, deny_list, SREACH_LENTH)
                        if len(virtual_path) == 1 and len(virtual_snake) == 2:
                            virtual_path = []
                        if len(virtual_path) > temp_tail_counter:
                            return next_action, True
                else:
                    return next_action, True

        return None, False


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen()
    ts, addr = s.accept()

    view_list = []
    head_list = []
    snake_list = []
    tail_stay_counter = 1
    step = -1
    scope_count = 0
    bad_fruit_counter = 0

    while True:
        data = get_state(ts)
        if not data:
            ts.close()
            break
        step += 1

        head_x, head_y, view = data['x'], data['y'], np.array(data['view'])

        if head_x < PRIORITY_THRESHOLD and head_y < PRIORITY_THRESHOLD:
            priority = 0
        elif head_x >= PRIORITY_THRESHOLD and head_y < PRIORITY_THRESHOLD:
            priority = 1
        elif head_x >= PRIORITY_THRESHOLD and head_y >= PRIORITY_THRESHOLD:
            priority = 2
        else:
            priority = 3

        action_table = default_action_table[priority:] + default_action_table[:priority]
        direction_table = default_direction_table[priority:] + default_direction_table[:priority]

        print("Position:")
        print("x:{}, y:{}".format(head_x, head_y))

        x_in_view, y_in_view = np.where(view == 12)
        x_in_view = x_in_view[0]
        y_in_view = y_in_view[0]

        if scope_count:
            scope_count -= 1
        scope = is_scope(x_in_view, y_in_view, view.shape[0], view.shape[1])
        if scope:
            scope_count = SCOPE_DELAY

        print("View:")
        print(view)

        global_map = np.zeros((99, 99))
        global_map[0, :] = 1
        global_map[:, 0] = 1
        global_map[-1, :] = 1
        global_map[:, -1] = 1

        head_list.append([head_x, head_y])
        view_list.append(view)
        if len(head_list) > VIEW_CAPACITY and len(view_list) > VIEW_CAPACITY:
            _ = head_list.pop(0)
            _ = view_list.pop(0)

        snake_list.append([head_x, head_y])

        for i in range(len(head_list)):
            h = head_list[i]  # global head position
            v = view_list[i]  # the view correspond to the head
            h_loc_x, h_loc_y = np.where(v == 12)  # head position in current view
            h_loc_x = h_loc_x[0]
            h_loc_y = h_loc_y[0]

            x0 = h[0] - h_loc_x
            y0 = h[1] - h_loc_y
            len_x, len_y = v.shape

            global_map[int(x0): int(x0 + len_x), int(y0): int(y0 + len_y)] = v

        global_map = fill_other(global_map, view_list[-1], head_list[-1], 9)
        global_map = fill_other(global_map, view_list[-1], head_list[-1], 10)
        global_map = fill_other(global_map, view_list[-1], head_list[-1], 11)

        global_map = avoid_others(global_map, view_list[-1], head_list[-1])

        viewed_body = np.where(global_map == 8)
        viewed_body = [[viewed_body[0][i], viewed_body[1][i]] for i in range(len(viewed_body[0]))]
        for body_pos in viewed_body:
            global_map[body_pos[0], body_pos[1]] = 0  # just clear the viewed body first, fix later

        for i, body_pos in enumerate(snake_list):
            if i == len(snake_list) - 1: # head
                continue
            elif i == 0:
                global_map[body_pos[0], body_pos[1]] = 7  # tail
            else:
                global_map[body_pos[0], body_pos[1]] = 8 # body

        print("Global Map:")
        print(global_map[max(head_x - 9, 0): min(head_x + 10, 99), # crop around
              max(head_y - 9, 0): min(head_y + 10, 99)].astype(int))

        has_five_fruits, _ = np.where(view == 6)
        has_one_fruit, _ = np.where(view == 5)

        if bad_fruit_counter > SUICIDE_THRESHOLD and not has_five_fruits.size and not has_one_fruit.size:
            next_action = randint(0, 3)

            do_action(ts, next_action) # random walk, hoping suicide itself

            if tail_stay_counter > 0:
                tail_stay_counter -= 1
            else:
                _ = snake_list.pop(0)

            got = global_map[head_x + direction_table[next_action][0],
                             head_y + direction_table[next_action][1]]
            if got == 5:
                tail_stay_counter += 1
            elif got == 6:
                tail_stay_counter += 5
            elif got == 4:
                bad_fruit_counter += 1

            continue

        big_view = 0
        if scope_count:
            big_view = 1

        target_finder = Finder(global_map, snake_list, tail_stay_counter)
        # print("In 3x3:")
        target_finder.build_search_view(3)
        next_action, is_found = target_finder.search(3, step)
        if is_found:
            do_action(ts, next_action)
            if tail_stay_counter > 0:
                tail_stay_counter -= 1
            else:
                _ = snake_list.pop(0)
            got = global_map[head_x + direction_table[next_action][0],
                             head_y + direction_table[next_action][1]]
            if got == 5:
                tail_stay_counter += 1
            elif got == 6:
                tail_stay_counter += 5
            elif got == 4:
                bad_fruit_counter += 1
            continue
        else:
            next_action, is_found = target_finder.search(6, step)
            if is_found:
                do_action(ts, next_action)
                if tail_stay_counter > 0:
                    tail_stay_counter -= 1
                else:
                    _ = snake_list.pop(0)
                got = global_map[head_x + direction_table[next_action][0],
                                 head_y + direction_table[next_action][1]]
                if got == 5:
                    tail_stay_counter += 1
                elif got == 6:
                    tail_stay_counter += 5
                elif got == 4:
                    bad_fruit_counter += 1
                continue
            else:
                next_action, is_found = target_finder.search(5, step)
                if is_found:
                    do_action(ts, next_action)
                    if tail_stay_counter > 0:
                        tail_stay_counter -= 1
                    else:
                        _ = snake_list.pop(0)
                    got = global_map[head_x + direction_table[next_action][0],
                                     head_y + direction_table[next_action][1]]
                    if got == 5:
                        tail_stay_counter += 1
                    elif got == 6:
                        tail_stay_counter += 5
                    elif got == 4:
                        bad_fruit_counter += 1
                    continue

        if big_view:
            # print("In 15x15:")
            target_finder.build_search_view(9)
        else:
            # print("In 9x9:")
            target_finder.build_search_view(6)

        next_action, is_found = target_finder.search(3, step)
        if is_found:
            do_action(ts, next_action)
            if tail_stay_counter > 0:
                tail_stay_counter -= 1
            else:
                _ = snake_list.pop(0)
            got = global_map[head_x + direction_table[next_action][0],
                             head_y + direction_table[next_action][1]]
            if got == 5:
                tail_stay_counter += 1
            elif got == 6:
                tail_stay_counter += 5
            elif got == 4:
                bad_fruit_counter += 1
            continue
        else:
            next_action, is_found = target_finder.search(6, step)
            if is_found:
                do_action(ts, next_action)
                if tail_stay_counter > 0:
                    tail_stay_counter -= 1
                else:
                    _ = snake_list.pop(0)
                got = global_map[head_x + direction_table[next_action][0],
                                 head_y + direction_table[next_action][1]]
                if got == 5:
                    tail_stay_counter += 1
                elif got == 6:
                    tail_stay_counter += 5
                elif got == 4:
                    bad_fruit_counter += 1
                continue
            else:
                next_action, is_found = target_finder.search(5, step)
                if is_found:
                    do_action(ts, next_action)
                    if tail_stay_counter > 0:
                        tail_stay_counter -= 1
                    else:
                        _ = snake_list.pop(0)
                    got = global_map[head_x + direction_table[next_action][0],
                                     head_y + direction_table[next_action][1]]
                    if got == 5:
                        tail_stay_counter += 1
                    elif got == 6:
                        tail_stay_counter += 5
                    elif got == 4:
                        bad_fruit_counter += 1
                    continue

        if step:
            goal = 7
            temp_map = global_map.copy()
            try:
                with time_limit(LIMIT_TIME):
                    for i in range(len(snake_list)):
                        # print("Search -{} tail:".format(i))
                        deny_list = [1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15]
                        next_action, found = longest_path(temp_map, snake_list[-1], goal, deny_list)
                        if not found:
                            deny_list = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15]
                            next_action, found = longest_path(temp_map, snake_list[-1], goal, deny_list)
                            if not found:
                                if i < len(snake_list) - 1:
                                    temp_map[snake_list[i][0], snake_list[i][1]] = 8
                                    temp_map[snake_list[i + 1][0], snake_list[i + 1][1]] = 7
                                else:
                                    next_action = randint(0, 3)
                            else:
                                break
                        else:
                            break
            except TimeoutException as e:
                deny_list = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15]
                for next_action, neighbor in enumerate(get_neighbors(snake_list[-1])):
                    if global_map[neighbor[0], neighbor[1]] not in deny_list and not_all_deny(global_map, neighbor, deny_list):
                        break
        else:
            deny_list = [1, 2, 8, 9, 10, 11, 12, 13, 14, 15]
            for next_action, neighbor in enumerate(get_neighbors(snake_list[-1])):
                if global_map[neighbor[0], neighbor[1]] not in deny_list and not_all_deny(global_map, neighbor, deny_list):
                    break

        do_action(ts, next_action)
        if tail_stay_counter > 0:
            tail_stay_counter -= 1
        else:
            _ = snake_list.pop(0)
        got = global_map[head_x + direction_table[next_action][0],
                         head_y + direction_table[next_action][1]]
        if got == 5:
            tail_stay_counter += 1
        elif got == 6:
            tail_stay_counter += 5
        elif got == 4:
            bad_fruit_counter += 1
