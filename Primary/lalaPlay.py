import torch
import os
import time
import numpy as np
from socket import *

class lalaSnake:
    def __init__(self, host, port):
        self.reward_table = [3, -10, 4, 5, 1, 10, 50, 0,
                             -10, -10, -10, -10, 0, -10, -10, -10]
        self.seed_table = [2020, 623500535, 700383151, 507690622, 41420402]
        self.action_table = ['w', 'a', 's', 'd']
        self.target_table = [[3, 4], [4, 3], [5, 4], [4, 5]]
        self.skate_table = [[2, 4], [4, 2], [6, 4], [4, 6]]
        self.seed_count = 0
        self.seed_num = len(self.seed_table)
        self.host = host
        self.port = port
        self.reset()

    def reset(self):
        self.score = 0
        self.fruits = 0
        self.steps = 0
        self.skate = 0
        self.is_up_near = 0
        self.is_left_near = 0
        self.is_down_near = 0
        self.is_right_near = 0
        self.previous_aciton = 2
        self.is_trick = False
        self.is_error = False
        self.gameover = False
        self.state = torch.FloatTensor(torch.zeros(1, 1, 9, 9))
        seed = self.seed_table[self.seed_count % self.seed_num]
        os.system('./subg.client -s {} -p {} &'.format(seed, self.port))
        time.sleep(0.02)
        self.ts = socket(AF_INET, SOCK_STREAM)
        self.ts.connect((self.host, self.port))
        self.seed_count += 1

    def get_state(self):
        self.ts.send(b'g')

        try:
            message = str(self.ts.recv(1024),  encoding="utf8")
        except:
            self.gameover = True
            self.is_error = True
        else:
            if message == '':
                self.gameover = True
            else:
                message = eval(message)
                view = np.array(message['view'])
                self.is_up_near = message['x'] == 1
                self.is_left_near = message['y'] == 1
                self.is_down_near = message['x'] == 197
                self.is_right_near = message['y'] == 197
                h, w = view.shape
                head_pos = np.array(np.where(view == 12))
                head_x = head_pos[1] + 1
                head_y = head_pos[0] + 1
                up = head_pos[0]
                left = head_pos[1]
                right = w - head_x
                down = h - head_y
                up_fix = 7 - up
                left_fix = 7 - left
                down_fix = 7 - down
                right_fix = 7 - right
                up_pad = np.ones((int(up_fix), 15))
                left_pad = np.ones((int(h), int(left_fix)))
                right_pad = np.ones((int(h), int(right_fix)))
                down_pad = np.ones((int(down_fix), 15))

                state = np.concatenate((left_pad, view, right_pad), axis=1)
                state = np.concatenate((up_pad, state, down_pad), axis=0)
                self.state = torch.FloatTensor(state[3:-3, 3:-3]).unsqueeze(0).unsqueeze(0)

        return self.state

    def step(self, action):
        self.is_trick = False

        if self.previous_aciton == 0 and action == 2:
            action = 0
            self.is_trick = True
        elif self.previous_aciton == 1 and action == 3:
            action = 1
            self.is_trick = True
        elif self.previous_aciton == 2 and action == 0:
            action = 2
            self.is_trick = True
        elif self.previous_aciton == 3 and action == 1:
            action = 3
            self.is_trick = True

        if self.is_up_near and self.is_left_near:
            if action == 0 or action == 1:
                action = 2
                self.is_trick = True
        elif self.is_up_near:
            if action == 0:
                action = 2
                self.is_trick = True
        elif self.is_left_near:
            if action == 1:
                action = 3
                self.is_trick = True
        elif self.is_down_near and self.is_right_near:
            if action == 2 or action == 3:
                action = 0
                self.is_trick = True
        elif self.is_down_near:
            if action == 2:
                action = 0
                self.is_trick = True
        elif self.is_right_near:
            if action == 3:
                action = 1
                self.is_trick = True

        self.previous_aciton = action

        try:
            self.ts.send(bytes('{}'.format(self.action_table[action]), encoding="utf8"))
        except:
            reward = -10
            self.score += reward
            self.gameover = True
            self.ts.close()
            return reward, self.state, self.gameover
        else:
            target_x = self.target_table[action][0]
            target_y = self.target_table[action][1]
            reward = self.reward_table[int(self.state[0, 0, target_x, target_y])]
            if self.is_trick:
                reward = -15

        self.score += reward
        self.steps += 1

        _ = self.get_state()

        if not self.gameover:
            if reward == 1:
                self.fruits -= 1
            elif reward == 10:
                self.fruits += 1
            elif reward == 50:
                self.fruits += 5
            elif reward == -10: # Luckily not die, fix reward and score
                self.score -= reward
                reward = 3
                self.score += reward
            elif reward == 4:
                self.skate = 1
        else:
            self.ts.close()
            if not reward == -10: # Unfortunately died, fix reward and score
                self.score -= reward
                reward = -10
                self.score += reward

        plus_reward = 0

        if self.skate > 0:
            skate_x = self.skate_table[action][0]
            skate_y = self.skate_table[action][1]
            plus_reward = self.reward_table[int(self.state[0, 0, skate_x, skate_y])]
            self.score += plus_reward

            if not self.gameover:
                if plus_reward == 1:
                    self.fruits -= 1
                elif plus_reward == 10:
                    self.fruits += 1
                elif plus_reward == 50:
                    self.fruits += 5
                elif plus_reward == -10:  # Luckily not die, fix reward and score
                    self.score -= plus_reward
                    plus_reward = 3
                    self.score += plus_reward
                elif plus_reward == 4:
                    self.skate = 0
            else:
                if not plus_reward == -10:  # Unfortunately died, fix reward and score
                    self.score -= plus_reward
                    plus_reward = -10
                    self.score += plus_reward

            self.skate += 1
            if self.skate == 25:
                self.skate = 0

        return reward + plus_reward, self.state, self.gameover