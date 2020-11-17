import argparse
import os
import shutil
import copy
from random import random, randint, sample

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import *
from tensorboardX import SummaryWriter

from lalaDQN import lalaDQN
from lalaPlay import lalaSnake
from collections import deque

host = 'localhost'
port = 8080

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play lalaSnake""")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--initial_epsilon", type=float, default=0.05)
    parser.add_argument("--final_epsilon", type=float, default=0.02)
    parser.add_argument("--num_decay_epochs", type=float, default=20000)
    parser.add_argument("--num_epochs", type=int, default=50000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--plot_interval", type=int, default=10000)
    parser.add_argument("--replay_memory_size", type=int, default=100000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--target_update_cycle", type=int, default=50)

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    env = lalaSnake(host, port)
    # model = lalaDQN()
    if torch.cuda.is_available():
        model = torch.load("saved_models/lalasnake_80000")
    else:
        model = torch.load("saved_models/lalasnake_80000", map_location=lambda storage, loc: storage)
    # target_model = lalaDQN()
    if torch.cuda.is_available():
        target_model = torch.load("saved_models/lalasnake_80000")
    else:
        target_model = torch.load("saved_models/lalasnake_80000", map_location=lambda storage, loc: storage)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.get_state()

    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    score_list = []
    loss_list = []
    temp_score_list = []
    temp_loss_list = []
    while epoch < opt.num_epochs:

        if torch.cuda.is_available():
            state = state.cuda()

        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon

        model.eval()
        with torch.no_grad():
            prediction = model(state)
        model.train()

        if random_action:
            action = randint(0, 3)
        else:
            action = torch.argmax(prediction).item()

        reward, next_state, done = env.step(action)

        if torch.cuda.is_available():
            next_state = next_state.cuda()

        if not env.is_error:
            replay_memory.append([state, reward, next_state, done])

        if done:

            final_score = env.score
            final_fruits = env.fruits
            final_steps = env.steps

            env.reset()
            state = env.get_state()

            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch)).squeeze(1)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).squeeze(1)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = torch.max(model(state_batch), 1).values.unsqueeze(1)
        target_model.eval()
        with torch.no_grad():
            next_prediction_batch = torch.max(target_model(next_state_batch), 1).values.unsqueeze(1)
        target_model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction
                  for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        if epoch % opt.target_update_cycle == 0:
            state_dict = copy.deepcopy(model.state_dict())
            target_model.load_state_dict(state_dict)

        print("Epoch: {}/{}, Action: {}, Score: {}, Fruits: {}, Steps: {}, Predictions: {}, Loss: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_fruits,
            final_steps,
            np.around(prediction.cpu().numpy()),
            loss))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Fruits', final_fruits, epoch - 1)
        writer.add_scalar('Train/Steps', final_steps, epoch - 1)

        temp_score_list.append(final_score)
        temp_loss_list.append(loss)

        if final_fruits > 150 or final_steps > 200:
            torch.save(model, "{}/GOOD_lalasnake_{}".format(opt.saved_path, epoch))

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/lalasnake_{}".format(opt.saved_path, epoch))
            score_list.append(torch.mean(torch.FloatTensor(temp_score_list)))
            loss_list.append(torch.mean(torch.FloatTensor(temp_loss_list)))
            temp_score_list = []
            temp_loss_list = []
            if epoch % opt.plot_interval == 0:
                x = np.linspace(1, len(score_list), len(loss_list))
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax1.plot(x, score_list)
                ax2.plot(x, loss_list)
                plt.savefig("figs/plot_{}.png".format(epoch))
                plt.close()
                # score_list = []
                # loss_list = []

    torch.save(model, "{}/lalasnake_final".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)