import argparse
import torch
import os

from lalaPlay import lalaSnake

host = 'localhost'
port = 8080
seed = [2020, 623500535, 700383151, 507690622, 41420402]

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play lalaSnake""")

    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    env = lalaSnake(host, port)

    model_files = os.listdir("{}/{}".format(os.curdir, opt.saved_path))

    for file in model_files:
        if torch.cuda.is_available():
            model = torch.load("{}/{}".format(opt.saved_path, file))
        else:
            model = torch.load("{}/{}".format(opt.saved_path, file), map_location=lambda storage, loc: storage)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        print("\tTesting on {}: ".format(file))

        for s in range(len(seed)):
            state = env.get_state()
            if torch.cuda.is_available():
                state = state.cuda()
            print("Testing on seed: {}".format(seed[s]))
            while True:
                prediction = model(state)
                action = torch.argmax(prediction).item()
                reward, next_state, done = env.step(action)
                # print("Reward: {}".format(reward))
                if torch.cuda.is_available():
                    next_state = next_state.cuda()
                state = next_state

                if done:
                    env.reset()
                    # print("Score: {}, Fruits: {}, Steps: {}".format(env.score, env.fruits, env.steps))
                    break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
