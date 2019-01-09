import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import logging
import os
import time

import cdqn_logging
import experience_replay
import gym
import image_preprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# ACTION_MEANING = {
#     0 : "NOOP",
#     1 : "FIRE",
#     2 : "UP",
#     3 : "RIGHT",
#     4 : "LEFT",
#     5 : "DOWN",
#     6 : "UPRIGHT",
#     7 : "UPLEFT",
#     8 : "DOWNRIGHT",
#     9 : "DOWNLEFT",
#     10 : "UPFIRE",
#     11 : "RIGHTFIRE",
#     12 : "LEFTFIRE",
#     13 : "DOWNFIRE",
#     14 : "UPRIGHTFIRE",
#     15 : "UPLEFTFIRE",
#     16 : "DOWNRIGHTFIRE",
#     17 : "DOWNLEFTFIRE",
# }
from cdqn_logging import logger
from gym import wrappers


class CNN(nn.Module):

    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

    def count_neurons(self, image_dim):
        x = torch.rand(1, *image_dim)
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SoftmaxBody(nn.Module):

    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=1, dtype=torch.float32)
        actions = probs.multinomial(num_samples=1)
        return actions


class AI:

    def __init__(self, brain, body, device) -> None:
        super().__init__()
        self.brain = brain
        self.body = body
        self.device = device

    def __call__(self, inputs):
        transformed = torch.from_numpy(np.array(inputs, dtype=np.float32))
        outputs = self.brain(transformed.to(device))
        actions = self.body(outputs.to(device))
        return actions.cpu().numpy()


class MA:

    def __init__(self, size) -> None:
        super().__init__()
        self.list_of_rewards = []
        self.size = size

    def add(self, cumul_reward):
        if isinstance(cumul_reward, list):
            self.list_of_rewards += cumul_reward
        else:
            self.list_of_rewards.append(cumul_reward)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


def eligibility_trace(batch, cnn, device):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        if len(series) == 0:
            continue
        input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
        output = cnn(input.to(device))
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + cumul_reward * gamma
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


def video_callable(episode_id):
    return True


def crop_image(img: np.ndarray):
    img_shape = img.shape
    cropped = img[5:img_shape[0] - 17, 16:img_shape[1] - 16]
    return cropped


def get_filehandler(file_dir):
    formatter = logging.Formatter(cdqn_logging.cdqn_logformat)
    file_handler = logging.FileHandler(os.path.join(file_dir, 'activity.log'), 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    return file_handler


videos_dir = str(time.time())
root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'videos'))
latest_path = os.path.join(root_dir, 'latest')
videos_path = os.path.realpath(os.path.join(root_dir, videos_dir))
os.mkdir(videos_path, 0o775)
logger.addHandler(get_filehandler(videos_path))

logger.info('Loading env')
env = gym.make('Alien-v0')
logger.info('env loaded')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger.info("I will use the device {}".format(device))

env = image_preprocessing.PreprocessImage(env, 80, 80, True, crop_image)

env = wrappers.Monitor(env, videos_path, video_callable=video_callable)

if os.path.exists(latest_path):
    os.unlink(latest_path)
os.symlink(videos_path, latest_path)

cnn = CNN(env.action_space.n)
cnn.to(device)
body = SoftmaxBody(0.5)
body.to(device)
ai = AI(brain=cnn, body=body, device=device)

n_steps = experience_replay.NStepProgress(env=env, ai=ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=1000, logger=logger)

ma = MA(100)

loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
nb_epochs = 50

for epoch in range(1, nb_epochs + 1):
    epoch_start = datetime.datetime.now()
    logger.info('Starting epoch {}'.format(epoch))
    logger.info('Running games')
    start = datetime.datetime.now()
    memory.run_games(3)
    end = datetime.datetime.now()
    reward_steps = n_steps.rewards_steps()
    ma.add(reward_steps)
    logger.info(
        'Games done in : {}s, avg score {}, min {}, max {}'.format((end - start).total_seconds(), np.mean(reward_steps),
                                                                   np.min(reward_steps), np.max(reward_steps)))
    for idx, batch in enumerate(memory.sample_batch(128)):
        start = datetime.datetime.now()
        inputs, targets = eligibility_trace(batch, cnn)
        predictions = cnn(inputs.to(device))
        loss_error = loss(predictions.to(device), targets.to(device))
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
        end = datetime.datetime.now()
        logger.info('Batch {} done'.format(idx))

    avg_reward = ma.average()
    epoch_end = datetime.datetime.now()
    logger.info(
        'Epoch: {}, Average reward: {}, in {}s'.format(epoch, avg_reward, (epoch_end - epoch_start).total_seconds()))

env.close()
