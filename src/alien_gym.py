import warnings
from collections import deque, namedtuple

warnings.simplefilter(action='ignore', category=FutureWarning)

import datetime
import logging
import os
import time

import cdqn_logging
import experience_replay
import gym
from image_preprocessing import ImageSize, PreprocessImage
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
from gym import wrappers


class CNN(nn.Module):

    def __init__(self, number_actions, image_w: int = 80, image_h: int = 80):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, image_w, image_h)), out_features=40)
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
        outputs = self.brain(transformed.to(self.device))
        actions = self.body(outputs.to(self.device))
        return actions.cpu().numpy()


class MA:

    def __init__(self, size) -> None:
        super().__init__()
        self.list_of_rewards = deque(maxlen=size)
        self.size = size

    def add(self, cumul_reward):
        if isinstance(cumul_reward, list):
            self.list_of_rewards += cumul_reward
        else:
            self.list_of_rewards.append(cumul_reward)

    def average(self):
        return np.mean(self.list_of_rewards)

    def min(self):
        return np.min(self.list_of_rewards)

    def max(self):
        return np.max(self.list_of_rewards)


Config = namedtuple('Config', ['nb_epoch',
                               'with_crop',
                               'with_color_pre',
                               'image_size',
                               'nb_games',
                               'softmax_temp',
                               'n_step',
                               'memory_capacity',
                               'optimizer_lr',
                               'gamma',
                               'record',
                               'is_train'])

AlienGymResult = namedtuple('AlienGymResult', ['config', 'final_mean', 'min', 'max', 'total_time', 'run_name'])

AlienGymCheckpoint = namedtuple('AlienGymCheckpoint',
                                ['config', 'model_state_dict', 'optimizer_state_dict', 'epoch', 'device'])
AlienGymAI = namedtuple('AlienGymAI', ['cnn', 'ai', 'loss', 'optimizer', 'n_step', 'replay_memory'])


class AlienGym:

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger('alien_gym')

    def eligibility_trace(self, batch, cnn, gamma):
        inputs = []
        targets = []
        for series in batch:
            if len(series) == 0:
                continue
            input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
            output = cnn(input.to(self.device))
            cumul_reward = 0.0 if series[-1].done else output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + cumul_reward * gamma
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

    @staticmethod
    def crop_image(img: np.ndarray):
        img_shape = img.shape
        cropped = img[5:img_shape[0] - 17, 16:img_shape[1] - 16]
        return cropped

    def prepare_env(self, config: Config, run_name: str):
        root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        latest_path = os.path.join(root_dir, 'latest')
        data_path = os.path.realpath(os.path.join(root_dir, run_name))
        os.mkdir(data_path, 0o775)

        if os.path.exists(latest_path):
            os.unlink(latest_path)
        os.symlink(data_path, latest_path)

        env = gym.make('Alien-v0')
        env = PreprocessImage(env, ImageSize.from_str(config.image_size), True, self.crop_image)
        if config.record is not None and callable(config.record):
            env = wrappers.Monitor(env, data_path, video_callable=config.record)
        return env, data_path

    def run(self, config: Config = None, run_number: int = 0) -> AlienGymResult:
        run_name = str(time.time())
        env, data_path = self.prepare_env(config=config, run_name=run_name)
        print(data_path)
        self.run_logger = cdqn_logging.create_runlogger(run_number=run_number, log_path=data_path,
                                                        filename=run_name)

        self.run_logger.info("I will use the device {}".format(self.device))
        self.run_logger.info('Using config : {}'.format(config))

        alien_ai: AlienGymAI = self.init_model(config, env)

        ma = MA(config.nb_epoch * config.nb_games)

        nb_epochs = config.nb_epoch
        total_chrono = datetime.datetime.now()

        for epoch in range(1, nb_epochs + 1):
            self.run_logger.info('Starting epoch {}'.format(epoch))
            start = datetime.datetime.now()
            alien_ai.replay_memory.run_games(config.nb_games)
            end = datetime.datetime.now()
            reward_steps = alien_ai.n_step.rewards_steps()
            ma.add(reward_steps)
            self.run_logger.info(
                'Games done in : {}s, avg score {}, min {}, max {}'.format((end - start).total_seconds(),
                                                                           np.mean(reward_steps),
                                                                           np.min(reward_steps), np.max(reward_steps)))
            if config.is_train:
                for batch in alien_ai.replay_memory.sample_batch(128):
                    inputs, targets = self.eligibility_trace(batch, alien_ai.cnn, config.gamma)
                    predictions = alien_ai.cnn(inputs.to(self.device))
                    loss_error = alien_ai.loss(predictions.to(self.device), targets.to(self.device))
                    alien_ai.optimizer.zero_grad()
                    loss_error.backward()
                    alien_ai.optimizer.step()

            avg_reward = ma.average()
            self.run_logger.info('Epoch: {}, Average reward: {}'.format(epoch, avg_reward))
            self.create_checkpoint(alien_ai, config, data_path, epoch, run_name)
        total_end = datetime.datetime.now()
        total_seconds = (total_end - total_chrono).total_seconds()
        env.close()
        return AlienGymResult(config=config, final_mean=ma.average(), min=ma.min(), max=ma.max(),
                              total_time=total_seconds, run_name=run_name)

    def create_checkpoint(self, alien_ai: AlienGymAI, config: Config, data_path: str, epoch: int, run_name: str):
        if not config.is_train:
            return
        checkpoint = AlienGymCheckpoint(config=config._asdict(), model_state_dict=alien_ai.cnn.state_dict(),
                                        optimizer_state_dict=alien_ai.optimizer.state_dict(), epoch=epoch,
                                        device=self.device)
        checkpoint_path = os.path.join(data_path, '{}_{}.pth'.format(run_name, epoch))
        self.run_logger.debug('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(checkpoint._asdict(), checkpoint_path)

    def init_model(self, config: Config, env) -> AlienGymAI:
        image_size: ImageSize = ImageSize.from_str(config.image_size)
        cnn = CNN(env.action_space.n, image_w=image_size.w, image_h=image_size.h)
        cnn.to(self.device)
        body = SoftmaxBody(config.softmax_temp)
        body.to(self.device)
        optimizer = optim.Adam(cnn.parameters(), lr=config.optimizer_lr)
        ai = AI(brain=cnn, body=body, device=self.device)
        n_steps = experience_replay.NStepProgress(env=env, ai=ai, n_step=config.n_step)
        memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=config.memory_capacity)
        return AlienGymAI(cnn=cnn, ai=ai, loss=nn.MSELoss(), optimizer=optimizer, n_step=n_steps, replay_memory=memory)
