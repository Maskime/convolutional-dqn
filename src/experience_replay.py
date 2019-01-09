# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


# Making the AI progress on several (n_step) steps

class NStepProgress:

    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
        self.game_rewards = []

    def __iter__(self):
        state = self.env.reset()
        history = deque(maxlen=self.n_step)
        reward = 0.0
        is_done = False
        while not is_done:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done))
            if (len(history) % self.n_step) == 0:
                self.game_rewards.append(reward)
                yield tuple(history)
                reward = 0.
                history.clear()
            state = next_state
        if (len(history) > 0) and (len(history) % self.n_step) != 0:
            self.game_rewards.append(reward)
            self.rewards.append(np.sum(self.game_rewards))
            self.game_rewards.clear()
            yield tuple(history)

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


# Implementing Experience Replay

class ReplayMemory:

    def __init__(self, n_steps, logger, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.buffer = deque(maxlen=capacity)
        self.logger = logger

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        batch_size = min(len(self.buffer), batch_size)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_games(self, nb_games):
        # for entry in self.n_steps_iter:
        game_number = 1
        while nb_games > 0:
            game_generator = iter(self.n_steps)
            for entry in game_generator:
                self.buffer.append(entry)  # we put 200 for the current episode
            game_number += 1
            nb_games -= 1
