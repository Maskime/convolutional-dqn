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

    def __iter__(self):
        state = self.env.reset()
        print('------------------------ RESET ------------------------')
        history = deque()
        reward = 0.0
        current_iter = 0
        while True:
            current_iter += 1
            final_layer = self.ai(np.array([state]))
            action = final_layer[0][0]
            next_state, r, is_done, _ = self.env.step(action)
            reward += r
            print('chosen action', action, 'reward', reward, 'iter', current_iter)
            history.append(Step(state=state, action=action, reward=r, done=is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
                self.rewards.append(reward)
                reward = 0.0
                history.clear()
            print('state', 'history length', len(history))
            state = next_state
            if is_done:
                print('env is done', 'iter', current_iter)
                if len(history) > self.n_step + 1:
                    history.popleft()
                yield tuple(history)
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                print('------------------------ RESET ------------------------')
                history.clear()
                current_iter = 0

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


# Implementing Experience Replay

class ReplayMemory:

    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size):  # creates an iterator that returns random batches
        print('buffer size', len(self.buffer))
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        batch_size = min(len(self.buffer), batch_size)
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def run_steps(self, samples):
        # for entry in self.n_steps_iter:
        print('Requested number of sample', samples)
        while samples > 0:
            print('sample counter', samples)
            entry = next(self.n_steps_iter)  # 10 consecutive steps
            self.buffer.append(entry)  # we put 200 for the current episode
            print('entry len', len(entry), 'buffer len', len(self.buffer))
            samples -= 1
        while len(self.buffer) > self.capacity:  # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
