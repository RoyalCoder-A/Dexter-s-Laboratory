import numpy as np


class Memory:
    def __init__(self, state_shape: tuple[int, ...], max_size: int):
        self.states = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.states_ = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.terminals = np.zeros(max_size, dtype=np.bool_)
        self.counter = 0
        self.max_size = max_size

    def remember(self, state, action, reward, state_, terminal):
        index = self.counter % self.max_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.terminals[index] = terminal
        self.counter += 1

    def sample(self, batch_size: int):
        max_size = min(self.counter, self.max_size)
        indices = np.random.choice(max_size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.states_[indices],
            self.terminals[indices],
        )
