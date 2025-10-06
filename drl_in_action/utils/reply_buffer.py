import numpy as np


class ReplyBuffer:
    def __init__(self, num_state: int, max_len: int):
        self.num_state = num_state
        self.max_len = max_len
        self.states = np.zeros((max_len, num_state), dtype=np.float64)
        self.actions = np.zeros((max_len,), dtype=np.int32)
        self.rewards = np.zeros((max_len,), dtype=np.float64)
        self.states_ = np.zeros((max_len, num_state), dtype=np.float64)
        self.terminals = np.zeros((max_len,), dtype=np.int8)
        self.counter = 0

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        state_: np.ndarray,
        terminate: bool,
    ):
        idx = self.counter % self.max_len
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.states_[idx] = state_
        self.terminals[idx] = int(terminate)
        self.counter += 1

    def sample(self, batch_size: int):
        max_len = min(self.max_len, self.counter)
        batches = np.random.choice(max_len, batch_size)
        return (
            self.states[batches],
            self.actions[batches],
            self.rewards[batches],
            self.states_[batches],
            self.terminals[batches],
        )
