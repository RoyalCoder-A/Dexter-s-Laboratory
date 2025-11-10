import numpy as np


class ReplyBuffer:
    def __init__(self, max_size: int, num_state: int) -> None:
        self.max_size = max_size
        self.counter = 0
        self.states = np.zeros(shape=(max_size, num_state), dtype=np.float64)
        self.actions = np.zeros(shape=(max_size,), dtype=np.long)
        self.rewards = np.zeros(shape=(max_size,), dtype=np.float64)
        self.states_ = np.zeros(shape=(max_size, num_state), dtype=np.float64)
        self.terminals = np.zeros(shape=(max_size,), dtype=np.bool)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        state_: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.counter % self.max_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.states_[idx] = state_
        self.terminals[idx] = done
        self.counter += 1
