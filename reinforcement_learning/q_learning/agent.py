import numpy as np


class Agent:

    def __init__(
        self,
        num_actions: int,
        eps: float,
        eps_min: float,
        eps_decay: float,
        discount: float,
        learning_rate: float,
    ) -> None:
        self.num_actions = num_actions
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q: dict[int, list[float]] = {}

    def get_action(self, state: int) -> int:
        action_values = self.Q.get(state, [0] * self.num_actions)
        if np.random.rand() < self.eps:
            return np.random.randint(self.num_actions)
        else:
            return int(np.argmax(action_values))

    def update_q_value(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        action_values = self.Q.get(state, [0] * self.num_actions)
        next_action_values = self.Q.get(next_state, [0] * self.num_actions)
        if done:
            target = reward
        else:
            target = reward + self.discount * np.max(next_action_values)
        action_values[action] += float(
            self.learning_rate * (target - action_values[action])
        )
        self.Q[state] = action_values
        self.eps = max(self.eps - self.eps_decay, self.eps_min)
