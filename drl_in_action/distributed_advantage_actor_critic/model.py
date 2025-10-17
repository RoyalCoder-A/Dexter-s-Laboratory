import torch


class ActorCriticModel(torch.nn.Module):
    def __init__(
        self, num_states: int, num_actions: int, num_hidden: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, num_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.LeakyReLU(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_actions),
            torch.nn.LogSoftmax(),
        )
        self.value_head = torch.nn.Linear(num_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.layers(x)
        log_prob = self.policy_head(features)
        state_value = self.value_head(features)
        return log_prob, state_value
