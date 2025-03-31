import torch


class PPOModel(torch.nn.Module):
    def __init__(self, obs_dim: int, n_action: int, hidden_layer_size: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_action = n_action
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_size, hidden_layer_size),
            torch.nn.ReLU(),
        )
        self.values_head = torch.nn.Linear(hidden_layer_size, 1)
        self.policy_head = torch.nn.Linear(hidden_layer_size, n_action)

    def forward(self, obs: torch.Tensor):
        hidden = self.linear_layers(obs)
        values = self.values_head(hidden)
        logits = self.policy_head(hidden)
        probs = torch.softmax(logits, dim=-1)
        return values, probs
