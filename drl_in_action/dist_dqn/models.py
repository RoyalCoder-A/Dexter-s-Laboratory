import torch


class DistDqnModel(torch.nn.Module):
    def __init__(
        self,
        num_state: int,
        num_hidden: int,
        num_actions: int,
        num_support: int,
        v_min: float,
        v_max: float,
    ) -> None:
        super().__init__()
        self.num_state = num_state
        self.num_actions = num_actions
        self.num_support = num_support
        self.v_min = v_min
        self.v_max = v_max
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_state, num_hidden),
            torch.nn.SELU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.SELU(),
            torch.nn.Linear(num_hidden, num_support * num_actions),
        )
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(
                    module.weight, mean=0, std=(1.0 / module.in_features) ** 0.5
                )
                torch.nn.init.zeros_(module.bias)
        support = torch.linspace(v_min, v_max, num_support)
        self.register_buffer("support", support)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        args:
            state: [BATCH, STATE_SIZE]
        returns:
            tensor: [BATCH, NUM_ACTIONS, NUM_SUPPORT]
        """
        return self.get_logits(state)

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """
        args:
            state: [BATCH, STATE_SIZE]
        returns:
            tensor: [BATCH, NUM_ACTIONS, NUM_SUPPORT]
        """
        x: torch.Tensor = self.layers(state)
        logits = x.reshape(-1, self.num_actions, self.num_support)
        return logits

    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        args:
            state: [BATCH, STATE_SIZE]
        returns:
            tensor: [BATCH, NUM_ACTIONS, NUM_SUPPORT]
        """
        return torch.nn.functional.softmax(self.get_logits(state), dim=-1)

    def get_q_values(
        self, state: torch.Tensor | None, probs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        args:
            state: [BATCH, STATE_SIZE]
            probs: [BATCH, NUM_ACTIONS, NUM_SUPPORT]
        returns:
            tensor: [BATCH, NUM_ACTIONS]
        """
        if state is not None:
            probs = self.get_probs(state)
        elif probs is None:
            raise ValueError("One of state or probs should be passed")
        values = torch.einsum("BAK,K->BA", probs, self.support)
        return values
