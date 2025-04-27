from pathlib import Path
import numpy as np
import torch

from ppo.src.model_implementation import PPOModel


class Agent:
    def __init__(
        self,
        horizon: int,
        state_dim: tuple[int, ...],
        n_envs: int,
        mini_batch_size: int,
        brain: PPOModel,
        n_epochs: int,
        discount: float,
        gae_lambda: float,
        clip_range: float,
        entropy_coef: float,
        value_loss_coef: float,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        init_lr: float,
        data_path: Path,
        device: str,
    ):
        self.sampled_states = np.zeros((horizon, n_envs, *state_dim), dtype=np.float32)
        self.sampled_actions = np.zeros((horizon, n_envs), dtype=np.int64)
        self.sampled_rewards = np.zeros((horizon, n_envs), dtype=np.float32)
        self.sampled_dones = np.zeros((horizon, n_envs), dtype=np.bool_)
        self.sampled_next_states = np.zeros(
            (horizon, n_envs, *state_dim), dtype=np.float32
        )
        self.log_probs = np.zeros((horizon, n_envs), dtype=np.float32)
        self.sampled_values = np.zeros((horizon, n_envs), dtype=np.float32)
        self.horizon = horizon
        self.state_dim = state_dim
        self.n_envs = n_envs
        self.current_step = 0
        self.mini_batch_size = mini_batch_size
        self.brain = brain
        self.n_epochs = n_epochs
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.init_lr = init_lr
        self.device = device
        self.data_path = data_path
        self.learning_step = 0

    def act(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tensors = torch.from_numpy(state).float().to(self.device)
        self.brain.eval()
        with torch.inference_mode():
            values, probs = self.brain(state_tensors)
        probs = probs.detach().cpu()
        values = values.detach().cpu()
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        actions = actions.numpy()
        log_probs = log_probs.numpy()
        return actions, log_probs, values.numpy()

    def save(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.brain.state_dict(),
            self.data_path / f"ppo_model_{self.learning_step}.pt",
        )

    def load(self):
        checkpoint_path = self.data_path / f"ppo_model_{self.learning_step}.pt"
        if checkpoint_path.exists():
            self.brain.load_state_dict(torch.load(checkpoint_path))

    def step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
    ):
        self._remember(states, actions, rewards, next_states, dones, log_probs, values)
        if self.current_step % self.horizon != 0:
            return
        self._train(next_states, dones)
        self._reset()

    def _train(self, final_next_state: np.ndarray, final_done: np.ndarray):
        dl = self._create_dataloader(final_next_state, final_done)
        for _ in range(self.n_epochs):
            for batch in dl:
                self._train_step(batch)
        self._update_lr()

    def _update_lr(self) -> float:
        alpha = 1 - (self.learning_step / self.max_steps)
        lr = self.init_lr * alpha
        for param in self.optimizer.param_groups:
            param["lr"] = lr
        self.learning_step += 1
        return lr

    def _train_step(self, batch: tuple[torch.Tensor, ...]):
        (
            sampled_states,
            sampled_actions,
            sampled_log_probs,
            sampled_advantages,
            sampled_returns,
        ) = batch
        sampled_states = sampled_states.to(self.device)
        sampled_actions = sampled_actions.to(self.device)
        sampled_log_probs = sampled_log_probs.to(self.device)
        sampled_advantages = sampled_advantages.to(self.device)
        sampled_returns = sampled_returns.to(self.device)
        self.brain.train()
        values, probs = self.brain(sampled_states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(sampled_actions)
        policy_ratio = torch.exp(log_probs - sampled_log_probs)
        policy_loss = -torch.min(
            policy_ratio * sampled_advantages,
            policy_ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
            * sampled_advantages,
        ).mean()
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        value_loss = self.value_loss_coef * ((values - sampled_returns).pow(2).mean())
        loss = policy_loss + value_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
        self.optimizer.step()

    def _create_dataloader(self, final_next_state: np.ndarray, final_done: np.ndarray):
        # Calculate value for the state AFTER the last state in the buffer
        with torch.inference_mode():
            final_next_value_t, _ = self.brain(
                torch.from_numpy(final_next_state).float().to(self.device)
            )
        final_next_value = (
            final_next_value_t.view((self.n_envs,)).detach().cpu().numpy()
        )
        # Calculate advantages and returns using the bootstrapped value
        advantages, returns = self._calculate_advantage(
            self.sampled_rewards,
            self.sampled_values,
            self.sampled_dones,  # Pass all dones
            final_next_value,  # Pass value of s_T
            final_done,  # Pass done signal for s_T
        )
        # --- Rest of the flattening and dataloader creation remains the same ---
        flattened_advantages = (
            torch.from_numpy(advantages).float().view(self.horizon * self.n_envs)
        )
        # ... (flatten returns, states, actions, log_probs) ...
        flattened_returns = (
            torch.from_numpy(returns).float().view(self.horizon * self.n_envs)
        )
        flattened_states = (
            torch.from_numpy(self.sampled_states)
            .float()
            .view(self.horizon * self.n_envs, *self.state_dim)
        )
        flattened_actions = (
            torch.from_numpy(self.sampled_actions)
            .long()
            .view(self.horizon * self.n_envs)
        )
        flattened_log_probs = (
            torch.from_numpy(self.log_probs).float().view(self.horizon * self.n_envs)
        )
        ds = PPODataset(
            flattened_states,
            flattened_actions,
            flattened_log_probs,
            flattened_returns,  # Use calculated returns
            flattened_advantages,
        )
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.mini_batch_size,
            shuffle=True,
            num_workers=0,  # Consider increasing if I/O bound
            pin_memory=True,  # Good if using GPU
        )
        return dataloader

    def _calculate_advantage(
        self,
        rewards: np.ndarray,  # Shape (horizon, n_envs)
        values: np.ndarray,  # Shape (horizon, n_envs)
        dones: np.ndarray,  # Shape (horizon, n_envs) - Dones FOR s_0 to s_{T-1}
        final_next_value: np.ndarray,  # Shape (n_envs,) - V(s_T)
        final_done: np.ndarray,  # Shape (n_envs,) - Done signal for transition s_{T-1} -> s_T
    ):
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        for t in reversed(range(self.horizon)):
            # Check if this is the last step in the buffer (t = horizon - 1)
            if t == self.horizon - 1:
                next_non_terminal = 1.0 - final_done
                next_value = final_next_value
            else:
                # For other steps, use done and value from the *next* step in the buffer
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = (
                rewards[t] + self.discount * next_value * next_non_terminal - values[t]
            )
            advantages[t] = (
                delta
                + self.discount * self.gae_lambda * next_non_terminal * last_advantage
            )
            last_advantage = advantages[t]

        returns = advantages + values  # Calculate returns using original values
        # Normalize advantages across all samples in the batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        values: np.ndarray,
    ):
        idx = self.current_step
        self.sampled_states[idx] = state
        self.sampled_actions[idx] = action
        self.sampled_rewards[idx] = reward
        self.sampled_next_states[idx] = next_state
        self.sampled_dones[idx] = done
        self.log_probs[idx] = log_prob
        self.sampled_values[idx] = values
        self.current_step += 1

    def _reset(self):
        self.sampled_states.fill(0)
        self.sampled_actions.fill(0)
        self.sampled_rewards.fill(0)
        self.sampled_next_states.fill(0)
        self.sampled_dones.fill(0)
        self.log_probs.fill(0)
        self.sampled_values.fill(0)
        self.current_step = 0


class PPODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.returns = returns
        self.advantages = advantages

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.returns[idx],
            self.advantages[idx],
        )
