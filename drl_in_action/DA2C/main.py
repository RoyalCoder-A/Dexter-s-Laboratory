from multiprocessing.managers import ListProxy, ValueProxy
import gymnasium
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp


class ActorCriticModel(torch.nn.Module):
    def __init__(
        self, num_states: int, num_actions: int, num_hidden: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.features_layers = torch.nn.Sequential(
            torch.nn.Linear(num_states, num_hidden),
            torch.nn.ReLU(),
        )
        self.policy = torch.nn.Linear(num_hidden, num_actions)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features_layers(state)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


def run(
    env_id: str,
    num_hidden: int,
    max_steps: int,
    num_workers: int,
    discount: float,
    clc: float,
    n_step: int,
):
    env = gymnasium.make(env_id)
    model = ActorCriticModel(
        env.observation_space.shape[0], env.action_space.n, num_hidden
    )
    model.share_memory()
    with mp.Manager() as mgr:
        counter: ValueProxy[int] = mgr.Value("i", 0)
        rewards: "ListProxy[tuple[int, float]]" = mgr.list()
        processes: list[mp.Process] = []
        for i in range(num_workers):
            p = mp.Process(
                target=_worker,
                args=(
                    i,
                    model,
                    counter,
                    rewards,
                    env_id,
                    max_steps,
                    discount,
                    clc,
                    n_step,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for i in range(len(processes)):
            p = processes[i]
            p.terminate()
            print(f"Process {i} exit code {p.exitcode}")
        df = pd.DataFrame(
            data=[{"step": x[0], "reward": x[1]} for x in rewards],
        )
    df["reward"].plot()
    plt.show()


def _worker(
    worker_id: int,
    model: ActorCriticModel,
    counter: ValueProxy[int],
    rewards: "ListProxy[tuple[int, float]]",
    env_id: str,
    max_steps: int,
    discount: float,
    clc: float,
    n_stp: int,
):
    env = gymnasium.make(env_id)
    optim = torch.optim.Adam(model.parameters())
    while counter.value < max_steps:
        reward = _run_episode(optim, env, model, counter, n_stp, discount, clc)
        rewards.append((counter.value, reward))
        print(f"Worker {worker_id}, Step {counter.value} rewards {reward}")


def _run_episode(
    optim: torch.optim.Optimizer,
    env: gymnasium.Env,
    model: ActorCriticModel,
    counter: ValueProxy[int],
    n_stp: int,
    discount: float,
    clc: float,
) -> int:
    init_state: np.ndarray | None = None
    reward = 0
    while True:
        optim.zero_grad()
        init_state, values, log_probs, ep_rewards, terminals = _run_partial_episode(
            init_state, env, model, counter, n_stp
        )
        if any(terminals):
            first_terminal_idx = terminals.numpy().tolist().index(True) + 1
            reward += ep_rewards[0:first_terminal_idx].sum().numpy()
        else:
            reward += ep_rewards.sum().numpy()
        _update_params(optim, values, log_probs, ep_rewards, discount, clc, terminals)
        if any(terminals):
            break
    return reward


def _run_partial_episode(
    init_state: np.ndarray | None,
    env: gymnasium.Env,
    model: ActorCriticModel,
    counter: ValueProxy[int],
    n_stp: int,
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if init_state is not None:
        state = init_state
    else:
        state, _ = env.reset()
    done = False
    rewards = []
    values = []
    log_probs = []
    terminals = []
    j = 0
    while j <= n_stp:
        model.train()
        policy, value = model(
            torch.from_numpy(state).float().view(1, env.observation_space.shape[0])
        )
        value: torch.Tensor = value.squeeze()
        policy: torch.Tensor = policy.squeeze()
        values.append(value)
        dist = torch.distributions.Categorical(logits=policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        state_, reward, terminated, truncated, _ = env.step(
            action.detach().cpu().numpy()
        )
        rewards.append(reward)
        done = terminated or truncated
        terminals.append(done)
        state = state_
        counter.value += 1
        j += 1
        if done:
            state, _ = env.reset()
    return (
        state if not done else None,
        torch.stack(values),
        torch.stack(log_probs),
        torch.from_numpy(np.array(rewards)).float(),
        torch.from_numpy(np.array(terminals)).float(),
    )


def _update_params(
    optim: torch.optim.Optimizer,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    discount: float,
    clc: float,
    terminals: torch.Tensor,
):
    returns_rev = []
    values_rev, rewards_rev, terminals_rev = (
        values.flip(0),
        rewards.flip(0),
        terminals.flip(0),
    )
    ret_ = 0
    for i in range(len(values_rev)):
        ret_ = rewards_rev[i] + discount * ret_ * (1 - terminals_rev[i])
        returns_rev.append(ret_)
    returns = torch.stack(returns_rev).flip(0)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    actor_loss = -1 * log_probs * (returns - values.detach())
    critic_loss = torch.pow(returns - values, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    optim.step()


if __name__ == "__main__":
    run("LunarLander-v3", 64, 1000000, 10, 0.99, 0.1, 2048)
