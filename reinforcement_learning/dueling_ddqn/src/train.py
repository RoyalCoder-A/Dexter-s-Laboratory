from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from reinforcement_learning.dueling_ddqn.src.agent import Agent
from reinforcement_learning.dueling_ddqn.src.atari_preprocessing import (
    create_env,
)


def train(
    env_id: str,
    checkpoint_path: str,
    runs_path: str,
    episodes: int = 500,
    device: str = "cuda",
):
    env = create_env(env_id)
    states_dim = env.observation_space.shape
    assert states_dim
    n_actions = env.action_space.n
    agent = Agent(
        device=device,
        eps=1.0,
        eps_min=0.1,
        eps_decay=1e-5,
        n_actions=n_actions,
        states_dim=states_dim,
        discount=0.99,
        checkpoint_path=Path(checkpoint_path),
        replace=1000,
        memory_size=50_000,
        batch_size=32,
        learning_rate=0.0001,
    )
    writer = SummaryWriter(runs_path)
    best_reward = float("-inf")
    warmup = episodes // 5
    for i in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, float(reward), next_state, done)
            agent.learn()
            rewards.append(reward)
            state = next_state
        final_reward = sum(rewards)
        if final_reward > best_reward and i > warmup:
            print(f"New best reward: {final_reward}")
            best_reward = final_reward
            agent.save()
        print(f"Episode {i}, reward: {final_reward}, eps: {agent.eps}")
        writer.add_scalar("reward", final_reward, i)
        writer.add_scalar("epsilon", agent.eps, i)
