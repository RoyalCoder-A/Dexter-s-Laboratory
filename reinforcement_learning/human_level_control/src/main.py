from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from reinforcement_learning.human_level_control.src.agent import Agent
from reinforcement_learning.human_level_control.src.atari_preprocessing import (
    create_env,
)

if __name__ == "__main__":
    device = "mps"
    env = create_env("PongNoFrameskip-v4")
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
        checkpoint_path=Path("checkpoints/"),
        replace=1000,
        memory_size=50_000,
        batch_size=32,
        learning_rate=0.0001
    )
    writer = SummaryWriter("runs/pong")
    best_reward = float("-inf")
    warmup = 10
    for i in tqdm(range(500)):
        state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            rewards.append(reward)
            state = next_state
        final_reward = sum(rewards)
        if final_reward > best_reward and i > warmup:
            print(f"New best reward: {final_reward}")
            best_reward = final_reward
            agent.save()
        print(f"Episode {i}, final reward: {final_reward}, epsilon: {agent.eps}")
        writer.add_scalar("reward", final_reward, i)
        writer.add_scalar("epsilon", agent.eps, i)
