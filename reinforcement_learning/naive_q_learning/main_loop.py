from pathlib import Path
import gymnasium
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from reinforcement_learning.naive_q_learning.agent import Agent
from reinforcement_learning.naive_q_learning.network import QNetwork

if __name__ == "__main__":
    device = "cuda"
    env = gymnasium.make("CartPole-v1")
    states_dim = env.observation_space.shape
    assert states_dim
    n_actions = env.action_space.n
    q_network = QNetwork(states_dim, n_actions).to(device)
    agent = Agent(
        q_network=q_network,
        device=device,
        eps=1,
        eps_min=0.01,
        eps_decay=1e-5,
        n_actions=n_actions,
        states_dim=states_dim,
        discount=0.99,
        checkpoint_path=Path(__file__).parent
        / "checkpoints"
        / "cartpole"
        / "checkpoint.pth",
    )
    writer = SummaryWriter("runs/cartpole")
    best_reward = float("-inf")
    for i in tqdm(range(10_000)):
        state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update_q_value(state, action, float(reward), next_state, done)
            rewards.append(reward)
            state = next_state
        final_reward = sum(rewards)
        if final_reward > best_reward:
            print(f"New best reward: {final_reward}")
            best_reward = final_reward
            agent.save()
        writer.add_scalar("reward", final_reward, i)
        writer.add_scalar("epsilon", agent.eps, i)
