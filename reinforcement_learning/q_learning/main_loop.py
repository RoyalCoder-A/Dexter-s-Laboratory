from reinforcement_learning.q_learning.agent import Agent
import gymnasium
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

if __name__ == "__main__":
    env = gymnasium.make("FrozenLake-v1")
    agent = Agent(
        num_actions=int(env.action_space.n),
        eps=1.0,
        eps_min=0.01,
        eps_decay=1e-5,
        discount=0.99,
        learning_rate=0.1,
    )
    writer = SummaryWriter("runs/frozenlake")
    for i in tqdm(range(500_000)):
        state, _ = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update_q_value(state, action, float(reward), next_state, done)
            rewards.append(reward)
            state = next_state
        final_reward = sum(rewards)
        writer.add_scalar("reward", final_reward, i)
        writer.add_scalar("epsilon", agent.eps, i)
