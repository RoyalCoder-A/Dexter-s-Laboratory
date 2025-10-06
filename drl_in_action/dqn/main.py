import gymnasium
import numpy as np
import pandas as pd

from drl_in_action.dqn.agent import Agent


if __name__ == "__main__":
    env = gymnasium.make("LunarLander-v3")
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.n,
        64,
        500_000,  # Smaller memory buffer
        32,  # Smaller batch size
        1000,  # Keep target update frequency
        1.0,  # Start with full exploration
        1e-5,  # Faster epsilon decay
        0.01,  # Keep epsilon min
        0.98,  # Higher discount for CartPole
        "mps",
    )
    loss_history = []
    reward_history = []
    while True:
        rewards = 0
        losses = []
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, state_, done)
            loss = agent.step()
            if loss is not None:
                losses.append(loss)
            rewards += reward
            state = state_
        loss_history.append(np.mean(losses))
        reward_history.append(rewards)
        print(
            f"Step: {agent.step_count}, eps: {agent.eps}, rewards: {rewards}, loss: {loss_history[-1]}"
        )
        if agent.step_count >= 1_000_000:
            break
    df = pd.DataFrame({"loss": loss_history, "rewards": reward_history})
    df.to_csv("./result.csv")
