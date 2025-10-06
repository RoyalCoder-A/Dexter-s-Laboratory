import gymnasium
import numpy as np
import pandas as pd

from drl_in_action.reinforce.agent import ReinforceAgent


if __name__ == "__main__":
    env = gymnasium.make("LunarLander-v3")
    agent = ReinforceAgent(
        env.observation_space.shape[0], env.action_space.n, 64, 0.99, "mps"
    )
    loss_history = []
    reward_history = []
    while True:
        rewards = 0
        losses = []
        state, _ = env.reset()
        loss = agent.reset()
        if loss is not None:
            losses.append(loss)
        done = False
        while not done:
            action = agent.get_action(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, state_, done)
            agent.step()
            rewards += reward
            state = state_
        loss_history.append(np.mean(losses))
        reward_history.append(rewards)
        print(f"Step: {agent.step_count}, rewards: {rewards}, loss: {loss_history[-1]}")
        if agent.step_count >= 1_000_000:
            break
    df = pd.DataFrame({"loss": loss_history, "rewards": reward_history})
    df.to_csv("./result.csv")
