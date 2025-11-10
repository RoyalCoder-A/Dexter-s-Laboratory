import gymnasium
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from drl_in_action.dist_dqn.agent import Agent


def run() -> None:
    epochs = 1300
    env = gymnasium.make("CartPole-v1")
    assert env.observation_space.shape
    action_space = int(getattr(env.action_space, "n"))
    agent = Agent(
        env.observation_space.shape[0],
        action_space,
        64,
        11,
        (0, 1),
        1.0,
        1e-5,
        0.01,
        0.9,
        200,
        5,
        75,
        "mps",
    )
    scores = []
    pbar = tqdm(range(epochs))
    for i in pbar:
        state, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.get_actions(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, float(reward), state_, done)
            agent.step()
            score += float(reward)
            state = state_
        pbar.set_description_str(f"Score: {score}")
        if i % 100 == 0:
            print(f"Epoch {i}, Eps {agent.eps} Score {score}")
        scores.append(score)
    df = pd.DataFrame([{"score": x} for x in scores])
    df["score"].plot()
    plt.show()


if __name__ == "__main__":
    run()
