import gymnasium
import numpy as np

if __name__ == "__main__":
    n_envs = 3
    env = gymnasium.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        vectorization_mode="async",
    )
    for _ in range(10):
        env.reset()
        done = [False] * n_envs  # (n_envs, )
        while not any(done):
            action = env.action_space.sample()  # (n_envs, )
            print(action)
            obs, reward, terminated, truncated, info = env.step(
                action
            )  # (n_envs, obs_dim)
            print(obs)
            done = np.array(terminated) | np.array(truncated)
            print(done)
