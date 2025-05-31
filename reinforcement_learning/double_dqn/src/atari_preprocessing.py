import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def create_env(env_id: str, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.ClipReward(env, min_reward=-1.0, max_reward=1.0)
    return env


if __name__ == "__main__":
    env = create_env("ALE/Boxing-v5")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
