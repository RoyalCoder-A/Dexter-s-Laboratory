from pathlib import Path
from ppo.src import train


if __name__ == "__main__":
    train.train(
        n_envs=8,
        horizon=128,
        minibatch_size=32 * 8,
        discount=0.99,
        gae_lambda=0.95,
        n_epochs=3,
        clip_range=0.1,
        value_coef=1,
        entropy_coef=0.01,
        max_steps=5_000_000,
        init_lr=2.5e-4,
        data_path=Path("data/ppo"),  # Path to save the data
        device="cuda",  # "cuda" or "cpu"
        reward_threshold=200,  # Reward threshold for the environment
        log_dir=Path("logs"),  # Path to save the logs
        env_id="LunarLander-v3",  # Environment ID
    )
