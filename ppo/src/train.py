from pathlib import Path
import gymnasium
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from ppo.src.agent import Agent
from ppo.src.model_implementation import PPOModel
from ppo.src.trainer import Trainer


def train(
    n_envs: int,
    horizon: int,
    minibatch_size: int,
    n_epochs: int,
    discount: float,
    gae_lambda: float,
    clip_range: float,
    entropy_coef: float,
    value_coef: float,
    max_steps: int,
    init_lr: float,
    data_path: Path,
    device: str,
    reward_threshold: float,
    log_dir: Path,
):
    env = gymnasium.make_vec(
        "CartPole-v1",
        num_envs=n_envs,
        vectorization_mode="async",
    )
    obs_shape = env.observation_space.shape[1:]
    n_actions = env.action_space.nvec[0]
    model = PPOModel(obs_shape, n_actions).to(device)
    if device == "gpu":
        model.compile()
    optimizer = torch.optim.Adam(model.parameters())
    summary_writer = SummaryWriter(str(log_dir))
    agent = Agent(
        horizon,
        obs_shape,
        n_envs,
        minibatch_size,
        model,
        n_epochs,
        discount,
        gae_lambda,
        clip_range,
        entropy_coef,
        value_coef,
        optimizer,
        max_steps,
        init_lr,
        data_path,
        device,
    )
    trainer = Trainer(env, agent, max_steps, reward_threshold, summary_writer, 100)
    trainer.train()
