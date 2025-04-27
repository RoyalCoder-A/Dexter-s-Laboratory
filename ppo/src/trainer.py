from torch.utils.tensorboard.writer import SummaryWriter
from gymnasium.vector import VectorEnv
import numpy as np
from collections import deque

from ppo.src.agent import Agent


class Trainer:
    def __init__(
        self,
        env: VectorEnv,
        agent: Agent,
        max_steps: int,
        reward_threshold: float,
        summary_writer: SummaryWriter,
        eval_window: int,
    ):
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        self.summary_writer = summary_writer
        self.eval_window = eval_window

    def train(self):
        current_total_steps = 0
        obs, _ = self.env.reset()
        episode_returns = deque(maxlen=self.eval_window)
        best_mean_episodic_return = -np.inf
        while current_total_steps < self.max_steps:
            actions, log_probs, values = self.agent.act(obs)
            obs_, reward, terminated, truncated, infos = self.env.step(actions)
            dones = terminated | truncated
            self.agent.step(obs, actions, reward, obs_, dones, log_probs, values)
            obs = obs_
            current_total_steps += self.env.num_envs
            if "_final_observation" in infos:
                for i, done in enumerate(dones):
                    if done:  # Check if the specific environment was done
                        # Extract episodic return from info
                        # Use .get() for safety if "episode" key might be missing
                        ep_info = infos["final_info"][i].get("episode")
                        if ep_info is not None:
                            ep_return = ep_info["r"]
                            episode_returns.append(ep_return)
                            # Log individual episode return
                            self.summary_writer.add_scalar(
                                "rollout/episodic_return",
                                ep_return,
                                current_total_steps,
                            )
                            # Optional: print(f"Step: {current_total_steps}, Env {i}, Episode Return: {ep_return}")

            # --- Logging and Evaluation (based on episodic returns) ---
            # Log metrics periodically or when the deque is full
            if len(episode_returns) == self.eval_window:  # Check if deque is full
                mean_episodic_return = np.mean(episode_returns)

                # Log mean episodic return
                self.summary_writer.add_scalar(
                    "rollout/mean_episodic_return",
                    mean_episodic_return,
                    current_total_steps,
                )

                # Save best model based on mean episodic return
                if mean_episodic_return > best_mean_episodic_return:
                    best_mean_episodic_return = mean_episodic_return
                    self.agent.save()
                    print(
                        f"\n*** New best mean return: {best_mean_episodic_return:.2f} at step {current_total_steps} ***"
                    )
                    self.summary_writer.add_scalar(
                        "rollout/best_mean_episodic_return",
                        best_mean_episodic_return,
                        current_total_steps,
                    )

                # Print progress
                print(
                    f"\rSteps: {current_total_steps}/{self.max_steps}, "
                    f"Mean Return ({self.eval_window} eps): {mean_episodic_return:.2f}, "
                    f"Best Mean Return: {best_mean_episodic_return:.2f}",
                    end="",
                )

                # Check for solving condition
                if mean_episodic_return >= self.reward_threshold:
                    print(f"\nEnvironment solved in {current_total_steps} steps!")
                    break

        print("\nTraining finished.")
        self.env.close()
        self.summary_writer.close()

        # ... (cleanup) ...
