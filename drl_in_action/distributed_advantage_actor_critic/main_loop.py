import multiprocessing as mp

import gymnasium
import pandas as pd

from drl_in_action.distributed_advantage_actor_critic.agent import Agent
from drl_in_action.distributed_advantage_actor_critic.model import ActorCriticModel


class MainLoop:
    def __init__(
        self,
        num_worker: int,
        max_step_count: int,
        num_states: int,
        num_actions: int,
        num_hidden: int,
        env_id: str,
        discount: float,
        n: int,
        clc: float,
        device: str,
    ):
        self.num_workers = num_worker
        self.counter = mp.Value("i", 0)
        self.max_step_count = max_step_count
        self.discount = discount
        self.device = device
        self.env_id = env_id
        self.clc = clc
        self.n = n
        self.model = ActorCriticModel(num_states, num_actions, num_hidden).to(device)
        self.model.share_memory()
        self.rewards: list[tuple[int, float]] = []
        self.rewards_lock = mp.Lock()

    def run(self):
        processes: list[mp.Process] = []
        for i in range(self.num_workers):
            p = mp.Process(target=self.worker, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        print(self.counter.value, processes[1].exitcode)
        df = pd.DataFrame(self.rewards, columns=["step", "reward"])
        df.to_csv("./rewards.csv")

    def worker(self, i: int):
        env = gymnasium.make(self.env_id)
        agent = Agent(
            self.model,
            self.n,
            env.observation_space.shape[0],
            env.action_space.n,
            self.discount,
            self.clc,
            self.device,
        )
        while self.counter.value < self.max_step_count:
            state, _ = env.reset()
            done = False
            rewards = 0
            while not done:
                action = agent.get_action(state)
                state_, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.memory.remember(state, action, reward, state_, done)
                agent.step()
                rewards += reward
                state = state_
                self.counter.value += 1
            print(f"Worker: {i}, step: {self.counter.value}, rewards: {rewards}")
            with self.rewards_lock:
                self.rewards.append((self.counter.value, rewards))
