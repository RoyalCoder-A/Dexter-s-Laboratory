import gymnasium
import numpy as np
import torch
from tqdm import tqdm
from drl_in_action.genetic_algo import model
from drl_in_action.genetic_algo.model import LayersType, unpack_params
from drl_in_action.genetic_algo.population import Population, mutate, recombine
import multiprocessing as mp


def next_generation(
    pop: list[Population], mut_rate: float, tournament_fraction: float
) -> list[Population]:
    new_pop: list[Population] = []
    while len(new_pop) < len(pop):
        tournament_size = int(tournament_fraction * len(pop))
        tournament_idx = np.random.randint(low=0, high=len(pop), size=tournament_size)
        tournament_population: list[Population] = np.array(pop)[tournament_idx].tolist()
        tournament_population.sort(key=lambda x: x.fit, reverse=True)
        parent_1, parent_2 = tournament_population[0], tournament_population[1]
        child_1, child_2 = recombine(parent_1, parent_2)
        child_1, child_2 = mutate(child_1, mut_rate), mutate(child_2, mut_rate)
        new_pop += [child_1, child_2]
    return new_pop


def test_model(x: Population, layers: LayersType, env_id: str) -> float:
    unpacked_params = unpack_params(x.params, layers)
    env = gymnasium.make(env_id)
    state, _ = env.reset()
    rewards = 0
    done = False
    while not done:
        action = model.model(
            torch.from_numpy(state).unsqueeze(0), unpacked_params
        ).squeeze()
        dist = torch.distributions.Categorical(probs=action)
        action = dist.sample().numpy()
        state_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards += reward
        state = state_
    return rewards


def evaluate_population(
    pop: list[Population], layers: LayersType, env_id: str
) -> tuple[list[Population], float, float]:

    with mp.Pool(mp.cpu_count()) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(
                    _test_model_wrapper,
                    [(x, layers, env_id) for x in pop],
                    chunksize=100,
                ),
                total=len(pop),
            )
        )
    scores = np.array([x[1] for x in result])
    return [x[0] for x in result], scores.mean(), scores.max()


def _test_model_wrapper(
    args: tuple[Population, LayersType, str],
) -> tuple[Population, float]:
    score = test_model(args[0], args[1], args[2])
    args[0].fit = score
    return args[0], score
