import gymnasium
from matplotlib import pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from drl_in_action.genetic_algo import model
from drl_in_action.genetic_algo.agent import evaluate_population, next_generation
from drl_in_action.genetic_algo.model import LayersType, unpack_params
from drl_in_action.genetic_algo.population import Population, spawn_population


def start_testing(pop: Population, env: gymnasium.Env, layers: LayersType):
    unpacked = unpack_params(pop.params, layers)
    while True:
        signal = input("Enter x to exit...")
        if signal == "x":
            break
        done = False
        score = 0.0
        state, _ = env.reset()
        while not done:
            env.render()
            probs = model.model(
                torch.from_numpy(state).unsqueeze(dim=0), unpacked
            ).squeeze()
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().numpy()
            state_, rewards, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += float(rewards)
            state = state_
        print(f"Score: {score}")


if __name__ == "__main__":
    num_generation = 50
    population_size = 5000
    population_rate = 0.001
    tournament_fraction = 0.5
    env_id = "LunarLander-v3"
    l1_count = 60
    l2_count = 30
    env = gymnasium.make(env_id)
    layers: LayersType = (
        (l1_count, env.observation_space.shape[0]),
        (l2_count, l1_count),
        (env.action_space.n, l2_count),
    )
    pop_fit: list[float] = []
    pop = spawn_population(population_size, layers)
    pbar = tqdm(range(num_generation))
    for i in pbar:
        pop, avg_fit, max_fit = evaluate_population(pop, layers, env_id)
        pbar.set_description_str(f"Avg fit: {avg_fit}, Max fit: {max_fit}")
        pop_fit.append(avg_fit)
        pop = next_generation(pop, population_rate, tournament_fraction)
    df = pd.Series(pop_fit).plot()
    plt.show()
    pop, _, _ = evaluate_population(pop, layers, env_id)
    best_child: Population | None = None
    for p in pop:
        if best_child is None or best_child.fit < p.fit:
            best_child = p
    assert best_child
    start_testing(best_child, env, layers)
