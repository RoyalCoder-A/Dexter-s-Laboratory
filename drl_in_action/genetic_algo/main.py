import gymnasium
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from drl_in_action.genetic_algo.agent import evaluate_population, next_generation
from drl_in_action.genetic_algo.model import LayersType
from drl_in_action.genetic_algo.population import spawn_population


if __name__ == "__main__":
    num_generation = 25
    population_size = 500
    population_rate = 0.01
    tournament_fraction = 0.2
    env_id = "CartPole-v1"
    env = gymnasium.make(env_id)
    layers: LayersType = (
        (25, env.observation_space.shape[0]),
        (10, 25),
        (env.action_space.n, 10),
    )
    pop_fit: list[float] = []
    pop = spawn_population(population_size, layers)
    for i in tqdm(range(num_generation)):
        pop, avg_fit = evaluate_population(pop, layers, env_id)
        pop_fit.append(avg_fit)
        pop = next_generation(pop, population_rate, tournament_fraction)
    df = pd.Series(pop_fit).plot()
    plt.show()
