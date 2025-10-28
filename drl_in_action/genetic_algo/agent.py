import gymnasium
import torch
from drl_in_action.genetic_algo import model
from drl_in_action.genetic_algo.model import LayersType, unpack_params
from drl_in_action.genetic_algo.population import Population


def test_model(x: Population, layers: LayersType, env_id: str) -> float:
    unpacked_params = unpack_params(x.params, layers)
    env = gymnasium.make(env_id)
    state, _ = env.reset()
    rewards = 0
    done = False
    while not done:
        action = model.model(state, unpacked_params)
        dist = torch.distributions.Categorical(probs=action)
        action = dist.sample().numpy()
        state_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards += reward
        state = state_
    return rewards


def evaluate_population(
    pop: list[Population], layers: LayersType, env_id: str
) -> tuple[list[Population], float]:
    total_fit = 0
    lp = len(pop)
    for agent in pop:
        score = test_model(agent, layers, env_id)
        agent.fit = score
        total_fit += score
    return pop, total_fit / lp
