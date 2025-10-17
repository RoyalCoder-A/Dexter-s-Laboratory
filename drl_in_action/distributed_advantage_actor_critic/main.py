import gymnasium

from drl_in_action.distributed_advantage_actor_critic.main_loop import MainLoop


if __name__ == "__main__":
    env_id = "CartPole-v1"
    env = gymnasium.make(env_id)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    main_loop = MainLoop(
        10, 1000000, num_states, num_actions, 64, env_id, 0.99, 10, 0.1, "cpu"
    )
    main_loop.run()
