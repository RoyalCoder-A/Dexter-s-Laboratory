from reinforcement_learning.double_dqn.src import train


if __name__ == "__main__":
    train.train("PongNoFrameskip-v4", "checkpoints/", "runs/pong", device="cpu")
