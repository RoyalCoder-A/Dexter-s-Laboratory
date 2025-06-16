from reinforcement_learning.dueling_ddqn.src import train


if __name__ == "__main__":
    train.train("PongNoFrameskip-v4", "checkpoints/", "runs/pong", device="mps")
