from attention_is_all_you_need.src import train, evaluate


if __name__ == "__main__":
    evaluate.evaluate("mps", train_name="fixed_causal_max")
