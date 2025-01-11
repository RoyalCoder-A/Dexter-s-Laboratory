import torchtext
from torch.utils.data import Dataset


def get_dataset() -> Dataset:
    dataset = torchtext.datasets.WMT14()
    return dataset
