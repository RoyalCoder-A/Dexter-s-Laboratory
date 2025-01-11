from pathlib import Path

import torchtext


def get_dataset():
    data_path = Path(__file__).parent / "data"
    if not data_path.exists():
        torchtext.datasets.WMT14.download(root=str(data_path))
    dataset = torchtext.datasets.WMT14(
        path=str(data_path),
        exts=("en", "de"),
        fields=[
            ("src", torchtext.data.Field(lower=True)),
            ("trg", torchtext.data.Field(lower=True)),
        ],
    )
    return dataset
