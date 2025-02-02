from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import tqdm


DATA_DIR_PATH = Path(__file__).parent.parent / "data"
MAX_LENGTH = 50
VOCAB_SIZE = 37_000


def get_tokenizer(data_path: Path = DATA_DIR_PATH) -> Tokenizer:
    bpe_path = data_path / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(bpe_path))
    return tokenizer


def train_bpe_tokenizer(data_path: Path = DATA_DIR_PATH) -> None:
    data_path.mkdir(parents=True, exist_ok=True)
    bpe_path = data_path / "bpe_tokenizer.json"
    bpe_dataset_path = data_path / "tmp.txt"
    if bpe_path.is_file():
        print("BPE tokenizer exists, skipping training")
        return
    dataset = load_dataset("wmt14", "de-en", split="train")
    if not bpe_dataset_path.is_file():
        with open(data_path / "tmp.txt", "a") as f:
            for data in tqdm.tqdm(dataset.iter(100), total=dataset.num_rows / 100):  # type: ignore
                for translation in data["translation"]:  # type: ignore
                    f.write(translation["en"] + "\n")
                    f.write(translation["de"] + "\n")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    # Train the tokenizer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,  # type: ignore
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],  # type: ignore
    )

    files = [str(bpe_dataset_path)]
    tokenizer.train(files, trainer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=MAX_LENGTH
    )
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.save(str(bpe_path))
    bpe_dataset_path.unlink()
