from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import tqdm


DATA_DIR_PATH = Path(__file__).parent.parent / "data"
MAX_LENGTH = 50
VOCAB_SIZE = 37_000


def get_tokenizer() -> Tokenizer:
    bpe_path = DATA_DIR_PATH / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(bpe_path))
    return tokenizer


def train_bpe_tokenizer() -> None:
    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    bpe_path = DATA_DIR_PATH / "bpe_tokenizer.json"
    bpe_dataset_path = DATA_DIR_PATH / "tmp.txt"
    if bpe_path.is_file():
        print("BPE tokenizer exists, skipping training")
        return
    dataset = load_dataset("wmt14", "de-en", split="train")
    if not bpe_dataset_path.is_file():
        with open(DATA_DIR_PATH / "tmp.txt", "a") as f:
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
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",  # Single sentence format
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",  # Paired sentence format
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )  # type: ignore
    tokenizer.save(str(bpe_path))
    bpe_dataset_path.unlink()
