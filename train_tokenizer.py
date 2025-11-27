#IMPORTS
import tokenizers
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split, Digits
from datasets import load_dataset

# SETUP DATASET
dataset = load_dataset("mydataset")

# TOKENIZER
trainer = BpeTrainer(special_tokens=["[UNK]", "[EOS]", "[BOS]", "[PAD]"],max_token_length=10)
pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True) ])
normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase()])
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer
tokenizer.trainer = trainer
tokenizer.train_from_iterator(dataset, length=len(dataset)).save("lamilaz_tokenizer.json")


