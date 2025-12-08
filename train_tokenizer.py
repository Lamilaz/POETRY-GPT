from tokenizers import Tokenizer,Regex, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split, Digits
from datasets import load_dataset
import re 
dataset = load_dataset("rojagtap/bookcorpus", split="train")

WhiteSpaceWithSplit = Split(
    pattern=Regex(r"\s+"), # Capture les espaces
    behavior="merged_with_next"    # Garde l'espace comme un token séparé
)

def get_training_corpus(batch_size=1000):
    batch = []
    column_name = "text" 
    for data in dataset:
        batch.append(data[column_name])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

trainer = BpeTrainer(
    special_tokens=["[UNK]", "[EOS]", "[BOS]", "[PAD]"],
    max_token_length=10,
    vocab_size=30000 # Pensez à définir une taille de vocabulaire cible
)

pre_tokenizer = Sequence([WhiteSpaceWithSplit, Digits(individual_digits=True)])
normalizer = normalizers.Sequence([
    normalizers.NFD(), 
    normalizers.StripAccents(), 
    normalizers.Lowercase()
])

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.save("/home/lamilaz/Documents/Code/POETRY-GPT/lamilaz_tokenizer.json")
print("Entraînement terminé et tokenizer sauvegardé.")
