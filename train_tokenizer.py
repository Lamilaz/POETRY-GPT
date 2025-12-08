from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel,Digits, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_dataset

dataset = load_dataset("rojagtap/bookcorpus", split="train")

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
    vocab_size=30000
)

# ByteLevel remplace WhiteSpaceWithSplit + Digits
# Il encode tout en bytes (espaces = Ġ, chiffres gérés automatiquement)
pre_tokenizer = Sequence([
    Digits(individual_digits=True),  # Split les chiffres d'abord
    ByteLevel(add_prefix_space=False)
])

normalizer = normalizers.Sequence([
    normalizers.NFD(),  # NFD avant StripAccents pour éviter les [UNK]
    normalizers.StripAccents(), 
    normalizers.Lowercase()
])

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Decoder ByteLevel pour reconvertir les bytes en texte
tokenizer.decoder = ByteLevelDecoder()

tokenizer.save("/home/LLM/jouanikomachon/lamilaz_tokenizer.json")
print("Entraînement terminé et tokenizer sauvegardé.")