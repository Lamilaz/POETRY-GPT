# 1. CORRECTION DES IMPORTS
from tokenizers import Tokenizer, normalizers, Regex  # <-- Regex est ici
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# On enlève 'Pattern' de cette ligne car il n'existe pas
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Replace 
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
    vocab_size=50000
)

pre_tokenizer = ByteLevel(add_prefix_space=False)

normalizer = normalizers.Sequence([
    NFD(), 
    StripAccents(), 
    Lowercase(),
    
    # 1. Supprimer les chiffres
    Replace(Regex(r"\d"), ""), 
    
    # 2. Supprimer TOUS les caractères spéciaux gênants
    # J'ai regroupé votre demande + d'autres symboles mathématiques/techniques
    # Explication du regex r"[%...]" :
    # \ est échappé en \\
    # ^ est mis à la fin pour ne pas être confondu avec la négation
    # * + = pour nettoyer les en-têtes de chapitres
    Replace(Regex(r"[%$£€@#~{}\[\]|<>^&*+=_`\\]()§;"), ""),

    # 3. Supprimer les guillemets bizarres et tirets de dialogue bizarres
    Replace(Regex(r"``"), ""),
    Replace(Regex(r"''"), ""),
    Replace(Regex(r"--"), ""), # Les doubles tirets sont fréquents
    
    # 4. Nettoyage des espaces (CRUCIAL pour votre syntaxe)
    Replace(Regex(r"\s+"), " "), # Transforme "  " en " "
    Replace(Regex(r"\s+\."), "."), # Colle le point au mot précédent
    Replace(Regex(r"\s+,"), ","),  # Colle la virgule au mot précédent
    Replace(Regex(r"\s+\?"), "?"), # Colle le ? au mot précédent
    Replace(Regex(r"\s+!"), "!"),  # Colle le ! au mot précédent
])

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.decoder = ByteLevelDecoder()

tokenizer.save("/home/lamilaz/lamilaz_tokenizer_latest.json")
print("Entraînement terminé et tokenizer sauvegardé.")