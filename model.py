# -*- coding: utf-8 -*-
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
import wandb
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from bert_score import score as bert_score_func
from sentence_transformers import SentenceTransformer, util
import logging
import random

# Configuration silencieuse
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout": 0.1, # Réduit légèrement pour converger plus vite si dataset large
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "block_size": 256,
    "batch_size": 64, # Augmenté car Mixed Precision économise de la VRAM
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_iters": 1000,
    "max_iters": 50000, # Ajusté pour un entrainement plus standard
    "eval_every": 50,
    "eval_iters": 200,
    "save_every": 500,
    "checkpoint_name": "modelv2",
    "wandb_project": "nano-gpt",
    "dataset_name": "rojagtap/bookcorpus", # Dataset cible
    "use_compile": False # Activer torch.compile (PyTorch 2.0+)
}

device = config["device"]
print(f"Using device: {device}")

# --- TOKENIZER & METRICS ---
# Assurez-vous que tokenizer.json existe, sinon il faut le créer ou utiliser un pretrained
if not os.path.exists("tokenizer.json"):
    print("⚠️ Attention: 'tokenizer.json' introuvable. Assurez-vous d'avoir entraîné votre tokenizer.")
    # Fallback pour le test si pas de fichier (à retirer si vous avez votre fichier)
    # from tokenizers import Tokenizer
    # tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
else:
    tokenizer = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenizer.get_vocab_size()

print("Loading semantic model...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# --- UTILS ---
def encode(s):
    return tokenizer.encode(s).ids if hasattr(tokenizer, 'encode') else []

def decode(ids):
    return tokenizer.decode(ids) if hasattr(tokenizer, 'decode') else ""

def get_lr(it):
    if it < config["warmup_iters"]:
        return config["max_lr"] * (it + 1) / (config["warmup_iters"] + 1)
    if it > config["max_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (config["max_iters"] - config["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["max_lr"] - config["min_lr"])

def upload_to_drive(filename, drive_folder_id=None):
    if not os.path.exists('token.json'): return
    try:
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.file'])
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': os.path.basename(filename)}
        if drive_folder_id: file_metadata['parents'] = [drive_folder_id]
        media = MediaFileUpload(filename, resumable=True)
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded: {filename}")
    except Exception as e:
        print(f"Drive Upload Error: {e}")

# --- DATASET OPTIMISÉ (NON-STREAMING) ---
class PretokenizedDataset(Dataset):
    """
    Dataset qui charge tout en mémoire (via HuggingFace Arrow) 
    et découpe des blocs aléatoires.
    """
    def __init__(self, hf_dataset, block_size):
        self.dataset = hf_dataset
        self.block_size = block_size
        # On calcule la taille totale virtuelle en tokens (approximatif pour l'indexage)
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # On récupère une ligne pré-tokenisée
        # Note: Pour un vrai entrainement LLM, on concatène souvent tout,
        # mais ici on garde la structure par "document" pour simplifier
        ids = self.dataset[idx]['ids']
        
        # Si le document est trop court, on pad ou on prend le suivant (simplifié ici par padding)
        if len(ids) <= self.block_size + 1:
            # Padding avec 0
            pad_len = (self.block_size + 1) - len(ids)
            ids = ids + [0] * pad_len
        
        # Choix aléatoire d'une fenêtre dans le document si long
        # ou prise du début si juste assez long
        max_start = len(ids) - (self.block_size + 1)
        if max_start > 0:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = 0

        chunk = ids[start_idx : start_idx + self.block_size + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def prepare_data():
    """Télécharge, tokenise et split le dataset."""
    print("⬇️ Téléchargement du dataset (cela peut prendre du temps la première fois)...")
    # Chargement standard (non-streaming) -> Stocké sur disque par HuggingFace
    dataset = load_dataset(config["dataset_name"], split="train") 
    
    # Pour le dev rapide, on peut décommenter la ligne suivante pour prendre juste 10%
    # dataset = dataset.select(range(100000)) 

    print("✂️ Tokenization en cours (multiprocess)...")
    def process_func(examples):
        return {"ids": [encode(text) for text in examples["text"]]}

    # .map va cacher le résultat sur disque. Le redémarrage sera instantané.
    tokenized_ds = dataset.map(
        process_func, 
        batched=True, 
        remove_columns=["text"], 
        num_proc=os.cpu_count() # Utilise tous les cœurs CPU
    )
    
    # Filtre les lignes vides
    tokenized_ds = tokenized_ds.filter(lambda x: len(x['ids']) > 0)

    # Split Train/Val (90/10)
    split_ds = tokenized_ds.train_test_split(test_size=0.10, seed=42)
    
    return split_ds['train'], split_ds['test']


# --- MODEL ARCHITECTURE ---
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config["dropout"])
        )

    def forward(self, x):
        return self.net(x)

class MainLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.multihead = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=config["dropout"])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)

    def compute_attn_mask(self, x):
        L = x.shape[1]
        # create_mask est plus efficace et ne recrée pas le tenseur s'il est caché
        return torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

    def forward(self, x):
        # Attention: PyTorch 2.x optimise automatiquement MultiheadAttention si possible
        qkv = self.norm1(x)
        attn_out, _ = self.multihead(qkv, qkv, qkv, attn_mask=self.compute_attn_mask(x), need_weights=False)
        x = x + attn_out
        x = x + self.feed_forward(self.norm2(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config["d_model"])
        self.position_embedding = nn.Embedding(config["block_size"], config["d_model"])
        self.layers = nn.Sequential(*[MainLayer(config["d_model"], config["n_heads"]) for _ in range(config["n_layers"])])
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.head = nn.Linear(config["d_model"], vocab_size)

        # Init weights (Best practice pour GPT)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        # Clamp pour éviter crash si idx > block_size
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.ln_f(x)
        return self.head(x)

def model_loss(model, x, targets):
    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=0)
    return loss

@torch.no_grad()
def generate(model, x, max_new_tokens):
    for _ in range(max_new_tokens):
        x_crop = x[:, -config["block_size"]:]
        logits = model(x_crop)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=1)
        x_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, x_next], dim=1)
    return x

@torch.no_grad()
def evaluate_loss(model, dataloader, max_batches):
    model.eval()
    losses = []
    # Utilisation d'itérateur pour ne pas reset le dataloader entier
    iter_dl = iter(dataloader)
    for _ in range(max_batches):
        try:
            x, y = next(iter_dl)
            x, y = x.to(device), y.to(device)
            # Pas besoin de AMP pour eval loss généralement, mais plus sûr
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = model_loss(model, x, y)
            losses.append(loss.item())
        except StopIteration:
            break
    model.train()
    return np.mean(losses) if losses else 0.0

@torch.no_grad()
def evaluate_semantics(model, dataset, num_samples=4, max_new=50):
    model.eval()
    refs, cands = [], []
    
    indices = torch.randint(0, len(dataset), (num_samples * 2,)).tolist()
    found = 0
    
    for idx in indices:
        if found >= num_samples: break
        
        x, _ = dataset[idx]
        
        # Filtre padding
        non_pad = (x != 0).nonzero(as_tuple=True)[0]
        if len(non_pad) < 10: continue
            
        real_len = non_pad[-1].item()
        mid = real_len // 2
        
        ctx = x[:mid].unsqueeze(0).to(device)
        end_target = min(mid + max_new, real_len + 1)
        target_ids = x[mid : end_target].tolist()
        ref_text = decode(target_ids).strip()
        
        if not ref_text: continue
            
        gen = generate(model, ctx, max_new)[0, mid:].tolist()
        cands.append(decode(gen))
        refs.append(ref_text)
        found += 1
    
    if not cands: return 0.0, 0.0, [], []
        
    emb1 = semantic_model.encode(cands, convert_to_tensor=True)
    emb2 = semantic_model.encode(refs, convert_to_tensor=True)
    cosine = torch.diag(util.cos_sim(emb1, emb2)).mean().item()
    
    try:
        _, _, F1 = bert_score_func(cands, refs, lang="en", verbose=False, device=device)
        bert_val = F1.mean().item()
    except Exception as e:
        # print(f"Erreur BERT Score: {e}") 
        bert_val = 0.0
        
    model.train()
    return bert_val, cosine, cands, refs

# --- MAIN ---
if __name__ == "__main__":
    
    # 1. Init WandB
    wandb.init(project=config["wandb_project"], resume="allow", config=config, id="7gfg4rby" )
    
    # 2. Préparation des données (One-time cost)
    train_data_hf, val_data_hf = prepare_data()
    
    print(f"📚 Train size: {len(train_data_hf)} docs | Val size: {len(val_data_hf)} docs")
    
    train_ds = PretokenizedDataset(train_data_hf, config["block_size"])
    val_ds = PretokenizedDataset(val_data_hf, config["block_size"])
    
    # Num_workers > 0 accélère le chargement, pin_memory=True pour GPU
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # 3. Modèle & Optimisation
    print("🧠 Création du modèle...")
    
    model = TransformerDecoder().to(device)
    model = torch.load("modelv2_1500.pt", weights_only=True)
    # Compile le modèle (PyTorch 2.0 optimization)
    if config["use_compile"] and hasattr(torch, "compile"):
        print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["max_lr"], weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") # Pour Mixed Precision

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 4. Boucle d'entrainement
    step = 1500
    print("🚀 Début de l'entraînement optimisé !")
    
    # On itère indéfiniment sur le loader (cycle)
    train_iter = iter(train_loader)
    
    t0 = time.time()
    
    while step < config["max_iters"]:
        step += 1
        
        # Récupération batch (avec reset automatique si fin d'epoch)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # --- TRAINING STEP (Mixed Precision) ---
        lr = get_lr(step)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        optimizer.zero_grad(set_to_none=True) # Plus efficace que zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss = model_loss(model, x, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # Nécessaire pour le clip_grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- LOGGING ---
        if step % config["eval_every"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            
            # Évaluation
            v_loss = evaluate_loss(model, val_loader, config["eval_iters"])
            t_loss = loss.item()
            
            # Semantic eval (un peu plus lourd, on peut le faire moins souvent)
            bert, cos, cands, refs = evaluate_semantics(model, val_ds, num_samples=2)
            
            # Génération sample
            prompts = torch.tensor([encode(p) for p in ["Hello", "Love", "Help"]]).to(device)
            outputs = generate(model, prompts, 100)
            outputs = [[decode(output.tolist())] for output in outputs]
            print(f"step {step}: train loss {t_loss:.4f}, val loss {v_loss:.4f}")
            
            wandb.log({
                "step": step, "lr": lr,
                "train_loss": t_loss, "val_loss": v_loss,
                "bert_score": bert, "cosine_sim": cos,
                "samples": wandb.Table(columns=["samples"], data=outputs)
            })

        # --- SAVE ---
        if step % config["save_every"] == 0:
            print(f"💾 Sauvegarde step {step}...")
            fname = f"{config['checkpoint_name']}_{step}.pt"
            # On sauvegarde le state_dict original (pas celui du compiled model)
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(raw_model.state_dict(), fname)
            upload_to_drive(fname)

    wandb.finish()
    print("✅ Entraînement terminé.")
