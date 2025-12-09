# -*- coding: utf-8 -*-
import math
import os
import random
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

# Configuration silencieuse
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- CONFIGURATION ---
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout": 0.2,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 6,
    "block_size": 256,
    "batch_size": 32,
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_iters": 300,
    "max_iters": 100000,
    "eval_every": 200,
    "eval_iters": 50,
    "save_every": 500,
    "checkpoint_name": "lamilaz_new.pt",
    "wandb_id": "uo2jduuz",
    "wandb_project": "nano-gpt"
}

device = config["device"]

# --- TOKENIZER & METRICS ---
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

class StreamingTextDataset(Dataset):
    def __init__(self, stream_dataset, block_size, max_samples=100000):
        self.block_size = block_size
        self.max_samples = max_samples
        # On transforme le stream en itérateur persistant pour reprendre où on s'est arrêté
        self.iterator = iter(stream_dataset) 
        self.samples = []
        
        # Premier chargement automatique
        self.renew_data()

    def renew_data(self):
        """Vide la mémoire et charge le paquet suivant de données"""
        print(f"🔄 Chargement de {self.max_samples} nouvelles données...")
        self.samples = [] # On vide la RAM
        count = 0
        
        try:
            while count < self.max_samples:
                # On récupère le prochain texte du stream (internet/cache)
                item = next(self.iterator)
                text = item["text"]
                
                if not text or len(text.strip()) == 0: 
                    continue
                
                tokens = encode(text)
                # On découpe le texte en morceaux de la taille du contexte
                num_chunks = max(1, len(tokens) - self.block_size)
                
                for start_idx in range(num_chunks):
                    if count >= self.max_samples: break
                    self.samples.append((text, start_idx))
                    count += 1
                    
        except StopIteration:
            print("⚠️ Fin du dataset atteinte ! On repart du début au prochain tour.")
            # Optionnel : relancer l'iterator si tu veux tourner en boucle à l'infini sur un petit dataset
            # self.iterator = iter(self.original_stream) 

        print(f"✅ Dataset rechargé : {len(self.samples)} nouveaux échantillons prêts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Sécurité si l'index est hors limite (peut arriver lors du switch)
        if idx >= len(self.samples):
            idx = idx % len(self.samples)

        text, start_idx = self.samples[idx]
        tokens = encode(text)
        
        x = tokens[start_idx : start_idx + self.block_size]
        y = tokens[start_idx + 1 : start_idx + self.block_size + 1]
        
        # Padding robuste
        if len(x) < self.block_size: x = x + [0] * (self.block_size - len(x))
        if len(y) < self.block_size: y = y + [0] * (self.block_size - len(y))
            
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
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
        return torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

    def forward(self, x):
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

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.ln_f(x)
        return self.head(x)

def model_loss(model, x, targets):
    logits = model(x)
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=0)
    return loss

def generate(model, x, max_new_tokens):
    for _ in range(max_new_tokens):
        x_crop = x[:, -config["block_size"]:]
        logits = model(x_crop)[:, -1, :]
        probs = F.softmax(logits, dim=1)
        x_next = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, x_next], dim=1)
    return x

@torch.no_grad()
def evaluate_loss(model, dataloader, max_batches):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches: break
        losses.append(model_loss(model, x.to(device), y.to(device)).item())
    model.train()
    return np.mean(losses) if losses else 0.0

@torch.no_grad()
def evaluate_semantics(model, dataset, num_samples=4, max_new=50):
    model.eval()
    refs, cands = [], []
    indices = torch.randperm(len(dataset))[:num_samples]
    
    for idx in indices:
        x, _ = dataset[idx]
        mid = len(x) // 2
        ctx = x[:mid].unsqueeze(0).to(device)
        target = x[mid:mid+max_new].tolist()
        
        gen = generate(model, ctx, max_new)[0, mid:].tolist()
        cands.append(decode(gen))
        refs.append(decode(target))
        
    emb1 = semantic_model.encode(cands, convert_to_tensor=True)
    emb2 = semantic_model.encode(refs, convert_to_tensor=True)
    cosine = torch.diag(util.cos_sim(emb1, emb2)).mean().item()
    
    try:
        _, _, F1 = bert_score_func(cands, refs, lang="en", verbose=False, device=device)
        bert_val = F1.mean().item()
    except:
        bert_val = 0.0
        
    model.train()
    return bert_val, cosine, cands, refs

# --- MAIN ---
if __name__ == "__main__":
    # =========================================================================
    # ÉTAPE 1 : INITIALISATION (À FAIRE UNE SEULE FOIS AU DÉBUT)
    # =========================================================================
    
    # 1. WandB & Config
    wandb.init(project=config["wandb_project"], resume="allow", config=config, id=config["wandb_id"])
    
    # 2. Création du Modèle & Optimiseur (Le "Cerveau")
    print("Création du modèle...")
    model = TransformerDecoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["max_lr"])
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. Reprise d'entraînement (Si le fichier existe)
    step = 0
    # if os.path.exists(config["checkpoint_name"]):
    #     print(f"♻️ Reprise depuis {config['checkpoint_name']}...")
    #     checkpoint = torch.load(config["checkpoint_name"], map_location=device)
    #     model.load_state_dict(checkpoint)
        # Idéalement, sauvegarder 'step' dans le checkpoint pour reprendre exactement au bon endroit
        # step = checkpoint.get('step', 0) 
    print("Streaming data ...")
    # On charge le stream UNE SEULE FOIS pour ne pas repartir du début à chaque refresh
    ds_stream = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
    
    # On instancie nos datasets intelligents
    # Train: prend les items 3 à 19 sur chaque bloc de 20
    train_ds = StreamingTextDataset(
        ds_stream.filter(lambda x, i: (i % 20) >= 3, with_indices=True), 
        config["block_size"]
    )
    # Val: prend les items 0, 1, 2 sur chaque bloc de 20
    val_ds = StreamingTextDataset(
        ds_stream.filter(lambda x, i: (i % 20) < 3, with_indices=True), 
        config["block_size"]
    )

    # =========================================================================
    # ÉTAPE 2 : BOUCLE INFINIE D'ENTRAÎNEMENT
    # =========================================================================
    model.train()
    print("🚀 Début de l'entraînement !")
    batch_num = 0
    while step < config["max_iters"]:
        batch_num+=1
        # A. Création des Loaders pour le chunk actuel
        # On doit les recréer car la taille du dataset change à chaque renew_data()
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)

        # B. Boucle sur les données actuelles (RAM)
        for x, y in train_loader:
            step += 1
            
            # --- LEARNING RATE & OPTIMIZER ---
            lr = get_lr(step)
            for param_group in optimizer.param_groups: param_group['lr'] = lr
            
            # --- EVALUATION ---
            if step % config["eval_every"] == 0:
                t_loss = evaluate_loss(model, train_loader, config["eval_iters"])
                v_loss = evaluate_loss(model, val_loader, config["eval_iters"])
                # Note: evaluate_semantics peut être lent, à commenter si besoin de vitesse
                bert, cos, cands, refs = evaluate_semantics(model, val_ds)
                
                print(f"Step {step}: Val Loss {v_loss:.4f} | BERT {bert:.4f} | Cos {cos:.4f}")
                wandb.log({
                    "step": step, "lr": lr,
                    "train_loss": t_loss, "val_loss": v_loss,
                    "bert_score": bert, "cosine_sim": cos,
                    "semantic_table": wandb.Table(columns=["Gen", "Ref"], data=list(zip(cands, refs))),
                    "batch_number" : batch_num
                })

            # --- SAUVEGARDE ---
            if step % config["save_every"] == 0:
                print(f"💾 Sauvegarde au step {step}...")
                fname = config["checkpoint_name"]
                torch.save(model.state_dict(), fname) # Sauvegarde simple des poids
                upload_to_drive(fname)

            # --- BACKPROPAGATION (APPRENTISSAGE) ---
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad() # Reset des gradients précédents
            loss = model_loss(model, x, y) # Calcul de l'erreur
            loss.backward() # Calcul de la correction nécessaire
            optimizer.step() # Application de la correction

            # Stop si on a fini
            if step >= config["max_iters"]: 
                break 

        # C. FIN DU CHUNK -> ON RECHARGE DE NOUVELLES DONNÉES
        if step < config["max_iters"]:
            print(f"--- Fin du chunk de données. Téléchargement de la suite... ---")
            train_ds.renew_data()
            val_ds.renew_data()
            
    wandb.finish()
    print("✅ Entraînement terminé avec succès.")
