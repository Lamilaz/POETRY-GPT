# IMPORTS
import tokenizers
import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Sequence, Split, Digits
from tokenizers import normalizers
from datasets import load_dataset
import random
import numpy as np

# HYPERPARAMETERS
dropout = 0.2
d_model = 512
vocab_size = 100
n_heads = 8
n_main_layers = 12
block_size = 512
batch_size = 32
lr = 1e-4
max_iters = 10000
eval_every = 100
eval_iters = 50
save_every = 1000
checkpoint = "lamilaz.pt"

# MODEL
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, 4*d_model)
        self.fc2 = torch.nn.Linear(4*d_model, d_model)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.relu(self.fc1(x))))

class MainLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        self.multihead = torch.nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.feed_forward = FeedForward()

    def compute_attn_mask(self, x):
        N, L = x.shape[:2]
        attn_mask = torch.tril(torch.ones(N, L, L))
        attn_mask = 1 - torch.unsqueeze(attn_mask, dim=1).repeat(1,self.n_head,1,1).reshape(-1, L, L)
        return attn_mask.to(dtype=bool, device=x.device)

    def forward(self, x):
        qkv = self.norm1(x)
        x = x + self.multihead(qkv, qkv, qkv, attn_mask=self.compute_attn_mask(x))[0]
        return x + self.ff(self.norm2(x))

class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        #hot one encode des tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        #hot one encode des positions
        self.position_embedding = nn.Embedding(block_size, d_model)
        #multihead1 & norm1
        self.main_layers = torch.nn.Sequential(*[MainLayer(d_model,n_heads) for i in range(n_main_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # embedding layer
        merged_embedding = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[0], device=device))
        #main layers
        merged_embedding = self.main_layers(merged_embedding)
        return self.linear(merged_embedding)

# SETUP DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# SETUP DATASET
dataset = load_dataset("rojagtap/bookcorpus")["train"]["text"][0:1_000_000]

# TOKENIZER
trainer = BpeTrainer(special_tokens=["[UNK]", "[EOS]", "[BOS]", "[PAD]"],max_token_length=10)
pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True) ])
normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase()])
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer
tokenizer.trainer = trainer
tokenizer.train_from_iterator(dataset, length=len(dataset)).save("lamilaz_tokenizer.json")
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# WANDB
wandb.init(
    project="nano-gpt",
    config={
        "model": "lamilaz",
        "batch_size": batch_size,
        "block_size": block_size,
        "d_model": d_model,
        "main_layers": n_main_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "learning_rate": lr
    }
)

# FORMAT
n = int(0.9 * len(dataset))
train_data = dataset[:n]
validation_data = dataset[n:]

# TRAIN UTILITIES
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def model_loss(model, x, targets):
    logits = model(x)
    #B : batch , T : longeur de la sequence, Vocabulary
    B, T, V = logits.shape
    logits = logits.view(B * T, V)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss

def generate(model, x, max_new_tokens):
    """
    Generate text through autoregressive next token prediction
    """

    # iterate until max new tokens reached
    # normally, special tokens [EOS] would be used to stop generation !
    for _ in range(max_new_tokens):
        # select in input a context windows of correct size
        x_crop = x[:, -block_size:]

        # predict
        logits = model(x_crop) # (B, T, C)

        # reshape and apply softmax
        logits = logits[:, -1, :] # (B, C)
        probs = F.softmax(logits, dim=1) # (B, C)

        # sample one token among the vocab,
        # each one having a probability of being sampled equal to the model predicted probability (= creativity, k=size(vocabulary))
        x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        x = torch.cat([x, x_next], dim=1)

    return x

@torch.no_grad()
def evaluate(model):
    model.eval()   # set model to test mode
    out =  {}
    for split, data in [("train", train_data), ("validation", validation_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data)
            loss = model_loss(model, x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

model = TransformerDecoder()
model = nn.DataParallel(model)
model = model.to(device)

print("Parameters:", sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for i in range(max_iters+1):
    if i % eval_every == 0:
        losses = evaluate(model)
        print(f"step {i}: train loss {losses['train']:.4f}, validation loss {losses['validation']:.4f}", flush=True)

        prompts = torch.tensor([encode(p) for p in ["Harry", "Hermione", "Ron", "Dumbledore", "Hagrid"]]).to(device)
        outputs = generate(model, prompts, 100)
        outputs = [[decode(output.tolist())] for output in outputs]
        wandb.log({
            "step": i,
            "train loss": losses["train"],
            "validation loss": losses["validation"],
            "samples": wandb.Table(columns=["samples"], data=outputs)
        })

    if i % save_every == 0:
        torch.save(model.module.state_dict(), f"{checkpoint}_{i}")

    model.train() # set model to train mode
    optimizer.zero_grad()
    x, y = get_batch(train_data)
    loss = model_loss(model, x, y)
    loss.backward()
    optimizer.step()

# save model
torch.save(model.module.state_dict(), checkpoint)


