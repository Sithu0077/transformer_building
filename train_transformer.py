import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import time
import os

# ============================================================
#  TRAIN YOUR OWN MINI-GPT ON SHAKESPEARE
#  Exercise 8 — Full Training Loop
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — CONFIGURATION
# Change these settings to experiment
# ────────────────────────────────────────────────────────────
class Config:
    # Data
    data_path   = "input.txt"       # your Shakespeare file

    # Model size — small enough to train on CPU in minutes
    d_model     = 64                # each token = 64 numbers
    num_heads   = 4                 # 4 attention heads
    d_ff        = 256               # feed forward inner size (4 × d_model)
    num_layers  = 4                 # number of transformer blocks
    max_len     = 128               # max sequence length
    dropout     = 0.1               # regularization

    # Training
    batch_size  = 16                # sentences per training step
    num_steps   = 5000              # total training steps
    lr          = 3e-4              # learning rate
    eval_every  = 500               # print loss every N steps
    eval_steps  = 50                # steps to average for eval loss

    # Generation
    gen_every   = 1000              # generate sample text every N steps
    gen_tokens  = 200               # how many tokens to generate
    temperature = 0.8               # creativity: low=safe, high=creative

    # Device
    device      = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
print(f"Device: {cfg.device}")
print(f"Training for {cfg.num_steps} steps")


# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD AND PREPARE DATA
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Loading Data")
print("="*60)

# Read the raw text
with open(cfg.data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters : {len(text):,}")
print(f"Sample text      : {repr(text[:100])}")

# ── Character-level tokenizer ──────────────────────────────
# Instead of GPT-2 tokenizer, we use CHARACTER level
# Why? Simpler vocab (65 chars) → model trains faster
# Each character gets a unique integer ID

# Build vocabulary from all unique characters in text
chars    = sorted(list(set(text)))
vocab_size = len(chars)

print(f"\nVocabulary size  : {vocab_size} unique characters")
print(f"Characters       : {''.join(chars[:30])}...")

# Create mappings: character ↔ integer
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode entire text to integers
def encode(s):
    return [char_to_id[c] for c in s]

def decode(ids):
    return ''.join([id_to_char[i] for i in ids])

# Convert full text to tensor
data    = torch.tensor(encode(text), dtype=torch.long)
print(f"Data tensor shape: {data.shape}")   # [total_chars]
print(f"First 20 IDs     : {data[:20].tolist()}")
print(f"Decoded back     : {repr(decode(data[:20].tolist()))}")

# ── Train / Validation split ───────────────────────────────
# 90% training, 10% validation
split    = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

print(f"\nTrain size: {len(train_data):,} characters")
print(f"Val size  : {len(val_data):,} characters")


# ────────────────────────────────────────────────────────────
# STEP 3 — BATCH LOADER
# ────────────────────────────────────────────────────────────
def get_batch(split='train'):
    """
    Randomly sample a batch of sequences from the data.

    For each sequence:
        x = characters at positions [i   : i+seq_len]   (input)
        y = characters at positions [i+1 : i+seq_len+1] (target)

    y is x shifted by 1 — the model must predict each next character.

    Example with seq_len=5:
        text  = "hello"
        x     = [h, e, l, l, o]
        y     = [e, l, l, o, !]
                  ↑
            each y[i] = next character after x[i]
    """
    source = train_data if split == 'train' else val_data

    # Random starting positions
    start_positions = torch.randint(
        low  = 0,
        high = len(source) - cfg.max_len - 1,
        size = (cfg.batch_size,)
    )

    # Build input and target tensors
    x = torch.stack([source[i   : i+cfg.max_len    ] for i in start_positions])
    y = torch.stack([source[i+1 : i+cfg.max_len + 1] for i in start_positions])

    return x.to(cfg.device), y.to(cfg.device)


# Test batch loader
x_test, y_test = get_batch('train')
print(f"\nBatch x shape: {x_test.shape}")   # [16, 128]
print(f"Batch y shape: {y_test.shape}")   # [16, 128]
print(f"\nFirst sequence input  : {repr(decode(x_test[0][:30].tolist()))}")
print(f"First sequence target : {repr(decode(y_test[0][:30].tolist()))}")
print("Notice: target is input shifted by 1 character ↑")


# ────────────────────────────────────────────────────────────
# STEP 4 — MODEL DEFINITION
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Building Model")
print("="*60)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.W_o  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Causal mask — pre-built for efficiency
        # Upper triangle = True = blocked (future positions)
        mask = torch.triu(torch.ones(512, 512), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        b, s, _ = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)

        # Apply causal mask — block future positions
        scores = scores.masked_fill(self.mask[:s, :s], float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.drop(weights)
        attn    = torch.matmul(weights, V)

        attn = attn.transpose(1,2).contiguous().view(b, s, self.d_model)
        return self.W_o(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha   = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm variant (more stable training)
        x = x + self.drop(self.mha(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class MiniGPT(nn.Module):
    """
    Complete GPT-style Language Model.

    Architecture:
        Token Embedding
        + Positional Encoding
        → N × TransformerBlock (with causal mask)
        → LayerNorm
        → Linear head → vocab logits

    Training objective:
        Given sequence x, predict next character at every position.
        Loss = cross entropy between predictions and true next chars.
    """
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Good initialization = faster, more stable training
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # x: [batch, seq_len]

        # Input pipeline
        x = self.embedding(x)          # [batch, seq, d_model]
        x = self.pos_enc(x)
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.head(x)           # [batch, seq, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross entropy
            # logits:  [batch, seq, vocab] → [batch*seq, vocab]
            # targets: [batch, seq]        → [batch*seq]
            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B*S, V),
                targets.view(B*S)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=200, temperature=0.8):
        """
        Generate text autoregressively.
        One token at a time. Each new token feeds back as input.

        Temperature:
            < 1.0 = more focused/repetitive
            > 1.0 = more random/creative
            = 1.0 = sample from raw probabilities
        """
        self.eval()
        x = prompt_ids.unsqueeze(0).to(cfg.device)   # [1, seq]

        for _ in range(max_new_tokens):
            # Only use last max_len tokens (context window)
            x_crop = x[:, -cfg.max_len:]

            # Forward pass
            logits, _ = self(x_crop)

            # Take logits at last position only
            logits_last = logits[:, -1, :]             # [1, vocab]

            # Apply temperature
            logits_last = logits_last / temperature

            # Convert to probabilities
            probs = F.softmax(logits_last, dim=-1)     # [1, vocab]

            # Sample next token
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Append to sequence
            x = torch.cat([x, next_id], dim=1)        # [1, seq+1]

        self.train()
        return x[0].tolist()


# ── Build model ────────────────────────────────────────────
model = MiniGPT(
    vocab_size = vocab_size,
    d_model    = cfg.d_model,
    num_heads  = cfg.num_heads,
    d_ff       = cfg.d_ff,
    num_layers = cfg.num_layers,
    max_len    = cfg.max_len,
    dropout    = cfg.dropout,
).to(cfg.device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Vocabulary size  : {vocab_size}")
print(f"Model parameters : {total_params:,}")
print(f"d_model          : {cfg.d_model}")
print(f"num_heads        : {cfg.num_heads}")
print(f"num_layers       : {cfg.num_layers}")
print(f"d_ff             : {cfg.d_ff}")


# ────────────────────────────────────────────────────────────
# STEP 5 — LOSS EVALUATION
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_loss():
    """
    Compute average loss on train and val sets.
    @torch.no_grad() = don't compute gradients (saves memory, faster)
    """
    model.eval()
    results = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(cfg.eval_steps):
            x, y      = get_batch(split)
            _, loss   = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


# ────────────────────────────────────────────────────────────
# STEP 6 — TRAINING LOOP
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6 — Training")
print("="*60)

# Optimizer — Adam works best for Transformers
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# Learning rate scheduler — reduce LR when loss plateaus
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.num_steps
)

print(f"\nStarting training for {cfg.num_steps} steps...")
print(f"Batch size : {cfg.batch_size}")
print(f"Seq length : {cfg.max_len}")
print(f"Learn rate : {cfg.lr}")
print("-"*60)

# Track losses for progress
train_losses = []
val_losses   = []
best_val_loss = float('inf')
start_time   = time.time()

# ── TRAINING LOOP ─────────────────────────────────────────────
for step in range(cfg.num_steps):

    # ── Evaluate periodically ──────────────────────────────────
    if step % cfg.eval_every == 0:
        losses   = evaluate_loss()
        elapsed  = time.time() - start_time
        lr_now   = optimizer.param_groups[0]['lr']

        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

        print(f"Step {step:5d}/{cfg.num_steps} | "
              f"train loss: {losses['train']:.4f} | "
              f"val loss: {losses['val']:.4f} | "
              f"lr: {lr_now:.6f} | "
              f"time: {elapsed:.0f}s")

        # Save best model
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), 'best_model.pt')

    # ── Generate sample text periodically ─────────────────────
    if step % cfg.gen_every == 0 and step > 0:
        print("\n--- Sample generation ---")
        # Start with "First Citizen:" as prompt
        prompt     = "First Citizen:"
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long)
        gen_ids    = model.generate(
            prompt_ids,
            max_new_tokens = cfg.gen_tokens,
            temperature    = cfg.temperature
        )
        gen_text = decode(gen_ids)
        print(f"Prompt: '{prompt}'")
        print(f"Generated:\n{gen_text[:300]}")
        print("-"*60)

    # ── Forward pass ───────────────────────────────────────────
    x, y       = get_batch('train')
    logits, loss = model(x, y)

    # ── Backward pass ──────────────────────────────────────────
    optimizer.zero_grad()    # clear old gradients
    loss.backward()          # compute new gradients

    # Gradient clipping — prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()         # update weights
    scheduler.step()         # update learning rate


# ────────────────────────────────────────────────────────────
# STEP 7 — FINAL EVALUATION
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7 — Final Results")
print("="*60)

final_losses = evaluate_loss()
total_time   = time.time() - start_time

print(f"Training complete!")
print(f"Total time     : {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
print(f"Final train loss: {final_losses['train']:.4f}")
print(f"Final val loss  : {final_losses['val']:.4f}")
print(f"Best val loss   : {best_val_loss:.4f}")
print(f"\nWhat does the loss mean?")
print(f"  Loss ~4.1 = random guessing (1/vocab_size = 1/{vocab_size})")
print(f"  Loss ~2.5 = model learned something")
print(f"  Loss ~1.5 = model learned patterns well")
print(f"  Loss ~1.0 = model learned very well")


# ────────────────────────────────────────────────────────────
# STEP 8 — GENERATE FINAL TEXT
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8 — Text Generation")
print("="*60)

# Load best model weights
model.load_state_dict(torch.load('best_model.pt', map_location=cfg.device))
print("Loaded best model weights\n")

# Generate from different prompts
prompts = [
    "First Citizen:",
    "ROMEO:",
    "To be or not",
    "\n",
]

for prompt in prompts:
    print(f"{'─'*50}")
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.long)
    gen_ids    = model.generate(
        prompt_ids,
        max_new_tokens = 300,
        temperature    = cfg.temperature
    )
    print(f"Prompt: '{prompt}'")
    print(f"Generated text:\n{decode(gen_ids)}\n")


# ────────────────────────────────────────────────────────────
# STEP 9 — INTERACTIVE MODE
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 9 — Interactive Generation")
print("Type your own prompt. Press Ctrl+C to exit.")
print("="*60)

try:
    while True:
        prompt = input("\nYour prompt: ")
        if not prompt:
            continue

        # Handle unknown characters
        clean_prompt = ''.join(c for c in prompt if c in char_to_id)
        if not clean_prompt:
            print("No valid characters found. Try again.")
            continue

        temp = input("Temperature (0.1-1.5, press Enter for 0.8): ").strip()
        temp = float(temp) if temp else 0.8

        prompt_ids = torch.tensor(encode(clean_prompt), dtype=torch.long)
        gen_ids    = model.generate(
            prompt_ids,
            max_new_tokens = 300,
            temperature    = temp
        )
        print(f"\nGenerated:\n{decode(gen_ids)}")

except KeyboardInterrupt:
    print("\n\nExiting interactive mode.")


# ────────────────────────────────────────────────────────────
# UNDERSTANDING YOUR LOSS CURVE
# ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("UNDERSTANDING YOUR TRAINING")
print("="*60)
print(f"""
Loss values you saw:
  Train losses: {[f"{l:.3f}" for l in train_losses]}
  Val   losses: {[f"{l:.3f}" for l in val_losses]}

What happened during training:
  Step 0      → random weights → loss ≈ 4.1 (pure guessing)
  Step 500    → learned basic patterns → loss dropping
  Step 1000+  → learning word structure → loss < 2.5
  Step 5000   → learned Shakespeare style → loss < 2.0

If val_loss > train_loss by a lot → overfitting
  Fix: increase dropout, reduce model size, get more data

If both losses are high → underfitting
  Fix: increase d_model, num_layers, train longer

If loss is not decreasing → learning rate issue
  Fix: try lr=1e-3 or lr=1e-4

Your model config:
  Parameters  : {total_params:,}
  d_model     : {cfg.d_model}
  num_heads   : {cfg.num_heads}
  num_layers  : {cfg.num_layers}
  Training    : {cfg.num_steps} steps
""")