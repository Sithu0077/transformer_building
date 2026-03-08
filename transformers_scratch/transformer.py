import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken

# ============================================================
#  TRANSFORMER FROM SCRATCH — Complete Code
#  Exercises 1 → 7 + Full Transformer Block
# ============================================================

# ────────────────────────────────────────────────────────────
# GLOBAL SETTINGS — change these to experiment
# ────────────────────────────────────────────────────────────
vocab_size = 50257   # GPT-2 tokenizer vocab
d_model    = 8       # each token = 8 numbers (use 64+ for real models)
num_heads  = 2       # number of attention heads
d_k        = d_model // num_heads   # 4 per head
d_ff       = d_model * 4            # feed forward inner size = 32
max_len    = 100     # max sentence length
dropout    = 0.1     # dropout rate

torch.manual_seed(42)


# ============================================================
# EXERCISE 1 — Raw Text → Token IDs
# ============================================================
print("\n" + "="*60)
print("EXERCISE 1 — Tokenization")
print("="*60)

enc       = tiktoken.get_encoding("gpt2")
sentence  = "I love cats"
token_ids = enc.encode(sentence)

print(f"Sentence    : '{sentence}'")
print(f"Token IDs   : {token_ids}")
print(f"Num tokens  : {len(token_ids)}")
print(f"Decoded back: '{enc.decode(token_ids)}'")

print("\nIndividual tokens:")
for id in token_ids:
    print(f"  ID {id:6d}  →  '{enc.decode([id])}'")


# ============================================================
# EXERCISE 2 — Token IDs → Embeddings
# ============================================================
print("\n" + "="*60)
print("EXERCISE 2 — Embeddings")
print("="*60)

embedding_table = nn.Embedding(vocab_size, d_model)

token_tensor = torch.tensor(token_ids)
embeddings   = embedding_table(token_tensor)

print(f"Embedding table shape : {embedding_table.weight.shape}")
print(f"Token IDs shape       : {token_tensor.shape}")
print(f"Embeddings shape      : {embeddings.shape}")
# [3, 8] → 3 words, 8 dims each

# Batch version
batch = torch.tensor([
    enc.encode("I love cats too"),
    enc.encode("cats love me too"),  # both 4 tokens
    # NOTE: in real training all sentences in batch must be same length
    # (or use padding — we learn that later)
])
# Oops — different lengths! Let's fix with same length sentences
s1 = enc.encode("I love big cats")   # 4 tokens
s2 = enc.encode("cats love me too")  # 4 tokens
batch = torch.tensor([s1, s2])
batch_embeddings = embedding_table(batch)
print(f"Batch shape           : {batch.shape}")          # [2, 4]
print(f"Batch embeddings shape: {batch_embeddings.shape}")# [2, 4, 8]


# ============================================================
# EXERCISE 3 — Positional Encoding
# ============================================================
print("\n" + "="*60)
print("EXERCISE 3 — Positional Encoding")
print("="*60)

class PositionalEncoding(nn.Module):
    """
    Adds position information to token embeddings.
    Each position gets a unique pattern of sin/cos waves.
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()

        # Build the full PE table once — shape [1, max_len, d_model]
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)         # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )                                                         # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims ← sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims ← cosine

        # register_buffer: saved in model but NOT trained
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]            # add position to meaning


class InputPipeline(nn.Module):
    """
    Full input pipeline:
    token IDs → embeddings → + positional encoding
    Output shape: [batch, seq_len, d_model]
    """
    def __init__(self, vocab_size, d_model, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)

    def forward(self, token_ids):
        x = self.embedding(token_ids)   # integers → vectors
        x = self.pos_enc(x)             # add position info
        return x


pipeline   = InputPipeline(vocab_size, d_model)
token_ids  = torch.tensor([enc.encode("I love cats")])  # [1, 3]
pe_output  = pipeline(token_ids)

print(f"Token IDs shape  : {token_ids.shape}")    # [1, 3]
print(f"Pipeline output  : {pe_output.shape}")    # [1, 3, 8]
print("✓ Text is now ready to enter the Transformer")


# ============================================================
# EXERCISE 4 — Self Attention
# ============================================================
print("\n" + "="*60)
print("EXERCISE 4 — Self Attention")
print("="*60)

class SelfAttention(nn.Module):
    """
    Scaled Dot-Product Self Attention.
    Formula: Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

    Steps:
        1. Project x → Q, K, V  (three different roles)
        2. Score   = Q @ Kᵀ     (similarity between every word pair)
        3. Scale   = Score / √d_k (prevent softmax collapse)
        4. Weights = softmax(Scale) (probabilities, rows sum to 1)
        5. Output  = Weights @ V   (weighted blend of values)
    """
    def __init__(self, d_model, d_k):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        Q = self.W_q(x)                                   # [batch, seq, d_k]
        K = self.W_k(x)
        V = self.W_v(x)

        scores  = torch.matmul(Q, K.transpose(-2,-1))     # [batch, seq, seq]
        scores  = scores / math.sqrt(self.d_k)            # scale
        weights = F.softmax(scores, dim=-1)               # probabilities
        output  = torch.matmul(weights, V)                # [batch, seq, d_k]

        return output, weights


x       = torch.rand(1, 3, d_model)
sa      = SelfAttention(d_model=d_model, d_k=d_model)
out, wt = sa(x)

print(f"Input  shape  : {x.shape}")    # [1, 3, 8]
print(f"Output shape  : {out.shape}")  # [1, 3, 8]
print(f"Weights shape : {wt.shape}")   # [1, 3, 3]
print(f"Row sums (must be 1.0): {wt.squeeze().sum(dim=-1)}")


# ============================================================
# EXERCISE 5 — Multi Head Attention
# ============================================================
print("\n" + "="*60)
print("EXERCISE 5 — Multi Head Attention")
print("="*60)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention = run self-attention H times in parallel
    on H different slices of the vector, then concat results.

    Why? Each head can learn a different type of relationship:
        Head 1 → grammatical relationships
        Head 2 → semantic meaning
        Head 3 → positional closeness  ...etc

    Shape journey:
        [batch, seq, d_model]
        → split → [batch, heads, seq, d_k]
        → attend → [batch, heads, seq, d_k]
        → concat → [batch, seq, d_model]
        → W_o   → [batch, seq, d_model]
    """
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # [batch, seq, d_model] → [batch, heads, seq, d_k]
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x):
        batch, seq, _ = x.shape

        # Project + split into heads
        Q = self.split_heads(self.W_q(x))   # [batch, heads, seq, d_k]
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Attention on all heads simultaneously
        scores  = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attn    = torch.matmul(weights, V)              # [batch, heads, seq, d_k]

        # Concat heads back
        attn = attn.transpose(1,2).contiguous()
        attn = attn.view(batch, seq, self.d_model)      # [batch, seq, d_model]

        return self.W_o(attn)                           # mix heads together


x   = torch.rand(1, 3, d_model)
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
out = mha(x)

print(f"Input  shape : {x.shape}")    # [1, 3, 8]
print(f"Output shape : {out.shape}")  # [1, 3, 8] ✓
total_params = sum(p.numel() for p in mha.parameters())
print(f"MHA parameters: {total_params}")


# ============================================================
# EXERCISE 6 — Feed Forward Network
# ============================================================
print("\n" + "="*60)
print("EXERCISE 6 — Feed Forward Network")
print("="*60)

class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    Applied to EACH word independently after attention.

    Why needed?
        Attention = words gather info from each other  (linear)
        FFN       = each word processes what it gathered (non-linear)

    Formula:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
        i.e.   = Linear(expand) → ReLU → Linear(contract)

    Why expand then contract?
        Expanding to 4×d_model gives the model a wider space
        to find complex patterns, then contract back to d_model.

    Shape:
        [batch, seq, d_model]
        → Linear → [batch, seq, d_ff]      d_ff = 4 × d_model
        → ReLU  → [batch, seq, d_ff]       kill negatives
        → Linear → [batch, seq, d_model]   contract back ✓
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # EXPAND  8 → 32
        self.relu    = nn.ReLU()                   # NON-LINEARITY
        self.linear2 = nn.Linear(d_ff, d_model)   # CONTRACT 32 → 8

    def forward(self, x):
        x = self.linear1(x)   # [batch, seq, d_ff]
        x = self.relu(x)      # kill negatives
        x = self.linear2(x)   # [batch, seq, d_model]
        return x


x   = torch.rand(1, 3, d_model)
ffn = FeedForward(d_model=d_model, d_ff=d_ff)
out = ffn(x)

print(f"d_model    : {d_model}")     # 8
print(f"d_ff       : {d_ff}")        # 32  (4 × d_model)
print(f"Input  shape : {x.shape}")   # [1, 3, 8]
print(f"Output shape : {out.shape}") # [1, 3, 8] ← same shape ✓

ffn_params = sum(p.numel() for p in ffn.parameters())
print(f"FFN parameters: {ffn_params}")


# ============================================================
# EXERCISE 7 — Full Transformer Encoder Block
# ============================================================
print("\n" + "="*60)
print("EXERCISE 7 — Full Transformer Encoder Block")
print("="*60)

class TransformerBlock(nn.Module):
    """
    One complete Transformer Encoder Block.

    Architecture:
        x → MultiHeadAttention → Add & Norm → FeedForward → Add & Norm → output

    Two sub-layers, each wrapped with:
        1. Residual connection (Add):  output = x + SubLayer(x)
        2. Layer Normalization (Norm): stabilize values

    Why residual connections?
        They create a "skip highway" for gradients.
        Even if SubLayer learns nothing, x passes through unchanged.
        Prevents vanishing gradients in deep networks.

    Why Layer Norm?
        Keeps values in a stable range after each sub-layer.
        Prevents exploding/vanishing activations during training.

    Shape in = Shape out = [batch, seq, d_model]
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Multi-Head Attention
        self.mha   = MultiHeadAttention(d_model, num_heads)

        # Sub-layer 2: Feed Forward
        self.ffn   = FeedForward(d_model, d_ff)

        # Layer Norms — one after each sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization (prevents overfitting)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # ── Sub-layer 1: Multi-Head Attention ──────────────────────────
        attn_out = self.mha(x)               # attend
        attn_out = self.drop(attn_out)       # regularize
        x        = self.norm1(x + attn_out) # Add & Norm (residual)
        #                      ↑
        #              original x added back = residual connection

        # ── Sub-layer 2: Feed Forward ───────────────────────────────────
        ffn_out  = self.ffn(x)               # process each word
        ffn_out  = self.drop(ffn_out)        # regularize
        x        = self.norm2(x + ffn_out)  # Add & Norm (residual)

        return x


# ── Test one block ─────────────────────────────────────────────────────────
x     = torch.rand(1, 3, d_model)
block = TransformerBlock(d_model=d_model, num_heads=num_heads,
                         d_ff=d_ff, dropout=0.0)
out   = block(x)

print(f"Input  shape : {x.shape}")    # [1, 3, 8]
print(f"Output shape : {out.shape}")  # [1, 3, 8] ✓

block_params = sum(p.numel() for p in block.parameters())
print(f"Block parameters: {block_params}")


# ── Stack multiple blocks ──────────────────────────────────────────────────
print("\n--- Stacking 4 Transformer Blocks ---")

class TransformerEncoder(nn.Module):
    """
    Stack N Transformer blocks on top of each other.
    Each block refines the representation further.

    GPT-2 small  = 12 blocks
    GPT-2 large  = 36 blocks
    GPT-3        = 96 blocks
    """
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Each layer refines the representation
        return x


encoder      = TransformerEncoder(d_model=d_model, num_heads=num_heads,
                                   d_ff=d_ff, num_layers=4)
x            = torch.rand(1, 3, d_model)
encoder_out  = encoder(x)

print(f"Input  shape : {x.shape}")           # [1, 3, 8]
print(f"Output shape : {encoder_out.shape}") # [1, 3, 8] ✓

enc_params = sum(p.numel() for p in encoder.parameters())
print(f"Encoder (4 layers) parameters: {enc_params}")


# ============================================================
# FULL PIPELINE — Everything Together
# ============================================================
print("\n" + "="*60)
print("FULL PIPELINE — Text → Transformer")
print("="*60)

class MiniTransformer(nn.Module):
    """
    Complete Mini Transformer:

    Raw Text
        ↓  tokenizer (outside model)
    Token IDs
        ↓  nn.Embedding
    Token Vectors
        ↓  PositionalEncoding
    Token + Position Vectors
        ↓  TransformerBlock × N
    Contextual Representations
        ↓  Linear head (for language modeling)
    Logits over vocabulary
    """
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, max_len=100, dropout=0.1):
        super().__init__()

        # Input pipeline
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len)
        self.drop      = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output head: project d_model → vocab_size
        # Used for predicting next token (language modeling)
        self.head      = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        # token_ids: [batch, seq_len]

        # Step 1: token IDs → embeddings + position
        x = self.embedding(token_ids)    # [batch, seq, d_model]
        x = self.pos_enc(x)              # add position info
        x = self.drop(x)

        # Step 2: pass through all transformer blocks
        for block in self.blocks:
            x = block(x)                 # [batch, seq, d_model]

        # Step 3: project to vocabulary size
        logits = self.head(x)            # [batch, seq, vocab_size]

        return logits


# ── Full dry run ────────────────────────────────────────────────────────────
model = MiniTransformer(
    vocab_size = vocab_size,   # 50257
    d_model    = d_model,      # 8
    num_heads  = num_heads,    # 2
    d_ff       = d_ff,         # 32
    num_layers = 2,            # 2 stacked blocks
    max_len    = 100,
    dropout    = 0.0           # off for testing
)

sentence  = "I love cats"
token_ids = torch.tensor([enc.encode(sentence)])   # [1, 3]
logits    = model(token_ids)

print(f"\nSentence    : '{sentence}'")
print(f"Token IDs   : {token_ids.shape}")           # [1, 3]
print(f"Logits shape: {logits.shape}")              # [1, 3, 50257]
#                                                         ↑  ↑   ↑
#                                                     batch seq  vocab

# For each word position, we get a probability over all 50257 tokens
# The highest probability = predicted next token
predicted_ids    = logits.argmax(dim=-1)
print(f"Predicted IDs: {predicted_ids}")

predicted_tokens = [enc.decode([id.item()]) for id in predicted_ids[0]]
print(f"Predicted next tokens: {predicted_tokens}")
# Random predictions for now — model hasn't trained yet

# Total parameters
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total:,}")


# ============================================================
# SUMMARY — What Each Piece Does
# ============================================================
print("\n" + "="*60)
print("SUMMARY — Shape Flow Through Full Model")
print("="*60)
print(f"""
Raw text: '{sentence}'
    ↓  enc.encode()
Token IDs:          {token_ids.shape}
    ↓  nn.Embedding({vocab_size}, {d_model})
Embeddings:         [1, 3, {d_model}]
    ↓  PositionalEncoding
Pos Embeddings:     [1, 3, {d_model}]
    ↓  TransformerBlock × 2
    │   ├─ MultiHeadAttention ({num_heads} heads, d_k={d_k})
    │   ├─ Add & Norm
    │   ├─ FeedForward ({d_model}→{d_ff}→{d_model})
    │   └─ Add & Norm
Contextual vecs:    [1, 3, {d_model}]
    ↓  Linear({d_model} → {vocab_size})
Logits:             {logits.shape}
    ↓  argmax
Predicted tokens:   {predicted_tokens}

Total parameters:   {total:,}
""")

print("🎉 Full Transformer built from scratch!")
print("Next step → Exercise 8: Train this on real text")