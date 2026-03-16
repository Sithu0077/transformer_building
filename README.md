# 🧠 Transformer from Scratch — Build Your Own Mini-GPT

A complete, beginner-friendly implementation of the Transformer architecture in PyTorch — built from absolute zero, trained on Shakespeare.

---

## 📁 Repository Structure

```
transformers/
│
├── transformer_from_scratch.py   # Core model — all building blocks
├── train_transformer.py          # Training loop — train on your own text
├── input.txt                     # Shakespeare dataset (~1MB)
├── best_model.pt                 # Saved best model weights (auto-generated)
└── README.md                     # This file
```

---

## 🗺️ What's Built — Exercise by Exercise

| Exercise | File | What it does |
|---|---|---|
| Exercise 1 | `transformer_from_scratch.py` | Tokenization — text → integers |
| Exercise 2 | `transformer_from_scratch.py` | Embeddings — integers → vectors |
| Exercise 3 | `transformer_from_scratch.py` | Positional Encoding — inject word order |
| Exercise 4 | `transformer_from_scratch.py` | Self-Attention — words attend to each other |
| Exercise 5 | `transformer_from_scratch.py` | Multi-Head Attention — multiple perspectives |
| Exercise 6 | `transformer_from_scratch.py` | Feed Forward Network — non-linear processing |
| Exercise 7 | `transformer_from_scratch.py` | Full Transformer Block — everything assembled |
| Exercise 8 | `train_transformer.py` | Training Loop — train on Shakespeare |

---

## 🏗️ Architecture Overview

```
Raw Text
    ↓  Character Tokenizer
Token IDs  [40, 1842, 11875, ...]
    ↓  nn.Embedding
Token Vectors  [batch, seq, d_model]
    ↓  Positional Encoding  (sin/cos waves)
Token + Position Vectors
    ↓  TransformerBlock × N
    │   ├─ Masked Multi-Head Attention
    │   │     Q = what am I looking for?
    │   │     K = what do I contain?
    │   │     V = what will I give?
    │   ├─ Add & Norm  (residual connection)
    │   ├─ Feed Forward  (expand → ReLU → contract)
    │   └─ Add & Norm
    ↓  LayerNorm
    ↓  Linear Head  (d_model → vocab_size)
    ↓  Softmax
Predicted Next Token
```

---

## ⚙️ Model Configuration

```python
d_model    = 64      # each token = 64 numbers
num_heads  = 4       # 4 attention heads  →  d_k = 16 per head
d_ff       = 256     # feed forward inner size  (4 × d_model)
num_layers = 4       # stacked transformer blocks
max_len    = 128     # context window — max tokens model sees at once
dropout    = 0.1     # regularization — prevents overfitting
vocab_size = 65      # unique characters in Shakespeare
```

**GPT comparison:**

| Model | d_model | num_heads | num_layers | Parameters |
|---|---|---|---|---|
| This Mini-GPT | 64 | 4 | 4 | ~200K |
| GPT-2 Small | 768 | 12 | 12 | 117M |
| GPT-3 | 12288 | 96 | 96 | 175B |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch tiktoken
```

### 2. Run the full model pipeline

```bash
python transformer_from_scratch.py
```

Expected output:
```
EXERCISE 1 — Tokenization
  Sentence    : 'I love cats'
  Token IDs   : [40, 1842, 11875]

EXERCISE 2 — Embeddings
  Embeddings shape : torch.Size([3, 64])

...

🎉 Full Transformer built from scratch!
```

### 3. Train on Shakespeare

```bash
# Put input.txt in same folder, then:
python train_transformer.py
```

Expected training output:
```
Step     0/5000 | train loss: 4.1738 | val loss: 4.1702
Step   500/5000 | train loss: 2.8431 | val loss: 2.9012
Step  1000/5000 | train loss: 2.4217 | val loss: 2.5108
Step  2000/5000 | train loss: 2.1043 | val loss: 2.2341
Step  5000/5000 | train loss: 1.8821 | val loss: 1.9654
```

---

## 📊 Understanding the Loss

```
Loss ~4.1  →  pure random guessing  (untrained model)
Loss ~2.5  →  learned basic patterns
Loss ~2.0  →  learning word structure
Loss ~1.5  →  generating good Shakespeare-style text
Loss ~1.0  →  very well trained
```

**Loss = cross-entropy** — measures how wrong the model's predictions are.
Lower is better. The model learns by minimizing this value.

---

## 🎭 Sample Generated Text

After 5000 steps of training, the model generates:

```
First Citizen:
What say you to the people? I will not
speak to thee, and let us have the world
to the people that we may be the man
that you have done the world to come
...
```

It's not perfect Shakespeare — but it has learned:
- Character names followed by colons
- Sentence structure and punctuation
- Common Shakespearean words and phrases

---

## 🧮 Key Math Behind the Model

### Self-Attention
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

Q = x · W_q    "what am I looking for?"
K = x · W_k    "what do I contain?"
V = x · W_v    "what will I give?"
```

### Why divide by √d_k?
```
Larger d_k → larger dot products → softmax collapses → gradients die
Dividing by √d_k keeps scores in a stable range
```

### Positional Encoding
```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

Each position gets a unique fingerprint of sine/cosine waves
Added to token embeddings so model knows word order
```

### Layer Normalization
```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Keeps values stable after each sub-layer
Prevents exploding/vanishing activations
```

### Feed Forward Network
```
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂

Expand: d_model → d_ff  (4× bigger)
ReLU:   kill negatives → add non-linearity
Contract: d_ff → d_model  (back to original)
```

---

## 🔧 Tuning the Model

**Training too slow?** Reduce model size:
```python
d_model    = 32     # smaller vectors
num_layers = 2      # fewer blocks
num_steps  = 2000   # fewer steps
batch_size = 8      # smaller batches
```

**Want better quality?** Make model bigger:
```python
d_model    = 128
num_layers = 6
num_steps  = 10000
```

**Generation too repetitive?** Increase temperature:
```python
temperature = 1.2   # more random/creative
```

**Generation too random?** Decrease temperature:
```python
temperature = 0.5   # more focused/safe
```

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Model too big for GPU | Reduce `d_model` or `batch_size` |
| `AssertionError` in MHA | `d_model` not divisible by `num_heads` | Make sure `d_model % num_heads == 0` |
| Loss not decreasing | Learning rate wrong | Try `lr=1e-3` or `lr=1e-4` |
| `KeyError` in tokenizer | Unknown character in prompt | Only use characters from Shakespeare text |
| `fatal: pathspec 'best_model.pt'` | Git error — not a Python issue | Add `best_model.pt` to `.gitignore` |

---

## 📦 .gitignore Recommendation

```gitignore
# Model weights — too large for Git
best_model.pt
*.pt
*.pth

# Python cache
__pycache__/
*.pyc

# VS Code
.vscode/
```

---

## 📚 Concepts Covered

```
✅ Tokenization                 text → integers
✅ Token Embeddings             integers → learned vectors
✅ Positional Encoding          sine/cosine position fingerprints
✅ Dot Product Attention        measuring word similarity
✅ Scaled Attention             dividing by √d_k
✅ Softmax                      turning scores into probabilities
✅ Q, K, V Projections          three roles for each word
✅ Multi-Head Attention          multiple parallel perspectives
✅ Causal Masking               no peeking at future tokens
✅ Residual Connections          skip highways for gradients
✅ Layer Normalization           keeping activations stable
✅ Feed Forward Network          non-linear per-word processing
✅ Cross-Entropy Loss            measuring prediction error
✅ Backpropagation              computing gradients
✅ Adam Optimizer               updating weights
✅ Autoregressive Generation    one token at a time
✅ Temperature Sampling         controlling creativity
```

---

## 📖 Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — original Transformer paper (Vaswani et al., 2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — visual guide
- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) — clean GPT reference implementation
- [PyTorch Documentation](https://pytorch.org/docs/) — official PyTorch docs

---

## 👤 Author

Built from scratch as a learning exercise — every line understood, every concept derived from first principles.

**Learning path followed:**
```
Math foundations → Tokenization → Embeddings → Positional Encoding
→ Self-Attention → Multi-Head Attention → Feed Forward
→ Full Transformer Block → Training → Text Generation
```

---

*"I fear not the man who has practiced 10,000 kicks once,
but I fear the man who has practiced one kick 10,000 times."*

You built a Transformer 10,000 times in your head before writing one line. That's the right way. 🚀
