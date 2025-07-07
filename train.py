# train.py
import torch
from gpt.model import BigramLanguageModel

# ── Hyperparameters ─────────────────────────────────────────
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# ── Load data ───────────────────────────────────────────────
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# ── Initialize model ────────────────────────────────────────
model = BigramLanguageModel(
    vocab_size, block_size, n_embd, n_head, n_layer, dropout
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ── Training loop ───────────────────────────────────────────
for step in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# ── Save checkpoint ─────────────────────────────────────────
torch.save(model.state_dict(), "shakespeare.pt")
