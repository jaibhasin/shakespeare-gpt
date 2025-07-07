# sample.py
import torch
from gpt.model import BigramLanguageModel

# ── Load vocab and model ────────────────────────────────────
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Match model params with what you trained ────────────────
model = BigramLanguageModel(
    vocab_size, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2
).to(device)
model.load_state_dict(torch.load("shakespeare.pt", map_location=device))
model.eval()

# ── Generate text ───────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000)[0].tolist()
print(decode(generated))
