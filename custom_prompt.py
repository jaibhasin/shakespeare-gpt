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

block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Match model params with what you trained ────────────────
model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
model.load_state_dict(torch.load("shakespeare.pt", map_location=device))
model.eval()

# ── Generate text ───────────────────────────────────────────
input_context = input('Enter a little context about the story you want :  ') # "Romeo : "
print(f"Your context : {input_context}")
# Convert the input context string to a list of token indices using the stoi mapping.
# If a character is not in the vocabulary, default to 0 (could also raise an error or skip).
context_indices = [stoi.get(ch, 0) for ch in input_context]
if not context_indices:
    # If the user entered nothing or only unknown chars, start with zero token.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
else:
    # Convert to tensor and reshape to (1, T)
    context = torch.tensor([context_indices], dtype=torch.long, device=device)

# Generate text from the model, using the context as a prompt.
generated = model.generate(context, max_new_tokens=1000)[0].tolist()
print("Since this is character level model, it doesn't understanding the meaning of the prompt!! \n")
print("We will have to use word level tokenizer for it .....\n ")
print("\n📝 Generated Story:\n" + "-"*30)
print(decode(generated))

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated = model.generate(context, max_new_tokens=1000)[0].tolist()
# print(decode(generated))
