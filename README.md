# Shakespeare GPT from Scratch

This repository contains a minimal transformer-based language model that learns directly from Shakespeare's text. The code is intentionally short and easy to follow, so you can understand how GPT-style models work under the hood.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (`requirements.txt` currently just lists PyTorch; install the appropriate build for your system.)

2. Train the model:
   ```bash
   python train.py
   ```
   The script saves the model weights to `shakespeare.pt`.

3. Generate text from the trained checkpoint:
   ```bash
   python sample.py
   ```

## Directory Overview

```
gpt/                 # Transformer model implementation
notebooks/           # Jupyter notebook walkthrough
shakespeare.txt      # Training data (tiny-shakespeare)
train.py             # Training script
sample.py            # Text generation script
requirements.txt     # Python dependencies
```

## Training on Your Own Data

Replace `shakespeare.txt` with any plain text file and run `python train.py` again. The model will learn the style and vocabulary of whatever data you provide.

### Example datasets
- **Song lyrics** – generate new verses in the style of a favorite artist
- **Movie or TV scripts** – create new scenes or dialogue
- **Classic novels** – emulate authors like Jane Austen or Herman Melville
- **Poetry collections** – craft new poems with a unique voice
- **Programming code** – experiment with autocompletion for your projects

Adjust hyperparameters at the top of `train.py` if your dataset is larger or smaller.

## Jupyter Notebook

The notebook `notebooks/built_from_scratch.ipynb` mirrors the code in a step-by-step format. Open it in Jupyter to explore the training loop, model architecture, and generation process interactively.

## Model Highlights

- Character-level tokenization
- Multi-head self-attention with residual connections
- Positional embeddings
- Minimal PyTorch code—no external training frameworks

## Credits

- Inspired by [Andrej Karpathy's GPT from scratch tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=4990s&pp=iAQB)
- Dataset derived from [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinysakespeare/input.txt)

Feel free to open issues or PRs to improve this project!
