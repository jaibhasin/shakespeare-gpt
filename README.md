# Shakespeare GPT from Scratch

A minimal GPT-style language model trained on Shakespeare's works, built from scratch in PyTorch. Inspired by [Andrej Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=4990s&pp=iAQB).

## Features
- Transformer-based character-level language model
- Trains on the complete works of Shakespeare
- Simple, readable PyTorch code
- Jupyter notebook for step-by-step exploration
- Easily generate Shakespearean text samples

## Directory Structure
```
.
├── gpt/
│   └── model.py         # Transformer model implementation
├── notebooks/
│   └── built_from_scratch.ipynb  # Step-by-step notebook
├── shakespeare.txt      # Training data (Shakespeare's works)
├── train.py             # Script to train the model
├── sample.py            # Script to generate text samples
├── requirements.txt     # Python dependencies
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Download the dataset**
   - The repo includes `shakespeare.txt`. If you want to use a different dataset, replace this file.

## Training
Train the model from scratch on Shakespeare's text:
```bash
python train.py
```
- This will save the trained model as `shakespeare.pt`.
- Training uses default hyperparameters (see `train.py`).

## Generating Text
After training, generate new Shakespearean text:
```bash
python sample.py
```
- This loads `shakespeare.pt` and prints generated text to the console.

## Jupyter Notebook
Explore and experiment interactively:
- Open `notebooks/built_from_scratch.ipynb` in Jupyter.
- The notebook walks through data processing, model building, training, and generation, with explanations and code cells.

## Model Architecture
- Multi-head self-attention transformer (configurable layers/heads)
- Character-level tokenization
- Positional embeddings
- Residual connections and layer normalization

## Credits
- Inspired by [Andrej Karpathy's GPT from scratch tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=4990s&pp=iAQB)
- Shakespeare dataset from [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinysakespeare/input.txt)

---

Feel free to open issues or PRs for improvements! 