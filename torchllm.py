#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from bs4 import BeautifulSoup
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Punctuation, Sequence, WhitespaceSplit
from tokenizers.trainers import BpeTrainer
from torch import nn
from torch.nn import functional as F

vocab_size = 20000
n_embd = 512
n_head = 8
n_layer = 4

batch_size = 4
block_size = 256
max_iters = 5000
learning_rate = 3e-4
device = "cuda"

MODEL_PATH = "model.pt"
TOKENIZER_PATH = "tokenizer.json"

logger = logging.getLogger()


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.80, device=0)


parser = argparse.ArgumentParser(description="fb2 gpt")

parser.add_argument(
    "--train",
    type=str,
    nargs="?",
    help="--train 'path to the fb2 books'",
)

parser.add_argument(
    "prompt",
    type=str,
    nargs="?",
    help="prompt",
)
args = parser.parse_args()


class Head(nn.Module):
    """
    A single head of causal self-attention.

    This layer maps input token representations into Query, Key, and Value vectors
    to compute attention affinities. It applies a lower-triangular mask (causal masking)
    to ensure that tokens can only attend to past and current positions, preventing
    information leakage from the future during autoregressive generation.
    """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections for attention mechanism (without bias as per standard GPT design)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask buffer: lower triangular matrix of ones to restrict future attention
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # Input shape: (Batch size, Time steps/Sequence length, Channels/Embedding dimension)
        B, T, C = x.shape

        # Project inputs to Keys and Queries
        k = self.key(x)  # Shape: (B, T, head_size)
        q = self.query(x)  # Shape: (B, T, head_size)

        # Compute attention scores (affinities) using scaled dot-product
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = (
            q @ k.transpose(-2, -1) * (C**-0.5)
        )  # Scaled by 1/sqrt(C) for variance stabilization

        # Apply causal masking: fill future tokens with -inf so Softmax zeroes them out
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Normalize weights along the last dimension to get probabilities
        wei = F.softmax(wei, dim=-1)  # Shape: (B, T, T)

        # Project inputs to Values and apply attention weights
        v = self.value(x)  # Shape: (B, T, head_size)

        # Weighted aggregation of values: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head parallel self-attention mechanism.

    This layer runs multiple Head modules in parallel, allowing the model
    to jointly attend to information from different representation subspaces
    at different positions. The independent outputs are concatenated and
    projected back to the residual pathway dimension.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # ModuleList containing independent instances of causal attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Projection layer to merge concatenated head outputs back into the hidden dimension (n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Input shape: (Batch size, Time steps, Channels/n_embd)

        # Compute forward pass for each head and concatenate outputs along the channel dimension
        # List of num_heads tensors of shape (B, T, head_size) -> (B, T, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Apply the linear projection to mix the signals across all attention heads
        # Shape: (B, T, n_embd) -> (B, T, n_embd)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    """
    A simple position-wise feed-forward neural network.

    This block is applied to every position independently and identically.
    It expands the representation space by a factor of 4 (as per the original
    Transformer architecture) to allow the model to process tokens individually
    and extract complex features, followed by a non-linear activation and a projection
    back to the embedding dimension.
    """

    def __init__(self, n_embd):
        super().__init__()
        # Sequential container to execute layers linearly
        self.net = nn.Sequential(
            # First linear layer expands the feature space (n_embd -> 4 * n_embd)
            nn.Linear(n_embd, 4 * n_embd),
            # Rectified Linear Unit (ReLU) introducing non-linearity to the network
            nn.ReLU(),
            # Second linear layer projects back to the original dimension (4 * n_embd -> n_embd)
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        # Input shape: (Batch size, Time steps, Channels/n_embd)
        # Output shape: (Batch size, Time steps, Channels/n_embd)
        return self.net(x)


class Block(nn.Module):
    """
    An isolated Transformer block combining Multi-Head Self-Attention and Feed-Forward networks.

    This layer follows the Pre-Layer Normalization (Pre-LN) architecture. It routes
    the input tokens through communication (attention) and computation (feed-forward) phases,
    using residual (skip) connections around both sub-blocks to prevent vanishing gradients
    during backpropagation in deep architectures.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # Calculate the feature dimensionality allocated to each independent attention head
        head_size = n_embd // n_head

        # Communication sub-block: orchestrates context sharing across different tokens
        self.sa = MultiHeadAttention(n_head, head_size)

        # Computation sub-block: processes feature states for each token independently
        self.ffwd = FeedForward(n_embd)

        # Layer normalization layers applied before the transformations (Pre-LN formulation)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Input shape: (Batch size, Time steps, Channels/n_embd)

        # 1. Multi-Head Attention phase with its corresponding Pre-LN and Residual Connection
        # x -> LayerNorm -> Attention -> Add original x (Skip Connection)
        x = x + self.sa(self.ln1(x))

        # 2. Feed-Forward network phase with its corresponding Pre-LN and Residual Connection
        # x -> LayerNorm -> FeedForward -> Add accumulated x (Skip Connection)
        x = x + self.ffwd(self.ln2(x))

        # Output shape matches the input shape exactly: (Batch size, Time steps, Channels/n_embd)
        return x


class GPTLanguageModel(nn.Module):
    """
    A Decoder-only Generative Pre-trained Transformer (GPT) language model.

    This class implements an autoregressive language model. It combines learnable token and
    positional embeddings, a modular stack of Transformer blocks, and a final linear
    projection layer to compute next-token vocabulary distributions.
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # Token embedding lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position embedding lookup table to provide positional awareness
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        # Final layer normalization layer
        self.ln_f = nn.LayerNorm(n_embd)

        # Linear layer mapping hidden states back to vocabulary dimensions
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Extract token and positional embeddings
        tok_emb = self.token_embedding_table(idx)  # Shape: (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # Shape: (T, n_embd)

        # Combine embeddings and process through the Transformer backbone
        x = tok_emb + pos_emb  # Shape: (B, T, n_embd)
        x = self.blocks(x)  # Shape: (B, T, n_embd)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Flatten the matrices to compute standard Cross-Entropy Loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context window to the maximum block size supported by positional embeddings
            idx_cond = idx[:, -self.block_size :]

            # Fetch the predictions for the current sequence
            logits, loss = self(idx_cond)

            # Focus only on the logits of the last time step
            logits = logits[:, -1, :]  # Shape: (B, vocab_size)

            # Apply softmax to get probability distributions
            probs = F.softmax(logits, dim=-1)

            # Sample the next token index from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if args.train:
    dir_path = Path(args.train)
    # Validate that the training path exists and is a directory
    if not dir_path.exists():
        raise FileNotFoundError(f"The specified path '{dir_path}' does not exist!")
    if not dir_path.is_dir():
        raise NotADirectoryError(
            f"The path '{dir_path}' must be a directory (folder), not a file!"
        )

    # Recursively find all .fb2 files (case-insensitive handling)
    fb2_files = list(dir_path.rglob("*.fb2")) + list(dir_path.rglob("*.FB2"))

    if not fb2_files:
        raise FileNotFoundError(
            f"No files with the .fb2 extension were found in the directory '{dir_path}'"
        )

    logger.warning(f"Books found: {len(fb2_files)}")

    paragraphs = []

    # Iterate through all discovered books and parse their contents
    for file_path in fb2_files:
        logger.warning(f"Parsing book: {file_path.name} ...")

        # Open the file in binary mode to allow BeautifulSoup to detect encoding safely
        with open(file_path, "rb") as file:
            soup = BeautifulSoup(file, "xml")

        # Extract textual content from all XML <p> tags
        book_paragraphs = [p.get_text() for p in soup.find_all("p")]
        paragraphs.extend(book_paragraphs)

    # Consolidate all book paragraphs into a single text corpus
    text = "\n".join(paragraphs)
    logger.warning(f"Successfully collected {len(paragraphs)} paragraphs of text.")

    # Initialize the Byte Pair Encoding (BPE) tokenizer with an unknown token fallback
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Configure pre-tokenization rules to split text by whitespace and handle punctuation
    tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation()])

    # Define control and padding special tokens required for model training and formatting
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[NL]"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    # Train the tokenizer subword units using the extracted text paragraphs iterator
    tokenizer.train_from_iterator(paragraphs, trainer)
    vocab_size = tokenizer.get_vocab_size()

    logger.warning(f"Resulting vocabulary size: {vocab_size}")

    # Convert the raw textual corpus into numerical token IDs
    encoded_text = tokenizer.encode(text).ids

    # Structure data into a 1D long integer tensor for PyTorch compatibility
    data = torch.tensor(encoded_text, dtype=torch.long)

    # Perform a 90/10 train-validation split across the text corpus
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split="train"):
        """
        Generates a small batch of inputs (x) and targets (y) for training or validation.

        Args:
            split (str): Specifies the dataset slice to sample from ('train' or 'val').

        Returns:
            tuple: Context window input tensors (x) and next-token target tensors (y) mapped to device memory.
        """
        current_data = train_data if split == "train" else val_data

        # Sample random starting index offsets for the batch sequences
        ix = torch.randint(len(current_data) - block_size, (batch_size,))

        # Stack sequential input contexts and shifted next-token target vectors
        x = torch.stack([current_data[i : i + block_size] for i in ix])
        y = torch.stack([current_data[i + 1 : i + block_size + 1] for i in ix])

        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model):
        """
        Evaluates the model's loss across training and validation splits
        without tracking gradients to save memory and compute.

        Args:
            model (nn.Module): The language model instance being evaluated.

        Returns:
            dict: Average cross-entropy loss values for both 'train' and 'val' splits.
        """
        out = {}
        # Set the model to evaluation mode (disables dropout and batchnorm updates)
        model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(100)
            # Sample multiple batches to get a stable, less noisy loss estimate
            for k in range(100):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            # Compute the arithmetic mean of the sampled batch losses
            out[split] = losses.mean()

        # Revert the model back to active training mode
        model.train()
        return out

    # Initialize the GPT model architecture and move its weights to the target hardware device (CPU/CUDA)
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size).to(device)

    # Setup AdamW optimizer, a standard choice for training stable Transformer networks
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    logger.warning("Training the GPT model...")
    for iter in range(max_iters):
        # Periodically evaluate model performance on training and validation splits
        if iter % 1000 == 0:
            losses = estimate_loss(model)
            logger.warning(
                f"Iteration {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # Sample a random batch of sequential data from the training split
        xb, yb = get_batch("train")

        # Run the forward pass to predict next tokens and compute the current batch loss
        logits, loss = model(xb, yb)

        # Flush accumulated gradients using set_to_none=True to reduce memory footlogger.warning overhead
        optimizer.zero_grad(set_to_none=True)

        # Execute backpropagation to calculate new structural gradients
        loss.backward()

        # Update model weights based on optimizer internal states and calculated gradients
        optimizer.step()

    logger.warning("\nSaving trained model weights and configurations...")

    # Pack the state dictionaries along with hyperparameter metadata for seamless reconstruction
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
    }

    # Serialize the checkpoint data and the trained tokenizer configuration onto the disk
    torch.save(checkpoint, MODEL_PATH)
    tokenizer.save(TOKENIZER_PATH)

    logger.warning(
        f"Artifacts successfully saved to:\nModel: {MODEL_PATH}\nTokenizer: {TOKENIZER_PATH}"
    )

else:
    # Verify that both the trained model weights and tokenizer artifact exist on disk
    if not Path(MODEL_PATH).exists() or not Path(TOKENIZER_PATH).exists():
        raise FileNotFoundError(
            f"\nError: Files {MODEL_PATH} or {TOKENIZER_PATH} were not found.\n"
            "Inference cannot be executed because the model has not been trained yet.\n"
            "Please run the script again with the training flag: --train"
        )

    # Reconstruct the tokenizer from the saved JSON configuration
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    # Load model checkpoints using weights_only=True to prevent unsafe unpickling code execution
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

    # Reconstruct the neural network's architecture using hyperparameters from the saved dictionary
    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        block_size=checkpoint["block_size"],
    ).to(device)

    # Populate the compiled model architecture with the saved learned weight tensors
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.warning("\n" + "=" * 50)
    # Calculate global parameter counts (total capacity and active training slices)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(f"Vocabulary Size:                {vocab_size}")
    logger.warning(f"Embedding Dimension (n_embd):   {n_embd}")
    logger.warning(f"Attention Heads (n_head):       {n_head}")
    logger.warning(f"Single Head Capacity (head):    {n_embd // n_head}")
    logger.warning(f"Transformer Blocks (n_layer):   {n_layer}")
    logger.warning(f"Context Window (block_size):    {block_size}")
    logger.warning(f"Compute Device (device):        {device.upper()}")
    logger.warning(f"Total Network Parameters:       {total_params:,}")
    logger.warning(f"Trainable Parameters:           {trainable_params:,}")
    logger.warning(
        f"Estimated File Footlogger.warning:       {total_params * 4 / (1024**2):.2f} MB"
    )
    logger.warning("=" * 50 + "\n")

if args.prompt:
    # Tokenize the user's string prompt to generate baseline input context IDs
    start_context = tokenizer.encode(args.prompt).ids
    context = torch.tensor([start_context], dtype=torch.long, device=device)

    # Switch the model to evaluation state to freeze any active dropout filters
    model.eval()

    # Run the forward generation pipeline loop to predict new sequential token IDs
    generated_tokens = model.generate(context, max_new_tokens=150).tolist()[0]

    # Convert the resulting collection of vocabulary IDs back into plain text string
    r = tokenizer.decode(generated_tokens)
    logger.warning(r)

    # Initialize the TTS module component to vocalize the newly synthesized string response
    from llmbot import BotActor

    bot = BotActor()
    bot.speak(r)
