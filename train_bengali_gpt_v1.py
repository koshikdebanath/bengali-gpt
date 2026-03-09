import os
import glob
import json
import math
import random
import pickle
import time
import sys
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Union
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Optional: progress bars and tensorboard
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# ------------------------------------------------------------------------------
# 1. BPE Tokenizer (trained on the Bengali corpus)
# ------------------------------------------------------------------------------

class BPETokenizer:
    """
    A simple Byte-Pair Encoding tokenizer that learns merge operations from text.
    Handles Bengali script (Unicode) by operating at the character level.
    Special tokens: <PAD>, <UNK>, <BOS>, <EOS> are added to the vocabulary.
    """
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # These will be filled during training
        self.char_to_id = {}          # mapping from characters to ids
        self.id_to_char = {}          # reverse mapping
        self.merges = {}               # mapping from pair (a,b) to merged token id
        self.vocab = []                # list of all tokens (strings)

    def _build_base_vocab(self, texts: List[str]) -> None:
        """Create initial vocabulary from all unique characters in the corpus."""
        chars = set()
        for text in texts:
            chars.update(text)
        # Add special tokens first
        base_tokens = self.special_tokens + sorted(chars)
        self.vocab = base_tokens[:]
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def _get_pair_stats(self, texts: List[str]) -> Counter:
        """Count frequencies of adjacent character pairs across the corpus."""
        pairs = Counter()
        for text in texts:
            # Represent text as list of token ids (initially characters)
            tokens = [self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text]
            for i in range(len(tokens)-1):
                pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def _merge_pair(self, texts: List[str], pair: Tuple[int, int], new_token_id: int) -> List[str]:
        """
        Replace all occurrences of the pair (a,b) in the corpus with the new token id.
        This function returns the updated texts (as strings of token ids, but we keep as list of ints for efficiency).
        For simplicity we work with token‑id lists.
        """
        new_texts = []
        a, b = pair
        for text in texts:
            # Represent as list of ids
            ids = [self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids)-1 and ids[i] == a and ids[i+1] == b:
                    new_ids.append(new_token_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            # Convert back to string (or keep as list of ids) – we'll keep as list for future merges.
            # But our merge function expects texts as strings? We'll keep everything in id lists to avoid re‑conversion.
            # We'll store the corpus as list of integer lists.
            new_texts.append(new_ids)
        return new_texts

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """
        Train the BPE tokenizer on a list of texts.
        Args:
            texts: list of raw strings.
            min_frequency: minimum frequency for a pair to be considered for merging.
        """
        # Build initial character vocabulary
        self._build_base_vocab(texts)
        
        # Convert texts to list of id lists for efficient processing
        corpus = [[self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text] for text in texts]
        
        # We'll maintain a mapping from merged token id to its string representation
        # Initially, every id corresponds to a single character or special token.
        # As we merge, we create new string tokens by concatenating the two parts.
        # We'll store the string representation in self.vocab (extend it).
        
        # Current vocabulary size
        current_vocab_size = len(self.vocab)
        
        # Continue merging until we reach desired vocab size
        pbar = tqdm(total=self.vocab_size - current_vocab_size, desc="Training BPE", disable=not tqdm) if tqdm else None
        while current_vocab_size < self.vocab_size:
            # Count pair frequencies across the corpus
            pair_counts = Counter()
            for ids in corpus:
                for i in range(len(ids)-1):
                    pair_counts[(ids[i], ids[i+1])] += 1
            
            if not pair_counts:
                break  # No more pairs to merge
            
            # Find the most frequent pair
            (best_pair), freq = max(pair_counts.items(), key=lambda x: x[1])
            if freq < min_frequency:
                break  # Stop if the best pair is too rare
            
            # Create a new token id and its string representation
            new_token_str = self.id_to_char[best_pair[0]] + self.id_to_char[best_pair[1]]
            new_token_id = current_vocab_size
            self.vocab.append(new_token_str)
            self.id_to_char[new_token_id] = new_token_str
            self.char_to_id[new_token_str] = new_token_id
            self.merges[best_pair] = new_token_id
            
            # Merge this pair in the entire corpus
            new_corpus = []
            for ids in corpus:
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids)-1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                        new_ids.append(new_token_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_corpus.append(new_ids)
            corpus = new_corpus
            current_vocab_size += 1
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a string into a list of token ids using the learned BPE merges.
        """
        # Start with character ids
        ids = [self.char_to_id.get(ch, self.char_to_id[self.unk_token]) for ch in text]
        
        # Apply merges repeatedly until no more merges can be applied
        # We can iterate through the merges in the order they were learned (greedy application)
        while True:
            # Find the first place where a merge can be applied (leftmost longest match is not strictly BPE,
            # but a simple greedy left‑to‑right pass works reasonably well.)
            i = 0
            merged = False
            new_ids = []
            while i < len(ids):
                if i < len(ids)-1 and (ids[i], ids[i+1]) in self.merges:
                    new_ids.append(self.merges[(ids[i], ids[i+1])])
                    i += 2
                    merged = True
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
            if not merged:
                break
        if add_special_tokens:
            ids = [self.char_to_id[self.bos_token]] + ids + [self.char_to_id[self.eos_token]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token ids back into a string.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_char[idx]
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return ''.join(tokens)

    def save(self, path: str) -> None:
        """Save the tokenizer to a file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'merges': self.merges,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)

    def load(self, path: str) -> None:
        """Load the tokenizer from a file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.char_to_id = data['char_to_id']
        self.id_to_char = data['id_to_char']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']


# ------------------------------------------------------------------------------
# 2. Dataset and DataLoader
# ------------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Loads all text files from a folder, tokenizes them with the provided tokenizer,
    and creates chunks of size `block_size` with stride `stride` for sliding window.
    """
    def __init__(self, folder_path: str, tokenizer: BPETokenizer, block_size: int = 512, stride: int = 256, split: str = "train", val_ratio: float = 0.1):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.examples = []
        
        # Read all .txt files in the folder (recursively)
        file_paths = glob.glob(os.path.join(folder_path, '**', '*.txt'), recursive=True)
        # Optional: split into train/val
        random.shuffle(file_paths)
        split_idx = int(len(file_paths) * (1 - val_ratio))
        if split == "train":
            file_paths = file_paths[:split_idx]
        else:
            file_paths = file_paths[split_idx:]
        
        for file_path in tqdm(file_paths, desc=f"Loading {split} data", disable=not tqdm):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Encode the entire text
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            # Create sliding windows
            for i in range(0, len(token_ids) - block_size, stride):
                self.examples.append(token_ids[i:i+block_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return input (x) and target (y) for next token prediction
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """
    Collate function for DataLoader. Pads sequences to the same length.
    """
    xs, ys = zip(*batch)
    # Find max length in batch
    max_len = max(x.shape[0] for x in xs)
    
    padded_xs = []
    padded_ys = []
    for x, y in zip(xs, ys):
        pad_len = max_len - x.shape[0]
        padded_xs.append(F.pad(x, (0, pad_len), value=0))  # pad with 0 (token id for <PAD>)
        padded_ys.append(F.pad(y, (0, pad_len), value=-100))  # use -100 to ignore in loss
    return torch.stack(padded_xs), torch.stack(padded_ys)


# ------------------------------------------------------------------------------
# 3. GPT Model Components
# ------------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch's default doesn't allow bias=False."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi‑head masked self‑attention."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Simple MLP with GELU activation."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre‑normalization."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT language model."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
      # Separate parameters into decay/no_decay sets
      decay = set()
      no_decay = set()
      whitelist_weight_modules = (torch.nn.Linear,)
      blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

      for mn, m in self.named_modules():
          for pn, p in m.named_parameters():
              fpn = f'{mn}.{pn}' if mn else pn
              if pn.endswith('bias'):
                  no_decay.add(fpn)
              elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                  decay.add(fpn)
              elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                  no_decay.add(fpn)

      # Get all parameter names from the model
      param_dict = {pn: p for pn, p in self.named_parameters()}

      # Verify that every parameter is assigned to exactly one group
      inter_params = decay & no_decay
      union_params = decay | no_decay
      missing_params = param_dict.keys() - union_params

      assert len(inter_params) == 0, f"Parameters {inter_params} made it into both decay/no_decay sets!"
      assert len(missing_params) == 0, f"Parameters {missing_params} were not separated into either decay/no_decay set!"

      # Build optimizer groups safely
      optim_groups = [
          {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": weight_decay},
          {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
      ]

      # Optional: fused AdamW for speed
      use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
      print(f"Using fused AdamW: {use_fused}")
      extra_args = dict(fused=True) if use_fused else {}
      optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
      return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ------------------------------------------------------------------------------
# 4. Configuration and Training Loop
# ------------------------------------------------------------------------------

class Config:
    """Simple object to hold model and training hyperparameters."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def train(config):
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load or train tokenizer
    tokenizer_path = os.path.join(config.data_folder, "bpe_tokenizer.pkl")
    if os.path.exists(tokenizer_path) and not config.retrain_tokenizer:
        print("Loading existing tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer...")
        # Read all text files to build corpus
        all_texts = []
        file_paths = glob.glob(os.path.join(config.data_folder, '**', '*.txt'), recursive=True)
        for file_path in tqdm(file_paths, desc="Reading texts", disable=not tqdm):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.train(all_texts, min_frequency=2)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}")

    config.vocab_size = len(tokenizer.vocab)
    config.block_size = config.block_size  # already set

    # 2. Create datasets and dataloaders
    train_dataset = TextDataset(config.data_folder, tokenizer, block_size=config.block_size, stride=config.stride, split="train", val_ratio=config.val_ratio)
    val_dataset = TextDataset(config.data_folder, tokenizer, block_size=config.block_size, stride=config.stride, split="val", val_ratio=config.val_ratio) if config.val_ratio > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn
        )
    else:
        val_loader = None

    print(f"Train dataset has {len(train_dataset)} examples.")
    if val_dataset:
        print(f"Validation dataset has {len(val_dataset)} examples.")

    # 3. Initialize model
    model = GPT(config)
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # 4. Optimizer and learning rate scheduler
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type
    )

    # Cosine learning rate scheduler with warmup
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    # 5. Training loop
    scaler = GradScaler(enabled=(device_type == 'cuda'))
    best_val_loss = float('inf')
    iter_num = 0
    running_loss = 0.0

    # TensorBoard writer
    if config.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(config.output_dir, "logs"))
    else:
        writer = None

    # Create checkpoint directory
    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}") if tqdm else enumerate(train_loader)
        for batch_idx, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Determine learning rate for this iteration
            lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass with mixed precision
            with autocast(enabled=(device_type == 'cuda')):
                _, loss = model(x, y)

            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Update running loss
                running_loss = running_loss * 0.9 + loss.item() * config.gradient_accumulation_steps * 0.1

                # Logging
                if iter_num % config.log_interval == 0:
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    print(f"Epoch {epoch}, iter {iter_num}, loss: {current_loss:.4f}, running: {running_loss:.4f}, lr: {lr:.2e}")
                    if writer:
                        writer.add_scalar('train/loss', current_loss, iter_num)
                        writer.add_scalar('train/lr', lr, iter_num)

                # Save checkpoint
                if iter_num % config.save_interval == 0 and iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'running_loss': running_loss
                    }
                    torch.save(checkpoint, os.path.join(config.output_dir, f"checkpoint_{iter_num}.pt"))
                    print(f"Checkpoint saved at iter {iter_num}")

                iter_num += 1

        # End of epoch: validation
        if val_loader:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for x_val, y_val in tqdm(val_loader, desc="Validating", disable=not tqdm):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    with autocast(enabled=(device_type == 'cuda')):
                        _, loss_val = model(x_val, y_val)
                    val_loss_total += loss_val.item()
                    val_steps += 1
            avg_val_loss = val_loss_total / val_steps
            print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}")
            if writer:
                writer.add_scalar('val/loss', avg_val_loss, epoch)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
                print(f"Best model saved with val loss {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final_model.pt"))
    if writer:
        writer.close()
    print("Training completed.")


# ------------------------------------------------------------------------------
# 5. Generation Example
# ------------------------------------------------------------------------------

def generate(config, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer_path = os.path.join(config.data_folder, "bpe_tokenizer.pkl")
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    config.vocab_size = len(tokenizer.vocab)

    # Load model
    model = GPT(config)
    model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt"), map_location=device))
    model.to(device)
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_new_tokens, temperature=temperature, top_k=top_k)
    output_text = tokenizer.decode(output_ids[0].tolist())
    print("\nGenerated text:")
    print(output_text)
    print()


# ------------------------------------------------------------------------------
# 6. Main entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model for Bengali text")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing .txt files")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and final model")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size for BPE tokenizer")
    parser.add_argument("--block_size", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=128, help="Stride for sliding window dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro batch size (per GPU)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--bias", type=bool, default=True, help="Whether to use bias in LayerNorm and Linear layers")
    parser.add_argument("--learning_rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="Number of warmup iterations")
    parser.add_argument("--lr_decay_iters", type=int, default=50000, help="Number of iterations for cosine decay (should match total steps)")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate after decay")
    parser.add_argument("--decay_lr", type=bool, default=True, help="Whether to decay learning rate")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N iterations")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N iterations")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count()), help="Number of DataLoader workers")
    parser.add_argument("--retrain_tokenizer", action="store_true", help="Force retraining of tokenizer even if saved file exists")
    parser.add_argument("--generate", action="store_true", help="Run generation mode instead of training")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top‑k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of data to use for validation (0 to disable)")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Build config object
    config = Config(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        decay_lr=args.decay_lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        num_workers=args.num_workers,
        retrain_tokenizer=args.retrain_tokenizer,
        seed=args.seed,
        val_ratio=args.val_ratio,
        use_tensorboard=args.use_tensorboard
    )

    if args.generate:
        generate(config, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    else:
        train(config)
