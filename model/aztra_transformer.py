# === model/aztra_transformer.py ===

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Causal mask oluştur
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, 512, 512)))

    def forward(self, x):
        B, T, C = x.shape

        # Q, K, V matrislerini hesapla
        q = self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, dk)
        k = self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, dk)
        v = self.w_v(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)  # (B, nh, T, dk)

        # Attention hesapla
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, nh, T, T)

        # Causal mask uygula
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, nh, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture (GPT-2 style)
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token ve position embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding = nn.Embedding(config['block_size'], config['d_model'])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config['d_model'],
                config['n_head'],
                config['ffn_hidden'],
                config.get('dropout', 0.1)
            ) for _ in range(config['n_layer'])
        ])

        # Final layer norm ve output projection
        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings (GPT-2 style)
        self.head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.config['block_size'], f"Sequence length {T} exceeds block size {self.config['block_size']}"

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)  # (1, T)

        # Embeddings
        tok_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_emb = self.position_embedding(pos)  # (1, T, d_model)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        logits = self.head(x)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits if loss is None else (logits, loss)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate new tokens autoregressively
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if needed
                idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]

                # Forward pass
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # Focus on last time step

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def count_parameters(self):
        """Model parametrelerini say"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)