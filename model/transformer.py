import torch as t
from einops import rearrange
import torch.nn.functional as F

class AttentionHeads(t.nn.Module): ## ... prevously called UniModule (?) 
    def __init__(self, hidden_size, num_heads):
        assert hidden_size % num_heads == 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.attention = t.nn.Linear(hidden_size, hidden_size * 3)
        self.output_proj = t.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        qkv = self.attention(x)
        q, k, v = t.split(qkv, self.hidden_size, dim=-1)
        q = rearrange(q, 'b s (n h) -> b n s h', n=self.num_heads)
        k = rearrange(k, 'b s (n h) -> b n s h', n=self.num_heads)
        v = rearrange(v, 'b s (n h) -> b n s h', n=self.num_heads)
        qk = t.einsum('bnil,bnjl->bnij', q, k)
        qk /= self.head_size**0.5
        qk = t.tril(qk)
        # setting everything qk can't attend to
        qk[t.triu(t.ones_like(qk), diagonal=1).bool()] = -1e4
        qk = F.softmax(qk, dim=-1)
        # qk: batch num_heads seq_len seq_len
        # v: batch seq_len num_heads head_size
        # out: batch num_heads seq_len head_size
        combined = t.einsum('bnij,bnjh->bnih', qk, v)
        combined = rearrange(combined, 'b n i h -> b i (n h)')
        return self.output_proj(combined)    

class TransformerBlock(t.nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        dropout: float, 
        layer_norm_epsilon: float
    ):
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)
        self.attention = AttentionHeads(hidden_size, num_heads)
        self.layer_norm2 = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)
        self.linear1 = t.nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = t.nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = t.nn.Dropout(dropout)

    def forward(self, x):
        layer_normed_1 = self.layer_norm1(x)
        attentioned = self.attention(layer_normed_1)
        attentioned_x = x + attentioned
        layer_normed_2 = self.layer_norm2(attentioned_x)
        mlped = self.dropout(self.linear2(F.gelu(self.linear1(layer_normed_2))))
        return attentioned_x + mlped

from dataclasses import dataclass
from torchtyping import TensorType

@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]

class Transformer(t.nn.Module):
    def __init__(self, 
        num_layers, 
        num_heads, 
        vocab_size, 
        hidden_size,
        max_position_embeddings, 
        dropout, 
        layer_norm_epsilon,
        device: t.cuda.Device,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = t.nn.Parameter(t.randn(vocab_size, hidden_size))
        self.position_embedding = t.nn.Parameter(t.randn(max_position_embeddings, hidden_size))
        self.dropout = t.nn.Dropout(dropout)
        self.blocks = t.nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                dropout,
                layer_norm_epsilon
            ) for _  in range(num_layers)
        ])
        self.layer_norm = t.nn.LayerNorm(normalized_shape=hidden_size, eps=layer_norm_epsilon)

        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads 
        self.num_layers = num_layers
        self.tokenizer = None
        self.device = device

    def forward(
        self, 
        input_ids
    ): # [batch, seq_len]
        seq_len = input_ids.shape[1]
        result = self.token_embedding[input_ids] + self.position_embedding[t.arange(seq_len)]
        result = self.dropout(result)
        for block in self.blocks:
            result = block(result)
        self._enc = result
        all_encodings = self.layer_norm(result)
        # print(f"all_encodings' shape is {all_encodings.shape}")
        final_encoding = all_encodings[:,-1,:]
        logits = t.einsum('bh,vh->bv', final_encoding, self.token_embedding)
        return GPT2Output(logits, final_encoding)

    def next_token(self, input_ids, temperature, freq_penalty=2.0):
        gpt2_output = self.forward(input_ids)
        logits = gpt2_output.logits[0]
        encoding = gpt2_output.final_encoding

        new_logits = logits / temperature ## - self.id_frequencies * freq_penalty
        probability_dist = F.softmax(new_logits, dim=0)
        # print(probability_dist)

        p = rearrange(probability_dist, '(d1 d2) -> d1 d2', d1=1)
        p = t.cumsum(p, dim=1) - t.rand(1, 1, device=self.device)
        p = (p > 0).float()
        result = t.argmax(p, dim=1).item()
        return result