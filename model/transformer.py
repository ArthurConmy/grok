import math
import torch as t
from einops import rearrange
import torch.nn.functional as F

class PositionalEncoding(t.nn.Module):
    """ From https://pyt.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.dropout = t.nn.Dropout(p=dropout)
        self.batch_first = batch_first
        
        if batch_first:
            pe = t.zeros(1, max_len, d_model)
            position = t.arange(0, max_len).unsqueeze(0).unsqueeze(2)
            div_term = t.exp(t.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[0, :, 0::2] = t.sin(position * div_term)
            pe[0, :, 1::2] = t.cos(position * div_term)
        else:
            pe = t.zeros(max_len, 1, d_model)
            position = t.arange(0, max_len).unsqueeze(1)
            div_term = t.exp(t.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0, 0::2] = t.sin(position * div_term)
            pe[:, 0, 1::2] = t.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            # print(x.shape, self.pe[:, :x.size(1)].shape)
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AttentionHeads(t.nn.Module):
    def __init__(self, hidden_size, num_heads):
        assert hidden_size % num_heads == 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.attention = t.nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output_proj = t.nn.Linear(hidden_size, hidden_size, bias=False)

        t.nn.init.xavier_uniform_(self.attention.weight, gain=1)
        t.nn.init.xavier_uniform_(self.output_proj.weight, gain=1)

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
    ):
        super().__init__()
        self.layer_norm1 = t.nn.LayerNorm(normalized_shape=hidden_size)
        self.attention = AttentionHeads(hidden_size, num_heads)
        self.layer_norm2 = t.nn.LayerNorm(normalized_shape=hidden_size)
        self.linear1 = t.nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.linear2 = t.nn.Linear(4 * hidden_size, hidden_size, bias = False)
        self.dropout = t.nn.Dropout(dropout)

        t.nn.init.xavier_uniform_(self.linear1.weight, gain=1)
        t.nn.init.xavier_uniform_(self.linear2.weight, gain=1)
        # t.nn.init.xavier_uniform_(self.linear1)

        self.layer_norm3 = t.nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, x):

        # x=self.attention(x)
        # x+=self.dropout(x)
        # x=self.layer_norm1(x)
        # x=self.dropout(F.gelu(self.linear1(x)))
        # return self.layer_norm2(x)

        # layer_normed_1 = self.layer_norm3(x)
        attentioned = self.attention(x)
        attentioned_x = x + attentioned
        layer_normed_2 = self.layer_norm2(attentioned_x)
        mlped = self.dropout(self.linear2(F.relu(self.linear1(layer_normed_2))))
        mlped = self.layer_norm1(mlped)
        return attentioned_x + mlped

from dataclasses import dataclass
from torchtyping import TensorType

@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class PositionalEncoding(t.nn.Module):
    """ From https://pyt.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.dropout = t.nn.Dropout(p=dropout)
        self.batch_first = batch_first
        
        if batch_first:
            pe = t.zeros(1, max_len, d_model)
            position = t.arange(0, max_len).unsqueeze(0).unsqueeze(2)
            div_term = t.exp(t.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[0, :, 0::2] = t.sin(position * div_term)
            pe[0, :, 1::2] = t.cos(position * div_term)
        else:
            pe = t.zeros(max_len, 1, d_model)
            position = t.arange(0, max_len).unsqueeze(1)
            div_term = t.exp(t.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0, 0::2] = t.sin(position * div_term)
            pe[:, 0, 1::2] = t.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            m = self.pe[:, :x.size(1)]
            # print(x.shape, m.shape)
            # input()
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(t.nn.Module):
    def __init__(self, 
        num_layers, 
        num_heads, 
        vocab_size, 
        hidden_size,
        dropout, 
        device: t.cuda.Device,
    ):
        super().__init__()
      
        self.vocab_size = vocab_size
        self.token_embedding = t.nn.Embedding(vocab_size, hidden_size) ## t.nn.Parameter(t.randn(vocab_size, hidden_size))
        self.position_embedding = PositionalEncoding(hidden_size, dropout-dropout, max_len=2, batch_first=True) ## t.nn.Parameter(t.randn(3, hidden_size)) # max position embeddings
        # t.nn.init.xavier_uniform_(self.position_embedding)

        self.dropout = t.nn.Dropout(dropout)
        self.blocks = t.nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                dropout,
            ) for _  in range(num_layers)
        ])
        self.token_unembedding = t.nn.Parameter(t.randn(vocab_size, hidden_size))
        t.nn.init.xavier_uniform_(self.token_unembedding)
        self.layer_norm = t.nn.LayerNorm(normalized_shape=hidden_size)

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
        # result = self.token_embedding(input_ids) + self.position_embedding[t.arange(seq_len)]
        result = self.position_embedding(self.token_embedding(input_ids))

        result = self.dropout(result)
        for block in self.blocks:
            result = block(result)
        self._enc = result
        all_encodings = self.layer_norm(result)

        final_encoding = all_encodings[:,-1,:]
        logits = t.einsum('bh,vh->bv', final_encoding, self.token_unembedding)
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

class TransposedLinear(t.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return x @ self.model.weight.T

    @property
    def weight(self):
        return self.model.weight

class BabyTransformer(t.nn.Module):
    def __init__(self, 
        # num_layers, 
        # num_heads, 
        vocab_size, 
        hidden_size,
        dropout, 
        device: t.cuda.Device,
    ):
        super().__init__()

        self.embedding = t.nn.Embedding(vocab_size, hidden_size)
        # self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=5, dropout=dropout, batch_first=True)
        # self.output = TransposedLinear(self.embedding)
        self.unembedding = t.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.unembedding(F.relu(self.embedding(x)))[:,-1,:]
        # return self.output(self.embedding(x) + self.pos_encoding(x))