import torch as t
from model.transformer import GPT2Output

class Embedding1D(t.nn.Module):
    def __init__(self, d, vocab_size):
        super().__init__()
        self.embeddings = t.nn.Parameter(t.randn(vocab_size, d))

    def forward(self, x):
        return self.embeddings[x[:, 0]] + self.embeddings[x[:, 1]]

class MLP(t.nn.Module):
    def __init__(self, vocab_size, d, device):
        super().__init__()
        self.module_list = t.nn.ModuleList([
            Embedding1D(d, vocab_size),
            t.nn.ReLU(),
            t.nn.Linear(d, 2000),
            t.nn.ReLU(),
            t.nn.Linear(2000, 2000),
            t.nn.ReLU(),
            t.nn.Linear(2000, 300),
            t.nn.ReLU(),
            t.nn.Linear(300, vocab_size),
        ]).to(device)

    def forward(self, x):
        output = x
        for module in self.module_list:
            output = module(output)
        return GPT2Output(output, t.zeros_like(output))