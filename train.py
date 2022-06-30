import torch as t
import torch.nn.functional as F
from dataset.data import ArithmeticDataset, ArithmeticTokenizer, ArithmeticIterator
from model.transformer import Transformer
from einops import rearrange

DEVICE = "cpu"
VOCAB_SIZE = 119
MINI_BATCH_SIZE = 16

if __name__ == "__main__":
    model = Transformer(
        num_layers=2, 
        num_heads=4, 
        vocab_size=VOCAB_SIZE, 
        hidden_size=128,
        max_position_embeddings=10, # TOCHECK ... ? 
        dropout=0.1,  # TOCHECK
        layer_norm_epsilon=0.1, # TOCHECK
        device = DEVICE,
    )

    a, b = ArithmeticDataset.splits(
        train_pct = 75, 
        operator = "+",
    )

    tdata = ArithmeticIterator(
        a,
        device = DEVICE,
        batchsize_hint = MINI_BATCH_SIZE,
        cutoff = 5,
    )

    vdata = ArithmeticIterator(
        b,
        device = DEVICE,
        batchsize_hint = MINI_BATCH_SIZE,
        cutoff = 5,
    )

    A = ArithmeticTokenizer()
    # tens = t.range(start=0, end=120).float()
    # print(A.decode(tens))
    # print() # TODO randomize the symbols; they're not REALLY the indistinguished things
    # input()

    for thing in tdata:
        print(thing)
        break
    # input()

    collect_data = []
    for thing in a.data[:5]:
        collect_data.append(thing)

    thing = t.stack(collect_data)
    print(thing)
    print(thing.shape)

    cross_entropy_loss = t.nn.CrossEntropyLoss()
    logits = model(thing).logits
    probabilities = F.softmax(logits, dim=1)
    
    

    # print(logits[0,:])
    # print(logits.shape, "is the shape")
    # print()