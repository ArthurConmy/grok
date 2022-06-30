import matplotlib.pyplot as plt
from tqdm import tqdm
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

    cross_entropy_loss = t.nn.CrossEntropyLoss()
    opt = t.optim.AdamW(model.parameters(), betas=(0.9, 0.98), weight_decay=1.0)
    sched = t.optim.lr_scheduler.LinearLR(opt, start_factor=1e-8, total_iters=10)

    for epoch_no in range(100):
        i = 0
        losses = []        

        for x, y in tqdm(tdata):
            i += 1
            logits = model(x).logits
            probabilities = F.softmax(logits, dim=1)

            y_one_hot = F.one_hot(y, num_classes=VOCAB_SIZE).float()

            opt.zero_grad()
            loss = cross_entropy_loss(probabilities, y_one_hot)
            losses.append(loss.item())
            loss.backward()
            opt.step()

            # if i==10: 
                # print(probabilities)
                # print(model.blocks[0].attention.attention.weight)
                # input()

        print(epoch_no)
        print(sum(losses) / len(losses))

        # plt.hist(losses, bins=50)
        # plt.show()

        sched.step()

            # print(y_one_hot.shape)
            # print(y_one_hot[0])
            # break