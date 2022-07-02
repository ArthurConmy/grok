import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from dataset.data import ArithmeticDataset, ArithmeticTokenizer, ArithmeticIterator
from model.transformer import get_transformer, BabyTransformer
from einops import rearrange
from time import ctime
import wandb
from utils import get_validation_data, VOCAB_SIZE

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
MINI_BATCH_SIZE = 512

if __name__ == "__main__":
    model = get_transformer(
        num_layers = 2,
        num_heads = 32,
        vocab_size = VOCAB_SIZE, 
        hidden_size = 256,
        dropout = 0.0,  # TOCHECK
        device = DEVICE,
    )
    
    nop = 0

    def list_prod(L):
        ans = 1
        for l in L:
            ans *= l
        return ans

    for p in model.parameters():
        nop += list_prod(p.shape)        
    print(nop, "NUMBER OF PARAMETERS")  

    a, b = ArithmeticDataset.splits(
        train_pct = 75, 
        operator = "+",
    )

    tdata = ArithmeticIterator(
        a,
        device = DEVICE,
        batchsize_hint = MINI_BATCH_SIZE,
    )

    vdata = ArithmeticIterator(
        b,
        device = DEVICE,
        batchsize_hint = MINI_BATCH_SIZE,
    )

    A = ArithmeticTokenizer()
    # tens = t.range(start=0, end=120).float()
    # print(A.decode(tens))
    # print() # TODO randomize the symbols; they're not REALLY the indistinguished things
    # input()

    string = str(ctime()).replace(":", ".")
    print(string)

    wandb.init(project=f"Arthur's Grok")
    wandb.run.name = f"LR 0.0005" # wandb.run.id

    cross_entropy_loss = t.nn.CrossEntropyLoss()
    opt = t.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1) # very fragile to a good learning rate
    # sched = t.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=50)
    sched = t.optim.swa_utils.SWALR(opt, anneal_strategy="linear", anneal_epochs=100, swa_lr=0.0001)
    is_greater_than_ninety = False

    for epoch_no in range(1000):
        i = 0
        losses = []        
        corrects = 0
        total = 0

        a, b = ArithmeticDataset.splits(
            train_pct = 75, 
            operator = "+",
        )

        tdata = ArithmeticIterator(
            a,
            device = DEVICE,
            batchsize_hint = MINI_BATCH_SIZE,
        )

        for x, y in tqdm(tdata):
            opt.zero_grad()
            i += 1

            logits = model(x).logits
            probabilities = F.softmax(logits, dim=1)

            y_one_hot = F.one_hot(y, num_classes=VOCAB_SIZE).float()
            corrects += t.sum((t.argmax(probabilities, dim=1) == y).float())
            total += min(probabilities.shape[0], y.shape[0])

            loss = cross_entropy_loss(probabilities, y_one_hot)
            losses.append(loss.item())
            loss.backward()
            opt.step()

        print(epoch_no)
        training_percentage_correct = ( 100 * corrects.item() ) / total

        if training_percentage_correct > 95 and not is_greater_than_ninety or is_greater_than_ninety:
            is_greater_than_ninety = True
            sched.step()

        vdata = ArithmeticIterator(
            b,
            device = DEVICE,
            batchsize_hint = -1,
        )

        for x, y in vdata:
            validation_percent_correct, validation_loss = get_percent_correct(model, x, y)

        lr = sched.get_last_lr()[0]
        # print(type(lr), lr)

        wandb_dict = {
            "percentage_correct" : training_percentage_correct,
            "training_loss" : sum(losses) / len(losses),
            "validation_percent_correct" : validation_percent_correct,
            "validation_loss" : validation_loss,
            "lr" : lr,
        }
        wandb.log(wandb_dict)