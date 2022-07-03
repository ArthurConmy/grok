import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from dataset.data import ArithmeticDataset, ArithmeticTokenizer, ArithmeticIterator, get_the_data
from model.transformer import get_transformer, BabyTransformer
from einops import rearrange
from time import ctime, perf_counter
import wandb
from utils import get_validation_data, get_no_parameters, VOCAB_SIZE, safe_dtime

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
MINI_BATCH_SIZE = 512
DEFAULT_MODEL_CONFIG = {
    "num_layers" : 2,
    "num_heads" : 32,
    "vocab_size" : VOCAB_SIZE, 
    "hidden_size" : 256,
    "dropout" : 0.0, 
    "device" : DEVICE,
}
DEFAULT_RUN_CONFIG = {
    "project_name" : "Arthur's Grok",
    "run_name" : f"Run at {safe_dtime()}",
    "model_function" : get_transformer,
    "model_config" : DEFAULT_MODEL_CONFIG,
    "operator" : "+",
    "train_proportion" : 0.75,    
    "device" : DEVICE,
    "mini_batch_size" : MINI_BATCH_SIZE,
    "lr" : 0.0005,
    "weight_decay" : 1,
    "no_epochs" : 1000,
    "save_models" : False,
}

def complete_run(
    project_name,
    run_name,
    model_function,
    model_config,
    operator,
    train_proportion,    
    device,
    mini_batch_size,
    lr,
    weight_decay,
    no_epochs,
    save_models,
):    
    if "device" in model_config:
        assert model_config["device"] == device, f"{model_config['device']} != {device}"

    wandb.init(project=project_name, reinit=True)
    wandb.run.name = run_name
    print(f"Starting run {run_name}")
    initial_time = perf_counter()

    model = model_function(**model_config)
    get_no_parameters(model)

    cross_entropy_loss = t.nn.CrossEntropyLoss()
    opt = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    
    sched = t.optim.swa_utils.SWALR(opt, anneal_strategy="linear", anneal_epochs=100, swa_lr=0.0001)
    train_is_greater = False
    val_is_greater = False

    for epoch_no in tqdm(range(no_epochs)):
        train_data, valid_data = get_the_data(
            operator = operator,
            train_proportion = train_proportion,
            mini_batch_size = mini_batch_size,
            device = device,
        )

        i = 0
        losses = []        
        corrects = 0
        total = 0

        for x, y in train_data:
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

        training_percentage_correct = ( 100 * corrects.item() ) / total

        if training_percentage_correct > 95 and not train_is_greater or train_is_greater:

            if not train_is_greater:
                t.save(model.state_dict(), f"first_greater_than_90_{ctime()}.pt")

            train_is_greater = True
            sched.step()

        for x, y in valid_data:
            validation_percent_correct, validation_loss = get_validation_data(model, x, y)

        if not val_is_greater and validation_percent_correct > 95:
            # t.save("First val greater")
            val_is_greater = True

        lr = sched.get_last_lr()[0]
        wandb_dict = {
            "percentage_correct" : training_percentage_correct,
            "training_loss" : sum(losses) / len(losses),
            "validation_percent_correct" : validation_percent_correct,
            "validation_loss" : validation_loss,
            "lr" : lr,
        }
        wandb.log(wandb_dict)
    wandb.run.name = run_name + " that took " + str(int(perf_counter() - initial_time))
    wandb.run.finish()

if __name__ == "__main__":    
    wandb.init(project=f"Arthur's Grok", reinit=True)

    for num_heads in [32, 4, 8, 64, 128]:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        model_config["num_heads"] = num_heads

        run_config = dict(DEFAULT_RUN_CONFIG)
        run_config["model_config"] = model_config
        run_config["run_name"] = f"{num_heads} heads"
        run_config["save_models"] = True

        complete_run(**run_config)
        break

    A = ArithmeticTokenizer()
    # tens = t.range(start=0, end=120).float()
    # print(A.decode(tens))
    # print() # TODO randomize the symbols; they're not REALLY the indistinguished things
    # input()
    # string = str(ctime()).replace(":", ".")
    # print(string)