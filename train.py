import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from dataset.data import ArithmeticDataset, ArithmeticTokenizer, ArithmeticIterator, get_the_data, get_metrics
from model.transformer import get_transformer, BabyTransformer
from einops import rearrange
from time import ctime, perf_counter, strftime
import wandb
from utils import get_percent_and_loss, get_no_parameters, VOCAB_SIZE, num_time, safe_dtime

PROJECT_NAME = "Arthur's Grok 3"
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
    "project_name" : PROJECT_NAME,
    "run_name" : f"Run at {safe_dtime()}",
    "model_function" : get_transformer,
    "model_config" : DEFAULT_MODEL_CONFIG,
    "operator" : "+",
    "train_proportion" : 0.75,    
    "device" : DEVICE,
    "mini_batch_size" : MINI_BATCH_SIZE,
    "lr" : 0.0005,
    "weight_decay" : 1,
    "epochs" : 1000,
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
    epochs,
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

    for epoch_no in tqdm(range(epochs)):
        train_data, _ = get_the_data(
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

        # if training_percentage_correct>90 and not train_is_greater or train_is_greater:
        #     train_is_greater = True
        #     sched.step()

        train_prop, train_loss, valid_prop, valid_loss = get_metrics(model, operator, train_proportion, device)

        if train_prop > 0.95 and not train_is_greater: 
            train_is_greater = True
            t.save(model.state_dict(), f"checkpoints/train_90_{num_time()}.pt")
        if train_is_greater:
            sched.step()

        if valid_prop > 0.8 and not val_is_greater:
            val_is_greater = True
            t.save(model.state_dict(), f"checkpoints/valid_90_{num_time()}.pt")

        lr = sched.get_last_lr()[0]
        wandb_dict = {
            "training_accuracy" : train_prop,
            "training_loss" : train_loss,
            "validation_accuracy" : valid_prop,
            "validation_loss" : valid_loss,
            "lr" : lr,
        }
        wandb.log(wandb_dict)

    wandb.run.name = run_name + " that took " + str(int(perf_counter() - initial_time))
    wandb.run.finish()

if __name__ == "__main__":    
    for num_heads in [128]:
        model_config = dict(DEFAULT_MODEL_CONFIG)
        model_config["num_heads"] = num_heads

        run_config = dict(DEFAULT_RUN_CONFIG)
        run_config["model_config"] = model_config
        run_config["run_name"] = f"{num_heads} heads"
        run_config["save_models"] = True
        run_config["epochs"] = 10

        complete_run(**run_config)

    A = ArithmeticTokenizer()
    # tens = t.range(start=0, end=120).float()
    # print(A.decode(tens))
    # print() # TODO randomize the symbols; they're not REALLY the indistinguished things
    # input()
    # string = str(ctime()).replace(":", ".")
    # print(string)