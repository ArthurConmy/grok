import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from dataset.data import ArithmeticDataset, ArithmeticTokenizer, ArithmeticIterator, get_the_data
from model.transformer import get_transformer, BabyTransformer
from model.tegmark import MLP
from einops import rearrange
from time import ctime, perf_counter, strftime
import wandb
from utils import get_no_parameters, num_time, safe_dtime

VOCAB_SIZE = 119
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
    "seed" : 0,
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
MLP_MODEL_CONFIG = {
    "vocab_size" : VOCAB_SIZE,
}
MLP_RUN_CONFIG = dict(DEFAULT_RUN_CONFIG)
MLP_RUN_CONFIG["run_name"] = f"MLP at {safe_dtime()}"
MLP_RUN_CONFIG["model_function"] = MLP
MLP_RUN_CONFIG["model_config"] = MLP_MODEL_CONFIG
MLP_RUN_CONFIG["lr"] = 0.0005

def get_percent_and_loss(model, x, y):
    with t.no_grad():
        logits = model(x).logits
        probabilities = F.softmax(logits, dim=1)
    
        y_one_hot = F.one_hot(y, num_classes=VOCAB_SIZE).float()
        corrects = t.sum((t.argmax(probabilities, dim=1) == y).float())

        cross_entropy_loss = t.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(probabilities, y_one_hot)

        assert probabilities.shape[0] == y.shape[0]
        return corrects / probabilities.shape[0], loss

def get_metrics(model, operator, train_proportion, device):
    train_data, valid_data = get_the_data(
        operator = operator,
        train_proportion = train_proportion,
        mini_batch_size = -1,
        device = device,
    )

    for x, y in train_data:
        train_prop, train_loss = get_percent_and_loss(model, x, y)

    for x, y in valid_data:
        valid_prop, valid_loss = get_percent_and_loss(model, x, y)

    return train_prop, train_loss, valid_prop, valid_loss

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
    print(model)

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
            probs = probabilities.detach().clone().cpu()
            y2 = y.detach().clone().cpu()
            print("Devices:", probs.device, y2.device)
            corrects += t.sum((t.argmax(probs, dim=1) == y2).float())
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
            if save_models == 1:
                t.save(model.state_dict(), f"checkpoints/train_90_{num_time()}.pt")
        if train_is_greater:
            sched.step()

        if valid_prop > 0.8 and not val_is_greater:
            val_is_greater = True
            if save_models == 1:
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

        if type(save_models) == type([]) and epoch_no in save_models:
            t.save(model.state_dict(), f"checkpoints/seed1at{epoch_no}.pt")

    wandb.run.name = run_name + " that took " + str(int(perf_counter() - initial_time))
    wandb.run.finish()

if __name__ == "__main__":    
    model_config = dict(MLP_MODEL_CONFIG)
    run_config = dict(MLP_RUN_CONFIG)
    print(run_config)
    complete_run(**run_config)
    input("Done")

    for it_no in range(1, 20):
        model_config = dict(DEFAULT_MODEL_CONFIG)
        model_config["num_heads"] = 128
        model_config["seed"] = it_no
        # 0 to 170 as well as 850 to 1000 are the interesting times
        # maybe 400 and 700 for good measure, too

        run_config = dict(DEFAULT_RUN_CONFIG)
        run_config["model_config"] = model_config
        run_config["run_name"] = f"{model_config['num_heads']} heads at {num_time()}"
        run_config["save_models"] = True
        run_config["epochs"] = 1000
        run_config["save_models"] = [0, 30, 60, 90, 120, 150, 400, 700, 850, 880, 910, 940, 970, 999]

        complete_run(**run_config)
        break

    A = ArithmeticTokenizer()
    
    # tens = t.range(start=0, end=120).float() ## 22 is not 0 !!!
    # print(A.decode(tens))
    # print()
    # string = str(ctime()).replace(":", ".")
    # print(string)