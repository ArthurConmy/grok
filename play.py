import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE
from dataset.data import get_the_data
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss

transformer = get_transformer(**DEFAULT_MODEL_CONFIG)
transformer.load_state_dict(t.load("newer_90.pt", map_location=t.device("cpu")))

for _ in range(10):
    train_data, valid_data = get_the_data(
        operator = "+",
        train_proportion = 0.5,
        mini_batch_size = MINI_BATCH_SIZE,
        device = DEVICE,
    )

    for data in [train_data, valid_data]:    
        for x, y in data:
            print(x)
            percent, loss = get_percent_and_loss(transformer, x, y)
            print(percent, loss)
            input()

    break

