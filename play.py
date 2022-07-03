import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE
from dataset.data import get_the_data
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss

transformer = get_transformer(**DEFAULT_MODEL_CONFIG)
transformer.load_state_dict(t.load("spice.pt", map_location=t.device("cpu"))) # spice is a good training data thing

for _ in range(10):
    train_data, valid_data = get_the_data(
        operator = "+",
        train_proportion = 0.75,
        mini_batch_size = MINI_BATCH_SIZE,
        device = DEVICE,
    )

    t2, v2 = get_the_data(
        operator = "+",
        train_proportion = 0.75,
        mini_batch_size = -1,
        device = DEVICE,
    )

    for x, y in t2:
        print(x)

    break