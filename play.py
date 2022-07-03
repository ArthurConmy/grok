import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE
from dataset.data import get_the_data, get_metrics
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss

my_model_config = dict(DEFAULT_MODEL_CONFIG)
my_model_config["num_heads"] = 128
transformer = get_transformer(**my_model_config)
transformer.load_state_dict(t.load("ep4.pt")) # , map_location=t.device("cpu"))) # spice is a good training data thing # tensor(0.0458) tensor(0.) for ep4 # tensor(0.0261) tensor(0.0038)
# m = transformer.state_dict()
# input()
# print(transformer.state_dict())

for _ in range(10):
# 0.75
# tensor([ 79,  56,  30,  ...,  81, 118,  66])
# tensor(0.0079) tensor(0.0098)

    for _ in range(10):
        train_prop, train_loss, valid_prop, valid_loss = get_metrics(transformer, "+", 0.75, "cpu")

        # myx = t.load("myx.pt")
        # input() tensor(0.1129) tensor(0.0043)

        print(train_prop, valid_prop)
        input()
        # break