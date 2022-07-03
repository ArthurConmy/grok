import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE
from dataset.data import get_the_data, get_metrics
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss
import seaborn
import matplotlib.pyplot as plt
from einops import repeat

my_model_config = dict(DEFAULT_MODEL_CONFIG)
my_model_config["num_heads"] = 128
transformer = get_transformer(**my_model_config)
transformer.load_state_dict(t.load("checkpoints/spice.pt", map_location=t.device("cpu")))

raw_range = t.arange(97)
first_operand = repeat(raw_range, "a -> (a a2)", a2=97).unsqueeze(1)
second_operand = repeat(raw_range, "a -> (a2 a)", a2=97).unsqueeze(1)
all_x = t.cat((first_operand, second_operand), dim=1)

answers = t.sum(all_x, dim=1) % 97

m = t.arange(97 * 97 * 97) % 97
m = m.reshape(())

# ax = seaborn.heatmap(all_x, annot=True, cmap="Blues")
# plt.show()