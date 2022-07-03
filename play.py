import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE
from dataset.data import get_the_data, get_metrics
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss

my_model_config = dict(DEFAULT_MODEL_CONFIG)
my_model_config["num_heads"] = 128
transformer = get_transformer(**my_model_config)
transformer.load_state_dict(t.load("checkpoints/spice.pt"))