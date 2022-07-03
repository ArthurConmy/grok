import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE

transformer = get_transformer(**DEFAULT_MODEL_CONFIG)
transformer.load_state_dict(t.load("my_random_model_4.pt"))