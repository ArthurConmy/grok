import torch as t
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE

transformer = get_transformer(**DEFAULT_MODEL_CONFIG)
transformer.load_state_dict(t.load("first_gre.pt", map_location=t.device("cpu")))
print("Done")