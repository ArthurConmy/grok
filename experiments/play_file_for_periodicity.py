import numpy as np
from tqdm import tqdm
import torch as t
import torch.nn.functional as F
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE, VOCAB_SIZE, get_metrics, get_percent_and_loss
from dataset.data import get_the_data, ArithmeticTokenizer
from train import MINI_BATCH_SIZE, DEVICE
import seaborn
import matplotlib.pyplot as plt
from einops import repeat, rearrange
OFFSET = 22
SAVED_MODELS = [0, 30, 60, 90, 120, 150, 400, 700, 850, 880, 910, 940, 970, 999]
MODEL_FILES = [f"checkpoints/seed1at{saved_model}.pt" for saved_model in SAVED_MODELS]
MODEL_FILE = MODEL_FILES[-1]
ts = []

def save_attention_pattern(self, inp, outp):
    self.saved_attentions.append(outp)

def get_all_x():
    raw_range = t.arange(97)
    first_operand = repeat(raw_range, "a -> (a a2)", a2=97).unsqueeze(1)
    second_operand = repeat(raw_range, "a -> (a2 a)", a2=97).unsqueeze(1)
    all_x = t.cat((first_operand, second_operand), dim=1)
    return all_x

def get_prob_matrix(model):
    all_x = get_all_x()
    answers = F.one_hot((t.sum(all_x, dim=1) % 97), num_classes=VOCAB_SIZE)

    with t.no_grad():
        logits = model(all_x).logits.detach().clone()

    probs = F.softmax(logits, dim=1)
    maxes = F.one_hot(t.argmax(logits, dim=1), num_classes=VOCAB_SIZE)

    correct_probs = t.sum(probs * answers, dim=1) 
    prob_matrix = rearrange(correct_probs, "(a b) -> a b", a=97, b=97)

    correct_maxes = t.sum(maxes * answers, dim=1)
    correct_matrix = rearrange(correct_maxes, "(a b) -> a b", a=97, b=97)

    return prob_matrix

def get_att_matrix(model, block, head):
    all_x = get_all_x()

    with t.no_grad():
        logits = model(all_x).logits.detach().clone()
    
    att = model.blocks[block].attention.last_att[:, head, 1, 0] # attention from place 2 applied to place 1    
    att_matrix = rearrange(att, "(a b) -> a b", a=97, b=97)
    return att_matrix

for block in range(0, 2):
    for head in range(0, 32):
        # print(f"Model {model_file}")
        my_model_config = dict(DEFAULT_MODEL_CONFIG)
        my_model_config["num_heads"] = 32
        model = get_transformer(**my_model_config)
        model.load_state_dict(t.load(MODEL_FILE, map_location=t.device("cpu")))

        prob_matrix = get_prob_matrix(model)
        att_matrix = get_att_matrix(model, block=block, head=head)
        
        # is_tdata = t.zeros(97 * 97).int()
        # write_data = t.ones_like(indices).int()
        # is_tdata.scatter_(0, indices, write_data)
        # t_data_matrix = rearrange(is_tdata, "(a b) -> a b", a=97, b=97)

        # A = ArithmeticTokenizer()
        # for i in range(VOCAB_SIZE):
        #     print(i)
        #     tens = t.range(start=0, end=118).float()[i:i+1]
        #     print(A.decode(tens))
        
        ax = seaborn.heatmap(att_matrix, annot=False, cmap="Blues")
        ax.set_title(f"Block {block} and head {head}")
        plt.show()
        ax.clear()
        
        # ax = seaborn.heatmap(correct_matrix, annot=False, cmap="Blues")
        # ax = seaborn.heatmap(t_data_matrix, annot=False, cmap="Blues")
        # plt.show()

        # ax = seaborn.heatmap(t_data_matrix, annot=False, cmap="Blues")
        # plt.show()