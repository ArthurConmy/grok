import torch as t
import torch.nn.functional as F
from model.transformer import Transformer, get_transformer
from train import DEFAULT_MODEL_CONFIG, DEVICE, VOCAB_SIZE
from dataset.data import get_the_data, get_metrics, ArithmeticTokenizer
from train import MINI_BATCH_SIZE, DEVICE
from utils import get_percent_and_loss
import seaborn
import matplotlib.pyplot as plt
from einops import repeat, rearrange
OFFSET = 22

my_model_config = dict(DEFAULT_MODEL_CONFIG)
my_model_config["num_heads"] = 128
model = get_transformer(**my_model_config)
model.load_state_dict(t.load("checkpoints/trainer.pt", map_location=t.device("cpu")))

raw_range = t.arange(97)
first_operand = repeat(raw_range, "a -> (a a2)", a2=97).unsqueeze(1)
second_operand = repeat(raw_range, "a -> (a2 a)", a2=97).unsqueeze(1)
all_x = t.cat((first_operand, second_operand), dim=1)
answers = F.one_hot((t.sum(all_x, dim=1) % 97) + 22)

logits = model(all_x).logits.detach().clone()
probs = F.softmax(logits, dim=1)
maxes = F.one_hot(t.argmax(logits, dim=1))

correct_probs = t.sum(probs * answers, dim=1) 
prob_matrix = rearrange(correct_probs, "(a b) -> a b", a=97, b=97)

correct_maxes = t.sum(maxes * answers, dim=1)
correct_matrix = rearrange(correct_maxes, "(a b) -> a b", a=97, b=97)

train_data, valid_data = get_the_data(
    operator = "+",
    train_proportion = 0.5,
    mini_batch_size = -1,
    device = DEVICE,
)
for x, y in train_data:
    indices = (x[:,1]-22) + 97 * (x[:,0]-22)
is_tdata = t.zeros(97 * 97).int()
write_data = t.ones_like(indices).int()
is_tdata.scatter_(0, indices, write_data)
t_data_matrix = rearrange(is_tdata, "(a b) -> a b", a=97, b=97)

# A = ArithmeticTokenizer()
# for i in range(VOCAB_SIZE):
#     print(i)
#     tens = t.range(start=0, end=118).float()[i:i+1]
#     print(A.decode(tens))

m = t.arange(97 * 97 * 97) % 97
m = rearrange(m, "(a b c) -> (a b) c", a=97, b=97, c=97)

# ax = seaborn.heatmap(prob_matrix, annot=False, cmap="Blues")
# plt.show()

ax = seaborn.heatmap(t_data_matrix, annot=False, cmap="Blues")
plt.show()