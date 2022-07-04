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

my_model_config = dict(DEFAULT_MODEL_CONFIG)
my_model_config["num_heads"] = 128
model = get_transformer(**my_model_config)
model.load_state_dict(t.load("checkpoints/new_trained.pt", map_location=t.device("cpu")))

raw_range = t.arange(97)
first_operand = repeat(raw_range, "a -> (a a2)", a2=97).unsqueeze(1)
second_operand = repeat(raw_range, "a -> (a2 a)", a2=97).unsqueeze(1)
all_x = t.cat((first_operand, second_operand), dim=1)
answers = F.one_hot((t.sum(all_x, dim=1) % 97), num_classes=VOCAB_SIZE)

logits = model(all_x).logits.detach().clone()
probs = F.softmax(logits, dim=1)
maxes = F.one_hot(t.argmax(logits, dim=1))

correct_probs = t.sum(probs * answers, dim=1) 
prob_matrix = rearrange(correct_probs, "(a b) -> a b", a=97, b=97)

# a1 = maxes * answers
# print("a1 shape", a1.shape)
# input()

correct_maxes = t.sum(maxes * answers, dim=1)
correct_matrix = rearrange(correct_maxes, "(a b) -> a b", a=97, b=97)

train_prop, train_loss, valid_prop, valid_loss = get_metrics(model=model, operator="+", train_proportion=0.5, device=DEVICE)
train_data, valid_data = get_the_data(
    operator = "+",
    train_proportion = 0.5,
    mini_batch_size = -1,
    device = DEVICE,
)
for x, y in train_data:
    indices = 97 * (x[:,0]) + x[:,1]
    print(x)
    print(y)
    print("Model")
    print(t.argmax(model(x).logits, dim=1))
is_tdata = t.zeros(97 * 97).int()
write_data = t.ones_like(indices).int()
is_tdata.scatter_(0, indices, write_data)
t_data_matrix = rearrange(is_tdata, "(a b) -> a b", a=97, b=97)
print(all_x)

# A = ArithmeticTokenizer()
# for i in range(VOCAB_SIZE):
#     print(i)
#     tens = t.range(start=0, end=118).float()[i:i+1]
#     print(A.decode(tens))

m = t.arange(97 * 97 * 97) % 97
m = rearrange(m, "(a b c) -> (a b) c", a=97, b=97, c=97)

# ax = seaborn.heatmap(prob_matrix, annot=False, cmap="Blues")
# plt.show()

ax = seaborn.heatmap(correct_matrix, annot=False, cmap="Blues")
plt.show()

# tensor([[66, 47],
#         [96, 82],
#         [41, 92],
#         ...,
#         [55, 82],
#         [46, 95],
#         [88, 42]])
# tensor([16, 81, 36,  ..., 40, 44, 33])