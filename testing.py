import torch as t
import wandb
from random import randint as ri
import matplotlib.pyplot as plt
import seaborn as sns

class NetMatrix(t.nn.Module):
  def __init__(self, size):
    super().__init__()
    self.mat = t.nn.Parameter(t.randn(size=size))

  def forward(self, x):
    return x @ self.mat

class Net(t.nn.Module):
  def __init__(self):
    super().__init__()
    self.m1 = NetMatrix(size=(5, 10))
    self.relu = t.nn.ReLU()
    self.m2 = NetMatrix(size=(10, 3))

  def forward(self, x):
    return self.m2(self.relu(self.m1(x)))

net = Net()
x = t.randn(2, 5)

def print_values(self, inp, outp):
  print("input", inp)
  print("output", outp)

print(x)
net.m1.register_forward_hook(print_values)
print(net(x))

def wandb_thing():
  wandb.init(project="my-test-project-1")
  for _ in range(10):
      wandb.log({"loss": ri(1,10)})

cf_matrix = [[0.0 for _ in range(97)] for _ in range(97)]

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
# plt.show()