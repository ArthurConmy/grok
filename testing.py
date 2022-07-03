import wandb
from random import randint as ri
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.show()