# Copy this key and paste it into your command line when asked to authorize your account
# b6f207a48ff1f20ad239d2de3257ada565801689
# At the top of your training script, start a new run
# Copy
import wandb

wandb.init(project="my-test-project-1")
# Capture a dictionary of hyperparameters with config

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}
# Log metrics inside your training loop to visualize model performance

from random import randint as ri

for _ in range(10):
    wandb.log({"loss": ri(1,10)})

# Optional
# wandb.watch(model)