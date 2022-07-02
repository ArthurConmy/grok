import torch as t
import torch.nn.functional as F
VOCAB_SIZE = 119


def get_percent_correct(model, x, y):
    with t.no_grad():
        logits = model(x).logits
        probabilities = F.softmax(logits, dim=1)
    
        y_one_hot = F.one_hot(y, num_classes=VOCAB_SIZE).float()
        corrects = t.sum((t.argmax(probabilities, dim=1) == y).float())

        assert probabilities.shape[0] == y.shape[0]
        return (100 * corrects) / probabilities.shape[0]