import torch as t
import torch.nn.functional as F
VOCAB_SIZE = 119

def list_prod(L):
    ans = 1
    for l in L:
        ans *= l
    return ans

def get_no_parameters(model):
    nop = 0
    def list_prod(L):
        ans = 1
        for l in L:
            ans *= l
        return ans
    for p in model.parameters():
        nop += list_prod(p.shape)        
    print("NUMBER OF PARAMETERS", nop)  


def get_validation_data(model, x, y):
    with t.no_grad():
        logits = model(x).logits
        probabilities = F.softmax(logits, dim=1)
    
        y_one_hot = F.one_hot(y, num_classes=VOCAB_SIZE).float()
        corrects = t.sum((t.argmax(probabilities, dim=1) == y).float())

        cross_entropy_loss = t.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(probabilities, y_one_hot)

        assert probabilities.shape[0] == y.shape[0]
        return (100 * corrects) / probabilities.shape[0], loss