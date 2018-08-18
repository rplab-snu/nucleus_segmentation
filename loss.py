import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class TverskyLoss:
    def __init__(self, alpha, torch_device):
        # super(TverskyLoss, self).__init__()
        self.a = alpha
        self.b = 1 - alpha 
        self.smooth = torch.tensor(1.0, device=torch_device)

    def __call__(self, predict, target_):
        predict = F.sigmoid(predict)
        target_f  = target_.view(-1) # g
        predict_f = predict.view(-1) # p

        # PG + a * P_G + b * G_P        
        PG  = (predict_f * target_f).sum() # p0g0
        P_G = (predict_f * (1 - target_f)).sum() * self.a # p0g1
        G_P = ((1 - predict_f) * target_f).sum() * self.b # p1g0

        loss = PG / (PG + P_G + G_P + self.smooth)
        return loss * -1

# https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    pred = F.sigmoid(pred)
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

if __name__ == "__main__":

    def get_grad(*args):
        print("Grad : \n", args)
        

    target = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]]], dtype=torch.float, requires_grad=True)
    predicted = torch.tensor([[[1,1,0],[0,0,0],[1,0,0]]], dtype=torch.float, requires_grad=True)
    print("Prediction : \n", predicted); print("GroudTruth : \n", target)
    predicted.register_hook(get_grad)

    loss = TverskyLoss(0.3, torch.device("cpu"))
    l = loss(predicted, target)
    print("Loss : ", l)
    l.backward()
