import torch 
import torch.nn as nn


def calcLoss(y_pred, y_true):
    pass

class YoloLoss(nn.Module):
    def __init__(self):
        pass
    def make(self):
        super(YoloLoss, self).__init__()
        out = lambda : calcLoss()
        return out
 
        
if __name__ == '__main__':    
    lossMaker = YoloLoss()
    loss = loss1.make()
    out1()