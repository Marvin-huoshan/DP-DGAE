import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
loss_history = []
Floss_history = []
Loss_history = []
class GAE_Loss(nn.Module):
    def __init__(self):
        super(GAE_Loss, self).__init__()
        return
    def forward(self, z, x, sample2, sample3, weight_tensor, num):

        Coefficient1 = torch.full((x.shape), np.log(2),device='cuda')
        #add noise to Coefficient1
        #Coefficient1 = Coefficient1 + sample1
        part1 = Coefficient1
        Coefficient2 = (1/2 - x)
        #print('Co2:',torch.mean(Coefficient2))
        Coefficient2 = Coefficient2 + sample2
        part2 = torch.mul(Coefficient2, z)
        #print('part2:',torch.mean(part2))
        Coefficient3 = torch.full((x.shape), 1/8,device='cuda')
        #print('Co3:',torch.mean(Coefficient3))
        Coefficient3 = Coefficient3 + sample3
        #print('Co3:', torch.mean(Coefficient3))
        part3 = torch.mul(Coefficient3,torch.pow(z,2))
        #print('part3:',torch.mean(part3))
        result = part1 + part2 + part3
        #result = torch.multiply(result,weight_tensor)
        #print('Coefficient1,mean:',torch.mean(Coefficient1))
        #print('Coefficient2,mean:',torch.mean(Coefficient2))
        #print('Coefficient3,mean:',torch.mean(Coefficient3))
        loss = torch.mean(result)
        loss_history.append(loss.item())
        print('loss:', loss)
        ABC = - torch.mean(torch.mul(x,torch.log(torch.sigmoid(z))) + torch.mul((1 - x),torch.log(1 - torch.sigmoid(z))))
        result2 = - (torch.mul(x,torch.log(torch.sigmoid(z))) + torch.mul((1 - x),torch.log(1 - torch.sigmoid(z))))
        Floss = F.binary_cross_entropy(torch.sigmoid(z), x, weight = weight_tensor)
        #Floss = F.binary_cross_entropy(torch.sigmoid(z), x)
        Floss_history.append(Floss.item())
        Loss = 0.9 * loss + 0.1 * Floss
        Loss_history.append(Loss.item())
        '''if num < 2000000:
            loss = (loss * 0.025 + Floss * 0.975 )
        else:
            loss = (loss * 0.975 + Floss * 0.025 )'''
        #loss = Floss
        #Loss_history.append(loss.item())
        print('Floss:', Floss)
        #loss = Floss
        #print('Floss:',Floss)
        #print('Floss:',Floss)
        return Loss
