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
    def forward(self, z, x, sample2, Coefficient3, weight_tensor):

        Coefficient1 = torch.full((x.shape), np.log(2),device='cuda')
        #add noise to Coefficient1
        #Coefficient1 = Coefficient1 + sample1
        part1 = Coefficient1
        Coefficient2 = (1/2 - x)
        #print('Co2:',torch.mean(Coefficient2))
        Coefficient2 = Coefficient2 + sample2
        #print('original Co2:',Coefficient2)
        #Coefficient2 = torch.sigmoid(Coefficient2)
        #Coefficient2 = Coefficient2 - 0.5
        #print('Co2:',Coefficient2)
        part2 = torch.mul(Coefficient2, z)
        #print('part2:',torch.mean(part2))
        '''Coefficient3 = torch.full((x.shape), 1/8,device='cuda')
        #print('Co3:',torch.mean(Coefficient3))
        Coefficient3 = Coefficient3 + sample2
        Coefficient3 = torch.reshape(Coefficient3,(1034,1034))
        (evals,evecs) = torch.eig(Coefficient3,eigenvectors=True)
        tmp = torch.zeros_like(evals)
        evals = torch.where(Coefficient3 < 0, tmp, Coefficient3)
        Coefficient3 = Coefficient3.view(-1)'''
        '''tmp = torch.zeros_like(x)
        Coefficient3 = torch.where(Coefficient3 < 0, tmp, Coefficient3)
        print('Co3:')
        print(torch.min(Coefficient3))'''
        #print('Co3:', torch.mean(Coefficient3))
        part3 = torch.mul(Coefficient3,torch.pow(z,2))
        #print('part3:',torch.mean(part3))
        result = part1 + part2 + part3
        result = result
        '''w1 = 1 / (2 * (sigma1.pow(2)))
        w2 = 1 / (2 * (sigma2.pow(2)))
        print('sg1:',sigma1)
        print('sg2:',sigma2)
        print('W1:', w1)
        print('W2:', w2)'''
        #result = torch.multiply(result,weight_tensor)
        #print('part1,mean:', torch.mean(part1))
        #print('part2,mean:', torch.mean(part2))
        #print('part3,mean:', torch.mean(part3))
        #print('Coefficient1,mean:',torch.mean(Coefficient1))
        #print('Coefficient2,mean:',torch.mean(Coefficient2))
        #print('Coefficient3,mean:',torch.mean(Coefficient3))
        loss = torch.mean(result)
        #loss = torch.abs(loss)
        loss_history.append(loss.item())
        print('loss:', loss)
        ABC = - torch.mean(torch.mul(x,torch.log(torch.sigmoid(z))) + torch.mul((1 - x),torch.log(1 - torch.sigmoid(z))))
        result2 = - (torch.mul(x,torch.log(torch.sigmoid(z))) + torch.mul((1 - x),torch.log(1 - torch.sigmoid(z))))
        #Floss = F.binary_cross_entropy(torch.sigmoid(z), x, weight = weight_tensor)
        Floss = F.binary_cross_entropy(z, x, weight=weight_tensor)
        #Floss = F.binary_cross_entropy(torch.sigmoid(z), x)
        Floss_history.append(Floss.item())
        #Loss = 0.9 * loss + 0.1 * Floss
        #Loss = w1 * loss + w2 * Floss + torch.log(sigma1 * sigma2)
        '''if num < 1000000:
            Loss = (loss * 0.8 + Floss * 0.2 )
        else:
            Loss = (loss * 0.2 + Floss * 0.8 )'''
        #Loss = Floss
        #Loss_history.append(Loss.item())
        #loss = Floss
        #Loss_history.append(loss.item())
        print('Floss:', Floss)
        #loss = Floss
        #print('Floss:',Floss)
        #print('Floss:',Floss)
        return loss,Floss
