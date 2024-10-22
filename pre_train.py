import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
from torch.optim import lr_scheduler
from input_data import load_data
from preprocessing import *
import pre_args
import pre_model
import AP_loss


# Train on CPU (hide GPU) due to memory constraints
#由于内存限制，使用CPU进行训练
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

#获得数据集的邻接矩阵与特征矩阵
adj, features = load_data(pre_args.dataset)

# Store original adjacency matrix (without diagonal entries) for later
#存储原始的邻接矩阵（排除对角线）
adj_orig = adj
#sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)->取出邻接矩阵主对角线的元素
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#移除零元素
adj_orig.eliminate_zeros()

#将数据集进行划分，test、validation、train、False
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
#增加自环与归一化：D^-0.5 * (A+I) * D^-0.5
#adj_norm中存储了处理后邻接矩阵的非零值坐标、非零值、shape
adj_norm = preprocess_graph(adj)

#点的数量
num_nodes = adj.shape[0]
'''
    sparse_to_tuple->获得矩阵中非零值的坐标、非零值、shape
'''
features = sparse_to_tuple(features.tocoo())
#特征的数量
num_features = features[2][1]
#特征非零值的数量
features_nonzero = features[1].shape[0]

# Create Model
#pos_weight = [n * n - num(edges)] / num(edges)
#??
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#norm = (n * n) / [(n * n - num(edges)) * 2]
#??
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

#sp.eye->生成n * n全为1的对角阵
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)


#adj_norm -> 行值[a,b,c,d,e]
#            列值[f,g,h,i,j]
#稀疏张量被表示为一对致密张量：一维张量和二维张量的索引。可以通过提供这两个张量来构造稀疏张量，以及稀疏张量的大小
#(a,f)->adj_norm[1][0];(b,g)->adj_norm[1][1]
#分别代表位置与对应的取值
#将adj_norm、adj_label、features转化为稀疏张量
'''
    adj_norm -> D^-0.5 * (A + I) * D^-0.5 (3种序列)
    adj_label -> (A + I) (3种序列)
    feature -> feature (3种序列) <按特征矩阵排列>
'''
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                            torch.FloatTensor(adj_norm[1]),
                            torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                            torch.FloatTensor(adj_label[1]),
                            torch.Size(adj_label[2]))
#features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                            #torch.FloatTensor(features[1]),
                            #torch.Size(features[2]))
#to_dense()->转化为稠密矩阵形式
#.view()将张量转化为一维向量形式
#=1的位置为True，做一张Mask tensor
#adj_label -> adj_train + I
#weight_mask中为1处为adj_label为1处
#相当于adj_label为1处为其加weight,其余地方为1
weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

# init model and optimizer
#getattr() 函数用于返回一个对象属性值。
#model.GAE(adj_norm)
print(torch.cuda.is_available())
adj_norm = adj_norm.cuda()
model = getattr(pre_model,pre_args.model)(adj_norm)
model = model.cuda()

#使用Adam优化器，参数为model.parammeters(),learning_rate
optimizer = Adam(model.parameters(), lr=pre_args.learning_rate)
scheduler1 = lr_scheduler.StepLR(optimizer,step_size=50000,gamma=0.9)

#validation_edges and validation_edges_false vs A_pred
def get_scores(edges_pos, edges_neg, adj_rec):
    #正向边、负向边、预测矩阵
    #平滑函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    #edges_pos中保存边的坐标[m,n]
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        #在tensor中使用item()函数取出坐标对应的数值，不实用则会取出一个单位的tensor
        #取出在预测矩阵中，正向边对应的值
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        #adj_orig->去除邻接矩阵对角线以及零元素
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    # 随机生成的假边（不与原来的边重合）
    for e in edges_neg:
        #预测矩阵中假边对应的预测值
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        #adj_orig->去除邻接矩阵对角线以及零元素
        neg.append(adj_orig[e[0], e[1]])
    #横向堆叠preds and preds_neg
    preds_all = np.hstack([preds, preds_neg])
    #pred->1; pred_neg->0
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    #labels_all [正边数量的1+反边数量的0]
    #roc_auc_score(y_true,y_score)
    #y_true与y_score对应位置重合所占的比例
    #labels_all正向边数量的的1+负向边数量的0
    #preds_all正向边对应的预测值+负向变对应的预测值
    roc_score = roc_auc_score(labels_all, preds_all)
    #被分为正实际为正比例
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_acc(adj_rec, feature_label):
    #adj_rec->预测矩阵；adj_label-> (A + I)
    #变为1维
    #为何用long
    #labels_all = feature_label.to_dense().view(-1).long()
    #feature_label = torch_2(feature_label)
    #adj_rec = torch_2(adj_rec)
    #labels_all = feature_label.view(-1).long()
    labels_all = (feature_label > 0.7).view(-1).long()
    #labels_all = feature_label.view(-1).long()
    #预测矩阵大于0.5的边置为1，小于0.5为0
    preds_all = (adj_rec > 0.7).view(-1).long()
    #preds_all = adj_rec.view(-1).long()
    #对应为值相等的数量之和/所有的边数
    #只是有矩阵，并未成图？
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

def torch_2(features):
    '''
    feature Retain two decimal places
    :param features:
    :return:
    '''
    features = features * 10
    frac = torch.frac(features)
    frac_round = torch.round(frac)
    features = torch.trunc(features)
    features = features + frac_round
    features = features / 10
    return features

with open('pre_weight/hidden_1_noise.pk','rb') as file_to_read:
    features = pickle.load(file_to_read)

features = features.clone().detach()

#Laplace noise
noise = torch.distributions.Laplace(
    torch.tensor([0.0]),
    torch.tensor([pre_args.delta_1/pre_args.epsilon]),
)
noise3 = torch.distributions.Laplace(
    torch.tensor([1.0]),
    torch.tensor([pre_args.delta_1/pre_args.epsilon]),
)
#features_tensor = features.to_dense().view(-1)
features_tensor = features.view(-1)
'''sample1 = noise.sample(sample_shape=features_tensor.shape)
sample1 = torch.reshape(sample1,features_tensor.shape)'''
sample1 = torch.full(features_tensor.shape,-1.0)
while torch.mean(sample1) < 0:
    sample1 = noise.sample(sample_shape=features_tensor.shape)
    sample1 = torch.reshape(sample1,features_tensor.shape)
    print(torch.mean(sample1))
#noise2
'''sample2 = noise.sample(sample_shape=features_tensor.shape)
sample2 = torch.reshape(sample2,features_tensor.shape)'''
sample2 = torch.full(features_tensor.shape,-1.0)
while torch.mean(sample2) < 0:
    sample2 = noise.sample(sample_shape=features_tensor.shape)
    sample2 = torch.reshape(sample2,features_tensor.shape)
    print(torch.mean(sample2))
#noise3
'''sample3 = noise.sample(sample_shape=features_tensor.shape)
sample3 = torch.reshape(sample3,features_tensor.shape)'''
sample3 = torch.full(features_tensor.shape,-1.0)
while torch.mean(sample3) < 0:
    sample3 = noise.sample(sample_shape=features_tensor.shape)
    sample3 = torch.reshape(sample3,features_tensor.shape)
    print(torch.mean(sample3))
print(torch.mean(sample1))
print(torch.mean(sample2))
print(torch.mean(sample3))
print('sample3:',torch.max(sample3),torch.min(sample3))
sample1 = sample1.cuda()
sample2 = sample2.cuda()
sample3 = sample3.cuda()
Loss = AP_loss.GAE_Loss()
Loss = Loss.cuda()
#print(sample1)
# train model
#开始训练
#features = data_normal_2d(features.to_dense())
features = features.cpu()
features = data_normal_2d(features)
#features = features.to_dense()
features = features.cuda()
acc_history = []
for epoch in range(pre_args.num_epoch):
    t = time.time()
    #feature -> 稀疏张量，张量形式的特征矩阵
    #A_pred = model.forward(features)
    A_pred = model(features)
    #print(A_pred)
    #print(features)
    #print(A_pred)
    #print(features)
    #梯度清零
    optimizer.zero_grad()
    #norm = (n * n) / [(n * n - num(edges)) * 2]
    #使用交叉熵->F.binary_cross_entropy([预测值的预测一维表示]，[A+I的一维表示])
    #在原本有边的地方，设置更高的权重（>1），原本无边的地方设置权重为1,更加注重对于原始边的学习
    #loss = log_lik = F.binary_cross_entropy(A_pred.view(-1), features.to_dense().view(-1))
    loss = log_lik = Loss(A_pred.view(-1),features.view(-1),sample1,sample2,sample3)
    #loss = log_lik = F.binary_cross_entropy(A_pred.view(-1), features.view(-1))
    if pre_args.model == 'VGAE':
        #kl_divergence = 1/2n * (1 + 2*logstd - mean^2 - [e^logstd]^2)
        #logstd->[n x 16]
        #mean->[n x 16]
        #sum(1)->按行相加
        #[n x 16]-> [n x 1] -> mean
        #kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
        kl_divergence = 1 / A_pred.size(0) * (1 + model.logstd - torch.abs(model.mean) - torch.exp(model.logstd)).sum(1).mean()
        loss -= kl_divergence
    #误差反向传播
    #print(model.base_gcn.weight)
    loss.backward()
    #梯度下降
    optimizer.step()
    print('epoch %d learning rate: %f' %(epoch,optimizer.param_groups[0]['lr']))
    scheduler1.step()
    #计算ACC
    train_acc = get_acc(A_pred,features)
    acc_history.append(train_acc.item())
    #validation_edges and validation_edges_false vs A_pred
    #在训练过程中使用validation; validation的主要作用是来验证是否过拟合、以及用来调节训练参数等
    #边训练边看到训练的结果，及时判断学习状态
    #val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    val_roc, val_ap = 1,1
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))
#torch.save(obj=model.base_gcn.weight, f = 'pre_weight/base_gcn_noise.pth')
torch.save(obj=model.gcn_mean.weight, f = 'pre_weight/gcn_mean.pth')
with open('pre_weight/hidden_2.pk','wb') as file_to_write:
    pickle.dump(model.hidden,file_to_write)

#test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
#print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      #"test_ap=", "{:.5f}".format(test_ap))
torch.cuda.empty_cache()

def plot_loss_with_acc(loss_history,Floss_history,Loss_history,acc_history):
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121)
    plot1 = plt.plot(range(len(loss_history)),loss_history,c = 'b',label = 'AP_loss')
    plot2 = plt.plot(range(len(Floss_history)),Floss_history,c = 'r',label = 'F_loss')
    plot3 = plt.plot(range(len(Loss_history)),Loss_history,c = 'g',label = 'Loss')
    ax1.legend(fontsize = 'large', loc = 'lower left')
    ax1.set_title('different loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2 = plt.subplot(122)
    plot4 = plt.plot(range(len(acc_history)),acc_history,c = 'y',label = 'ACC')
    ax2.set_title('ACC')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('ACC')
    ax2.legend(fontsize = 'large', loc = 'upper right')
    plt.savefig('Loss_&_ACC_H2_1_0_with.png')


plot_loss_with_acc(AP_loss.loss_history,AP_loss.Floss_history,AP_loss.Loss_history,acc_history)
'''print(model.Z)
Z = model.Z
ZZT = torch.matmul(Z, Z.t())
print(A_pred)
print(model.logstd)
print(model.mean)
import pandas as pd
signumpy = ZZT.detach().numpy()
data_df = pd.DataFrame(signumpy)
writer = pd.ExcelWriter('save_Excel_VGAE_laplace_origin.xlsx')
data_df.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
writer.save()'''
