import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

import pre_args

class GraphConvSparse(nn.Module):
    # **kwargs是将一个可变的关键字参数的字典传给函数实参
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        # weight矩阵通过限定input_dim与out_dim的维度，可以控制weight的大小，从而实现维度变化
        # weight->[input_dim,output_dim]
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        # feature * weight
        x = torch.mm(x, self.weight)
        # A * feature * weight
        x = torch.mm(self.adj, x)
        # outputs = relu(A * feature * weight)
        outputs = self.activation(x)
        return outputs

    def dforward(self, inputs):
        x = inputs
        # feature * weight.T
        x = torch.mm(x, self.weight.T)
        # A * feature * weight.T
        x = torch.mm(self.adj, x)
        # outputs = relu(A * feature * weight)
        outputs = self.activation(x)
        return outputs

def glorot_init(input_dim, output_dim):
    # init_range = (6/((输入纬度)+(输出纬度)))^(1/2)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    # torch.rand(input_dim,output_dim) -> 从[0,1)正态分布抽样，初始化一个(input_dim * output_dim)的矩阵
    # 为何inital要如此处理？
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    # nn.Parameter将一个张量注册为可训练参数
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        # base_gcn = GraphConvSparse(1433,32,adj)
        self.base_gcn = GraphConvSparse(pre_args.input_dim, pre_args.hidden1_dim, adj, activation=torch.sigmoid)
        # gcn_mean = GraphConvSparse(32,16,adj)
        self.gcn_mean = GraphConvSparse(pre_args.hidden1_dim, pre_args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_out = GraphConvSparse(pre_args.input_dim, pre_args.num, adj, activation=torch.sigmoid)

    # 使用GCN作为解码器
    # self.gcn_meanT = GraphConvSparse(args.hidden2_dim, args.hidden1_dim, adj, activation=torch.sigmoid)
    # self.base_gcnT = GraphConvSparse(args.hidden1_dim)

    def encode(self, X):
        # hidden = base_gcn.forward(X)
        # hidden = relu(A * feature * weight<正态>)
        # hidden维度为：n x D
        hidden = self.base_gcn(X)
        # z的维度为n x hidden2_dim
        #z = self.mean = self.gcn_mean(hidden)
        return hidden

    def decode(self, X):
        '''A_P = self.gcn_mean.dforward(X)
        A_P = self.base_gcn.dforward(A_P)
        A_P = self.gcn_out(A_P)'''
        A_P = self.base_gcn.dforward(X)
        return A_P

    # 前向传播
    def forward(self, X):
        # 通过encode获得隐变量Z
        Z = self.encode(X)
        # 使用Z*Z.T乘积解码
        # A_pred 为一个n x n矩阵
        # A_pred = dot_product_decode(Z)
        A_pred = self.decode(Z)
        # 返回预测矩阵
        return A_pred

# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out