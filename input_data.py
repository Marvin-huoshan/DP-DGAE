'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        #将每一行去除前导和尾部的空白，返回副本
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    # load the data: x, tx, allx, graph
    #加载数据x，tx，allx，graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        #with open as f 读文件
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            #判断python版本
            if sys.version_info > (3, 0):
                #Pickle,可以将对象转换为一种可以传输或存储的格式
                #pkl.dump->序列化  pkl.load->反序列化
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    #list->tuple
    x, tx, allx, graph = tuple(objects)
    #存储每一行（测试集）
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    #排序(默认按行)
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        #citeseer中存在孤立节点，需要特殊处理
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        #一个shape为len(test)*feature的基于行连接的稀疏矩阵
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        #按行赋值
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    #vstack->按行堆叠数组
    #tolil()->将矩阵转换为List of Lists格式。
    features = sp.vstack((allx, tx)).tolil()
    #为了让特征向量和邻接矩阵的索引一致，把乱序的特征数据读取出来，按正确的ID顺序重新排列。
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
