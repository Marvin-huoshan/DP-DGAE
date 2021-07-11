'''
****************NOTE*****************
CREDITS : Thomas Kipf
since datasets are the same as those in kipf's implementation, 
Their preprocessing source was used as-is.
*************************************
'''
import math

import numpy as np
import scipy.sparse as sp
import torch

import args


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        #转化为稀疏化格式
        sparse_mx = sparse_mx.tocoo()
    #np.vstack->按行堆叠数组
    #transpose->行列互换（转置）
    #sparse_mx.row->所有行坐标；    sparse_mx.col->所有列坐标
    #相当于存储所有非零元素坐标
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    #存储所有非零元素
    values = sparse_mx.data
    #存储矩阵的shape
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    #转化为稀疏矩阵形式
    adj = sp.coo_matrix(adj)
    #增加自环
    adj_ = adj + sp.eye(adj.shape[0])
    #矩阵按行相加
    rowsum = np.array(adj_.sum(1))
    #D^-0.5(对角阵)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    #transpose->类似于转置
    #tocoo->Convert this matrix to COOrdinate format.
    #矩阵归一化(AB).T*B->B.T*A.T*B?
    #??
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    #去除对角线元素
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    #移除零元素
    adj.eliminate_zeros()
    # Check that diag is zero:
    #Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    assert np.diag(adj.todense()).sum() == 0

    #用稀疏格式返回矩阵上三角部分
    adj_triu = sp.triu(adj)
    #adj_tuple中存储邻接矩阵所有非零元素坐标、元素数值、矩阵shape
    adj_tuple = sparse_to_tuple(adj_triu)
    #将上三角所有非零元素坐标存储为edges(上三角中的边)[[0 1],[1,0]...,[n,m]]
    edges = adj_tuple[0]
    #矩阵中所有的边
    edges_all = sparse_to_tuple(adj)[0]
    #np.floor->向下取整
    #test
    num_test = int(np.floor(edges.shape[0] / 10.))
    #validation
    num_val = int(np.floor(edges.shape[0] / 20.))
    #从零开始编号
    all_edge_idx = list(range(edges.shape[0]))
    #洗牌
    np.random.shuffle(all_edge_idx)
    #开始划分validation和test index
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    #选出validation和test边
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    #np.hstack->水平方向平铺
    #删除test_edges & val_edges 剩余的为train_edges
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        #np.all按照axis方向做与运算
        #查看b中是否存在a
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    #当test_edges_false的长度小于test_edges
    while len(test_edges_false) < len(test_edges):
        #randint->Return random integers from `low` (inclusive) to `high` (exclusive).
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        #如果idx_i = idx_j跳过
        if idx_i == idx_j:
            continue
        #[idx_i,idx_j]是否存在于edges_all
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    #未测试val_edges_false是否与edge_all重合?
    #!!
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    #测试集合之间是否重合
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)


    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    #a[row_ind[k], col_ind[k]] = data[k]
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def GPU_data_normal_2d(orign_data):
    '''
    normalization
    :param orign_data:
    :return:
    '''
    orign_data = orign_data.reshape(args.num,-1)
    dim = 0
    d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    d_one = torch.ones(dst.shape).cuda()
    dst = torch.where(dst != 0,dst,d_one)
    d_min = d_min.expand(orign_data.shape[0],orign_data.shape[1])
    dst = dst.expand(orign_data.shape[0],orign_data.shape[1])
    sq = math.sqrt(orign_data.shape[0])
    normal_data = (orign_data - d_min) / (dst * sq)
    return normal_data

def CPU_data_normal_2d(orign_data):
    '''
    normalization
    :param orign_data:
    :return:
    '''
    orign_data = orign_data.reshape(args.num,-1)
    dim = 0
    d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    d_one = torch.ones(dst.shape)
    dst = torch.where(dst != 0,dst,d_one)
    d_min = d_min.expand(orign_data.shape[0],orign_data.shape[1])
    dst = dst.expand(orign_data.shape[0],orign_data.shape[1])
    sq = math.sqrt(orign_data.shape[0])
    normal_data = (orign_data - d_min) / (dst * sq)
    return normal_data