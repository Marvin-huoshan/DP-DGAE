import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd

def load_data(file1,file2):
    # dataset file
    #edge_file = 'facebook/107.edges'
    edge_file = file1
    #feature_file = 'facebook/107.feat'
    feature_file = file2
    G = nx.read_edgelist(edge_file)
    list_node = nx.nodes(G)
    print('number of nodes:',len(list_node))
    list_node = sorted(list_node, key=lambda x: int(x))
    # matrix
    a = nx.to_numpy_array(G, list_node)
    #feature = np.loadtxt(feature_file, dtype=str)
    feature_file = 'data/IMDB/IMDB_feature_onehot.txt'
    feature = file2array(feature_file)
    print(feature)
    feature_arg = np.argsort(feature[:, 0])
    feature = feature[feature_arg]
    list_node_int = [int(i) for i in list_node]
    feature_item = list(feature[:, 0])
    differ = list(set(feature_item).difference(list_node_int))
    # feature
    for i in differ:
        row = np.where(feature == i)[0]
        feature = np.delete(feature, row, 0)
    feature = np.delete(feature, 0, 1)
    coo_A = coo_matrix(a)
    coo_feature = coo_matrix(feature)
    return coo_A, coo_feature
#facebook-delimeter:' '
def file2array(path, delimeter='\t'):
    #delimiter = ' '
    fp = open(path, 'r', encoding='utf-8')
    string = fp.read()
    fp.close()
    row_list = string.splitlines()
    data_list = [[float(i) for i in row.strip().split(delimeter)] for row in row_list]
    return np.array(data_list)

def preprocess_txt_edge(file):
    df = pd.read_csv(file,'\t')
    print(df.info)
    df_new = df[['from','to']]
    df_new.to_csv('data/IMDB/IMDB_edge_unweight.txt', header=True, index=None, sep='\t')

def preprocess_txt_feature(file):
    df = pd.read_csv(file,'\t')
    print(df.info)
    df_new = df[['id','movies_95_04','main_genre']]
    df_new.to_csv('data/IMDB/IMDB_feature_un.txt', header=True, index=None, sep='\t')

def feature_onehot(file):
    df = pd.read_csv(file,'\t')
    print(df.info)
    df['movies_95_04'] = df['movies_95_04'].apply(lambda x:int(x/10))
    dummies = pd.get_dummies(df['main_genre'])
    df_new = df[['id','movies_95_04']]
    df_new = pd.concat([df_new,dummies], axis=1)
    df_new.to_csv('data/IMDB/IMDB_feature_onehot.txt',header=False, index=None, sep='\t')
