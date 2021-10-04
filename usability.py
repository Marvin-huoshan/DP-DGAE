import numpy as np
import torch
import numpy
import networkx as nx
from to_graph import load_data
import args

file = 'pre_matrix/A_p_MTL_test.pth'
if file != 'None':
    weight = torch.load(file)
print(weight)
print(torch.max(weight))
#weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
weight = (weight > 0.5).long()
G1 = load_data(args.dataset)
b = nx.to_numpy_array(G1)
print('original matrix:')
print(b)
a = weight.detach().numpy()

'''for j in range(2708):
    for i in range(j, 2708):
        a[i][j] = 0
for i in range(2708):
    for j in range(i, 2708):
        a[j][i] = a[i][j]'''
row, col = np.diag_indices_from(a)
a[row,col] = 0
for i in range(2708):
    for j in range(2708):
        if a[i][j] == 1:
            a[j][i] = 1
print((a == 1).sum())
print(a)
G2 = nx.from_numpy_matrix(a)
print('mod graph edges:')
print(nx.number_of_edges(G2))
num = 0

for i in range(2708):
    for j in range(2708):
        if (a[i][j] == 1 and b[i][j] == 1):
            num += 1
print('sim edges:')
print(num / 2)
print('sim probability:')
print((num / 2) / nx.number_of_edges(G1))

dict0 = dict(sorted(nx.degree(G1), key=lambda x: x[1], reverse=True))
dict1 = dict(sorted(nx.degree(G2), key=lambda x: x[1], reverse=True))

print(dict0)
print(dict1)