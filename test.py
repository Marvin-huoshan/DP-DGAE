import numpy as np
import scipy.sparse as sp
import torch
b = np.arange(12).reshape((6,2))
b = sp.coo_matrix(b).todense()
b = torch.tensor(b)
print('b:',b)
print('b**2:',(b**2).sum(1))
print(b.row)
print(b.col)
coords = np.vstack((b.row, b.col))
print('coords:',coords.transpose())

data = b.data
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(coords),
                            torch.FloatTensor(data),
                            torch.Size(b.shape))
print(data)
print(adj_norm)
adj_norm = adj_norm.to_dense()
print('adj_norm:',adj_norm)
print('123:',adj_norm[0,1].item())
print(adj_norm.view(-1).long())
a = (adj_norm > 2).view(-1).long()
print('a:',a)
adj_norm = adj_norm.view(-1) == 1
print(adj_norm.size(0))
print(adj_norm)
init_range = np.sqrt(6.0/(1433 + 32))
print('init_range:',init_range)
initial = torch.rand(1433, 32)*2*init_range - init_range
print(torch.rand(1433, 32)*2*init_range)
print(initial)
print('z:',z)
a = [2,2]
print(b)
tol = 5
print(a)
print(np.round(a - b[:,None], tol) == 0)
rows_close = np.all(np.round(a - b[:,None], tol) == 0, axis=-1)
print(rows_close)