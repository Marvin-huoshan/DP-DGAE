import numpy as np
import torch
import numpy
import networkx as nx
#from to_graph import load_data
from data.data_struct import load_data
import args
import matplotlib.pyplot as plt

edge_file = 'data/facebook/107.edges'
feature_file = 'data/facebook/107.feat'
file = 'pre_matrix/A_p_MTL.pth'
if file != 'None':
    weight = torch.load(file)
print(weight)
print(torch.max(weight))
#weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
weight_copy = weight.detach().numpy()
row, col = np.diag_indices_from(weight_copy)
weight_copy[row,col] = 0
weight = (weight == 1).long()
adj,feature = load_data(edge_file, feature_file)
adj = adj.A
G1 = nx.from_numpy_array(adj)
G_1 = list(max(nx.connected_components(G1)))
G_1 = nx.subgraph(G1, G_1)
#print('diameter')
#print(nx.diameter(G_1))
b = nx.to_numpy_array(G_1)
print('original matrix:')
print(b)
a = weight.detach().numpy()

row, col = np.diag_indices_from(a)
a[row,col] = 0
for i in range(1034):
    for j in range(1034):
        if a[i][j] == 1:
            a[j][i] = 1
print((a == 1).sum())
print(a)
G2 = nx.from_numpy_matrix(a)

G_2 = list(max(nx.connected_components(G2), key=len))
G_2 = nx.subgraph(G2, G_2)
dict0 = dict(sorted(nx.degree(G_1), key=lambda x: x[1], reverse=True))
dict1 = dict(sorted(nx.degree(G_2), key=lambda x: x[1], reverse=True))
node_origin = list(dict0.keys())
node_mod = list(dict1.keys())
#add node and edge
def mod_node(node_origin, node_mod, weight_matrix, G_mod):
    node_add = list(set(node_origin).difference(set(node_mod)))
    unfrozen_graph = nx.Graph(G_mod)
    unfrozen_graph.add_nodes_from(node_add)
    for i in node_add:
        edge_max = weight_matrix[i].argmax()
        unfrozen_graph.add_edge(i,edge_max)
    return unfrozen_graph

G_2 = mod_node(node_origin, node_mod, weight_copy, G_2)

dict1 = dict(sorted(nx.degree(G_2), key=lambda x: x[1], reverse=True))

print('origin node:', nx.number_of_nodes(G_1))
print('mod node:', nx.number_of_nodes(G_2))
print('mod graph edges:')
print(nx.number_of_edges(G_2))
num = 0

for i in range(1034):
    for j in range(1034):
        if (a[i][j] == 1 and b[i][j] == 1):
            num += 1
print('sim edges:')
print(num / 2)
print('sim probability:')
print((num / 2) / nx.number_of_edges(G_1))

length = 100
node_origin = list(dict0.keys())[:length]
node_mod = list(dict1.keys())[:length]
list_inter = list(set(node_mod).intersection(set(node_origin)))


print('similarity: {:.2%}'.format(len(list_inter)/length))
print('origin AVE:', 2 * nx.number_of_edges(G_1) / nx.number_of_nodes(G_1))
print('mod AVE:', 2 * nx.number_of_edges(G_2) / nx.number_of_nodes(G_2))
print('origin ACC:', nx.average_clustering(G_1))
print('mod ACC:', nx.average_clustering(G_2))
print('origin APL:', nx.average_shortest_path_length(G_1))
print('mod APL:', nx.average_shortest_path_length(G_2))

dict2 = dict(sorted(nx.degree(G_1), key=lambda x: x[0], reverse=True))
dict3 = dict(sorted(nx.degree(G_2), key=lambda x: x[0], reverse=True))
degree_mod = list(dict1.values())
degree_origin = list(dict0.values())
degree_dis = [degree_mod[i] - degree_origin[i] for i in range(len(degree_origin))]

fig = plt.figure(figsize=(18, 10))
plt.title('degree')
#plot1 = plt.plot(range(len(degree_dis)), degree_dis, c = 'r', label = 'distence')
plot1 = plt.plot(range(len(degree_origin)),degree_origin,color = 'b',label = 'origin')
plot2 = plt.plot(range(len(degree_mod)), degree_mod, color ='r', label='mod')
plt.legend(fontsize='large', loc='upper right')
plt.savefig('degree_match.png')

