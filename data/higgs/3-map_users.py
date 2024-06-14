import numpy as np
import pandas as pd
import networkx as nx
import json
from tqdm.auto import tqdm
import pickle as pkl
from datetime import datetime, timedelta
import datetime as dt
from scipy import sparse
import matplotlib.pyplot as plt

dataset = 'higg'
data_path = '../{0}'.format(dataset)
infected_users = np.load('{0}/infected_users.npy'.format(data_path))
with open('./{0}/infected_records.pkl'.format(data_path), 'rb') as f:
    infected_records = pkl.load(f)

# G = nx.read_edgelist("higgs-social_network.edgelist", create_using=nx.DiGraph)
# subgraph = G.subgraph(infected_users)
#
# subgraph = nx.convert_node_labels_to_integers(subgraph)
# nx.write_edgelist(G, 'g.edgelist', delimiter=' ', data=False)
#
# # with open('./{0}/graph.pkl'.format(data_path), 'wb') as f:
# #     pkl.dump(subgraph, f)
# exit(0)

# with open('./{0}/graph.pkl'.format(data_path), 'rb') as f:
#     subgraph = pkl.load(f)

subgraph = nx.read_edgelist('g.edgelist', create_using=nx.DiGraph(), nodetype=int, delimiter=' ')
print('loaded subgraph')
feature_graph = np.zeros(((max(np.array(subgraph.nodes).astype(int)) + 1), len(infected_records)))

# generate user in networks and their infos
for index, day in tqdm(enumerate(infected_records)):
    day_infect = np.array(infected_records[day]).astype(int)
    if day == 'original':
        feature_graph[day_infect, 0] = 1
    else:
        feature_graph[:, index] = feature_graph[:, index - 1]
        feature_graph[day_infect, index] = 1
feature_graph = feature_graph[np.array(subgraph.nodes).astype(int), :]
feature_graph = sparse.csr_matrix(feature_graph)
sparse.save_npz('dir_feature_graphs.npz', feature_graph)

no_index_subgraph = nx.convert_node_labels_to_integers(subgraph)
edges = {'source': [], 'target': []}
for edge in tqdm(no_index_subgraph.edges()):
    node1, node2 = edge
    edges['source'].append(node1)
    edges['target'].append(node2)
edges = pd.DataFrame(edges)
edges.to_csv('./dir_new_edges.csv', index=False)
with open('./{0}/dir_g.pkl'.format(data_path), 'wb') as f:
    pkl.dump(no_index_subgraph, f)
exit(0)
