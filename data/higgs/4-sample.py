import numpy as np
import pandas as pd
import networkx as nx
import random
from tqdm.auto import tqdm
import pickle as pkl
from scipy import sparse
import matplotlib.pyplot as plt

dataset = 'higg'
load_data_path = '../{0}'.format(dataset)
save_data_path = '../{0}_sample'.format(dataset)
infected_users = np.load('{0}/infected_users.npy'.format(load_data_path))
with open('./{0}/infected_records.pkl'.format(load_data_path), 'rb') as f:
    infected_records = pkl.load(f)

with open('./{0}/dir_g.pkl'.format(load_data_path), 'rb') as f:
    G = pkl.load(f)
random.seed(1)
nodes = random.sample(np.array(G.nodes).tolist(), 30000)
subgraph = G.subgraph(nodes)
feature_graph = sparse.load_npz('dir_feature_graphs.npz').toarray()
feature_graph = feature_graph[nodes]
feature_graph = sparse.csr_matrix(feature_graph)
sparse.save_npz('{0}/dir_feature_graphs.npz'.format(save_data_path), feature_graph)

no_index_subgraph = nx.convert_node_labels_to_integers(subgraph)
edges = {'source': [], 'target': []}
for edge in tqdm(no_index_subgraph.edges()):
    node1, node2 = edge
    edges['source'].append(node1)
    edges['target'].append(node2)
edges = pd.DataFrame(edges)
edges.to_csv('{0}/dir_new_edges.csv'.format(save_data_path), index=False)
with open('{0}/dir_g.pkl'.format(save_data_path), 'wb') as f:
    pkl.dump(no_index_subgraph, f)
exit(0)
