import networkx as nx
import pickle as pkl

# G = nx.read_edgelist("dir_g.edgelist", create_using=nx.DiGraph)
with open('dir_g.pkl', 'rb') as f:
    g = pkl.load(f)
print(1)
