import concurrent.futures
from tqdm import tqdm
import networkx as nx

def process_node(node):
    two_hop_neighbors = list(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
    two_hop_subgraph = G.subgraph(two_hop_neighbors)
    nx.write_edgelist(two_hop_subgraph, './sub-graphs/node_{0}.edgelist'.format(node))

G = nx.read_edgelist('./higgs-social_network.edgelist')
print("节点数:", G.number_of_nodes())
print("边数:", G.number_of_edges())

# 创建线程池或进程池
with concurrent.futures.ThreadPoolExecutor() as executor:
# with concurrent.futures.ProcessPoolExecutor() as executor:

    # 提交任务给线程池或进程池
    futures = [executor.submit(process_node, node) for node in G.nodes]

    # 使用 tqdm 进度条追踪任务完成情况
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass
