import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain

# 空手道俱乐部
G=

com = community_louvain.best_partition(G)

#节点大小设置，与度关联
node_size = [G.degree(i)**1*20 for i in G.nodes()]


#格式整理
df_com = pd.DataFrame({'Group_id':com.values(),
                       'object_id':com.keys()}
                    )
# 统计每个团伙人数 并降序
df_com.groupby('Group_id').count().sort_values(by='object_id', ascending=False)


# 颜色设置
colors = ['DeepPink','orange','DarkCyan','#A0CBE2','#3CB371','b','orange','y','c','#838B8B','purple','olive','#A0CBE2','#4EEE94']*500
colors = [colors[i] for i in com.values()]



#使用 kamada_kawai_layout spring_layout 布局
plt.figure(figsize=(4,3),dpi=500)
nx.draw_networkx(G,

                 pos = nx.spring_layout(G),
                 node_color = colors,
                 edge_color = '#2E8B57',
                 font_color = 'black',
                 node_size = node_size,
                 font_size = 5,
                 alpha = 0.9,
                 width = 0.1,
                 font_weight=0.9
                 )
plt.axis('off')
plt.show()