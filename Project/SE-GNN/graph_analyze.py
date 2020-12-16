# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:54:08 2020

@author: user
"""

from utils import get_train_data,get_query
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

config={'alias':'GNN-TCP',
        'model_dir':'checkpoints/{}_Epoch{}_valF1{:.4f}.model',
        'train_path':'data/train/',
        'valid_query_path':'data/valid_query/',
        'valid_answer_path':'data/valid_answer/',
        'test_query_path':'data/test_query/',
        'attack_types':['-', 'apache2', 'back', 'dict', 'guest', 'httptunnel-e', 'ignore', 'ipsweep', 'mailbomb', 'mscan',
                'neptune', 'nmap', 'pod', 'portsweep', 'processtable', 'rootkit', 'saint', 'satan', 'smurf', 'smurfttl',
                'snmpgetattack', 'snmpguess', 'teardrop', 'warez', 'warezclient'],
        
        'node_number':23398,
        'node_hidden_size':8,
        
        'n_connection_types':25,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        'connection_hidden_size':8,
        
        'n_times_types':4,
        'appearance_time_hidden_size':4,
        
        'n_intervals':3,
        'interval_hidden_size':4,
        
        'score_mlp_layers':[24,24,8],
        'batch_size':200,
        'num_epoch':200,
        'optimizer':'adam',
        'adam_lr':1e-3,
        'l2_regularization':0,
        'test_size':500,
        'GNNStep':3}

train_data=get_train_data(config['train_path'],config['attack_types'])
valid_data=get_query(config['valid_query_path'])
test_data=get_query(config['test_query_path'])

train=np.concatenate(train_data)
valid=np.concatenate(valid_data)
test=np.concatenate(test_data)
total=np.concatenate([train[:,:-1],valid,test])

edges=[]
for line in total:
    edges.append((int(line[0]),int(line[1])))

G=nx.DiGraph()
G.add_edges_from(edges)

nx.write_edgelist(G,'wholeGraph.csv',data=False)

G.number_of_nodes()
G.number_of_edges()

in_degree={}
out_degree={}
clustering_coefficient={}
for node in list(G.nodes):
    in_degree[node]=G.in_degree[node]
    out_degree[node]=G.out_degree[node]
    clustering_coefficient[node]=nx.clustering(G,node)

# 画图
# in degree
plt.hist(np.array(list(in_degree.values())),bins=40)
plt.xlabel('in degree')
plt.ylabel('appearance number')
plt.title('distribution of in_degrees')
plt.show()


max(list(in_degree.values())) # 8061
a=[x for x in list(in_degree.values()) if x >5]
len(a)

# out degree
plt.hist(np.array(list(out_degree.values())),bins=40)
plt.xlabel('out degree')
plt.ylabel('appearance number')
plt.title('distribution of out_degrees')
plt.show()


max(list(out_degree.values())) #7294

# clustering_coefficient
plt.hist(np.array(list(clustering_coefficient.values())),bins=40)
plt.xlabel('clustering_coefficient')
plt.ylabel('appearance number')
plt.title('distribution of clustering_coefficient')
plt.show()



max(list(clustering_coefficient.values())) #7294
