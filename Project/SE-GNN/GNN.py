# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:56:12 2020

@author: user
"""

import dgl
from dgl.nn.pytorch.conv import SAGEConv

g=dgl.graph(data=(train[:,0],train[:,1]))

GNN=SAGEConv(8,8,'mean')

a=GNN(g,node_embedding.weight)
