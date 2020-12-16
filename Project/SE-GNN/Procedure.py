# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:36:11 2019

@author: User
"""

import torch
from engine import Engine
from MLPs import score_MLP
from dgl.nn.pytorch.conv import SAGEConv


class Procedure(torch.nn.Module):
    def __init__(self,config):
        super(Procedure,self).__init__()
        self.config=config
        self.node_embedding=torch.nn.Embedding(config['node_number'],
                                                config['node_hidden_size'])
        
        self.times_embedding=torch.nn.Embedding(config['n_times_types'],
                                                config['appearance_time_hidden_size'])
        
        self.interval_embedding=torch.nn.Embedding(config['n_intervals'],
                                                config['interval_hidden_size'])
        
        self.connection_embedding=torch.nn.Embedding(config['n_connection_types'],
                                                config['connection_hidden_size'])
        
        self.score_MLP=score_MLP(config)
        
        self.GNN=SAGEConv(config['node_hidden_size'],config['node_hidden_size'],'mean')
        
        if torch.cuda.is_available():
            self.node_embedding=self.node_embedding.cuda()
            self.connection_embedding=self.connection_embedding.cuda()
            self.score_MLP=self.score_MLP.cuda()
        
    def GNN_forward(self,graph):
        '''
        

        Parameters
        ----------
        graph : dgl graph
            only includes the relations, not contain the node features 

        Returns
        -------
        hidden : tensor
            the updated embedding vectors of each node. shape== (config['node_number'],
                                                config['node_hidden_size'])
        '''
        if torch.cuda.is_available():
            all_nodes=torch.LongTensor([x for x in range(self.config['node_number'] )]).cuda()
        else:
            all_nodes=torch.LongTensor([x for x in range(self.config['node_number'] )])
        hidden=self.node_embedding(all_nodes)
        hidden=self.GNN(graph,hidden)
        return hidden
        
    def forward(self,hidden, source,destination,times, intervals, connection_types ): 
        '''
        Parameters
        ----------
        hidden: tensor
            updated embedding vectors for each node
        source : np array
            (batch_size,)
            
        destination : np array
            (batch_size,)
            
        connection_types : tensor
            (batch_size)

        Returns
        -------
        scores : TYPE
            DESCRIPTION.

        '''
        source_embedding=hidden[source]
        destination_embedding=hidden[destination]
        times_embedding=self.times_embedding(times)
        interval_embedding=self.interval_embedding(intervals)
        connection_embedding=self.connection_embedding(connection_types)
        
        final_scores=self.score_MLP(source_embedding,destination_embedding,times_embedding, interval_embedding, connection_embedding)
        
        return final_scores

    
class ProcedureEngine(Engine):
    def __init__(self,config):
        self.model=Procedure(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        super(ProcedureEngine,self).__init__(config)
        
        model_dict=self.model.state_dict()
        
        pair=torch.load('GNN-TCP_Best.model')
        pretrained_pair={k:v for k,v in pair.items() if k in model_dict}
        model_dict.update(pretrained_pair)
        
        self.model.load_state_dict(model_dict)