# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:02:40 2020

@author: user
"""
import torch

class score_MLP(torch.nn.Module):
    def __init__(self,config):
        super(score_MLP,self).__init__()
        self.fc_layers=torch.nn.ModuleList()
        for idx, (in_size,out_size) in enumerate(zip(config['score_mlp_layers'][:-1],config['score_mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size,out_size))
        self.affine_output = torch.nn.Linear(in_features=config['score_mlp_layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()
            
    def forward(self,source_embedding,destination_embedding,times_embedding, interval_embedding, connection_embedding):     
        vector=torch.cat([source_embedding,destination_embedding,times_embedding, interval_embedding, connection_embedding],dim=-1)
        for idx in range(len(self.fc_layers)):
            vector=self.fc_layers[idx](vector)
            vector=torch.nn.ReLU()(vector)  
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating    