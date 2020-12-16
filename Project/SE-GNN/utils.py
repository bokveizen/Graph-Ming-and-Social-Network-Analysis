# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:33:01 2019

@author: User
"""

import torch
import os
import numpy as np 

def use_optimizer(model,config):
    if config['optimizer']=='adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=config['adam_lr'],weight_decay=config['l2_regularization'])        
    return optimizer    

def save_checkpoint(model,model_dir):
    torch.save(model.state_dict(),model_dir)

def get_train_data(train_path,attack_types):
    '''
        input: 
            train_path: str that record train data directory
            attack_type: list of attach types
        return:
            train_data (numpy array)
    '''
    
    train_list=os.listdir(train_path)
    train_list=[train_path+x for x in train_list]
    train_data = []
    for filePath in train_list:
        one_file=[]
        with open(filePath) as f:
            data = f.readlines()
        for line in data:
            if line[-1] == '\n':
                line = line[:-1]
            info_list = line.split('\t')
            info_tuple = [int(info_list[0]),  # Source id
                          int(info_list[1]),  # Destination id
                          int(info_list[2]),  # Port
                          int(info_list[3]),  # Timestamp
                          attack_types.index(info_list[4])  # Type of connection
                          ]
            one_file.append(info_tuple)
        train_data.append(np.array(one_file))
    return train_data

def get_query(query_path):
    '''
        input: 
            query_path: str that record query data directory (valid or test)
        return:
            list of (numpy array). one file one array.
    '''
    
    query_list=os.listdir(query_path)
    query_list=[query_path+x for x in query_list]
    query_data = []
    for filePath in query_list:
        with open(filePath) as f:
            data = f.readlines()
            
        one_file=[]
        for line in data:
            if line[-1] == '\n':
                line = line[:-1]
            info_list = line.split('\t')
            info_tuple = [int(info_list[0]),  # Source id
                          int(info_list[1]),  # Destination id
                          int(info_list[2]),  # Port
                          int(info_list[3]),  # Timestamp
                          ]
            one_file.append(info_tuple)
        query_data.append(np.array(one_file))
    return query_data    
    
def weighted_f1(answer_path,preds,attack_types):
    '''
    Parameters
    ----------
    answer_path : str
        path of the answer data ('data/valid_answer/')
    preds : list
        prediction result. [set(),set([1]),...]
    attack_types : list
        attack_types

    Returns
    -------
    final_score: int
    weighted f1 score

    '''
    answer_list=os.listdir(answer_path)
    answer_list=[answer_path+x for x in answer_list]
    A_type_count = B_type_count = score_sum_A = score_sum_B = 0
    for path,prediction in zip(answer_list,preds):
        with open(path) as f:
            answer_data = f.readlines()
        if not answer_data:
            ground_truth = set()
            B_type_count += 1
            prec = 1 / (1 + len(prediction))
            recall = 1
            score_sum_B += 2 * prec * recall / (prec + recall)
        else:
            ground_truth = set([attack_types.index(t) for t in answer_data[0].split('\t')])
            A_type_count += 1
            inter = prediction.intersection(ground_truth)
            prec = (1 + len(inter)) / (1 + len(prediction))
            recall = (1 + len(inter)) / (1 + len(ground_truth))
            score_sum_A += 2 * prec * recall / (prec + recall)
    final_score = (score_sum_A / A_type_count + score_sum_B / B_type_count) / 2
    print('final score:', final_score)
    return final_score