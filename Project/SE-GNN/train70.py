# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:36:29 2019

@author: User
"""
from utils import get_train_data,get_query

from data import SampleGenerator
from Procedure import ProcedureEngine


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
        
        'score_mlp_layers':[32,32,8],
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

sample_generator=SampleGenerator(config,train_data,valid_data,test_data)
engine=ProcedureEngine(config)

for epoch in range(config['num_epoch']):
    print('Epoch{}starts !'.format(epoch))
    print('_'*80)

    #engine.train_an_epoch(sample_generator,epoch)
    #val_f1=engine.evaluate(sample_generator,epoch)
    #engine.save(config['alias'],epoch,val_f1)
    
    engine.get_result(sample_generator,epoch)
    