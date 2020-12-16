# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:13:18 2019

@author: User
"""
import torch
from utils import use_optimizer,save_checkpoint,weighted_f1
import numpy as np


class Engine(object):
    def __init__(self,config):
        self.config=config
        self.opt=use_optimizer(self.model,config)
        self.crit=torch.nn.BCELoss()
        torch.autograd.set_detect_anomaly(True)
        
        if torch.cuda.is_available():
            self.crit=self.crit.cuda()
        
    def train_single_batch(self,hidden,source,destination,times,intervals,connection_types,targets):

        self.opt.zero_grad()
        scores=self.model(hidden,source,destination,times,intervals,connection_types) #(batch_size)
        loss=self.crit(scores,targets)
        loss.backward()
        self.opt.step()
        loss=loss.item()
        return loss

    
    def train_an_epoch(self,sample_generator,epoch_id):
        self.model.train()
        total_loss=0
        #hidden=self.model.GNN_forward(sample_generator.graph)
        files=sample_generator.train_data
        for i,file in enumerate(files):
            print('file',i)
            batches=sample_generator.generate_train_batch(file)
            if len(batches) ==0:
                continue
            for batch_id, batch in enumerate(batches):
                print('batch',batch_id)
                (source,destination,times,intervals,connection_types,targets) = sample_generator.get_train_batch(batch)
                hidden=self.model.GNN_forward(sample_generator.graph)
                loss=self.train_single_batch(hidden,source,destination,times,intervals,connection_types,targets)
                total_loss+=loss
        print('[Training Epoch{}] Batch {}, loss {}'.format(epoch_id,batch_id,loss))  
    def evaluate(self,sample_generator,epoch_id):
        self.model.eval()
        files=sample_generator.valid_data # list of arrays, every array is a valid_file
        preds=[]
        for i,file in enumerate(files):
            single_pred=set()
            batches=sample_generator.generate_val_batch(file) #list of batches,每一个batch都是一个（s,d）和所有的connection type
            hidden=self.model.GNN_forward(sample_generator.graph)
            for batch in batches:
                (source,destination,times, intervals, connection_types) = sample_generator.get_val_batch(batch)
                scores=self.model(hidden,source,destination,times, intervals, connection_types) # shape (n_types, 1)
                scores=scores.cpu().detach().numpy().reshape([-1])
                most_possible=scores.argmax()
                if most_possible != 0:
                    single_pred.add(most_possible)
            preds.append(single_pred) 
            print('finish {}-th evaluation file'.format(i))
            print(single_pred)
        F1=weighted_f1(self.config['valid_answer_path'],preds,self.config['attack_types'])
        
        return F1
    
    def get_result(self,sample_generator,epoch_id):
        self.model.eval()
        #get valid result
        files=sample_generator.valid_data # list of arrays, every array is a valid_file
        preds=[]
        valid_result={}
        for i,file in enumerate(files):
            single_pred=set()
            batches=sample_generator.generate_val_batch(file) #list of batches,每一个batch都是一个（s,d）和所有的connection type
            hidden=self.model.GNN_forward(sample_generator.graph)
            for batch in batches:
                (source,destination,times, intervals, connection_types) = sample_generator.get_val_batch(batch)
                scores=self.model(hidden,source,destination,times, intervals, connection_types) # shape (n_types, 1)
                scores=scores.cpu().detach().numpy().reshape([-1])
                valid_result[(int(source[0]), int(destination[0]))]=scores
                most_possible=scores.argmax()
                if most_possible != 0:
                    single_pred.add(most_possible)
            preds.append(single_pred) 
            print('finish {}-th evaluation file'.format(i))
            print(single_pred)
        F1=weighted_f1(self.config['valid_answer_path'],preds,self.config['attack_types'])
        np.save('valid.npy', valid_result)
        
        #get test result
        files=sample_generator.test_data # list of arrays, every array is a test_file
        preds=[]
        test_result={}
        for i,file in enumerate(files):
            single_pred=set()
            batches=sample_generator.generate_val_batch(file) #list of batches,每一个batch都是一个（s,d）和所有的connection type
            hidden=self.model.GNN_forward(sample_generator.graph)
            for batch in batches:
                (source,destination,times, intervals, connection_types) = sample_generator.get_val_batch(batch)
                scores=self.model(hidden,source,destination,times, intervals, connection_types) # shape (n_types, 1)
                scores=scores.cpu().detach().numpy().reshape([-1])
                test_result[(int(source[0]), int(destination[0]))]=scores
            print('finish {}-th evaluation file'.format(i))
        np.save('test.npy', valid_result)
        
        return F1
   
    
    def save(self,alias,epoch_id,val_f1):
        model_dir=self.config['model_dir'].format(alias,epoch_id,val_f1)
        save_checkpoint(self.model,model_dir)