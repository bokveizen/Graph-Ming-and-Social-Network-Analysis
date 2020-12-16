# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 22:38:59 2019

@author: User
"""

import torch
import numpy as np
import dgl
import random

class SampleGenerator(object):
    def __init__(self,config,train_data,valid_data,test_data):
        '''
        

        Parameters
        ----------
        config : dict
            setting parameters
        train_data : list of numpy.array (~,5)
            array of the train_data. [source, destination, port, timestamp, connection type]
        valid_data : list of numpy.array (~,4)
            [source, destination, port, timestamp]
        test_data : list of numpy.array (~,4)
            [source, destination, port, timestamp]

        Returns
        -------
        None.

        '''
        self.config=config
        self.train_data=train_data
        self.valid_data=valid_data
        self.test_data=test_data
        self.graph=self.graph()
        
    def graph(self):
        '''
        Returns
        -------
        graph : dgl graph
            range of node id is (0~23397)
            total nonredundant nodes number is 23395

        '''
        train=np.concatenate(self.train_data)
        valid=np.concatenate(self.valid_data)
        test=np.concatenate(self.test_data)
        total=np.concatenate([train[:,:-1],valid,test])
        
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        graph=dgl.graph(data=(total[:,0], total[:,1]), num_nodes=self.config['node_number'], device=device)
        return graph
    

    def generate_train_batch(self,file):
        '''
        Parameters
        ----------
        file : np array
            array of the train_data. [source, destination, port, timestamp, connection type]

        Returns
        -------
        batches: list of np arrays [(batch_size,6),...]
            [s(id),d,times(e.g. 10次), interval (e.g. 296), connection types, target]

        '''
        #build the dictionaries for sd
        sd_times={}
        sd_start={}
        sd_end={}
        sd_types={}
        for line in file:
            if (int(line[0]),int(line[1])) not in sd_times.keys():
                sd_times[(int(line[0]),int(line[1]))]=1
                sd_start[(int(line[0]),int(line[1]))]=int(line[3])
                sd_end[(int(line[0]),int(line[1]))]=int(line[3])
                sd_types[(int(line[0]),int(line[1]))]=[int(line[-1])]
                continue
            sd_times[(int(line[0]),int(line[1]))]+=1
            sd_end[(int(line[0]),int(line[1]))]=int(line[3])
            if int(line[-1]) not in sd_types[(int(line[0]),int(line[1]))]:
                sd_types[(int(line[0]),int(line[1]))].append(int(line[-1]))
                
        # generate sd_pairs [s,d,times,interval,type,target]    
        sd_pairs=[]
        for sd in sd_times.keys():
            interval=sd_end[sd]-sd_start[sd]
            
            #positive sampling
            for t in sd_types[sd]:
                positive_instance=list(sd)+[sd_times[sd]]+[interval]+[t]+[1]
                sd_pairs.append(positive_instance)
             
            #negative sampling
            neg_set=set([x for x in range(self.config['n_connection_types'])]).difference(set(sd_types[sd]))   
            neg_chosen=random.sample(list(neg_set),4)
            if (0 not in sd_types[sd]) and (0 not in neg_chosen):
                neg_chosen.append(0)
            for t in neg_chosen:
                negative_instance=list(sd)+[sd_times[sd]]+[interval]+[t]+[0]
                sd_pairs.append(negative_instance)
        sd_pairs=np.array(sd_pairs) # [s,d,times,interval, type, target]
        
        if sd_pairs.shape[0] ==0:
            batches= []
        elif int(sd_pairs.shape[0]/self.config['batch_size']) > 0:
            last_batch_length=sd_pairs.shape[0]%self.config['batch_size']
            batches=np.split(sd_pairs[:-last_batch_length,:],int(sd_pairs.shape[0]/self.config['batch_size']),axis=0 ) 
            batches.append(sd_pairs[-last_batch_length:,:])
        else:
            batches=[sd_pairs]
            
        return batches 
            
    def generate_test_batch(self,test_size):
        
        return 
    
    def generate_val_batch(self,file):
        '''
        Parameters
        ----------
        file : numpy array shape(~,4)
            all the connections in a file. [source, destination, port, timestamp]

        Returns
        -------
        batches: list of batches. 每一个batch形状为(25,5)
            每一个batch都是一个（s(id),d,times (10次),interval）和所有的connection type的np array

        '''
        # generate sd_pairs [s,d,times,interval]
        sd_times={}
        sd_start={}
        sd_end={}
        for line in file:
            if (int(line[0]),int(line[1])) not in sd_times.keys():
                sd_times[(int(line[0]),int(line[1]))]=1
                sd_start[(int(line[0]),int(line[1]))]=int(line[3])
                sd_end[(int(line[0]),int(line[1]))]=int(line[3])
                continue
            sd_times[(int(line[0]),int(line[1]))]+=1
            sd_end[(int(line[0]),int(line[1]))]=int(line[3])
            
        sd_pairs=[]
        for sd in sd_times.keys():
            interval=sd_end[sd]-sd_start[sd]
            instance=list(sd)+[sd_times[sd]]+[interval]
            sd_pairs.append(instance)
        sd_pairs=np.array(sd_pairs) # [s,d,times,interval]
        
        # for every sd_pair, generate a batch 
        batches=[]
        for line in sd_pairs:
            sd=np.array([int(line[0]),int(line[1]),int(line[2]),int(line[3])])
            sd=np.tile(sd,(len(self.config['attack_types']),1))
            types=np.array([i for i in range(len(self.config['attack_types']))]).reshape([-1,1])
            sdt=np.concatenate((sd,types),axis=1)
            batches.append(sdt)
        
        return batches  
    
    def get_train_batch(self,batch):
        '''
        Parameters
        ----------
        batch : numpy array
            shape (batch_size,5) [source, destination, times, interval, connection type,target]

        Returns
        -------
        source : np array
            shape==(batch_size,)
        destination : np array
            shape==(batch_size,)
        times: LongTensor
            divided times
        intervals: LongTensor
            divided intervals
        connection_types : tensor
            DESCRIPTION.
        targets : tensor
            DESCRIPTION.

        '''
        
            
        source=batch[:,0]
        destination=batch[:,1]
            
        times=torch.LongTensor(self.divide_times([x for x in batch[:,2]]))
        intervals=torch.LongTensor(self.divide_intervals([x for x in batch[:,3]]))
        connection_types=torch.LongTensor([x for x in batch[:,-2]])
        targets=torch.FloatTensor([x for x in batch[:,-1]])
        targets=targets.view([-1,1])
        
        
        return_contents=[times,intervals,connection_types,targets]
        if torch.cuda.is_available():
            for i in range(len(return_contents)):
                return_contents[i]=return_contents[i].cuda()
        (times,intervals,connection_types,targets)=tuple(return_contents)

        return (source,destination,times,intervals,connection_types,targets)
    
    def get_val_batch(self, batch):
        '''

        Parameters
        ----------
        batch : np array. shape(25,5)
            (s,d,times,interval, t) for all the connection types

        Returns
        -------
        source : np array
            shape==(batch_size,)
        destination : np array
            shape==(batch_size,)
        times: LongTensor
            divided times
        intervals: LongTensor
            divided intervals
        connection_types : tensor
            DESCRIPTION.

        '''
       
        source=batch[:,0]
        destination=batch[:,1]  
        
        times=torch.LongTensor(self.divide_times([x for x in batch[:,2]]))
        intervals=torch.LongTensor(self.divide_intervals([x for x in batch[:,3]]))
        connection_types=torch.LongTensor([x for x in batch[:,-1]])   
                
        return_contents=[times, intervals, connection_types]
        if torch.cuda.is_available():
            for i in range(len(return_contents)):
                return_contents[i]=return_contents[i].cuda()
        (times,intervals,connection_types)=tuple(return_contents)
        
        return (source,destination,times,intervals,connection_types)
    
    def divide_times(self, times):
        '''
        Parameters
        ----------
        times : list
            lisf of real appearance times. [53,64,1,...] 

        Returns
        -------
        times : list
            list of types of appearance times. [2,2,0,...]

        '''
        for i in range(len(times)):
            if times[i]==1:
                times[i]=0
            elif times[i] <10:
                times[i]=1
            elif times[i] <100:
                times[i]=2
            else:
                times[i]=3
        
        return times
    
    def divide_intervals(self, intervals):
        '''
        Parameters
        ----------
        intervals : list
            lisf of real intervals [68,262,0,...] 

        Returns
        -------
        intervals : list
            list of types of intervals [1,1,0,...]

        '''
        for i in range(len(intervals)):
            if intervals[i]<10:
                intervals[i]=0
            elif intervals[i] <300:
                intervals[i]=1
            else:
                intervals[i]=2
        
        return intervals
        
