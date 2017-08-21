#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:56:31 2017

@author: py
"""
import os

class configs():
    
    def __init__(self):
        
        self.data_folder = '/Users/py/Python/Python3.6/rnn/'
        
        self.data_path = os.path.join('example.csv')
        
        self.freq_ratio = 0.8
        
        self.pad_id = 0
        
        self.unk_id = 1
        
        self.start_vocabs = ['_pad', '_unk']
        
        self.buckets = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300]
        
        self.batch_size = 10
        
        self.steps_per_checkpoint = 1000
        
        self.hidden_size = 128
        
        self.max_gradient_norm = 1
        
        self.learning_rate = 1.0
        
        self.train_size = 50000
        
