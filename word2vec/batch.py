#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:25:19 2017

@author: py
"""


"""This file will be imported in word2vec.py"""

import numpy as np

class batch:
    
    def __init__(self, centers, targets, batch_size):
        
        self.centers = centers
        
        self.targets = targets
        
        self.batch_size = batch_size
      
    def next_(self, batch_group):

        batch_index = self.batch_size * batch_group
        
        s = batch_index % len(self.centers)
        
        e = (s + self.batch_size)%len(self.centers)
        
        if s < e :
        
            centers_bp = np.reshape(self.centers[s: e],[self.batch_size])
            
            targets_bp = np.reshape(self.targets[s: e], [self.batch_size, 1])
            
        else:
        
            centers_bp = np.reshape(self.centers[:e] + self.centers[s:],[self.batch_size])
            
            targets_bp = np.reshape(self.targets[:e] + self.targets[s:], [self.batch_size, 1])
            
        return centers_bp, targets_bp
    
if __name__ == '__main__':

    batch_ = batch()
    
    batch_.next_()

