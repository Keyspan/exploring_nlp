#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:42:42 2017

@author: py
"""

import nltk

from nltk.tokenize import ToktokTokenizer

import csv

import itertools

from config import configs

config = configs()

train_size = config.train_size


class clean_data():
    
    def __init__(self):
        
        data_path = config.data_path

        ratio = config.freq_ratio
        
        start_vocabs = config.start_vocabs
        
        self.buckets = config.buckets
        
        print("Reading 'tasks.csv' file...")
        
        with open(data_path, 'r', encoding = "utf-8") as f:
            
            reader = csv.reader(f, skipinitialspace=True)
            
            next(reader)

            sentences = [x[0].lower() for x in reader]
            
            self.sentences = sentences[:train_size]
    
        print("{} sentences loaded.".format(len(self.sentences)))


        # tokenize sentences
        
        tok = ToktokTokenizer()

        self.tokenized_sens = [tok.tokenize(sen) for sen in self.sentences] 
        
        # clean sentences and only consider sentences with length > 1
        
        self.tokenized_sens = [[x for x in sen if x.isalpha()] for sen in self.tokenized_sens]
    
        self.tokenized_sens = [sen for sen in self.tokenized_sens if sen != [] and len(sen) > 1]



        # remove low frequency words and index them
        
        frequency_words = nltk.FreqDist(itertools.chain(*self.tokenized_sens))
        
        size = len(list(set(itertools.chain(*(self.tokenized_sens)))))
        
        self.vocabs = start_vocabs + [w[0] for w in frequency_words.most_common(int(size*ratio))]
    
        self.vocab_size = len(self.vocabs)
           
        self.word_to_index = dict([(w,i) for i, w in enumerate(self.vocabs)])
        
        self.tokenized_sens = [[w if w in self.vocabs else '_unk' for w in sen
                                ] for sen in self.tokenized_sens]
        # create train data

        self.x_train = [[self.word_to_index[w] for w in sen[:-1]] for sen in self.tokenized_sens]
        
        self.y_train = [[self.word_to_index[w] for w in sen[1:]] for sen in self.tokenized_sens]
        

    # bucket data
    
    def data_bucket(self, x, y):        
        
        data_buckets = [[] for _ in self.buckets]
        
        current_process = 0
    
        for current_process in range(len(x)):
                
            if (current_process + 1) % 10000 == 0:
                
                print("Bucketing the", current_process + 1, "sentences")
            
            x_current = x[current_process]
            
            y_current = y[current_process]
            
            for bucket_id, sentence_size in enumerate(self.buckets):
            
                if len(x_current) <= sentence_size:
                    
                    data_buckets[bucket_id].append([x_current, y_current])
                                        
                    break
                
            current_process += 1
            
        return data_buckets
    
    
        
        
        
        