#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:26:40 2017

@author: py
"""

# skip_window: represents how far you go from target word for labels at left or right side.
# num_skips: how many pairs you'd like produce for specific target word.
# batch_size: how many pairs are processed each step.
# vocab_size: remember we initial embed the word with one-hot vector, the dimension of the vector is the vocab_size.
# hidden_size: how many neurons we set up for each hidden layer
# n_epochs: how many loops we'd like to run.
# num_sampled: we use noise-contrastive estimation to ease the computation. See reference paper getting to now nce.
# I am using TensorFlow 1.0 and Python 3.5

import numpy as np

import collections

import random

import math

import tensorflow as tf

from batch import batch  # remember the batch.py we define above?

# define word2vec function

def word2vec(data, num_skips, skip_window, 
             batch_size,  hidden_size, vocab_size,
             learning_rate, n_epochs, num_sampled):    

    # generate target-label pairs for specific sentence, 'data' -> 'one sentence'
    
    def generate_skip(data):
    
        data_index = 0
        
        size = 2*skip_window*len(data)
        
        assert num_skips <= 2 * skip_window
        
        centers = np.ndarray(shape=(size), dtype=np.int32)
        
        labels = np.ndarray(shape=(size, 1), dtype=np.int32)
        
        span = 2 * skip_window + 1  # [structure: skip_window, target , skip_window ]
        
        buffer = collections.deque(maxlen=span)
        
        for _ in range(span):
        
            buffer.append(data[data_index])
            
            data_index = (data_index + 1) % len(data)
            
        for i in range(size // num_skips):
        
            target = skip_window  # target label at the center of the buffer
            
            targets_to_avoid = [skip_window]
            
            for j in range(num_skips):
            
                while target in targets_to_avoid:
                
                    target = random.randint(0, span - 1)
                    
                targets_to_avoid.append(target)
                
                centers[i * num_skips + j] = buffer[skip_window]
                
                labels[i * num_skips + j, 0] = buffer[target]
                
            buffer.append(data[data_index])
            
            data_index = (data_index + 1) % len(data) # Move one forward
            
        return centers, labels


    # produce skip pairs for each sentence then concatenate   
    
    centers = []
    
    targets = []
    
    for sequence in data:
    
        centers_element, targets_element = generate_skip(sequence)
        
        centers.append(centers_element)
        
        targets.append(targets_element)
        
    centers = list(np.concatenate(centers))
    
    targets = list(np.concatenate(targets))
    
    """Tensorflow Infrastructure"""
    
    # Step 1: define the placeholders for input and output 
            
    with tf.name_scope("batch_data"):
    
        center_words = tf.placeholder(tf.int32, shape=[batch_size], name='center_words')
        
        target_words = tf.placeholder(tf.int32, shape=[batch_size, 1], name='target_words')
    
    # Step 2: define weights. In word2vec, embed matrix is the weight matrix

    with tf.name_scope("embed"):
    
        embed_matrix = tf.Variable(tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0), name = "embed_matrix")
    
    # Step 3+4: define inference + loss function

    with tf.name_scope("loss"):
        
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name = "embed")
        
        nce_weight = tf.Variable(tf.truncated_normal([vocab_size, hidden_size], 
                                                 stddev = 1.0/math.sqrt(hidden_size)), name = "nce_weight" )
        
        nce_bias = tf.Variable(tf.zeros([vocab_size]), name = "bias")
        
        loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, biases = nce_bias, 
                                         labels = target_words, inputs=embed, 
                                         num_sampled = num_sampled, num_classes = vocab_size, name = "loss"))
    
    # Step 5: define optimizer

    with tf.name_scope("optimizer"):
        
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


    
    # Step 6: run the graph

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        
        current_epoch = 0
        
        print('Initialized, now begin Word2Vec!')
        
        writer = tf.summary.FileWriter('./my_graph', sess.graph)
        
        batch_ = batch(centers, targets, batch_size)
        
        for current_epoch in range(n_epochs):
        
            centers_batch, targets_batch = batch_.next_(current_epoch)
            
            feed = {center_words: centers_batch, target_words: targets_batch}
            
            loss_, _ = sess.run([loss, optimizer], feed_dict = feed)
            
            if (current_epoch+1) % 10000 == 0:
            
                print('The loss for {} iteration is {}'.format(current_epoch, loss_))
                
            current_epoch += 1
            
        embedding = sess.run(embed_matrix, feed_dict = feed)    
        
        writer.close()
        
    return embedding 
    
