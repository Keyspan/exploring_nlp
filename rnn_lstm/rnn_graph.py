#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:59:28 2017

@author: py
"""

import random

from config import config

import numpy as np

import tensorflow as tf

from clean_data import clean_data

config = config()

clean_data = clean_data()

batch_size = config.batch_size

buckets = config.buckets

hidden_size = config.hidden_size

n_epochs = config.n_epochs

x_train = clean_data.x_train

y_train = clean_data.y_train

data_buckets = clean_data.data_bucket(x_train, y_train)

vocab_size = clean_data.vocab_size

def get_batch(data_buckets, bucket_id):
        
        size = buckets[bucket_id]
        
        inputs, targets = [], []
        
        for _ in range(batch_size):
                
            input_, target = random.choice(data_buckets[bucket_id])
            
            input_pad = [config.pad_id] * (size - len(input_))
                
            inputs.append(input_ + input_pad)        
                       
            target_pad = [config.pad_id] * (size - len(target))
                
            targets.append(target + target_pad)
    

        # create batch-major vectors from the data selected above.
            
        batch_inputs, batch_targets, batch_weights = [], [], []
        
        for length_idx in range(size):
            
            batch_inputs.append(
                    np.array([inputs[batch_idx][length_idx
                              ] for batch_idx in range(batch_size)], 
                dtype=np.int32))

        # batch decoder inputs are re-indexed decoder_inputs.
        
        for length_idx in range(size):
            
            batch_targets.append(
                    np.array([targets[batch_idx][length_idx
                              ] for batch_idx in range(batch_size)], 
                dtype=np.int32))

            # create target_weights to be 0 for targets that are padding.
            
            batch_weight = np.ones(batch_size, dtype=np.float32)
        
            for batch_idx in range(batch_size):

                # last target word is empty and mask off the loss for pad.
                
                if batch_targets[batch_idx] == 0:
                    
                    batch_weight[batch_idx] = 0.0
                
            batch_weights.append(batch_weight)
            
        return batch_inputs, batch_targets, batch_weights

def get_random_bucket(data_buckets):
    
    train_bucket_size = [len(data_buckets[b]) for b in range(len(buckets))]
    
    total = sum(train_bucket_size)
    
    train_bucket_ratio = [train_bucket_size[b]/total for b in range(len(train_bucket_size))]
    
    bucket_id = np.random.choice([i for i in range(len(train_bucket_ratio))], p = train_bucket_ratio)
    
    return(bucket_id, train_bucket_ratio)

""" Recurrent Neural Network in TensorFlow """

# Still need dynamic rnn, the length of sentence is different

# Placeholder
with tf.name_scope('data_input'):
    
    inputs = tf.placeholder(tf.int32, [batch_size,])
    
    targets = tf.placeholder(tf.int32, [batch_size,])
    
    seq_len = tf.placeholder(tf.int32, [batch_size])
    
    keep_prob = tf.constant(1.0)
    
    cell_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
    
    hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_size])

# embed layer
with tf.variable_scope('embed'):
    
    embed_matrix = tf.get_variable('embed_matrix', [vocab_size, hidden_size])
    
    rnn_inputs = tf.nn.embedding_lookup(embed_matrix, inputs)

# softmax layer
with tf.variable_scope('softmax'):
    
        W = tf.get_variable('W', [hidden_size, vocab_size])
        
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))
        
# rnn
with tf.name_scope('RNN'):
    
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

    init_state = state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
       
    hidden_states, last_states_tuple = tf.nn.dynamic_rnn(
            cell=cell,
            inputs = rnn_inputs,
            sequence_length= seq_len,
            time_major = False,
            dtype = tf.float32
            )
    
    last_rnn_output = tf.gather_nd(hidden_states, tf.stack([tf.range(batch_size), seq_len-1], axis=1))
    
with tf.name_scope('loss'):    
    
    logits = tf.matmul(last_rnn_output, W)  + b
    
    preds = tf.nn.softmax(logits)
    
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), targets)
    
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = targets)


with tf.name_scope('optimize'):
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(losses)

""" Run the graph """
with tf.name_scope('run_graph'):
    
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
    
    current_epoch = 0
    
    size = 100
    
    full_rounds = int(n_epochs//size)
    
    final_ten_accuracy = np.zeros(shape = 10)
       
    bucket_id = get_random_bucket(data_buckets)
    
    batches = get_batch(data_buckets, bucket_id)

    
    while current_epoch < n_epochs:
        
        inputs_batches = batches.next_(current_epoch)
        
        feed = {inputs: inputs_batches[0], targets: inputs_batches[1], seq_len: inputs_batches[2]}
        sess.run(train_step, feed_dict=feed)
        
        i = current_epoch//size
        if i in range(full_rounds-10, full_rounds):
            accuracy_ = sess.run(accuracy, feed_dict = feed)
            final_ten_accuracy[i-full_rounds+10] += accuracy_
        current_epoch += 1
    sess.close()   

for i in range(10):
    print("The average accuracy for round {} is {} ".format(i + full_rounds-10, final_ten_accuracy[i]/size))
    