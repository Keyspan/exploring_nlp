#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:59:28 2017

@author: py
"""

import random

import math

import os

import tensorflow as tf

import numpy as np

from clean_data import clean_data

from config import configs

config = configs()

clean_data = clean_data()
        
batch_size = config.batch_size

buckets = config.buckets

hidden_size = config.hidden_size

steps_per_checkpoint = config.steps_per_checkpoint
            
learning_rate = config.learning_rate
        
max_gradient_norm = config.max_gradient_norm
        
x_train = clean_data.x_train
        
y_train = clean_data.y_train
        
data_buckets = clean_data.data_bucket(x_train, y_train)
                
vocab_size = clean_data.vocab_size

data_folder = config.data_folder

def get_batch(data_buckets, bucket_id):
        
        size = buckets[bucket_id]
        
        inputs, targets, seq_lens = [], [], []
        
        for _ in range(batch_size):
                
            input_, target = random.choice(data_buckets[bucket_id])
            
            input_pad = [config.pad_id] * (size - len(input_))
                
            inputs.append(input_ + input_pad)      
            
            seq_lens.append(len(input_))
                       
            target_pad = [config.pad_id] * (size - len(target))
                
            targets.append(target + target_pad)

            
        return inputs, targets, seq_lens
    
def get_random_bucket(data_buckets):
    
    train_bucket_size = [len(data_buckets[b]) for b in range(len(buckets))]
    
    total = sum(train_bucket_size)
        
    train_bucket_ratio = [train_bucket_size[b]/total for b in range(len(train_bucket_size))]
    
    bucket_id = np.random.choice([i for i in range(len(train_bucket_ratio))], p = train_bucket_ratio)
    
    return bucket_id

                
with tf.name_scope('data_input'):
                
    inputs = tf.placeholder(tf.int32, [batch_size, None])
            
    targets = tf.placeholder(tf.int32, [batch_size, None])
                
    seq_len = tf.placeholder(tf.int32, [batch_size])
                
    cell_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
                
    hidden_state = tf.placeholder(tf.float32, [batch_size, hidden_size])
    
    global_step = tf.Variable(0, trainable=False)
            
# embed layer
with tf.variable_scope('embed'):
              
    embed_matrix = tf.get_variable('embed_matrix', [vocab_size, hidden_size])
    
    rnn_inputs = tf.nn.embedding_lookup(embed_matrix, inputs)

# softmax layer
with tf.name_scope('softmax'):
            
    W = tf.get_variable('W', [hidden_size, vocab_size])
            
    b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))
        
    learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        
# rnn
with tf.name_scope('rnn'):
            
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    
    #init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
    
    hidden_states, last_states_tuple = tf.nn.dynamic_rnn(
            cell=cell,
            inputs = rnn_inputs,
            sequence_length= seq_len,
            time_major = False,
            dtype = tf.float32
            )
    
    
with tf.name_scope('loss'):    
            
    rnn_outputs = tf.unstack(hidden_states)    
    
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    
    preds = tf.nn.softmax(logits)
    
    correct = tf.equal(tf.cast(tf.argmax(preds,2),tf.int32), targets)
    
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
    labels_series = tf.unstack(targets)
            
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = logit, labels = target) for logit, target in zip(logits, labels_series)]
    
    loss_ave = tf.reduce_mean(losses)
        
        
with tf.name_scope('optimize'):
            
    learning_rate_decay_operation = learning_rate.assign(learning_rate * tf.exp(-1.0))
    
    trainable = tf.trainable_variables()
        
    gradients_norms = []
        
    updates = []
        
    opt = tf.train.GradientDescentOptimizer(learning_rate)   
        
    gradients = tf.gradients(loss_ave, trainable)
    
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    
    gradients_norms.append(norm)
    
    updates.append(opt.apply_gradients(zip(clipped_gradients, trainable), 
                                       global_step= global_step))
        
    saver = tf.train.Saver(tf.global_variables())


def train():
    
    loss = 0.0  
    
    ave_accuracy = 0.0
            
    previous_losses = []
        
    current_step = 0
        
    with tf.Session() as sess:                
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/py/checkpoint'))   
        
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        
            print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        
            saver.restore(sess, os.path.join(ckpt.model_checkpoint_path))
            
        else:
            
            print("Created model with fresh parameters.")
        
            sess.run(tf.global_variables_initializer())
            
                
        while True:
                        
            bucket_id = get_random_bucket(data_buckets)
        
            batch = get_batch(data_buckets, bucket_id)
            
            current_step += 1
            
            feed = {inputs: batch[0], targets: batch[1], seq_len: batch[2]}
            
            _, _, step_loss, step_accuracy, step_learning_rate, step_global_step = sess.run(
                    [updates, gradients_norms, loss_ave, accuracy,
                     learning_rate, global_step], feed_dict=feed)   
    
            loss += step_loss/steps_per_checkpoint   
            
            ave_accuracy += step_accuracy/steps_per_checkpoint
                
            if current_step % steps_per_checkpoint == 0:                    
                
                # print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                
                print ("global step {}, learning rate {}, accuracy {}, perplexity {}".format(
                        step_global_step, 
                        step_learning_rate, ave_accuracy, perplexity))
                
                # decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-2:]):
                            
                    sess.run(learning_rate_decay_operation)
                    
                    previous_losses.append(loss)
                    
                # save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join("rnn.ckpt")
                
                saver.save(sess, checkpoint_path, global_step = step_global_step)
            
                loss = 0.0
            
                ave_accuracy = 0.0
        
                
 
#def decode():
    
    #ckpt = tf.train.get_checkpoint_state(os.path.dirname('/checkpoint')) 

       
    

    