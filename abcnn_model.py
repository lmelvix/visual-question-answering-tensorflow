import os,sys
import numpy as np
import tensorflow as tf 
import data_loader as dl
import cv2
import pickle
import skimage
import skimage.io
import skimage.transform
from itertools import cycle
import word2glove as w2g 
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import pylab
import Image


def model(sess, batch_size):

    # Placeholders for input image, question and output answer
    image = tf.placeholder("float", [None, 14, 14, 512], name="VGG")
    ques = tf.placeholder("float", [None, 22, 50], name="ques")
    ans = tf.placeholder("float", [None, 1000], name="ans")

    # TODO: Extract image histogram
    image_feature = image
    
    # Learn semantics of question using LSTM 
    with tf.variable_scope('Question') as scope:
        ques_w = tf.Variable(tf.random_normal([256, 512]), name="W_qa") 
        ques_b = tf.Variable(tf.random_normal([512]), name="b_A")
        ques_sem = ques_semantics(ques, ques_w, ques_b)
        ques_vec = tf.reshape(ques_sem, shape=[-1, 1, 1, 512])
        variable_summaries(ques_w)
    
    # Convolve CCK with Image features
    with tf.variable_scope('Attention') as scope:
    	with tf.variable_scope('embedding') as scope:
        	visual_embed = tf.add(image_feature, ques_vec)
        	visual_embed_activ = tf.nn.relu(visual_embed, name="h_A")
        	variable_summaries(visual_embed_activ)

        with tf.variable_scope('probability') as scope:
        	embed_squeeze = tf.reduce_sum(tf.reduce_sum(visual_embed_activ, axis=2), axis=1)
        	attention_prob = tf.nn.softmax(embed_squeeze, name="p_I")
    
    # Build reduced Image feature using attention map
    with tf.variable_scope('ReducedImage') as scope:
    	attention_prob = tf.reshape(attention_prob, shape=[-1,1,1,512])
        rimage_feature = tf.multiply(image_feature, attention_prob)
        variable_summaries(rimage_feature)
        reduced_image_visualize = tf.reshape(tf.reduce_sum(rimage_feature, axis=3), shape=[-1, 14, 14, 1])
        attention_summary = tf.summary.image('weighted-features', reduced_image_visualize)
        
    # Adding a convolution layer to reduce dimensions
    with tf.variable_scope('RImage') as scope:
        red_img_w = tf.Variable(tf.random_normal([1,1,512,8]), name="weight")
        red_img_b = tf.Variable(tf.random_normal([8]), name="bias")
        red_img_conv = tf.nn.conv2d(rimage_feature, red_img_w, strides=[1,1,1,1], padding='SAME')
        red_img_activ = tf.nn.relu(red_img_conv + red_img_b, name="activation")
        reduced_image_feature = tf.reshape(red_img_activ, shape=[-1, 1568])
        variable_summaries(red_img_w)

    # with tf.variable_scope('Image') as scope:
    #     img_w = tf.Variable(tf.random_normal([1,1,512,8]), name="weight") # 64-->8
    #     img_b = tf.Variable(tf.random_normal([8]), name="bias") # 64 --> 8
    #     img_conv = tf.nn.conv2d(image_feature, img_w, strides=[1,1,1,1], padding='SAME')
    #     img_activ = tf.nn.relu(img_conv + img_b, name="activation")
    #     image_feature = tf.reshape(img_activ, shape=[-1, 1568])
    #     variable_summaries(img_w)

    
    # Combine all three features to build dense layer
    with tf.variable_scope('Dense') as scope:
    	# with tf.variable_scope('question') as scope:
     #    	sem_dense_w = tf.Variable(tf.random_normal([128, 1000]), name="q_weight")
     #    	variable_summaries(sem_dense_w)

     #    with tf.variable_scope('image') as scope:
     #    	img_dense_w = tf.Variable(tf.random_normal([1568, 1000]), name="i_weight") 
     #    	variable_summaries(img_dense_w)

        with tf.variable_scope('attention') as scope:
        	reduced_img_dense_w = tf.Variable(tf.random_normal([1568, 1000]), name="ri_weight") 
        	variable_summaries(reduced_img_dense_w)

        with tf.variable_scope('bias') as scope:
        	dense_b = tf.Variable(tf.random_normal([1000]), name="ans_bias")
        	variable_summaries(dense_b)
    
        dense = tf.matmul(reduced_image_feature, reduced_img_dense_w) + \
                dense_b
        		# tf.matmul(ques_sem, sem_dense_w) + \
                # tf.matmul(image_feature, img_dense_w) + \
                
    # Apply softmax on dense layer to get answers
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=ans)
    mean_cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('mean_cross_entropy', mean_cross_entropy)

    # Create Optimizer for reducing loss
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(mean_cross_entropy)

    # Evaluate model
    correct_prediction = tf.equal(tf.argmax(ans, 1), tf.argmax(dense, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Return Model parameters
    return image, ques, ans, optimizer, mean_cross_entropy, accuracy

def rgb_histogram(im):
    rhist,_ = np.histogram(np.reshape(im[:,:,0], (1,-1)), bins=np.arange(257))  
    ghist,_ = np.histogram(np.reshape(im[:,:,1], (1,-1)), bins=np.arange(257))    
    bhist,_ = np.histogram(np.reshape(im[:,:,2], (1, -1)), bins=np.arange(257))  
    hist = np.append((rhist, ghist, bhist), 0)
    return hist

def ques_semantics(word, weight, bias):
    with tf.variable_scope('LSTM') as scope:
        word = tf.unstack(word, 22, 1)
        lstm_cell = rnn.BasicLSTMCell(256, forget_bias=1.0)
        output, states = rnn.static_rnn(lstm_cell, word, dtype=tf.float32)
        ques_sem = tf.matmul(states[-1], weight) + bias
        return tf.nn.relu(ques_sem, "ques-semantics-acitvation")

def write_tensorboard(sess):
    writer = tf.summary.FileWriter('graph/log/', sess.graph)
    return writer
    
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)