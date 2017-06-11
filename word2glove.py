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

def build_glove_dict(datapath, dest_file):

    # Check if Glove model exists
    if os.path.isfile(datapath):
        print "Glove file exists. Preparing dictionary"
    else:
        print "ERROR: Glove dict not found"
        return -1

    # Read glove file as dictionary
    glove_dict = {}
    progress = 0
    with open(datapath) as f:
        for line in f.readlines():
            word = line.split()
            glove_dict[word[0]] = [float(vec) for vec in word[1:]]
            progress += 1

            if progress%1000==0:
                print "Processed %d vectors" % (progress)

    # Saving glove dictionary
    with open(dest_file, 'wb') as handle:
        pickle.dump(glove_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print "Saved glove dictionary in %s" % (dest_file)
    return 0

def get_glove_dict(src_file):

    # Check if Glove dict exists
    if os.path.isfile(src_file):
        print "Glove dictionary exists. Retrieving data"
    else:
        print "ERROR: Glove dict not found"
        return -1

    with open(src_file, 'rb') as handle:
        glove_dict = pickle.load(handle)

    print "Completed Glove dict retrieval"
    return glove_dict

def build_missing_w2g(word_dict, glove_dict, dest_file):

    missing_words = []
    total_words = len(word_dict.keys())
    progress = 0

    # Append words not found in Glove dict to list
    for word in word_dict.keys():
        progress += 1
        if word.lower() not in glove_dict:
            missing_words.append(word_dict[word])
        if progress%1000 == 0:
            print "Processed %d out of %d words" %(progress, total_words)

    with open(dest_file, 'wb') as handle:
        pickle.dump(missing_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print "Saved missing words in %s" %(dest_file)
    return 0

def is_missing_encoding(words, missing_pkl):

    # Load list of missing words
    with open(missing_pkl, 'rb') as handle:
        missing_list = pickle.load(handle)

    # Return True if any word in question does not have Glove vector
    if not set(words).isdisjoint(missing_list):
        return True
    else:
        return False

def encode(word_idx, vocab, glove_dict):
        
    # Handle padding separately
    if word_idx not in vocab.values():
        return np.zeros((len(glove_dict['the'])))
    else:
        word = vocab.keys()[vocab.values().index(word_idx)]
        return np.array(glove_dict[word.lower()])

def encode_onehot(word_idx, vocab, dummy):
	return np.eye(len(vocab.keys()))[int(word_idx)]