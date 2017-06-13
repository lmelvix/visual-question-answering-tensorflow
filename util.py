import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np 
import tensorflow as tf 
import skimage
import skimage.io
import skimage.transform
from itertools import cycle
import word2glove as w2g 
import data_loader as dl 
import abcnn_model as abc 

def one_hot_ans(vqa, ans):
    return np.eye(len(vqa['answer_vocab'].keys()))[ans]

def get_batch(sess, vqa, batch_size, mode='training'):

    missing_pkl = 'data/missing_glove_ques.pkl'
    glove_pkl = 'data/glove_6B_50.pkl'
    image_batch = np.array([])
    image_vgg_batch = np.array([])
    answer_batch = np.array([])
    question_batch = np.array([])
    vgg, images = dl.getVGGhandle()
    batch_count = 0
    glove_dict = w2g.get_glove_dict(glove_pkl)

    if mode == 'training':
    	purpose = 'train'
    	img_datapath = 'data/train2014'
    else:
    	purpose = 'val'
    	mode = 'validation'
    	img_datapath = 'data/val2014'
    
    for data in cycle(vqa[mode]): # Changed to smaller subset
        vqa_id    =  data['image_id']
        vqa_ans   =  data['answer']
        vqa_ques =  data['question']

        # Skip question if it does not have Glove vector encoding
        if w2g.is_missing_encoding(vqa_ques, missing_pkl) == True:
            continue

        # Filter non-YES/NO/2 answers to avoid skew
        if vqa_ans < 3:
        	continue
        	
        # Get image and build batch
        img = dl.getImage(img_datapath, vqa_id, purpose)
        if( len(img.shape) < 3 or img.shape[2] < 3 ):
            continue
        img = skimage.transform.resize(img, (224, 224))
        img = img.reshape((1, 224, 224, 3))
        if image_batch.size==0:
            image_batch = abc.rgb_histogram(img)
        else:
            image_batch = np.concatenate((image_batch, abc.rgb_histogram(img)), axis=0)

        # Get VGG image features and build batch
        img_vgg = dl.getImageFeatures(sess, vgg, images, img.reshape((224,224,3)))
        if image_vgg_batch.size==0:
            image_vgg_batch = img_vgg
        else:
            image_vgg_batch = np.concatenate((image_vgg_batch,img_vgg),0)
        
        # Get answer and build batch
        if answer_batch.size==0:
            answer_batch = one_hot_ans(vqa, vqa_ans)
        else:
            answer_batch = np.vstack((answer_batch, one_hot_ans(vqa, vqa_ans)))

        # Get Glove encoded question
        question_word = np.array([])
        for word in vqa_ques:
            encoded_word = w2g.encode(word, vqa['question_vocab'], glove_dict)
            if question_word.size==0:
                question_word = encoded_word
            else:
                question_word = np.dstack((question_word, encoded_word))
                
        if question_batch.size==0:
            question_batch = question_word
        else:
            question_batch = np.concatenate((question_batch, question_word), 0)
        
        batch_count += 1
        if batch_count==batch_size:
            yield image_batch, image_vgg_batch, np.transpose(question_batch,(0,2,1)), answer_batch 
            batch_count = 0
            image_batch = np.zeros((0, 224, 224, 3))
            question_batch = np.array([])
            answer_batch = np.array([])
            image_vgg_batch = np.array([])
