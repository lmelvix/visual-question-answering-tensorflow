import os,sys
import numpy as np
import pickle
import warnings

import tensorflow as tf 
from tensorflow.contrib import rnn

import data_loader as dl
import word2glove as w2g
import abcnn_model as abc
import util

def preprocess_question():
	glove_source = 'data/glove.6B.50d.txt'
	glove_pkl = 'data/glove_6B_50.pkl'
	missing_pkl = 'data/missing_glove_ques.pkl'

	# Prepare Glove Dictionary
	if os.path.isfile(glove_pkl):
		print "Glove dictionary already exists!"
	else:
		if w2g.build_glove_dict(glove_source, glove_pkl) == 0:
			print "COMPLETED: Glove dictionary parsing"

	# Identify missing words in Glove dictionary
	if os.path.isfile(missing_pkl):
		print "Missing question vectors already processed!"
	else:
		# Load Glove Dictionary
		glove_dict = w2g.get_glove_dict(glove_pkl)

		# Load VQA training data
		vqa_data = dl.load_questions_answers('data') 
		print "COMPLETED: VQA data retrieval"
		ques_vocab = vqa_data['question_vocab']
		if w2g.build_missing_w2g(ques_vocab, glove_dict, missing_pkl) == 0:
			print "COMPLETED Missing question words identification"

def train():
	batch_size = 10
	print "Starting ABC-CNN training"
	vqa = dl.load_questions_answers('data')

	# Create subset of data for over-fitting
	sub_vqa = {}
	sub_vqa['training'] = vqa['training'][:10]
	sub_vqa['validation'] = vqa['validation'][:10]
	sub_vqa['answer_vocab'] = vqa['answer_vocab']
	sub_vqa['question_vocab'] = vqa['question_vocab']
	sub_vqa['max_question_length'] = vqa['max_question_length']

	train_size = len(vqa['training'])
	max_itr = (train_size // batch_size) * 10

	with tf.Session() as sess:
		image, ques, ans, optimizer, loss, accuracy = abc.model(sess, batch_size)
		print "Defined ABC model"

		train_loader = util.get_batch(sess, vqa, batch_size, 'training')
		print "Created train dataset generator"

		valid_loader = util.get_batch(sess, vqa, batch_size, 'validation')
		print "Created validation dataset generator"

		writer = abc.write_tensorboard(sess)
		init = tf.global_variables_initializer()	    
		merged = tf.summary.merge_all()
		sess.run(init)
		print "Initialized Tensor variables"

		itr = 1

		while itr < max_itr:
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()

			_, vgg_batch, ques_batch, answer_batch = train_loader.next()
			_, valid_vgg_batch, valid_ques_batch, valid_answer_batch = valid_loader.next() 
			sess.run(optimizer, feed_dict={image: vgg_batch, ques: ques_batch, ans: answer_batch})
			[train_summary, train_loss, train_accuracy] = sess.run([merged, loss, accuracy], 
			                                        feed_dict={image: vgg_batch, ques: ques_batch, ans: answer_batch},
			                                        options=run_options,
			                                        run_metadata=run_metadata)
			[valid_loss, valid_accuracy] = sess.run([loss, accuracy],
													feed_dict={image: valid_vgg_batch, 
													ques: valid_ques_batch, 
													ans: valid_answer_batch})

			writer.add_run_metadata(run_metadata, 'step%03d' % itr)
			writer.add_summary(train_summary, itr)
			writer.flush()
			print "Iteration:%d\tTraining Loss:%f\tTraining Accuracy:%f\tValidation Loss:%f\tValidation Accuracy:%f"%(
				itr, train_loss, 100.*train_accuracy, valid_loss, 100.*valid_accuracy)
			itr += 1

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	# preprocess_question()
	train()
	
