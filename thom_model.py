#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import sys as system
import cv2
import time
# from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import rnn_cell
# from tensorflow.nn import rnn_cell
# from thom_preprocessing import get_features_from_video 
# from keras.preprocessing import sequence
import ipdb

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps

        with tf.device("/gpu:1"):#
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        # self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
        # self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)

        self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden,state_is_tuple=True)
        self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden,state_is_tuple=True)


        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    #used
    def build_generator(self):
        
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])
        
        state1 = (tf.zeros([1,self.dim_hidden]),)*2
        state2 = (tf.zeros([1,self.dim_hidden]),)*2
        padding = tf.zeros([1,self.dim_hidden])

        # ipdb.set_trace()
        # state1 = tf.zeros([1, self.lstm1.state_size])
        # state2 = tf.zeros([1, self.lstm2.state_size])
        # padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []
        for i in range(self.n_lstm_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( image_emb[:,i,:], state1 )

            with tf.variable_scope("LSTM2"):
                # ipdb.set_trace()
                output2, state2 = self.lstm2( tf.concat([padding,output1],1), state2 )
        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()
            
            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])
            # ipdb.set_trace()
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat([current_embed,output1],1), state2 )

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:1"):  # '/gpu:1'  "/cpu:0"
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)
        return video, video_mask, generated_words, probs, embeds,output1,output2


def get_feat_list(video_feat_path):
    # ipdb.set_trace()
    if os.path.exists(video_feat_path):
        f_list = os.listdir(video_feat_path)
        if len(f_list)!=0:
            f_list= map(lambda x: os.path.join(video_feat_path, x),f_list)
        return f_list
        #os.path.splitext(): # split name and . name
    else:
        print "NO TARGET PATH"


def test_file(model_path, video_feat_path,ixtoword_path,n_frame_step):
    
    # read the word-index table
    ixtoword = pd.Series(np.load(ixtoword_path).tolist())
    # load pre trained model
    model = Video_Caption_Generator(
             dim_image=dim_image,
             n_words=len(ixtoword),
             dim_hidden=dim_hidden,
             batch_size=batch_size,
             n_lstm_steps=n_frame_step,
             bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf,output1_tf,output2_tf = model.build_generator()
    sess = tf.InteractiveSession()   
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    # start transfer from features to the words
    feature_path_list = get_feat_list(video_feat_path)

    for temp_feat_path in feature_path_list:
        # print temp_feat_path
        # print(temp_feat_path)
        if not os.path.isfile(temp_feat_path):
            continue
        video_feat = np.load(temp_feat_path)[None,...] # add a dimension from 20,2046 to 1,20,4096
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
        embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]
        
        output1_val = sess.run(output1_tf,feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        output2_val = sess.run(output2_tf,feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        #ipdb.set_trace()  # JIANBO use values here, output1_val,output2_val
        punctuation = np.argmax(np.array(generated_words) == '.')+1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        print time.time(),"  ",generated_sentence

############### Global Parameters ###############
root_path = '/home/bibo/Data/video_to_sequence/'
############## Train Parameters #################
dim_image = 4096
dim_hidden= 256
n_epochs = 1000
batch_size = 100
learning_rate = 0.001
##################################################

if __name__=="__main__":
    # test vides
    model_path = root_path + 'models_new/model-2900'
    video_feat_path = root_path + 'data/test_one_feature/'
    ixtoword_path = root_path + 'data/ixtoword.npy'
    n_frame_step = 20  # 80 20 
    system.exit(test_file(model_path,video_feat_path,ixtoword_path,n_frame_step))


    # # test frames
    # model_path = root_path + 'models_new/model-2900'
    # video_feat_path = root_path + 'data/test_one_feature/task_vi/'
    # ixtoword_path = root_path + 'data/ixtoword.npy'
    # system.exit(test_file(model_path,video_feat_path,ixtoword_path))
