#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
import Global_Params
import ipdb

########################### Global Parameters ###############################
num_frames = Global_Params.num_frames #80
n_frame_step = num_frames
dim_image = 4096
dim_hidden = 256
batch_size = 10 # 100
########################### Global Parameters ###############################

############################ Initialization #################################

############################ Initialization #################################

## Generator Model (only for test)
class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, bias_init_vector = None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name = 'Wemb')

        self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
        self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)

        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name = 'encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name = 'encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name = 'embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name = 'embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name = 'embed_word_b')

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []
        probs = []
        embeds = []
        for i in range(self.n_lstm_steps):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)

        for i in range(self.n_lstm_steps):
            tf.get_variable_scope().reuse_variables()
            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [current_embed, output1]), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds, output1, output2

############################### Initialization ###################################
# LSTM sequence model
ixtoword = pd.Series(np.load('/home/thomas/project/video_to_sequence-master/data/ixtoword.npy').tolist())
# load WORD list
model = Video_Caption_Generator(
        dim_image = dim_image,
        n_words = len(ixtoword),
        dim_hidden = dim_hidden,
        batch_size = batch_size,
        n_lstm_steps = n_frame_step,
        bias_init_vector = None)
video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf, output1_tf, output2_tf = model.build_generator()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, Global_Params.model_path)# RESTORE models
############################### Initialization ###################################

## test model and output sentence
def test_model(feats_fullpath):
    # ipdb.set_trace()
    video_feat = np.load(feats_fullpath)[None,...]
    video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
    generated_word_index = sess.run(caption_tf, feed_dict = {video_tf: video_feat, video_mask_tf: video_mask})
    generated_words = ixtoword[generated_word_index]

    # JIANBO use values here: output1_val, output2_val
    # probs_val = sess.run(probs_tf, feed_dict = {video_tf:video_feat})
    # embed_val = sess.run(last_embed_tf, feed_dict = {video_tf:video_feat})
    # output1_val = sess.run(output1_tf,feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
    # output2_val = sess.run(output2_tf,feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

    punctuation = np.argmax(np.array(generated_words) == '.') + 1
    generated_words = generated_words[:punctuation] # extract longest sentence by punctuation
    generated_sentence = ' '.join(generated_words) # add spaces
    print time.time(), "  ", generated_sentence

## MAIN
def main():
    while True:
        test_model(Global_Params.RT_feats_save_fullpath)
        time.sleep(2)

##########################
if __name__ == '__main__':
    sys.exit(main())