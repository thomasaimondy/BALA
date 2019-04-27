#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import cv2
import ipdb
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

filt = 10 # don't calculate the nodes frequencey < filt
def getEnglishwords(readcsvpath):
    csv_data = pd.read_csv(readcsvpath, sep=',')
    # ipdb.set_trace()
    csv_data = csv_data[csv_data['Language'] == 'English']
    sentences = csv_data['Description']
    # print(sentences)
    
    # nodes
    nodes_id = []
    nodes_modularity = []
    nodes_times = {}
    noun_count = 0;
    verb_count = 0;
    for sent in sentences:
        try:
            text1 = nltk.word_tokenize(sent)
            words = nltk.pos_tag(text1) #[('A', 'DT'),]
        except:
            continue
        for word in words:
            # calculate times
            if word[0] in nodes_times.keys():
                nodes_times[word[0]]  = nodes_times[word[0]] +1
            else:
                nodes_times[word[0]] = 1

            if word[0] in nodes_id: # word[0] is A, word[1] is DT
                continue
            # Word vector of nouns ['NN','NNS','NNPS'] 
            # Word vector of verbs ['VB','VBD','VBG','VBN','VBP','VBZ']

            # NN  Noun, singular or mass
            # NNS Noun, plural
            # NNP Proper noun, singular
            # NNPS    Proper noun, plural


            # VB  Verb, base form
            # VBD Verb, past tense
            # VBG Verb, gerund or present participle
            # VBN Verb, past participle
            # VBP Verb, non-3rd person singular present
            # VBZ Verb, 3rd person singular present


            if word[1] in ['NN','NNS','NNPS']:
                nodes_id.append(word[0])
                nodes_modularity.append(word[1])
                noun_count = noun_count + 1 
            if word[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                nodes_id.append(word[0])
                nodes_modularity.append(word[1])
                verb_count = verb_count + 1

    print('The nount number is : ' + str(noun_count));
    print('The verb number is : ' + str(verb_count));
    df = pd.DataFrame({"id":nodes_id, "label":nodes_id, "modularity_class":nodes_modularity})
    df.to_csv('nodes.csv')

    # connections
    connections = {}
    for sent in sentences:
        try:
            text1 = nltk.word_tokenize(sent)
            words = nltk.pos_tag(text1) #[('A', 'DT'),]
        except:
            continue
        for i in range(len(words)):
            for j in range(len(words)):
                if i==j:
                    continue
                w_from = words[i]
                w_to = words[j]

                if nodes_times[w_from[0]] < filt or  nodes_times[w_to[0]] < filt:  # filter freguqence
                    continue

                if w_from[0] not in nodes_id or  w_to[0] not in nodes_id:
                    continue

                # both w_from and w_to are nouns or verbs
                key = (w_from[0] , w_to[0])
                if key in connections:
                    connections[key] = connections[key] +1
                else:
                    connections[key] = 1
    conn_source = [key[0] for key in connections.keys()]
    conn_tar = [key[1] for key in connections.keys()]
    conn_weight = connections.values()


    df = pd.DataFrame({"Source":conn_source, "Target":conn_tar, "Weight":conn_weight})
    df.to_csv('connections_filt_'+str(filt)+'.csv')




if __name__=="__main__":
    readcsvpath = 'video_corpus.csv'
    getEnglishwords(readcsvpath)