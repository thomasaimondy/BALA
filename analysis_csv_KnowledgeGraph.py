#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import cv2
import ipdb


def getEnglishwords(readcsvpath):
    csv_data = pd.read_csv(readcsvpath, sep=',')
    # ipdb.set_trace()
    csv_data = csv_data[csv_data['Language'] == 'English']
    sentences = csv_data['Description']
    print(sentences)



if __name__=="__main__":
    readcsvpath = 'video_corpus.csv'
    getEnglishwords(readcsvpath)