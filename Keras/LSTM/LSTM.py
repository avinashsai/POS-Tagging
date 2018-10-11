import os
import sys
import re
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import nltk
from nltk.corpus import treebank

import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM
from keras.layers import Bidirectional,Dropout
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import Embedding,Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


nltk.download('treebank')
nltk.download('universal_tagset')

corpus = treebank.tagged_sents(tagset='universal')

print(len(corpus1))

print(corpus[0:2])

def preprocess(word):
    return word.lower()

processed = []
tags = []


for sentence in corpus:
    temp = []
    new_sentence = " "
    for word,tag in sentence:
        if(tag!='.'):
            temp.append(tag)
            new_sentence+=preprocess(word)+" "
    tags.append(temp)
    processed.append(new_sentence)
    assert (len(new_sentence.split())==len(temp))


tag_dict = {}
index = 0
for tag in tags:
    for temp in tag:
        if temp not in tag_dict:
            tag_dict[temp] = index
            index+=1


tag_dict['PAD'] = index


out_len = len(tag_dict)

print(tag_dict)

corpus_length = len(corpus)


max_length = 0
for sentence in processed:
    if(len(sentence.split())>max_length):
        max_length = len(sentence.split())


max_sentence_length = 10


def generate_labels(tag):
    temp_tag = np.zeros((max_sentence_length,out_len))
    min_len = min(len(tag),max_sentence_length)
    for i in range(min_len):
        temp_tag[i,:] = np.eye(out_len)[tag_dict[tag[i]]]
    if(min_len==len(tag)):
        for i in range(min_len,max_sentence_length):
            temp_tag[i,:] = np.eye(out_len)[tag_dict['PAD']]
    return temp_tag


labels = np.zeros((corpus_length,max_sentence_length,out_len))



for i in range(corpus_length):
    labels[i,:,:] = generate_labels(tags[i])



Xtrain,Xtest,ytrain,ytest = train_test_split(processed,labels,test_size=0.2,random_state=42)



train_length = len(Xtrain)
test_length = len(Xtest)


print(Xtrain[0:2])


assert (len(Xtrain)==len(ytrain))


assert (len(Xtest)==len(ytest))


max_words = 5000


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(Xtrain)


train_indices = pad_sequences(tokenizer.texts_to_sequences(Xtrain),maxlen=max_sentence_length)
test_indices = pad_sequences(tokenizer.texts_to_sequences(Xtest),maxlen=max_sentence_length)


print(train_indices.shape)


print(ytrain.shape)


embedding_dim = 100


model = Sequential()
model.add(Embedding(max_words,embedding_dim,trainable=False))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(TimeDistributed(Dense(out_len,activation='softmax')))


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


model.fit(train_indices,ytrain,batch_size=32,validation_data=(test_indices,ytest),epochs=50)

