import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import Constant
from keras.layers import Bidirectional, TimeDistributed
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, Input, CuDNNGRU, GlobalMaxPooling1D, BatchNormalization, TimeDistributed, Flatten
from keras.layers import Convolution1D, Dropout, GRU
from keras_preprocessing import sequence


import seaborn as sns
from keras import backend as K
from keras.engine.topology import Layer

from nltk.tokenize import TweetTokenizer
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import re

import nltk
from sklearn.utils import shuffle
from tqdm import tqdm
#nltk.download('stopwords')
from sklearn import metrics
train_path = '/LIAR-PLUS/dataset/train2.tsv'
test_path = '/LIAR-PLUS/dataset/test2.tsv'
val_path = '/LIAR-PLUS/dataset/val2.tsv'

train_df = pd.read_csv(train_path, sep="\t", header=None)
test_df = pd.read_csv(test_path, sep="\t", header=None)
val_df = pd.read_csv(val_path, sep="\t", header=None)

train = train_df.values
test = test_df.values
val = val_df.values

labels = {'train':[train[i][2] for i in range(len(train))], 'test':[test[i][2] for i in range(len(test))], 'val':[val[i][2] for i in range(len(val))]}
statements = {'train':[train[i][3] for i in range(len(train))], 'test':[test[i][3] for i in range(len(test))], 'val':[val[i][3] for i in range(len(val))]}
subjects = {'train':[train[i][4] for i in range(len(train))], 'test':[test[i][4] for i in range(len(test))], 'val':[val[i][4] for i in range(len(val))]}
speaker = {'train':[train[i][5] for i in range(len(train))], 'test':[test[i][5] for i in range(len(test))], 'val':[val[i][5] for i in range(len(val))]}
job = {'train':[train[i][6] for i in range(len(train))], 'test':[test[i][6] for i in range(len(test))], 'val':[val[i][6] for i in range(len(val))]}
state = {'train':[train[i][7] for i in range(len(train))], 'test':[test[i][7] for i in range(len(test))], 'val':[val[i][7] for i in range(len(val))]}
affiliation = {'train':[train[i][8] for i in range(len(train))], 'test':[test[i][8] for i in range(len(test))], 'val':[val[i][8] for i in range(len(val))]}
credit = {'train':[train[i][9:14] for i in range(len(train))], 'test':[test[i][9:14] for i in range(len(test))], 'val':[val[i][9:14] for i in range(len(val))]}
context = {'train':[train[i][14] for i in range(len(train))], 'test':[test[i][14] for i in range(len(test))], 'val':[val[i][14] for i in range(len(val))]}
justification = {'train':[train[i][15] for i in range(len(train))], 'test':[test[i][15] for i in range(len(test))], 'val':[val[i][15] for i in range(len(val))]}

def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = [1,0,0,0,0,0]
        elif a[i]=='mostly-true':
            a_cat[i] = [0,1,0,0,0,0]
        elif a[i]=='half-true':
            a_cat[i] = [0,0,1,0,0,0]
        elif a[i]=='barely-true':
            a_cat[i] = [0,0,0,1,0,0]
        elif a[i]=='false':
            a_cat[i] = [0,0,0,0,1,0]
        elif a[i]=='pants-fire':
            a_cat[i] = [0,0,0,0,0,1]
        else:
            print('Incorrect label')
    return a_cat

def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = [1,0]
        elif a[i]=='mostly-true':
            a_cat[i] = [1,0]
        elif a[i]=='half-true':
            a_cat[i] = [1,0]
        elif a[i]=='barely-true':
            a_cat[i] = [0,1]
        elif a[i]=='false':
            a_cat[i] = [0,1]
        elif a[i]=='pants-fire':
            a_cat[i] = [0,1]
        else:
            print('Incorrect label')
    return a_cat

labels_onehot = {'train':to_onehot(labels['train']), 'test':to_onehot(labels['test']), 'val':to_onehot(labels['val'])}
BASE_DIR = '/home/ubuntu/fake_news'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'LIAR-PLUS/dataset')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
NUM_FILTERS = 50
WINDOW_SIZE = 8

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Train
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(statements['train'])
sequences_train = tokenizer.texts_to_sequences(statements['train'])
word_index = tokenizer.word_index
print('Found %s unique tokens in Train.' % len(word_index))
data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of Train data tensor:', data_train.shape)

# Test
#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(statements['test'])
sequences_test = tokenizer.texts_to_sequences(statements['test'])
word_index = tokenizer.word_index
print('Found %s unique tokens in Test.' % len(word_index))
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of Test data tensor:', data_test.shape)

# Val
#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(statements['val'])
sequences_val = tokenizer.texts_to_sequences(statements['val'])
word_index = tokenizer.word_index
print('Found %s unique tokens in val.' % len(word_index))
data_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of val data tensor:', data_val.shape)

#word_index_ = {**word_index_train, **word_index_test, **word_index_val}

x_train = data_train
y_train = np.asarray(labels_onehot['train'])
x_val = data_val
y_val = np.asarray(labels_onehot['val'])
x_test = data_test
y_test = np.asarray(labels_onehot['test'])

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
print(num_words)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

class Position_Embedding(Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


'''
output dimention: [batch_size, time_step, nb_head*size_per_head]
every word can be represented as a vector [nb_head*size_per_head]
'''
class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

config = {
    "trainable": False,
    "max_len": 70,
    "max_features": 95000,
    "embed_size": 300,
    "units": 64,
    "num_heads": 8,
    "dr": 0.5,
    "epochs": 2,
    "model_checkpoint_path": "best_weights",
}

def build_model(config):

    inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    x = embedding_layer(inp)
    x = Position_Embedding()(x)
    x = Attention(config["num_heads"], config["units"])([x, x, x])  #output: [batch_size, time_step, nb_head*size_per_head]
    x = GlobalAveragePooling1D()(x)
    x = Dropout(config["dr"])(x)
    
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs = inp, outputs = x)
    model.compile(
        loss = "categorical_crossentropy", 
        #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
        optimizer = Adam(lr=0.0001),
        metrics = ["accuracy"])
    
    return model

'''model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
'''
model = build_model(config)
model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          validation_data=(x_val, y_val))
model.save('model_attention.h5')
