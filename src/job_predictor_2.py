#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:12:25 2020

@author: mcgaritym
"""

# =============================================================================
# STEP 0: IMPORT LIBRARIES
# =============================================================================

import re
import glob
import string
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import inspect
import gensim
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature
import keras
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
#nltk.download('wordnet')

local_vars = {}

def job_keyword_counter(file):
    
    global local_vars
    
    # load files into dataframe from csv
    df = pd.read_csv(file)

    for index,row in df.iterrows():
        row['job_text'] = row['job_text'].replace("',", '.')
        row['job_text'] = row['job_text'].replace("' ", ",")
        row['job_text'] = row['job_text'].replace(". '", ". ")
        row['job_text'] = row['job_text'].replace('. "', '. ')
        row['job_text'] = row['job_text'].replace(",", "")
        row['job_text'] = row['job_text'].replace("'", "")
        row['job_text'] = row['job_text'].replace('"', '')
        row['job_text'] = row['job_text'].replace('..', '.')
        row['job_text'] = row['job_text'].replace(' .', '.')
        row['job_text'] = row['job_text'].replace('..', '.')    
        row['job_text'] = row['job_text'].replace(' •', '')
        row['job_text'] = row['job_text'].replace('•', '')
        row['job_text'] = row['job_text'].replace(' ·', '')
        row['job_text'] = row['job_text'].replace('·', '')    
        row['job_text'] = row['job_text'].replace(':.', ':')
        row['job_text'] = row['job_text'].replace('.  ', '. ')
        re.sub(r'(?<=[.,])(?=[^\s])', r' ', row['job_text']) 
        row['job_text'] = row['job_text'].replace(u'\xa0', u' ')   
        row['job_text'] = " ".join(row['job_text'].split())
        row['job_text'] = str(row['job_text'])[1:-1] 
       
    df = df[df['job_text'].apply(lambda x: len(x) > 0)]
    
    # find number of job postings used in analysis
    num_jobs = len(df)    
    
    # Tokenize the article: tokens
    tokens = [word_tokenize(x) for x in df['job_text']]
    
    tokens = [item for sublist in tokens for item in sublist]
    
    # Convert the tokens into lowercase: lower_tokens
    lower_tokens = [t.lower() for t in tokens]
    
    # # Retain alphabetic words: alpha_only
    alpha_only = [t for t in lower_tokens if t.isalpha()]
    
    # set stop words
    stop_words = set(stopwords.words('english')) 
    
    # # Remove all stop words: no_stops
    no_stops = [t for t in alpha_only if t not in stop_words]
    
    # # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # # Lemmatize all tokens into a new list: lemmatized
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
    
    # # Create the bag-of-words: bow
    bow = Counter(lemmatized)
    
    # create dataframe from dictionry
    df_count = pd.DataFrame.from_dict(bow, orient='index').reset_index()
    df_count.columns = ['keywords', 'counts']
    df_count = df_count.sort_values(by ='counts' , ascending=False, ignore_index=True)
    df_count['avg_frequency_per_job'] = df_count['counts'] / num_jobs
    # # create list of data tools
    # data_tools = ['bi', 'mongodb', 'sas', 'hive', 'git', 'linux', 'pig', 'pearl', 'airflow', 'bigquery', 'tensorflow', 'jupyter', 'keras', 'scikit-learn', 'd3', 'excel', 'matlab', 'python', 'sql', 'java', 'aws', 'spark', 'azure', 'javascript', 'c', 'microsoft', 'scala', 'docker', 'nosql', 'hadoop', 'react', 'kafka', 'mysql', 'r', 'react', 'ruby', 'redshift', 'google', 'kuberenetes', 'jenkins', 'amazon', 'tableau']
    
    # df_count = df_count[df_count['keywords'].isin(data_tools)]
    # df_count.reset_index(drop=True, inplace=True)
    
    # # Print the 10 most common tokens
    # print('\n' + file.split(sep='_')[0] + ' ' + file.split(sep='_')[1] + ' Top Keywords:\n\n', df_count.iloc[:20])
    local_vars = inspect.currentframe().f_locals
    return df_count

    # # plot keyword frequency
    # sns.set()
    # fig, ax = plt.subplots(figsize = (6, 11), dpi=250)   
    # sns.barplot(x="avg_frequency_per_job", y="keywords", data=df_count, palette="Blues_d")
    # ax.set(title = 'Linkedin Jobs - ' + file.split(sep='_')[0] + ' ' + file.split(sep='_')[1] + ' Keyword Frequency')
    # plt.show()
    
        

# load and clean dataframes
#job_keyword_counter('DATA_ENGINEER_linkedin_jobs_2020-08-29__22-50-06.csv')
#job_keyword_counter('DATA_SCIENTIST_linkedin_jobs_2020-08-29__22-29-51.csv')


# =============================================================================
# STEP 1: GATHER DATA
# =============================================================================

# collect files in local directory, concatenate to dataframe
files = glob.glob('DATA*jobs_*.csv')
print(files)
df_jobs = pd.DataFrame()
for f in files:
    data = pd.read_csv(f)
    df_jobs = pd.concat([df_jobs, data], axis=0)    
    
# =============================================================================
# STEP 2: CLEAN DATA
# =============================================================================

# clean dataframe
df_jobs = df_jobs.dropna(subset=['job_text', 'job_title'])
df_jobs = df_jobs[df_jobs['job_text'] != '[]']
for index,row in df_jobs.iterrows():
    row['job_text'] = row['job_text'].replace("',", '.')
    row['job_text'] = row['job_text'].replace("' ", ",")
    row['job_text'] = row['job_text'].replace(". '", ". ")
    row['job_text'] = row['job_text'].replace('. "', '. ')
    row['job_text'] = row['job_text'].replace(",", "")
    row['job_text'] = row['job_text'].replace("'", "")
    row['job_text'] = row['job_text'].replace('"', '')
    row['job_text'] = row['job_text'].replace('..', '.')
    row['job_text'] = row['job_text'].replace(' .', '.')
    row['job_text'] = row['job_text'].replace('..', '.')    
    row['job_text'] = row['job_text'].replace(' •', '')
    row['job_text'] = row['job_text'].replace('•', '')
    row['job_text'] = row['job_text'].replace(' ·', '')
    row['job_text'] = row['job_text'].replace('·', '')    
    row['job_text'] = row['job_text'].replace(':.', ':')
    row['job_text'] = row['job_text'].replace('.  ', '. ')
    re.sub(r'(?<=[.,])(?=[^\s])', r' ', row['job_text']) 
    row['job_text'] = " ".join(row['job_text'].split())
    row['job_text'] = str(row['job_text'])[1:-1]  
    row['job_text'] = re.sub(r'<[A-Za-z/]*\>+', '', row['job_text'])
    row['job_text'] = re.sub(r'\\xa0', '', row['job_text'])
    row['job_text'] = re.sub(r'\\n', '', row['job_text'])    
    
# filter for data scientist and data engineer 
df_jobs = df_jobs[(df_jobs['job_title'].str.contains('Data Scientist', case=False)) | (df_jobs['job_title'].str.contains('Data Engineer', case=False))]

# add identifying column for data scientist (#0) and data engineer (#1)
df_jobs['label'] = df_jobs['job_title'].apply(lambda x: 0 if 'Scientist' in x or 'scientist' in x or 'SCIENTIST' in x else 1 if 'Engineer' in x or 'engineer' in x or 'ENGINEER' in x else '')
df_jobs = df_jobs.dropna(subset=['label'])
df_jobs.reset_index(inplace=True, drop=True)

# train and test split the data 80/20
split = int(len(df_jobs)*(0.8))
print(split)
df_jobs_train = df_jobs[:split]
df_jobs_test = df_jobs[split:]

# =============================================================================
# MODEL 1: random forest model based on document vectors and labels
# =============================================================================

# generate the doc2vec tagged documents and generate labels
def tokenize_text_labels_doc2vec_train(text_and_labels_df):
    #this function generates our doc2vec tagged documents, and provies our labels as a numpy matrix
    text = text_and_labels_df['job_text'].tolist()
    text_tokenized = []
    for i in range(len(text)):
        text_tokenized.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text[i]), [i]))
    
    labels = text_and_labels_df['label'].tolist()
    numpy_labels = np.zeros((text_and_labels_df.shape[0], 1))
    for i, label in enumerate(labels):
        if label == 1:
            numpy_labels[i, 0] = 1
    return (text_tokenized, numpy_labels)

# assign function outputs to train variables
train_text, train_labels = tokenize_text_labels_doc2vec_train(df_jobs_train)

# build and train doc2vec
gensim_model = gensim.models.doc2vec.Doc2Vec(vector_size=150, window=10, epochs=5, workers=12, sample=1e-4, dbow_words = 1, compute_loss = True)
gensim_model.build_vocab(train_text)
gensim_model.train(train_text, total_examples=gensim_model.corpus_count, epochs=gensim_model.epochs)
gensim_model.save('gensim.model')

#get  document vectors for training set
doc_vecs = [gensim_model.docvecs[i] for i in range(len(train_text))]
doc_vecs, train_labels = np.asarray(doc_vecs), np.asarray(train_labels)

# train random forest model (note the input values for n_estimators and max_depth)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0, max_features = 150)
clf.fit(doc_vecs, train_labels.ravel())
#here we print feature importances, this is interesting for knowing which nodes are important
#however, these nodes are not interpretable, so it's generally unusable
print(clf.feature_importances_)

# prepare test data for random forest
def tokenize_text_and_labels_doc2vec_test(text_and_labels_df):
    text = text_and_labels_df['job_text'].tolist()
    text_tokenized = [preprocess_string(x) for x in text]
    labels = text_and_labels_df['label'].tolist()
    numpy_labels = np.zeros((text_and_labels_df.shape[0], 1))
    for i, label in enumerate(labels):
        if label == 1:
            numpy_labels[i, 0] = 1
    return (text_tokenized, numpy_labels)

# assign function outputs as variables
test_text, test_labels = tokenize_text_and_labels_doc2vec_test(df_jobs_test)

# vectorize test dataset
vectors = [gensim_model.infer_vector(text) for text in test_text]

# get predictions and accuracy metrics
preds = clf.predict_proba(vectors)
precision, recall, thresholds = precision_recall_curve(test_labels.ravel(), preds[:,1].ravel())
    
# plot precision vs. recall in matplotlib
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
sns.set()
fig, ax = plt.subplots(dpi=200)                     
plt.step(recall, precision, color='b', alpha=0.5, where='post')
plt.fill_between(recall, precision, alpha=0.5, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.show()

# =============================================================================
# MODEL 2: keras LSTM
# =============================================================================

# tokenize data
def tokenize_text_labels_keras(text_and_labels_df):
    text = text_and_labels_df['job_text'].tolist()
    text_tokenized = [preprocess_string(x) for x in text]
    labels = text_and_labels_df['label'].tolist()
    numpy_labels = np.zeros((text_and_labels_df.shape[0], 1))
    for i, label in enumerate(labels):
        if label == 1:
            numpy_labels[i, 0] = 1
    return (text_tokenized, numpy_labels)

train_text, train_labels = tokenize_text_labels_keras(df_jobs_train)

# define val, test split 20/80
val_test_split = int(len(df_jobs_test)*(0.2))
print(val_test_split)
val_text, val_labels = tokenize_text_labels_keras(df_jobs_test[:val_test_split])
test_text, test_labels = tokenize_text_labels_keras(df_jobs_test[val_test_split:])

#here we set MAX_SEQUENCE_LENGTH, which represents the longest document we expect to see
#all documents will be normalized to this size
MAX_SEQUENCE_LENGTH = 50

# encode documents as word indices
def encode_and_pad_text(list_of_tokenized_text, gensim_model):
    encoded_text_train = []
    for tokenized_text in list_of_tokenized_text[:]:
        encoded_text = []
        for word in tokenized_text:
            if word in gensim_model.wv.vocab:
                encode_num = gensim_model.wv.vocab[word].index+1
                encoded_text.append(encode_num)
        encoded_text_train.append(encoded_text)
    
    #if a document is longer than MAX_SEQUENCE_LENGTH, then it is truncated
    #if a document is shorter than MAX_SEQUENCE_LENGTH, then we pad it with zeros
    batch = sequence.pad_sequences(encoded_text_train, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post',\
                                    truncating = 'post', dtype = 'float32')
    return batch

# assign function outputs as variables
train_encode = encode_and_pad_text(train_text, gensim_model)
val_encode = encode_and_pad_text(val_text, gensim_model)
test_encode = encode_and_pad_text(test_text, gensim_model)

# Look at an example encoding of a document below: this is now a list of 50 integers, each representing a word in the document
train_encode[500]

# build embedding matrix - this embedding matrix allows models to feed in the word indexes and return the word embedding. This allows us to feed models just the documents as lists of word indexes and the embedding matrix, and the model can convert the documents to sequences of word embeddings on the fly. The benefit is that this dramatically shrinks the size of our dataset as we feed it to the model, but the conversions do provide slight computational overhead during training.
#here we set the embedding dimension for the model and the vocab size
#these allow us to initialize a blank embedding matrix of size vocab_size*embedding_dim
embedding_dim = 150
vocab_size = len(gensim_model.wv.vocab)

def make_embedding_matrix(model, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size+1, embedding_dim))
    count = 0
    
    for i in range(vocab_size):
        current_word = model.wv.index2word[i]
        current_word_embedding = model.wv[current_word]
        
        if current_word_embedding is not None:
            #we add one here because we want our 0 index to be entirely zeros
            #the zero index will represent words that are out of our vocabulary
            embedding_matrix[i+1, :] = current_word_embedding
        
    np.save('embedding_matrix', embedding_matrix)
        
    return embedding_matrix

embedding_matrix = make_embedding_matrix(gensim_model, vocab_size, embedding_dim)

print(embedding_matrix)

# use keras to train LSTM model
max_features = vocab_size+1
batch_size = 256

#we initialize the model with Sequential() and then add layers using Sequential.add()
lstm_model = Sequential()
#this layer converts our integers into embeddings
lstm_model.add(Embedding(vocab_size+1,
                    embedding_dim,
                    weights = [embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable = False))
#this layer adds out bidirectional LSTM model
lstm_model.add(Bidirectional(LSTM(64)))
#dropout randomly excludes nodes to prevent overfitting
lstm_model.add(Dropout(0.5))
#add a "Dense" layer of size 1 to represent our output class
lstm_model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
adam = keras.optimizers.Adam(lr = 0.0001)

lstm_model.compile(adam, 'binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()
print('Train...')
lstm_model.fit(train_encode, train_labels,
          batch_size=batch_size,
          epochs=30,
          validation_data=[val_encode, val_labels], shuffle = True)

# =============================================================================
# MODEL 3: keras convolutional neural network (CNN)
# =============================================================================


print('Build model...')
conv_model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
conv_model.add(Embedding(vocab_size+1,
                    embedding_dim,
                    weights = [embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable = False))
conv_model.add(Dropout(0.5))

#The conv1D learns filters to represent words, here we use 256 filters of size 6
#this means that the conv net is reviewing the sentiment six words at a time
#and it's creating 256 different combination of words to review
conv_model.add(Conv1D(256,
                 6,
                 padding='valid',
                 activation='relu',
                 strides=1))

# we use max pooling, this gives us the max value for each filter as it slid over the document
conv_model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
conv_model.add(Dense(300))
conv_model.add(Dropout(0.5))
conv_model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
conv_model.add(Dense(1))
conv_model.add(Activation('sigmoid'))

adam = keras.optimizers.Adam(lr = 0.0001)

conv_model.summary()

conv_model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
conv_model.fit(train_encode, train_labels,
          batch_size=256,
          epochs=20,
          validation_data=(val_encode, val_labels))


