#!/usr/bin/env python
# coding: utf-8

# # Data Jobs

# <u>TABLE OF CONTENTS:</u>
# 0. Introduction
# 1. Data Input
# 2. Data Processing/Cleaning
# 3. Data Visualization/EDA
#     * 3.1. Most Frequent Tools Per Job  
#     * 3.2. Most Frequent Skills Per Job  
#     * 3.3. Job Postings by Company
# 4. Data Predicting/Forecasting
#     * 4.1. Logistic Regression (One vs. All) w/ TFIDF
#     * 4.2. Logistic Regression (Multinomial) w/ TFIDF
#     * 4.3. Decision Tree w/ TFIDF
#     * 4.4. K-Nearest Neighbors w/ TFIDF
#     * 4.5. Naive Bayes w/ TFIDF
#     * 4.6. Choose Best Model and Optimize Hyperparameters
# 5. Conclusion

# ## 0. Introduction: 

# The purpose of this project is to examine the skills and tools desired by employers for data related jobs (i.e. *Data Analyst*, *Data Scientist*, *Data Engineer*). The motivation for the project is two-fold. First, I am personally interested in data-related careers, and the skills and tools in demand from employers. Second, while job boards are helpful in searching for jobs, there is a lack of consistency in displaying which skills/tools are desired. In other words, job boards such as Indeed and Linkedin do not have any filtering functions or ways to aggregate by skills/tools mentioned in the job advertisement. There is usually filtering functions for location, seniority, industry, etc but the filtering does not go down to the necessary level of detail for skills/tools. Furthermore, job announcements are inconsistent on where they place the text for required skills/tools. Sometimes, it is under 'Qualifications', 'Requirements', 'Skills', 'Responsibilities', or other sections. Thus, it is necessary to do some level of web scraping and text preprocessing prior to analysis.  
# 
# This project will try to answer these questions:
# - What tools/skills are most in demand for a Data Analyst?
# - What tools/skills are most in demand for a Data Engineer?
# - What tools/skills are most in demand for a Data Scientist?
# - Which companies post the most data-related job openings?
# - Can a classifier be built which predicts job role/title (Data Analyst, Data Engineer or Data Scientist) based on job description?

# ## 1. Data Input:
# 
# Data was collected from Linkedin and Indeed job sites via a custom, seperate web scraping script.

# In[70]:


# load required libraries
import os
import re
import glob
import string
import inspect
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from collections import Counter

# NLP libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize, TweetTokenizer, sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
from nltk.util import bigrams, trigrams, ngrams
from gensim.parsing.preprocessing import preprocess_documents, preprocess_string
#nltk.download('wordnet')

# sklearn libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, plot_confusion_matrix, precision_recall_curve, auc, average_precision_score, plot_precision_recall_curve
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.decomposition import PCA


# In[71]:


def get_jobs_data(filename):
    
    # get current parent directory and data folder path
    par_directory = os.path.dirname(os.getcwd())
    data_directory = os.path.join(par_directory, 'data/raw')

    # retrieve job files
    files = glob.glob(os.path.join(data_directory, filename))

    # create empty dataframe, loop over files and concatenate data to dataframe
    df_jobs = pd.DataFrame()
    for f in files:
        data = pd.read_csv(f)
        df_jobs = pd.concat([df_jobs, data], axis=0, sort='False')

    # reset index 
    df_jobs = df_jobs.reset_index(drop=True)
    
    return df_jobs
    
df_jobs = get_jobs_data('*DATA*jobs_*.csv')

# print data and length 
df_jobs.head()


# ## 2. Data Processing/Cleaning:
# 
# 

# In[72]:


# view descriptive info on dataframe
df_jobs.info()
df_jobs.describe()


# In[73]:


# clean jobs description data 
def clean_jobs(df):
    
    # clean job text data with empty list, reg expressions, and appending to list
    clean_text = []

    for x in df['job_text']:
    
        x = re.sub(r'(?<=[.,])(?=[^\s])', r' ', x) 
        x = re.sub(r'<[A-Za-z/]*\>+', '', x)
        x = re.sub(r'\\xa0', '', x)
        x = re.sub(r'\\n', '', x)    
        x = re.sub(r'Data Analyst', '', x)            
        x = re.sub(r'Data Engineer', '', x)    
        x = re.sub(r'Data Scientist', '', x) 
        
        clean_text.append(x)  
    
    df['clean_text'] = clean_text
    
    # clean date columns by filling in missing values, converting to pd datetime format
    df['date'].update(df.pop('date_scraped'))
    df['date'] = pd.to_datetime(df['date'])
    
    # clean rating columns by extracting rating string
    df['company_rating'] = [x.split('out')[0] if type(x) != float else x for x in df['company_rating']]

    # drop NA, duplicates, and uncessary columns
    df = df.drop_duplicates(subset=['clean_text', 'job_title'])
    df = df.dropna(subset=['clean_text', 'job_title'])
    df = df[df['clean_text'] != '[]']
    df = df.drop(columns=['date_posted', 'seniority_level', 'applicants', 'job_function', 'employment_type'])
        
    # filter for jobs with description length greater than 10 words
    df['job_text_length'] = df['clean_text'].apply(lambda x: len(x))
    df = df[df['job_text_length'] >= 10]  
    
    # reset dataframe index 
    df.reset_index(drop=True, inplace=True)
    
    return df
            
df_jobs = clean_jobs(df_jobs)

# print number of jobs and data sample
print('Number of Jobs: {}'.format(len(df_jobs)))
df_jobs.head()


# In[74]:


# function used to geocode locations and override timeout error
#def geocode_location(location):
#    time.sleep(1)
#    geopy = Nominatim(user_agent="my_project")
#    try:
#        return geopy.geocode(location,exactly_one=True, country_codes='us')
#    except GeocoderTimedOut:
#        return do_geocode(location)

#df_jobs['geocoded_location']=df_jobs['location'].apply(lambda x: geocode_location(x) if x != None else None)

# create latitude and longitude column from geocoded location
#df_jobs['latitude'] = df_jobs['geocoded_location'].apply(lambda x: x[1][0] if x != None else None)
#df_jobs['longitude'] = df_jobs['geocoded_location'].apply(lambda x: x[1][1] if x != None else None)
#print(df_jobs.head())


# In[75]:


# filter for job titles with Data Scientist, Data Engineer, or Data Analyst
def filter_jobs(df):
    
    # filter for data scientist, data engineer, or data analyst
    df_jobs = df[(df['job_title'].str.contains('Data Scientist', case=False)) | (df['job_title'].str.contains('Data Engineer', case=False)) | (df['job_title'].str.contains('Data Analyst', case=False))].copy()

    # add identifying column for data scientist (#0) and data engineer (#1), or data analyst (#2)
    df_jobs['label'] = df_jobs['job_title'].apply(lambda x: 0 if 'Scientist' in x or 'scientist' in x or 'SCIENTIST' in x else 1 if 'Engineer' in x or 'engineer' in x or 'ENGINEER' in x else 2 if 'Analyst' in x or 'analyst' in x or 'ANALYST' in x else '')
    df_jobs = df_jobs.dropna(subset=['label'])
    df_jobs.reset_index(inplace=True, drop=True)

    # print number of jobs and counts of each job title
    print('\nCounts of Job Titles (0=Data Scientist, 1=Data Engineer, 2=Data Analyst): \n\n{}'.format(df_jobs['label'].value_counts()))

    return df_jobs

df_jobs = filter_jobs(df_jobs)


# ## 3. Data Visualization/EDA

# ### 3.1 Most Frequent Tools per Job 

# In[76]:


# visualize job skills/tools per job title
def data_tools(df, title):
    
    # filter jobs for job title & count
    #df_jobs = df[df['label'] == label]
    df_jobs = df[(df['job_title'].str.contains(title, case=False))]
    num_jobs = len(df_jobs)     
    
    # Tokenize the article: tokens
    tokens = [word_tokenize(x) for x in df_jobs['job_text']]
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
    print(type(lemmatized))
    
    # # Create the bag-of-words: bow
    bow = Counter(lemmatized)
    
    # create dataframe from dictionary
    df_count = pd.DataFrame.from_dict(bow, orient='index').reset_index()
    df_count.columns = ['keywords', 'counts']
    df_count = df_count.sort_values(by ='counts' , ascending=False, ignore_index=True)
    df_count['avg_frequency_per_job'] = df_count['counts'] / num_jobs
    
    # create list of data tools
    data_tools = ['airflow', 'azure', 'aws', 'bi', 'bigquery', 'c', 'c++', 'd3', 'docker', 'ec2', 'excel', 'git', 'hadoop', 'hive', 
                  'java','javascript', 'jenkins','jupyter', 'kafka', 'keras', 'kuberenetes', 'linux', 'luigi', 'matlab', 'mongodb', 
                  'pearl', 'python', 'pytorch', 'r', 'react', 'redshift', 'ruby', 'sas', 'scala', 'scikit-learn', 'sql', 'spark', 'tableau', 'tensorflow']
        
    # filter keyword counts for data tools 
    df_count = df_count[df_count['keywords'].isin(data_tools)]
    df_count.reset_index(drop=True, inplace=True)
    
    # Print the 20 most common tools
    print('\n' + title + ' Top Keywords:\n\n', df_count.iloc[:20])
    
    # plot the 20 most common tools
    sns.set()
    fig, ax = plt.subplots(dpi=300) #, figsize = (3, 5), )   
    sns.barplot(x="avg_frequency_per_job", y="keywords", data=df_count.iloc[:20], palette="Blues_d")
    plt.title(label = "'" + title + "'"  + ' Keywords on Linkedin/Indeed', fontsize=13)
    plt.xlabel('Frequency (per job posting)', fontsize=8)
    plt.ylabel('Keywords', fontsize=8)
    
    # ax.set(title = "'" + title + "'"  + ' Keywords on Linkedin/Indeed')
    plt.show()


# In[77]:


data_tools(df_jobs, 'Data Analyst')


# In[78]:


data_tools(df_jobs, 'Data Engineer')


# In[79]:


data_tools(df_jobs, 'Data Scientist')


# ### 3.2 Most Frequent Skills per Job 

# In[80]:


# preprocess job description string 
def preprocess_text(df):
    
    df = df.copy()

    processed_text = []
    
    for x in df['clean_text']:

        # # Convert the tokens into lowercase: lower_tokens
        lower_tokens = [x.lower()]
        
        # # Convert tokens
        tokenize = [word_tokenize(x) for x in lower_tokens]
  
        # # # Retain alphabetic words: alpha_only
        alpha_only = [i for item in tokenize for i in item if i.isalpha()]

        # set stop words
        stop_words = set(stopwords.words('english')) 

        # # Remove all stop words: no_stops
        no_stops = [i for i in alpha_only if i not in stop_words]

        # # Instantiate the WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()

        # # Lemmatize all tokens into a new list: lemmatized
        lemmatized = [wordnet_lemmatizer.lemmatize(i) for i in no_stops]

        processed_text.append(lemmatized)
    
    df['processed_text'] = processed_text
    
    return df
            
df_jobs = preprocess_text(df_jobs)

df_jobs.head()


# In[81]:


def ngram_generator(df, n, title):
    
    # filter for job title
    df_jobs = df[(df['job_title'].str.contains(title, case=False))]
    
    # create empty list
    ngram_list = []

    # loop over every row in df_jobs['clean_text']
    for x in df_jobs['processed_text']:
        
        # join each list of strings into sentence
        joined = [' '.join(x)]        
        
        # create list of trigrams within each row
        ngram = [list(ngrams(item.split(), n)) for item in joined]

        # append list of trigrams to empty list
        ngram_list.append(ngram)
    
    # extract item within each sublist
    ngram_list = [item for sublist in ngram_list for item in sublist]
    
    # extract each item within sublist again for Counter (list is unhashable)
    ngram_list = [item for sublist in ngram_list for item in sublist]
    
    # print the 20 most common grams via Counter function
    
    return list(Counter(ngram_list).most_common(20))


# ### Data Analyst

# In[82]:


ngram_generator(df_jobs, 2, 'Data Analyst')


# ### Data Engineer

# In[83]:


ngram_generator(df_jobs, 2, 'Data Engineer')


# ### Data Scientist

# In[84]:


ngram_generator(df_jobs, 2, 'Data Scientist')


# ### 3.3 Most Frequent Companies Posting Jobs 

# In[85]:


def job_companies(df, title):
    
    # replace various Amazon company names with Amazon
    df['company'] = df['company'].replace(to_replace = 'Amazon Web Services (AWS)', value = 'Amazon')
    df['company'] = df['company'].replace(to_replace = 'Amazon.com Services LLC', value = 'Amazon')
    df['company'] = df['company'].replace(to_replace = 'Amazon Web Services, Inc.', value = 'Amazon')

    # print value counts for companies to determine top 20 companies for each type of job posting
    title_companies = df[(df['job_title'].str.contains(title, case=False))]['company'].value_counts().iloc[:20]   
    print('\nMost Frequent Companies for {} Job Postings: \n\n{}'.format(title, title_companies))
 
    # plot most frequent companies
    sns.set()
    fig, ax = plt.subplots(figsize = (12,8), dpi=250)
    sns.barplot(x = title_companies.iloc[:10].index, y= title_companies.iloc[:10].values, palette="Blues_d")
    plt.title(label = "Most Frequent Companies for " + title + " Jobs", fontsize=14) 
    plt.xlabel('Companies', fontsize=14)
    plt.ylabel('No. of Postings', fontsize=14)
    plt.xticks(rotation=45, fontsize=12) 
    plt.show()


# ### Data Analyst

# In[86]:


job_companies(df_jobs, 'Data Analyst')


# ### Data Engineer

# In[87]:


job_companies(df_jobs, 'Data Engineer')


# ### Data Scientist

# In[88]:


job_companies(df_jobs, 'Data Scientist')


# ## 4. Data Prediction

# In[89]:


# # make series of job text for just Data Engineer and Data Scientist roles
def remove_obvious_words(df):
    
    updated_words = []

    for x in df['processed_text']:
        
        x = str(x)
        x = x.replace('scientist', '')
        x = x.replace('engineer', '')
        x = x.replace('analyst', '')
       
        # append updated strings to list
        updated_words.append(x)

    df['processed_text'] = updated_words
    
    return df
            
df_jobs = remove_obvious_words(df_jobs)
df_jobs.head()


# In[90]:


# create feature and target series from dataframe
df_feature = df_jobs[(df_jobs['label'] == 0) | (df_jobs['label'] == 1) | (df_jobs['label'] == 2)].loc[:, 'processed_text']
df_target = df_jobs[(df_jobs['label'] == 0) | (df_jobs['label'] == 1) | (df_jobs['label'] == 2)].loc[:, 'label']                


# In[91]:


# apply TF-IDF based feature representation
tfidf_vectorizer = TfidfVectorizer()
df_feature_TFIDF = tfidf_vectorizer.fit_transform(df_feature)

# split train/test data 80/20
X_train, X_test, y_train, y_test = train_test_split(df_feature_TFIDF, 
                                                    df_target, 
                                                    train_size=0.8, 
                                                    random_state=20)


# In[92]:


# create empty dataframe to store model results and scores
model_results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1'])
model_results


# ### 4.1 Logistic Regression (One vs. All) w/ TFIDF (Model 1)

# In[93]:


# define logistic regression, fit to model 
logreg_OVR_TFIDF = LogisticRegression(multi_class = 'ovr')
#log_reg_CV = LogisticRegression(solver='liblinear', penalty='l1')
logreg_OVR_TFIDF.fit(X_train, y_train)


# In[94]:


# compute y-prediction and accuracy, recall, precision, and f1 scores
y_pred = logreg_OVR_TFIDF.predict(X_test)
mod1_acc = accuracy_score(y_test, y_pred)
mod1_recall = recall_score(y_test, y_pred, average='macro')
mod1_precision = precision_score(y_test, y_pred, average='macro')
mod1_f1 = f1_score(y_test, y_pred, average='macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 1 accuracy: ', round(mod1_acc,3))
print('Model 1 recall: ', round(mod1_recall,3))
print('Model 1 precision: ', round(mod1_precision,3))
print('Model 1 f1 : ', round(mod1_f1,3))
print('Model 1 classification report: \n',classification_report(y_test, y_pred))


# In[95]:


# append model results to dataframe
    # append to dataframe
model_results = model_results.append({'model': 'LogReg_OVR_TFIDF',
                      'accuracy':round(mod1_acc,3),
                      'recall': round(mod1_recall,3),
                      'precision': round(mod1_precision,3),
                      'f1': round(mod1_f1,3)}, ignore_index=True)

model_results


# In[96]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(logreg_OVR_TFIDF, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('LogReg_OVR_TFIDF Confusion Matrix')

plt.grid(None)
plt.show()


# ### 4.2 Logistic Regression (Multinomial) w/ TF-IDF (Model 2)

# In[97]:


# define logistic regression, fit to model                                                    
logreg_multi_TFIDF = LogisticRegression(multi_class = 'multinomial')
logreg_multi_TFIDF.fit(X_train, y_train)


# In[98]:


# compute y-prediction and accuracy, recall, precision, and f1 scores
y_pred = logreg_multi_TFIDF.predict(X_test)
mod2_acc = accuracy_score(y_test, y_pred)
mod2_recall = recall_score(y_test, y_pred, average = 'macro')
mod2_precision = precision_score(y_test, y_pred, average = 'macro')
mod2_f1 = f1_score(y_test, y_pred, average = 'macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 2 accuracy: ', round(mod2_acc,3))
print('Model 2 recall: ', round(mod2_recall,3))
print('Model 2 precision: ', round(mod2_precision,3))
print('Model 2 f1 : ', round(mod2_f1,3))
print('Model 2 classification report: \n',classification_report(y_test, y_pred))


# In[99]:


# append model results to dataframe
model_results = model_results.append({'model': 'LogReg_Multi_TFIDF',
                      'accuracy':round(mod2_acc,3),
                      'recall': round(mod2_recall,3),
                      'precision': round(mod2_precision,3),
                      'f1': round(mod2_f1,3)}, ignore_index=True)

model_results


# In[100]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(logreg_multi_TFIDF, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('LogReg_Multi_TFIDF Confusion Matrix')

plt.grid(None)
plt.show()


# In[101]:


# get importance
importance = logreg_multi_TFIDF.coef_[0]

# create empty dataframe
feature_list = []

# summarize feature importance
for i,v in enumerate(importance):
    if v > 0.5:
        feature_list.append(dict({'number': i, 'feature': tfidf_vectorizer.get_feature_names()[i], 'weight': v})) 

    if v < -0.5:
        feature_list.append(dict({'number': i, 'feature': tfidf_vectorizer.get_feature_names()[i], 'weight': v}))        
        
df_important_features = pd.DataFrame(feature_list, columns=['number', 'feature', 'weight'])
print('Top Words (Features) for Predicting Data Scientist job: \n\n', df_important_features.sort_values(by='weight', ascending=False).iloc[:10])
print('\nTop Words (Features) for Predicting Data Engineering job: \n\n', df_important_features.sort_values(by='weight', ascending=True).iloc[:10])


# ### 4.3 Decision Tree w/ TF-IDF (Model 3)

# In[102]:


# train a Decision Tree Classifier 
dtree_TFIDF = DecisionTreeClassifier(max_depth = 3)
dtree_TFIDF.fit(X_train, y_train) 
y_pred = dtree_TFIDF.predict(X_test) 


# In[103]:


# compute y-prediction and accuracy, recall, precision, and f1 scores
y_pred = dtree_TFIDF.predict(X_test) 
mod3_acc = accuracy_score(y_test, y_pred)
mod3_recall = recall_score(y_test, y_pred, average = 'macro')
mod3_precision = precision_score(y_test, y_pred, average = 'macro')
mod3_f1 = f1_score(y_test, y_pred, average = 'macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 3 accuracy: ', round(mod3_acc,3))
print('Model 3 recall: ', round(mod3_recall,3))
print('Model 3 precision: ', round(mod3_precision,3))
print('Model 3 f1 : ', round(mod3_f1,3))
print('Model 3 classification report: \n',classification_report(y_test, y_pred))


# In[104]:


# append model results to dataframe
model_results = model_results.append({'model': 'DecisionTree_TFIDF',
                      'accuracy':round(mod3_acc,3),
                      'recall': round(mod3_recall,3),
                      'precision': round(mod3_precision,3),
                      'f1': round(mod3_f1,3)}, ignore_index=True)

model_results


# In[105]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(dtree_TFIDF, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('DecisionTree_TFIDF Confusion Matrix')

plt.grid(None)
plt.show()


# ### 4.4 K-Nearest Neighbors w/ TF-IDF (Model 4)

# In[106]:


# setup plot to determine optimal number of neighbors
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors= k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[107]:


# train an KNN Classifier 
KNN_TFIDF = KNeighborsClassifier(n_neighbors = 3)
KNN_TFIDF.fit(X_train, y_train) 


# In[108]:


# compute y-prediction and accuracy, recall, precision, and f1 scores
y_pred = KNN_TFIDF.predict(X_test)
mod4_acc = accuracy_score(y_test, y_pred)
mod4_recall = recall_score(y_test, y_pred, average = 'macro')
mod4_precision = precision_score(y_test, y_pred, average = 'macro')
mod4_f1 = f1_score(y_test, y_pred, average = 'macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 4 accuracy: ', round(mod4_acc,3))
print('Model 4 recall: ', round(mod4_recall,3))
print('Model 4 precision: ', round(mod4_precision,3))
print('Model 4 f1 : ', round(mod4_f1,3))
print('Model 4 classification report: \n',classification_report(y_test, y_pred))


# In[109]:


# append model results to dataframe
model_results = model_results.append({'model': 'KNN_TFIDF',
                      'accuracy':round(mod4_acc,3),
                      'recall': round(mod4_recall,3),
                      'precision': round(mod4_precision,3),
                      'f1': round(mod4_f1,3)}, ignore_index=True)

model_results


# In[110]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(KNN_TFIDF, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('KNN_TFIDF Confusion Matrix')

plt.grid(None)
plt.show()


# ### 4.5 Naive Bayes w/ TF-IDF (Model 5)

# In[111]:


# convert to non-sparse matrix for GaussianNB
X_train = X_train.todense()
X_test = X_test.todense()

# train a GaussianNB Classifier 
GaussianNB_TFIDF = GaussianNB()
GaussianNB_TFIDF.fit(X_train, y_train) 


# In[112]:


# compute y-prediction and accuracy, recall, precision, and f1 scores
y_pred = GaussianNB_TFIDF.predict(X_test)
mod5_acc = accuracy_score(y_test, y_pred)
mod5_recall = recall_score(y_test, y_pred, average = 'macro')
mod5_precision = precision_score(y_test, y_pred, average = 'macro')
mod5_f1 = f1_score(y_test, y_pred, average = 'macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 5 accuracy: ', round(mod5_acc,3))
print('Model 5 recall: ', round(mod5_recall,3))
print('Model 5 precision: ', round(mod5_precision,3))
print('Model 5 f1 : ', round(mod5_f1,3))
print('Model 5 classification report: \n',classification_report(y_test, y_pred))


# In[113]:


# append model results to dataframe
model_results = model_results.append({'model': 'GaussianNB_TFIDF',
                      'accuracy':round(mod5_acc,3),
                      'recall': round(mod5_recall,3),
                      'precision': round(mod5_precision,3),
                      'f1': round(mod5_f1,3)}, ignore_index=True)

model_results


# In[114]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(GaussianNB_TFIDF, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('GaussianNB_TFIDF Confusion Matrix')

plt.grid(None)
plt.show()


# ### 4.6 Choose Best Model and Optimize Hyperparameters

# In[115]:


# split train/test data 80/20
X_train, X_test, y_train, y_test = train_test_split(df_feature_TFIDF, 
                                                    df_target, 
                                                    train_size=0.8, 
                                                    random_state=20)

# scale data
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[116]:


# choose parameters
tuned_parameters = [{'estimator__C': [100, 10, 1, 0.1, 0.01]}]

# find optimal C by grid search and fit 
logreg_OVR = OneVsRestClassifier(LogisticRegression(max_iter=1000))
grid = GridSearchCV(logreg_OVR, tuned_parameters, scoring = 'f1_weighted', verbose=2, cv=3)
grid.fit(X_train, y_train)

# print best score/parameter
print(grid.best_score_)
print(grid.best_params_)

# compute y-prediction
y_pred = grid.predict(X_test)


# In[117]:


mod6_acc = accuracy_score(y_test, y_pred)
mod6_recall = recall_score(y_test, y_pred, average = 'macro')
mod6_precision = precision_score(y_test, y_pred, average = 'macro')
mod6_f1 = f1_score(y_test, y_pred, average = 'macro')

# print accuracy, recall, precision, f1 score, detailed report
print('---------------------------------------')
print('Model 6 accuracy: ', round(mod6_acc,3))
print('Model 6 recall: ', round(mod6_recall,3))
print('Model 6 precision: ', round(mod6_precision,3))
print('Model 6 f1 : ', round(mod6_f1,3))
print('Model 6 classification report: \n',classification_report(y_test, y_pred))


# In[118]:


# append model results to dataframe
model_results = model_results.append({'model': 'LogReg_OVR_TFIDF_optimized',
                      'accuracy':round(mod6_acc,3),
                      'recall': round(mod6_recall,3),
                      'precision': round(mod6_precision,3),
                      'f1': round(mod6_f1,3)}, ignore_index=True)

model_results = model_results.sort_values(by='accuracy')
model_results


# In[120]:


fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

plot_confusion_matrix(grid, X_test, y_test,
                             display_labels=['Data Scientist', 'Data Engineer', 'Data Analyst'],
                             cmap=plt.cm.Blues, ax=ax)
ax.set_title('LogReg_OVR_TFIDF_optimized Confusion Matrix')

plt.grid(None)
plt.show()


# ## 5. Conclusion

# This project reviewed data-related job postings and tried to answer the following questions. The answers and results are shown below:
# 
# ### **What tools/skills are most in demand for a *Data Analyst?***
# 
# For a Data Analyst, the top desired tools are: *sql, excel, tableau, python, and r*. The top desired skills (after parsing through some useless bigrams) are: *data analysis, communication skill, and data visualization*. See section 3.1 for the full list and plot of tools/skills frequency. 
# 
# ### **What tools/skills are most in demand for a *Data Engineer?***
# 
# For a Data Engineer, the top desired tools are: *sql, python, aws, spark, and azure*. The top desired skills (after parsing through some useless bigrams) are: *data pipeline, big data, and data warehouse*. See section 3.1 for the full list and plot of tools/skills frequency. 
# 
# ### **What tools/skills are most in demand for a *Data Scientist?***
# 
# For a Data Scientist, the top desired tools are: *python, r, sql, spark, and tableau*. The top desired skills (after parsing through some useless bigrams) are: *machine learning, data analysis, and communication skill*. See section 3.1 for the full list and plot of tools/skills frequency. 
# 
# ### **Which companies post the mose job openings?**
# 
# For Data Analyst postings, the top companies are: *ClearedJobs.Net, GEICO, CyberCoders, Booz Allen Hamilton, and Apex Systems*. See section 3.3 for the full list and plot of job company frequency. 
# 
# For Data Engineer postings, the top companies are: *Amazon, Optello, Facebook, Apple, and CyberCoders*. See section 3.3 for the full list and plot of job company frequency. 
# 
# For Data Scientist postings, the top companies are: *Amazon, Facebook, Booz Allen Hamilton, Apple, and Optello*. See section 3.3 for the full list and plot of job company frequency. 
# 
# ### **Can a classifier be built which predicts job role/title (Data Analyst, Data Scientist, or Data Engineer) based on job description?**
# 
# Five different initial models were chosen for the multi-class (three class) classification problem. These models included the following: Logisitic Regression (One vs. All), Logistic Regression (Multinomial), K-Nearest Neighbor, Decision Tree, and Naive Bayes. Based on an initial fitting of the model types to the classification problem and collecting model metrics, the Logistic Regression model performed the best. This model type was then further optimized via a hyperpareter grid search. The resulting best model, LogReg_OVR_TFIDF_optimized, is shown below with model performace metrics
# 
# | Model | Accuracy | Precision | Recall | f1
# | :- | :-: | :-: | :-: | :-: |
# | GaussianNB_TFIDF | 0.613 | 0.660 | 0.643 | 0.611 
# | DecisionTree_TFIDF | 0.648 | 0.697 | 0.599 | 0.611 
# | KNN_TFIDF | 0.716 | 0.802 | 0.680 | 0.697 
# | LogReg_OVR_TFIDF | 0.872 | 0.863 | 0.853 | 0.857 
# | LogReg_Multi_TFIDF | 0.880 | 0.870 | 0.865 | 0.867 
# | LogReg_OVR_TFIDF_optimized | 0.891 | 0.882 | 0.873 | 0.877 
# 
# Next steps for improving model performance include trying additional model types, additional hyperparameter tuning (e.g. more solvers, more C estimators, more penalty terms, etc), review imbalanced classes and upsample/downsample appropriately, and try different NLP cleaning approaches (e.g. add more stopwords). 
# 
