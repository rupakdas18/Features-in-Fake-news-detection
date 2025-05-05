#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:40:28 2024

@author: rupak_das18
"""


import numpy as np
import pandas as pd
import os
import matplotlib as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import torch.nn.functional as F
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize



def data_load(dataset_name,class_number,paraphraser,language,classifier):


    train_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_train.csv")
    test_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_test.csv")
    valid_df = pd.read_csv(f"Dataset/{dataset_name}/{paraphraser}/{dataset_name}_{paraphraser}_valid.csv")

    print("Shape before removing NAN values")
    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


    train_df = train_df.dropna(how='any')
    test_df = test_df.dropna(how='any')
    valid_df = valid_df.dropna(how='any')

    print("Shape after removing NAN values")

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)

    merged_df = pd.concat([train_df, test_df, valid_df], axis =0 )

    if dataset_name == 'kaggle':
        
        merged_df.reset_index(inplace = True)
        # drop the index column
        merged_df.drop(["index"], axis = 1, inplace = True)
        # Check the number of columns containing "null"

        
        merged_df['label'] = merged_df['label'].astype(int)
        train_df['label'] = train_df['label'].astype(int)
        test_df['label'] = test_df['label'].astype(int)
        valid_df['label'] = valid_df['label'].astype(int)

        label_mapping = {
        0: 0,
        1: 1,
        }

      
    
    ################################################################################################################################
    elif dataset_name == 'TALLIP':

        headers = ['domain', 'topic', 'text', 'label']
        train_df.columns = headers
        valid_df.columns = headers
        test_df.columns = headers

        train_df['label'] = train_df['label'].map(label_mapping)
        test_df['label'] = test_df['label'].map(label_mapping)
        valid_df['label'] = valid_df['label'].map(label_mapping)

    elif dataset_name == 'covid_19':
         label_mapping = {
        0: 0,
        1: 1,
        }

    
    elif dataset_name == 'liar_2' or dataset_name == 'liar_6'or dataset_name == 'liar':

        if class_number == 6:
                    label_mapping = {
                    'pants-fire': 0,
                    'false': 1,
                    'barely-true': 2,
                    'half-true': 3,
                    'mostly-true': 4,
                    'true': 5}
                    
        elif class_number == 2 and paraphraser == 'human':
            label_mapping = {
        'pants-fire': 0,
        'false': 0,
        'barely-true': 0,
        'half-true': 1,
        'mostly-true': 1,
        'true': 1
        }
            
        elif class_number == 2 and paraphraser != 'human':
             
             print("I am here")
             
             label_mapping = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1
        }
             
    train_df['label'] = train_df['label'].map(label_mapping)
    test_df['label'] = test_df['label'].map(label_mapping)
    valid_df['label'] = valid_df['label'].map(label_mapping)
    merged_df['label'] = merged_df['label'].map(label_mapping)
                    
    label_counts = merged_df['label'].value_counts()
    print(label_counts)

   
    

              
    return merged_df, train_df, test_df, valid_df
    
    

# def clean_text(text,word_count):
#     """Process text function.
#     Input:
#         tweet: a string containing a tweet
#     Output:
#         tweets_clean: a list of words containing the processed tweet
#     """

#     try:
#         lemmatizer = WordNetLemmatizer()
#         stopwords_english = stopwords.words('english')
#         text= re.sub('\[[^]]*\]', '', text)
#         # remove stock market tickers like $GE
#         text = re.sub(r'\$\w*', '', text)
#         #removal of html tags
#         review =re.sub(r'<.*?>',' ',text)
#         # remove old style retweet text "RT"
#         text = re.sub(r'^RT[\s]+', '', text)
#         # remove hyperlinks
#         text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
#         # remove hashtags
#         # only removing the hash # sign from the word
#         text = re.sub(r'#', '', text)
#         text = re.sub("["
#                             u"\U0001F600-\U0001F64F"  # removal of emoticons
#                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                             u"\U00002702-\U000027B0"
#                             u"\U000024C2-\U0001F251"
#                             "]+",' ',text)
#         text = re.sub('[^a-zA-Z]',' ',text)
#         text = text.lower()
#         text_tokens =word_tokenize(text)

#         text_clean = []
#         for word in  text_tokens:
#             if (word not in stopwords_english and  # remove stopwords
#                     word not in string.punctuation):  # remove punctuation
#                 lem_word =lemmatizer.lemmatize(word)  # lemmitiging word
#                 text_clean.append(lem_word)
#         text_mod=[i for i in text_clean if len(i)>2]
#         text_clean=' '.join(text_mod)

#         words = text_clean.split()
#         first_n_words = words[:word_count]
#         trunced_clean_text = ' '.join(first_n_words)

#         return  trunced_clean_text

#     except:
#         pass
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


def clean_text(text,word_count):
    """Process text function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    """

    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    text= re.sub('\[[^]]*\]', '', text)
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    #removal of html tags
    review =re.sub(r'<.*?>',' ',text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = re.sub("["
                        u"\U0001F600-\U0001F64F"  # removal of emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+",' ',text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    text_tokens =word_tokenize(text)

    text_clean = []
    for word in  text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            lem_word =lemmatizer.lemmatize(word)  # lemmitiging word
            text_clean.append(lem_word)
    text_mod=[i for i in text_clean if len(i)>2]
    text_clean=' '.join(text_mod)

    words = text_clean.split()
    first_n_words = words[:word_count]
    trunced_clean_text = ' '.join(first_n_words)

    return  trunced_clean_text




    