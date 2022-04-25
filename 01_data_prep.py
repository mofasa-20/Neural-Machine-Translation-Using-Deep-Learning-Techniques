#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import pickle
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

def create_data_folder(folder_name):
    parent_dir = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT' 
    directory = 'project_NMT' ## folder name
    ## join path 
    path = os.path.join(parent_dir, directory)
    try: os.mkdir(path)
    except: pass

    ## create folder
    data_path = os.path.join(path, folder_name)
    try: os.mkdir(data_path)
    except: pass
    return data_path


## function to convert unicode to ascii
def unicode_to_ascii(sentence):
    """
    This function converts unicode into ascii format.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn')


## function to preprocess the test 
def preprocessing(sentence):
    """
    This function normalizes the string by doing preprocessing.
    """
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r'([!.?])', r' \1', sentence)  #removing special character
    sentence = re.sub(r'[^a-zA-Z.!?]+', r' ',sentence) #keeping only letters and words
    sentence = re.sub(r'\s+', r' ', sentence)  #removing extra spaces to avoid treating as a sentence
    return sentence

if __name__ == "__main__":

    ## get path 
    path = create_data_folder("data")
    print(f"Input File Path : {path}")

    ## extracting details from the text file
    text_file_path = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT\Data\fra.txt'
    with open(text_file_path, "r",  encoding="utf8") as f:
        data = f.readlines()
    f.close()
    print("Input Text Loading Completed.")

    ## extracting both english and french text from the text file
    sentences = [[sentence.split("\t")[0], sentence.split("\t")[1]] for sentence in data]

    ## creating a dataframe
    df = pd.DataFrame(sentences, columns=["English", "French"])

    ### getting the length of the words of each sentence
    df["french_length"] = df['French'].str.split().apply(len)
    df["english_length"] = df['English'].str.split().apply(len)

    ## keeping the threshold of words for each sentence as 22
    ## removing the sentences where words are more than 22
    threshold = 22
    mod_df = df[(df["french_length"] <= threshold) & (df["english_length"] <= threshold)]


    ## keeping test size as 20%
    xtrain, xval = train_test_split(mod_df, test_size=0.05) 

    # ## bifurcating english and french sentences into train and test datasets
    xtrain_eng = list(xtrain["English"].values); xval_eng = list(xval["English"].values)
    xtrain_fre = list(xtrain["French"].values); xval_fre = list(xval["French"].values)

    #normalize string for train data
    xtrain_french = [preprocessing(sent) for sent in xtrain_fre]
    xtrain_eng_in = ['<start> ' + preprocessing(sent) for sent in xtrain_eng]
    xtrain_eng_out = [preprocessing(sent) + ' <end>' for sent in xtrain_eng]

    #normalize string for validation data
    xval_french = [preprocessing(sent) for sent in xval_fre]
    xval_english = [preprocessing(sent) for sent in xval_eng]
    print("Data Preprocessing Completed.")

    ## dumping data as a pickle file
    pickle.dump(xtrain_eng_in, open(os.path.join(path, "train_english_in.lst"), 'wb'))
    pickle.dump(xtrain_eng_out, open(os.path.join(path, "train_english_out.lst"), 'wb'))
    pickle.dump(xval_english, open(os.path.join(path, "validation_english.lst"),'wb'))
    pickle.dump(xtrain_french, open(os.path.join(path, "train_french.lst"), 'wb'))
    pickle.dump(xval_french, open(os.path.join(path, "validation_french.lst"),'wb'))
    print("Input and Validation Text Dumped in the above printed location.")