#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pickle
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


def unicode_to_ascii(sentence):
    """
    This function converts unicode into ascii format.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn')


def preprocessing(sentence):
    """
    This function normalizes the string by doing preprocessing.
    """
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r'([!.?])', r' \1', sentence)  #removing special character
    sentence = re.sub(r'[^a-zA-Z.!?]+', r' ',sentence) #keeping only letters and words
    sentence = re.sub(r'\s+', r' ', sentence)  #removing extra spaces to avoid treating as a sentence
    return sentence


def predict(input_sentence, max_len, model, fra_token, eng_token):
    """
    This function takes an input sentence and predicts the tranlated output.
    """

    batch_size = 1
    inp_text_seq = input_sentence

    #given input sentence, convert the sentence into integers using tokenizer used earlier
    tokenize_text_data = fra_token.texts_to_sequences([inp_text_seq])
    sequenced_text_data = pad_sequences(tokenize_text_data, 
                                        maxlen=max_len, 
                                        dtype='int32', 
                                        padding='post')

    #Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
    initial_states = model.layers[0].initialize_states(batch_size)
    encoder_outputs, state_h, state_c = model.layers[0](tf.constant(sequenced_text_data), initial_states)

    #Initialize index of <start> as input to decoder. and encoder final states as input_states to decoder
    #till we reach max_length of decoder or till the model predicted word <end>:
    decoder_input = tf.expand_dims([eng_token.word_index["<start>"]],0)
    out_words = list()
    states = [state_h, state_c]
    while True:
        decoder_output, decoder_state_h, decoder_state_c = model.layers[1](decoder_input, states)
        decoder_output = model.layers[2](decoder_output)
        decoder_input = tf.argmax(decoder_output, -1)

        out_words.append(eng_token.index_word[decoder_input.numpy()[0][0]])
        if out_words[-1] == "<end>":
            out_words.remove("<end>")
            break
        
        #update the input_to_decoder with current predictions
        states = [decoder_state_h, decoder_state_c]
        decoder_input = tf.expand_dims([eng_token.word_index[eng_token.index_word[decoder_input.numpy()[0][0]]]],0)

    return (" ".join(out_words))


class Encoder(tf.keras.Model):
        
    '''Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c'''
    
    def __init__ (self, inp_vocab_size, embedding_size, lstm_size, input_length):
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0

        #Initialize Embedding layer
        self.embedding = Embedding(input_dim=self.inp_vocab_size, 
                                   output_dim=self.embedding_size, 
                                   input_length=self.input_length,
                                   mask_zero=True, 
                                   name="embedding_layer_encoder")
        #Intialize Encoder LSTM layer
        self.lstm = LSTM(self.lstm_size, 
                         return_state=True, 
                         return_sequences=True, 
                         name="Encoder_LSTM")

    def call(self,input_sequence,*kwargs):

        '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
        '''
        input_embedd = self.embedding(input_sequence)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h, self.lstm_state_c
    
    def initialize_states(self,batch_size):
        
        '''
        Given a batch size it will return intial hidden state and intial cell state.
        If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
        '''
        self.batch_size = batch_size
        state_h = tf.zeros((batch_size,self.lstm_size))
        state_c = tf.zeros((batch_size,self.lstm_size))
        initial_states = [state_h, state_c]
        return initial_states 

class Decoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,out_vocab_size,embedding_size,lstm_size,input_length):

        super().__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
      
        #Initialize Embedding layer
        self.embedding = Embedding(input_dim=out_vocab_size,
                                   output_dim=embedding_size,
                                   input_length=input_length,
                                   mask_zero=True,
                                   name="embedding_layer_decoder")
        #Intialize Decoder LSTM layer
        self.lstm = LSTM(self.lstm_size, 
                         return_sequences=True, 
                         return_state=True, 
                         name="Encoder_LSTM")


    def call(self,input_sequence,initial_states,*kwargs):
        '''
        This function takes a sequence input and the initial states of the encoder.
        Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to decoder_lstm
        
        returns -- decoder_output,decoder_final_state_h,decoder_final_state_c
      '''
        target_embedd = self.embedding(input_sequence)
        decoder_output, decoder_final_state_h, decoder_final_state_c = self.lstm(target_embedd, 
                                                                                 initial_state = initial_states)
        return decoder_output, decoder_final_state_h, decoder_final_state_c
      

class Encoder_decoder(tf.keras.Model):
    
    def __init__(self, encoder_inputs_length, decoder_inputs_length, 
                 input_vocab_size, output_vocab_size, 
                 emb_size, lstmsize):
        
        super().__init__()
        #Create encoder object
        self.encoder = Encoder(inp_vocab_size = input_vocab_size, 
                               embedding_size = emb_size, 
                               input_length = encoder_inputs_length, 
                               lstm_size = lstmsize)
        
        #Create decoder object
        self.decoder = Decoder(out_vocab_size = output_vocab_size, 
                               embedding_size = emb_size, 
                               input_length = decoder_inputs_length, 
                               lstm_size = lstmsize)
        
        #Intialize Dense layer(out_vocab_size) with activation='softmax'
        self.dense = Dense(output_vocab_size, 
                           activation='softmax')
    
    
    def call(self,data, *kwargs):
        '''
        A. Pass the input sequence to Encoder layer -- Return encoder_output,encoder_final_state_h,encoder_final_state_c
        B. Pass the target sequence to Decoder layer with intial states as encoder_final_state_h,encoder_final_state_C
        C. Pass the decoder_outputs into Dense layer 
        
        Return decoder_outputs
        '''
        input_sequence, output_sequence = data[0], data[1]
        encoder_output, encoder_final_state_h, encoder_final_state_c = self.encoder(input_sequence,*kwargs)
        initial_states = [encoder_final_state_h, encoder_final_state_c]
        decoder_output,_,_ = self.decoder(output_sequence, initial_states)
        output = self.dense(decoder_output)

        return output



if __name__ == "__main__":
    
    token_dir = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT\Tokenizers' 
    ## load tokens and input data
    fra_token = pickle.load(open(os.path.join(token_dir, "french_tokens.tkn"),'rb'))
    eng_token = pickle.load(open(os.path.join(token_dir, "english_tokens.tkn"),'rb'))
    
    input_data_dir = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT\Input_Data'
    ## importing input sentences
    english_train_in = pickle.load(open(os.path.join(input_data_dir, "train_english_in.lst"),'rb'))
    english_train_out = pickle.load(open(os.path.join(input_data_dir, "train_english_out.lst"),'rb'))
    french_train = pickle.load(open(os.path.join(input_data_dir, "train_french.lst"),'rb'))
    
    threshold = 22

    ## tokenizing data
    fra_train_data = fra_token.texts_to_sequences(french_train)
    eng_train_in_data = eng_token.texts_to_sequences(english_train_in)
    eng_train_out_data = eng_token.texts_to_sequences(english_train_out)
    
    # post padding means adding 0's after sentence
    french_train = pad_sequences(fra_train_data, maxlen = threshold, dtype='int32', padding='post')
    english_train_in = pad_sequences(eng_train_in_data, maxlen = threshold, dtype='int32', padding='post')
    english_train_out = pad_sequences(eng_train_out_data, maxlen = threshold, dtype='int32', padding='post')
    
    ## getting vocabulary
    french_vocab = len(fra_token.word_index.keys()) + 1
    english_vocab = len(eng_token.word_index.keys()) + 1
    
    
    epoch = 1
    embsize = 128
    batchsize = 1024
    lstm_size = 512
    threshold = 22
    
    clear_session()

    #create an object of encoder_decoder Model class
    model  = Encoder_decoder(encoder_inputs_length = threshold, decoder_inputs_length = threshold, 
                             input_vocab_size = french_vocab, output_vocab_size = english_vocab,
                             emb_size = embsize, lstmsize = lstm_size)
    
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy') 
    
    history = model.fit([french_train[0:1], english_train_in[0:1]], english_train_out[0:1], 
                        batch_size = batchsize, epochs=epoch)

    ## loading weights
    model_path = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT\Weights\Encoder_Decoder.h5'
    model.load_weights(model_path)
    
    #taking input
    print("Please enter a french sentence which can have maximum 22 words.\nEnter: ")
    fre_input = preprocessing(input())
    
    
    #prediction
    eng_output = predict(fre_input, threshold, model, fra_token, eng_token)
    
    print(eng_output)
    
    
    
    


