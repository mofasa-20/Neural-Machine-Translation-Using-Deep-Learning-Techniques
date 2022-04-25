#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import pickle
import random
import datetime
import unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk.translate.bleu_score as bleu

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

class Encoder(tf.keras.Model):
        
    '''Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c'''
    
    def __init__ (self,inp_vocab_size,embedding_size,lstm_size,input_length):
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0

        #Initialize Embedding layer
        self.embedding = Embedding(input_dim=self.inp_vocab_size, output_dim=self.embedding_size, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        #Intialize Encoder LSTM layer
        self.lstm = LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM")

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


class Attention(tf.keras.layers.Layer):
    '''
    Class the calculates score based on the scoring_function using Bahdanu attention mechanism.
    '''
    def __init__(self,scoring_function, att_units):
        super(Attention, self).__init__()
        self.scoring_function = scoring_function
        self.att_units = att_units
        
        if self.scoring_function=='dot':
            # Intialize variables needed for Dot score function here
            pass

        if scoring_function == "general":
            # Intialize variables needed for General score function here
            self.W=tf.keras.layers.Dense(att_units)
            pass

        elif scoring_function == "concat":
            # Intialize variables needed for Concat score function here
            self.W1=tf.keras.layers.Dense(att_units)
            self.W2=tf.keras.layers.Dense(att_units)
            self.V=tf.keras.layers.Dense(1)
            pass

  
    def call(self,decoder_hidden_state,encoder_output):
        '''
        Attention mechanism takes two inputs current step -- decoder_hidden_state and all the encoder_outputs.
        * Based on the scoring function we will find the score or similarity between decoder_hidden_state and encoder_output.
        Multiply the score function with your encoder_outputs to get the context vector.
        Function returns context vector and attention weights(softmax - scores)
        '''
    
        if self.scoring_function == 'dot':
            # Implement Dot score function here
            decoder_with_time_axis = tf.expand_dims(decoder_hidden_state,2)
            score = tf.matmul(encoder_output, decoder_with_time_axis)
            pass

        elif self.scoring_function == 'general':
            # Implement General score function here
            decoder_with_time_axis = tf.expand_dims(decoder_hidden_state,2)
            score = tf.matmul(self.W(encoder_output), decoder_with_time_axis)
            pass

        elif self.scoring_function == 'concat':
            # Implement concat score function here
            """
            1. score shape == (batch_size, max_length, 1)
            2. we get 1 at the last axis because we are applying score to self.V
            3. the shape of the tensor before applying self.V is (batch_size, max_length, units)
            4. decoder hidden state shape == (batch_size, hidden size)
            5. decoder_with_time_axis shape == (batch_size, 1, hidden size)
            6. encoder_output shape == (batch_size, max_length, hidden size)
            7. we are doing this to broadcast addition along the time axis to calculate the score
            """
            decoder_with_time_axis = tf.expand_dims(decoder_hidden_state,1)
            score = self.V(tf.nn.tanh(self.W1(decoder_with_time_axis) + self.W2(encoder_output)))
            pass

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights=tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights*encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)  

        return context_vector, attention_weights


class One_Step_Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

        # Initialize decoder embedding layer, LSTM and any other objects needed
        super(One_Step_Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units

        #Initialize Embedding layer
        self.embedding = Embedding(input_dim=out_vocab_size, output_dim=embedding_dim, input_length=input_length,
                           mask_zero=True, name="embedding_layer_decoder")
        #Intialize Decoder LSTM layer
        self.lstm = LSTM(self.dec_units, return_state=True, return_sequences=True, name="Decoder_LSTM")
        #Intialize Dense Layer 
        self.dense = tf.keras.layers.Dense(out_vocab_size)
        #Intialize attention model
        self.attention = Attention(score_fun, att_units)

    
    def call(self, input_to_decoder, encoder_output, state_h, state_c):
        
        #B. Using the encoder_output and decoder hidden state, compute the context vector.
        context_vector, attention_weights = self.attention(state_h, encoder_output)

        #A. Pass the input_to_decoder to the embedding layer and then get the output(batch_size,1,embedding_dim)
        dec_output = self.embedding(input_to_decoder)
        
        #C. Concat the context vector with the step A output
        concat_output = tf.concat([tf.expand_dims(context_vector, 1), dec_output], axis=-1)
        
        #D. Pass the Step-C output to LSTM/GRU and get the decoder output and states(hidden and cell state)
        decoder_output, dec_state_h, dec_state_c = self.lstm(concat_output)
        decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))
        
        #E. Pass the decoder output to dense layer(vocab size) and store the result into output.
        output = self.dense(decoder_output)
        #F. Return the states from step D, output from Step E, attention weights from Step -B

        return output, dec_state_h, dec_state_c, attention_weights, context_vector


class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        #Intialize necessary variables and create an object from the class onestepdecoder
        super(Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.onestepdecoder = One_Step_Decoder(out_vocab_size, embedding_dim, input_length, dec_units ,
                                               score_fun, att_units)

    @tf.function  
    def call(self, input_to_decoder, encoder_output, decoder_hidden_state, decoder_cell_state):
        
        #Initialize an empty Tensor array, that will store the outputs at each and every time step
        #Create a tensor array as shown in the reference notebook
        all_outputs = tf.TensorArray(tf.float32, size=input_to_decoder.shape[1], name="output_arrays")
        
        #Iterate till the length of the decoder input
        for timestep in range(input_to_decoder.shape[1]):
            # Call onestepdecoder for each token in decoder_input
            # Store the output in tensorarray
            output, dec_state_h, dec_state_c, attention_weights, context_vector = self.onestepdecoder(input_to_decoder[:,timestep:timestep+1], encoder_output,
                                                                                                        decoder_hidden_state, decoder_cell_state)
            all_outputs = all_outputs.write(timestep, output)
        
        # Return the tensor array
        all_outputs = tf.transpose(all_outputs.stack(), [1,0,2])

        return all_outputs

class encoder_decoder(tf.keras.Model):
    
    def __init__(self, encoder_inputs_length, decoder_inputs_length, score_fun, 
                 att_units, encoder_lstm_units, decoder_lstm_units, 
                 embsize, input_vocab, output_vocab):

        super().__init__()
        #Create encoder object
        self.encoder = Encoder(inp_vocab_size = input_vocab, 
                               embedding_size=embsize, 
                               lstm_size = encoder_lstm_units, 
                               input_length=encoder_inputs_length)
        #Create decoder object
        self.decoder = Decoder(out_vocab_size = output_vocab,
                               embedding_dim=embsize, 
                               input_length=decoder_inputs_length, 
                               dec_units = decoder_lstm_units, 
                               score_fun=score_fun, 
                               att_units=att_units)

  
    def call(self, data, *kwargs):

        input_sequence, output_sequence = data[0], data[1]

        #Intialize encoder states, Pass the encoder_sequence to the embedding layer
        #Decoder initial states are encoder final states, Initialize it accordingly
        encoder_output, encoder_final_state_h, encoder_final_state_c = self.encoder(input_sequence,*kwargs)
        

        #Pass the decoder sequence,encoder_output,decoder states to Decoder
        decoder_output = self.decoder(output_sequence, encoder_output, encoder_final_state_h, 
                                      encoder_final_state_c)
        
        return decoder_output



## custom loss function
def custom_lossfunction(targets, logits):
    """
    This function takes targets and logits as input and produce loss values.
    """
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    loss_ = loss_object(targets, logits)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def predict(input_sentence, max_len, model, fra_token, eng_token):
    """
    This function takes an input sentence and predicts the tranlated output.
    """

    batch_size = 1
    attention_plot = np.zeros((max_len, max_len))

    #Given input sentence, convert the sentence into integers using tokenizer used earlier
    tokenize_text_data = fra_token.texts_to_sequences([input_sentence])
    sequenced_text_data = pad_sequences(tokenize_text_data, 
                                        maxlen=max_len, 
                                        dtype='int32', 
                                        padding='post')
    sequenced_text_data = tf.convert_to_tensor(sequenced_text_data)

    #Pass the input_sequence to encoder. we get encoder_outputs, last time step hidden and cell state
    initial_states = model.layers[0].initialize_states(batch_size)
    encoder_outputs, state_h, state_c = model.layers[0](sequenced_text_data, 
                                                        initial_states)

    #Initialize index of <start> as input to decoder. and encoder final states as input_states to onestepdecoder.    
    decoder_input = tf.expand_dims([eng_token.word_index["<start>"]],0)
    out_sent = str()
    
    #till we reach max_length of decoder or till the model predicted word <end>:
    for i in range(0,max_len):
        #predictions, input_states, attention_weights = model.layers[1].onestepdecoder(input_to_decoder, encoder_output, input_states)
        predictions, dec_state_h, dec_state_c, attention_weights, context_vector = model.layers[1].onestepdecoder(decoder_input, encoder_outputs, 
                                                                                                                  state_h, state_c, training=False)
        #Save the attention weights
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[i] = attention_weights.numpy()

        #And get the word using the tokenizer(word index) and then store it in a string.
        predicted_id = tf.argmax(predictions[0]).numpy()

        out_sent += eng_token.index_word[predicted_id] + ' '  

        if eng_token.index_word[predicted_id] == '<end>':
            out_sent = out_sent.replace('<end>','')
            break
    
        decoder_input = tf.expand_dims([predicted_id], 0)
        #state_h, state_c = dec_state_h, dec_state_c

    return out_sent, input_sentence, attention_plot


def create_data_folder(folder_name):
    parent_dir = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT' ## current parent path
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

def loss_plot(data, file_name):
    """This function plots the loss graph."""
    plt.figure(figsize = (15,5))
    plt.plot(range(1,len(data)+1),data)
    plt.title("log loss", fontsize=15)
    plt.xlabel("epochs"); plt.ylabel("loss values")
    plt.savefig(file_name)
    return plt.show()

if __name__ == "__main__":

    ## get path 
    path = create_data_folder("data")
    print(f"Input File Path : {path}")
    
    ## load training and validation data
    with open(os.path.join(path, "train_english_in.lst"), "rb") as f: xtrain_eng_in = pickle.load(f)
    with open(os.path.join(path, "train_english_out.lst"), "rb") as f: xtrain_eng_out = pickle.load(f)
    with open(os.path.join(path, "validation_english.lst"), "rb") as f: xval_english = pickle.load(f)
    with open(os.path.join(path, "train_french.lst"), "rb") as f: xtrain_french = pickle.load(f)
    with open(os.path.join(path, "validation_french.lst"), "rb") as f: xval_french = pickle.load(f)
    print("Loading Data Completed.")

    #Tokenizing data with no filters and fitting it with french data
    fre_token = Tokenizer(filters='')
    fre_token.fit_on_texts(xtrain_french)
    fra_train_tkn = fre_token.texts_to_sequences(xtrain_french)

    #Tokenizing data with no filters and fitting it with english data
    eng_token = Tokenizer(filters='')

    eng_token.fit_on_texts(xtrain_eng_in)
    eng_token.fit_on_texts(xtrain_eng_out)

    eng_train_in_tkn = eng_token.texts_to_sequences(xtrain_eng_in)
    eng_train_out_tkn = eng_token.texts_to_sequences(xtrain_eng_out)
    print("Tokenization Completed.")

    # post padding means adding 0's after sentence
    threshold = 22
    xtrain_fre_pad = pad_sequences(fra_train_tkn, maxlen = threshold, dtype='int32', padding='post') 
    xtrain_eng_pad_in = pad_sequences(eng_train_in_tkn, maxlen = threshold, dtype='int32', padding='post')
    xtrain_eng_pad_out = pad_sequences(eng_train_out_tkn, maxlen = threshold, dtype='int32', padding='post')
    print("Padding Completed.")

    ## getting vocabulary size
    french_vocab = len(fre_token.word_index.keys()) + 1
    english_vocab = len(eng_token.word_index.keys()) + 1

    ## loading callbacks
    ## Tensorboard
    log_directory = os.path.join(create_data_folder('TensorBoard'), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"log_directory for tensorboard : {log_directory}")
    tensorboard = TensorBoard(log_dir = log_directory, histogram_freq=1,  write_graph=True, write_grads=False)

    ## Early Stopping
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=2, restore_best_weights=True)

    epoch = 150
    threshold = 22
    emb_size = 512
    batchsize = 1024
    attention_units = 64

    encoder_lstm_unit = 64
    decoder_lstm_unit = 64

    #create an object of encoder_decoder Model class
    clear_session()
    model  = encoder_decoder(encoder_inputs_length = threshold, decoder_inputs_length = threshold, 
                            score_fun = "concat",  att_units = attention_units,
                            encoder_lstm_units= encoder_lstm_unit, decoder_lstm_units= decoder_lstm_unit,
                            embsize = emb_size, input_vocab = french_vocab, output_vocab = english_vocab)

    model.compile(optimizer = Adam(0.001), loss = custom_lossfunction) #compile the model


    #fitting the model and training the attention layers
    history = model.fit([xtrain_fre_pad, xtrain_eng_pad_in], xtrain_eng_pad_out, 
                        batch_size = batchsize, epochs=epoch, callbacks = [tensorboard, earlystop])

    # dumping weights into h5 files
    weight_path = os.path.join(create_data_folder('weights'), 'Attention_Concat.h5')
    print(f"weight to save path : {weight_path}")
    model.save_weights(weight_path)

    ## creating loss plot folder
    plots_path = create_data_folder('plots') 
    loss_plot(history.history["loss"], os.path.join(plots_path, 'attention_concat_loss.jpeg'))

    ## get english prediction sentence output
    eng_sent_out = [predict(xval_french[num], threshold, model, fre_token, eng_token)[0] 
                                   for num in tqdm(range(0,len(xval_french)))]

    ## getting BLEU score
    score = sum([bleu.sentence_bleu(xval_english[val].lower(), eng_sent_out[val].strip()) 
            for val in range(0, len(xval_english))])
    print('BLEU score for Attention Concatenation Model is {:.4f}.'.format(score/len(xval_english)))
    print('Attention Concatenation modelling is completed.')

