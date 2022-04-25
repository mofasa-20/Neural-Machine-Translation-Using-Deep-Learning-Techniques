#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import random
import datetime
import nltk.translate.bleu_score as bleu

import matplotlib.pyplot as plt
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

def create_data_folder(folder_name):
    parent_dir = r'C:\Users\dell\Desktop\Deep Learning Project\Project_NMT'  ## current parent path
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
    vocab_fre = len(fre_token.word_index.keys()) + 1
    vocab_eng = len(eng_token.word_index.keys()) + 1

    ## loading callbacks
    ## Tensorboard
    log_directory = os.path.join(create_data_folder('TensorBoard'), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"log_directory for tensorboard : {log_directory}")
    tensorboard = TensorBoard(log_dir = log_directory, histogram_freq=1,  write_graph=True, write_grads=False)

    ## Early Stopping
    earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=2, restore_best_weights=True)

    epoch = 100
    embsize = 128
    batchsize = 1024
    lstm_size = 512

    #create an object of encoder_decoder Model class
    clear_session() ## clearing all previous sessions
    model  = Encoder_decoder(encoder_inputs_length = threshold, decoder_inputs_length = threshold, 
                            input_vocab_size = vocab_fre, output_vocab_size = vocab_eng,
                            emb_size = embsize, lstmsize = lstm_size)

    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy') #compile the model

    ## training the model
    history = model.fit([xtrain_fre_pad, xtrain_eng_pad_in], xtrain_eng_pad_out, batch_size = batchsize, epochs=epoch, 
                        callbacks = [tensorboard, earlystop])
    print('Training is Completed.')

    # dumping weights into h5 files
    weight_path = os.path.join(create_data_folder('weights'), 'Encoder_Decoder.h5')
    print(f"weight to save path : {weight_path}")
    model.save_weights(weight_path)

    ## creating loss plot folder
    plots_path = create_data_folder('plots') 
    loss_plot(history.history["loss"], os.path.join(plots_path, 'encoder_decoder_loss.jpeg'))

    ## get english prediction sentence output
    eng_sent_out = [predict(xval_french[num], threshold, model, fre_token, eng_token) 
                    for num in tqdm(range(0,len(xval_french)))]

    ## getting BLEU score
    score = sum([bleu.sentence_bleu(xval_english[val].lower(), eng_sent_out[val]) 
    for val in range(0, len(xval_english))])
    print('BLEU score for Encoder Decoder Model is {:.4f}.'.format(score/len(xval_english)))
    print('Encoder - Decoder modelling is completed.')