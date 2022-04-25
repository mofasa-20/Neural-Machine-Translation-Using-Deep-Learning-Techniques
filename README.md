# Neural Machine Translation Using Deep Learning Techniques

<p align="justify">
    Neural Machine Translation has become a benchmark in machine translation, and even Google Translate is using Neural Machine Translation as its backbone. Though it has achieved a remarkable feat in recent years, there are still many scopes to improve the models using various new approaches. Here I have implemented several Neural Machine Translation models using various deep learning techniques that can effectively and robustly predict the sentences from French to English language.
</p>

## Packages

Below are the python packages version details. Incase to install any package the easiet way to do it using "pip:"

```bash
python == 3.7.13
re == 2.2.1       
numpy == 1.21.6       
pandas == 1.3.5        
nltk == 3.2.5
sklearn == 1.0.2
matplotlib == 3.2.2
tqdm == 4.64.0
seaborn == 0.11.2
tensorflow == 2.8.0
keras == 2.8.0

Installation Example:

pip install tensorflow==2.8.0
```

## Features

* State-of-the-art models implemented using Encoder-Decoder, and Attention mechanism.
* Utilities to evaluate models performance and compare their accuracy using BLEU score and log loss
* Building blocks to define custom models and quickly experiment

## About Data

* The dataset is taken from the given source [data](http://www.manythings.org/anki/).
* It is a "Tab-delimited Bilingual Sentence Pairs" and present from one natural language to another natural language.
* I have selected French to English sentence pair for machine translation.

## Available Models

Name                             | Local/global | Data layout              | Architecture/method | Implementation 
---------------------------------|--------------|--------------------------|---------------------|----------------
NMT using Encoder-Decoder        | Global       | Text                     | Encoder Decoder using LSTM | [Path](provide path)|
NMT using Attention              | Global       | Text                     | Attention with Dot product of context vector | [Path](provide path)|
NMT using Attention              | Global       | Text                     | Attention with General Method (added dense layer with an extra attention unit) | [Path](provide path)|
NMT using Attention              | Global       | Text                     | Attention with concatenation of outputs (added tanh acitvation) | [Path](provide path)|

## Running on System

There are 6 files present in the code set, explanations are given below.


Name                             | Information  | 
---------------------------------|--------------|
Data Prep       | Data preparation using text preprocesssing, teacher forcing |
Encoder Decoder          | Implemenation of Encoder Decoder Model | 
Attention Dot            | Implemenation of Attention Dot Model |
Attention General              | Implementation of Attention General Model |
Attention Concat              | Implementation of Attention Concatenation Model |
Prediction Pipeline          | Prediction Pipeline Using Best model weights(Encoder Decoder Model)|

<p align="justify">

### Part 1:
* As a first step, we need to provide the file path in the data prep code to preprocess the input (French sentences) and output (English sentences) text. 
* After text preprocessing, the code will apply teacher forcing method to generate text for model input, and the generated sentences will be stored as a pickle file.
    
### Part 2:
* After data preparation, we can run any 4 models for training of input data. 
* There are 4 models which I have implemented such as Encoder Decoder, and Attention with Dot, General, Concatenation method. 
* All the models are trained at least for 100 iterations/epochs (if model doesn't overfit before reaching provided iterations), and loss plots are generated and saved in the provided locations.
* Best model weight is saved for future use.
* Model validation is done using log loss and BLEU score to check the quality of transaltion on validation datatset.
    
### Part 3:
* Prediction Pipeline is implemented using the saved weights of the best model (Encoder Decoder) which can be used as an intercative platform. 
* provide  french sentence as an input (max 22 tokens/words limitation), and get the model generated translation.
</p>

## Data Preparation:

There is an extensive data prepraration has been done on the dataset.

* Preprocessing steps involve conversion of unicode text and characters to ascii format, removal of extra spaces, emojis, and special characters present in unicode, out of context alphanumeric characters, etc.
```bash
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
```



* Teacher Forecing method has been applied. Example:
<pre>
    In French: Nous nous sommes accordés sur un prix.
    English Input to the model:    &lt;start>   We agreed on a price	
                                             |     |    | |   |
    English Input from the model:            We agreed on a price	&lt;end>
</pre>

## Model Training:

There are 4 models has been training using tensorflow/keras framework and loss scores and plots has been generated.
Example:

<pre>
model  = Encoder_decoder(encoder_inputs_length = 22, decoder_inputs_length = 22, input_vocab_size = 23492, output_vocab_size = 14544,
                         emb_size = 128, lstmsize = 512)

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy')
model.fit([train_french_sentence, train_english_sentence_in], train_english_sentence_out, batch_size = 1024, epochs=100, 
           callbacks = [tensorboard, early_stopping])
</pre>

![Log Loss Plot Encoder Decoder](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/LL-Encoder.JPG)
![Log Loss Plot Attention](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/LL-Data Dot)

## Documentation:

Provide Report

## Other resources:

[Translation Engines](https://user-images.githubusercontent.com/74071047/164985524-71b5dd60-be4c-4a0f-a210-cbc4f1253cac.png)

[ALPAC- (Automatic Language Processing Advisory Committee](https://en.wikipedia.org/wiki/ALPAC)
