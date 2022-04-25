# Neural Machine Translation Using Deep Learning Techniques

[![Python](https://warehouse-camo.ingress.cmh1.psfhosted.org/582ab2eba9d0e0f4acbea2fd883f604349908147/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f74656e736f72666c6f772e7376673f7374796c653d706c6173746963)](https://pypi.org/project/tensorflow/2.8.0/)
[![PyPI](https://warehouse-camo.ingress.cmh1.psfhosted.org/76cd0764983d405a55b91b028b8ea467797f1816/68747470733a2f2f62616467652e667572792e696f2f70792f74656e736f72666c6f772e737667)](https://pypi.org/project/tensorflow/2.8.0/)
[![tqdm](https://warehouse-camo.ingress.cmh1.psfhosted.org/6c7e16a4732b3e24d08c464d155bde3b89d95f80/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f7471646d2e7376673f6c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465)](https://pypi.org/project/tqdm/4.64.0/)

.
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
NMT using Encoder-Decoder        | Global       | Text                     | Encoder Decoder using LSTM | [Path](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Codes/02_encoder_decoder_model.py)|
NMT using Attention Dot Product  | Global       | Text                     | Attention with Dot product of context vector | [Path](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Codes/03_attention_dot.py)|
NMT using Attention General      | Global       | Text                     | Attention with General Method (added dense layer with an extra attention unit) | [Path](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Codes/04_attention_general.py)|
NMT using Attention Concat       | Global       | Text                     | Attention with concatenation of outputs (added tanh acitvation) | [Path](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Codes/05_attention_concat.py)|

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

![Log Loss Plot Attention](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/Attention%20model.JPG)

![Log Loss Plot Data Dot Product](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/LL-Data%20Dot.JPG)

![Log Loss Plot Attention Concat](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/Attention%20model%20concat.JPG)

## Documentation:

[Report](https://github.com/mofasa-20/Neural-Machine-Translation-Using-Deep-Learning-Techniques/blob/main/Report/DS8013_Project_Report_Mohammed_Abdul_Faheem.pdf)

## Other resources:

[ALPAC- Automatic Language Processing Advisory Committee](https://en.wikipedia.org/wiki/ALPAC)

[Recursive hetero-associative memories for translation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.1968&rep=rep1&type=pdf)

[Translation Engines](https://www.academia.edu/5965803/Translation_Engines_Techniques_for_Machine_Translation_Arturo_Trujillo_Springer_Verlag_Applied_Computing_Heidelberg_1999_ISBN_1_85233_057_0111)

[JANUS: a speech-to-speech translation system using connectionist and symbolic processing Strategies](https://isl.anthropomatik.kit.edu/downloads/CP_1991_JANUS-_A_Speech-to-Speech_Translation_System_Using_Connectionist_and_Symbolic_Processing_Strategies(1).pdf)

[Recurrent Continuous Translation Models](https://aclanthology.org/D13-1176/)

[Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473#)

[Neural Computation - Long Short-Term Memory - LSTM ](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)

[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)


## Remark:
Path - Path is hardcoded in the coding so please change the path before you run the code.


