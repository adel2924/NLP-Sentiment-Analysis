#Imports
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt
import re
!pip install contractions
import contractions
import gensim
import tensorboard
# Google Drive
from google.colab import drive
# Preprocessing
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Model
from keras.models import Sequential
from keras import layers
from keras.initializers import Constant
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
# K-fold Cross-validation
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from tensorflow.keras import regularizers
# Plot_model
from keras.utils.vis_utils import plot_model
# Word2Vec
!pip install --upgrade gensim
from gensim import models
from gensim.models import Word2Vec
# EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
# Tensorboard 
from datetime import datetime
from packaging import version
from tensorflow import keras

#Load dataframes from drive
drive.mount('/content/drive')

df_train = pd.read_csv('/content/drive/My Drive/NAPIER/train.csv', names=['sentence', 'label'])
df_test = pd.read_csv('/content/drive/My Drive/NAPIER/test.csv', names=['sentence', 'label'])
df_val = pd.read_csv('/content/drive/My Drive/NAPIER/val.csv', names=['sentence', 'label'])

#Seperate by coloumns
X_train = df_train['sentence'].values
y_train = df_train['label'].values

X_test = df_test['sentence'].values
y_test = df_test['label'].values

X_val = df_val['sentence'].values
y_val = df_val['label'].values

# TOKENISING
X_train_token = []
for x in X_train:
    # Replacing "," , converting to lower and then splitting
    X_train_token.append(x.replace(","," ").lower().split())

# NORMALISING
def not_alpha(data):
    for i, array in enumerate(data):
        for j, word in enumerate(array):
            # Removing numbers and punctuations (only)
            if not word.islower():
                (data)[i][j] = ''

not_alpha(X_train_token)

def remove_punc(data):
    for i, array in enumerate(data):
        for j, word in enumerate(array):
            # Remove punctuation before and after occurrence of an alphabet char
            if re.search(r'[a-z]', word, re.I) is not None:
                (data)[i][j] = word[((re.search(r'[a-z]', word, re.I).start())):(
                            list(re.finditer(r'[a-z]', word, re.I))[-1].start() + 1)]

remove_punc(X_train_token)

def expand_contraction(data):
    for i, array in enumerate(data):
        for j, word in enumerate(array):
            # Expanding contractions
            (data)[i][j] = contractions.fix(word)

expand_contraction(X_train_token)

def split_contraction(data):
    for i, array in enumerate(data):
        for j, word in enumerate(array):
            # Splitting contraction words
            if ' ' in word:
                data[i][j] = word.split()[0]
                array.insert(j+1, word.split()[1])

split_contraction(X_train_token)

# Setting stopwords
stop = stopwords.words('english')

def remove_stopwords(data):
    for i, array in enumerate(data):
        for j, word in enumerate(array):
            # Removing stopwords
            if word in stop:
                (data)[i][j] = ''

remove_stopwords(X_train_token)

# Removing white space and one char typos
X_train_new = [[instance for instance in sublist if len(instance)>1] for sublist in(X_train_token)]

# WORD 2 VEC Model
w2v_model = gensim.models.Word2Vec(X_train_new, min_count=1, workers=4, vector_size=100, epochs=4, sg=1, window=4)

# Training word2vec model
w2v_model.train(X_train_new,epochs=10,total_examples=len(X_train_new))

# Dictionary of word2vec vectors
vocab = list(w2v_model.wv.index_to_key)

word_vec_dict={}
for word in vocab:
  word_vec_dict[word]=w2v_model.wv.get_vector(word)

# Define the tokenizer
tokenizer = Tokenizer(num_words=2000)

# Use tokenisation only on the training data!
tokenizer.fit_on_texts(X_train_new)

# tranform to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_new)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Embedding params
maxlen=50
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
embed_dim=100

# Add padding
X_train_seq = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
X_test_seq = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)
X_val_seq = pad_sequences(X_val_seq, padding='post', maxlen=maxlen)

# Creating the embedding matrix
embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tokenizer.word_index.items():
  embed_vector=word_vec_dict.get(word)
  if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
    embed_matrix[i]=embed_vector
  # if word is not found then embed_vector corresponding to that vector will stay zero.

# LSTM model
def lstm_text_classifier():
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, embeddings_initializer=Constant(embed_matrix))) #The Word2Vec embedding layer
    model.add(layers.LSTM(4, return_sequences=True, dropout=0.4))
    model.add(layers.LSTM(4,  dropout=0.4)) #Our last LSTM layer, by removing return_sequences=True,
    model.add(Flatten())
    model.add(layers.Dense(1,activation='sigmoid', kernel_regularizer='l2'))
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# TENSORFLOW GRAPH

# Load the TensorBoard notebook extension.
%load_ext tensorboard

# Clear any logs from previous runs
!rm -rf ./logs/

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Training LSTM model

early_stopping = EarlyStopping()
model = lstm_text_classifier()

training = model.fit(X_train_seq, y_train, epochs=10, verbose=True, validation_data=(X_val_seq,y_val), batch_size=1, callbacks=[early_stopping, tensorboard_callback])

# Model evaluation
loss, accuracy = model.evaluate(X_train_seq, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

print('---------------------------')

loss, accuracy = model.evaluate(X_test_seq, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Will load Tensorboard, for the model graph click "GRAPHS" on the top left, to see the sequential model double click on "Sequential" within the graph
%tensorboard --logdir logs

# Plot_model
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, embeddings_initializer=Constant(embed_matrix))) #The Word2Vec embedding layer
model.add(layers.LSTM(4, return_sequences=True, dropout=0.4))
model.add(layers.LSTM(4,  dropout=0.4)) #Our last LSTM layer, by removing return_sequences=True,
model.add(Flatten())
model.add(layers.Dense(1,activation='sigmoid', kernel_regularizer='l2'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Val loss vs accuracy
def plot_history(training):
# Plot history: MAE
  plt.plot(training.history['val_loss'], label='Validation loss')
  plt.plot(training.history['accuracy'], label='Accuracy')
  plt.title('Sentiment analysis')
  plt.ylabel('value')
  plt.xlabel('epoch')
  plt.legend(loc="lower right")
  plt.show()

plot_history(training)

# Train vs Val accuracy
def plot_accuracy(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

plot_accuracy(training)

# Train vs Val loss
def plot_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()

plot_loss(training)

# Train vs Val loss & accuracy
def plot_all(history):
  #pd.DataFrame(history.history).plot(figsize=(8, 5))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
  plt.ylabel('value')
  plt.xlabel('epoch')
  plt.grid(True)
  plt.gca().set_ylim(0,1)
  plt.show()

plot_all(training)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X_train_seq, y_train):

  # Define the model architecture
  model = Sequential()
  model.add(layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, embeddings_initializer=Constant(embed_matrix))) #The Word2Vec embedding layer
  model.add(layers.LSTM(4, return_sequences=True, dropout=0.4))
  model.add(layers.LSTM(4,  dropout=0.4)) #Our last LSTM layer, by removing return_sequences=True,
  model.add(Flatten())
  model.add(layers.Dense(1,activation='sigmoid', kernel_regularizer='l2'))
  model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(X_train_seq, y_train, epochs=10, verbose=True, validation_data=(X_val_seq,y_val), batch_size=1, callbacks=[early_stopping])

  # Generate generalization metrics
  scores = model.evaluate(X_test_seq, y_test, verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold = []
  loss_per_fold = []
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

loss, accuracy = model.evaluate(X_train_seq, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

print('---------------------------')

loss, accuracy = model.evaluate(X_test_seq, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)