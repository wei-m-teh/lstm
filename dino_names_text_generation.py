from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

# Read input file and prints the length of the text
input_file = "dinos.txt"
with io.open(input_file, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
data_size, vocab_size = len(text), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

names = text.split('\n')
np.random.shuffle(names)
max_char = len(max(names, key=len)) + 1
# Creates a character mapper (character to index - index to character mappings in memory)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((len(names), max_char, vocab_size))
Y = np.zeros((len(names), max_char, vocab_size))

for n in range(len(names)):
   i = 0
   for c in range(len(names[n]) + 1):
      # always make the first element an non character because it's how we want to train the model.
      if c > 0:
        X[n, c, char_indices[names[n][c-1]]] = 1
        Y[n, c-1, char_indices[names[n][c-1]]] = 1
   Y[n, len(names[n]), char_indices['\n']] = 1

test_word=[]
for l in range(len(Y[0])):
    for m in range(len(Y[0][l])):
        if Y[0][l][m] == 1:
            test_word.append(indices_char[m])
print(test_word)

test_word=[]
for l in range(len(X[0])):
    for m in range(len(X[0][l])):
        if X[0][l][m] == 1:
            test_word.append(indices_char[m])
print(test_word)



# build the model network architecture: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(max_char, len(chars))))
model.add(Dense(len(chars), activation='softmax'))


# Here I am using RMSprop as the optimizer, but it could be changed for something else.
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample():
    stop = False
    cur_index = 0
    word = []
    x_pred = np.zeros((1, max_char, vocab_size))
    while not stop:
        probs = list(model.predict(x_pred)[0, cur_index])
        probs = probs / np.sum(probs)
        char_index = np.random.choice(range(vocab_size), p=probs)
        pred_char = indices_char[char_index]
        if pred_char == "\n" or cur_index == max_char:
            stop = True
            break
        word.append(pred_char)
        cur_index += 1
        x_pred[0, cur_index, char_index] = 1

    print("".join(word))

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print('----- Generating text after Epoch: %d' % epoch)
    sample()

# Training the model, then predict the next 400 characters for every opoch
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
print(model.summary(90))

model.fit(X, Y,
          batch_size=32,
          epochs=100,
          callbacks=[print_callback])