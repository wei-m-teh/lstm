from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

input_file = "nietzsche.txt"
with io.open(input_file, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))
