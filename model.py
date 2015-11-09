from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import generic_utils
import numpy as np

from unidecode import unidecode
import random
import sys
import string
from collections import Counter
import itertools as IT
import gc

from gutenburg import Bookshelf


'''
    At least 20 epochs are required before the generated text
    starts sounding coherent.

    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.

    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

bookshelf = Bookshelf((
    "/dataset_archive/gutenburg/pgsfcd-032007/data/", 
))
random.shuffle(bookshelf.books)

#chars_counter = Counter(
#    c
#    for book in bookshelf
#    for chapter in book.chapters 
#    for c in "\n".join(chapter['content']).lower()
#    if len(book) > min_sections
#)
#max_count = 0.01 * chars_counter.most_common(10)[-1][1]
#chars = set(char for char, count in chars_counter.most_common() if count >
#        max_count)
#max_samples = 1e7
max_samples = 5e6
chars = string.ascii_letters + r"""-;:.,?!'"()\n1234567890 """
print(chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
def parse_dataset(maxlen, step=3):
    sentences = []
    next_chars = []
    books_used = 0
    for book in bookshelf:
        if len(book) < 3 or book.meta['language'] != b'english':
            print(book.meta['language'], len(book))
            continue
        books_used += 1
        for chapter in book.chapters[1:-1]:
            text = unidecode(r"\n".join(chapter['content']))
            for i in range(0, len(text) - maxlen, step):
                if all(c in chars for c in text[i:i+maxlen+1]):
                    sentences.append(text[i: i + maxlen])
                    next_chars.append(text[i + maxlen])
        if len(sentences) >= max_samples:
            break
    print('books used: ', books_used)
    print('nb sequences:', len(sentences))
    return sentences, next_chars

maxlen = 52
step = 3
sentences, next_chars = parse_dataset(maxlen, step)
data = list(zip(sentences, next_chars))
random.shuffle(data)
sentences, next_chars = list(zip(*data))

valid_idx = int(len(sentences) * 0.10)
valid_sentences, valid_next_char = sentences[:valid_idx], next_chars[:valid_idx]
train_sentences, train_next_char = sentences[valid_idx:], next_chars[valid_idx:]

def samples_generator(sentences, next_chars, char_indices, batch_size,
        num_samples=None, X=None, y=None):
    indexes = list(range(len(sentences)))
    random.shuffle(indexes)
    if X is None:
        X = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
    if y is None:
        y = np.zeros((batch_size, len(chars)), dtype=np.bool)
    num_samples = int(num_samples or len(indexes))
    for i, idx in zip(range(num_samples), indexes):
        for t, char in enumerate(sentences[idx]):
            X[i % batch_size, t, char_indices[char]] = True
        y[i % batch_size, char_indices[next_chars[idx]]] = True
        if i > 0 and i % batch_size == 0:
            yield X, y
            X.fill(False)
            y.fill(False)

# build the model: 2 stacked LSTM
print('Build model...')
gc.collect()

hidden_size = 1536
dropout = 0.3
num_layers = 4

model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(dropout))
for i in range(num_layers-2):
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(dropout))
model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dropout(dropout))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def sample_model(model, char_indices, indices_char, seed_text, diversities=(0.2,0.5,1.0,1.2)):
    for diversity in diversities:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = seed_text
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# train the model, output generated text after each iteration
gc.collect()
batch_size = 128
samples_frac = 0.05
train_num_samples = int(samples_frac * len(train_sentences))
valid_num_samples = int(samples_frac * len(valid_sentences))
X = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
y = np.zeros((batch_size, len(chars)), dtype=np.bool)
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print("Training")
    progbar = generic_utils.Progbar(train_num_samples)
    gen = samples_generator(train_sentences, train_next_char, char_indices,
            batch_size, num_samples=train_num_samples, X=X, y=y)
    for X, y in gen:
        loss, accuracy = model.train_on_batch(X, y, accuracy=True)
        progbar.add(batch_size, values=[("train loss", loss), ("train acc",
            accuracy)])

    print("Validating")
    progbar = generic_utils.Progbar(valid_num_samples)
    gen = samples_generator(valid_sentences, valid_next_char, char_indices,
            batch_size, num_samples=valid_num_samples, X=X, y=y)
    for X, y in gen:
        loss, accuracy = model.test_on_batch(X, y, accuracy=True)
        progbar.add(batch_size, values=[("valid loss", loss), ("valid acc",
            accuracy)])

    sample_model(model, char_indices, indices_char, random.choice(sentences))

