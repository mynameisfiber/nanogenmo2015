from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils import generic_utils
from keras.models import model_from_json
import numpy as np

import random
import os


# dataset params
seqlen = 8
step = 2

# model params
vector_size = 4800
hidden_size = 1408
dropout = 0.5
num_layers = 7

# training sampling
batch_size = 128
samples_frac = 1.0


def save_model(model, path):
    open(path + ".json", "w+").write(model.to_json())
    model.save_weights(path + ".weights", overwrite=True)


def load_model(path):
    model = model_from_json(open(path + '.json').read())
    model.load_weights(path + '.weights')
    return model


def samples_generator(sequences, batch_size, num_samples=None,
                      X=None, y=None, seqlen=seqlen, normalize=True):
    random.shuffle(sequences)
    if X is None:
        X = np.zeros((batch_size, seqlen, vector_size), dtype=np.float)
    if y is None:
        y = np.zeros((batch_size, vector_size), dtype=np.float)
    num_samples = int(num_samples or len(sequences))
    for sample_num, seq in zip(range(num_samples), sequences):
        i = sample_num % batch_size
        for j, s in enumerate(seq[:-1]):
            X[i, j, :] = np.load(s)
        y[i, :] = np.load(seq[-1])
        if i == batch_size - 1:
            if normalize:
                X /= np.linalg.norm(X, axis=2)[..., np.newaxis]
                y /= np.linalg.norm(y, axis=1)[..., np.newaxis]
            yield X, y


def filter_files(path, files, EOP, EOC):
    for f in files:
        fullf = os.path.join(path, f)
        if f.endswith('.npy') and not (fullf in EOP or fullf in EOC):
            yield fullf

try:
    print("Trying to load model")
    model = load_model("sentence_model")
except:
    print("Creating new model")
    model = Sequential()
    model.add(LSTM(
        hidden_size,
        return_sequences=True,
        input_shape=(seqlen, vector_size)
    ))
    model.add(Dropout(dropout))
    for i in range(num_layers-2):
        model.add(LSTM(hidden_size, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(hidden_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(vector_size, activation='linear', W_regularizer=l2(0.01)))
    model.compile(loss='mse', optimizer='adam')

EOP_files = set(map(str.strip, open('./sentence_vectors/EOP.txt')))
EOC_files = set(map(str.strip, open('./sentence_vectors/EOC.txt')))
dataset = [
    list(filter_files(p, f, EOP_files, EOC_files))
    for p, d, f in os.walk('./sentence_vectors')
]
del EOP_files
del EOC_files
print("{} Books, {} Sentences".format(len(dataset), sum(map(len, dataset))))
sequences = []
for sents in dataset:
    for i in range(0, len(sents) - seqlen - 1, step):
        sequences.append(sents[i:i+seqlen+1])
print("Sequences: ", len(sequences))
valid_idx = int(len(sequences) * 0.10)
valid_sequences = sequences[:valid_idx]
train_sequences = sequences[valid_idx:]

# train the model, output generated text after each iteration
train_num_samples = int(samples_frac * len(train_sequences)) 
valid_num_samples = int(samples_frac * len(valid_sequences))
train_num_samples = (train_num_samples // batch_size) * batch_size
valid_num_samples = (valid_num_samples // batch_size) * batch_size
best_loss = 10000
iteration = 0
while True:
    iteration += 1
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print("Training")
    progbar = generic_utils.Progbar(train_num_samples)
    gen = samples_generator(train_sequences, batch_size,
                            num_samples=train_num_samples)
    for X, y in gen:
        loss, accuracy = model.train_on_batch(X, y, accuracy=True)
        progbar.add(batch_size, values=[("train loss", loss),
                    ("train acc", accuracy)])
    print()

    print("Validating")
    progbar = generic_utils.Progbar(valid_num_samples)
    gen = samples_generator(valid_sequences, batch_size,
                            num_samples=valid_num_samples)
    valid_loss = 0
    for X, y in gen:
        loss, accuracy = model.test_on_batch(X, y, accuracy=True)
        progbar.add(batch_size, values=[("valid loss", loss),
                    ("valid acc", accuracy)])
        valid_loss += loss
    print()
    valid_loss /= float(valid_num_samples)

    print("Valid Loss: {}, Best Loss: {}".format(valid_loss, best_loss))
    if valid_loss < best_loss:
        print("Saving model")
        save_model(model, "sentence_model")
        best_loss = valid_loss
