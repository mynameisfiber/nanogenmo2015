import numpy as np
import itertools as IT

def argmaxk(arr, k):
    max_indicies = np.argsort(arr.flat)
    for i in reversed(max_indicies[-k:]):
        yield np.unravel_index(i, arr.shape)

def beam_search(model, seed, char_to_idx, idx_to_char, width=10, history=None):
    beams = np.zeros((width, len(seed), len(idx_to_char)), dtype=np.bool)
    beams_swap = np.zeros_like(beams)
    for i, c in enumerate(seed):
        print(c, char_to_idx[c])
        idx = char_to_idx[c]
        beams[:, i, idx] = 1
    sentences = [seed,]*width
    probabilities = np.ones(width)
    for i in IT.count():
        predict = model.predict(beams)
        predict *= probabilities[:,np.newaxis]
        if i == 0:
            # prime the beams
            best_predictions_idx = [(0,i[0]) for i in argmaxk(predict[0], width)]
        else:
            best_predictions_idx = list(argmaxk(predict, width))
        if history and beams_swap.shape[1] < history:
            beams_swap = np.hstack(
                beams_swap,
                np.zeros(beams.shape[:-1] + (2,), dtype=np.bool)
            )
            offset = 0
        else:
            beams_swap.fill(0)
            offset = 1
        next_sentences = []
        print(best_predictions_idx)
        for i, idx in enumerate(best_predictions_idx):
            next_sentences.append(sentences[idx[0]] + idx_to_char[idx[1]])
            probabilities[i] = predict[idx]
            beams_swap[i, :-1] = beams[idx[0], offset:]
            beams_swap[i, -1, idx[1]] = 1
        beams, beams_swap = beams_swap, beams
        sentences = next_sentences
        yield zip(probabilities, sentences, beams)
