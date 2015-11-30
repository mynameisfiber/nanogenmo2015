from __future__ import print_function
from gutenburg import Bookshelf
from skipthoughts import skipthoughts
import numpy as np
from unidecode import unidecode
import progressbar as PB
import os
import time
import string
from nltk import sent_tokenize

skipthoughts_model = skipthoughts.load_model(
    data_path="/home/micha/work/tldr/skipthoughts/data/"
)

EOP = np.ones(4800)
EOC = -1 * np.ones(4800)

def null(*args, **kwargs):
    pass
print = null

def skipthoughts_encode(sentences, model=skipthoughts_model):
    vectors = None
    print(time.time(), len(sentences), "encoding")
    vectors = skipthoughts.encode(
        model, sentences, preprocess=lambda x: x,
        use_norm=False, verbose=False
    )
    print(time.time(), vectors.shape, "done")
    for i, sent in enumerate(sentences):
        if sent == "EOP":
            vectors[i, :] = EOP
        elif sent == "EOC":
            vectors[i, :] = EOC
    return vectors


def parse_sentences(bookshelf):
    for book in bookshelf:
        if len(book) <= 1 or book.meta['language'] != b'english':
            continue
        text = []
        for chapter in book.chapters:
            sentences = []
            for paragraph in chapter['content']:
                paragraph = unidecode(unicode(paragraph))
                sentences.extend(sent_tokenize(paragraph))
                sentences.append("EOP")
            text.extend(filter(None, sentences))
            text.append("EOC")
        yield book.meta['title'], text


def save_sentences(title, sentences, data_root="./sentence_vectors/"):
    print("Saving: {} ({} sentences)".format(title, len(sentences)))
    title = title.replace(' ', '_').strip(string.punctuation)
    path = os.path.join(data_root, title)
    try:
        os.mkdir(path)
    except:
        if len(os.listdir(path)) == len(sentences):
            print("Already calculated for: ", title)
            return
    vectors = skipthoughts_encode(sentences)
    for i, v in enumerate(vectors):
        vector_path = os.path.join(path, "{:06d}.npy".format(i))
        sent_path = os.path.join(path, "{:06d}.txt".format(i))
        np.save(open(vector_path, 'wb+'), v)
        open(sent_path, "w+").write(sentences[i])

            
if __name__ == "__main__":
    bookshelf = Bookshelf((
        "/dataset_archive/gutenburg/pgsfcd-032007/data/",
    ))
    book = parse_sentences(bookshelf)
    pb = PB.ProgressBar(maxval=len(bookshelf.books), widgets=[PB.Percentage(), PB.Bar(), PB.ETA()]).start()
    for title, sentences in pb(book):
        save_sentences(title, sentences)
