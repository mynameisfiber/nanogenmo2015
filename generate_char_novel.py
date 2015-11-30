from gutenburg import Bookshelf
import model
import math
from operator import itemgetter
from tqdm import tqdm
import ujson as json
import string


def find_book(bookshelf, title):
    for book in bookshelf:
        if book.meta['title'] == title:
            return book


if __name__ == "__main__":
    bookshelf = Bookshelf((
        "/dataset_archive/gutenburg/pgsfcd-032007/data/",
    ))
    book_title = "Edison's Conquest of Mars"
    book = find_book(bookshelf, book_title)
    words_per_chapter = math.ceil(50000.0 / len(book))
    print("Generating {} words per chapter for the {} chapters of '{}'".
          format(words_per_chapter, len(book), book_title))

    print("Loading Model")
    char_model = model.load_model("language_model")
    diversities = (0.2, 0.3, 0.5, 0.75, 1.0)
    punct = string.punctuation
    max_iter = words_per_chapter * 15
    generated_chapters = []
    for i, chapter in enumerate(book.chapters):
        title = chapter['title']
        chapter_data = {}
        seed = ''
        for paragraph in chapter['content']:
            seed += paragraph[:model.maxlen]
            if len(seed) >= model.maxlen:
                break
        print("\nSeed: ", seed)
        text_generator = model.sample_model2(char_model, seed, diversities)
        tqdm_param = dict(desc="{} ({})".format(title, i), leave=True)
        for num_iter, result in tqdm(enumerate(text_generator), **tqdm_param):
            for d, (text, probability) in result.items():
                if d not in chapter_data:
                    num_words = text.count(' ') + text.count('\n')
                    if num_words > words_per_chapter and text[-1] in punct:
                        chapter_data[d] = {
                            "text": text,
                            "probability": probability,
                            "diversity": d,
                        }
            if len(chapter_data) == len(diversities) or num_iter > max_iter:
                break
        content = sorted(
            chapter_data.values(),
            key=itemgetter('probability'),
            reverse=True
        )
        generated_chapters.append({
            "content": content,
            "title": chapter['title'],
        })
        json.dump(generated_chapters, open("generate_char_novel.json", "w+"))
