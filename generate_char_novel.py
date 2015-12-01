from gutenburg import Bookshelf
import model
import math
from operator import itemgetter
from tqdm import tqdm
import ujson as json
import textwrap
import string


def find_book(bookshelf, title):
    for book in bookshelf:
        if book.meta['title'] == title:
            return book


def clean_text(text, maxlen):
    text = "*" + text[:maxlen] + "*" + text[maxlen:]
    paragraphs = text.split('\\n')
    paragraphs = map(textwrap.wrap, paragraphs)
    return "\n\n".join("\n".join(p) for p in paragraphs)


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
    if 'char_model' not in globals():
        char_model = model.load_model("language_model")
    diversities = (0.4, 0.5, 0.6)
    punct = '".!?\'\n'
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
        text_generator = model.sample_model2(
            char_model,
            seed,
            diversities=diversities
        )
        tqdm_param = dict(
            desc="{} ({})".format(title, i),
            leave=True,
            total=max_iter,
        )
        for num_iter, result in tqdm(enumerate(text_generator), **tqdm_param):
            for d, (text, probability) in result.items():
                if d not in chapter_data:
                    num_words = text.count(' ') + text.count('\\n')
                    if num_words > words_per_chapter and text[-1] in string.ascii_letters:
                        # we generate at least 1 extra character (by forcing
                        # the last character generated to be a letter) to close
                        # off any potential stray punctuation (for example when
                        # you end a paragraph with a quote, we need to generate
                        # both the . and the "
                        tmp = max(text[-5:].rfind(p) for p in punct)
                        if tmp >= 0:
                            punct_idx = len(text) - 5 + tmp + 1
                            chapter_data[d] = {
                                "text": text[:punct_idx],
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
            "seed": seed,
        })
        json.dump(generated_chapters, open("generate_char_novel.json", "w+"))
        novel_content = "\n\n\n".join(
            "## {} ({:.1f})\n\n{}".format(
                d['title'],
                d['content'][0]['diversity'],
                clean_text(d['content'][0]['text'], model.maxlen)
            )
            for d in generated_chapters
        )
        novel = "# {}\nBy: Micha Gorelick & Fast Forward Labs\n\n\n{}".format(
            book_title,
            novel_content
        )
        open("novel.md", 'w+').write(novel)
