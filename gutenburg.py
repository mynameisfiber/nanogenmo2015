"""
Read in an HTM file from project Gutenburg and parse the contents.  This is
done by first extracting out chapters and their respective contents and then
scoring each section.  This'll allow us to get rid of some of the Gutenburg
fluf.
"""

from lxml import html
import os
import re
import itertools as IT
from functools import reduce
from collections import OrderedDict


_HEADINGS = ["h{}".format(i) for i in range(1, 10)]
_DELINIATIONS = ["hr", ] + _HEADINGS


def clean_string(content):
    content, _ = re.subn(r"\s+", " ", content.strip())
    return content

class Book(object):
    """
    Parses a Gutenburg project html file.  Currently only gets basic metadata
    (title/author) and the content of the book, split up into chapters.
    Currently, images are ignored.
    """
    def __init__(self, filename):
        self.filename = filename
        self.dom = html.fromstring(open(filename, 'rb').read())
        self.chapters = self._parse_chapters()
        self.meta = self._parse_meta()

    def _parse_meta(self):
        title_node = self.dom.xpath("/html/head/title")[0]
        title = clean_string(title_node.text_content())
        print(title)
        # Parse titles of the form "The Project Gutenberg eBook of Brigands of
        # the Moon, by Ray Cummings".  This is done by first eliminating words
        # starting with upper case, then words starting with lowercase, then
        # assuming the title starts followed by the author.
        match = re.search(
            r"^(The Project Gutenberg eBook of )?(?P<title>.+?[^,])([,]? by (?P<author>.+))?$", 
            title
        )
        if match is not None:
            return match.groupdict()
        else:
            return {"title" : None, "author" : None}

    def _parse_chapters(self):
        """
        This function should get all the metadata we need from a given dom
        structure.  Right now all we do is get rid of the license and then find
        chapters
        """
        # remove the license
        for license in self.dom.xpath(".//pre"):
            license.getparent().remove(license)
        # remove page labels
        for page in self.dom.xpath(".//span[@class='pagenum']"):
            page.getparent().remove(page)
        # remove comments
        for comment in self.dom.xpath(".//comment()"):
            comment.getparent().remove(comment)
        # find headers and parse their contents
        chapters = []
        headers = self.dom.xpath(".//*[" + " or ".join("self::"+h for h in _HEADINGS) + "]")
        header_queue = OrderedDict()
        for header in headers:
            if header.getparent().tag.lower() not in ("body", "div"):
                header = header.getparent()
            header_type = int(header.tag[-1])
            header_queue[header_type] = header
            content = self._parse_header(header)
            score = self._score_content(content)
            if score > 25:
                header_list = [v for k,v in header_queue.items() if k <= header_type]
                chapters.append({
                    "header" : header_list,
                    "content" : content, 
                    "score" : score,
                    "content" : [clean_string(node.text_content()) for node in content],
                })
        similar_headers = reduce(lambda a,b : a&b, (set(c['header']) for c in chapters))
        for chapter in chapters:
            chapter['header'] = [h for h in chapter['header'] if h not in similar_headers]
            chapter['title'] = ": ".join(clean_string(h.text_content()) for h in chapter['header'])
        return chapters

    def _parse_header(self, node):
        """
        Starting at a "header" node, we add all siblings of the node until we
        hit another heading or a horizontal rule.  I chose these elements to
        stop on because it looked right... may be a source of pain in the
        future.
        """
        contents = []
        while True:
            node = node.getnext()
            if node is None or node.tag.lower() in _DELINIATIONS:
                break
            contents.append(node)
        return contents

    def _score_content(self, content):
        """
        Score is a simple text/node density.  A high density means that each
        dom node contains a good amount of text.  We don't need to do the usual
        recursive shenanigans with this since Gutenburg dom structures are
        quite flat.
        """
        score = 0
        for node in content:
            score += len(node.text_content().strip()) / (1.0 + sum(1 for _ in node.iterdescendants()))
        return score / float(1 + len(content))

    def __len__(self):
        return len(self.chapters)

    def __repr__(self):
        return '<gutenburg.book, "{title}" by {author}>'.format(**self.meta)

class Bookshelf(object):
    def __init__(self, location):
        self.location = location
        self.books = self._find_books()

    def _find_books(self):
        books = []
        for dirpath, dirnames, filenames in os.walk(self.location):
            for filename in filenames:
                if filename.lower().endswith(".htm"):
                    books.append(os.path.join(self.location, dirpath, filename))
        return books
                    
    def __iter__(self):
        for book in self.books:
            yield Book(book)

