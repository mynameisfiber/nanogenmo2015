"""
Read in an HTM file from project Gutenburg and parse the contents.  This is
done by first extracting out chapters and their respective contents and then
scoring each section.  This'll allow us to get rid of some of the Gutenburg
fluf.
"""
from lxml import html

class Book(object):
    def __init__(self, filename):
        self.filename = filename
        self.dom = html.fromstring(open(filename).read())
        self._parse()

    def _parse(self):
        """
        This function should get all the metadata we need from a given dom
        structure.  Right now all we do is get rid of the license and then find
        chapters
        """
        # remove the license
        for license in self.dom.xpath(".//pre"):
            license.getparent().remove(license)
        # find headers and parse their contents
        chapters = []
        headers = self.dom.xpath(".//*[self::h1 or self::h2 or self::h3 and p]")
        for header in headers:
            content = self._parse_header(header)
            score = self._score_content(content)
            if score > 0:
                chapters.append({
                    "header" : header, 
                    "content" : content, 
                    "score" : score,
                    "content" : [node.text_content().strip() for node in content],
                })
        self.chapters = chapters

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
            if node.tag.lower() in ("h1", "h2", "h3", "hr"):
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
        return score / float(len(content))
