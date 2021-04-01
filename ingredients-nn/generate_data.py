#!/usr/bin/python3
#
# Generates training/test/validation data from listed ingredient txt + html files.
#
# TODO
# - perhaps keep newlines during tokenization (sometimes used as section separator)
#
import os
import sys
import html.parser
from nltk.tokenize import RegexpTokenizer

def debug(s):
    #sys.stderr.write(s)
    #sys.stderr.write('\n')
    pass

class CustomTokenizer(RegexpTokenizer):
    def __init__(self):
        # almost the same as WordPuncTokenizer
        RegexpTokenizer.__init__(self, r"\w+|[^\w\s]")

class HTMLIngredientsParser(html.parser.HTMLParser):
    def feed_file(self, filename):
        self.result = []
        self._cur_text = ''
        self._cur_parsed = []
        self._next_result()
        with open(filename) as f:
            while True:
                chunk = f.read(1024)
                if not chunk: break
                self.feed(chunk)

    def handle_starttag(self, tag, attrs):
        debug('handle_starttag("%s", "%s") lvl=%d' % (tag, attrs, self._cur_tag_lvl))
        cls = [a[1] for a in attrs if a[0] == 'class'][0].split(r'\s+')
        self._cur_tag_lvl += 1
        self._cur_tag_cls.append(cls)
        if 'root' in cls:
            self._next_result()
            self._cur_tag_lvl = 1
        elif 'name' in cls:
            self._next_tag()
            self._cur_tag = 'INGR'
        elif 'amount' in cls:
            self._next_tag()
            self._cur_tag = 'AMNT'
        elif 'note' in cls:
            self._next_tag()
            self._cur_tag = 'NOTE'
        else:
            self._next_tag()
            self._cur_tag = 'UNKN'

    def handle_data(self, data):
        debug('handle_data("%s")' % (data))
        self._cur_text += data

    def handle_endtag(self, tag):
        debug('handle_endtag("%s")' % (tag))
        self._cur_tag_lvl -= 1
        self._cur_tag_cls.pop()
        if self._cur_tag_lvl == 0:
            self._next_result()
        else:
            self._next_tag()

    def _next_tag(self):
        if self._cur_text:
            debug('_next_tag(), _cur_tag="%s", _cur_text="%s"' % (self._cur_tag, self._cur_text))
            new_pos = self._cur_pos + len(self._cur_text)
            self._cur_parsed.append({ 'text': self._cur_text, 'start': self._cur_pos, 'end': new_pos, 'tag': self._cur_tag })
            self._cur_pos = new_pos
        self._cur_text = ''
        self._cur_tag = 'PUNC'

    def _next_result(self):
        debug('_next_result(), parsed=%s' % (self._cur_parsed))
        if len(self._cur_parsed) > 0:
            self.result.append(self._cur_parsed)
        self._next_tag()
        self._cur_pos = 0
        self._cur_tag_lvl = 0
        self._cur_tag_cls = [['body']]
        self._cur_parsed = []

        #debug('  _cur_pos=%s, _cur_tag_lvl=%s, _cur_tag_cls=%s, _cur_parsed=%s, _cur_text=%s, _cur_tag=%s'%(self._cur_pos, self._cur_tag_lvl, self._cur_tag_cls, self._cur_parsed,self._cur_text, self._cur_tag))

def clean_parsed_text_newline(r):
    # remove trailing newline token
    return r[0:-2] if r[-1]['text'] == '\n' else r

def clean_parsed_text_whitespace(r):
    # remove whitespace at start and end of string, correcting start and end positions
    # does not handle whitespace in the middle of tokens (not really expected)
    def filterfun(s, right=True):
        t = { 'text': s['text'], 'start': s['start'], 'end': s['end'], 'tag': s['tag'] }
        pos_attr = 'end' if right else 'start'
        t['text'] = s['text'].rstrip() if right else s['text'].lstrip()
        t[pos_attr] += (-1 if right else 1) * (len(s['text']) - len(t['text']))
        return t
    new = [filterfun(filterfun(s, False), True) for s in r]
    return [n for n in new if n['text'] != '']

def combine_tokens(tokenized, parsed):
    # Parsed may contain larger spans than tokenized. This method looks up tags in parsed
    # for each token in tokenized.
    r = []
    for token in tokenized:
        start, end = token['start'], token['end']
        parsed_tokens = [p for p in parsed if start >= p['start'] and end <= p['end']]
        if len(parsed_tokens) == 1:
            parsed_token = parsed_tokens[0]
            r.append({ 'text': token['text'], 'start': start, 'end': end, 'tag': parsed_token['tag'] })
        elif len(parsed_tokens) == 0:
            r.append({ 'text': token['text'], 'start': start, 'end': end, 'tag': None })
        elif len(parsed_tokens) > 1:
            sys.stderr.write('Unexpected condition: multiple parsed tokens for single tokenized for "%s" at position %d\n' % (token['text'], start))
            continue
    return r

def print_results(tokenized_texts, parsed_texts, combined_texts):
    for tokens, parsed, combined in zip(tokenized_texts, parsed_texts, combined_texts):
        print('---')
        print('_'.join([t['text'] for t in tokens]))
        print('_'.join([t['text'] for t in parsed]))
        print('_'.join([t['tag'] or '?' for t in parsed]))
        print()
        print('_'.join([t['text'] for t in combined]))
        print('_'.join([t['tag'] or '?' for t in combined]))

def write_tf_ner(base, combined_texts):
    fwords = open(base + '.words.txt', 'w')
    ftags = open(base + '.tags.txt', 'w')
    for combined in combined_texts:
        fwords.write(' '.join([t['text'] for t in combined]) + '\n')
        ftags.write(' '.join([t['tag'] or '?' for t in combined]) + '\n')

def print_pos(combined_texts):
    for combined in combined_texts:
        print(' '.join(['%s/%s' % (t['text'], t['tag']) for t in combined]))

def main(action, base):
    # read full ingredient texts
    full_texts = [s.rstrip('\r\n') for s in open(base + '.txt').readlines()]
    # tokenize
    tokenizer = CustomTokenizer()
    tokenized_texts = []
    for text in full_texts:
        r = []
        for start, end in tokenizer.span_tokenize(text):
            r.append({ 'text': text[start:end], 'start': start, 'end': end })
        tokenized_texts.append(r)

    # read parsed ingredients as tokens
    htmldoc = HTMLIngredientsParser(convert_charrefs=True)
    htmldoc.feed_file(base + '.html')
    parsed_texts = htmldoc.result
    # cleanup parsed ingredient tokens
    parsed_texts = [clean_parsed_text_newline(s) for s in parsed_texts]
    parsed_texts = [clean_parsed_text_whitespace(s) for s in parsed_texts]

    # match html-based with text-based tokens
    combined_texts = [combine_tokens(t, p) for t, p in zip(tokenized_texts, parsed_texts)]

    if action == 'info':
        print_results(tokenized_texts, parsed_texts, combined_texts)
    elif action == 'tf_ner':
        write_tf_ner(base, combined_texts)
    elif action == 'pos':
        print_pos(combined_texts)
    else:
        sys.stderr.write('Unknown action: %s\n' % (action))
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s <info|tf_ner|pos> <basename>\n' % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
