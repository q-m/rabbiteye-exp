#!/usr/bin/env python
#
# Tokenizes product nut features.
#
#
# Either import in Python, or pipe a jsonlines file with product nuts, like
#
#    cat data/product_nuts.jsonl | python featurize.py >data/product_nut_features.jsonl
#
import json
import re
import sys
from unidecode import unidecode

STOPWORDS = '''
    het de deze
    en of om te hier nog ook al
    in van voor mee per als tot uit bij
    waar waardoor waarvan wanneer
    je uw ze zelf jezelf
    ca bijv bijvoorbeeld
    is bevat hebben kunnen mogen zullen willen
    gemaakt aanbevolen
    belangrijk belangrijke heerlijk heerlijke handig handige dagelijks dagelijkse
    gebruik allergieinformatie bijdrage smaak hoeveelheid
'''.split()


def clean(s):
    if s is None: return None
    # @todo keep '0.50%' and the like (or extract separately) - relevant for alcohol-free
    s = unidecode(s).strip()
    s = re.sub(r'[^A-Za-z0-9\'\s]', '', s, flags=re.MULTILINE)
    s = re.sub(r'\s+', ' ', s, flags=re.MULTILINE)
    return s

def get_brand_name(j):
    '''Return brand name from brand_name or brand_url'''
    s = j.get('brand_name', '').strip()
    if s == '':
        s = j.get('brand_url', '').strip()
        s = re.sub(r'(\Ahttps?://(www\.)?|\Awww\.|\.\w{2,3}\/?\Z)', '', s, flags=re.MULTILINE|re.IGNORECASE)
    return s


def f_name(j):
    f = clean(j.get('name', '').lower())
    # strip brand from front of name, would be twice featurized
    brand_name_clean = clean(get_brand_name(j).lower())
    if brand_name_clean != '' and f.startswith(brand_name_clean):
        f = f[len(brand_name_clean):].strip()

    if f == '': return []
    return f.split()


def f_brand(j):
    f = clean(get_brand_name(j))

    if f == '': return []
    return ['BRN:' + f]


def f_first_ingredient(j):
    if 'ingredients' not in j or len(j['ingredients']) == 0: return []

    f = j['ingredients'][0].strip().lower()

    # we're more interested in whether the ingredient is composed, than its exact content
    if re.search(r'[({:;,\n]', f, flags=re.MULTILINE):
        f = '(COMPOSED)'

    f = clean(f)

    if f == '': return []
    return ['ING:' + f]


def tokenize(j):
    '''Returns array of tokens for product nut dict'''
    tokens = f_name(j) + f_brand(j) + f_first_ingredient(j)
    tokens = filter(lambda s: s not in STOPWORDS, tokens)
    tokens = filter(lambda s: len(s) > 1, tokens)

    return tokens


def tokenize_dict(j):
    '''Returns a dict with id, tokens and optional usage_name and product_id'''
    d = {'id': j['id'], 'tokens': tokenize(j)}
    if 'usage'      in j: d['usage']      = j['usage']
    if 'product_id' in j: d['product_id'] = j['product_id']

    return d


if __name__ == '__main__':
    for line in map(str.rstrip, sys.stdin):
        j = json.loads(line)
        d = tokenize_dict(j)
        print(json.dumps(d))
