#!/usr/bin/env python
#
# Merges data/products.jsonl and data/product_nuts.jsonl into data/product_nuts_with_product_info.jsonl
# (uses about 3GB of memory with 1M products)
#
import csv
import json
from collections import defaultdict

products      = 'data/products.jsonl'
products_nuts = 'data/product_nuts.jsonl'
result        = 'data/product_nuts_with_product_info.jsonl'

def get_data(filename):
    '''Returns lines of text file'''
    with open(filename, 'r') as f:
        return [line for line in f]

products_data = get_data(products)
products_nuts_data = get_data(products_nuts)


# takes only the entries in products_data and products_nuts_data if they are a json object
# and loads the json object so that it can be used
data_p = []
for x in products_data:
    if x[0] == '{':
        data_p.append(json.loads(x))

data_pn = []
for x in products_nuts_data:
    if x[0] == '{':
        data_pn.append(json.loads(x))

data_pn = data_pn[1:]


# - Merging the data

# makes a dict_p where for each id in data_p, so all ids with a usage, are added with their corresponding 
# product_nut_id usage and product_id.
dict_p = {}
for x in data_p:
    for _id in x['product_nut_ids']:
        _dict = {}
        _dict['product_nut_id'] = _id
        _dict['usage'] = x['usage_name']
        _dict['product_id'] = x['id']
        dict_p[_id] = _dict


# makes a list with all the data_pn items that have a usage (so appear in dict_p.keys())
l = set(dict_p.keys())

data_pn_usage = []
for x in data_pn:
    if x['id'] in l:
        data_pn_usage.append(x)


# makes the new data_pn where the usage and product_id are both new keys
new_data_pn = []
for x in data_pn_usage:
    x['usage'] = dict_p[x['id']]['usage']
    x['product_id'] = dict_p[x['id']]['product_id']
    new_data_pn.append(x)


# write the new data_pn to the path stored under result
with open(result, 'w') as f:
    f.write('\n'.join(map(json.dumps, new_data_pn)))

