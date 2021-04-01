# Ingredient parsing

We developed a [rule-based food ingredient declaration parser](https://github.com/q-m/food-ingredient-parser-ruby), which
works fine for around 80% of the ingredient declarations. Because there are so many errors in these ingredient declarations,
a rule-based parser is not very suitable for the remaining 20% (and sometimes a typo can cause the parser to come to a wrong
conclusion).

A next step would be to use machine learning for parsing the ingredients. The existing rule-based parser can be used to
generate training data, augmented with human annotation.

This is very much a work in progress.

There can be multiple approaches:
- Sequence-to-sequence, where for each character the model determines if it is an ingredient, amount, etc.
- Tagging, where the ingredients are first tokenized and the model determines for each token what it is.

## Generating training data (tagging)

1. put a list of ingredient declarations in a .txt file (one per line), e.g. `train.txt`
2. convert them to HTML:
   `gem install food_ingredient_parser`
   `food_ingredient_parser --html -f train.txt > train.html`
3. generate tagged data, one of:
   - `python3 tf_ner train` - generates `train.words.txt` and `train.tags.txt` e.g. for [tf_ner](https://github.com/guillaumegenthial/tf_ner)
   - `python3 pos train` - prints as POS corpus e.g. for use with [NLTK-Trainer](https://nltk-trainer.readthedocs.io/en/latest/train_tagger.html)

Note that currently the script is very limited. Things to improve:
- nesting (currently only flat lists without brackets etc. are handled)
- distinguishing between start and intermediate tags (e.g. S-INGR and I-INGR, like S-ORG and I-ORG in NER)
- probably more

