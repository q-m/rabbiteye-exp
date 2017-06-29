# Product categorization improvements

After a first step in [applying machine learning for product categorization](../categorization-svm),
there were a lot of things to improve.

Code is provided for a more accurate evaluation criterion, and (later) feature reduction.

Read the [cross-validation explanation](CROSS_VALIDATION.md).


## Dependencies

To run the examples, one needs [Python 2.7](http://python.org/) with [sckit-learn](http://www.scikit-learn.org/).
To work with the `.ipynb` files, [Jupyter](http://jupyter.org/) is required.

## Pre-processing

**TODO** include sample data files in `data/`

The source data in `data/products.jsonl` and `data/product_nuts.jsonl` first needs to be put together:

```sh
$ python merge.py
```

resulting in `data/product_nuts_with_usage_and_product_id.jsonl`, containing both product attributes and
an assigned `usage` (as well as `product_id` for cross-validation with bubbles).

Now you're ready to run the notebooks.

## Cross-validation

Basic cross-validation using linear SVM can be found in [cross_validation.ipynb](cross_validation.ipynb).
An improved version that keeps product nuts belonging to the same product together, is found in
[cross_validation_bubble.ipynb](cross_validation_bubbles.ipynb).

The two cross validation files are implementations of machine learning to train an algorithm. This algorithm
is then evaluated with a macro recall score, and in the code without bubbles the accuracy is higher because
it does not account for almost duplicate data, which causes overfitting. For more background,
[read the explanation](CROSS_VALIDATION.md).

## Feature reduction

Existing code used all features for classification, even though some of them would not be relevant. In this
step, features that don't influence the result were removed - see [feature_reduction.ipynb](feature_reduction.ipynb).
This resulted in a 25% percent feature reduction, where the macro recall score ('with bubbles') did not
deteriorate more than 1%. [Read more details](FEATURE_REDUCTION.md).

