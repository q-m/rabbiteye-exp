Product Categorisation with Machine Learning - 2 

The code in this directory is to asses the macro recall score of a machine learning algorithm designed to classify products. 

The two main files are ‘Cross Validation with Bubble.ipynb’ and the ‘Cross Validation without Bubble.ipynb’. These are two Jupiter notebook files, and work with python 2.7. The features.py file is standard python code, and functions with python 2.7 as well. Merge product and product nuts is also a notebook, and runs on python 2.7.

features.py returns a list of json types with the usage, product_id, product nuts id and the normalised tokens, for each product given to feauturize_all. 

The data that is given to featurize.py is data that is made in ‘Merge product and product nuts.ipynb’. Here products and products nuts are combined, and a person of product nuts is returned, with two extra keys in each dictionary: usage and product_id. 

The two cross validation files are implementations of machine learning to train an algorithm. This algorithm is then evaluated with a macro recall score, and in the code without bubbles the accuracy is higher because it does not account for almost duplicate data, which causes overfitting. 