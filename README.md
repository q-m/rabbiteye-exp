Nutrient image recognition experiment
=====================================

When looking for product data for [Questionmark](http://www.thequestionmark.org/)'s
sustainability and health scores, we also consider online sources. While most data
online is textual, sometimes they're in images.

This is an experiment to find out how hard it would be to extract the data from the
image. A good opportunity for some image processing and recognition.

As a sample dataset, we're looking at some images from the Dutch supermarket webshop
[Hoogvliet](https://www.hoogvliet.com/), which has nutritional values in an image.


Method
------

First the table is split into separate components. Then a basic
[k-nearest neighbour algorithm](http://docs.opencv.org/3.1.0/d0/d72/tutorial_py_knn_index.html)
trained, and then we can classify new images. This works well for fixed terms
like _energy_ and _salt_, which is just what we need for nutritional tables.

Header, nutrient name and nutrient units are trained separately.

Make sure you have [Python](http://www.python.org/) 2.5+, [numpy](http://www.numpy.org) 1.8+
and [OpenCV](http://www.opencv.org) 2.4+ (with Python bindings).

In this example, we'll be training with images from [imgs/](imgs) with labels
from [train/content.txt](train/content.txt), and test it on
[an image](imgstest/VOED665279000.png) that was not in the training set.

```sh
# First split img/*.png into separate bits for recognition in train/
$ ./split.py

# Edit train/content.{header,nam,unt}.txt and add a label for each file (only if you added images)
$ vi train/content.*.txt

# Train a simple k-nearest neighbour algorithm
$ ./knn_train.py

# Classify a new image with k=3
$ ./knn_test.py imgstest/VOED665279000.png 3
per 100 gram
Energie: Kilojoule
Energie: Kilocalorie
Vetten: Gram
Vetzuren, totaal verzadigd: Gram
Koolhydraten: Gram
Suikers: Gram
Eiwitten: Gram
Zout: Gram
```

So far, so good!


Future steps
------------

* Segment digits from value, and train a network to recognise them.
* Recognise text rows from the image, since not all images have equal dimensions.
* Validate by measuring errors
* Make sure new texts are detected (and not wrongly classified).

