Nutrient image recognition experiment
=====================================

When looking for product data for [Questionmark](http://www.thequestionmark.org/)'s
sustainability and health scores, we also consider online sources. While most data
online is textual, sometimes they're in images.

This is an experiment to find out how hard it would be to extract the data from the
image. A good opportunity for some image processing and recognition.

As a sample dataset, we're looking at some images from the Dutch supermarket webshop
[Hoogvliet](https://www.hoogvliet.com/), which has nutritional values in an image.

The initial approach was to do word-based recognition using a couple of custom-trained
kNN-networks (see [nutrient-ocr-knn](../nutrient-ocr-knn)), but in the end just
using tesseract was more convenient. That's what you're seeing here.


Running
-------

Needs [Python](http://python.org/) 2.5+ with [PIL](http://pythonware.com/products/pil/)
and [tesseract](https://github.com/tesseract-ocr/tesseract) 3.04 (or higher).

Since the images are quite low-resolution, the program scales them up three times,
does thresholding, and calls tesseract. Common misdetections are fixed.

The following example returns ingredients from [an image](../nutrient-ocr-knn/imgstest/VOED665279000.png).

```sh
$ ./tesseract.py ../nutrient-ocr-knn/imgstest/VOED665279000.png | grep -v '^$'
Voedingswaarde per 100 Gram
Energie 1050 Kilojoule
Energie 251 Kilocalorie
Vetten 12.3 Gram
Vetzuren, totaal verzadigd 8.2 Gram
Koolhydraten 31.4 Gram
Suikers 30.2 Gram
Eiwitten 3.3 Gram
Zout 0.32 Gram
```

That's looking great, and pretty easy.
