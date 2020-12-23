# Certificate logo detection with ImageAI

One important aspect of gathering data on food products in the supermarkets is
about which certificates the product has (like Organic, Fairtrade or MSC). Many
data sources have this kind of information, but it not always consistent with
reality.

One way to improve this data, is to look at the product images: often a logo
can be found on the package, and sometimes it is added as an overlay on top of
the photo.

Since there is a relatively small set of known certificate labels, it shouldn't
be too hard to automate this. That's what first a couple of trainees from
[Xomnia](https://www.xomnia.com/) set out to do, as you can read in
[this blog post](https://www.xomnia.com/post/ready-to-shop-more-sustainably-the-ai-solution-making-food-purchases-more-transparent/).
They selected [ImageAI](https://imageai.readthedocs.io/)'s custom object detection
as approach, prepared some data and trained a model. That was a good starting point.

Then mathematics intern Britt took it into her eager hands and set out improving
the model. Her result is what you can find here.

_Report (pending)_ |
[Presentation](2020-12-final-presentation-nl.pdf)

## Data

As a first step we set out to work on clean product images by the producer,
like the ones found in retailer webshops. These are high-quality photos with
the product clear on a white background, which is a lot easier to handle than
manually taken pictures taken from many directions and many lighting-conditions.

These photos were manually annotated using [LabelImg](https://github.com/tzutalin/labelImg).
Download the latest [Windows binary here](https://tzutalin.github.io/labelImg/) (at the bottom).

## Software setup

ImageAI uses Tensorflow 1, which doesn't work with Python 3.8 or higher, which has
become the default nowadays. To manage the version issue, we resort to using Docker.

First build the image using

```sh
docker build -t imageai .
```

Then we can start an interactive container as follows:

```sh
docker run --rm -v `pwd`:/app --name=imageai -ti -w /app -e HOME=/tmp -u `id -u`:`id -g` imageai
```

You get into an environment with all dependencies installed. Note that all
commands below are expected to be run inside this container. To exit the
container, use the `exit` command (or press <kbd>Ctrl-D</kbd>).

## Data preparation

Before we can train a model, the source data needs to be split into training,
evaluation and test sets. This is done by [`split_and_summarize.py`](split_and_summarize.py).
It expects JPEG-images in `data/Alles/images/` and XML-annotations in `data/Alles/annotations/`.
You can [download training data](https://qm-import-export.s3.amazonaws.com/2020-12-certificate-logo-nn-training-data.zip) (58MB),
extract this inside the `data/` directory.

With these files in place, run

```sh
python3 split_and_summarize.py
```

This results in a training, test and validation set with only the certificates
selected in [`config.py`](./config.py).

## Training

After data preparation, you still need a pretrained YOLOv3 file, which
can be downloaded [here](https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5) (237MB).
Save it in `data/models/`. Then you can run the training:

```sh
python3 train.py
```

Note that this can take a long, long time, and would preferably be run on a
rather large instance with a GPU, e.g. on AWS.

## Evaluation

After training is done, the result can be evaluated.
First find the model you want to use, look in `data/models/`. Edit [`config.py`](config.py) and
update `MODEL_PATH` and `JSON_PATH` (at the bottom). Then evaluate by running

```sh
python3 evaluate.py
```

which prints information and saves the confusion matrix to the file `confusion_matrix.png`.

## Prediction

Finally, the model can be put to use and predict which certificates are present
in images it has not seen before. You can give multiple image filenames as argument
to the script. Here we download a sample image (that was not present in the original dataset)
and detect its certificates.

```sh
wget https://productimages.thequestionmark.org/images/medium/1c8a97526cd20c26c15db10d6b2c1796.jpg
python3 predict.py 1c8a97526cd20c26c15db10d6b2c1796.jpg
```
```
Using TensorFlow backend.
-- 1c8a97526cd20c26c15db10d6b2c1796.jpg
{'name': 'msc', 'percentage_probability': 99.92098212242126, 'box_points': array([352, 453, 498, 517])}
{'name': 'asc', 'percentage_probability': 98.02358746528625, 'box_points': array([230, 451, 349, 513])}
```

Which corresponds to the two certificates shown on the product image.
