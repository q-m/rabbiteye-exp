#!/bin/sh
#
# Cleans up all generated files, including the models (!)
# (so don't do this right after training)
#
# Only the source images and the pretrained YOLOv3 model are kept.
#

rm -f data/Alles/annotations.csv
rm -Rf data/train data/validation data/test data/cache data/logs
rm -Rf data/json data/models/detection_model*
