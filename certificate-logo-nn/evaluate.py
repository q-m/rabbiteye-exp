#!/usr/bin/env python3
"""Evaluate.py

Derived from: https://colab.research.google.com/drive/1KF03i-i1OYImGS_ySbeKoT4GKoKgHkCG
"""
import os
import copy
import glob
import math
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches
import xml.etree.ElementTree as ET

from imageai.Detection.Custom import CustomObjectDetection

from config import *
from custom_model import new_trainer

TEST_DIR = DATA_DIR + "test/"


### Evaluation functions
#

def iou(box1, box2):
    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(xi2-xi1, 0)
    inter_height = max(yi2-yi1,0)
    inter_area = inter_width * inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2-box1_x1)* (box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)* (box2_y2-box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    return iou

def yolo_non_max_suppression(scores, boxes, classes, iou_threshold=0.5): 
    # sort by probability from high to low
    sorted_scr = (-scores).argsort()
    scores = scores[sorted_scr]
    boxes = boxes[sorted_scr]
    classes = classes[sorted_scr]
    
    # vector to remember which detections we want to keep
    keep = [True] * len(scores)

    # iterate over all boxes
    for i in range(len(scores)):
      if keep[i] == True :
        # iterate over all following boxes (coming after this one)
        for j in range(i+1, len(scores)):
          if (iou(boxes[i], boxes[j]) > iou_threshold) and (keep[j] == True) :  #(classes[i] == classes[j]) and 
            keep[j] = False

    keep = np.array(keep)
    return scores[keep], boxes[keep], classes[keep]


### Confusion matrix
#

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize = (17,17))
    plt.imshow(cm, interpolation='nearest', cmap=cmap) # ,aspect='auto')
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=10,
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if filename is not None: fig.savefig(filename)

    plt.close(fig)

def create_confusion_matrix(df_confusion, cm_plot_labels, true_negs) : 
  df_cm = pd.DataFrame(columns = cm_plot_labels)
  df_cm.set_index(keys=cm_plot_labels)

  for lab1 in cm_plot_labels :
    if lab1 != 'true_negatives':
      for lab2 in cm_plot_labels :
        df_cm.loc[lab1, lab2] = len(df_confusion[(df_confusion['label_true'] == lab1) & (df_confusion['label_predicted'] == lab2)])
    else:
      for lab in cm_plot_labels: 
        df_cm.loc[lab1, lab] = 0
        df_cm.loc[lab, lab1] = 0
        df_cm.loc[lab1,lab1] = true_negs
  return df_cm

def calc_precision_recall(cm, labels):
  #calculates precision and recall
  #precision = TP / (TP+FP). (where something is a false positive if anything other than the accurate certificate was detected)
  #recall = TP / (TP+FN). 
  #no certificate scores not included in mean
  TP = cm.diagonal()[0:-3].tolist()
  FP = cm[-3,0:-3].tolist()
  TN = cm[-1,-1]
  FN = []

  for index in range(len(labels)):
    FN.append(sum(cm[index,:]) - cm[index,index])

  m_labels =  copy.copy(labels)
  m_labels.append('mean')

  accuracy = (sum(TP) + TN) / (sum(TP) + TN + sum(FP) + sum(FN)) 

  df_results = pd.DataFrame(columns = ['AP', 'AR'])

  for label in labels:
    if label in ['vegan']:
      continue
    i = labels.index(label)
    df_results.loc[label, 'AP'] = TP[i] / (TP[i]+FP[i])
    df_results.loc[label, 'AR'] = TP[i] / (TP[i]+FN[i])

  df_results.loc['mean','AP'] = df_results['AP'].mean()
  df_results.loc['mean','AR'] = df_results['AR'].mean()

  df_results.loc['no_certificate', 'AP'] = TN / (TN + sum(FN))
  df_results.loc['no_certificate', 'AR'] = TN / (TN + sum(cm[:,-2]))
  df_results.loc['accuracy', :] = accuracy
  return df_results


def create_cm_plot_labels(labels):
  cm_plot_labels = copy.copy(labels)
  cm_plot_labels.append('no_certificate')
  cm_plot_labels.append('not_detected')
  cm_plot_labels.append('true_negatives')
  return cm_plot_labels

def df_confusion_to_numpy(df_cm):
  return df_cm.to_numpy().astype('int')

# Function that translate xml file to dataframe
# (note that this is somewhat different from split_and_summarize.py!)
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'label_true', 'xmin', 'ymin', 'xmax', 'ymax'] #'width', 'height',
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# returns list of labels for image path
def what_labels_in_xml(image_path):
  xml_path = image_path.replace('.jpg', '.xml').replace('/images/','/annotations/')
  tree = ET.parse(xml_path)
  root = tree.getroot()
  km = []
  for member in root.findall('object'):
    km.append(member[0].text)
  return(km)

def plot_yolo(src_image_path, dst_image_path, probs, boksen, klassen):
  fig = plt.figure()
  image = plt.imread(src_image_path)
  plt.imshow(image)
  # get the context for drawing boxes
  ax = plt.gca()
  # get coordinates from the first face
  print('Prediction (+ certainty)')
  for i in range(len(probs)):
      # set proper coordinates
      x, y, x2, y2 = boksen[i]
      width = x2-x
      height = y2-y
      # create the rectangle
      rect = pltpatches.Rectangle((x, y), width, height, fill=False, color='red')
      # draw the box
      ax.add_patch(rect)
      # add name and probabilty
      ax.annotate(klassen[i] + ' '+ str(math.ceil(probs[i])), (x,y), color='blue')
      print(klassen[i], ":",str(math.ceil(probs[i])), "%")
  fig.savefig(dst_image_path)
  plt.close(fig)

def predict_test(df_true, print_results=False, min_prob=30, iou_tres=0.2):
    # Model path variables
    #make dataframe with FILENAME, LABEL_TRUE, LABEL_PRED, only do this if iou of boxes overlaps
    df_confusion = pd.DataFrame(columns=['filename', 'label_true', 'label_predicted'])
    df_true['gelinked'] = False
    true_negs = 0 

    image_path = TEST_DIR + '/images/'
    results_dir = TEST_DIR + '/results/'
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    results2_dir = TEST_DIR + '/results2/'
    if not os.path.exists(results2_dir): os.makedirs(results2_dir)

    nr_certificates = 0

    dtc = CustomObjectDetection()
    dtc.setModelTypeAsYOLOv3()
    dtc.setModelPath(MODEL_PATH)
    dtc.setJsonPath(JSON_PATH)
    dtc.loadModel()

    for f in os.listdir(image_path):
          print(f)
          km = what_labels_in_xml(image_path + f)
          print('True certificates: ' + str(km))
          detections = dtc.detectObjectsFromImage(input_image=image_path + f,
                                                  output_image_path=results_dir + f,
                                                  minimum_percentage_probability=70,
                                                  display_percentage_probability=True,
                                                  display_object_name=True,
                                                  nms_treshold=0.5)

          # Print detection results detected in input image
          if print_results is True:
            probs = []
            boksen = []
            klassen = []
            for detection in detections:
               # make a list 
                probs.append(detection['percentage_probability'])
                boksen.append(detection['box_points'])
                klassen.append(detection['name'])

          #apply personal NMS if anything has been found
          probs = np.array(probs)
          boksen = np.array(boksen)
          klassen = np.array(klassen)

          true_cert_indexes = df_true[df_true['filename'] == f].index

          if (len(probs) > 0):
            probs, boksen, klassen = yolo_non_max_suppression(scores = probs, boxes = boksen, classes = klassen)

            #in df_confusion: link detectie aan gelabeld keurmerk met iou >= 0.5, als deze er niet zijn, voeg een false positive toe
            #sla ook false negatives op in df_true
            #gaat per gevonden label alle daadwerkelijke labels af
            for i in range(len(klassen)): 
              gelinked = False #indicator of deze vinding gelinked is aan een true_certificate
              box_pred = boksen[i]
              certificate_pred = klassen[i]

              #ga alle gelabelde keurmerken af (indien die er zijn)
              #als die er niet zijn, dan blijft deze vinding ongelinked (en dus gaat hij als false positive df_confusion in)
              if (len(true_cert_indexes) > 0):
                for i in true_cert_indexes:
                  certificate_true = df_true.loc[i, 'label_true']
                  box_true = df_true.loc[i, ['xmin','ymin','xmax','ymax']]
                  
                  #als iou >= 0.5, maak een df aan met filename, echte certificaat, gedetecteerde certificaat
                  #zet gelinked op true voor deze detectie
                  #zet in dataframe dat dit keurmerk gelinked is
                  if iou(box_true ,box_pred ) >= iou_tres :
                    nr_certificates = nr_certificates + 1
                    df_confusion = df_confusion.append(pd.DataFrame(data =[[f, certificate_true, certificate_pred]], columns=['filename', 'label_true', 'label_predicted']), ignore_index=False)
                    gelinked = True
                    df_true.loc[i, 'gelinked'] = True
            
              #als deze detectie niet gelinked is, voeg dan rij aan dt toe waarin een 'false positive' wordt aangegeven als true label
              if (gelinked == False) :
                df_confusion = df_confusion.append(pd.DataFrame(data = [[f, 'no_certificate', certificate_pred]], columns=['filename', 'label_true', 'label_predicted']), ignore_index=True)            
          elif (km == []) and (len(probs) == 0):
            true_negs = true_negs + 1
            print('True negative!')
          else:
            print('False negative :(')

          plot_yolo(image_path + f, results2_dir + f, probs, boksen, klassen)
          print('------------------------------------------------------')

    #add columns of certificates that were not detected (false negatives)
    # add false negatives to df_true
    df_false_negs = df_true.loc[df_true['gelinked'] == 0, ['filename', 'label_true']]
    df_false_negs['label_predicted'] = 'not_detected' 
    df_confusion = df_confusion.append(df_false_negs[['filename','label_true',  'label_predicted']], ignore_index=True)
    return df_confusion, nr_certificates, true_negs


if __name__ == '__main__':
    ### Load, prepare and evaluate model
    #

    # Note that it is very important to have the same validation set that has been trained on,
    # otherwise the evaluation does not work (or would give wrong results).
    trainer = new_trainer()

    print("=== ImageAI evaluation ===")
    metrics = trainer.evaluateModel(model_path=MODEL_PATH, json_path=JSON_PATH, iou_threshold=0.2, object_threshold=0.8, nms_threshold=0.5)
    # print(metrics)


    ### analyze the mAP for different IoUs
    #

    # ious = np.linspace(start=0.2, stop=0.24, num=5)
    # maps = []
    # for x in ious:
    #   metrics = trainer.evaluateModel(model_path=MODEL_PATH, json_path=JSON_PATH, iou_threshold=x, object_threshold=0.8, nms_threshold=0.5)
    #   maps.append(metrics[0]['map'])


    # MAPs = [0.8053, 0.8061, 0.8063, 0.8053, 0.8055, 0.8054, 0.8081, 0.8053, 0.8051, 0.8055, 0.8056, 0.8088, 0.8053, 0.8112, 0.8052, 0.8052, 0.8056, 0.8112,0.8156,0.8073,0.8034, 0.8042, 0.8037]
    # IOUs = np.linspace(start=0.02, stop= 0.24, num=23)
    # pyplot.plot(IOUs, MAPs)
    # pyplot.grid()

    print("=== Custom evaluation ===")
    df_true = xml_to_csv(TEST_DIR + '/annotations/')
    df_confusion, nr_certificates, true_negs = predict_test(df_true=df_true, print_results=True)

    cm_plot_labels = create_cm_plot_labels(labels)
    df_cm = create_confusion_matrix(df_confusion, cm_plot_labels, true_negs)
    cm = df_confusion_to_numpy(df_cm)

    plot_confusion_matrix(cm, cm_plot_labels, filename='confusion_matrix.png')
    precision_recall = calc_precision_recall(cm, labels)
    print(precision_recall)

    print('Total false positives :' +  str(len(df_confusion[df_confusion['label_true'] == 'no_certificate'])) )
    print('Total false negatives :' +  str(len(df_confusion[df_confusion['label_predicted'] == 'not_detected'])) )
    print('Total true negatives : ' + str(true_negs))

