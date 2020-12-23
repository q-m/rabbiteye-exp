#!/usr/bin/env python3
"""split_and_summarize.py

Derived from: https://colab.research.google.com/drive/1aAU9sik9ymXQrA_JNS0sNmU4i0YuHu4q
"""
import os
import glob
import random
import shutil

import xml.etree.ElementTree as ET
import lxml.etree as ETT

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import *

### Harmonize image filenames
#

def jpeg_to_jpg_xml(path):
    img_names = os.listdir(path + 'images/')
    y = [s for s in img_names if s.find('.jpeg') > -1]
    for jpeg_file in y:
      os.rename(path+'images/'+jpeg_file,path+'images/'+jpeg_file.replace('.jpeg', '.jpg'))

    xml_list = []
    for xml_file in glob.glob(path+'annotations' + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        root.find('filename').text = (root.find('filename').text.replace('.jpeg', '.jpg'))
        root.find('path').text = (root.find('path').text.replace('.jpeg', '.jpg'))
        tree.write(xml_file)
        print(xml_file)

jpeg_to_jpg_xml(DATA_DIR + 'Alles/')


### Split files into train, test and validation set
#

def split_filenames(DATA_DIR, fold = 0.2) :
    # make dataframe with imagenames and corresponding annotationnames
    img_names = os.listdir(DATA_DIR+"Alles/images")
    anno_names = [w.replace('.jpg', '.xml') for w in img_names]

    img_names = [DATA_DIR+'Alles/images/'+w for w in img_names]
    anno_names = [DATA_DIR+'Alles/annotations/'+w for w in anno_names]

    data_names_ar = np.transpose(np.array([img_names, anno_names]))
    data_names_df = pd.DataFrame.from_records(data_names_ar, columns=['image', 'annotation'])

    #split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(data_names_df, data_names_df['annotation'], test_size=fold)

    #split into test and train data
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, X_train['annotation'], test_size=fold)

    #print the directory sizes:
    print('Supposed sizes of directories (images : annotations):')
    print('Train: ' + str(X_train.shape[0]) + " : " + str(y_train.shape[0]))
    print('Test: ' + str(X_test.shape[0]) + " : " + str(y_test.shape[0]))
    print('Validation: ' + str(X_valid.shape[0]) + " : " + str(y_valid.shape[0]))

    if (X_train.shape[0] != y_train.shape[0]) or (X_test.shape[0] != y_test.shape[0]) or (X_valid.shape[0] != y_valid.shape[0]) :
      raise Exception('Directory sizes are not equal')

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def write_df_to_csv(dir, df):
  #removes the images and annotations directories in the 'dir' directory, creates new, empty ones
  path = DATA_DIR + dir
  file_path = DATA_DIR + dir + '/'+ dir
  os.makedirs(path)
  if (dir == 'validation') or (dir == 'train') or (dir == 'test'): 
    if os.path.exists(file_path+ '_anno.txt'):
      os.remove(file_path+ '_anno.txt')
    if os.path.exists(file_path+ '_img.txt'):
      os.remove(file_path+ '_img.txt')
    df.image.to_csv(file_path+ '_img.txt', header=None, index=None, sep=' ', mode='a')
    df.annotation.to_csv(file_path+ '_anno.txt', header=None, index=None, sep=' ', mode='a')
  else :
    raise Exception('Dir str not train, test or validation')

def split_and_write(DATA_DIR, fold):
  #split
  X_train, X_valid, X_test, y_train, y_valid, y_test = split_filenames(DATA_DIR, fold)
  #write df to csv
  write_df_to_csv(dir = 'train', df = X_train)
  write_df_to_csv(dir = 'test', df = X_test)
  write_df_to_csv(dir = 'validation', df = X_valid)

if split_and_csv_bool:
  split_and_write(DATA_DIR=DATA_DIR, fold=0.2)


### Write files to directories
#

def clear_img_anno(dir):
  #removes the images and annotations directories in the 'dir' directory, creates new, empty ones
  path = DATA_DIR + dir
  if (dir == 'validation') or (dir == 'train') or (dir == 'test'): 
    if os.path.exists(path+'/images'): shutil.rmtree(path+'/images/')
    if os.path.exists(path+'/annotations'): shutil.rmtree(path+'/annotations/')
    if not os.path.exists(path + '/images'): os.makedirs(path + '/images')
    if not os.path.exists(path + '/annotations'): os.makedirs(path + '/annotations')
  else :
    raise Exception('Dir str not train, test or validation')

def check_missing_files(dir):
  #searches for files that exist in either the images or annotations directory, that are not present in the other
  if (dir == 'validation') or (dir == 'train') or (dir == 'test') or (dir == 'Alles') or (dir == 'alles'): 
      if (dir == 'alles'):
        dir = 'Alles'
      imgs = os.listdir(DATA_DIR+dir+'/images/')
      imgs = [w.replace('.jpg', '') for w in imgs]
      imgs = [w.replace('.jpeg', '') for w in imgs]
      imgs = [w.replace('.JPG', '') for w in imgs]
      imgs = [w.replace('.JPEG', '') for w in imgs]
      annos = os.listdir(DATA_DIR+dir+'/annotations/')
      annos = [w.replace('.xml', '') for w in annos]
      print('Files that don\'t appear in both image or '+dir+' directory: (Should be two empty lists)')
      print(list(set(imgs) - set(annos)))
      print(list(set(annos) - set(imgs)))

      if (len(list(set(imgs) - set(annos))) > 0 ) :
        raise Exception('There exist files in images that do not exist in annotations, namely: ' + str(list(set(imgs) - set(annos))) )
      if (len(list(set(annos) - set(imgs))) > 0):
        raise Exception('There exist files in annotations that do not exist in images, namely: ' + str(list(set(annos) - set(imgs)) ) )
  else :
    raise Exception('Dir str not Alles, train, test or validation')


def write_split_to_dir(dir, data_dir):
    print(dir)
    dir_path = data_dir + dir + '/'

    IMPATH = dir_path + 'images/'
    ANNOPATH = dir_path + 'annotations/'
    IMTXT = dir_path + dir + '_img.txt'
    ANNOTXT = dir_path + dir + '_anno.txt'

    # clear out image and annotations directories
    clear_img_anno(dir)

    print('Files that appear in both image or annotation directory: (Should be nothing)')
    print('Files that appear in both image or test directory: (Should be nothing)')
    os.system('ls "%s"' % IMPATH)
    os.system('ls "%s"' % ANNOPATH)
    print('Total amount of files in images and annotations in the %s directory:' % dir)
    os.system('ls "%s" | wc -l' % IMPATH)
    os.system('ls "%s" | wc -l' % ANNOPATH)

    # copy files to directory
    os.system('cat "%s" | xargs -I @ ln @ "%s"' % (IMTXT, IMPATH))
    os.system('cat "%s" | xargs -I @ cp @ "%s" ' % (ANNOTXT, ANNOPATH ))

    # test if every image has an annotation
    check_missing_files(dir)


if reassign_csv_bool :
  write_split_to_dir('train', DATA_DIR)
  write_split_to_dir('test', DATA_DIR)
  write_split_to_dir('validation', DATA_DIR)


### Filter temporary data (and output results to see if it worked)
#

# Function that translate xml file to dataframe
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    #  int(root.find('size')[0].text),
                    #  int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'certificate', 'xmin', 'ymin', 'xmax', 'ymax'] #'width', 'height',
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

# writes all xml files into one dataframe
def transformers_assemble(directory):
    image_path = os.path.join(directory, 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(image_path + '.csv', index=None)
    # print('Successfully converted xml to csv.')
    return xml_df

def filter_labels_in_subset(path, labels):
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            lab = (member.find('name').text)
            if lab not in labels : 
              # print(root.find('filename').text)
              # print(lab)
              root.remove(member)
        tree.write(open(xml_file, 'w'), encoding='unicode')


def filter_labels_everywhere(path, labels):
  for subset in ['train','test', 'validation']:
      filter_labels_in_subset(path+subset+'/annotations', labels)


filter_labels_everywhere(DATA_DIR, labels)

### Balance dataset by crossing out labels
#

def draw_over_label_and_alter_xml(path, filename, label):
    #draws a purple square over a certificate, and removes it from the xml file
    #path should link to directory on top of images / annotations subdirs
    image_name = path+'images/'+filename.replace('.xml', '.jpg')
    xml_name = path+'annotations/'+filename
    #open picture in OpenCV
    img = cv2.imread(image_name)

    #read out coordinates of certificate location and draw over it
    tree = ET.parse(xml_name)
    root = tree.getroot()
    for member in root.findall('object'):
      if member.find('name').text == label:
        pt_min = (int(member.find('bndbox').find('xmin').text),int(member.find('bndbox').find('ymin').text) )
        pt_max = (int(member.find('bndbox').find('xmax').text),int(member.find('bndbox').find('ymax').text) )
        img = cv2.rectangle(img=img, pt1=pt_min, pt2=pt_max, color=(125, 0, 125) , thickness=-1)
        root.remove(member)
        tree.write(open(xml_name, 'w'), encoding='unicode') #.replace('.xml', '_altered.xml')
        cv2.imwrite(image_name, img) #.replace('.jpg', '_altered.jpg')


def list_only_label(path, label):
    #makes list of all files that contain a certificate in labels
    # path should link to annotations file
    files_with_label = []
    for xml_file in glob.glob(path+'*.xml'):
      tree = ET.parse(xml_file)
      root = tree.getroot()
      for member in root.findall('object'):
        if member.find('name').text == label:
            files_with_label.append((root.find('filename').text).replace('.jpg', '.xml'))
    return files_with_label 


def remove_perc_labels_in_dir(path, label, percentage_to_remove):
    #iterates over a __% random list of files with certain label, and alters them
    #path should link to the directory above the images/annotations subdir. 
    files_with_label = list_only_label(path + 'annotations/', label)
    files_to_alter = random.sample(files_with_label, int(percentage_to_remove*len(files_with_label)))
    for filename in files_to_alter:
      draw_over_label_and_alter_xml(path, filename, label)


def remove_perc_everywhere(path, label, percentage_to_remove):
    # remove certain percentage of certain label out of all 3 split-datasets
    for dir in ['train/','test/','validation/']:
      remove_perc_labels_in_dir(path=path+dir, label=label, percentage_to_remove=percentage_to_remove)


if balance_certificates_bool: 
  percentage_to_remove=0.55
  label = 'organic'
  remove_perc_everywhere(path=DATA_DIR, label=label, percentage_to_remove=percentage_to_remove)

  percentage_to_remove=0.61
  label = 'fairtrade'
  remove_perc_everywhere(path=DATA_DIR, label=label, percentage_to_remove=percentage_to_remove)


### Preprocessing
#

def prepr_dir(dir, ks = 0):
  for file in glob.glob(dir+"*.jpg"):
      img = cv2.imread(file)
      blur = cv2.GaussianBlur(img,(ks,ks),cv2.BORDER_DEFAULT)
      filename = file
      cv2.imwrite(filename,img-blur)

def preprocess(DATA_DIR):
  for dset in ['train','test','validation']: 
    prepr_dir(DATA_DIR+dset+'/images/')

if preproces_bool: 
  preprocess(DATA_DIR)

if visualisation_bool:
  visualise_data_for_label(DATA_DIR, 'beterleven3')

### Check splits
#

# Returns for any directory how many of each certificate are present
def certificates_in_dir(dir):
    if (dir == 'Alles') or (dir == 'alles') :
      path = DATA_DIR + 'Alles/'
    elif (dir == 'validation') or (dir == 'train') or (dir == 'test'):
      path = DATA_DIR+ dir+ '/'
    else :
      raise Exception('Split value not train, test or validation')

    annotations_df = transformers_assemble(path)

    classes = (
        annotations_df
        ['certificate']
        .unique()
    )
    print('Nr of files in ' + dir + ' directory: ' + str(annotations_df.shape))
    print('Nr of certificate types in ' + dir + ': ' + str(len(classes)))


    # count how many pictures there are per certificate
    distrib = (
        annotations_df[['certificate', 'filename']]
        .groupby(['certificate'])
        .agg(['count'])
    )
    distrib['percentage'] = distrib['filename']['count']*100/annotations_df.shape[0]
    # print(distrib)
    # print('\n')
    return distrib


print(certificates_in_dir('Alles'))
print('\n')
print(certificates_in_dir('train'))
print('\n')
print(certificates_in_dir('test'))
print('\n')
print(certificates_in_dir('validation'))

