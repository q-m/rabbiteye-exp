#
# ImageAI Custom YOLOv3 model.
# Some customizations can be configured in config.py
# Note that this is tightly coupled with ImageAI version 2.1.5.
#
import os
import copy

import cv2
import numpy as np

from imageai.Detection.Custom.generator import BatchGenerator as BG 
from imageai.Detection.Custom import DetectionModelTrainer as DMT
from imageai.Detection.Custom.utils.utils import normalize

from config import *

### Edit ImageAI augmentation implementation
#

def _rand_scale(scale):
    scale = np.random.uniform(0, scale)
    return scale if np.random.randint(2) == 0 else 1./scale


def _constrain(min_v, max_v, value):

    if value < min_v:
        return min_v

    if value > max_v:
        return max_v

    return value 


def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin']*sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax']*sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin']*sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax']*sy + dy))

        if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
            zero_boxes += [i]
            continue

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]

    return boxes


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    
    # change hue
    image[:, :, 0] += dhue
    # image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    # image[:, :, 0] += (image[:, :, 0] < 0)   * 180
    
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):

    im_sized = cv2.resize(image, (new_w, new_h))
    
    if dx > 0: 
        im_sized = np.pad(im_sized, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0, 0), (0, net_w - (new_w+dx)), (0, 0)), mode='constant', constant_values=127)
               
    if dy > 0: 
        im_sized = np.pad(im_sized, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]
        
    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h+dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)
        
    return im_sized[:net_h, :net_w, :]



class BatchGenerator(BG):

    # def load_image(self, i):
    #     image = cv2.imread(self.instances[i]['filename'])
    #     image = self.blurr_image(image)
    #     return image
        
    def blurr_image(self, image, ksize):
        image_gaussian = cv2.GaussianBlur(
            image, (ksize, ksize), cv2.BORDER_DEFAULT
        )
        return image_gaussian


    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)   # BGR image
        
        # ksize = np.random.randint(1,101) # deze moet odd zijn
        # image = blurr_image(image, ksize)
        
        if image is None:
            print('Cannot find ', image_name)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB image
            
        image_h, image_w, _ = image.shape
        
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h

        new_ar = (image_w + ratio_distortion*np.random.uniform(-dw, dw)) / (image_h + ratio_distortion*np.random.uniform(-dh, dh))# De verhoudingen tussen hoogte en breedte minder extreem maken
        scale = np.random.uniform(zoom_min, zoom_max) #De zoom minder intens maken (was eerst (0.25, 2))

        if new_ar < 1:
            new_h = int(scale * net_h)
            new_w = int(net_h * new_ar)
        else:
            new_w = int(scale * net_w)
            new_h = int(net_w / new_ar)
            
        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))
        
        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
        
        # randomly distort hsv space
        im_sized = random_distort_image(im_sized, hue=hue_distortion, saturation=sat_distortion, exposure=exp_distortion)
        
                # # randomly flip
        # flip = np.random.randint(2)
        # im_sized = random_flip(im_sized, flip)
        flip=0
            
        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
        
        return im_sized, all_objs   



class DetectionModelTrainer(DMT):
  def __init__(self):
    super().__init__()
    #### CHANGE WEIGHTS ####
    self.__train_obj_scale = train_obj_scale
    self.__train_noobj_scale = train_noobj_scale
    self.__train_xywh_scale = 1
    self.__train_class_scale = train_class_scale
    #### CHANGE LEARNING RATE ####
    self.__train_learning_rate = 1e-4

  
  def trainModel(self):
    
    """
    'trainModel()' function starts the actual model training. Once the training starts, the training instance
    creates 3 sub-folders in your dataset folder which are:
    - json,  where the JSON configuration file for using your trained model is stored
    - models, where your trained models are stored once they are generated after each improved experiments
    - cache , where temporary traing configuraton files are stored
    :return:
    """

    train_ints, valid_ints, labels, max_box_per_image = self._create_training_instances(
        self.__train_annotations_folder,
        self.__train_images_folder,
        self.__train_cache_file,
        self.__validation_annotations_folder,
        self.__validation_images_folder,
        self.__validation_cache_file,
        self.__model_labels

    )
    if self.__training_mode:
        print('Training on: \t' + str(labels) + '')
        print("Training with Batch Size: ", self.__train_batch_size)
        print("Number of Training Samples: ", len(train_ints))
        print("Number of Validation Samples: ", len(valid_ints))
        print("Number of Experiments: ", self.__train_epochs)

    ###############################
    #   Create the generators
    ###############################
    train_generator = BatchGenerator(
        instances=train_ints,
        anchors=self.__model_anchors,
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=self.__train_batch_size,
        min_net_size=self.__model_min_input_size,
        max_net_size=self.__model_max_input_size,
        shuffle=True,
        jitter=0.3,
        norm=normalize
    )

    valid_generator = BatchGenerator(
        instances=valid_ints,
        anchors=self.__model_anchors,
        labels=labels,
        downsample=32,  # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=max_box_per_image,
        batch_size=self.__train_batch_size,
        min_net_size=self.__model_min_input_size,
        max_net_size=self.__model_max_input_size,
        shuffle=True,
        jitter=0.0,
        norm=normalize
    )

    ###############################
    #   Create the model
    ###############################
    if os.path.exists(self.__pre_trained_model):
        self.__train_warmup_epochs = 0
    warmup_batches = self.__train_warmup_epochs * (self.__train_times * len(train_generator))

    os.environ['CUDA_VISIBLE_DEVICES'] = self.__train_gpus
    multi_gpu = [int(gpu) for gpu in self.__train_gpus.split(',')]

    train_model, infer_model = self._create_model(
        nb_class=len(labels),
        anchors=self.__model_anchors,
        max_box_per_image=max_box_per_image,
        max_grid=[self.__model_max_input_size, self.__model_max_input_size],
        batch_size=self.__train_batch_size,
        warmup_batches=warmup_batches,
        ignore_thresh=self.__train_ignore_treshold,
        multi_gpu=multi_gpu,
        lr=self.__train_learning_rate,
        grid_scales=self.__train_grid_scales,
        obj_scale=self.__train_obj_scale,
        noobj_scale=self.__train_noobj_scale,
        xywh_scale=self.__train_xywh_scale,
        class_scale=self.__train_class_scale,
    )

    ###############################
    #   Kick off the training
    ###############################
    callbacks = self._create_callbacks(self.__train_weights_name, infer_model)

    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator) * self.__train_times,
        validation_data=valid_generator,
        validation_steps=len(valid_generator) * self.__train_times,
        epochs=self.__train_epochs + self.__train_warmup_epochs,
        verbose=1,
        callbacks=callbacks,
        workers=4,
        max_queue_size=8
    )


# Returns a new trainer
def new_trainer(gpus=1):
    # Set model type
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()

    # Data folder
    trainer.setDataDirectory(data_directory=DATA_DIR)

    # Training configuration
    trainer.setGpuUsage(gpus)

    trainer.setTrainConfig(
        object_names_array=labels,
        batch_size=batch_size, # batch size is zo klein omdat je anders OOM errors krijgt
                      # Out of Memory. Plaatjes zijn 700x700x3. Best wel groot.
        num_experiments=nr_epochs,
        train_from_pretrained_model=PRETRAINED_MODEL_PATH
    )

    return trainer

