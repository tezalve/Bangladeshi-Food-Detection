# Mask R-CNN for Bangladeshi Food

This is a project of North South University Senior Project design under guidance of Mirza Mohammad Lutfe Elahi.

The codes are based on implementation of Mask R-CNN by (https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository includes:

- Source code of Mask R-CNN built on FPN and ResNet101.
- Instruction and training code for the surgery robot dataset.
- Pre-trained weights on MS COCO and ImageNet.
- Example of training on your own dataset, with emphasize on how to build and adapt codes to dataset with multiple classes.
- Jupyter notebooks to visualize the detection result.

# Training on Your own Dataset

Pre-trained weights from MS COCO and ImageNet are provided for you to fine-tune over new dataset. 

In summary, to train the model you need to modify two classes in 'surgery.py':

1. 'SurgeryConfig' This class contains the default configurations. Modify the attributes for your training, most importantly the NUM_CLASSES.
2. 'SurgeryDataset' This class inherits from utils.Dataset which provides capability to train on new dataset without modifying the model. In this project I will demonstrate with a dataset labeled by VGG Image Annotation(VIA). If you are also trying to label a dataset for your own images, start by reading this blog post about the [balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46/) . First of all, for training you need to add class in function load_VIA

self.add_class("SourceName", ClassID, "ClassName")
#For example:
self.add_class("food", 1, "beef")  #means add a class named "beef" with class_id "1" from source "food"
......
