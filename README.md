# Mask R-CNN for Bangladeshi Food

This is a project of North South University Senior Project design under guidance of Mirza Mohammad Lutfe Elahi.

The codes are based on implementation of Mask R-CNN by (https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

The repository includes:

- Source code of Mask R-CNN built on FPN and ResNet101.
- Instruction and training code for the BangladeshiFoodData dataset.
- Pre-trained weights on MS COCO and ImageNet.
- Example of training on your own dataset, with emphasize on how to build and adapt codes to dataset with multiple classes.
- Jupyter notebooks to visualize the detection result.

# Training on Your own Dataset

Pre-trained weights from MS COCO and ImageNet are provided for you to fine-tune over new dataset. 

In summary, to train the model you need to modify two classes in `food.py`:

1. `FoodConfig` This class contains the default configurations. Modify the attributes for your training, most importantly the NUM_CLASSES.
2. 'FoodDataset' This class inherits from utils.Dataset which provides capability to train on new dataset without modifying the model. In this project I will demonstrate with a dataset labeled by VGG Image Annotation(VIA). If you are also trying to label a dataset for your own images, start by reading this blog post about the [balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46/) . First of all, for training you need to add class in function load_VIA

```
self.add_class("SourceName", ClassID, "ClassName")
#For example:
self.add_class("food", 1, "beef")  #means add a class named "beef" with class_id "1" from source "food"
......
```
Then extend function `load_mask` for reading different class names from annotations For example, if you assign name "a" to class "beef" when you are labelling, according to its class_id defined in `load_VIA`

```
class_ids = np.zeros([len(info["polygons"])])
for i, p in enumerate(class_names):
   if p['name'] == 'a':
      class_ids[i] = 1
      ......
```

Now you should be able to start training on your own dataset! Training parapeters are mainly included in function `train` in `food.py`.

```
#Train a new model starting from pre-trained COCO weights
python food.py train --dataset=/home/.../mask_rcnn/samples/food/data/ --weights=coco 

#Train a new model starting from pre-trained ImageNet weights
python surgery.py train --dataset=/home/.../mask_rcnn/samples/food/data/ --weights=imagenet

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python surgery.py train --dataset=/home/.../mask_rcnn/samples/food/data/ --weights=last
```

# Prediction, Visualization, Evaluation
Function `detect_and_color_splash` in `food.py` are provided in this project. To use `detect_and_color_splash`, you need to add class_names according to your dataset

```
class_names = ['BG', 'beef', 'chicken']
```

You can make prediction on a specific image, images in a specific directory or even a video, by

```
#Detect and color splash on a image with the last model you trained.
#This will find the last trained weights in the model directory.
python food.py splash --weights=last --image=/home/...../*.jpg
```
`prediction.ipynb` provides step-by-step prediction and visualization on your own dataset. You can also roughly evaluate the model with metrics of overall accuracy and precision.

# Instance Segmentation Samples on BangladeshiFoodData Dataset

The model is trained based on pre-trained weights for MS COCO.
![test1](https://github.com/tezalve/Bangladeshi-Food-Detection/blob/master/assets/17.PNG)
![test2](https://github.com/tezalve/Bangladeshi-Food-Detection/blob/master/assets/22.PNG)
![test3](https://github.com/tezalve/Bangladeshi-Food-Detection/blob/master/assets/8.PNG)
