# **Behavioral Cloning** 
---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview
---
This repository contains the implementation of the Udacity driving behavior project. The trained model is trained, validated and tested using Keras library, which the model outputs steering angle to an autonomous vehicle in the provided simulator.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Dependencies
This repo requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* README.md

The simulator can be downloaded from the [here](https://github.com/udacity/self-driving-car-sim) and sample data from [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` is required to save the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, FPS (frames per second) can be defined for the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

[//]: # (Image References)

[image1]: ./examples/left.jpg "left camera"
[image2]: ./examples/center.jpg "center camera"
[image3]: ./examples/right.jpg "right camera"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
This project was developed over itarations. First, I developed a solid preprocessing and data loading pipeline to automize later adjustment. Alongside, I established data denerator to stream training and validation data from hard disk in a way that it can easily add more training and validation data (model.py lines 30-46). To collect more data, three cameras are placed on the vehicle, left, center and right camera. The left and right cameras are transferred and projected to the center camera with the correction factor of 0.2 (model.py lines 60-63).

#### 1. An appropriate model architecture has been employed

I came up with the final model through progressive iterations. My first step was to use a convolution neural network model similar to the LeNet architecture but as a regression model, I thought this model might be appropriate because it uses a stack of convolution layers along with fully connected ones. I didn't achieve desirable results with my first approach, then I utilized the [model architecture developed by Nvidia team](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and could fully meet the requirements to pass this project. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the gap between the train and validation set as an objective metric to monitor the overfitting status of my model. The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track in the first approach, yet with my final trained model I could cover the whole track q1 without falling off the road. 

#### 2. Reduce overfitting and parameter tuning

Since I used a proven architecture developed by Nvidia group, I didn't want to modify it so much so I tried to resolve the overfitting issue by trying different number of epochs. The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

#### 3. Final Model Architecture

The final model architecture (model.py lines 99-117) consisted of a convolution neural network with the following layers and layer sizes:

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________
```

The input images are normalized between `[-1, 1]` and cropped them to focus more on the regions with richer features as in the first layers of the model:

```sh
model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
...                           ...                     ...     

```

#### 3. Creation of the Training Set & Training Process

To develop a good driving behavior I used only the sample data provided by Udacity. On the other hand, I tried to well augment the data to be able to train my model being capable of autonomously driving in the udacity simulator - track 1.
Here are the three views of the left, center and right camera for a scene:

|  Left Camera   |   Center Camera  |  Right Camera  |
|:-------------:|:-------------:|:-------------:|
| ![alt text][image1] | ![alt text][image2] | ![alt text][image3] |

To augment the data sat, I also flipped images and angles to overcome the overfitting for specific maneruvers. Data shufflind was also encorporated in my pipeline.


