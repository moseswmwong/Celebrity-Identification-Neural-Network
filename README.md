# Celebrity-Identification-Neural-Network
Identify a celebrity base on a person picture

This neural network is training with Keras components including Conv, Dense, combined with own writen ResNet50 Identity and Convolutional blocks. Then trained with 20,000 Celebrity pictures. 


# Performance Assessment 1 - First Test Report
Test against webcam images

Result: Successful at 97% Accuracy. Among 3200 images provided to the inferrence network, only one image is identified as similar to celebrity. The test is being performed with three people appears for around 7 seconds in front of a webcam which takes motion detection filtered pictures around every one second. The three person appeared in front of the cameras are a male adult, a female adult, and a young boy. And only the female adult is classified as similar to celebrity, and the AI is able to identify this single picture. However, as the resulting accuracy calculation is to find out how many error pictures were mis-classifed and there are 100 images which are being mis-classified, therefore the accuracy is 97%. 

Total number of images being captures for people subjects:
- Male adult: 4
- Female adult: 2
- Boy: 12

After reviewed the mis-classified images, these are complex living room image which has few people sitting and sometime chatting on a dinner table with many colorful background small sub-images, therefore, the most probably explaination of mis-classification is due to Adversial problem of Convolutional Neural Network, which means many sub-sampling sections of the image are misregarded as Celebrity on the final Fully Connected layer.

For more detail explaination about the intuition of Adversial problem of CNN model, please read www.deeplearningbook.org


# Performance Assessment 2 - Second Test Report
Test against people images

Result: Successful at 91% Accuracy. There are 33 images provided, most of them are celebrity images, two are non people images. The two non people images are accurately predicted as non celebrity, nine were classified as celebrity, including Ariana Grand, Gal Gadot, Kate Upton, Maggot Robbie etc. However, within the people images of the rest 22 images, there are three errors which are errorously identified as non celebrity which they obviously are. Error rate is 3 false-negatives (and 0 false-positive) divided by 33 equals 9%, therefore accuracy is 91%.

More details is available from the test report - [celebrity-images-test-report1.txt](https://github.com/moseswmwong/Celebrity-Identification-Neural-Network/blob/master/celebrity-images-test-report1.txt)

As a supplementary informtion, you may download the file ci4.html and open with a browser to check the Python Notebook browser screen capture as an alternative reference for a particular test run of these input images. This Python Notebook ran on the cloud with Paperspace.com using GPU+ machine and run time is about 30 seconds.

# Data

TBD

# Residual Network - The Algorithm Used

TBD+

# Google Tensorflow Model and Weights
The file is in hdf5 format (.h5 file extension), and can be download from from Dropbox public folder [here](https://www.dropbox.com/s/x2ck81r3lokeq57/my_model-100epochs.h5?dl=0).

# GPU
This system is created with GPU, including Deep Learning Training, Inference processes.

GPU Used : 1 nVidia GeForce GTX 1080

# AI Computing Environment

  The following are the key requirements of the Deep Learning system for running the Neural Network
  
  - Python 3.6
  - nVidia CUDA 9.0
  - Google Tensorflow 1.5.0 or above
  - iPython Notebook
  - (Optional) Anaconda

# Capturing Client Devices - Embedded System Device used as webcam for image capturing and motion detection

TBD

# Reporting Client Devices

TBD

# Training Arrangements
### Number of Epochs
The weights used in the local test is 300 epoch, however the weights published is trained with only 100 epoch. You may find less accuracy with this 100 epoch when you apply the model on your own test images. To get a higher epochs such as 300 epoch or above, you need the training model and run for the destinated iterations.

# How to use this program - the computer vision model inference process

## Preparations

1. Download all files from this repository.

2. Download trained model file [my_model-100epochs.h5](https://www.dropbox.com/s/x2ck81r3lokeq57/my_model-100epochs.h5?dl=0), and store it the same directory of this repository. Modify the source code to use this model file, it can be easily found with the iPython Notebook file.

3. Creates a subfolder name "localtest-celebrity".

4. Install all necessaries Python libraries either through conda or pip, includes numpy, keras-gpu, scipy, matplotlib, gc, pillow
   - keras is a Machine Learning library provided by Google for creating ML models or modules base on Tensorflow, Torch or other supported backend. The '-gpu' variant of keras library ensure it runs on GPU.
   - numpy is a numerical operation python library created for scientific computation use, for ML specific applications, vectorized matrix operations are dependent on numpy, but DL mainly depend on CUDA and nVidia GPU instead.
   - scipy.misc are miscellanous library from this scientific computation library.
   - matplotlib is used to plot graph.
   - gc is the gabbage colleciton library for releasing resources during the resources intensive ML/DL operations.
   - pillow is an image processing library used to import and convert image files into specific dimensions and other characteristics to fit subsequent computer vision processing with the particular Deep Learning architecture I created for this application.

## Running the inference engine

TBD

# Comparing other solutions

## Comparing AutoML

TBD

## Comparing Amazon Rekonition cloud services

TBD

# Future Enhancements
- Face detection pre-processing to reduce error rate
- Include more background images as negative examples
- De-bias to add more Celebrity images from countries all over the world
- Port to mobile phone and embedded devices using Tensorflow Lite and Apple's Neural Engine

# Final words ...

If you are curious everything about Artificial Intelligence, I hope you will find my Quora answers and questions interesting since I am a super curious person and decided to find answers by asking. As of 2018 boxing day, 2018-12-26, my answers and questions was being read by 66700 visitors, and I was chosen by Quora as a Quora Partner. Here is the [link](https://www.quora.com/profile/Moses-Wong-1)

AI is an extremely complex topic, so do the computer systems, softwares, algorithms and mathematics that runs it. I believes 'The best way to learn is to DO it'. My most favorite Quora content is [If humans can create artificial intelligence, then, are we a creation?](https://www.quora.com/If-humans-can-create-artificial-intelligence-then-are-we-a-creation)

