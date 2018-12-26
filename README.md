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


# Data

TBD

# Google Tensorflow Model and Weights
The file is in hdf5 format (.h5 file extension), and can be download from from Dropbox public folder [here](https://www.dropbox.com/s/x2ck81r3lokeq57/my_model-100epochs.h5?dl=0).

# GPU
This system is created with GPU, including Deep Learning Training, Inference processes.

GPU Used : 1 nVidia GeForce GTX 1080

# Computing Environment

TBD

# Client Devices

TBD

# Trainig Arrangements
### Number of Epochs
The weights used in the local test is 300 epoch, however the weights published is trained with only 100 epoch. You may find less accuracy with this 100 epoch when you apply the model on your own test images. To get a higher epochs such as 300 epoch or above, you need the training model and run for the destinated iterations.

### A Brief Introduction of Residual Network

TBD

### How to use this program - the computer vision model inferrence process

TBD


# Future Enhancements
- Face detection pre-processing to reduce error rate
- Include more background images as negative examples
- De-bias to add more Celebrity images from countries all over the world

