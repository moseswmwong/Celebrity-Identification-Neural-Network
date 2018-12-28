# Celebrity-Identification-Neural-Network
Identify a potential celebrity base on a person picture

This neural network was being engineered with Keras components including Conv, Dense, combined with own writen ResNet50 Identity and Convolutional blocks. Then trained with 20,000 Celebrity pictures. 

# Business Potential Use and Initiatives

This AI is well trained to recognises a person whose picture reveal that she is very similar to a white female celebrity, using 10,000 white woman celebrity images this AI is trained through Residual Network 50 (ResNet50), an advance Neural Network architecture for computer vision. Model companies, advertising agency may find this application automate the supply of beautiful people discovery. The only reason this AI is trained with white female is because the Github is to serve as a demonstration purpose only. And the same AI can be extended to male or people from other colors and nations working equally well.

# Performance Assessment 1 - First Test Report
Test against webcam images

Result: Successful at 97% Accuracy. Among 3200 images provided to the inferrence network, only one image is identified as similar to celebrity. The test is being performed with three people appears for around 7 seconds in front of a webcam which takes motion detection filtered pictures around every one second. The three person appeared in front of the cameras are a male adult, a female adult, and a young boy. And only the female adult is classified as similar to celebrity, and the AI is able to identify this single picture. However, as the resulting accuracy calculation is to find out how many error pictures were mis-classifed and there are 100 images which are being mis-classified, therefore the accuracy is 97%. 

Total number of images being captures for people subjects:
- Male adult: 4
- Female adult: 2
- Boy: 12

After reviewed the mis-classified images, these are complex living room image which has few people sitting and sometime chatting on a dinner table with many colorful background small sub-images, therefore, the most probably explaination of mis-classification is due to Adversial problem of Convolutional Neural Network, which means many sub-sampling sections of the image are misregarded as Celebrity on the final Fully Connected layer.

For more detail explaination about Convolutional Neural Network and the intuition of Adversial problem of CNN model, please read this book www.deeplearningbook.org or this book [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)


# Performance Assessment 2 - Second Test Report
Test against people images

Result: Successful at 91% Accuracy. There are 33 images provided, most of them are celebrity images, two are non people images. The two non people images are accurately predicted as non celebrity, nine were classified as celebrity, including Ariana Grand, Gal Gadot, Kate Upton, Maggot Robbie etc. However, within the people images of the rest 22 images, there are three errors which are errorously identified as non celebrity which they obviously are. Error rate is 3 false-negatives (and 0 false-positive) divided by 33 equals 9%, therefore accuracy is 91%.

More details is available from the test report - [celebrity-images-test-report1.txt](https://github.com/moseswmwong/Celebrity-Identification-Neural-Network/blob/master/celebrity-images-test-report1.txt)

As a supplementary informtion, you may download the file ci4.html and open with a browser to check the Python Notebook browser screen capture as an alternative reference for a particular test run of these input images. This Python Notebook ran on the cloud with Paperspace.com using GPU+ machine and run time is about 30 seconds.

# Data

The process is to source most relevant data from the Internet, i.e. Celebrity images, select the target images, most famous white female celebrity, and clean up images that is too dark, too bright, people are too big or too small in the picture, and any errors, e.g. images of machines are occaptionally get its way into the dataset. I also spend some time to make sure small percentage of alternate format, for instance, cartoon style, are also included to ensure diversity of image format. 

I received these images mostly from AI research institutions which had already accumlated and processed many images that has similar requirements as stated.

Due to copyright no images used by this AI can be released by any means.

*** DISCLAIMER - all materials provided in this github.com page does not contains any celebrity image, images are all pre-processed to generate weights with the Neural Network and these processed weights only represent more general features, and they can only be understands by relevant AI program not human eyes. And there is no way, program, nor methodologies whatsoever to reverse engineering for recovering the original image in any format. The author and user of this program take no responsibiilty of claim of copyright for these images.

# Residual Network - The Algorithm Used

Residual Network is an advanced Deep Learning algorithm mainly designed and used in solving Computer Vision problems, it is a kind of Convolutional Neural Network (CNN) which excel in solving the vanishing gradient descent problem of Deep Learning which enable deeper layer stacking for the CNN Deep Learning architecture.

As a general rule with support from many laboratory reports, Computer Vision performance of a Deep Learning architecture increases as the number of layers increases. Therefore with this 50 layers Residual Network the Computer Vision performance and prediction accuracy is high, and can products very good prediction results.

There are many intricate differences between person faces and high degree of visual capability is needed for this project, hence it is chosen for this project.

The only drawback of deeper architecture is increases in computational resources requirements, thus I decided to use 50 layers for the computer vision project, at some early evaluation ResNet50 can provides pretty fast prediction performance down to 80 milliseconds processing time per image over tight resources platform, such as, a CPU-based mobile phone using Google's Androiod system running Tensorflow Lite.

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

Embedded Linux system connected with webcam with motion-based software runs on Debian Linux system is used for image capturing. Motion detection is applied on the captured images before passing onto the AI helped reduce traffic and AI loading. Raspberry Pi Model B+ was used but the same OS and Software can be used on other Embedded Linux hardware, e.g. BeagleBone Black.

Motion provides a platform to support common webcam hardware on Linux platform with motion detection and image processing modules. I setup one web camera on the hardware with devices modules activated which supports PI cameras, V4L2 (Video for Linux) webcams,and movie file processing. 

Motion detection is achieved by comparing pixel differences and count-base threshold check, false positive was pre-filtered with noise level detection to avoid too much false motion identification generated by electric noise in the camera.

# Reporting Client Devices

After AI processing, results can be found inside subfolder 'result'. Subsequent processes can display them on local machine or to perform further processings for example to display them through a web server farm.

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

The program is very easy to operate, just upload your image files to the subfolder 'localtest-celebrity' and run the Python Notebook, the processing time for one image is around 1 second on the nVidia GTX 1080 GPU.

After processing all files, you may collect image files being recognized as Celebrity or Celebrity-like in the sub-folder 'result'.

# Comparing other solutions

## Compare with automatic Machine Learning and visual recognition cloud services

Automatic Machine Learning, AutoML, is a new research direction which goal is to simplify hyper-parameter turning process, the benefit is simplified AI architecture design and development process, but the drawbacks are
1. Expensive services - pay for training on a per image per training basis, server and storage rental, and inference fee for AI, server and network usages
2. Very difficult to customize
3. Zero data ownership
4. Lost of competitive edge due to AI weights sharing after training with your data
5. Zero program ownership
6. Lost of competitive edge due to program logic sharing
7. Limited client device integration
8. Offline use impossible
9. IoT local Machine Learning processing impossible
10. General security concerns of data privacy using cloud services
11. More and more difficult to comply with the ever increasing privacy protection government regulations, result in increasing reputational and legal risk of business

# Future Enhancements
- Face detection pre-processing to reduce error rate
- Include more background images as negative examples
- De-bias to add more Celebrity images from countries all over the world
- Port to mobile phone and embedded devices using Tensorflow Lite and Apple's Neural Engine
- Port to nVidia Jetson TX2 Embedded Linux module for training and inference

# Artificial Intelligence

If you are curious everything about Artificial Intelligence, I hope you will find my Quora answers and questions interesting since I am a super curious person and decided to find answers by asking. As of 2018 boxing day, 2018-12-26, my answers and questions was being read by 66,700 visitors, and I was chosen by Quora as a Quora Partner. Here is the [link](https://www.quora.com/profile/Moses-Wong-1) to my Quora page.

AI is an extremely complex topic, so do the computer systems, softwares, algorithms and mathematics that runs it. I believes 'The best way to learn is to DO it', and my most favorite Quora content is [If humans can create artificial intelligence, then, are we a creation?](https://www.quora.com/If-humans-can-create-artificial-intelligence-then-are-we-a-creation)


