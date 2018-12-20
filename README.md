# Celebrity-Identification-Neural-Network
Identify a celebrity base on a person picture

This neural network is training with Keras components including Conv, Dense, combined with own writen ResNet50 Identity and Convolutional blocks. Then trained with 20,000 Celebrity pictures. 



Result: Successful at 97% Accuracy. Among 3200 images provided to the inferrence network, only one image is identified as similar to celebrity. The test is being performed with three people appears for around 7 seconds in front of a webcam which takes motion detection filtered pictures around every one second. The three person appeared in front of the cameras are a male adult, a female adult, and a young boy. And only the female adult is classified as similar to celebrity, and the AI is able to identify this single picture. However, as the resulting accuracy calculation is to find out how many error pictures were mis-classifed and there are 100 images which are being mis-classified, therefore the accuracy is 97%. 

Total number of images being captures for people subjects:
- Male adult: 4
- Female adult: 2
- Boy: 12

After reviewed the mis-classified images, these are complex living room image which has few people sitting and sometime chatting on a dinner table with many colorful background small sub-images, therefore, the most probably explaination of mis-classification is due to Adversial problem of Convolutional Neural Network, which means many sub-sampling sections of the image are misregarded as Celebrity on the final Fully Connected layer.



Future enhancmeents:
- Face detection pre-processing to reduce error rate
- Include more background images as negative examples
- De-bias to add more Celebrity images from countries all over the world


### Number of Epochs
The weights used in the local test is 300 epoch, however the weights published is trained with only 100 epoch. You may find less accuracy with this 100 epoch when you apply the model on your own test images. To get a higher epochs such as 300 epoch or above, you need the training model and run for the destinated iterations.


![alt text](https://mir-s3-cdn-cf.behance.net/project_modules/disp/c17cc352639217.59173719c2a57.jpg)

![alt text](http://url/to/img.png)

