# Image-descriptors-HoG-
Implementation of a Histogram of Oriented Gradients (HoG) - Image processing
This model will attempt to accurately determine whether a given image contains a human or not, using the Histogram of Oriented Gradients. In this repository, you will find a folder with some images for testing.

## Introduction

It was first introduced by Dalal and Triggs for human
detection in CCTV images. We will follow in their steps and implement a solution for
the same task using both the descriptor and the Machine Learning algorithm K-nearestneighbours.


#### Steps:
1. Load images
2. Transform all images to black&white using the Luminance technique
3. Implement and compute the HoG descriptor of all images.
4. Implement and Train a K-nearest-neighbours model with K = 3 using the training images.
5. Predict the classes for each of the test images.


### Processing
Before computing the image descriptors we need to transform the images to black&white.
For this task we are are going to use the common Luminance method, for each RGB pixel:<br>
f(x, y) = ⌊0.299R + 0.587G + 0.114B⌋,<br>
where ⌊.⌋ is the floor operation and letters R, G, B represent each colour channel.

### Histogram of Oriented Gradients

This descriptor is a good way of capturing how the textures in an image are “arranged”
by looking at the angle of the gradients (the rate-of-change from one pixel to the
next at a certain angle). To compute this we must first obtain the gradient of the
image in both x and y directions. This can be done by performing convolution (use
scipy.ndimage.convolve) of an image with the sobel operator in each direction:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/f1bc7248-4d9c-4eab-ad21-369b25d3ec93) <br>
After computing the convolution between the image and each filter, we will have the image
gradients gx and gy. Using those matrices, we can separate the gradient’s magnitude and
angle into two other matrices. The magnitude can be computed:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/5a1687c4-5ae8-4d3b-9528-b65918be61fa) <br>
and the angles:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/7d9bc547-67f6-4ecc-b715-1efabfdc077b) <br>
Now, we need to use those angles to accumulate magnitudes in a binned fashion. First,
(i) sum π/2 to all values in ϕ to make the angle range shift from [−π/2, π/2] to [0, π], then,
(ii) convert the angles from radians to degrees (see np.degree); (iii) digitise the angles
into 9 bins, this means slicing the angle range in 20 degree intervals [0, 19], [20, 39], ... and
then for each ϕ(x, y), determine in which of the 9 bins the value falls into.
We want to use the bins determined for each x, y position to accumulate the magnitudes. Let’s consider an example with post-shift and conversion matrix:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/bdadf254-86f8-4d05-b736-0aa7e7246b49) <br>
And the magnitude matrix:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/aaecc047-fdd6-4c13-a563-297eea773a83) <br>
Computing the bins for each position we would have:
<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/c4bf7aa9-0415-43f7-9562-c06d6b4f67e8)<br>
where, for example, ϕd(1, 2) = 4 because ϕ(1, 2) = 91 falls into the range [80, 99], which
is bin 4. Now, using this setup to accumulate magnitudes using the bins, we’d have a
resulting descriptor:

<br> ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/9132ef8f-bdc8-40cb-9c97-1c32e8581e7c) <br>
where, for example, dg(1) = 5 because positions ϕd(0, 1) and ϕd(0, 2) are in bin 1 and
the magnitude for those positions is M(0, 1) = 2, M(0, 2) = 3.
This descriptor is then a vector with 9 dimensions, dg.
NOTE: In this example there will be some divisions by zero that will result in
infinity. This is ok as np.arctan works with infinity. Your code will however produce
some warnings that are flagged by the auto-grader.

### K-Nearest Neighbours

Just as the Histogram of Oriented Gradients is a classic of image feature extractors,
K-Nearest Neighbours (KNN) is a classic of Machine Learning literature. The concept
for KNN is very simple: there is no training step to the model and inference/prediction
is done by comparing new samples against the training set
With a selection of labelled feature vectors (which we have in the form of the features
extracted from X0 and X1), prediction with KNN consists of comparing a new sample
(a feature vector from an image in Xtest for instance) against all of the vectors in the
labelled training set. For doing such comparison we will make use of euclidean distance
between the vectors:

![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/57ac18ff-a34d-4e68-8c78-94cb1035b9cf)

where a and b are images and dg is the HoG feature extractor.
After computing the euclidean distance between a test sample and all of the training
sample, the predicted class (has human or does not) is determined by a “voting system”.
The K (here defined as K = 3) training samples closest to the test sample each get “a
vote”, which is determined by their original class. For example, if 2 of the closest samples
came from X0 and one came from X1, then the majority of votes goes to class 0, meaning
that no human was detected in the test sample (the predicted label is 0).

Your goal then is to run this prediction algorithm for each of the test samples in Xtest
and print out the resulting predictions for weather there is a human (1) or not (0) in
each image. The output of your code consists of printing out the predictions for each
test sample separated by a single space.

### Input and Output

In the case it was used images with and without humans, the model is trained with labeled images and we are going to pass 10 images without humans, 10 images with humans and 10 images for testing<br><br>
![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/8beb8b88-292c-49db-bfff-0b2169285eee)     ![image](https://github.com/pedrodbatista/image-descriptors-HoG-/assets/80288516/a6e24556-497d-4713-96d9-0cf1f26d8787)<br>
_examples of the inputs that the program will use for training_<br>

In the code you have to pass all the file names separeded by space for the same sample, and separeded by a line brak for differente samples.<br>

The result of using the HoG descriptor of the 20 input training
images to train a KNN model and then using this model to predict whether the 10 testing
samples have humans or not.

### Testing
Running the model with
```
python3 imageDescriptors.py < input.in
```
To run use the 'input.in' file to simplify the process of specifying the directory where the image files are located and the files itself. In this input, we allocated 60% of the images for training and used 20 images for testing. The model has achieved a 70% accuracy, which appears to be suitable for the project's scope.t.

#### References

_N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In 2005
IEEE computer society conference on computer vision and pattern recognition (CVPR’05),
volume 1, pages 886–893. Ieee, 2005._
