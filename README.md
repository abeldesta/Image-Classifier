# Whose the Artist?
Capstone 1 project for Galvanize Data Science Immersive, Week 4

*By Abel Desta*

# Introduction
## Data
I found a dataset on kaggle that contains most of the artwork from the 50 influential artist ranging from everytime period. The data came with a csv file that contains information about each artist, such as a small bio, genre and nationality of the artist. The artwork was scraped from [artchallenge](http://artchallenge.ru/?lang=en) website. 

There is around 8500 images scaped. All the images in the files are RGB images in JPG format. RGB images are represented as 3D matrices. The rows and columns give us the number of pixels in eac dimension. The more pixels, the larger the matrix. The depth gives a 2D matrix with the pixel intensity of each color (Red, Green, and Blue) at each pixel. Values range from 0 to 250.

The images came resized but still high resolution. Also, most images came in a various different pixel size and shape.

## Goal
My goal for this capstone is to build a convolution neural network to be able to take an image of artists' artwork and classify the piece to the correct artists. To start, I will try to correctly classify the three artists out the 50 in the dataset.

## EDA

## Pipline 
