# Who's the Artist?
Image Classification - Convolution Neural Network project 

*By Abel Desta*

<p align="center">
    <img src="img/starry_night.png" width ='700' height ='400'/>
<p/>

# Introduction
Built an image‑classification model that identifies the artist of a painting. I designed and trained a convolutional neural network (CNN) on a curated dataset of digitized artworks, then evaluated its performance to refine architecture, hyper‑parameters, and data‑augmentation strategy. The final model reliably distinguishes artists’ styles, demonstrating practical use of deep‑learning workflows—from data collection and preprocessing through training, validation, and deployment.

        Tools: [Python, Pandas, Numpy, Sci-kit Image, Tensorflow Keras, AWS EC2]


## Data
I sourced a Kaggle dataset containing roughly 8,500 high‑resolution RGB JPEG images of paintings by 50 influential artists spanning multiple periods. A companion CSV file provides metadata for each work, including the artist’s name, brief biography, genre, and nationality. All images were scraped from the [artchallenge](http://artchallenge.ru/?lang=en) website. 

Although the images were resized during scraping, they remain high quality and vary in both dimensions and aspect ratios. Each image is stored as a three‑dimensional matrix: height × width pixels, with a depth of three channels (Red, Green, Blue). Pixel‑intensity values range from 0–255 for each channel.

<p align="center">
    <img src="img/Pablo_Picasso_3.jpg" width ='400'/>
    <img src="img/resize_Pablo_Picasso_3.jpg" width ='300'/>
<p/>

## Goal
For my capstone, I aim to build a convolutional neural network capable of identifying the artist of a painting from its image. I will first validate the pipeline by training and testing on a subset of three artists selected from the full 50‑artist dataset. Because training CNNs on my local hardware would be prohibitively slow, all experiments will be executed on an AWS EC2 p2.xlarge instance with a dedicated GPU, allowing faster iteration and more thorough hyper‑parameter tuning.

## EDA
There are 31 different genres in the dataset. Some artists belonged in more than one genre.

|Genres |  Number of Artists|Genres |  Number of Artists|
|--------|----------------|------------|--------|      
|Northern Renaissance|  4| Post-Impressionism   |  4|
|Impressionism        |    4| Baroque    |   4|
 Romanticism|     3| High Renaissance     |                      3|
 Surrealism|      2|Primitivism |    2|
|Impressionism,Post-Impressionism|           2| Early Renaissance |  1|
|Symbolism,Art Nouveau |                     1| Symbolism     |  1|
|Realism                 |       1| Social Realism,Muralism  |   1|
|Pop Art      |                 1| Neoplasticism            |    1|
|Expressionism,Abstractionism,Surrealism|    1| Symbolism,Expressionism   |    1|
|High Renaissance,Mannerism |   1| Surrealism,Impressionism    |               1|
|Cubism        |  1|Suprematism                   |             1|
|Realism,Impressionism        |              1| Expressionism   |        1|
|Symbolism,Post-Impressionism |             1 | Abstract Expressionism |            1|
|Mannerism                    |           1| Primitivism,Surrealism       |        1|
|Expressionism,Abstractionism |              1| Byzantine Art                |      1|
|Proto Renaissance    |                  1|

**Table 1. The number of artist in each genre.**

The dataset is highly imbalanced—some artists are represented by many more paintings than others, a disparity that likely stems from differences in artistic output and the uneven availability of digitized works. Such class imbalance can bias a convolutional neural network toward artists with larger image counts, so it will need to be mitigated during model development.




<p align="center">
    <img src="img/paintings.png" width ='700'/>
<p/>

**Figure 1. This bar chart shows severe discrepancy in images for artists.**



# Building the Artist CNN

## Pipeline 

Because the source images differ in size and aspect ratio, the first step in my pipeline is standardized resizing: each painting is converted to an RGB tensor of 100 × 100 × 3. To expand the training set, the pipeline then generates ten augmented variants (e.g., random rotations, flips, and slight color shifts) for every original image. For the initial model, I focus on the three artists with the largest representation in the dataset—Vincent van Gogh, Edgar Degas, and Pablo Picasso—to ensure a robust proof of concept before scaling to all fifty classes.

### Pipeline Flow

<p align="center">
    <img src="img/pipline_creation.png" />
<p/>

**Figure 2. The pipeline process.**


## CNN 
| Layers | Output Shape | # Parameter |
|--------|--------------|-------------|
| Convolution Layer | 98 x 98 x 32 | 896 |
| Convolution Layer | 96 x 96 x 32 | 9248 |
| Max Pool | 48 x 48 x 32 | 0 |
| Convolution Layer | 46 x 46 x 64 | 18496 |
| Convolution Layer | 44 x 44 x 64 | 36928 |
| Max Pool | 22 x 22 x 64 | 0 |
| Convolution Layer | 20 x 20 x 96| 55392 |
| Convolution Layer | 18 x 18 x 96| 83040 |
| Max Pool | 9 x 9 x 96 | 0 |
| Flatten | 7776 | 0 |
| Dense | 128 | 995456 |
| 60% Dropout | 128 | 0 |
| Dense | 3 | 387 |

* Input Image resolution: 250 x 250 x 3
* Trained on 1842 images
* Total Parameters: 1,199,843

# Transfer Learning 

After learning about transfer learning—which repurposes knowledge from a model trained on one task to boost performance on another—I decided to test whether a pre‑trained architecture could outperform my CNN built from scratch. I chose Keras’s Xception network and adopted a feature‑extraction strategy: I removed the classification head, passed each painting through the frozen convolutional base, and captured the resulting feature vectors. These one‑dimensional embeddings then served as inputs to two classical classifiers—a Random Forest and a Gradient Boosting model—allowing me to compare their performance against the baseline CNN.

# Results 
### Initial CNN Results
<p align="center">
    <img src="img/OG_CNN_acc.png" width ='400'/>
    <img src="img/OG_CNN_loss.png" width ='400'/>
<p/>

In my initial experiments, a four‑layer CNN without dropout severely overfit the data: training accuracy soared while validation accuracy lagged, and the widening gap in the loss curves confirmed the model’s excessive complexity for the available dataset.

### Some Improvement
<p align="center">
    <img src="img/CNN_acc1.png" width ='400'/>
    <img src="img/CNN_loss1.png" width ='400'/>
<p/>

This second iteration—a three‑layer CNN with a small amount of dropout—performed markedly better than the first model. It reached about 70 % accuracy on the training set and 65 % on the test set, showing reduced overfitting while still leaving room for further improvement.

### Final CNN model 

<p align="center">
    <img src="img/CNN_acc_other.png" width ='400'/>
    <img src="img/CNN_loss_other.png" width ='400'/>
<p/>

The final model kept the three‑convolution‑layer architecture but raised dropout from 0.3 to 0.8 and doubled the filters in the last layer. Trained for 50 epochs, it reached 76 % training accuracy and 74 % test accuracy by epoch 10—an improvement, though smaller than expected. After that, the growing divergence between training and validation metrics signaled renewed overfitting.

| Holdout metrics | My CNN | TL |
|-----------------|--------|----|
| Accuracy | 70% | 74% |
| Precision | 72% | 75% |
| Recall | 69% | 74% |



<p align="center">
    <img src="img/confuse_OG.png" width ='400'/>
    <img src="img/confuse_GDBC.png" width ='400'/>
<p/>

## Future Work

* Better treatment for imbalance classes and/or try 3 artists with relative the same amount of images 
* Work more on the model architeture and tuning
* After achieving good score on 3 artists, start adding artists, i.e. increase classes.
