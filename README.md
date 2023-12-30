# Praktikum_Machine-Learning_Leta-Anindya-R_202010370311231
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="logo leta.png" alt="Logo" width="360" height="180">

<h1 align="center">Rock, Paper, Scissors Prediction</h1>
  <p align="center">
    This project focuses on creating a deep learning model to predict
    images of rock, paper, scissors.
  </p>
</div>


### Authors
- Leta Anindya Riyadi ([@231_Leta Anindya Riyadi](https://www.github.com/letaanindyariyadi))


## Dataset
The dataset used in this project contains a total of 2520 images with an equal proportion of images per class: 840 images for rock, paper, and scissors, respectively. [The link to the dataset can be accessed here.](https://drive.google.com/file/d/1X9jFokn9AXMMVTmlBQ7XZpBsLKVFnp-d/view?usp=drive_link)

<div>
    <img src="download.png" alt="dataset" width="75%">
</div>

### Data Preprocessing
The dataset is first splitted using *splitfolders* library into 3 sets: training, validation, and testing with proportion of 80, 10, and 10 percent respectively.
```python
splitfolders.ratio('/content/drive/MyDrive/Modul5/rps', output="/content/drive/MyDrive/Modul5/rps_split",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
```
Then, the images are loaded using ImageDataGenerator() from the *keras.preprocessing.image* library. To prevent overfitting, the images are augmented with the paramaters below:
- rotation_range=30
- shear_range=0.2
- zoom_range=0.025
- horizontal_flip=True
- vertical_flip=True
- rescale=1./255
- brightness_range=(1,1.1)

<div>
    <img src="downloadd.png" alt="augmented_dataset" width="75%">
</div>

## Deep Learning Model
The modelling involves training the dataset with a pre-trained VGG-19

<div>
    <img src="arsitektur-convolutional-neural-networks (1).webp" alt="pretrained_architecture" width="75%">
</div>

### Model Training
Model is trained on the dataset with RMSprop optimizer and *categorical_crossentropy* loss for 3 epochs. Based on the training history graph, the model was able to highly recognize each images in the label without losing the validation accuracy.
|   Training and Validation accuracy | Training and Validation loss |
| ------------- | ------------- |
| ![](download model.png)  | ![](download model 2.png)  |
