# Brain Tumor Classification: Deep Learning Using CNNs
**This repository contains an end-to-end project aimed at classifying brain tumors from MRI scans using Deep Learning techniques. The project leverages Convolutional Neural Networks (CNNs) for image classification, with the goal of assisting medical professionals in the early detection and categorization of brain tumors.**

## Problem Statement
Brain tumors are a serious medical condition requiring early detection for effective treatment. This project provides a Deep Learning-based classification system for brain tumor diagnosis, helping to distinguish between different tumor types from MRI scan images. The solution is designed to support medical experts by providing accurate predictions and facilitating faster diagnosis.

## Solution Overview
Deep Learning Model:

- A CNN-based model was built and trained on MRI images of brain tumors.
The model is capable of classifying different types of brain tumors such as glioma, meningioma, and pituitary tumors.

Image Preprocessing:

- MRI images were preprocessed for the model through resizing, normalization, and augmentation.

Transfer Learning:

- To improve accuracy, we employed Transfer Learning techniques by using pre-trained models like ResNet50 and VGG16.

Model Optimization:

- The model was optimized using techniques such as quantization and pruning, ensuring efficient deployment without compromising accuracy.

## Table of Contents
1. Project Overview
2. Features
3. Tech Stack
4. Model Architecture
5. Dataset
6. Installation
7. Deployment
8. Project Structure

## Project Overview
- The Brain Tumor Classification project demonstrates the use of Convolutional Neural Networks (CNNs) for medical image classification. MRI images are processed and fed into the CNN model, which predicts the type of brain tumor present in the image. This tool is designed to aid in faster diagnosis and better patient outcomes.

## Features
- Convolutional Neural Networks (CNNs) for image classification.
Transfer Learning with pre-trained models like ResNet50 and VGG16.
Image Preprocessing (normalization, augmentation).
- Visualization of predictions and model performance.

## Tech Stack
1. Python (v3.x): Programming language used for building the models .
2. TensorFlow and Keras: Deep Learning libraries for building and training the CNN model.
3. OpenCV and PIL: Image processing libraries.
4. Matplotlib: For visualizing results and metrics.

## Model Architecture
Convolutional Neural Networks (CNNs):

The model consists of convolutional layers for extracting features from MRI images, followed by pooling layers for downsampling and fully connected layers for classification.

Transfer Learning:

- Pre-trained models like ResNet50 and VGG16 are used for feature extraction, helping to achieve higher accuracy with fewer labelled images.

Model Optimization:

- Techniques such as model pruning and quantization were applied to reduce the model's size for easier deployment without compromising performance.

Dataset
- Kaggle Brain Tumor Dataset: MRI scans of brain tumors categorized into different tumor types (glioma, meningioma, pituitary tumors).
Images were preprocessed (resized, normalized) for consistent input size before training the CNN model.

## Installation
Clone the repository:
```
git clone https://github.com/TARAK0506/BRAIN-TUMOUR-CLASSIFICATION.git
cd BRAIN-TUMOUR-CLASSIFICATION
```
Create a virtual environment:
```
python3 -m venv venv
venv\Scripts\activate
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Download the dataset:

- Download the brain tumor MRI dataset and place it in the appropriate directory for training and testing.
- The dataset used consists of MRI scans of brain tumors categorized into different classes. The images were preprocessed to ensure consistent input size and format for the model. For more information on the dataset, please refer to the Kaggle Brain Tumor Dataset.


## Deployment Video

https://github.com/user-attachments/assets/660098d1-36ce-4132-a67a-9cd7056cbf63



