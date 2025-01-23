# Emotions Classifier

## Overview

The Emotions Classifier project focuses on detecting and classifying emotions from audio recordings. It leverages machine learning techniques to analyze audio features and assign them to specific emotional categories. This project has applications in areas such as human-computer interaction, sentiment analysis, and mental health monitoring.

## Features

#### Audio Feature Extraction:
Utilizes MFCC (Mel-Frequency Cepstral Coefficients) and Mel-spectrogram features to represent audio inputs.

#### Emotion Classification: 
Classifies emotions into predefined categories such as happiness, sadness, anger, etc.

#### Model Architecture:

Simple Convolutional Neural Network (CNN) for audio feature learning.

Support Vector Machine (SVM) for baseline performance comparison.

#### Visualization: 
Tracks and visualizes loss and accuracy metrics during training.

## Dataset

The project uses the RAVDESS audio dataset (public dataset) that containing labeled emotional expressions. 
Each audio file corresponds to a specific emotion.

## Installation

Clone the repository:

git clone https://github.com/Livnatc/emotions_clf.git
cd emotions_clf

Install the required dependencies:

pip install -r requirements.txt


## Results

CNN Performance: Achieved XX% accuracy on the test set with MFCC features.

SVM Performance: Achieved XX% accuracy using MFCC features.

Comparison of MFCC vs Mel-spectrogram inputs showed that [add your observation here].

