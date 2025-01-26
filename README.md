# Emotions Classifier

## Overview

The Emotions Classifier project focuses on detecting and classifying emotions from audio recordings. It leverages machine learning techniques to analyze audio features and assign them to specific emotional categories. This project has applications in areas such as human-computer interaction, sentiment analysis, and mental health monitoring.

## Features

#### Audio Feature Extraction:
Utilizes MFCC (Mel-Frequency Cepstral Coefficients) and Mel-spectrogram features to represent audio inputs.

#### Emotion Classification: 
Classifies emotions into predefined categories such as happiness, sadness, anger, etc.

#### Model Architecture:
Two models were tested to evaluate their performance on emotion classification:

1. Simple Convolutional Neural Network (CNN)

2. Support Vector Machine (SVM)

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

Comparison of MFCC vs Mel-spectrogram inputs showed that Mel-spectrogram achieved significantly better results.

CNN Performance: Achieved total accuracy of 35% on the test set with Mel features.
                 more results:
                          
                        precision    recall  f1-score  
        
                calm       0.67      0.55      0.60        
               happy       0.50      0.38      0.43        
                 sad       0.62      0.71      0.67         
               angry       0.30      0.50      0.38         
        
               accuracy                        0.35

SVM Performance: Achieved total accuracy of 51% using Mel features. 
                 more results:

                      precision    recall  f1-score  
        
                calm       0.67      0.55      0.60        
               happy       0.50      0.38      0.43        
                 sad       0.62      0.71      0.67         
               angry       0.30      0.50      0.38         
        
               accuracy                        0.51

confusion matrix of svm results on mel-spectrogram:
![confusion_matrix_mel_svm](https://github.com/user-attachments/assets/f4261178-1a05-43b7-b993-d5f71bd744db)

confusion matrix of cnn results on mel-spectrogram:
![Confusion_cnn_mel](https://github.com/user-attachments/assets/f37e818a-7ab7-485e-b6cf-fc4511ee19c5)


