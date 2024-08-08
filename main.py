import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.manifold import TSNE
from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer, Wav2Vec2Model
import torch
import librosa


if __name__ == '__main__':

    ds_path = "dataset"
    # load DATASET
    dataset = []
    labels = []
    ds_folders = os.listdir(os.path.join(ds_path, "dataset"))
    ds_folders.remove('.DS_Store')

    for fold in ds_folders:
        files = os.listdir(os.path.join(ds_path, "dataset", fold))
        for f in files:
            dataset.append(
                librosa.load(os.path.join(ds_path, "dataset", fold, f),
                             sr=16000)[0])
            labels.append(int(f.split('-')[2].split('0')[1]))

    # data exploration:
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # Create a dictionary to map emotions to numbers
    number_to_emotion = {idx + 1: emotion for idx, emotion in enumerate(emotions)}

    # Map the list of emotions to a list of numbers
    emotions_list = [number_to_emotion[num] for num in labels]

    # Plotting
    plt.figure(figsize=(14, 7))  # Make it 14x7 inch
    plt.style.use('seaborn-v0_8-pastel')  # Nice and clean grid

    # Create histogram with specified colors
    n, bins, patches = plt.hist(emotions_list, bins=len(emotions), range=(0.5, len(emotions) + 0.5),
                                edgecolor='white', linewidth=0.9, width=0.8)
    # Add title and labels
    plt.title('Classes Distribution', fontsize=16)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Values', fontsize=14)

    # Additional aesthetics
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid for better readability
    plt.yticks(fontsize=12)

    # Show plot
    plt.tight_layout()  # Adjust layout to ensure everything fits without overlap
    plt.show()

    print('PyCharm')

