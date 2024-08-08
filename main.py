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

    print('PyCharm')

