import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa


class Emotions_clf:
    def __init__(self):
        self.dataset = []
        self.labels = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.clf = SVC()
        self.sr = 16000

    def load_dataset(self, ds_path):
        ds_folders = os.listdir(os.path.join(ds_path, "dataset"))
        ds_folders.remove('.DS_Store')

        for fold in ds_folders:
            files = os.listdir(os.path.join(ds_path, "dataset", fold))
            for f in files:
                self.dataset.append(
                    librosa.load(os.path.join(ds_path, "dataset", fold, f),
                                 sr=self.sr)[0])
                self.labels.append(int(f.split('-')[2].split('0')[1]))

    def extract_features(self, audio, mfcc=False, chroma=False, mel=True):

        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40).T, axis=0)
            features.extend(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=self.sr).T, axis=0)
            features.extend(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=self.sr).T, axis=0)
            features.extend(mel)

        return features

    def prepare_dataset(self):

        X, y = [], []

        for idx,f in enumerate(self.dataset):
            features = self.extract_features(f)
            X.append(features)
            y.append(self.labels[idx])

        X = np.array(X)
        y = np.array(y)

        return X, y


if __name__ == '__main__':

    ds_path = "dataset"
    clf = Emotions_clf()
    clf.load_dataset(ds_path)
    X, y = clf.prepare_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    report = classification_report(y_test, y_pred, target_names=emotions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Calculate percentages

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_Mel.png')
    plt.show()

    print('PyCharm')
