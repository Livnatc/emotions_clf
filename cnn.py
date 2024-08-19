import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features(file_path, n_mels=128):
    with open(file_path, 'rb') as f:
        audio, sr = librosa.load(f, sr=None)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if mel_spectrogram.shape[1] > max_len:
        # Truncate the spectrogram to max_len
        mel_spectrogram = mel_spectrogram[:, :max_len]
    else:
        # Pad the spectrogram with zeros to make it max_len
        pad_width = max_len - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mel_spectrogram


def extract_features_mfcc(file_path, n_mfcc=13):
    with open(file_path, 'rb') as f:
        audio, sr = librosa.load(f, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Optionally, normalize the MFCC features
    mfcc = librosa.util.normalize(mfcc)

    if mfcc.shape[1] > max_len:
        # Truncate the MFCCs to max_len
        mfcc = mfcc[:, :max_len]
    else:
        # Pad the MFCCs with zeros to make it max_len
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfcc


class Emotions_clf(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        features = extract_features_mfcc(file_path)

        if self.transform:
            features = self.transform(features)

        features = torch.tensor(features, dtype=torch.float32)
        features = features.unsqueeze(0)  # Add channel dimension

        return features, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = self.fc1 = nn.Linear(64 * 32 * (max_len // 4), 128)  # Adjust the input size based on your data
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        embd = x  # Save the embeddings
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, embd


class SimpleCNN_mfcc(nn.Module):
    def __init__(self, num_classes=8, n_mfcc=13):
        super(SimpleCNN_mfcc, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.calculate_output_size(n_mfcc)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(128, num_classes)

    def calculate_output_size(self, n_mfcc):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mfcc, max_len)
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        embd = x  # Save the embeddings
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x, embd


if __name__ == '__main__':

    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  # Assuming these are your emotion labels

    file_paths = []
    labels = []
    base_path = 'dataset/dataset'
    internal_folds = os.listdir(os.path.join(base_path))
    max_len = 350

    for fold in internal_folds:
        if fold != '.DS_Store':
            for filename in os.listdir(os.path.join(base_path, fold)):
                file_path = os.path.join(base_path, fold, filename)
                file_paths.append(file_path)
                labels.append(int(filename.split('-')[2].split('0')[1]) - 1)

    dataset = Emotions_clf(file_paths, labels)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = torch.tensor(labels, dtype=torch.long)

    # Split into train and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleCNN_mfcc(num_classes=len(set(labels.numpy())))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 50
    # Lists to store training and validation loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        # Calculate average validation loss for this epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print("Training complete")

    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-bright')
    # Plotting the loss
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('Loss_cnn_mfcc.png')
    plt.show()

    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_embeddings = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, embd = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_embeddings.extend(embd.cpu().numpy())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Calculate percentages

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_cnn_mfcc.png')
    plt.show()

    # T-SNE visualization
    from sklearn.manifold import TSNE
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(np.array(all_embeddings))

    # Plot t-SNE embeddings
    scatter_x = tsne_embeddings[:, 0]
    scatter_y = tsne_embeddings[:, 1]

    unique_labels = np.unique(all_labels)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = all_labels == label
        label_emotion = emotions[label]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=[colors[i]], label=label_emotion)

    plt.title('t-SNE Embedding Visualization with Different Colors for Each Label')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('tsne_cnn_mfcc.png')
    plt.show()

    # plot tsne for only 2 best labels:
    conf_label = [cm_percentage[k][k] for k in range(len(cm_percentage))]
    max_conf_label = np.argmax(conf_label)
    conf_label[max_conf_label] = 0
    second_max_conf_label = np.argmax(conf_label)

    unique_labels = [max_conf_label, second_max_conf_label]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = all_labels == label
        label_emotion = emotions[label]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=[colors[i]], label=label_emotion)

    plt.title('t-SNE Embedding Visualization with Different Colors for Each Label')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('tsne_2labels_mfcc.png')
    plt.show()
    print('PyCharm')
