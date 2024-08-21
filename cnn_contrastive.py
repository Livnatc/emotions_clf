import os
import numpy as np
import librosa
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def create_pairs(dataset, labels, inds):
    pairs = []
    pair_labels = []

    num_classes = len(set(labels))  # Ensure labels are converted to numpy before getting the set
    label_indices = {i: [] for i in range(num_classes)}

    # Group the data by labels
    for idx in range(len(inds)):
        label = int(labels[inds[idx]])  # Convert tensor to integer
        label_indices[label].append(dataset[idx])

    for idx, sample in enumerate(dataset):
        label = int(labels[inds[idx]])  # Convert tensor to integer

        # Create a positive pair (same class)
        positive_pair = random.choice(label_indices[label])
        pairs.append((sample, positive_pair))
        pair_labels.append(0)  # Same class, label is 0

        # Create a negative pair (different class)
        negative_label = random.choice([l for l in range(num_classes) if l != label])
        negative_pair = random.choice(label_indices[negative_label])
        pairs.append((sample, negative_pair))
        pair_labels.append(1)  # Different class, label is 1

    return pairs, pair_labels


def extract_features_mfcc(file_path, n_mfcc=13, max_len=300):
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


class SiameseNetworkWithClassifier(nn.Module):
    def __init__(self, embedding_net, num_classes):
        super(SiameseNetworkWithClassifier, self).__init__()
        self.embedding_net = embedding_net
        self.fc1 = nn.Linear(embedding_net.embedding_dim, 64)  # Add a classifier layer
        self.fc2 = nn.Linear(64, num_classes)  # Add a classifier layer

    def forward(self, x):
        embeddings = self.embedding_net(x)
        logits = self.fc1(embeddings)
        logits = self.fc2(logits)
        probabilities = F.softmax(logits, dim=1)
        return probabilities


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
    def __init__(self, embedding_dim, n_mfcc=13, max_len=300):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size of the output from the convolutional layers
        self._to_linear = None
        self.calculate_output_size(n_mfcc, max_len)

        # Fully connected layer to output embeddings
        self.fc1 = nn.Linear(self._to_linear, embedding_dim)
        self.dropout = nn.Dropout(0.4)  # Dropout layer

        self.embedding_dim = embedding_dim  # Set embedding dimension

    def calculate_output_size(self, n_mfcc, max_len):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mfcc, max_len)
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc1(x)  # Output the embedding
        return x


# Contrastive Loss Definition
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between the two embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)

        # Contrastive loss formula
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


if __name__ == '__main__':

    file_paths = []
    labels = []
    base_path = 'dataset/dataset'
    internal_folds = os.listdir(os.path.join(base_path))
    max_len = 300
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']  # Assuming these are your emotion labels

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
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example usage
    embedding_dim = 64
    embedding_net = SimpleCNN(embedding_dim=embedding_dim)

    # Loss function
    contrastive_loss = ContrastiveLoss(margin=1.0)

    num_epochs = 50
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=0.0001)

    pairs, pair_labels = create_pairs(train_dataset, train_dataset.dataset.labels, train_dataset.indices)
    val_pairs, val_pair_labels = create_pairs(val_dataset, val_dataset.dataset.labels, val_dataset.indices)
    # Convert to PyTorch tensors
    pair_labels = torch.tensor(pair_labels, dtype=torch.float32)
    val_pair_labels = torch.tensor(val_pair_labels, dtype=torch.float32)

    train_losses = []
    val_losses = []
    # Example Training Loop
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        for (input1, input2), label in zip(pairs, pair_labels):
            optimizer.zero_grad()

            # Unpack the input pairs (if they are tuples)
            if isinstance(input1, tuple):
                input1 = input1[0]
            if isinstance(input2, tuple):
                input2 = input2[0]

            if isinstance(input1, np.ndarray):
                input1 = torch.from_numpy(input1).float()
            if isinstance(input2, np.ndarray):
                input2 = torch.from_numpy(input2).float()

            # Add batch dimension (assuming input1 and input2 are 2D or 3D arrays)
            input1 = input1.unsqueeze(0)  # Add batch dimension
            input2 = input2.unsqueeze(0)  # Add batch dimension

            # Ensure label is a tensor if not already
            label = torch.tensor(label).float()

            # Forward pass through the Siamese network
            output1 = embedding_net(input1)
            output2 = embedding_net(input2)

            # Compute the contrastive loss
            loss = contrastive_loss(output1, output2, label)
            epoch_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        epoch_loss /= len(pairs)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation loop
        with torch.no_grad():
            val_loss = 0.0
            for (val_input1, val_input2), val_label in zip(val_pairs, val_pair_labels):
                # Unpack the validation input pairs
                if isinstance(val_input1, tuple):
                    val_input1 = val_input1[0]
                if isinstance(val_input2, tuple):
                    val_input2 = val_input2[0]

                if isinstance(val_input1, np.ndarray):
                    val_input1 = torch.from_numpy(val_input1).float()
                if isinstance(val_input2, np.ndarray):
                    val_input2 = torch.from_numpy(val_input2).float()

                # Add batch dimension for validation inputs
                val_input1 = val_input1.unsqueeze(0)
                val_input2 = val_input2.unsqueeze(0)

                # Ensure validation label is a tensor if not already
                val_label = torch.tensor(val_label).float()

                # Forward pass through the Siamese network
                val_output1 = embedding_net(val_input1)
                val_output2 = embedding_net(val_input2)

                # Compute the contrastive loss for validation
                val_loss += contrastive_loss(val_output1, val_output2, val_label).item()

            # Average validation loss over the validation set
            val_loss /= len(val_pairs)
            val_losses.append(val_loss)
            print(f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {val_loss:.4f}")

    print("Training and Validation complete")

    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-bright')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('training_validation_loss_CONTRASTIVE.png')
    plt.show()

    # evaluate the embeddings and visualize them:
    embedding_net.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            embeddings = embedding_net(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    # T-SNE visualization
    from sklearn.manifold import TSNE
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(np.array(all_embeddings))

    # Plot t-SNE embeddings
    scatter_x = tsne_embeddings[:, 0]
    scatter_y = tsne_embeddings[:, 1]

    unique_labels = np.unique(all_labels)
    # unique_labels = np.unique([3, 4])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = all_labels == label
        em_label = emotions[label]
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=[colors[i]], label=em_label)

    plt.title('t-SNE Embedding Visualization with Different Colors for Each Label')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('tsne_contrastive.png')
    plt.show()

    # Assuming embedding_net is your pre-trained network
    model = SiameseNetworkWithClassifier(embedding_net, num_classes=8)

    # Freeze the embedding network's weights
    for param in model.embedding_net.parameters():
        param.requires_grad = False

    # Define optimizer and loss function for the classifier
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train the classifier
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Forward pass through the model
            probabilities = model(inputs)

            # Compute the loss
            loss = criterion(probabilities, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs in test_loader:
            probabilities = model(inputs[0])
            predicted_class = torch.argmax(probabilities, dim=1)
            all_labels.extend(inputs[1].cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

    #
    # # Print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Calculate percentages

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print('PyCharm')
