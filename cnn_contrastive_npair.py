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


def create_n_pairs(dataset, labels, n_negatives=4):
    pairs = []
    pair_labels = []

    num_classes = len(set(labels.numpy()))
    label_indices = {i: [] for i in range(num_classes)}

    # Group the data by labels
    for idx, label in enumerate(labels):
        label = int(label.item())
        label_indices[label].append(dataset[idx])

    for idx, sample in enumerate(dataset):
        label = int(labels[idx].item())

        # Create a positive pair (same class)
        positive_pair = random.choice(label_indices[label])
        negative_pairs = []

        for _ in range(n_negatives):
            # Create a negative pair (different class)
            negative_label = random.choice([l for l in range(num_classes) if l != label])
            negative_pair = random.choice(label_indices[negative_label])
            negative_pairs.append(negative_pair)

        pairs.append((sample, positive_pair, negative_pairs))
        pair_labels.append(0)  # Same class, label is 0

    return pairs, pair_labels


# N-Pair Loss Definition
class NPairLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(NPairLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        # Compute the similarity between anchor and positive
        positive_loss = torch.mean((1 - positive)**2)

        # Compute the similarity between anchor and each negative
        negative_loss = torch.mean(torch.clamp(self.margin - negatives, min=0.0)**2)

        # Compute the N-pair loss
        loss = positive_loss + negative_loss

        return loss


# Update your training loop
def train_npair(model, pairs, pair_labels, optimizer, criterion, device='cpu'):
    model.train()
    for (anchor, positive, negatives), label in zip(pairs, pair_labels):
        optimizer.zero_grad()

        # Unpack the tuples if needed
        if isinstance(anchor, tuple):
            anchor = anchor[0]
        if isinstance(positive, tuple):
            positive = positive[0]

        # Convert anchor and positive to tensors
        if isinstance(anchor, list) or isinstance(anchor, np.ndarray):
            anchor = torch.tensor(anchor, dtype=torch.float32).to(device)
        if isinstance(positive, list) or isinstance(positive, np.ndarray):
            positive = torch.tensor(positive, dtype=torch.float32).to(device)

        # Ensure the tensors have the correct dimensions (e.g., add batch dimension if needed)
        if isinstance(anchor, torch.Tensor) and len(anchor.shape) == 3:  # Assuming anchor is 2D (e.g., (128, 300))
            anchor = anchor.unsqueeze(0)  # Convert to (1, 128, 300)
        if isinstance(positive, torch.Tensor) and len(positive.shape) == 3:
            positive = positive.unsqueeze(0)

        # Convert each negative sample to a tensor individually
        negative_embeddings = []
        for neg in negatives:
            if isinstance(neg, tuple):
                neg = neg[0]
            if isinstance(neg, list) or isinstance(neg, np.ndarray):
                neg_tensor = torch.tensor(neg, dtype=torch.float32).to(device)
                if len(neg_tensor.shape) == 3:  # Assuming neg is 2D (e.g., (128, 300))
                    neg_tensor = neg_tensor.unsqueeze(0)  # Convert to (1, 128, 300)
                negative_embeddings.append(neg_tensor)
            if isinstance(neg, torch.Tensor):
                neg_tensor = torch.tensor(neg, dtype=torch.float32).to(device)
                if len(neg_tensor.shape) == 3:  # Assuming neg is 2D (e.g., (128, 300))
                    neg_tensor = neg_tensor.unsqueeze(0)  # Convert to (1, 128, 300)
                negative_embeddings.append(neg_tensor)

        # Ensure negative_embeddings is not empty
        if not negative_embeddings:
            print("No negative samples generated; skipping this batch.")
            continue

        # Forward pass through the model
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embeddings = [model(neg) for neg in negative_embeddings]

        # Compute cosine similarities
        positive_similarity = F.cosine_similarity(anchor_embedding, positive_embedding)
        negative_similarities = torch.stack([F.cosine_similarity(anchor_embedding, neg) for neg in negative_embeddings], dim=0)

        # Compute N-pair loss
        loss = criterion(anchor=None, positive=positive_similarity, negatives=negative_similarities)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    return model


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

    def forward(self, anchor, positive, negatives):
        # Forward pass to get embeddings
        anchor_embedding = self.embedding_net(anchor)
        positive_embedding = self.embedding_net(positive)
        negative_embeddings = [self.embedding_net(neg) for neg in negatives]
        return anchor_embedding, positive_embedding, negative_embeddings


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
        x = self.fc1(x)  # Output the embedding
        return x


if __name__ == '__main__':

    file_paths = []
    labels = []
    base_path = 'dataset/dataset'
    internal_folds = os.listdir(os.path.join(base_path))
    max_len = 300

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
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example usage
    embedding_dim = 128
    embedding_net = SimpleCNN(embedding_dim=embedding_dim)

    # Loss function
    n_pair_loss = NPairLoss(margin=1.0)

    num_epochs = 3
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=0.001)

    # Generate N-pairs
    pairs, pair_labels = create_n_pairs(dataset, labels, n_negatives=4)

    # Example Training Loop
    for epoch in range(num_epochs):
        embd_model = train_npair(embedding_net, pairs, pair_labels, optimizer, n_pair_loss)

    print("Training complete")

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
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    tsne_embeddings = tsne.fit_transform(np.array(all_embeddings))

    # Plot t-SNE embeddings
    scatter_x = tsne_embeddings[:, 0]
    scatter_y = tsne_embeddings[:, 1]

    unique_labels = np.unique(all_labels)
    # unique_labels = np.unique([0, 1])

    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = all_labels == label
        plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], c=[colors[i]], label=label)

    plt.title('t-SNE Embedding Visualization with Different Colors for Each Label')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('tsne_contrastive_npair.png')
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
