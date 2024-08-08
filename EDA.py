import matplotlib.pyplot as plt


def plot_classes_distribution(labels):

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


if __name__ == '__main__':
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    plot_classes_distribution(labels)

    print('PyCharm')