import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import yaml
import os

def read_dataset_from_yaml(yaml_path):
    """
    Reads dataset (X, Y, and dataset_name) using a YAML configuration file.

    Parameters:
        yaml_path (str): Path to the YAML file specifying dataset configurations.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Feature data.
            - Y (numpy.ndarray): Labels.
            - dataset_name (str): Name of the dataset.
    """
    # Load the YAML configuration
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract data_root, paths to X and Y, and dataset_name from the configuration
    data_root = config.get('data_root', '')
    x_path = config.get('X_path', '').replace('${data.data_root}', data_root)
    y_path = config.get('Y_path', '').replace('${data.data_root}', data_root)
    dataset_name = config.get('dataset_name', 'Unknown')

    if not x_path or not y_path:
        raise ValueError("The YAML file must specify 'X_path' and 'Y_path'.")

    # Load the data
    X = np.load(x_path)
    Y = np.load(y_path)

    return X, Y, dataset_name

def plot_tsne_2d(x, y=None, title="t-SNE Visualization", random_state=42, save_path=None):
    """
    Plots a 2D t-SNE visualization and optionally saves it to a file.

    Parameters:
        x (numpy.ndarray): Input data of shape (n_samples, n_features).
        y (numpy.ndarray, optional): Labels for coloring the points. Default is None.
        title (str, optional): Title of the plot. Default is "t-SNE Visualization".
        random_state (int, optional): Random state for reproducibility. Default is 42.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed. Default is None.

    Returns:
        None
    """
    # Map string labels to numbers if y is provided
    label_encoder = None
    if y is not None and isinstance(y[0], str):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=random_state)
    x_embedded = tsne.fit_transform(x)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    if y is not None:
        unique_classes = np.unique(y)
        scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap=plt.cm.get_cmap('tab10', len(unique_classes)), s=30)
        cbar = plt.colorbar(scatter, ticks=unique_classes)
        cbar.set_label('Class Labels')
        if label_encoder is not None:
            cbar.set_ticklabels(label_encoder.inverse_transform(unique_classes))
        else:
            cbar.set_ticklabels(unique_classes)
    else:
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=30)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Read dataset from YAML and plot t-SNE visualization.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--title", type=str, default="t-SNE Visualization", help="Title of the t-SNE plot.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--save_path", type=str, default="plots/imgs/tsne_plot.png", help="Path to save the plot.")
    args = parser.parse_args()

    # Read dataset from YAML
    X, Y, dataset_name = read_dataset_from_yaml(args.yaml_path)

    # Flatten the last two dimensions of X if it has 3 dimensions
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  # Shape becomes (num_samples, 300*3)

    # Update the title to include the dataset name
    title = f"{args.title} of {dataset_name} dataset"

    # Update the save path to include the dataset name
    save_path = args.save_path.replace('.png', f'_{dataset_name}.png')

    # Plot t-SNE visualization and save it
    plot_tsne_2d(X, y=Y, title=title, random_state=args.random_state, save_path=save_path)