import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import yaml
import os
import torch
import torch.nn as nn
from sslearning.models.accNet import cnn1, SSLNET, Resnet, EncoderMLP
import hydra
from matplotlib.colors import Normalize


def init_model(cfg, my_device):
    if cfg.model.resnet_version > 0:
        model = Resnet(
            output_size=cfg.data.output_size,
            is_eva=True,
            resnet_version=cfg.model.resnet_version,
            epoch_len=cfg.dataloader.epoch_len,
        )
    elif cfg.model.name.split("_")[0] == "MLP":
        model = IMUMLPClassifier(
            input_dim=cfg.data.input_size,
            embed_dim=cfg.model.embed_dim,
            num_classes=cfg.data.output_size,
        )
    elif cfg.model.name.split("_")[0] == "Transformer":
        model = IMUTransformerClassifier(
            input_dim=cfg.evaluation.input_size,
            embed_dim=cfg.model.embed_dim,
            seq_length=cfg.evaluation.input_size,
            num_heads=cfg.model.transformer_num_heads,
            num_layers=cfg.model.transformer_num_layers,
            num_classes=cfg.data.output_size
        )
    else:
        model = SSLNET(
            output_size=cfg.data.output_size, flatten_size=1024
        )  # VGG

    if cfg.multi_gpu:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)

    print(model)
    model.to(my_device, dtype=torch.float)
    return model
class IMUMLPClassifier(nn.Module):
    def __init__(self, input_dim=300*3, embed_dim=64, num_classes=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        self.hidden_dim = hidden_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x_embedding = self.embedding(self.flatten(x))
        x_encoded = self.mlp(x_embedding)
        output = self.head(x_encoded)
        return output
    
class IMUTransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, seq_length=300, num_heads=4, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim*4)#, batch_first=True) not supported in Pytorch 1.7.0
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B, T, _ = x.shape
        x = self.embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embedding(pos)
        x = x.transpose(0, 1)  # Transpose to (T, B, embed_dim) comment this if using batch_first=True and PyTorch >= 1.9.0
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Transpose back to (B, T, embed_dim) comment this if using batch_first=True and PyTorch >= 1.9.0
        x = x.mean(dim=1)  # Global average pooling over time
        out = self.head(x)
        return out

def read_dataset(cfg):
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
    # # Load the YAML configuration
    # with open(yaml_path, 'r') as file:
    #     config = yaml.safe_load(file)

    # # Extract data_root, paths to X and Y, and dataset_name from the configuration
    # data_root = config.get('data_root', '')
    # x_path = config.get('X_path', '').replace('${data.data_root}', data_root)
    # y_path = config.get('Y_path', '').replace('${data.data_root}', data_root)
    # dataset_name = config.get('dataset_name', 'Unknown')

    # if not x_path or not y_path:
    #     raise ValueError("The YAML file must specify 'X_path' and 'Y_path'.")

    # Load the data
    X = np.load(cfg.evaluation_data.X_path)
    Y = np.load(cfg.evaluation_data.Y_path, allow_pickle=True)
    P = np.load(cfg.evaluation_data.PID_path, allow_pickle=True)
    dataset_name = cfg.evaluation_data.dataset_name

    return X, Y, P, dataset_name

def read_features(cfg):
    """
    Reads features from a precomputed file if it exists, otherwise returns None.

    Parameters:
        cfg: Configuration object containing the path to the feature file.

    Returns:
        numpy.ndarray or None: Loaded features if the file exists, otherwise None.
    """
    # Extract the directory of the X_path
    feature_dir = os.path.dirname(cfg.evaluation_data.X_path)

    # Look for a file matching the pattern "extracted_features*.npz"
    feature_files = [f for f in os.listdir(feature_dir) if f.startswith("extracted_features") and f.endswith(".npz")]

    if feature_files:
        # Load the first matching feature file
        feature_path = os.path.join(feature_dir, feature_files[-1])
        print(f"Loading features from {feature_path}")
        with np.load(feature_path) as data:
            features = data['features']
        return features
    else:
        print("No precomputed feature file found.")
        return None
def read_logits(cfg):
    """
    Reads logits from a precomputed file if it exists, otherwise returns None.

    Parameters:
        cfg: Configuration object containing the path to the logits file.

    Returns:
        numpy.ndarray or None: Loaded logits if the file exists, otherwise None.
    """
    # Extract the directory of the X_path
    feature_dir = os.path.dirname(cfg.evaluation_data.X_path)

    # Look for a file matching the pattern "extracted_logits*.npz"
    logit_file = [f for f in os.listdir(feature_dir) if f.startswith("personalized_logits") and f.endswith(".npz")]

    if logit_file:
        # Load the first matching feature file
        feature_path = os.path.join(feature_dir, logit_file[-1])
        print(f"Loading features from {feature_path}")
        with np.load(feature_path) as data:
            features = data['logits']
        return features
    else:
        print("No precomputed feature file found.")
        return None

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
        scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', len(unique_classes)), s=30)
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

def plot_tsne_multi_modes1(X, features,logits,P, Y=None, dataset_name="Dataset", random_state=42, save_path=None):
    """
    Plots t-SNE visualizations for raw input and extracted features side by side if both exist.

    Parameters:
        X (numpy.ndarray): Raw input data.
        features (numpy.ndarray or None): Extracted features.
        Y (numpy.ndarray, optional): Labels for coloring the points. Default is None.
        dataset_name (str, optional): Name of the dataset. Default is "Dataset".
        random_state (int, optional): Random state for reproducibility. Default is 42.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed. Default is None.

    Returns:
        None
    """
    # Map string labels to numbers if y is provided
    label_encoder = None
    if Y is not None and isinstance(Y[0], str):
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        unique_classes = np.unique(Y)
        # Create a colormap and normalization object
    cmap = plt.cm.get_cmap('viridis', len(unique_classes))
    norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes))
    if P is not None:
        participants = np.unique(P)
        print(f"Unique participants: {participants}")
    fig, axes = plt.subplots(len(participants), 3 if features is not None else 1, figsize=(20, 6*len(participants)))
    for index,participant in enumerate(participants):
        # axes[index].set_title(f"Participant {participant}")
        axes[index][0].set_title(f"Raw Input ({dataset_name})")
        axes[index][1].set_title(f"Extracted Features ({dataset_name})")
        axes[index][2].set_title(f"Logits ({dataset_name})")
        # Filter data for the current participant
        X_participant = X[P == participant]
        Y_participant = Y[P == participant]
        features_participant = features[P == participant] if features is not None else None
        logits_participant = logits[P == participant] if logits is not None else None
        # Plot t-SNE for raw input
        tsne_raw = TSNE(n_components=2, random_state=random_state)
        X_embedded = tsne_raw.fit_transform(X_participant)
        axes[index][0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30)
        axes[index][0].set_xticks([])
        axes[index][0].set_yticks([])
        
        # Plot t-SNE for extracted features
        if features_participant is not None:
            tsne_features = TSNE(n_components=2, random_state=random_state)
            features_embedded = tsne_features.fit_transform(features_participant)
            scatter = axes[index][1].scatter(features_embedded[:, 0], features_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30)
            axes[index][1].set_xticks([])
            axes[index][1].set_yticks([])
        
        if logits_participant is not None:
            # Plot t-SNE for logits
            tsne_logits = TSNE(n_components=2, random_state=random_state)
            logits_embedded = tsne_logits.fit_transform(logits_participant)
            axes[index][2].scatter(logits_embedded[:, 0], logits_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30)
            axes[index][2].set_xticks([])
            axes[index][2].set_yticks([])
        # Add a colorbar to the second plot
        # cbar = fig.colorbar(scatter, ax=axes[index][2], ticks=unique_classes)
        # cbar.set_label('Class Labels')
        # if label_encoder is not None:
        #     cbar.set_ticklabels(label_encoder.inverse_transform(unique_classes))
        # else:
        #     cbar.set_ticklabels(unique_classes)
# Create a single legend for all subplots
    handles, labels = scatter.legend_elements()  # Get legend elements from the scatter plot
    print(handles, labels)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5, title="Activities")
    plt.subplots_adjust(bottom=0.2)
    # if features is not None:
    #     # Plot t-SNE for raw input
    #     tsne_raw = TSNE(n_components=2, random_state=random_state)
    #     X_embedded = tsne_raw.fit_transform(X)
    #     axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap=plt.cm.get_cmap('viridis', len(np.unique(Y))), s=30)
    #     axes[0].set_title(f"Raw Input ({dataset_name})")
    #     axes[0].set_xticks([])
    #     axes[0].set_yticks([])
    #     # axes[0].set_aspect('equal') 
    #     # axes[0].set_xlabel("t-SNE Dimension 1")
    #     # axes[0].set_ylabel("t-SNE Dimension 2")

    #     # Plot t-SNE for extracted features
    #     tsne_features = TSNE(n_components=2, random_state=random_state)
    #     features_embedded = tsne_features.fit_transform(features)
    #     scatter = axes[1].scatter(features_embedded[:, 0], features_embedded[:, 1], c=Y, cmap=plt.cm.get_cmap('viridis', len(np.unique(Y))), s=30)
    #     axes[1].set_title(f"Extracted Features ({dataset_name})")
    #     axes[1].set_xticks([])
    #     axes[1].set_yticks([])
    #     # axes[1].set_aspect('equal') 

    #     if logits is not None:
    #         # Plot t-SNE for logits
    #         tsne_logits = TSNE(n_components=2, random_state=random_state)
    #         logits_embedded = tsne_logits.fit_transform(logits)
    #         axes[2].scatter(logits_embedded[:, 0], logits_embedded[:, 1], c=Y, cmap=plt.cm.get_cmap('viridis', len(np.unique(Y))), s=30)
    #         axes[2].set_title(f"Logits ({dataset_name})")
    #         axes[2].set_xticks([])
    #         axes[2].set_yticks([])
    #         # axes[2].set_aspect('equal') 
    #     # axes[1].set_xlabel("t-SNE Dimension 1")
    #     # axes[1].set_ylabel("t-SNE Dimension 2")

    #     # # Add a colorbar to the second plot
    #     # cbar = fig.colorbar(scatter, ax=axes[2], ticks=np.unique(Y))
    #     # cbar.set_label('Class Labels')
    #     # if label_encoder is not None:
    #     #     cbar.set_ticklabels(label_encoder.inverse_transform(unique_classes))
    #     # else:
    #     #     cbar.set_ticklabels(unique_classes)
    # else:
    #     # Plot t-SNE for raw input only
    #     tsne_raw = TSNE(n_components=2, random_state=random_state)
    #     X_embedded = tsne_raw.fit_transform(X)
    #     axes.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, cmap=plt.cm.get_cmap('viridis', len(np.unique(Y))), s=30)
    #     axes.set_title(f"Raw Input ({dataset_name})")
    #     # axes.set_xlabel("t-SNE Dimension 1")
    #     # axes.set_ylabel("t-SNE Dimension 2")

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_tsne_multi_modes(X, features, logits, P, Y=None, dataset_name="Dataset", random_state=42, save_path=None):
    """
    Plots t-SNE visualizations for raw input, extracted features, and logits side by side for each participant.
    Adds a common legend at the bottom of the figure.

    Parameters:
        X (numpy.ndarray): Raw input data.
        features (numpy.ndarray or None): Extracted features.
        logits (numpy.ndarray or None): Logits from the model.
        P (numpy.ndarray): Participant identifiers for each sample.
        Y (numpy.ndarray, optional): Labels for coloring the points. Default is None.
        dataset_name (str, optional): Name of the dataset. Default is "Dataset".
        random_state (int, optional): Random state for reproducibility. Default is 42.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed. Default is None.

    Returns:
        None
    """
    # Map string labels to numbers if y is provided
    label_encoder = None
    if Y is not None and isinstance(Y[0], str):
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        unique_classes = np.unique(Y)
    # unique_classes = np.unique(Y) if Y is not None else []

    cmap = plt.cm.get_cmap('viridis', len(unique_classes)) 
    norm = Normalize(vmin=min(unique_classes), vmax=max(unique_classes)) 
    label_names = label_encoder.inverse_transform(np.array(sorted(unique_classes)))
    # Create custom legend handles and labels
    handles = [plt.Rectangle((0, 0), width=10, height=4, color=cmap(norm(i))) for i in unique_classes]
    labels_legend = [label_names[i] for i in range(len(unique_classes))]

    participants = np.unique(P)
    participants = sorted(participants, key=lambda x: int(x.split('_p')[-1]))
    print(f"Unique participants: {participants}")
    num_participants = len(participants)
    num_cols = 3 if features is not None and logits is not None else (2 if features is not None or logits is not None else 1)
    fig, axes = plt.subplots(num_participants, num_cols, figsize=(20, 6 * num_participants))
    if num_participants == 1:
        axes = [axes]  # Make sure axes is always a list of lists for consistent indexing

    scatter_elements = []
    label_names = []

    for index, participant in enumerate(participants):
        row_axes = axes[index]

        # Filter data for the current participant
        X_participant = X[P == participant]
        Y_participant = Y[P == participant] if Y is not None else None
        features_participant = features[P == participant] if features is not None else None
        logits_participant = logits[P == participant] if logits is not None else None

        # Plot t-SNE for raw input
        tsne_raw = TSNE(n_components=2, random_state=random_state)
        X_embedded = tsne_raw.fit_transform(X_participant)
        scatter_raw = row_axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30) if Y_participant is not None else row_axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], s=30)
        row_axes[0].set_title(f"Raw Input (Participant {participant}, {dataset_name})")
        row_axes[0].set_xticks([])
        row_axes[0].set_yticks([])

        col_idx = 1
        # Plot t-SNE for extracted features
        if features_participant is not None:
            tsne_features = TSNE(n_components=2, random_state=random_state)
            features_embedded = tsne_features.fit_transform(features_participant)
            scatter_features = row_axes[col_idx].scatter(features_embedded[:, 0], features_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30) if Y_participant is not None else row_axes[col_idx].scatter(features_embedded[:, 0], features_embedded[:, 1], s=30)
            row_axes[col_idx].set_title(f"Extracted Features (Participant {participant}, {dataset_name})")
            row_axes[col_idx].set_xticks([])
            row_axes[col_idx].set_yticks([])
            col_idx += 1

        # Plot t-SNE for logits
        if logits_participant is not None:
            tsne_logits = TSNE(n_components=2, random_state=random_state)
            logits_embedded = tsne_logits.fit_transform(logits_participant)
            scatter_logits = row_axes[col_idx].scatter(logits_embedded[:, 0], logits_embedded[:, 1], c=Y_participant, cmap=cmap, norm=norm, s=30) if Y_participant is not None else row_axes[col_idx].scatter(logits_embedded[:, 0], logits_embedded[:, 1], s=30)
            row_axes[col_idx].set_title(f"Logits (Participant {participant}, {dataset_name})")
            row_axes[col_idx].set_xticks([])
            row_axes[col_idx].set_yticks([])
        
        # Collect scatter elements and labels from the 4th participant's P13 feature plot (if it exists)
        if index == 4 and features_participant is not None and Y is not None:
            scatter_elements, label_indices = scatter_features.legend_elements()
            if label_encoder:
                label_names = label_encoder.inverse_transform(np.array(sorted(unique_classes)))
            else:
                label_names = [f"Class {i}" for i in sorted(unique_classes)]
    
    # if scatter_elements.any() and label_names.any():
    # legend = fig.legend(scatter_elements, label_names, loc="lower center", bbox_to_anchor=(0.5, 1 - (0.05 * num_participants)), ncol=len(unique_classes) // 2 + 1, fontsize='large', title="Activities")
    # legend.get_title().set_fontsize('large')
    # plt.subplots_adjust(bottom=0.0)
    # else:
    #     plt.tight_layout()
        # Try adding the legend to the axes
    legend = row_axes[1].legend(handles, labels_legend, loc="lower center", 
                                bbox_to_anchor=(0.5, -0.25), ncol=len(unique_classes) // 2 + 1, fontsize='x-large', title="Activities")
    legend.get_title().set_fontsize('x-large')
    # Adjust spacing between legend items
    plt.rcParams['legend.handlelength'] = 5  # Length of the legend handles
    plt.rcParams['legend.handleheight'] = 5  # Height of the legend handles
    plt.rcParams['legend.labelspacing'] = 3  # Spacing between legend labels
    # Adjust layout to make space for the legend
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for title if needed

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def initialize_model_and_extract_features(model, weight_path, input_data):
    """
    Initialize a model, load its weights, and use it to extract features using the feature_extractor module.

    Parameters:
        model_class (nn.Module): The class of the model to initialize.
        weight_path (str): Path to the .pt file containing the model weights.
        input_data (torch.Tensor): Input data to extract features from.

    Returns:
        torch.Tensor: Extracted features from the feature_extractor module.
    """
    # Initialize the model
    # model = model_class()
    
    # Load the model weights
    model.load_state_dict(torch.load(weight_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {weight_path}")   
    # Extract the feature_extractor module
    feature_extractor = model.module.feature_extractor if isinstance(model, nn.DataParallel) else model.feature_extractor

    # Extract features
    with torch.no_grad():
        features = feature_extractor(input_data)

    return features

@hydra.main(config_path="conf", config_name="config_eva_pretrained")
def main(cfg):

    # Detect the available device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_FEATURE = True
    # Initialize the model using the init_model function
    # model = init_model(cfg, device)
    # Read dataset from YAML
    X, Y, P, dataset_name = read_dataset(cfg)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, dataset_name: {dataset_name}")
    
    features = read_features(cfg)
    logits = read_logits(cfg)
    print(f"logits shape: {logits.shape}")
    print(f"features shape: {features.shape}")
    if features is None:
        # If no precomputed features are found, initialize the model and extract features
        print("No precomputed features found. Initializing model and extracting features...")
    
        # features = initialize_model_and_extract_features(cfg, device,  "/home/hossein/ssl-wearables/model_check_point/2025-04-15_00:30:31tmp.pt", X)
        # model = init_model(cfg, device)
        # print(f"Model loading...")
        # features = initialize_model_and_extract_features(model,  "/home/hossein/ssl-wearables/model_check_point/2025-04-08_10:30:51tmp.pt", X)
        
    

    # Update the title to include the dataset name
    title = f"t-SNE Visualization of {dataset_name} dataset"

    # Update the save path to include the dataset name
    # if USE_FEATURE:
    #     # model_name = cfg.model.name.split("_")[0]
    #     save_path = f"/home/hossein/ssl-wearables/plots/imgs/tsne_plots_{dataset_name}_features.png"
    # else:
    #     save_path = f"/home/hossein/ssl-wearables/plots/imgs/tsne_plots_{dataset_name}.png"
    save_path = f"/home/hossein/ssl-wearables/plots/imgs/tsne_plots_{dataset_name}_raw_features_logits.png"
    print(f"save_path: {save_path}")
    # Plot t-SNE visualization and save it
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
        print(f"X reshaped to: {X.shape}")
    # # Flatten the last two dimensions of X if it has 3 dimensions
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)  # Shape becomes (num_samples, 300*3)
        print(f"features reshaped to: {features.shape}")
    if logits.ndim == 3:
        logits = logits.reshape(logits.shape[0], -1)
        print(f"logits reshaped to: {logits.shape}")
    # plot_tsne_2d(X, y=Y, title=title, random_state=42, save_path=save_path)
    plot_tsne_multi_modes(X, features, logits, P, Y=Y, dataset_name=dataset_name, random_state=42, save_path=save_path)
if __name__ == "__main__":
    main()