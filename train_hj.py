import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import yaml

"""
We can further try ...
- GRU instead of self.embedding (if it's sequential data)
"""

""" 
Configuration
"""
np.random.seed(42)
batch = 64
embed_dim = 128
epochs = 100
save_dir = 'data/reports/training_logs_hj'

""" 
Data Loader
"""
def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

data_conf = read_config('conf/data/realworld_wisdm.yaml')
# Load data
X_path = os.path.join(data_conf['data_root'], os.path.basename(data_conf["X_path"]))
Y_path = os.path.join(data_conf['data_root'], os.path.basename(data_conf["Y_path"]))
PID_path = os.path.join(data_conf['data_root'], os.path.basename(data_conf["PID_path"]))
X = np.load(X_path)   # (15317, 300, 3); data history
_, seq_len, input_dim = X.shape
# print(f"X shape: {X.shape}")
Y = np.load(Y_path)   # (15317); annotation of each data ['Sitting', 'Running', 'Sitting', ... ]
classes = np.unique(Y)
num_classes = len(classes)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
Y_numeric = np.array([class_to_idx[y] for y in Y], dtype=np.int64) 

pid = np.load(PID_path) # (15317); user ID

# Random split - #can split based on user
indices = np.arange(len(X))
np.random.shuffle(indices)
split = int(0.85 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
Y_train, Y_test = Y_numeric[indices[:split]], Y_numeric[indices[split:]]

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=64)

""" 
Model Definition
"""
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
    def __init__(self, input_dim=3, embed_dim=128, seq_length=300, num_heads=4, num_layers=2, num_classes=num_classes):
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

# Training and Evaluation
def train_and_evaluate(model, model_name, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    logs = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
        logs['train_loss'].append(total_loss / total)
        logs['train_acc'].append(correct / total)

        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
        logs['test_loss'].append(total_loss / total)
        logs['test_acc'].append(correct / total)

        print(f"{model_name} | Epoch {epoch}: Train Acc {logs['train_acc'][-1]:.4f}, Test Acc {logs['test_acc'][-1]:.4f}")

    return logs

"""
Run
"""
results = {}
os.makedirs(save_dir, exist_ok=True)

# MLP baseline
mlp_model = IMUMLPClassifier(input_dim=seq_len*input_dim, embed_dim=embed_dim, num_classes=num_classes)
results["MLP"] = train_and_evaluate(mlp_model, "MLP", epochs = epochs)

# Transformer (with heads = 2,4,8 for ablation)
for heads in [2, 4, 8]:
    transformer_model = IMUTransformerClassifier(input_dim=input_dim, embed_dim=embed_dim, seq_length=seq_len, num_heads=heads, num_layers=2, num_classes=num_classes)
    key = f"Transformer_{heads}h"
    results[key] = train_and_evaluate(transformer_model, key, epochs = epochs)

# Save logs
for key, log in results.items():
    np.savez(f"{save_dir}/{key}_log.npz", **log)

# Plotting
import matplotlib.pyplot as plt
for key, log in results.items():
    plt.figure()
    plt.plot(log['train_acc'], label='Train Acc')
    plt.plot(log['test_acc'], label='Test Acc')
    plt.title(f"{key} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{key}_accuracy_curve.png")
    plt.close()

    # plt.figure()
    # plt.plot(log['train_loss'], label='Train Loss')
    # plt.plot(log['test_loss'], label='Test Loss')
    # plt.title(f"{key} Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{save_dir}/{key}_loss_curve.png")
    # plt.close()
