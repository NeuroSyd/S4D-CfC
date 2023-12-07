# Code created by Zhaojing Huang for evaluating the model on 05 December 2023

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import onnx
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

import os
import argparse

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D
from tqdm.auto import tqdm
from Wavenet import WaveNetClassifier
import csv

from ncps import wirings
from ncps.torch import CfC, LTC

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch Predicting')
parser.add_argument('--file_name', default='S4D_NCP_test1', type=str, help='Folder Name')
# Optimizer
parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
# Scheduler
# parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--epochs', default=300, type=int, help='Training epochs')
# Dataset
# parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
# parser.add_argument('--grayscale', action='store_true', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=1, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

# Define the directory path and file name where you want to save the text file
output_directory = '/s4_results/' + args.file_name

if not os.path.exists(output_directory):
    # If it doesn't exist, create the directory
    os.makedirs(output_directory)
    print(f"Directory '{output_directory}' created successfully.")
else:
    print(f"Directory '{output_directory}' already exists.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_layers = args.n_layers
batch_size =  args.batch_size

d_input = 1
d_output = 6

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # NCP decoder
        wiring = wirings.AutoNCP(20, d_output)
        ncp = CfC(d_model, wiring, batch_first=True)  # , return_sequences=False

        self.decoder = ncp

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x , _ = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

# Open the HDF5 file
with h5py.File('/x.hdf5', 'r') as f:
    y = pd.read_csv('/y.csv').values.reshape(-1, 6)
    X = f['tracings'][:, :, 1].reshape(-1, 4096, 1) #[:]
label = ['1dAVb','RBBB','LBBB','SB','AF','ST']

# Define a custom PyTorch dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.X))

    def __getitem__(self, index):
        # Get the input feature and target label for the given index
        idx = self.indices[index]
        x = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        # Convert to PyTorch tensor and return
        return torch.tensor(x), torch.tensor(label)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.X)


def min_max_normalize(x):
    # Get the shape of the input tensor
    batch_size, num_readings, num_channels = x.shape

    # Reshape the input tensor to (batch_size, num_readings * num_channels)
    x_flat = x.view(batch_size, -1)

    # Calculate the min and max values along the second dimension (num_channels)
    min_values = x_flat.min(dim=1, keepdim=True)[0]
    max_values = x_flat.max(dim=1, keepdim=True)[0]

    # Handle zero division by setting max_values and min_values to 1 for rows where all values are zero
    all_zeros = (min_values == 0) & (max_values == 0)
    max_values[all_zeros] = 1
    min_values[all_zeros] = 0

    # Normalize the data
    normalized_x_flat = (x_flat - min_values) / (max_values - min_values)

    # Reshape the normalized data back to the original shape
    normalized_x = normalized_x_flat.view(batch_size, num_readings, num_channels)

    return normalized_x

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model from .pt file
state_dict = torch.load(output_directory + '/brazil_model_' + str(n_layers) + '_' + str(args.lr) +'.pt') #

# Load the state dictionary into the model
model.load_state_dict(state_dict)
model = model.to(device)

# Perform inference in batches
all_probs = []  # List to store probabilities for all batches

for start_idx in range(0, len(X_val), batch_size):
    end_idx = start_idx + batch_size
    batch_x_val = X_val[start_idx:end_idx]

    # Convert batch_x_val to a PyTorch tensor and move to the device
    batch_x_val_tensor = torch.tensor(batch_x_val, dtype=torch.float32, device=device)
    batch_x_val_tensor = min_max_normalize(batch_x_val_tensor)

    with torch.no_grad():
        outputs = nn.functional.sigmoid(model(batch_x_val_tensor))


    # Convert outputs to probabilities for the positive class
    all_probs.append(outputs.cpu().numpy())  # Store batch probabilities
    all_probs_array = np.concatenate(all_probs, axis=0)
    print(all_probs_array.shape)

# Initialize lists to store per-class metrics
class_metrics = []

for class_index, class_label in enumerate(label):
    y_val_binary = y_val[:, class_index]
    predictions = (all_probs_array[:, class_index] > 0.5).astype(int)

    accuracy = accuracy_score(y_val_binary, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_binary, predictions, average='binary')
    auroc = roc_auc_score(y_val_binary, all_probs_array[:, class_index])

    class_metrics.append({
        'Class': class_label,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'AUROC': auroc,
    })

# Calculate average metrics
average_metrics = {
    'Class': 'Average',
    'Recall': np.mean([metric['Recall'] for metric in class_metrics]),
    'Precision': np.mean([metric['Precision'] for metric in class_metrics]),
    'F1': np.mean([metric['F1'] for metric in class_metrics]),
    'AUROC': np.mean([metric['AUROC'] for metric in class_metrics]),
}

# Append average metrics to the list of class metrics
class_metrics.append(average_metrics)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame(class_metrics)

# Save the DataFrame to a CSV file
metrics_df.to_csv(output_directory + '/eval_metrics.csv', index=False)
# metrics_df.to_csv(output_directory + '/eval_metrics_quantization.csv', index=False)

print('Completed')
