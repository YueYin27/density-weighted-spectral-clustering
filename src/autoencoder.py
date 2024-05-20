import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure output is in the range [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(data, input_dim, hidden_dim, epochs=50, batch_size=256, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = Autoencoder(input_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    data_loader = DataLoader(TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=batch_size,
                             shuffle=True)

    for epoch in range(epochs):
        for batch in data_loader:
            inputs = batch[0].to(device)
            _, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return autoencoder
