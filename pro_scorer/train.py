import sys
sys.path.insert(0, '/home/itm/msadasivam/simcortex')
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from pro_scorer.scorer import PROScorer

class PRODataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        latent = torch.tensor(sample['latent'], dtype=torch.float32)
        labels = torch.tensor([
            sample['progress'],
            sample['risk'],
            sample['efficiency'],
        ], dtype=torch.float32)
        return latent, labels


def train():
    print("="*50)
    print("Training PRO Scorer on mimic-video latents")
    print("="*50)

    # load dataset
    dataset_path = '/home/itm/msadasivam/simcortex/data/pro_dataset_mimic.pkl'
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total samples: {len(data)}")
    print(f"Perturbations: {set(d['perturbation'] for d in data)}")

    # print label distributions
    for pert in ['normal', 'slippery', 'heavy', 'extreme_pose']:
        subset = [d for d in data if d['perturbation'] == pert]
        if subset:
            risks = [d['risk'] for d in subset]
            progs = [d['progress'] for d in subset]
            print(f"  {pert}: n={len(subset)} risk={np.mean(risks):.3f} progress={np.mean(progs):.3f}")

    # split train/val
    np.random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(PRODataset(train_data), batch_size=8, shuffle=True)
    val_loader = DataLoader(PRODataset(val_data), batch_size=8)

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    model = PROScorer(latent_dim=2048).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # train
    best_val_loss = float('inf')
    epochs = 100

    print(f"\nTraining for {epochs} epochs...")
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    print("-"*50)

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        for latents, labels in train_loader:
            latents = latents.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(latents)

            pred = torch.stack([
                out['progress'],
                out['risk'],
                out['efficiency']
            ], dim=1)

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for latents, labels in val_loader:
                latents = latents.to(device)
                labels = labels.to(device)
                out = model(latents)
                pred = torch.stack([
                    out['progress'],
                    out['risk'],
                    out['efficiency']
                ], dim=1)
                val_loss += criterion(pred, labels).item()
        val_loss /= max(len(val_loader), 1)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                '/home/itm/msadasivam/simcortex/pro_scorer/pro_weights_mimic.pt')

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Saved: pro_scorer/pro_weights_mimic.pt")
    print("\nDone!")


if __name__ == '__main__':
    train()
