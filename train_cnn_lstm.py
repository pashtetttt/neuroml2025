import os, time, h5py, math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 41  # phoneme count (from LOGIT_TO_PHONEME)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4


# HDF5 dataset utilities

def find_hdf5_files(split="train"):
    base = "data/hdf5_data_final"
    files = []
    for sess in os.listdir(base):
        sess_dir = os.path.join(base, sess)
        if not os.path.isdir(sess_dir):
            continue
        path = os.path.join(sess_dir, f"data_{split}.hdf5")
        if os.path.exists(path):
            files.append(path)
    print(f"Found {len(files)} {split} files.")
    return files


def get_max_channels(hdf5_paths):
    print("Scanning for global max channel count...")
    max_c = 0
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            for tr in f.keys():
                if tr.startswith("trial_"):
                    c = f[tr]["input_features"].shape[0]
                    max_c = max(max_c, c)
    print(f"  → Global max channel count: {max_c}")
    return max_c


def compute_channel_stats(hdf5_paths, max_trials=200):
    print("Computing normalization stats (up to 1024 channels)...")
    max_keep = 1024
    sum_x = np.zeros(max_keep)
    sum_x2 = np.zeros(max_keep)
    total_T, count = 0, 0

    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            for tr in f.keys():
                if not tr.startswith("trial_"):
                    continue
                x = f[tr]["input_features"][:].astype(np.float64)
                C, T = x.shape
                c_use = min(C, max_keep)
                x = x[:c_use, :]
                sum_x[:c_use] += x.sum(axis=1)
                sum_x2[:c_use] += (x ** 2).sum(axis=1)
                total_T += T
                count += 1
                if count >= max_trials:
                    break
        if count >= max_trials:
            break

    mean = sum_x / total_T
    std = np.sqrt(np.maximum((sum_x2 / total_T) - (mean ** 2), 1e-12))
    print(f"  → Used {count} trials for normalization.")
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# Dataset class

class Brain2TextNestedHDF5(Dataset):
    def __init__(self, hdf5_paths, normalize=None, max_channels=None):
        self.paths = hdf5_paths
        self.normalize = normalize
        self.max_channels = max_channels
        self.index = []
        for f_idx, path in enumerate(hdf5_paths):
            with h5py.File(path, "r") as f:
                for tr in f.keys():
                    if tr.startswith("trial_"):
                        self.index.append((f_idx, tr))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f_idx, trial_key = self.index[idx]
        path = self.paths[f_idx]
        with h5py.File(path, "r") as f:
            grp = f[trial_key]
            x = grp["input_features"][:].astype(np.float32)
            y = grp["seq_class_ids"][:].astype(np.int32)
            C, T = x.shape

            # Normalize first up to 1024 channels
            if self.normalize:
                mean, std = self.normalize["mean"], self.normalize["std"]
                c_use = min(len(mean), C)
                x[:c_use, :] = (x[:c_use, :] - mean[:c_use, None]) / np.where(std[:c_use, None] < 1e-6, 1.0, std[:c_use, None])

            # Pad / truncate channels to global max
            if self.max_channels:
                if C < self.max_channels:
                    pad = np.zeros((self.max_channels - C, T), dtype=np.float32)
                    x = np.vstack([x, pad])
                elif C > self.max_channels:
                    x = x[:self.max_channels, :]

        return {
            "feats": torch.from_numpy(x),
            "targets": torch.from_numpy(y),
            "T": T,
            "L": len(y),
        }


# Collate function

def collate_batch(batch):
    batch = sorted(batch, key=lambda b: int(b["T"]), reverse=True)
    feats = [b["feats"] for b in batch]
    targets = [b["targets"] for b in batch]
    T_list = torch.tensor([b["T"] for b in batch])
    L_list = torch.tensor([b["L"] for b in batch])

    C = feats[0].shape[0]
    T_max = int(T_list.max())
    B = len(batch)
    x = torch.zeros((B, C, T_max), dtype=torch.float32)
    for i, f in enumerate(feats):
        x[i, :, :f.shape[1]] = f

    y = torch.cat(targets)
    return {
        "inputs": x,
        "input_lengths": T_list,
        "targets_concat": y,
        "target_lengths": L_list,
    }


# Model

class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512, num_classes)
        self.ctc_log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, input_lengths):
        # x: (B, C, T)
        x = self.cnn(x)  # -> (B, 512, T')
        x = x.permute(0, 2, 1)  # -> (B, T', 512)

        # Correct length after pooling (each MaxPool1d halves T)
        reduced_lengths = torch.div(input_lengths, 4, rounding_mode='floor')

        # Pack the reduced sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logp = self.ctc_log_softmax(self.classifier(out))
        return logp, reduced_lengths



# Training loop

def train():
    train_files = find_hdf5_files("train")
    val_files = find_hdf5_files("val")

    max_channels = get_max_channels(train_files)
    stats = compute_channel_stats(train_files)

    train_ds = Brain2TextNestedHDF5(train_files, normalize=stats, max_channels=max_channels)
    val_ds = Brain2TextNestedHDF5(val_files, normalize=stats, max_channels=max_channels)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_batch)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_batch)

    model = CNNBiLSTMCTC(NUM_CLASSES, in_channels=max_channels).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"\nStarting training on {DEVICE} ({max_channels} channels)...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start_time = time.time()
        epoch_loss = 0.0

        progress = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        for i, batch in enumerate(progress, 1):
            x = batch["inputs"].to(DEVICE)
            in_l = batch["input_lengths"].to(DEVICE)
            y = batch["targets_concat"].to(DEVICE)
            out_l = batch["target_lengths"].to(DEVICE)

            optimizer.zero_grad()
            logp, out_l2 = model(x, in_l)
            logp = logp.permute(1, 0, 2)  # (T, B, C)
            loss = criterion(logp, y, out_l2, out_l)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / i
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(progress) - i)
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "eta": f"{eta/60:.1f}m"})

        print(f"Epoch {epoch} finished in {time.time() - start_time:.1f}s | loss={avg_loss:.4f}")

        # Optional: save checkpoint
        torch.save(model.state_dict(), f"checkpoints/cnn_bilstm_epoch{epoch:02d}.pt")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
