import os, sys, time, glob, math, h5py, argparse, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# =========================
# Phoneme inventory (paper)
# =========================
LOGIT_TO_PHONEME = [
    'BLANK', # 0 CTC blank
    'AA','AE','AH','AO','AW','AY',
    'B','CH','D','DH',
    'EH','ER','EY','F','G',
    'HH','IH','IY','JH','K',
    'L','M','N','NG','OW',
    'OY','P','R','S','SH',
    'T','TH','UH','UW','V',
    'W','Y','Z','ZH',
    ' | ',  # SIL
]
PHONEMES = LOGIT_TO_PHONEME[1:]          # exclude BLANK for diphone classes
SIL_TOKEN = ' | '                         # matches paper's SIL definition
PH2ID = {p:i for i,p in enumerate(LOGIT_TO_PHONEME)}  # includes BLANK & SIL
SIL_ID = PH2ID[SIL_TOKEN]
CTC_BLANK = 0

# ================
# Utility helpers
# ================
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def list_split_files(root: str, split: str) -> List[str]:
    """Find all data_{split}.hdf5 under session folders; skip sessions where the split is missing."""
    pattern = os.path.join(root, "*", f"data_{split}.hdf5")
    return sorted(glob.glob(pattern))

def h5_trial_groups(h5file: str) -> List[str]:
    with h5py.File(h5file, "r") as f:
        return [k for k in f.keys() if k.startswith("trial_")]

def read_trial_shapes(h5file: str, trial: str) -> Tuple[int,int]:
    """Return (C, T) — we support either [T,C] or [C,T] in file."""
    with h5py.File(h5file, "r") as f:
        x = f[trial]["input_features"]
        shape = x.shape
        if len(shape) != 2: raise ValueError(f"{h5file}/{trial}: input_features must be 2D, got {shape}")
        # normalize to (C, T)
        if shape[0] < shape[1]:
            C, T = shape[0], shape[1]
        else:
            C, T = shape[1], shape[0]
    return C, T

def robust_channel_stats(sample_files: List[str], max_trials: int = 300) -> Dict[str, np.ndarray]:
    """
    Compute per-channel robust normalization stats (median and MAD) across multiple sessions
    with variable numbers of channels. Pads to max channel count and ignores NaNs.
    """
    chans = []
    used = 0
    maxC = 0

    # First pass: find global max channel count
    for fpath in sample_files:
        with h5py.File(fpath, "r") as f:
            for k in f.keys():
                if not k.startswith("trial_"):
                    continue
                x = np.array(f[k]["input_features"])
                C = x.shape[0] if x.shape[0] < x.shape[1] else x.shape[1]
                if C > maxC:
                    maxC = C
                used += 1
                if used >= max_trials:
                    break
        if used >= max_trials:
            break

    # Second pass: collect with padding
    used = 0
    for fpath in sample_files:
        with h5py.File(fpath, "r") as f:
            for k in f.keys():
                if not k.startswith("trial_"):
                    continue
                x = np.array(f[k]["input_features"]).astype(np.float32)

                # ensure shape (T, C)
                if x.shape[1] < x.shape[0]:
                    x = x.T  # if second dim < first, time is likely along axis 0

                T, C = x.shape
                if T > 1200:
                    s = np.random.randint(0, T - 1200)
                    x = x[s:s+1200]

                # pad to maxC (number of channels)
                pad = np.full((x.shape[0], maxC), np.nan, dtype=np.float32)
                pad[:, :C] = x
                chans.append(pad)
                used += 1
                if used >= max_trials:
                    break
        if used >= max_trials:
            break


    X = np.concatenate(chans, axis=0)  # (sum_T, maxC)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med[None, :]), axis=0)
    mad = np.where(mad < 1e-6, 1.0, mad)
    print(f"  → Used {used} trials for normalization. Global max channels: {maxC}")
    return {"median": med.astype(np.float32), "mad": mad.astype(np.float32), "n_channels": np.array([maxC], dtype=np.int32)}


def zscore(x: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return (x - med[None, :]) / mad[None, :]

def make_diphone_map() -> Tuple[Dict[Tuple[int,int], int], Dict[int, Tuple[int,int]], int, int, int]:
    """
    Build mapping for diphone (prev -> curr).
    prev ∈ {SIL} ∪ {all non-BLANK phonemes}, curr ∈ {all non-BLANK phonemes}.
    We keep a separate CTC blank class for diphone head.
    Returns: (pair2id, id2pair, n_prev, n_curr, n_diphone_classes_incl_blank)
    """
    nonblank_ph_ids = [i for i,p in enumerate(LOGIT_TO_PHONEME) if p != 'BLANK']
    curr_ids = [i for i in nonblank_ph_ids if i != SIL_ID]  # curr cannot be SIL per usual definition
    prev_ids = [SIL_ID] + curr_ids                           # allow SIL as prev

    pair2id = {}
    id2pair = {}
    idx = 0
    for pj in prev_ids:
        for ck in curr_ids:
            pair2id[(pj, ck)] = idx
            id2pair[idx] = (pj, ck)
            idx += 1
    n_prev, n_curr = len(prev_ids), len(curr_ids)
    n_pairs = idx
    # +1 diphone-blank for CTC on the diphone stream
    return pair2id, id2pair, n_prev, n_curr, n_pairs + 1

def phonemes_to_diphones(seq: List[int]) -> List[int]:
    """Convert phoneme id sequence (LOGIT_TO_PHONEME ids) into diphone ids (pair prev->curr).
       Insert SIL at sentence start. Ignore BLANKs in the label sequence."""
    pairs = []
    prev = SIL_ID
    for pid in seq:
        if pid == CTC_BLANK:  # skip blanks in labels
            continue
        if pid == SIL_ID:
            prev = SIL_ID
            continue
        key = (prev, pid)
        if key in DIP_PAIR2ID:
            pairs.append(DIP_PAIR2ID[key])
        prev = pid
    return pairs

# ==================
# Dataset + Collate
# ==================
@dataclass
class TrialRef:
    path: str
    trial: str
    C: int
    T: int
    y_ph: np.ndarray  # int phoneme ids
    y_len: int

class BrainTextDCoND(Dataset):
    def __init__(self, files: List[str], stats: Dict[str,np.ndarray], patch_w=14, patch_stride=7, seed=1337):
        self.files = files
        self.stats = stats
        self.patch_w = patch_w
        self.patch_stride = patch_stride
        self.trials: List[TrialRef] = []
        set_seed(seed)
        self._index_trials()

    def _index_trials(self):
        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                for k in f.keys():
                    if not k.startswith("trial_"): continue
                    grp = f[k]
                    if "input_features" not in grp or "seq_class_ids" not in grp: continue
                    x = grp["input_features"]
                    C, T = (x.shape[0], x.shape[1]) if x.shape[0] < x.shape[1] else (x.shape[1], x.shape[0])
                    y = np.array(grp["seq_class_ids"]).astype(np.int32)
                    # filter empty labels
                    if y.size == 0: continue
                    self.trials.append(TrialRef(fpath, k, C, T, y, len(y)))
        random.shuffle(self.trials)

    def __len__(self): return len(self.trials)

    def __getitem__(self, idx):
        ref = self.trials[idx]
        with h5py.File(ref.path, "r") as f:
            x = np.array(f[ref.trial]["input_features"]).astype(np.float32)
            # to (T,C)
            if x.shape[0] > x.shape[1]: x = x.T
        # robust z-score per channel; if stats.C < x.C, tile; if stats.C > x.C, crop
        med, mad = self.stats["median"], self.stats["mad"]
        if med.shape[0] != x.shape[1]:
            if med.shape[0] < x.shape[1]:
                reps = int(np.ceil(x.shape[1]/med.shape[0]))
                med = np.tile(med, reps)[:x.shape[1]]
                mad = np.tile(mad, reps)[:x.shape[1]]
            else:  # more stats than channels -> crop
                med = med[:x.shape[1]]
                mad = mad[:x.shape[1]]
        x = zscore(x, med, mad)  # (T,C)

        # ---- Patchify along time: produce sequence length T' with feature dim C*W
        T, C = x.shape
        W, S = self.patch_w, self.patch_stride
        if T < W:
            # pad time with zeros to at least one patch
            pad = np.zeros((W - T, C), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
            T = x.shape[0]
        starts = list(range(0, max(1, T - W + 1), S))
        patches = [x[s:s+W].reshape(-1) for s in starts]
        Xp = np.stack(patches, axis=0)  # (T', C*W)

        # labels
        y_ph = ref.y_ph
        y_di = np.array(phonemes_to_diphones(list(y_ph)), dtype=np.int32)

        return torch.from_numpy(Xp), torch.from_numpy(y_ph), torch.from_numpy(y_di)

def collate_batch(batch):
    """Pad time dim for inputs; pack later for RNN. We also keep label lengths."""
    xs, ys_ph, ys_di = zip(*batch)
    Tlens = [x.shape[0] for x in xs]
    Fdim  = xs[0].shape[1]
    B     = len(xs)

    maxT = max(Tlens)
    Xpad = torch.zeros(B, maxT, Fdim, dtype=xs[0].dtype)
    for i,x in enumerate(xs):
        Xpad[i, :x.shape[0]] = x

    # concatenate labels & lengths (for CTC)
    yph = torch.cat(ys_ph, dim=0).to(torch.int32)
    ydi = torch.cat(ys_di, dim=0).to(torch.int32)
    yph_lens = torch.tensor([len(y) for y in ys_ph], dtype=torch.int32)
    ydi_lens = torch.tensor([len(y) for y in ys_di], dtype=torch.int32)
    xin_lens = torch.tensor(Tlens, dtype=torch.int32)

    return Xpad, xin_lens, yph, yph_lens, ydi, ydi_lens

# =============
# DCoND model
# =============
class DCoND_GRU(nn.Module):
    def __init__(self, in_dim, hidden=512, num_layers=2, dropout=0.1,
                 n_prev=0, n_curr=0, n_diphone_classes_incl_blank=0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.proj = nn.Linear(2*hidden, n_diphone_classes_incl_blank)  # diphone head (+blank)
        # cache shapes for marginalization
        self.n_prev = n_prev
        self.n_curr = n_curr
        self.n_pairs = n_diphone_classes_incl_blank - 1  # excluding diphone blank
        assert self.n_pairs == n_prev * n_curr, "Mismatch in diphone class layout."

    def forward(self, x, x_lens):
        # x: (B, T', in_dim), lengths: (B,)
        packed = pack_padded_sequence(x, lengths=x_lens.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(packed)
        y, out_lens = pad_packed_sequence(y, batch_first=True)  # (B, T', 2H)

        logits_di = self.proj(y)                                # (B, T', n_pairs+1)
        # Marginalize diphones → monophones (log-sum-exp over prev axis):
        # Exclude diphone-blank from reshape.
        di_main = logits_di[..., :self.n_pairs]                 # (B, T', n_pairs)
        B, Tprime, _ = di_main.shape
        di_reshaped = di_main.view(B, Tprime, self.n_prev, self.n_curr)
        # logsumexp over prev → get mono logits over curr
        mono_logits = torch.logsumexp(di_reshaped, dim=2)       # (B, T', n_curr)

        # CTC requires a blank at index 0; we build monophone logits with its own blank at idx 0.
        # Our phoneme vocab for CTC is: [BLANK] + all phonemes (including SIL).
        # mono_logits currently corresponds to "curr" = all nonblank & non-SIL phonemes.
        # We need to insert columns for BLANK and SIL into the mono stream.
        # Map order to LOGIT_TO_PHONEME:
        # idx 0 -> BLANK
        # then phonemes including SIL in the same order as LOGIT_TO_PHONEME[1:]
        device = x.device
        BZ = torch.zeros(B, Tprime, 1, device=device, dtype=mono_logits.dtype)  # blank column
        # build full [BLANK | nonblank_nonSIL | SIL] according to LOGIT_TO_PHONEME order
        # Find indices of "curr" inside LOGIT_TO_PHONEME:
        nonblank_ids = [i for i,p in enumerate(LOGIT_TO_PHONEME) if p != 'BLANK']
        curr_ids = [i for i in nonblank_ids if i != SIL_ID]
        # mono_logits order matches curr_ids; we need to interleave to the paper’s LOGIT_TO_PHONEME order excluding BLANK
        # Construct [nonblank_nonSIL..., SIL] into the LOGIT_TO_PHONEME[1:] order
        # Build gather index to align to LOGIT_TO_PHONEME[1:]
        order = [i for i in range(1, len(LOGIT_TO_PHONEME))]  # target order excluding BLANK
        # map each order id to mono position:
        mono_cols = []
        for pid in order:
            if pid == SIL_ID:
                # SIL is not in mono_logits; synthesize it by summing diphone probs where curr=SIL (rare for “curr”; often excluded).
                # If you prefer to exclude SIL from mono CTC targets, you can set this to a very low logit.
                # Here we set SIL column to a small value to avoid dominating.
                mono_cols.append(torch.full((B, Tprime, 1), -10.0, device=device, dtype=mono_logits.dtype))
            else:
                # find its index in curr_ids
                j = curr_ids.index(pid)
                mono_cols.append(mono_logits[..., j:j+1])
        mono_body = torch.cat(mono_cols, dim=-1)  # (B, T', len(LOGIT_TO_PHONEME)-1)
        logits_mono = torch.cat([BZ, mono_body], dim=-1)  # add BLANK at index 0

        return logits_di, logits_mono, out_lens

# ==================
# Training utilities
# ==================
def ctc_loss_from_logits(logits, out_lens, targets, targ_lens, blank_idx):
    # logits: (B, T, C); CTC expects (T,B,C)
    l = logits.transpose(0,1)     # (T,B,C)
    logp = F.log_softmax(l, dim=-1)
    return F.ctc_loss(
        logp, targets, out_lens, targ_lens,
        blank=blank_idx, zero_infinity=True
    )

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

# =========
#  Runner
# =========
def train():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/hdf5_data_final")
    p.add_argument("--epochs", type=int, default=120)  # diphone per paper
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_w", type=int, default=14)      # paper A.6
    p.add_argument("--patch_stride", type=int, default=7)  # overlap allowed; not specified; use W//2
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save", type=str, default="checkpoints_dcond")
    args = p.parse_args()

    set_seed(args.seed)
    Path(args.save).mkdir(parents=True, exist_ok=True)

    # device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    # files (handle missing splits)
    train_files = list_split_files(args.data_root, "train")
    val_files   = list_split_files(args.data_root, "val")
    if not train_files:
        print("No train files found. Exiting.")
        sys.exit(1)
    if not val_files:
        print("Warning: no val files found; will report train loss only.")

    print(f"Found {len(train_files)} train files.")
    if val_files: print(f"Found {len(val_files)} val files.")

    # normalization stats from a subset of train files
    print("Computing robust per-channel stats...")
    stats = robust_channel_stats(train_files, max_trials=300)
    in_channels = int(stats["n_channels"][0])

    # diphone mapping (global, fixed)
    global DIP_PAIR2ID, DIP_ID2PAIR, DIP_NPREV, DIP_NCURR, DIP_NCLASSES
    DIP_PAIR2ID, DIP_ID2PAIR, DIP_NPREV, DIP_NCURR, DIP_NCLASSES = make_diphone_map()

    # datasets
    train_ds = BrainTextDCoND(train_files, stats, args.patch_w, args.patch_stride, args.seed)
    val_ds   = BrainTextDCoND(val_files,   stats, args.patch_w, args.patch_stride, args.seed) if val_files else None

    # infer input dim after patching
    # sample one item
    Xs, _, _, _, _, _ = collate_batch([train_ds[0]])
    in_dim = Xs.shape[-1]  # (B=1, T', C*W) → in_dim
    print(f"Starting DCoND training on {dev} with in_dim={in_dim}, channels≈{in_channels}, "
          f"diphone_classes={DIP_NCLASSES} (prev={DIP_NPREV}, curr={DIP_NCURR})")

    # model
    model = DCoND_GRU(in_dim=in_dim,
                      hidden=args.hidden,
                      num_layers=args.layers,
                      dropout=args.dropout,
                      n_prev=DIP_NPREV,
                      n_curr=DIP_NCURR,
                      n_diphone_classes_incl_blank=DIP_NCLASSES).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(dev.type=="cuda"),
                              collate_fn=collate_batch, drop_last=False)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(dev.type=="cuda"),
                                collate_fn=collate_batch, drop_last=False)

    best_val = float('inf')
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        # alpha schedule per paper: start emphasizing diphone (1-alpha), increase mono share by +0.1 every 10 epochs → 0.6
        alpha = min(0.6, 0.1 * (epoch // 10))
        model.train()
        ep_loss = 0.0
        n_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for Xpad, XinLens, yph, yph_lens, ydi, ydi_lens in loop:
            Xpad = Xpad.to(dev)
            XinLens = XinLens.to(dev)
            yph = yph.to(dev, non_blocking=True)
            yph_lens = yph_lens.to(dev, non_blocking=True)
            ydi = ydi.to(dev, non_blocking=True)
            ydi_lens = ydi_lens.to(dev, non_blocking=True)

            opt.zero_grad()
            logits_di, logits_mono, out_lens = model(Xpad, XinLens)

            # losses
            loss_di = ctc_loss_from_logits(logits_di, out_lens, ydi, ydi_lens, blank_idx=DIP_NCLASSES-1)
            loss_mo = ctc_loss_from_logits(logits_mono, out_lens, yph, yph_lens, blank_idx=CTC_BLANK)
            loss = alpha*loss_mo + (1.0-alpha)*loss_di

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ep_loss += loss.item()
            n_batches += 1

            # live ETA
            elapsed = time.time() - start_time
            iters_done = (epoch-1) * len(train_loader) + n_batches
            iters_total = args.epochs * max(1, len(train_loader))
            eta = (elapsed / max(1,iters_done)) * (iters_total - iters_done)
            loop.set_postfix(loss=f"{loss.item():.4f}", alpha=f"{alpha:.1f}", spent=human_time(elapsed), eta=human_time(eta))

        train_loss = ep_loss / max(1, n_batches)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vloss = 0.0; vb = 0
                for Xpad, XinLens, yph, yph_lens, ydi, ydi_lens in val_loader:
                    Xpad = Xpad.to(dev); XinLens = XinLens.to(dev)
                    yph = yph.to(dev); yph_lens = yph_lens.to(dev)
                    ydi = ydi.to(dev); ydi_lens = ydi_lens.to(dev)
                    logits_di, logits_mono, out_lens = model(Xpad, XinLens)
                    loss_di = ctc_loss_from_logits(logits_di, out_lens, ydi, ydi_lens, blank_idx=DIP_NCLASSES-1)
                    loss_mo = ctc_loss_from_logits(logits_mono, out_lens, yph, yph_lens, blank_idx=CTC_BLANK)
                    loss = alpha*loss_mo + (1.0-alpha)*loss_di
                    vloss += loss.item(); vb += 1
                val_loss = vloss/max(1,vb)
        else:
            val_loss = float("nan")

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} (alpha={alpha:.1f})")

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "stats": stats
        }
        torch.save(ckpt, os.path.join(args.save, f"dcond_epoch{epoch:03d}.pt"))
        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save, f"dcond_best.pt"))

if __name__ == "__main__":
    train()
