import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from scipy import signal
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import *

cudnn.benchmark = True

seed = 2023
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class EEGDataset:
    def __init__(self, mode):
        path = 'data/eeg/' + mode + '.npy'
        self.data = np.load(path, allow_pickle=True)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460, :]
        label = self.data[i]["label"]
        HZ = 1000
        low_f, high_f = 1, 70
        b, a = signal.butter(2, [low_f * 2 / HZ, high_f * 2 / HZ], 'bandpass')
        eeg = signal.lfilter(b, a, eeg).copy()

        return eeg, label


dataset = {mode: EEGDataset(mode) for mode in ['train', 'val', 'test']}
loader = {split: DataLoader(dataset[split], batch_size=64, drop_last=False, shuffle=True) for split in ["train", "val", "test"]}


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# =========================training==========================
epochs = 100
for epoch in range(1, epochs + 1):
    for split in ("train", "val", "test"):
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        for i, (eeg, target) in enumerate(loader[split]):
            eeg = eeg.to(device)
            target = target.to(device)
            eeg, target = Variable(eeg.type(torch.cuda.FloatTensor)), Variable(target.type(torch.cuda.LongTensor))
            output = model(eeg)
            loss = F.cross_entropy(output, target)
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
