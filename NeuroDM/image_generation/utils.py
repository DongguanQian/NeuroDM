import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class EEGDataset(Dataset):
    def __init__(self, split="train"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        path = f'../data/feature/{split}.npy'
        self.data = np.load(path, allow_pickle=True)

    def __getitem__(self, index):
        item = self.data[index]['eeg'], \
               self.transform(Image.fromarray(self.data[index]['image'].numpy())), \
               self.data[index]['label'], \
               self.data[index]['image_label']

        return item

    def __len__(self):
        return len(self.data)


def prepare_dataloaders(batch_size):
    train_dataset = EEGDataset(split='train')
    test_dataset = EEGDataset(split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    return train_dataloader, test_dataloader
